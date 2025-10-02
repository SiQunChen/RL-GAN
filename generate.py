# generate_with_rl.py

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os # 引入 os 模組

# 導入 RL 相關套件和自定義環境
from stable_baselines3 import SAC
from hvac_env import GanHvacEnv

# 由於 hvac_env.py 需要導入 LSTMGenerator，我們也需要確保這個類別是可用的
# 即使這裡不直接使用，hvac_env 的 unpickling 過程也可能需要它
from gan_models import LSTMGenerator

def post_process_data(df, real_data_path):
    # ... 此函式內容完全不變 ...
    """
    對生成的數據進行後處理，以確保其符合物理約束和領域知識。
    此函數的功能類似於您原版 GAN 腳本中的 safe_post_process。
    """
    print("正在執行數據後處理...")
    
    # 重新計算 EHVAC_y
    if 'EHVAC_y' in df.columns and 'PHVAC_y' in df.columns:
        print("根據公式重新計算 'EHVAC_y'...")
        try:
            # 從真實數據中獲取初始能量值
            real_df = pd.read_csv(real_data_path)
            initial_energy = real_df['EHVAC_y'].iloc[0]
            df.loc[0, 'EHVAC_y'] = initial_energy
            print(f"使用真實數據的初始能量值: {initial_energy:.2f}")

            p_values = df['PHVAC_y'].values
            e_values = df['EHVAC_y'].values
            for t in range(1, len(df)):
                # 使用梯形法則進行積分
                e_values[t] = e_values[t-1] + (p_values[t] + p_values[t-1]) / 120.0
            df['EHVAC_y'] = e_values
        except Exception as e:
            print(f"警告：計算 'EHVAC_y' 失敗: {e}。")

    # 強制設定/修正特定欄位值
    if 'chi_on' in df.columns:
        df['chi_on'] = 1 # 假設冰機一直開啟

    # 強制修正物理上必須為正值的欄位
    positive_cols = ['PHVAC_y', 'chi_P', 'fan_P', 'Qflow_total', 'chi_QCon_flow', 'chi_m2_flow', 'fan_m_flow_actual']
    for col in positive_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
    
    if 'EHVAC_y' in df.columns:
        df['EHVAC_y'] = df['EHVAC_y'].clip(lower=0)

    # 修正熱力學約束
    if 'T_WetBul' in df.columns and 'T_DryBul' in df.columns:
        df['T_WetBul'] = np.minimum(df['T_WetBul'], df['T_DryBul'] - 0.1)
    if 'TCHWLeaChi_T' in df.columns and 'TCHWEntChi_T' in df.columns:
        df['TCHWLeaChi_T'] = np.minimum(df['TCHWLeaChi_T'], df['TCHWEntChi_T'] - 0.1)
        
    print("後處理完成。")
    return df


def generate_data_with_rl_agent(
    rl_model_path,
    gan_model_path,
    data_path,
    output_csv_path,
    num_steps
):
    # ... 此函式內容完全不變 ...
    """
    使用訓練好的 RL Agent 來引導 GAN 環境生成數據。
    """
    # --- 1. 參數設定 ---
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用設備: {DEVICE}")

    # --- 2. 建立環境並載入模型 ---
    print("正在以『生成模式』建立 GAN-based HVAC 環境...")
    env = GanHvacEnv(
        gan_model_path=gan_model_path, 
        data_path=data_path, 
        device=DEVICE,
        training_mode=False  # << 關鍵修改！
    )
    print("環境建立成功。")

    print(f"正在從 '{rl_model_path}' 載入訓練好的 RL Agent...")
    # 載入 SAC 模型，並將其與我們剛剛建立的環境關聯
    model = SAC.load(rl_model_path, env=env, device=DEVICE)
    print("RL Agent 載入成功。")

    # --- 3. 執行生成迴圈 ---
    print(f"開始生成數據，總步數: {num_steps}...")
    generated_states_normalized = []
    
    # 重置環境，取得第一個觀測狀態
    obs, info = env.reset()
    generated_states_normalized.append(obs)

    # 使用 tqdm 顯示進度條
    for _ in tqdm(range(num_steps - 1), desc="RL Agent Generating Data"):
        # 讓 RL Agent 根據當前狀態預測最佳動作
        action, _states = model.predict(obs, deterministic=True)
        
        # 在環境中執行動作，獲得下一個狀態
        obs, reward, done, truncated, info = env.step(action)
        
        # 儲存生成的狀態
        generated_states_normalized.append(obs)
        
        # 如果一個 episode 結束，就重置環境以開始新的序列
        # if done or truncated:
        #     obs, info = env.reset()

    print("數據生成完畢。")
    
    # --- 4. 數據轉換與後處理 ---
    # 將狀態列表轉換為 NumPy 陣列
    generated_data_normalized = np.array(generated_states_normalized)
    
    print("正在將數據逆轉換回原始物理單位...")
    # 使用環境中的 scaler 將標準化數據還原
    scaler = env.scaler
    generated_data_original_scale = scaler.inverse_transform(generated_data_normalized)
    
    # 建立 DataFrame
    feature_names = env.feature_names
    df_generated = pd.DataFrame(generated_data_original_scale, columns=feature_names)

    # 插入時間和 EHVAC_y 欄位 (EHVAC_y 會在後處理中重新計算)
    df_generated.insert(0, 'time', range(0, len(df_generated) * 60, 60))
    df_generated.insert(1, 'EHVAC_y', 0.0) # 初始化為0

    # 執行後處理
    df_final = post_process_data(df_generated, real_data_path=data_path)

    # --- 5. 儲存結果 ---
    # 儲存前確保目標資料夾存在
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_final.to_csv(output_csv_path, index=False)
    print(f"成功生成並儲存 {len(df_final)} 筆數據至 '{output_csv_path}'")


if __name__ == '__main__':
    # --- 這個區塊現在僅供獨立測試使用 ---
    # 當您執行 `python generate.py` 時，它才會運行
    # 主控腳本 `run_evaluation.py` 會直接呼叫上面的函式，繞過這裡
    print(">> 警告：此腳本被獨立執行，僅用於測試目的 <<")
    
    # --- 使用者設定 ---
    MODEL_LOG_DIR = './logs/20250922-223235_也還不錯' # 範例資料夾
    RL_MODEL_PATH = os.path.join(MODEL_LOG_DIR, 'best_model.zip')
    GAN_MODEL_PATH = './saved_models/hvac_cgan_500ping.pth'
    DATA_PATH = './data/ChillerFinal500Ping_res.csv'
    OUTPUT_CSV_PATH = os.path.join(MODEL_LOG_DIR, 'hvac_rl_gan_generated_test.csv')
    NUM_STEPS_TO_GENERATE = 5000 # 測試時生成少量數據即可

    generate_data_with_rl_agent(
        rl_model_path=RL_MODEL_PATH,
        gan_model_path=GAN_MODEL_PATH,
        data_path=DATA_PATH,
        output_csv_path=OUTPUT_CSV_PATH,
        num_steps=NUM_STEPS_TO_GENERATE
    )