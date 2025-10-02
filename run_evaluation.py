# run_evaluation.py
import os
import argparse
import sys
import pandas as pd
from datetime import datetime

# --- 導入您的三個分析腳本的核心功能 ---
# 我們假設這四個 .py 檔案都在同一個資料夾底下
from generate import generate_data_with_rl_agent
from 仿真度評估 import HVACComprehensiveAnalyzer
from 效能指標評估 import load_data, calculate_performance_metrics, plot_distributions, plot_correlation_heatmaps, plot_autocorrelation

def main(model_log_dir):
    """
    主控流程函式
    1. 根據指定的模型日誌資料夾，自動設定所有檔案路徑。
    2. 執行 RL-GAN 數據生成。
    3. 執行仿真度綜合評估。
    4. 執行效能與保真度比較。
    """
    print("="*80)
    print(f"🚀 開始執行針對模型 '{os.path.basename(model_log_dir)}' 的完整評估流程")
    print("="*80)

    # --- 1. 自動化路徑設定 ---
    # 這些是固定的檔案，不太會變動
    REAL_DATA_PATH = './data/ChillerFinal500Ping_res.csv'
    VANILLA_GAN_PATH = './synthetic_data_500ping.csv' # 用於比較的純GAN數據
    BASE_GAN_MODEL_PATH = './saved_models/hvac_cgan_500ping.pth' # 基礎GAN模型

    # 這些是根據您指定的資料夾動態決定的
    RL_MODEL_PATH = os.path.join(model_log_dir, 'best_model.zip')
    GENERATED_CSV_PATH = os.path.join(model_log_dir, 'hvac_rl_gan_generated.csv')
    
    # 檢查所有必要的輸入檔案是否存在
    required_files = [REAL_DATA_PATH, VANILLA_GAN_PATH, BASE_GAN_MODEL_PATH, RL_MODEL_PATH]
    for f_path in required_files:
        if not os.path.exists(f_path):
            print(f"❌ 錯誤：找不到必要的檔案 '{f_path}'。請檢查路徑。")
            sys.exit(1) # 中斷程式

    print("\n【步驟 1/4】路徑設定完成")
    print(f"  - 真實數據: {REAL_DATA_PATH}")
    print(f"  - RL Agent: {RL_MODEL_PATH}")
    print(f"  - 生成數據將儲存至: {GENERATED_CSV_PATH}")
    print("-" * 60)

    # --- 2. 執行數據生成 ---
    print("\n【步驟 2/4】使用 RL Agent 引導 GAN 生成數據...")
    # 假設原始數據有 522362 筆，我們就生成同樣長度的數據
    # 您也可以根據需求調整 num_steps
    real_df_for_length = pd.read_csv(REAL_DATA_PATH)
    num_steps_to_generate = len(real_df_for_length)
    
    generate_data_with_rl_agent(
        rl_model_path=RL_MODEL_PATH,
        gan_model_path=BASE_GAN_MODEL_PATH,
        data_path=REAL_DATA_PATH,
        output_csv_path=GENERATED_CSV_PATH,
        num_steps=num_steps_to_generate
    )
    print("✅ 數據生成完畢！")
    print("-" * 60)

    # --- 3. 執行仿真度綜合評估 ---
    print("\n【步驟 3/4】執行仿真度綜合評估...")
    # 評估報告會直接存在模型資料夾底下，方便管理
    analysis_output_dir = os.path.join(model_log_dir, 'comprehensive_analysis_results')
    analyzer = HVACComprehensiveAnalyzer(
        real_data_file=REAL_DATA_PATH,
        synthetic_data_file=GENERATED_CSV_PATH
    )
    analyzer.comprehensive_analysis(output_dir=analysis_output_dir)
    print("✅ 仿真度評估完成！")
    print("-" * 60)

    # --- 4. 執行效能與保真度比較 ---
    print("\n【步驟 4/4】執行效能與保真度比較分析...")
    # 建立一個字典，包含所有要比較的檔案路徑
    comparison_paths = {
        'Real Data': REAL_DATA_PATH,
        'Vanilla GAN': VANILLA_GAN_PATH,
        'RL+GAN': GENERATED_CSV_PATH
    }
    
    # 載入數據
    dataframes = load_data(comparison_paths)
    
    if dataframes:
        # 執行效能指標計算
        calculate_performance_metrics(dataframes)
        
        # 繪製圖表 (注意：圖表會直接顯示，需要手動關閉才會繼續)
        plot_distributions(dataframes, columns=['roo_TRooAir', 'PHVAC_y', 'TCHWLeaChi_T'])
        plot_correlation_heatmaps(dataframes)
        plot_autocorrelation(dataframes, column='roo_TRooAir')
    print("✅ 效能比較分析完成！")
    print("-" * 60)

    print("\n🎉🎉🎉 所有評估流程已成功執行完畢！ 🎉🎉🎉")
    print(f"所有產出的報告與數據都可以在這個資料夾找到: {model_log_dir}")


if __name__ == '__main__':
    # --- 使用 argparse 讓您可以從命令列傳入資料夾路徑 ---
    parser = argparse.ArgumentParser(description="HVAC 模型生成與評估主控腳本")
    parser.add_argument(
        "model_log_dir", 
        type=str, 
        help="指定包含 best_model.zip 的模型訓練日誌資料夾路徑。例如: ./logs/20250922-223235_也還不錯"
    )
    
    args = parser.parse_args()

    # 檢查使用者提供的路徑是否存在
    if not os.path.isdir(args.model_log_dir):
        print(f"❌ 錯誤：找不到您指定的資料夾 '{args.model_log_dir}'。請確認路徑是否正確。")
        sys.exit(1)
        
    main(args.model_log_dir)