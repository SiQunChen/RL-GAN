# train_rl.py (再次優化版：強化全局真實性權重)

import torch
import gymnasium as gym
import os
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from hvac_env import GanHvacEnv

def main():
    # --- 超參數設定 ---
    GAN_MODEL_PATH = './saved_models/hvac_cgan_500ping.pth'
    DATA_PATH = './data/ChillerFinal500Ping_res.csv'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    LOG_DIR = f'./logs/{run_timestamp}/'
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"本次運行的日誌與模型將儲存於: {LOG_DIR}")

    TOTAL_TIMESTEPS = 1_000_000
    
    # --- 建立用於「評估」的環境 ---
    eval_env = GanHvacEnv(
        gan_model_path=GAN_MODEL_PATH, 
        data_path=DATA_PATH, 
        device=DEVICE,
        w_realism=0.8, # 評估環境的權重通常保持固定，這裡維持不變
        w_physics=1.5
    )
    eval_env = Monitor(eval_env)

    # --- 建立用於「訓練」的環境 ---
    train_env = GanHvacEnv(
        gan_model_path=GAN_MODEL_PATH, 
        data_path=DATA_PATH, 
        device=DEVICE,
        # <--- 核心修改：提高真實性獎勵的權重 --->
        w_realism=8.0,  # << 從 5.0 提高到 8.0，強化對多變量結構的約束
        w_physics=1.0
    )
    train_env = Monitor(train_env)

    # --- 設定 EvalCallback ---
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=LOG_DIR,
        log_path=LOG_DIR, 
        eval_freq=20000,
        n_eval_episodes=10,
        deterministic=True, 
        render=False
    )
    
    # --- 初始化 SAC 模型 (超參數保持不變) ---
    model = SAC(
        policy='MlpPolicy',
        env=train_env, 
        verbose=1,
        tensorboard_log=LOG_DIR,
        device=DEVICE,
        learning_rate=1e-4,
        buffer_size=1000000,
        batch_size=512,
        learning_starts=25000,
        ent_coef='auto',
        policy_kwargs=dict(net_arch=[400, 300])
    )

    # --- 開始訓練 ---
    print("\n--- 開始進行混合目標優化訓練 (強化全局真實性) ---")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=True
    )
    
    model.save(f"{LOG_DIR}/sac_hvac_final_model")
    print("--- 訓練完成 ---")
    print(f"訓練過程中的最佳模型已儲存至: {LOG_DIR}best_model.zip")


if __name__ == '__main__':
    main()