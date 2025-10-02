# hvac_env.py (優化版：注入物理一致性獎勵)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import pandas as pd
from collections import deque
from gan_models import LSTMGenerator, CNNCritic 

class GanHvacEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    # <--- 核心修改 1: 在 __init__ 的參數中增加物理獎勵的權重 w_physics --->
    def __init__(self, gan_model_path, data_path, device='cpu', training_mode=True, w_realism=0.5, w_physics=1.0):
        super().__init__()
        
        self.device = torch.device(device)
        self.training_mode = training_mode
        
        # --- 1. 載入模型與數據處理器 ---
        print(f"從 '{gan_model_path}' 載入預訓練的 GAN...")
        checkpoint = torch.load(gan_model_path, map_location=self.device, weights_only=False)
        self.feature_names = checkpoint['feature_names']
        self.scaler = checkpoint['scaler']
        
        state_dim = len(self.feature_names)
        action_dim = 1
        
        # --- 2. 實例化生成器 (Generator) ---
        self.generator = LSTMGenerator(
            scaler=self.scaler,
            input_dim=state_dim, 
            output_dim=state_dim
        ).to(self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.generator.eval()
        
        # --- 3. 實例化判別器 (Discriminator) ---
        self.seq_length_for_critic = 24 
        self.discriminator = CNNCritic(
            input_dim=state_dim,
            seq_length=self.seq_length_for_critic
        ).to(self.device)
        if 'critic' in checkpoint:
            self.discriminator.load_state_dict(checkpoint['critic'])
            self.discriminator.eval()
            print("GAN 生成器與判別器均載入成功。")
        else:
            print("警告：在模型檔案中未找到判別器權重，真實性獎勵將被禁用。")
            w_realism = 0.0

        # --- 4. 初始化狀態歷史緩衝區 ---
        self.state_history = deque(maxlen=self.seq_length_for_critic)
        
        # --- 5. 定義環境參數 ---
        self.PHVAC_y_idx = self.feature_names.index('PHVAC_y')
        self.roo_TRooAir_idx = self.feature_names.index('roo_TRooAir')
        self.controlled_feature = 'TCHWLeaChi_T'
        self.controlled_idx = self.feature_names.index(self.controlled_feature)

        # <--- 核心修改 2: 取得計算物理獎勵所需特徵的索引 --->
        self.chi_P_idx = self.feature_names.index('chi_P')
        self.chi_QCon_flow_idx = self.feature_names.index('chi_QCon_flow')
        self.chi_QEva_flow_idx = self.feature_names.index('chi_QEva_flow')
        print(f"RL Agent 將直接控制特徵: '{self.controlled_feature}' (索引: {self.controlled_idx})")
        
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        
        # --- 6. 載入完整數據用於初始化和外部條件注入 ---
        print("正在載入完整數據用於環境模擬...")
        full_real_df = pd.read_csv(data_path)
        df_for_state = full_real_df.drop(['time', 'EHVAC_y'], axis=1, errors='ignore')
        df_for_state = df_for_state[self.feature_names]
        self.initial_conditions_normalized = self.scaler.transform(df_for_state.values)
        self.external_conditions_features = [
            'T_DryBul', 'T_WetBul', 'qRadGai_flow', 
            'qLatGai_flow', 'qConGai_flow'
        ]
        self.external_conditions_features = [f for f in self.external_conditions_features if f in self.feature_names]
        self.external_conditions_df = df_for_state[self.external_conditions_features]
        self.external_indices = [self.feature_names.index(f) for f in self.external_conditions_features]
        print(f"將在每一步強制更新 {len(self.external_indices)} 個外部條件特徵。")

        self.current_step = 0
        self.max_steps = 200
        self.last_action = np.zeros(self.action_space.shape)
        
        # <--- 核心修改 3: 保存獎勵權重 --->
        self.w_realism = w_realism
        self.w_physics = w_physics # 保存物理獎勵的權重
        
        print(f"環境初始化完成。訓練模式: {self.training_mode}, Episode 最大步數: {self.max_steps}")
        print(f"獎勵權重 -> (HVAC 獎勵) + 真實性: {self.w_realism} + 物理一致性: {self.w_physics}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_start_idx = self.np_random.integers(0, len(self.initial_conditions_normalized) - self.max_steps)
        self.state_normalized = self.initial_conditions_normalized[self.episode_start_idx]
        self.current_step = 0
        self.state_history.clear()
        for _ in range(self.seq_length_for_critic):
            self.state_history.append(self.state_normalized)
        initial_condition_tensor = torch.FloatTensor(self.state_normalized).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            batch_size = 1
            h0 = self.generator.condition_encoder(initial_condition_tensor.squeeze(1))
            h0 = h0.view(self.generator.num_layers, batch_size, self.generator.hidden_dim)
            c0 = torch.zeros_like(h0)
        self.h_prev, self.c_prev = h0, c0
        self.last_action = np.zeros(self.action_space.shape)
        return self.state_normalized.astype(np.float32), {}

    def step(self, action):
        modified_state_normalized = np.copy(self.state_normalized)
        modified_state_normalized[self.controlled_idx] = action[0]
        modified_state_tensor = torch.FloatTensor(modified_state_normalized).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            next_state_normalized_tensor, h_next, c_next = self.generator.predict_next_step_from_state(
                modified_state_tensor, self.h_prev, self.c_prev
            )
        
        self.state_normalized = next_state_normalized_tensor.squeeze().cpu().numpy()
        self.h_prev, self.c_prev = h_next, c_next
        
        step_in_episode = self.current_step
        total_rows_external = len(self.external_conditions_df)
        current_index = (self.episode_start_idx + step_in_episode) % total_rows_external
        current_external_conditions_row = self.external_conditions_df.iloc[current_index]
        
        for i, feature_idx in enumerate(self.external_indices):
            feature_name = self.external_conditions_features[i]
            original_value = current_external_conditions_row[feature_name]
            s_min_val = self.scaler.data_min_[feature_idx]
            s_range = self.scaler.data_range_[feature_idx]
            normalized_value = ((original_value - s_min_val) / s_range) * 2.0 - 1.0 if s_range > 1e-9 else 0.0
            self.state_normalized[feature_idx] = np.clip(normalized_value, -1.0, 1.0)

        self.state_history.append(self.state_normalized)
        
        # --- 4. 計算各項獎勵 ---
        state_original = self.scaler.inverse_transform(self.state_normalized.reshape(1, -1))[0]
        
        # 4.1 真實性獎勵
        realism_reward = 0.0
        if len(self.state_history) == self.seq_length_for_critic and self.w_realism > 0:
            history_tensor = torch.FloatTensor(np.array(self.state_history)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                realism_score = torch.sigmoid(self.discriminator(history_tensor)).item()
            realism_reward = realism_score

        # 4.2 HVAC 效能獎勵
        hvac_reward = self._calculate_hvac_reward(state_original, action)

        # <--- 核心修改 4: 計算新的物理一致性獎勵 --->
        physics_reward = self._calculate_physics_reward(state_original)

        # 4.3 組合最終總獎勵
        total_reward = hvac_reward + (self.w_realism * realism_reward) + (self.w_physics * physics_reward)
        
        self.last_action = action
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return self.state_normalized.astype(np.float32), total_reward, done, False, {}

    def _calculate_hvac_reward(self, state_original, action):
        current_temp = state_original[self.roo_TRooAir_idx]
        power_consumption = state_original[self.PHVAC_y_idx]
        target_temp, comfort_zone, energy_penalty_multiplier = 25.0, 0.3, 3.0
        w_comfort, w_energy, w_stability = 1.0, 1.5, 0.2
        
        temp_deviation = abs(current_temp - target_temp)
        comfort_reward = np.exp(- (temp_deviation**2) / (2 * 1.0**2))
        
        max_power_realistic = 150.0 
        base_energy_penalty = (power_consumption / max_power_realistic)**2
        final_energy_penalty = base_energy_penalty * energy_penalty_multiplier if temp_deviation <= comfort_zone else base_energy_penalty
        final_energy_penalty = min(final_energy_penalty, 2.0)

        action_change = np.sum((action - self.last_action)**2)
        stability_penalty = min(1.0, action_change / 2.0)

        return (w_comfort * comfort_reward) - (w_energy * final_energy_penalty) - (w_stability * stability_penalty)

    # <--- 核心修改 5: 新增計算物理一致性獎勵的函式 --->
    def _calculate_physics_reward(self, state_original):
        """
        根據能量守恆定律計算物理一致性獎勵。
        Q_condenser ≈ -Q_evaporator + P_chiller
        """
        # 從反正規化後的狀態中獲取所需值
        chi_p = state_original[self.chi_P_idx]
        q_con = state_original[self.chi_QCon_flow_idx]
        q_eva = state_original[self.chi_QEva_flow_idx]

        # 計算能量守恆的等式右側
        # 注意：q_eva 在數據中是負值，代表吸熱，所以要取負號
        energy_sum = -q_eva + chi_p
        
        # 計算誤差百分比
        # 加上一個極小值 epsilon (1e-6) 來避免除以零
        error_percentage = np.abs(q_con - energy_sum) / (np.abs(q_con) + 1e-6)
        
        # 將誤差轉換為獎勵 (0-1之間)。誤差越小，獎勵越高。
        # 使用 exp(-error) 曲線，小誤差幾乎不懲罰，大誤差懲罰會很顯著
        physics_reward = np.exp(-error_percentage)
        
        return physics_reward

    def close(self):
        print("環境關閉。")