# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import warnings
import argparse

warnings.filterwarnings('ignore')

# --- 條件 LSTM 生成器 (維持不變) ---
class LSTMGenerator(nn.Module):
    # --- 修改點 1: 在 __init__ 方法中增加 scaler 參數 ---
    def __init__(self, scaler, noise_dim=100, input_dim=30, output_dim=30, seq_length=24, hidden_dim=256, num_layers=2):
        super().__init__()
        # --- 修改點 2: 將傳入的 scaler 保存為類別屬性 ---
        self.scaler = scaler
        
        # --- 以下原始程式碼保持不變 ---
        self.seq_length = seq_length
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.condition_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim * num_layers)
        )

        self.lstm = nn.LSTM(
            input_size=noise_dim + input_dim + 1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        
        self.fc_output = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, initial_condition):
        device = next(self.parameters()).device
        batch_size = initial_condition.size(0)
        
        h0 = self.condition_encoder(initial_condition.squeeze(1))
        h0 = h0.view(self.num_layers, batch_size, self.hidden_dim)
        c0 = torch.zeros_like(h0)

        noise_sequence = torch.randn(batch_size, self.seq_length - 1, 100, device=device)
        condition_repeated = initial_condition.repeat(1, self.seq_length - 1, 1)

        lstm_input = torch.cat([noise_sequence, condition_repeated], dim=2)
        
        lstm_out, _ = self.lstm(lstm_input, (h0, c0))
        
        generated_sequence = self.fc_output(lstm_out)
        full_sequence = torch.cat([initial_condition, generated_sequence], dim=1)
        
        return full_sequence

    def predict_next_step(self, current_state_normalized, action, h_prev, c_prev):
        device = next(self.parameters()).device
        
        # 創建一個假的噪聲輸入
        dummy_noise = torch.randn(current_state_normalized.size(0), 1, 100, device=device) # <--- 使用隨機噪聲而非全零
        
        # 【修改點 6】: 將 action 和 state 拼接作為 LSTM 的輸入
        lstm_input = torch.cat([dummy_noise, current_state_normalized, action], dim=2)
        
        lstm_out, (h_next, c_next) = self.lstm(lstm_input, (h_prev, c_prev))
        
        next_state_normalized = self.fc_output(lstm_out)
        
        return next_state_normalized, h_next, c_next

    def predict_next_step_from_state(self, modified_state_normalized, h_prev, c_prev):
        """
        只根據被修改後的狀態來預測下一步，更符合原始GAN的自回歸邏輯。
        """
        device = next(self.parameters()).device
        batch_size = modified_state_normalized.size(0)
        
        # 準備噪聲和一個假的 action (因為 LSTM 的輸入層維度是固定的)
        # 這個假的 action 就像原始訓練中的隨機靈感
        noise = torch.randn(batch_size, 1, self.lstm.input_size - modified_state_normalized.size(2) - 1, device=device)
        dummy_action = torch.zeros(batch_size, 1, 1, device=device)

        # 將 (被修改的狀態, 假動作, 噪聲) 拼接起來
        # 注意：這裡的拼接順序和維度需要嚴格對應您 LSTM 的 input_size
        # 假設原始 input_size 是 noise_dim + input_dim + 1
        # 這裡的 modified_state_normalized 就是 input_dim
        lstm_input = torch.cat([modified_state_normalized, dummy_action, noise], dim=2)
        
        lstm_out, (h_next, c_next) = self.lstm(lstm_input, (h_prev, c_prev))
        
        next_state_normalized = self.fc_output(lstm_out)
        
        return next_state_normalized, h_next, c_next

# --- 評論家 CNNCritic (維持不變, 名稱已很合適) ---
class CNNCritic(nn.Module):
    def __init__(self, input_dim=30, seq_length=24):
        super().__init__()
        l_out_2, l_out_3 = seq_length // 2, seq_length // 4
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(input_dim, 64, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, 3, 2, 1), nn.LayerNorm([128, l_out_2]), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, 3, 2, 1), nn.LayerNorm([256, l_out_3]), nn.LeakyReLU(0.2, inplace=True),
        )
        self.mlp_layers = nn.Sequential(
            nn.Linear(256 * l_out_3, 512), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x_permuted = x.permute(0, 2, 1)
        cnn_out = self.cnn_layers(x_permuted)
        x_flat = cnn_out.view(cnn_out.size(0), -1)
        return self.mlp_layers(x_flat)