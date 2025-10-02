# compare_models.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
import warnings

# --- 設定 ---
warnings.filterwarnings('ignore')
# 移除或註解掉中文的字體設定，以使用 Matplotlib 預設的英文字體
# plt.rcParams['font.sans-serif'] = ['Heiti TC'] 
plt.rcParams['axes.unicode_minus'] = False


def load_data(file_paths: dict) -> dict:
    """載入多個 CSV 檔案並回傳 DataFrame 字典"""
    dataframes = {}
    print("正在載入數據...")
    for name, path in file_paths.items():
        try:
            dataframes[name] = pd.read_csv(path)
            print(f"- 已成功載入 '{name}' ({len(dataframes[name])} 筆)")
        except FileNotFoundError:
            print(f"錯誤：找不到檔案 '{path}'。請檢查路徑。")
            return None
    return dataframes


def calculate_performance_metrics(dataframes: dict, target_temp: float = 25.0):
    """
    計算並比較舒適度、能耗，以及新的綜合效能指標 (CPI)。
    
    Comfort-Power Index (CPI) 解釋:
    - 公式: CPI = Avg. Power * Comfort RMSE
    - 意義: 代表了為了將溫度每偏離目標 1°C 所付出的平均能量代價。
    - 結論: CPI 值越低，代表模型在節能與控溫上的綜合性能越好。
    """
    print("\n--- 1. 效能指標評估 ---")
    results = []
    
    for name, df in dataframes.items():
        # 舒適度: 均方根誤差 (RMSE) - 維持不變
        temp_rmse = np.sqrt(np.mean((df['roo_TRooAir'] - target_temp)**2))
        
        # 平均功耗 - 維持不變
        avg_power = df['PHVAC_y'].mean()
        
        # <--- 核心修改：移除舊的、有誤導性的 Efficiency 指標 --->
        # comfort_hours_ratio = ((df['roo_TRooAir'] >= target_temp - comfort_range) & 
        #                        (df['roo_TRooAir'] <= target_temp + comfort_range)).mean()
        # efficiency = avg_power / comfort_hours_ratio if comfort_hours_ratio > 0 else float('inf')
        
        # <--- 核心修改：計算新的 Comfort-Power Index (CPI) --->
        # 這個指標更公平地反映了能耗和舒適度的平衡
        comfort_power_index = avg_power * temp_rmse
        
        results.append({
            'Model': name,
            'Comfort RMSE': temp_rmse,
            'Avg. Power': avg_power,
            'Comfort-Power Index': comfort_power_index # << 使用新指標
        })
        
    results_df = pd.DataFrame(results).set_index('Model')
    print("效能指標比較表：")
    print(results_df.round(3))
    return results_df


def plot_distributions(dataframes: dict, columns: list):
    """繪製關鍵特徵的機率密度分佈圖"""
    print("\n--- 2. 數據保真度評估 (特徵分佈) ---")
    num_cols = len(columns)
    fig, axes = plt.subplots(num_cols, 1, figsize=(12, 5 * num_cols))
    if num_cols == 1: axes = [axes] # 確保單圖時仍可迭代

    colors = {'Real Data': 'blue', 'Vanilla GAN': 'orange', 'RL+GAN': 'green'}

    for i, col in enumerate(columns):
        ax = axes[i]
        for name, df in dataframes.items():
            sns.kdeplot(df[col], ax=ax, label=name, color=colors.get(name), fill=True, alpha=0.2)
        # --- 圖表標示改為英文 ---
        ax.set_title(f"Probability Density Distribution for '{col}'", fontsize=15)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmaps(dataframes: dict):
    """繪製特徵相關性熱圖"""
    print("\n--- 3. 數據保真度評估 (特徵關聯性) ---")
    num_dfs = len(dataframes)
    fig, axes = plt.subplots(1, num_dfs, figsize=(8 * num_dfs, 7))
    if num_dfs == 1: axes = [axes]

    for i, (name, df) in enumerate(dataframes.items()):
        # 選擇部分關鍵特徵進行比較，避免熱圖過於擁擠
        key_features = ['PHVAC_y', 'roo_TRooAir', 'TCHWLeaChi_T', 'TCHWEntChi_T', 'T_DryBul', 'chi_P']
        key_features = [f for f in key_features if f in df.columns]
        
        corr = df[key_features].corr()
        sns.heatmap(corr, ax=axes[i], annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
        # --- 圖表標示改為英文 ---
        axes[i].set_title(f"Feature Correlation Heatmap for '{name}'", fontsize=15)
        
    plt.tight_layout()
    plt.show()

def plot_autocorrelation(dataframes: dict, column: str, lags: int = 48):
    """繪製單一特徵的自相關函數圖"""
    print(f"\n--- 4. 數據保真度評估 (時間依賴性: {column}) ---")
    num_dfs = len(dataframes)
    fig, axes = plt.subplots(num_dfs, 1, figsize=(12, 4 * num_dfs))
    if num_dfs == 1: axes = [axes]
    
    for i, (name, df) in enumerate(dataframes.items()):
        # --- 圖表標示改為英文 ---
        plot_acf(df[column], lags=lags, ax=axes[i], title=f"Autocorrelation Function (ACF) for '{column}' in '{name}'")
        axes[i].grid(True, linestyle='--', alpha=0.6)
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- 這個區塊現在僅供獨立測試使用 ---
    print(">> 警告：此腳本被獨立執行，僅用於測試目的 <<")
    
    # --- 請在此處設定您的三個檔案路徑 ---
    # 將字典的 'key' 改成英文，因為它會被當作圖例 (legend) 的標籤
    FILE_PATHS_FOR_TESTING = {
        'Real Data': './data/ChillerFinal500Ping_res.csv',  # 請替換為您的真實數據檔案
        'Vanilla GAN': './synthetic_data_500ping.csv', # 之前純 GAN 生成的數據
        'RL+GAN': './logs/20250924-015030_剩效率不好/hvac_rl_gan_generated.csv' # 測試時請確保此檔案存在
    }
    
    # 載入數據
    dataframes = load_data(FILE_PATHS_FOR_TESTING)
    
    if dataframes:
        # 1. 執行效能指標計算
        performance_df = calculate_performance_metrics(dataframes)
        
        # 2. 繪製關鍵特徵分佈
        # 我們關心：室內溫度、總功耗，以及 RL 主要控制的冰水出水溫度
        plot_distributions(dataframes, columns=['roo_TRooAir', 'PHVAC_y', 'TCHWLeaChi_T'])
        
        # 3. 繪製相關性熱圖
        plot_correlation_heatmaps(dataframes)
        
        # 4. 繪製室內溫度的自相關圖，觀察時間特性
        plot_autocorrelation(dataframes, column='roo_TRooAir')

        print("\n=== 分析完成 ===")