# -*- coding: utf-8 -*-
"""
HVAC 仿真資料集綜合評估工具 (優化版本)

優化與新增功能：
1. 機器學習實用性評估 (Train on Synthetic, Test on Real - TSTR)
2. 時間序列分析加入 動態時間規整 (Dynamic Time Warping - DTW)
3. 多變量分佈評估 (PCA-based)
4. 調整綜合評分權重，納入新指標
5. 程式碼結構優化與錯誤處理增強
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wasserstein_distance
from statsmodels.tsa.stattools import acf, pacf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import warnings
import os
from datetime import datetime
from tabulate import tabulate

# 嘗試導入 dtw，若失敗則提示安裝
try:
    from dtw import dtw
except ImportError:
    print("警告：缺少 'dtw-python' 套件。時間序列 DTW 分析將被跳過。")
    print("請執行 'pip install dtw-python' 來安裝。")
    dtw = None

warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class HVACComprehensiveAnalyzer:
    """HVAC資料集綜合評估分析器 (優化版)"""

    def __init__(self, real_data_file: str, synthetic_data_file: str):
        self.real_data_file = real_data_file
        self.synthetic_data_file = synthetic_data_file
        self._load_data()
        self.results = {} # 初始化結果容器

    def _load_data(self):
        """載入數據"""
        try:
            self.real_data = pd.read_csv(self.real_data_file)
            self.synthetic_data = pd.read_csv(self.synthetic_data_file)
            
            # 確保資料長度一致以便比較
            min_len = min(len(self.real_data), len(self.synthetic_data))
            self.real_data = self.real_data.iloc[:min_len]
            self.synthetic_data = self.synthetic_data.iloc[:min_len]

            self.numeric_columns = self.real_data.select_dtypes(include=np.number).columns.tolist()
            if 'time' in self.numeric_columns:
                self.numeric_columns.remove('time')

            print("="*80)
            print("📊 HVAC 仿真資料集綜合評估系統 (優化版)")
            print("="*80)
            print(f"✅ 資料載入成功")
            print(f"   原始資料維度: {self.real_data.shape}")
            print(f"   仿真資料維度: {self.synthetic_data.shape}")
            print(f"   分析特徵數量: {len(self.numeric_columns)} 個")
            print("="*80)
            
        except FileNotFoundError as e:
            print(f"❌ 錯誤：找不到資料檔案 {e.filename}")
            raise

    def comprehensive_analysis(self, output_dir: str = "analysis_results"):
        """執行完整的綜合分析"""
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        print("\n🔍 開始執行綜合分析...\n")
        
        # 依次執行各項分析
        self._descriptive_statistics_analysis()
        self._fidelity_assessment()
        self._hvac_specific_analysis()
        self._correlation_analysis()
        self._time_series_analysis()
        self._utility_assessment()  # 新增：TSTR實用性評估
        self._multivariate_analysis() # 新增：多變量結構評估
        self._calculate_overall_scores()
        self._generate_visualizations()
        self._generate_comprehensive_report()
        
        print(f"\n✅ 分析完成！所有結果已保存至: {output_dir}")
    
    # --- 各項分析方法 (為簡潔，僅展示新增與修改的核心部分) ---
    
    def _descriptive_statistics_analysis(self):
        """1. 描述性統計分析 (與原版相同)"""
        print("📈 1. 描述性統計分析")
        print("-" * 60)
        stats_real = self.real_data.describe().T
        stats_synth = self.synthetic_data.describe().T
        comparison_data = []
        for feature in stats_real.index:
            if feature in stats_synth.index and feature in self.numeric_columns:
                real_mean, synth_mean = stats_real.loc[feature, 'mean'], stats_synth.loc[feature, 'mean']
                real_std, synth_std = stats_real.loc[feature, 'std'], stats_synth.loc[feature, 'std']
                mean_error = abs(synth_mean - real_mean) / abs(real_mean) * 100 if real_mean != 0 else 0
                std_error = abs(synth_std - real_std) / abs(real_std) * 100 if real_std != 0 else 0
                comparison_data.append({
                    '特徵': feature, '真實均值': f"{real_mean:.3f}", '仿真均值': f"{synth_mean:.3f}",
                    '均值誤差(%)': f"{mean_error:.1f}", '真實標準差': f"{real_std:.3f}",
                    '仿真標準差': f"{synth_std:.3f}", '標準差誤差(%)': f"{std_error:.1f}"
                })
        print(tabulate(comparison_data, headers='keys', tablefmt='grid'))
        self.results['descriptive_stats'] = pd.DataFrame(comparison_data)
        print("\n")

    def _fidelity_assessment(self):
        """2. 保真度評估 (與原版相同)"""
        print("🎯 2. 分佈保真度評估 (Fidelity Assessment)")
        print("-" * 60)
        fidelity_data = []
        for feature in self.numeric_columns:
            real_v, synth_v = self.real_data[feature].dropna(), self.synthetic_data[feature].dropna()
            if len(real_v) < 10 or len(synth_v) < 10: continue
            awd_score = wasserstein_distance(real_v, synth_v)
            stat_grade = "✅優秀" if awd_score < 1 else "⚠️良好" if awd_score < 2 else "❌需改善"
            fidelity_data.append({'特徵': feature, 'Wasserstein距離': f"{awd_score:.4f}", '統計等級': stat_grade})
        print(tabulate(fidelity_data, headers='keys', tablefmt='grid'))
        self.results['fidelity_metrics'] = pd.DataFrame(fidelity_data)
        print("\n")

    def _hvac_specific_analysis(self):
        """3. HVAC系統專屬分析 (與原版相同，僅作微調)"""
        print("🏭 3. HVAC系統專屬分析")
        print("-" * 60)
        # (此處程式碼與原版幾乎相同，為節省篇幅省略，實際程式碼中應保留)
        # ...
        hvac_results = [] # 假設已有計算結果
        # Chi_V 常數檢查
        if 'Chi_V' in self.real_data.columns and 'Chi_V' in self.synthetic_data.columns:
            real_chi_v_mode = self.real_data['Chi_V'].mode()[0]
            synth_chi_v_mode = self.synthetic_data['Chi_V'].mode()[0]
            chi_v_status = "✅正確" if real_chi_v_mode == synth_chi_v_mode else "❌錯誤"
            hvac_results.append({'檢查項目': 'Chi_V常數值', '真實資料': f"{real_chi_v_mode}", '仿真資料': f"{synth_chi_v_mode}", '狀態': chi_v_status})
        # 開關邏輯分析
        if 'chi_on' in self.real_data.columns and 'chi_on' in self.synthetic_data.columns:
            real_on_rate = self.real_data['chi_on'].mean() * 100
            synth_on_rate = self.synthetic_data['chi_on'].mean() * 100
            on_rate_diff = abs(real_on_rate - synth_on_rate)
            logic_status = "✅優秀" if on_rate_diff < 5 else "⚠️良好" if on_rate_diff < 10 else "❌需改善"
            hvac_results.append({'檢查項目': '開啟率(%)', '真實資料': f"{real_on_rate:.1f}", '仿真資料': f"{synth_on_rate:.1f}", '狀態': logic_status})
        
        if hvac_results:
             print(tabulate(hvac_results, headers='keys', tablefmt='grid'))
             self.results['hvac_specific'] = pd.DataFrame(hvac_results)
        else:
            print("未找到可供分析的HVAC特定欄位。")
        print("\n")

    def _correlation_analysis(self):
        """4. 相關性結構分析 (與原版相同)"""
        print("🔗 4. 相關性結構分析")
        print("-" * 60)
        common_features = [f for f in self.numeric_columns if f in self.synthetic_data.columns]
        real_corr, synth_corr = self.real_data[common_features].corr(), self.synthetic_data[common_features].corr()
        corr_diff = np.abs(real_corr - synth_corr)
        self.mean_corr_diff = np.nanmean(corr_diff.values[np.triu_indices_from(corr_diff, k=1)])
        print(f"平均相關性差異: {self.mean_corr_diff:.4f}")
        corr_grade = "✅優秀" if self.mean_corr_diff < 0.1 else "⚠️良好" if self.mean_corr_diff < 0.2 else "❌需改善"
        print(f"相關性保持程度: {corr_grade}")
        self.results['correlation_analysis'] = {'mean_diff': self.mean_corr_diff, 'grade': corr_grade}
        print("\n")

    def _time_series_analysis(self):
        """5. 時間序列特性分析 (整合ACF, FFT, DTW - 使用降採樣以相容舊版套件)"""
        print("⏰ 5. 時間序列特性分析")
        print("-" * 60)
        
        if dtw is None:
            print("DTW 分析已跳過，因為 'dtw-python' 套件未安裝。")
            self.results['time_series_analysis'] = pd.DataFrame()
            return

        key_features = ['PHVAC_y', 'chi_P', 'TAirLeaRoo_T', 'T_DryBul']
        time_results = []

        for feature in key_features:
            if feature not in self.numeric_columns or feature not in self.synthetic_data.columns:
                continue
            
            real_v = self.real_data[feature].dropna().values
            synth_v = self.synthetic_data[feature].dropna().values
            if len(real_v) < 50: continue

            # ACF 相似度
            real_acf = acf(real_v, nlags=10, fft=True)
            synth_acf = acf(synth_v, nlags=10, fft=True)
            acf_diff = np.mean(np.abs(real_acf - synth_acf))
            acf_score = max(0, 1 - acf_diff / 0.5)

            # FFT 相似度
            fft_diff = self._calculate_fft_diff(real_v, synth_v)
            fft_score = max(0, 1 - fft_diff / 0.2)
            
            # --- 🔧 DTW 相似度 (使用降採樣以相容舊版套件) ---
            # 正規化資料
            scaler = StandardScaler()
            real_v_norm = scaler.fit_transform(real_v.reshape(-1, 1)).flatten()
            synth_v_norm = scaler.transform(synth_v.reshape(-1, 1)).flatten()

            # 將長序列降採樣至可計算的長度，例如 2000
            sample_size = 2000
            if len(real_v_norm) > sample_size:
                # 透過選取等間隔的點來進行降採樣
                indices = np.linspace(0, len(real_v_norm) - 1, sample_size, dtype=int)
                real_v_sampled = real_v_norm[indices]
                synth_v_sampled = synth_v_norm[indices]
            else:
                real_v_sampled = real_v_norm
                synth_v_sampled = synth_v_norm
            
            # 使用降採樣後的短序列進行DTW計算，不需 window 參數
            alignment = dtw(real_v_sampled, synth_v_sampled, keep_internals=True)
            
            dtw_dist = alignment.distance
            # 使用採樣後序列的長度進行正規化
            dtw_score = max(0, 1 - dtw_dist / len(real_v_sampled))

            # 綜合時間分數 (ACF:40%, FFT:30%, DTW:30%)
            time_score = (acf_score * 0.4 + fft_score * 0.3 + dtw_score * 0.3) * 100
            time_grade = "✅優秀" if time_score >= 85 else "⚠️良好" if time_score >= 70 else "❌需改善"
            
            time_results.append({
                '特徵': feature, 'ACF差異': f"{acf_diff:.4f}", 'FFT差異': f"{fft_diff:.4f}",
                'DTW距離': f"{dtw_dist:.4f}", '綜合時間分數': f"{time_score:.1f}", '時序相似性': time_grade
            })
        
        if time_results:
            print(tabulate(time_results, headers='keys', tablefmt='grid'))
            self.results['time_series_analysis'] = pd.DataFrame(time_results)
        else:
            print("沒有找到可供分析的時間序列特徵。")
            self.results['time_series_analysis'] = pd.DataFrame()
            
        print("\n")

    def _utility_assessment(self):
        """6. 機器學習實用性評估 (TSTR)"""
        print("🔧 6. 機器學習實用性評估 (TSTR)")
        print("-" * 60)
        
        target = 'PHVAC_y'
        if target not in self.numeric_columns:
            print(f"目標變數 '{target}' 不存在，跳過 TSTR 評估。")
            self.results['utility_assessment'] = {'score': 0, 'r2': -999}
            return

        features = [col for col in self.numeric_columns if col != target]
        
        X_train_synth = self.synthetic_data[features]
        y_train_synth = self.synthetic_data[target]
        X_test_real = self.real_data[features]
        y_test_real = self.real_data[target]
        
        model = RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train_synth, y_train_synth)
        
        predictions = model.predict(X_test_real)
        r2 = r2_score(y_test_real, predictions)
        
        # 將 R2 分數轉換為 0-100 的實用性分數
        utility_score = max(0, r2) * 100
        
        utility_grade = "✅優秀" if utility_score > 70 else "⚠️良好" if utility_score > 50 else "❌需改善"
        
        print(f"在仿真資料上訓練，在真實資料上測試的 R² 分數: {r2:.4f}")
        print(f"實用性分數: {utility_score:.1f} / 100")
        print(f"實用性等級: {utility_grade}")
        
        self.results['utility_assessment'] = {'score': utility_score, 'r2': r2}
        print("\n")

    def _multivariate_analysis(self):
        """7. 多變量結構評估 (PCA-based)"""
        print("🌐 7. 多變量結構評估 (PCA)")
        print("-" * 60)
        
        features = self.numeric_columns
        
        scaler = StandardScaler()
        real_scaled = scaler.fit_transform(self.real_data[features])
        synth_scaled = scaler.transform(self.synthetic_data[features])
        
        pca = PCA(n_components=min(len(features), 5)) # 最多取5個主成分
        pca.fit(real_scaled)
        
        real_pca = pca.transform(real_scaled)
        synth_pca = pca.transform(synth_scaled)
        
        pca_results = []
        distances = []
        for i in range(pca.n_components_):
            dist = wasserstein_distance(real_pca[:, i], synth_pca[:, i])
            distances.append(dist)
            pca_results.append({
                '主成分': f"PC-{i+1}",
                '解釋變異比例': f"{pca.explained_variance_ratio_[i]:.2%}",
                '分佈距離(WD)': f"{dist:.4f}"
            })
            
        df_pca_results = pd.DataFrame(pca_results) # 將結果轉為 DataFrame
        print(tabulate(pca_results, headers='keys', tablefmt='grid'))
        
        # 將平均距離轉換為分數 (0-100)
        avg_dist = np.mean(distances)
        # 假設平均距離 > 0.5 意味著結構差異很大
        multivar_score = max(0, 100 * (1 - avg_dist / 0.5)) 
        multivar_grade = "✅優秀" if multivar_score > 85 else "⚠️良好" if multivar_score > 70 else "❌需改善"
        
        print(f"\n平均主成分分佈距離: {avg_dist:.4f}")
        print(f"多變量結構分數: {multivar_score:.1f} / 100")
        print(f"結構相似度: {multivar_grade}")
        
        # 將詳細表格也存入 results
        self.results['multivariate_analysis'] = {'score': multivar_score, 'avg_dist': avg_dist, 'details': df_pca_results}
        print("\n")
        
    def _calculate_overall_scores(self):
        """8. 計算綜合評分 (已更新權重與維度)"""
        print("🏆 8. 綜合評分計算")
        print("-" * 60)
        
        scores = {}
        # 統計準確性
        mean_errors = self.results['descriptive_stats']['均值誤差(%)'].str.replace('%', '').astype(float)
        avg_error = mean_errors.mean()
        scores['統計準確性'] = max(0, 100 - avg_error * 4)

        # 分佈相似性
        dist_scores = []
        stats_df = self.results['descriptive_stats'].set_index('特徵')
        for _, row in self.results['fidelity_metrics'].iterrows():
            w_dist = float(row['Wasserstein距離'])
            real_std = float(stats_df.loc[row['特徵'], '真實標準差'])
            if real_std > 1e-9:
                normalized_dist = w_dist / real_std
                dist_scores.append(max(0, 100 * (1 - normalized_dist)))
        scores['分佈相似性'] = np.mean(dist_scores) if dist_scores else 0

        # HVAC邏輯
        hvac_score = 100
        if 'hvac_specific' in self.results and self.results['hvac_specific'] is not None:
             for _, row in self.results['hvac_specific'].iterrows():
                 if '❌' in str(row['狀態']): hvac_score -= 30
                 elif '⚠️' in str(row['狀態']): hvac_score -= 15
        scores['HVAC邏輯'] = max(0, hvac_score)

        # 時間特性
        scores['時間特性'] = self.results['time_series_analysis']['綜合時間分數'].astype(float).mean()

        # 相關性結構
        scores['相關性結構'] = max(0, 100 * (1 - self.mean_corr_diff / 0.3))

        # 實用性 (TSTR)
        scores['機器學習實用性'] = self.results['utility_assessment']['score']

        # 多變量結構
        scores['多變量結構'] = self.results['multivariate_analysis']['score']

        # 更新後的權重
        weights = {
            '統計準確性': 0.10, '分佈相似性': 0.15, 'HVAC邏輯': 0.20,
            '時間特性': 0.15, '相關性結構': 0.10, '多變量結構': 0.15,
            '機器學習實用性': 0.15
        }
        
        total_score = sum(scores[key] * weights[key] for key in scores)
        grade = "A+(卓越)" if total_score >= 90 else "A(優秀)" if total_score >= 80 else "B(良好)" if total_score >= 70 else "C(合格)" if total_score >= 60 else "D(需改善)"

        score_table = [{'評估維度': dim, '分數': f"{score:.1f}", '權重': f"{weights[dim]:.0%}", '加權分數': f"{score * weights[dim]:.1f}"} for dim, score in scores.items()]
        
        print(tabulate(score_table, headers='keys', tablefmt='grid'))
        print(f"\n{'='*40}\n📊 綜合評分: {total_score:.1f} / 100\n🎖️ 品質等級: {grade}\n{'='*40}")
        
        self.results['overall_scores'] = {'scores': scores, 'weights': weights, 'total_score': total_score, 'grade': grade}
        print("\n")
        
    # --- 輔助函式與視覺化 (為簡潔省略大部分，僅展示必要修改) ---

    def _calculate_fft_diff(self, s1, s2, n_peaks=5):
        """計算FFT差異 (從原 _calculate_tc_score 改名)"""
        min_len = min(len(s1), len(s2))
        s1, s2 = s1[:min_len], s2[:min_len]
        amp1, amp2 = np.abs(np.fft.fft(s1))[:min_len//2], np.abs(np.fft.fft(s2))[:min_len//2]
        peaks1, peaks2 = np.argsort(amp1[1:])[-n_peaks:]+1, np.argsort(amp2[1:])[-n_peaks:]+1
        all_peaks = np.union1d(peaks1, peaks2)
        norm_amp1, norm_amp2 = amp1/(np.sum(amp1)+1e-10), amp2/(np.sum(amp2)+1e-10)
        return np.mean(np.abs(norm_amp1[all_peaks] - norm_amp2[all_peaks]))

    def _generate_visualizations(self):
        """生成所有可視化圖表"""
        print("🎨 9. 生成可視化分析圖表")
        print("-" * 60)
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        # (這裡應包含所有繪圖函式的呼叫)
        self._plot_radar_chart(vis_dir) # 雷達圖需要更新
        # ... 其他繪圖函式 ...
        print(f"✅ 所有圖表已保存至: {vis_dir}\n")

    def _plot_radar_chart(self, output_dir):
        """繪製綜合評分雷達圖 (已更新維度)"""
        if 'overall_scores' not in self.results: return
            
        scores_data = self.results['overall_scores']
        categories = list(scores_data['scores'].keys())
        values = list(scores_data['scores'].values())
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2, color='red', label='實際得分')
        ax.fill(angles, values, alpha=0.25, color='red')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=12)
        ax.set_ylim(0, 100)
        ax.set_title(f"綜合品質評估雷達圖\n總分: {scores_data['total_score']:.1f}/100 ({scores_data['grade']})", 
                     size=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quality_radar.png'), dpi=300)
        plt.close()

    def _generate_comprehensive_report(self):
        """生成綜合文字報告 (已更新)"""
        # --- 💥 關鍵修改！ 💥 ---
        # 原本的路徑是寫死的，現在我們讓它動態生成在指定的 output_dir 中
        # 這樣報告就會跟圖表一起儲存在正確的位置
        report_path = os.path.join(self.output_dir, 'comprehensive_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # ... (後面所有寫入檔案的 f.write(...) 內容都完全不變) ...
            f.write("="*80 + "\nHVAC 仿真資料集綜合評估報告 (優化版)\n" + "="*80 + "\n\n")
            f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"真實資料: {self.real_data_file}\n")
            f.write(f"仿真資料: {self.synthetic_data_file}\n\n")

            # --- 1. 總體評估 ---
            if 'overall_scores' in self.results:
                scores_data = self.results['overall_scores']
                f.write("--- 🏆 總體評估 ---\n")
                f.write(f"綜合評分: {scores_data['total_score']:.1f}/100\n")
                f.write(f"品質等級: {scores_data['grade']}\n\n")
                f.write("各維度得分:\n")
                df_scores = pd.DataFrame({
                    '評估維度': scores_data['scores'].keys(),
                    '分數': [f"{s:.1f}" for s in scores_data['scores'].values()],
                    '權重': [f"{w:.0%}" for w in scores_data['weights'].values()],
                    '加權分數': [f"{s * w:.1f}" for s, w in zip(scores_data['scores'].values(), scores_data['weights'].values())]
                })
                f.write(df_scores.to_string(index=False))
                f.write("\n\n" + "="*80 + "\n\n")
            
            f.write("--- 📜 詳細分析結果 ---\n\n")

            # --- 2. 描述性統計分析 ---
            if 'descriptive_stats' in self.results and not self.results['descriptive_stats'].empty:
                f.write("--- 📈 1. 描述性統計分析 ---\n")
                f.write(self.results['descriptive_stats'].to_string(index=False))
                f.write("\n\n")

            # --- 3. 分佈保真度評估 ---
            if 'fidelity_metrics' in self.results and not self.results['fidelity_metrics'].empty:
                f.write("--- 🎯 2. 分佈保真度評估 (Fidelity Assessment) ---\n")
                f.write(self.results['fidelity_metrics'].to_string(index=False))
                f.write("\n\n")
                
            # --- 4. HVAC系統專屬分析 ---
            if 'hvac_specific' in self.results and not self.results['hvac_specific'].empty:
                f.write("--- 🏭 3. HVAC系統專屬分析 ---\n")
                f.write(self.results['hvac_specific'].to_string(index=False))
                f.write("\n\n")

            # --- 5. 相關性結構分析 ---
            if 'correlation_analysis' in self.results:
                corr_data = self.results['correlation_analysis']
                f.write("--- 🔗 4. 相關性結構分析 ---\n")
                f.write(f"平均相關性差異: {corr_data['mean_diff']:.4f}\n")
                f.write(f"相關性保持程度: {corr_data['grade']}\n\n")

            # --- 6. 時間序列特性分析 ---
            if 'time_series_analysis' in self.results and not self.results['time_series_analysis'].empty:
                f.write("--- ⏰ 5. 時間序列特性分析 ---\n")
                f.write(self.results['time_series_analysis'].to_string(index=False))
                f.write("\n\n")
            
            # --- 7. 機器學習實用性評估 (TSTR) ---
            if 'utility_assessment' in self.results:
                utility_data = self.results['utility_assessment']
                f.write("--- 🔧 6. 機器學習實用性評估 (TSTR) ---\n")
                f.write(f"在仿真資料上訓練，在真實資料上測試的 R² 分數: {utility_data['r2']:.4f}\n")
                f.write(f"實用性分數: {utility_data['score']:.1f} / 100\n\n")

            # --- 8. 多變量結構評估 (PCA) ---
            if 'multivariate_analysis' in self.results:
                multi_data = self.results['multivariate_analysis']
                f.write("--- 🌐 7. 多變量結構評估 (PCA) ---\n")
                if 'details' in multi_data and not multi_data['details'].empty:
                    f.write("主成分分佈比較:\n")
                    f.write(multi_data['details'].to_string(index=False) + "\n\n")
                f.write(f"平均主成分分佈距離: {multi_data['avg_dist']:.4f}\n")
                f.write(f"多變量結構分數: {multi_data['score']:.1f} / 100\n\n")

        print(f"📄 綜合報告已生成 (包含詳細資訊): {report_path}")

# --- 主程式入口 ---
if __name__ == "__main__":
    # --- 這個區塊現在僅供獨立測試使用 ---
    print(">> 警告：此腳本被獨立執行，僅用於測試目的 <<")
    
    # 這裡仍然可以使用寫死的路徑來快速測試這個腳本的功能
    real_data_path = 'data/ChillerFinal500Ping_res.csv'
    synth_data_path = './logs/20250922-223235_也還不錯/hvac_rl_gan_generated.csv' # 測試時請確保此檔案存在

    if os.path.exists(real_data_path) and os.path.exists(synth_data_path):
        print("檢測到資料檔案，開始執行優化版分析...\n")
        analyzer = HVACComprehensiveAnalyzer(real_data_path, synth_data_path)
        # 測試時，結果會存在一個名為 "analysis_results_test" 的資料夾
        analyzer.comprehensive_analysis(output_dir="analysis_results_test") 
    else:
        print(f"❌ 錯誤：請確保 '{real_data_path}' 和 '{synth_data_path}' 檔案存在於當前目錄。")