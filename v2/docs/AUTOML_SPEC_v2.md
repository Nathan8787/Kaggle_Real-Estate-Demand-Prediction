# Real Estate Demand Prediction – AutoML Specification v2 (XGBoost GPU Edition)

## 1. Objectives
- 建立涵蓋 2019-01 至 2025-07 的月 × sector panel，整合官方提供的訓練資料並產生未來月份的外推欄位。
- 依據既定的缺值補值與外推規則建構完整特徵矩陣，避免未來月份特徵退化為 0 或 NaN。
- 使用 GPU 加速的 XGBoost 搭配 `np.log1p` 目標轉換與客製競賽評分指標，並透過 FLAML 在 6 小時內搜尋最適超參數。
- 採用固定的時間序列折 (含 2024 Q2、Q3) 進行交叉驗證，並支援 walk-forward 重新訓練與殘差修正。

## 2. Data Inventory & Integration Rules
| File | Description | Keys | Pre-processing & Integration |
| --- | --- | --- | --- |
| `train/new_house_transactions.csv` | 目標與新屋指標 | `month`, `sector` | 讀入後 join；缺少的月 × sector 組合代表成交金額為 0，於 panel 建立時補上。欄位：`num_new_house_transactions`, `area_new_house_transactions`, `price_new_house_transactions`, `amount_new_house_transactions`, `area_per_unit_new_house_transactions`, `total_price_per_unit_new_house_transactions`, `num_new_house_available_for_sale`, `area_new_house_available_for_sale`, `period_new_house_sell_through`. |
| `train/new_house_transactions_nearby_sectors.csv` | 鄰近區新屋指標 | `month`, `sector` | 讀入後欄位加 `_nearby` 後 join；欄位結構同上。 |
| `train/pre_owned_house_transactions.csv` | 二手市場指標 | `month`, `sector` | 欄位：`num_pre_owned_house_transactions`, `area_pre_owned_house_transactions`, `price_pre_owned_house_transactions`, `amount_pre_owned_house_transactions`. |
| `train/pre_owned_house_transactions_nearby_sectors.csv` | 鄰近區二手指標 | `month`, `sector` | 欄位加 `_nearby` 後 join。 |
| `train/land_transactions.csv` | 土地市場指標 | `month`, `sector` | 欄位：`num_land_transactions`, `construction_area`, `planned_building_area`, `transaction_amount`. |
| `train/land_transactions_nearby_sectors.csv` | 鄰近區土地指標 | `month`, `sector` | 欄位加 `_nearby` 後 join。 |
| `train/sector_POI.csv` | Sector 靜態 POI/density | `sector` | 所有欄位廣播到每個月；欄位包含人口、商業、交通等密度。 |
| `train/city_search_index.csv` | 城市月度搜尋量 | `month`, `keyword`, `source` | 讀入後：1) 將 `keyword` 正規化 (`unicodedata.normalize('NFKC')`) 及 slug 化 (`unidecode` → `replace(' ', '_').lower()`)，保留 `keyword_original`；2) 以 `keyword_slug` 群組並依月排序執行 `ffill` → `bfill`；3) 使用 `rolling(window=3, min_periods=1).mean()` 產生 `search_kw_<slug>`；4) 取方差最高的 30 個 slug，pivot 成 `month` × `search_kw_*`；5) 缺值補 0 並產生 `search_kw_<slug>_was_missing`。 |
| `train/city_indexes.csv` | 城市年度指標 | `city_indicator_data_year` | 每欄位依年份排序後計算 CAGR。若近 3 年資料不足則使用全部歷史資料；含 0 或負值時改用平均年增量。使用最後一年資料外推 2023、2024、2025，超出部分前述結果前向填補。外推或填補的欄位加 `city_<col>_was_interpolated`。 |
| `test.csv` | 測試樣本 | `id` | 測試期間：2024-08 ~ 2025-07；僅用於生成 submission 順序。 |
| `sample_submission.csv` | 範例提交 | `id`, `new_house_transaction_amount` | 生成最終 submission 時維持相同排序。 |

- 所有 CSV 以 UTF-8 讀入，`month` 以 `%Y-%b` 或 `dateutil` 解析為月首日期。
- 把 `sector` 欄位保留原字串，另擷取 `sector_id = int(re.search(r"(\\d+)", sector).group(1))`。

## 3. Environment & Tooling
- Python 3.13.2，強制 `PYTHONUTF8=1`。
- 裝好下列 pip 套件：`pandas`, `numpy`, `pyarrow`, `scikit-learn`, `scipy`, `Unidecode`, `PyYAML`, `joblib`, `xgboost>=2.0.0`, `flaml>=2.1.2`, `catboost`, `lightgbm`, `pytest`。
- GPU 驅動：需確認 `nvidia-smi` 可執行並顯示至少一張 GPU，CUDA toolkit 已安裝。
- logging 採 JSON 檔案，無需其他 logging 套件。

## 4. Panel Builder (`src/panel_builder_v2.py`)
1. `load_raw_tables_v2(config_path)` 讀取 YAML 指定路徑：
   - 所有表以 UTF-8 讀入，依欄位分類設定 `dtype`（`month`→string、計數欄→float）。
   - 欄名統一 snake_case；物件欄位去除首尾空白。
   - `city_search_index` 新增 `keyword_original` 與 slug (`keyword_slug`)，`source` 正規化。
2. `_expand_city_indexes_with_growth(city_index_df, months)`：
   - 取最近三個年度 (不足三年則使用全部) 計算 CAGR：`cagr = (value[last] / value[first]) ** (1/(n_years-1))`。若任一年度≤0，改為平均年增量 `mean_delta`。
   - 對 2023、2024、2025 外推：`value[year] = value[year-1] * cagr` 或 `value[year-1] + mean_delta`。
   - 超過最後一年後使用 `ffill`。外推的欄位在對應列標記 `city_<col>_was_interpolated = 1`，其餘為 0。
3. `_build_search_features_v2(search_df, months)`：
   - 依 `month`, `keyword_slug` 彙總 `search_volume`，取方差最高的 30 個 slug (不足 30 則取全部)。
   - 對每個 slug：`series = groupby('keyword_slug')` → `ffill()` → `bfill()` → `rolling(window=3, min_periods=1).mean()`。
   - Pivot 成 `month` × `search_kw_<slug>`，缺失補 0；建立 `search_kw_<slug>_was_missing` (原始值在當月缺 → 1)。
4. `_extend_monthly_tables(raw_tables, months)`：
   - 對 `new_house`, `pre_owned`, `land` 及對應 `_nearby` 表逐 sector 操作：
     1. 建立 6 個月移動平均 (至少 4 個非 NaN)，用於 2024-08 之後外推。
     2. 若 6 月不符合條件，改用 3 個月移動平均 (至少 2 個非 NaN)。
     3. 仍不足時，採用同月城市中位數 (所有 sector)；若仍缺則使用全期間中位數。
     4. 無任何歷史時填 0。
     5. 新增 `<column>_synthetic = 1` 表示外推，其他為 0。
5. `build_calendar(raw_tables)`：
   - Calendar 範圍固定為 2019-01-01 至 2025-07-01，每月第一天。
   - 解析 `sector_id = int`；若無法解析， raise `ValueError`。
6. `merge_sources(calendar_df, raw_tables)`：
   - 依順序 merge：new_house → new_house_nearby → pre_owned → pre_owned_nearby → land → land_nearby → sector_poi → city_indexes (外推結果) → search_features。
   - 所有 `_synthetic`、`_was_missing`、`city_*_was_interpolated` flag 一併保留。
7. `attach_target_v2(panel_df)`：
   - `target` = `amount_new_house_transactions`，若缺值視為 0。
   - `target_filled` = `target` (補 0 後值)，供 lag/rolling 使用。
   - `is_future` = `month > '2024-07-01'`。
8. `save_panel(panel_df, panel_path)`：刪除舊檔，依 `year` partition 輸出 `data/processed/panel_v2.parquet`。

## 5. Missing Value Policy (`src/features_v2.py`)
- 所有需補值欄位先新增 `*_was_missing = df[col].isna().astype(np.int8)`。
- `target_filled`：保持 0 填補；新增 `target_filled_was_missing`。
- 計數/金額/面積/密度欄位 (`num_`, `area_`, `amount_`, `construction_`, `transaction_`, `planned_building_`, `number_of_`, `_dense`)
  1.  sector 內 `ffill()`。
  2.  sector rolling median (`window=12`, `min_periods=3`) 補值。
  3.  月內城市中位數。
  4.  全期間中位數。
  5.  仍缺者填 0。
- 價格/比例欄 (`price` 但非 `city_`)：sector `ffill()` → `_fill_with_sector_history` (排序後逐列取過去 median) → 全期間 median → 若仍缺補 0。
- 期間欄 (`period_`)：sector `ffill()` → sector median → 全期間 median。
- 城市欄 (`city_`)：sector `ffill()` → month `ffill()` → 全期間 median。
- 搜尋欄 (`search_kw_*`)：rolling 補值後無資料者填 0，`was_missing` 旗標保持 1。
- 補值統計寫入 `reports_v2/missing_value_report.json`，每欄位需記錄填補策略與各步驟填補數量。

## 6. Feature Engineering (`src/features_v2.py`)
1. `month` 轉為 `datetime64[ns]`，DataFrame 依 `sector_id`, `month` 排序。
2. 時間特徵：`year`, `month_num`, `quarter`, `month_index = 12*(year-2019) + (month_num-1)`, `is_year_start`, `is_year_end`, `days_in_month`。
3. Lag/rolling：對 `target_filled` 與所有 `num_/area_/amount_/price_/total_price_` 欄位計算：
   - Lag：1, 3, 6, 12；若缺值，lag 欄會為 NaN。
   - Rolling：mean, std, min, max, count，window = 3, 6, 12，`min_periods=1`；新增 `rolling_count` 對應實際樣本數。
4. 成長率：
   - `growth_1 = (value - lag_1) / (abs(lag_1) + 1e-6)`；
   - `growth_3 = (value - lag_3) / (abs(lag_3) + 1e-6)`；
   - 若 lag 為 NaN，則 growth = 0 並設定 `growth_*_was_missing = 1`。
5. 人口權重 share：
   - 欄位 `weight = resident_population`，若 ≤0 或 NaN 則改用 `population_scale`；仍缺則取 1.0。
   - 以 month 為單位計算加權平均：`city_weighted_mean = sum(value*weight)/sum(weight)`，權重全為 0 時改用月平均。
   - share = `value / (city_weighted_mean + 1e-6)`。
6. 互動欄位：針對 `_nearby` 結尾欄位產生 `diff = base - nearby`。
7. 搜尋衍生欄位：
   - `search_kw_<slug>_pct_change`：依 sector 分組計算 `(curr - lag1) / (abs(lag1)+1e-6)`。
   - `search_kw_<slug>_zscore_3m`：使用 3 月 rolling mean/std 的 z-score（缺值填 0）。
   - 對以 `_dense` 結尾欄位建立 `density_adj = dense * mean(zscores)`。
8. `np.log1p`：對所有計數/密度欄及 `target_filled` 落單欄加 `_log1p` 字尾。
9. POI PCA：取 log 後密度欄位 (含 `_dense_log1p`)，若欄位數 ≥ 2 且總變異 > 0，執行 PCA 保留累積變異 ≥ 95% 的主成分，命名 `poi_pca_1`…；否則跳過。
10. 結果輸出 `data/processed/features_v2.parquet`（按年 partition），並在 `reports_v2/feature_metadata.json` 紀錄所有新增欄位。

## 7. Target Transformation & Scaling
- y 轉換：
  - `y_raw = features_df['target']`
  - `y_log = np.log1p(y_raw)`
  - 所有模型訓練與 cross-validation 均使用 `y_log`。
- 評估：
  - 模型輸出 `y_pred_log` → `y_pred = np.clip(np.expm1(y_pred_log), 0.0, None)`。
  - 真實值 `y_true_log` 同步 `np.expm1` 還原；評分在原尺度計算。
- `StandardScaler`：
  - 每折先 fit 訓練子集特徵，保存 `reports_v2/scaler_fold_<k>.json`（包含 mean, scale）。
  - 最終在全部訓練資料 (`month <= 2024-07`) 上 fit，輸出 `models_v2/scaler.pkl`。
- `reports_v2/automl_summary.json` 需包含：
  ```json
  {
    "best_estimator": "xgboost",
    "best_fold": 7,
    "target_transform": "log1p_expm1",
    "best_config": { ... },
    "folds": [ ... ]
  }
  ```

## 8. Dataset Splitting (`src/splits_v2.py`)
```
FOLD_DEFINITIONS_V2 = [
  ("2019-01-01", "2022-12-31", "2023-01-01", "2023-03-31"),
  ("2019-01-01", "2023-03-31", "2023-04-01", "2023-06-30"),
  ("2019-01-01", "2023-06-30", "2023-07-01", "2023-09-30"),
  ("2019-01-01", "2023-09-30", "2023-10-01", "2023-12-31"),
  ("2019-01-01", "2023-12-31", "2024-01-01", "2024-03-31"),
  ("2019-01-01", "2024-03-31", "2024-04-01", "2024-05-31"),
  ("2019-01-01", "2024-05-31", "2024-06-01", "2024-07-31"),
]
```
- `generate_time_series_folds_v2(features_df)` 先依 `month`, `sector_id` 排序並保留原 index，返回 7 組 `(train_idx, valid_idx)`。
- 每組 `train_idx` 包含從第一個月至 validation_start-1 全部觀測。

## 9. Custom Metric (`src/metrics_v2.py`)
```python
import numpy as np

def competition_score(y_true_raw, y_pred_raw):
    eps = 1e-9
    y_true = np.asarray(y_true_raw, dtype=float)
    y_pred = np.asarray(y_pred_raw, dtype=float)
    mask = ~np.isnan(y_true)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    ape = np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))
    if (ape > 1.0).mean() > 0.30:
        return 0.0
    mask_valid = ape <= 1.0
    if mask_valid.sum() == 0:
        return 0.0
    scaled_mape = ape[mask_valid].mean() / mask_valid.mean()
    return 1.0 - scaled_mape

def competition_metric(X_val, y_val_log, estimator, labels, X_train, y_train, weight_val=None, weight_train=None, *args, **kwargs):
    y_pred_log = estimator.predict(X_val)
    y_true = np.clip(np.expm1(np.asarray(y_val_log, dtype=float)), 0.0, None)
    y_pred = np.clip(np.expm1(y_pred_log), 0.0, None)
    score = competition_score(y_true, y_pred)
    loss = 1.0 - score
    return loss, {'competition_score': score}
```
- 所有評估流程（cross-validation、最終報表、walk-forward）均使用上述還原與 clip。

## 10. Training (`src/automl_runner_v2.py`)
1. 載入 `config/automl_v2.yaml`：
   ```yaml
   seed: 42
   time_budget: 21600
   metric: competition_score
   estimator_list: [xgboost]
   n_jobs: -1
   fit_kwargs_by_estimator:
     xgboost:
       tree_method: gpu_hist
       predictor: gpu_predictor
       gpu_id: 0
   search_space:
     xgboost:
       n_estimators:
         _type: randint
         _value: [100, 4000]
       max_depth:
         _type: randint
         _value: [3, 16]
       learning_rate:
         _type: loguniform
         _value: [0.005, 0.3]
       subsample:
         _type: uniform
         _value: [0.4, 1.0]
       colsample_bytree:
         _type: uniform
         _value: [0.4, 1.0]
       min_child_weight:
         _type: loguniform
         _value: [0.001, 50]
       reg_alpha:
         _type: loguniform
         _value: [1e-4, 50]
       reg_lambda:
         _type: loguniform
         _value: [1e-4, 50]
   ```
2. `run(features_df, folds, config_path)`：
   - 過濾 `train_df = features_df[month <= '2024-07-31']`。
   - `X_full, y_raw, feature_cols = _prepare_features_v2(train_df)`；`y_full_log = np.log1p(y_raw)`。
   - 對每折：
     - 擷取 `fold_X`, `fold_y`, 建立 `fold_y_log = np.log1p(fold_y)`。
     - `StandardScaler` fit 訓練位置；保存至 `reports_v2/scaler_fold_<k>.json`。
     - 建立 `SingleFoldSplitter(train_positions, valid_positions)`。
     - `AutoML().fit` 設定：
       ```python
       automl_settings = {
           'time_budget': config['time_budget'],
           'metric': competition_metric,
           'task': 'regression',
           'estimator_list': config['estimator_list'],
           'log_file_name': str(logs_path),
           'seed': config['seed'],
           'n_jobs': config['n_jobs'],
           'eval_method': 'cv',
           'split_type': splitter,
           'fit_kwargs_by_estimator': config['fit_kwargs_by_estimator'],
           'custom_hp': config['search_space'],
           'verbose': 1,
       }
       automl.fit(X_train=fold_X_scaled, y_train=fold_y_log, **automl_settings)
       ```
     - 驗證：`y_pred_log = automl.predict(...)` → `y_pred = np.clip(np.expm1(y_pred_log), 0.0, None)`；`y_true = np.clip(np.expm1(fold_y_log.iloc[valid_positions]), 0.0, None)`；計算 `competition_score`。
     - 將 fold 成績保存至 `reports_v2/fold_results.csv`。
   - `automl.best_config` 需去除非模型參數後保存為 `models_v2/best_config.json`。
3. 最終模型：
   - `global_scaler = StandardScaler().fit(X_full)` → `models_v2/scaler.pkl`。
   - `final_params = {**best_config, **config['fit_kwargs_by_estimator']['xgboost']}`。
   - `final_model = XGBRegressor(**final_params).fit(global_scaler.transform(X_full), y_full_log)`。
   - 儲存 `models_v2/best_model.pkl`。
   - `reports_v2/automl_summary.json` 寫入最佳設定與折數資訊。

## 11. Prediction (`src/predict_v2.py`)
1. 讀取 `panel_v2.parquet`、`features_v2.parquet`、`models_v2/scaler.pkl`、`models_v2/best_model.pkl`、`models_v2/best_config.json`、`reports_v2/automl_summary.json`。
2. `forecast_months = sorted(panel_df.loc[panel_df['is_future'], 'month'].unique())`。
3. 逐月執行：
   - 以最新 `working_panel` 呼叫 `build_feature_matrix_v2` 取得當月特徵。
   - `X_iter_scaled = scaler.transform(X_iter)` → `preds_log = model.predict(X_iter_scaled)` → `preds = np.clip(np.expm1(preds_log), 0.0, None)`。
   - 若提供殘差修正 (`models_v2/residual_model.pkl`)，則載入 `residual_scaler.pkl` 與 `residual_features.json`，對當月特徵套用修正：
     ```python
     residual_features = metadata['features']
     residual_X = residual_scaler.transform(current_features[residual_features])
     preds += residual_model.predict(residual_X)
     preds = np.clip(preds, 0.0, None)
     ```
   - 更新 `working_panel[target_filled == month] = preds`，並寫入 `target`。
   - 保存 `preds` 與 `id` 至列表。
4. 合併所有月的預測，與 `sample_submission.csv` 基於 `id` merge，若存在 NaN raise error。
5. 產出 `submission_v2/submission.csv`，並撰寫 `reports_v2/prediction_summary.json` (count, mean, std, min, max)。

## 12. Walk-forward Retraining (`scripts/walk_forward_train.py`)
- 預設關閉；如需啟用：
  1. 從 `models_v2/best_config.json` 讀取最佳參數。
  2. For each `forecast_month`：
     - `train_df = features_v2[month <= forecast_month - 1 month]`
     - `StandardScaler().fit(train_features)`；`XGBRegressor(**best_config, **gpu_args)` fit `np.log1p(train_target)`。
     - 保存 `models_v2/scaler_<month>.pkl`、`models_v2/best_model_<month>.pkl`。
     - 使用 `predict_v2` 的核心函式在該月產生預測，回寫 `target_filled`。
  3. 產出 `reports_v2/walk_forward_summary.json` 參考各月預測統計。

## 13. Residual Correction (`src/residual_correction.py`)
1. 載入 `reports_v2/fold_results.csv`，計算 `residual = target - prediction`。
2. 特徵集合：`['prediction', 'month_index', 'target_city_weighted_mean', 'target_share', ...]`（需在程式內固定清單）。
3. 使用 `StandardScaler` 對上述特徵做標準化；保存為 `models_v2/residual_scaler.pkl`。
4. 訓練 `sklearn.linear_model.Ridge(alpha=1.0)`，保存為 `models_v2/residual_model.pkl`。
5. 將特徵欄位清單存於 `reports_v2/residual_features.json`：`{"features": [...]}。

## 14. CLI (`run_pipeline_v2.py`)
提供下列子命令與參數：
1. `build_panel`：
   ```bash
   python run_pipeline_v2.py build_panel \
       --data-config config/data_paths_v2.yaml \
       --panel-path data/processed/panel_v2.parquet
   ```
2. `build_features`：
   ```bash
   python run_pipeline_v2.py build_features \
       --panel-path data/processed/panel_v2.parquet \
       --features-path data/processed/features_v2.parquet \
       --reports-dir reports_v2
   ```
3. `run_xgb`：
   ```bash
   python run_pipeline_v2.py run_xgb \
       --features-path data/processed/features_v2.parquet \
       --config config/automl_v2.yaml
   ```
4. `predict`：
   ```bash
   python run_pipeline_v2.py predict \
       --panel-path data/processed/panel_v2.parquet \
       --features-path data/processed/features_v2.parquet \
       --model-path models_v2/best_model.pkl \
       --scaler-path models_v2/scaler.pkl \
       --sample-submission sample_submission.csv \
       --output-path submission_v2/submission.csv \
       [--residual-model models_v2/residual_model.pkl \
        --residual-scaler models_v2/residual_scaler.pkl \
        --residual-features reports_v2/residual_features.json]
   ```
5. `walk_forward_train` (選用)：
   ```bash
   python run_pipeline_v2.py walk_forward_train \
       --panel-path data/processed/panel_v2.parquet \
       --features-path data/processed/features_v2.parquet \
       --config config/automl_v2.yaml
   ```

## 15. Artifacts & Directories
```
project/
├── config/
│   ├── automl_v2.yaml
│   └── data_paths_v2.yaml
├── data/processed/
│   ├── panel_v2.parquet
│   └── features_v2.parquet
├── logs_v2/automl.log
├── models_v2/
│   ├── scaler.pkl
│   ├── best_model.pkl
│   ├── best_config.json
│   ├── residual_model.pkl (若訓練殘差校正)
│   ├── residual_scaler.pkl (若訓練殘差校正)
│   └── best_model_<YYYY-MM>.pkl (walk-forward 時產生)
├── reports_v2/
│   ├── automl_summary.json
│   ├── best_config.json (若需備份)
│   ├── fold_results.csv
│   ├── missing_value_report.json
│   ├── feature_metadata.json
│   ├── prediction_summary.json
│   └── residual_features.json (若訓練殘差校正)
├── submission_v2/submission.csv
├── scripts/
│   ├── run_pipeline_v2.py
│   ├── walk_forward_train.py
│   └── residual_correction.py
└── docs/AUTOML_SPEC_v2.md
```

## 16. Testing & Verification
1. 語法檢查：`python -m py_compile src/**/*.py scripts/**/*.py`
2. 單元測試 (需撰寫)：
   - `pytest tests/test_panel_builder_v2.py::test_extend_monthly_tables_fills_future_months`
   - `pytest tests/test_features_v2.py::test_missing_value_policy`
   - `pytest tests/test_metrics_v2.py::test_competition_metric_log_handling`
3. Smoke test：
   ```bash
   python run_pipeline_v2.py build_panel --data-config config/data_paths_v2.yaml
   python run_pipeline_v2.py build_features --panel-path data/processed/panel_v2.parquet --features-path data/processed/features_v2.parquet --reports-dir reports_v2
   python run_pipeline_v2.py run_xgb --features-path data/processed/features_v2.parquet --config config/automl_v2.yaml
   python run_pipeline_v2.py predict --panel-path data/processed/panel_v2.parquet --features-path data/processed/features_v2.parquet --model-path models_v2/best_model.pkl --scaler-path models_v2/scaler.pkl --sample-submission sample_submission.csv --output-path submission_v2/submission.csv
   ```
4. 如需殘差校正，於 smoke test 前執行 `python scripts/residual_correction.py`，後續 `predict` 命令加上 `--residual-model` 等參數。

---
以上規格為 v2 版本之唯一實作標準，不得另行更改工具或流程。所有開發、部署與驗證皆必須依照此文件執行。
