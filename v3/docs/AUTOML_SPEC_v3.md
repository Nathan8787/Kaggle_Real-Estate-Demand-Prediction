# Real Estate Demand Prediction – AutoML Specification v3

## 1. 核心目標
1. **校正訓練/推論落差**：所有訓練、交叉驗證、留存月與最終推論資料皆使用相同的前瞻特徵投影流程，避免再出現 v2 中「驗證接近 1.0、實測僅 0.47」的落差。

2. **控制特徵維度**：最終輸出特徵總數必須落在 552±15（即 537–567），且若超出區間立即拋出錯誤；允許名單與分類資訊必須記錄於報告供審計。
3. **資料安全界線**：任何模型、殘差修正或投影程序不得使用 2024-07 之後的真實標籤；投影僅可基於 `month <= 2024-07` 的觀測。
4. **完整審計鏈**：對每一步驟輸出指標、特徵清單、投影誤差及模型設定，確保日後能追溯「投影 → 特徵 → 訓練 → 推論」的每一環。

---

## 2. 資料來源與面板建構

### 2.1 檔案清單與欄位
| 檔案 | 鍵值 | 主要欄位 | 備註 |
| --- | --- | --- | --- |
| `../train/new_house_transactions.csv` | `month`, `sector` | 新建案本地 9 欄 | 目標來源 |
| `../train/new_house_transactions_nearby_sectors.csv` | 同上 | 新建案鄰近 9 欄 |  |
| `../train/pre_owned_house_transactions.csv` | 同上 | 二手房本地 4 欄 |  |
| `../train/pre_owned_house_transactions_nearby_sectors.csv` | 同上 | 二手房鄰近 4 欄 |  |
| `../train/land_transactions.csv` | 同上 | 土地本地 4 欄 |  |
| `../train/land_transactions_nearby_sectors.csv` | 同上 | 土地鄰近 4 欄 |  |
| `../train/sector_POI.csv` | `sector` | 9 個核心 POI 欄 + 其他 `_dense` 欄位 | `_dense` 僅供 PCA |
| `../train/city_search_index.csv` | `month`,`keyword`,`source` | `search_volume` | 僅用到 2024-07 |
| `../train/city_indexes.csv` | `city_indicator_data_year` | 48 個年度指標 | 僅保留精選 12 欄 |
| `../test.csv` | `id` | 2024-08 ~ 2024-12 | |
| `../sample_submission.csv` | `id` | 提交格式 | |

所有 CSV 以 `encoding='utf-8'` 讀取，欄位命名轉成 snake_case，文字欄位做 NFKC 正規化並去除前後空白。

### 2.2 `panel_builder_v3.py`
建立模組 `src/panel_builder_v3.py`，對應函式需具備型別標註、docstring 與 logging。

1. **常數**
   ```python
   MONTH_RANGE = pd.date_range('2019-01-01', '2024-12-01', freq='MS')
   TARGET_CUTOFF = pd.Timestamp('2024-07-31')
   FORECAST_MONTHS = pd.date_range('2024-08-01', '2024-12-01', freq='MS')
   ```
2. **`load_raw_tables_v3(config_path: Path) -> dict[str, pd.DataFrame]`**
   - 讀取 `config/data_paths_v3.yaml`。除路徑解析外，流程等同 v2。
   - 回傳字典鍵值固定：`new_house_transactions`, `new_house_transactions_nearby_sectors`, `pre_owned_house_transactions`, `pre_owned_house_transactions_nearby_sectors`, `land_transactions`, `land_transactions_nearby_sectors`, `sector_poi`, `city_search_index`, `city_indexes`, `test`, `sample_submission`。
3. **`build_calendar_v3(raw_tables)`**
   - 取所有出現過的 `sector`，使用 `pd.MultiIndex.from_product([MONTH_RANGE, sectors])`。
   - `sector_id = int(re.search(r'(\d+)', sector).group(1))`；若解析失敗直接 `ValueError`。
4. **`merge_sources_v3(calendar_df, raw_tables)`**
   - 僅合併實際觀測值（不再在此階段外插未來月份）；即把 v2 的 `_extend_monthly_tables` 功能移除。
   - 合併順序：六個月度表 → `sector_poi` → `city_indexes`（年資料轉月、forward fill）→ `city_search_index`（維持原始觀測，不做變異數篩選）。
   - 合併後保留 `sector` 文字欄位供稽核，於最終輸出前刪除。
5. **`attach_target_v3(panel_df)`**
   - `target_raw = amount_new_house_transactions`。
   - `target_available_flag = (month <= TARGET_CUTOFF).astype(np.int8)`。
   - `target = np.where(target_available_flag == 1, target_raw, np.nan)`.
   - `target_filled = target_raw.fillna(0.0)`（僅供歷史 lag；未來月份維持 0 以便預測後更新）。
   - `target_filled_was_missing = target_raw.isna().astype(np.int8)`。
   - `is_future = (month > TARGET_CUTOFF).astype(np.int8)`。
   - 生成 `id = month.strftime('%Y %b') + '_sector ' + sector_id.astype(str)`。
6. **`save_panel_v3(panel_df, panel_path)`**
   - 加入 `year = month.dt.year`，輸出至 `data/processed/panel_v3.parquet`（依年份分割）。
   - 另寫 `reports_v3/panel_build_summary.json`：包含 `month_min`, `month_max`, `sector_count`, `labeled_rows`, `future_rows`, `missing_target_ratio`。
7. **清理**
   - 在返回特徵階段前即刪除 `sector` 字串欄位。

---

## 3. 前瞻特徵投影（Projection）

### 3.1 模組結構
新增 `src/projection_v3.py`，公開函式：

- `apply_projection(panel_df: pd.DataFrame, forecast_start: pd.Timestamp) -> pd.DataFrame`
- `collect_projection_drift(original_df, projected_df, forecast_start) -> dict`

### 3.2 投影欄位集合
```
PROJECTION_COLUMNS = [
    新建案本地 9 欄,
    新建案鄰近 9 欄,
    二手房本地 4 欄,
    二手房鄰近 4 欄,
    土地本地 4 欄,
    土地鄰近 4 欄,
    全部 search_kw_<slug> 欄位（後續特徵選擇時再篩 12 個），
    city 指標（後續精選 12 欄）
]
```
POI 靜態欄、時間欄與 `target_*` 不在投影範圍。

### 3.3 投影演算法
對每個欄位、每個 `sector_id` 依時間排序後執行：

1. **歷史截取**：僅使用 `month < forecast_start` 且非 NaN 的觀測。
2. **首選**（資料量 ≥ 12）：計算過去 12 筆的指數加權平均  
   `weights = 0.7 ** np.arange(len(history))[::-1]`，正規化後乘上值。
3. **次選**（6 ≤ 資料量 < 12）：使用簡單平均。
4. **第三層**（3 ≤ 資料量 < 6）：使用中位數。
5. **月度共享**（資料量 < 3）：使用同月份（`month.dt.month`）且 `month < forecast_start` 的跨 sector 中位數。
6. **全域備援**：若仍缺，使用整體歷史中位數；最後保底 0。
7. 任何負值經 `np.clip(value, 0, None)` 截為 0。
8. 寫入 `*_proj_source` 欄位：`0` 表示觀測值、`1` 表示第 2 步（EWMA）、`2` 表示平均/中位數、`3` 表示月度共享或全域備援。
9. 若原始值在 `forecast_start` 之後仍存在（理論上不會），以投影覆蓋並標記 `*_proj_overridden = 1`。

### 3.4 API 約定
- `apply_projection` 回傳與輸入同欄位的 DataFrame，惟 `PROJECTION_COLUMNS` 在 `month >= forecast_start` 的欄位值被投影結果覆蓋，並新增對應的 `*_proj_source` / `*_proj_overridden`。
- `collect_projection_drift` 比較同一 `forecast_start` 下的「投影值 vs. 原始值」，計算 MAE、MAPE、RMSE、P10、P90 等指標，輸出到 `reports_v3/projection_drift_<YYYYMM>.json`；若 `forecast_start` 尚未有真實值，則寫出空指標並註記 `actual_available = False`。

### 3.5 使用時機
- 交叉驗證：`forecast_start = validation_start`。
- 留存月：`forecast_start = pd.Timestamp('2024-07-01')`。
- 最終推論：`forecast_start = pd.Timestamp('2024-08-01')`，並在每個月推論後把預測寫回 `target_filled` 再次呼叫投影以維持自我迭代。

---

## 4. 特徵工程 (`src/features_v3.py`)

### 4.1 入口
`build_feature_matrix_v3(panel_path, forecast_start, features_path, reports_dir) -> tuple[pd.DataFrame, dict]`

1. 載入 `panel_v3.parquet`，套用 `apply_projection(panel_df, forecast_start)`。
2. 僅保留 `month <= TARGET_CUTOFF` 的真實標籤；投影後的月份 `target` 維持 NaN。
3. 執行缺值旗標與補值流程後，再產生派生特徵。
4. 寫入 `data/processed/features_v3/<YYYYMM>.parquet`（依年份分割）與 `reports_v3/feature_inventory_v3.json`。

### 4.2 原始欄位允許名單
- 月度觀測（34 欄）：
  - 新建案本地：`amount_new_house_transactions`, `num_new_house_transactions`, `area_new_house_transactions`, `price_new_house_transactions`, `area_per_unit_new_house_transactions`, `total_price_per_unit_new_house_transactions`, `num_new_house_available_for_sale`, `area_new_house_available_for_sale`, `period_new_house_sell_through`。
  - 新建案鄰近：上述 + `_nearby_sectors`。
  - 二手房本地：`amount_pre_owned_house_transactions`, `num_pre_owned_house_transactions`, `area_pre_owned_house_transactions`, `price_pre_owned_house_transactions`。
  - 二手房鄰近：上述 + `_nearby_sectors`。
  - 土地本地：`transaction_amount`, `num_land_transactions`, `construction_area`, `planned_building_area`。
  - 土地鄰近：上述 + `_nearby_sectors`。
- 目標欄位：`target`, `target_filled`, `target_available_flag`, `target_filled_was_missing`, `is_future`。
- POI 核心：`resident_population`, `resident_population_dense`, `population_scale`, `population_scale_dense`, `office_population`, `office_population_dense`, `surrounding_housing_average_price`, `surrounding_shop_average_rent`, `sector_coverage`。
- 時間：`year`, `month_num`, `quarter`, `month_index`, `days_in_month`, `is_year_start`, `is_year_end`。
- `sector_id`（整數）。
- 城市指標：僅保留以下 12 欄（及對應 `was_interpolated` 旗標）：  
  `city_gdp_100m`, `city_secondary_industry_100m`, `city_tertiary_industry_100m`, `city_gdp_per_capita_yuan`, `city_total_households_10k`, `city_year_end_resident_population_10k`, `city_total_retail_sales_of_consumer_goods_100m`, `city_per_capita_disposable_income_absolute_yuan`, `city_annual_average_wage_urban_non_private_employees_yuan`, `city_number_of_universities`, `city_hospital_beds_10k`, `city_number_of_operating_bus_lines`。
- Search 原始欄位：投影後再選出 12 個 slug（第 4.6 節）。

### 4.3 缺值旗標與補值
1. 對第 4.2 中所有月度欄位建立 `*_was_missing` (`np.int8`)。
2. 補值策略（先依 `sector_id` 分組再排序）：
   - `num_`, `area_`, `amount_`, `transaction_`, `planned_building_`：`ffill()` → 12 月滾動中位數 (`min_periods=3`) → 全資料中位數 → 0。
   - `price_`, `total_price_`, `period_`：`ffill()` → 分組中位數 → 全資料中位數 → 0。
   - 城市指標：`ffill()` → 月度中位數 → 全資料中位數。
   - Search 欄位：在投影階段已處理缺值，仍為 NaN 則填 0。
3. 任何補值步驟寫入 `reports_v3/missing_value_report.json`，格式同 v2，但新增欄位 `projection_stage`（`observed` 或 `forecast`）。

### 4.4 指標集合
```
METRIC_SET_FULL = 上述 34 個月度欄位
METRIC_SET_LONG = [
  amount_new_house_transactions, num_new_house_transactions, price_new_house_transactions,
  amount_new_house_transactions_nearby_sectors, num_new_house_transactions_nearby_sectors, price_new_house_transactions_nearby_sectors,
  amount_pre_owned_house_transactions, num_pre_owned_house_transactions,
  amount_pre_owned_house_transactions_nearby_sectors, num_pre_owned_house_transactions_nearby_sectors,
  transaction_amount, transaction_amount_nearby_sectors
]
GROWTH_METRICS = [
  amount_new_house_transactions, num_new_house_transactions, price_new_house_transactions,
  amount_pre_owned_house_transactions, num_pre_owned_house_transactions, transaction_amount
]
SHARE_METRICS = [
  amount_new_house_transactions, num_new_house_transactions, price_new_house_transactions,
  amount_new_house_transactions_nearby_sectors, num_new_house_transactions_nearby_sectors,
  amount_pre_owned_house_transactions, num_pre_owned_house_transactions,
  transaction_amount, transaction_amount_nearby_sectors, num_land_transactions
]
```

### 4.5 派生特徵
1. **Lag / Rolling**
   - `lag_1`, `lag_3`, `lag_6` for `METRIC_SET_FULL`。
   - `lag_12` for `METRIC_SET_LONG`。
   - `rolling_mean_3`, `rolling_mean_6` for `METRIC_SET_FULL`；`rolling_mean_12` for `METRIC_SET_LONG`。計算前先 `shift(1)`，`min_periods=2`。
   - `rolling_std_3`, `rolling_std_6` for `METRIC_SET_FULL`，`ddof=0`。
   - 缺值再次 `ffill()`（同 sector），確保無 NaN。
2. **Growth**
   - `growth_1`、`growth_3` 對 `GROWTH_METRICS`：`(value - lag) / (abs(lag) + 1e-6)`；lag 缺值時填 0，並加 `*_growth_*_was_missing`.
3. **Share / Weighted Mean**
   - 使用 `population_weight = np.where(resident_population>0, resident_population, np.where(population_scale>0, population_scale, 1.0))`。
   - `metric_city_weighted_mean` = 加權平均（權重和為 0 則使用全體平均）。
   - `metric_share = value / (metric_city_weighted_mean + 1e-6)`。
4. **Log1p**
   - 對 `METRIC_SET_FULL` 及其 lag/rolling/growth/weighted_mean/share 統一建立 `_log1p` 版本（避免重覆計算；使用 `np.log1p(np.clip(value, 0, None))`）。
5. **Search 關鍵字**
   - 根據 `search_kw_<slug>` 的 `variance`（計算範圍：`month <= TARGET_CUTOFF`）排序，選取前 12 個 slug，寫入 `reports_v3/search_keywords_v3.json`。
   - 對每個保留 slug：保留原始值與 `lag_1`, `rolling_mean_3`, `pct_change_1`, `zscore_3m`（以 `rolling(window=3, min_periods=1).mean()` 與標準差計算 z-score）；並保留 `_was_missing`。
6. **POI PCA**
   - 將 `sector_POI` 中除核心欄位外的 `_dense`/`number_of_` 欄位以 0 補值後取 `np.log1p`。
   - 使用 `sklearn.decomposition.PCA`，保留前兩個成分（若第二成分解釋率不足 5% 仍保留以維持固定欄位數）。輸出 `poi_pca_1`, `poi_pca_2`，並寫 `reports_v3/poi_pca_summary.json`（包含原始欄位數、解釋率）。
7. **欄位清單與配額**
   - 彙整所有欄位分類，格式：
     ```json
     {
       "total_features": 556,
       "by_category": {
         "raw": [...],
         "missing_flags": [...],
         "lag": [...],
         "lag_long": [...],
         "rolling_mean": [...],
         "rolling_mean_long": [...],
         "rolling_std": [...],
         "growth": [...],
         "share": [...],
         "log1p": [...],
         "search": [...],
         "poi_pca": ["poi_pca_1","poi_pca_2"],
         "time": [...]
       }
     }
     ```
   - 若 `total_features` 不在 537–567，直接 `raise ValueError(f"feature count {n_features} outside expected range")`。

---

## 5. 資料切分與資料集組裝 (`src/splits_v3.py`)

### 5.1 定義
```python
FOLD_DEFINITIONS_V3 = [
    ("2019-01-01", "2023-11-30", "2023-12-01", "2024-01-31"),
    ("2019-01-01", "2024-01-31", "2024-02-01", "2024-03-31"),
    ("2019-01-01", "2024-03-31", "2024-04-01", "2024-05-31"),
    ("2019-01-01", "2024-04-30", "2024-05-01", "2024-06-30")
]
HOLDOUT_WINDOW = ("2019-01-01", "2024-06-30", "2024-07-01", "2024-07-31")
```
- 交叉驗證僅涵蓋 2023-12 ~ 2024-06，以對齊測試月前的季節性。
- 留存月專門評估 2024-07。

### 5.2 `generate_time_series_folds_v3(features_df)`
1. 依 `month`、`sector_id` 排序後，建立索引。
2. 針對每個 fold：
   - `train_mask = (month >= train_start) & (month <= train_end)`
   - `valid_mask = (month >= valid_start) & (month <= valid_end)`
   - `forecast_start = pd.Timestamp(valid_start)`
   - 呼叫 `apply_projection` 產生 `projected_df`，並僅保留 `forecast_start` 前的列做為訓練、`forecast_start` 起的列做為驗證。
3. 返回 `list[tuple[np.ndarray, np.ndarray]]`，並寫入 `reports_v3/folds_definition_v3.json`（含 `forecast_start`）。

---

## 6. AutoML 與模型訓練 (`src/automl_runner_v3.py`)

### 6.1 設定檔 `config/automl_v3.yaml`
```yaml
seed: 42
time_budget_per_fold: 900
min_trials_per_fold: 25
metric: competition_score
task: regression
estimator_list: ["xgboost"]
n_jobs: -1
gpu_per_trial: 1
fit_kwargs:
  tree_method: gpu_hist
  predictor: gpu_predictor
  gpu_id: 0
search_space:
  n_estimators:
    _type: randint
    _value: [400, 1600]
  max_depth:
    _type: randint
    _value: [3, 8]
  max_leaves:
    _type: randint
    _value: [0, 256]
  learning_rate:
    _type: loguniform
    _value: [0.02, 0.12]
  min_child_weight:
    _type: loguniform
    _value: [0.3, 8]
  subsample:
    _type: uniform
    _value: [0.6, 0.95]
  colsample_bytree:
    _type: uniform
    _value: [0.5, 0.9]
  reg_alpha:
    _type: loguniform
    _value: [0.0001, 5]
  reg_lambda:
    _type: loguniform
    _value: [0.5, 10]
  gamma:
    _type: uniform
    _value: [0, 5]
```

### 6.2 流程
1. **資料準備**
   - 僅使用 `is_future == 0` 且 `month <= train_end` 的列作訓練資料。
   - 特徵欄位：排除 `['month', 'id']`。保留 `sector_id` 以捕捉結構性差異。
   - `float_cols = df.select_dtypes(include=['float32','float64']).columns`。
2. **標準化**
   - 每個 fold 訓練前 fit `StandardScaler` (with_mean=True, with_std=True) on `float_cols` 的訓練切片。
   - 儲存 `models_v3/fold_<k>_scaler.pkl` 以及均值/標準差 JSON (`reports_v3/fold_<k>_scaler_stats.json`)。
3. **FLAML 設定**
   - 使用 `automl = AutoML()`，`automl.add_metric("competition_score", flaml_metric_callback)`，callback 需回傳 `(1-score)` 為 loss。
   - `automl.fit(...)` 使用 `time_budget=config['time_budget_per_fold']`。若 `automl.best_iteration < config['min_trials_per_fold']` 則再呼叫一次 `automl.fit` 補到 1,200 秒。
4. **再訓練**
   - 取得 `best_config = automl.best_config`，加入 `fit_kwargs`，再以 `xgboost.XGBRegressor(**params, objective='reg:squarederror', random_state=42)` 訓練。
   - 以 `eval_set=[(X_valid, y_valid_log)]`, `early_stopping_rounds=100`, `verbose=False`。
   - 儲存模型為 `models_v3/fold_<k>_model.json`（使用 `booster.save_model`）。
5. **預測與評估**
   - 轉回原尺度：`np.clip(np.expm1(y_pred_log), 0, None)`。
   - 指標：`competition_score`, `SMAPE`, `MAPE`, `MAE`, `RMSE`, `R2`。寫入 `reports_v3/fold_metrics_v3.csv`（每 fold 一列）。
   - 預測結果保存 `reports_v3/predictions_fold_<k>.parquet`。
6. **AutoML 審計**
   - 將每次試驗資訊（config、score、wall_time）追加到 `logs_v3/automl_trials.csv`。
   - `reports_v3/automl_summary.json`：包含 `best_config`, `fold_scores`, `avg_score`, `feature_count`, `target_transform: "log1p_expm1"`。

### 6.3 留存月與全量訓練
1. 選取 `avg competition_score` 最高的折數設定作為 `final_config`。
2. **留存月模型**
   - 特徵來源：`forecast_start = 2024-07-01`。
   - 訓練資料：`month <= 2024-06-30`。
   - 保存模型為 `models_v3/best_model_holdout.json`、標準化器 `models_v3/scaler_holdout.pkl`。
   - 評估結果寫入 `reports_v3/holdout_metrics_v3.json`，並附上 `projection_drift_202407.json`。
   - 若 `competition_score < 0.70` 或 `MAPE > 0.45`，流程即時停止並 raise。
3. **全量模型**
   - 特徵來源：`forecast_start = 2024-08-01`。
  - 訓練資料：`month <= 2024-07-31`（標籤僅到 2024-07）。
  - 保存 `models_v3/best_model_full.json`、`models_v3/scaler_full.pkl`。
  - 產出 `reports_v3/feature_importance_gain.csv`（`booster.get_score(importance_type='gain')`）與 `reports_v3/top_features.txt`（前 50 名）。

---

## 7. 殘差修正 (`scripts/residual_correction_v3.py`)
1. 載入所有 fold 與留存月預測，計算 `residual = target - prediction`，僅使用 `target_available_flag == 1`。
2. 特徵集：`prediction`, `month_index`, `sector_id`, `resident_population`, `population_scale`, `amount_new_house_transactions`, `num_new_house_transactions`, `amount_pre_owned_house_transactions`, `num_pre_owned_house_transactions`, `transaction_amount`, `transaction_amount_nearby_sectors`、12 個保留的 `search_kw_<slug>`。
3. 使用時間序 KFold（5 折，按 `month` 排序），每折評估 `MAPE`、`RMSE`。若平均 `MAPE >= 0.40` 則不寫模型、直接輸出 `reports_v3/residual_training_metrics.json` 並標記 `status = "rejected"`。
4. 通過門檻時：
   - `StandardScaler` → `sklearn.linear_model.Ridge(alpha=0.5, random_state=42)`。
   - 儲存 `models_v3/residual_scaler.pkl`, `models_v3/residual_model.pkl`, `reports_v3/residual_features.json`（排序後的特徵清單）與 `residual_training_metrics.json`（含每折分數）。

---

## 8. 預測與提交 (`src/predict_v3.py`)
1. 入口 `generate_submission(panel_path, features_path, model_path, scaler_path, sample_submission, output_path, residual_paths=None)`.
2. 預測流程：
   - 起始 `working_panel = panel_v3`。
   - 對 `forecast_month` 依序為 `2024-08` → `2024-12`：
     1. 呼叫 `apply_projection(working_panel, month)` 取得當前月的投影特徵。
     2. `build_feature_matrix_v3(..., forecast_start=month)` 取得對應特徵列。
     3. 取出 `month == forecast_month` 的資料，套用 scaler、模型，輸出預測。
     4. 若提供殘差模型，套用後再 `np.clip(…, 0, None)`。
     5. 將預測寫回 `working_panel.loc[mask, 'target_filled']`，供下一月 lag 使用。
   - 累積所有 `id`、`prediction`，與 `sample_submission` merge，確保順序一致。
3. 寫出 `submission_v3/submission.csv`、`reports_v3/test_window_summary.json`（月別平均/標準差/最值/投影來源比例）。
4. 輸出診斷圖：
   - `reports_v3/plots/residual_vs_target_2024-07.png`
   - `reports_v3/plots/prediction_distribution_train.png`
5. 若任何月份投影來源比例 `proj_source in {1,2,3}` 超過 85%，需於 `test_window_summary.json` 標註 `high_projection_dependence = true`。

---

## 9. 測試與驗證

### 9.1 靜態檢查
- `python -m py_compile src/**/*.py scripts/**/*.py`

### 9.2 單元測試
1. `pytest tests/test_projection_v3.py::test_projection_respects_cutoff`
2. `pytest tests/test_features_v3.py::test_feature_counts_and_categories`
3. `pytest tests/test_splits_v3.py::test_fold_alignment`
4. `pytest tests/test_metrics_v3.py::test_competition_metric_cap`

### 9.3 煙霧流程（需依序執行）
```bash
python scripts/run_pipeline_v3.py build_panel --data-config config/data_paths_v3.yaml --panel-path data/processed/panel_v3.parquet --reports-dir reports_v3
python scripts/run_pipeline_v3.py build_features --panel-path data/processed/panel_v3.parquet --features-path data/processed/features_v3.parquet --reports-dir reports_v3 --forecast-start 2024-06-01
python scripts/run_pipeline_v3.py run_xgb --features-path data/processed/features_v3.parquet --config config/automl_v3.yaml --reports-dir reports_v3
python scripts/run_pipeline_v3.py evaluate_holdout --panel-path data/processed/panel_v3.parquet --features-path data/processed/features_v3.parquet --model-path models_v3/best_model_holdout.json --scaler-path models_v3/scaler_holdout.pkl --reports-dir reports_v3
python scripts/run_pipeline_v3.py predict --panel-path data/processed/panel_v3.parquet --features-path data/processed/features_v3.parquet --model-path models_v3/best_model_full.json --scaler-path models_v3/scaler_full.pkl --sample-submission ../sample_submission.csv --output-path submission_v3/submission.csv --reports-dir reports_v3 [--residual-model models_v3/residual_model.pkl --residual-scaler models_v3/residual_scaler.pkl --residual-features reports_v3/residual_features.json]
```

---

## 10. 產出清單
- `data/processed/panel_v3.parquet`
- `data/processed/features_v3.parquet`（依年份分割）
- `config/data_paths_v3.yaml`, `config/automl_v3.yaml`
- `models_v3/` 內所有模型與 scaler
- `reports_v3/`:  
  `panel_build_summary.json`, `missing_value_report.json`, `projection_drift_*.json`, `feature_inventory_v3.json`, `search_keywords_v3.json`, `poi_pca_summary.json`, `fold_metrics_v3.csv`, `holdout_metrics_v3.json`, `test_window_summary.json`, `feature_importance_gain.csv`, `top_features.txt`, `residual_training_metrics.json`, `residual_features.json`, `plots/` 目錄
- `logs_v3/automl.log`, `logs_v3/automl_trials.csv`
- `submission_v3/submission.csv`

---

### Testing
⚠️ 未執行實際程式與測試（本次工作僅為文件審查與規格撰寫）。
