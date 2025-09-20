# Real Estate Demand Prediction – AutoML Specification v3.1

## 1. 核心目標
1. **校正訓練/推論落差**：所有訓練、交叉驗證、留存月與最終推論資料皆使用相同的前瞻特徵投影流程，避免再出現 v2 中「驗證接近 1.0、實測僅 0.47」的落差。

2. **控制特徵維度**：最終輸出特徵總數須落在 450–650 的彈性區間；若低於 450 或高於 650 需立即拋出錯誤。於 `feature_inventory_v3.json` 中寫入 `total_features`, `min_features`, `max_features` 與各分類差異，並在自動化報告內標記當前計數相對 v3.0 的變化量。
3. **資料安全界線**：任何模型、殘差修正或投影程序不得使用 2024-07 之後的真實標籤；投影僅可基於 `month <= 2024-07` 的觀測。
4. **完整審計鏈**：對每一步驟輸出指標、特徵清單、投影誤差及模型設定，確保日後能追溯「投影 → 特徵 → 訓練 → 推論」的每一環。
5. **指標與驗證一致性**：全流程僅允許使用 `metrics/competition_score.py` 實作的官方評分（含 APE>1 置零規則）。CV、留存月、偽測試與最終報告皆須呼叫該模組，以避免自訂實作造成評分偏差。

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
| `config/sector_city_map_v3.csv` | `sector` | `city_id` | 手動維護部門隸屬城市 |
| `../train/city_search_index.csv` | `month`,`keyword`,`source` | `search_volume` | 僅用到 2024-07 |
| `../train/city_indexes.csv` | `city_indicator_data_year` | 48 個年度指標 | 僅保留精選 12 欄 |
| `../test.csv` | `id` | 2024-08 ~ 2024-12 | |
| `../sample_submission.csv` | `id` | 提交格式 | |

所有 CSV 以 `encoding='utf-8'` 讀取，欄位命名轉成 snake_case，文字欄位做 NFKC 正規化並去除前後空白。`sector_city_map_v3.csv` 為
手動維護檔，固定兩欄：`sector`（與原始表一致）與 `city_id`（整數 1~N），僅含歷史出現過的 sector；若遇到測試階段未見過的
sector，需於管線初始化時即 raise。

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
   - 另讀取 `config/sector_city_map_v3.csv`，確保所有 `sector` 均存在對應 `city_id`；若缺漏立即 raise。
   - 回傳字典鍵值固定：`new_house_transactions`, `new_house_transactions_nearby_sectors`, `pre_owned_house_transactions`, `pre_owned_house_transactions_nearby_sectors`, `land_transactions`, `land_transactions_nearby_sectors`, `sector_poi`, `city_search_index`, `city_indexes`, `sector_city_map`, `test`, `sample_submission`。
3. **`build_calendar_v3(raw_tables)`**
   - 取所有出現過的 `sector`，使用 `pd.MultiIndex.from_product([MONTH_RANGE, sectors])`。
   - `sector_id = int(re.search(r'(\d+)', sector).group(1))`；若解析失敗直接 `ValueError`。
   - 透過 `sector_city_map` 取得 `city_id`（型別 `np.int16`）；將 `city_id` 納入索引欄位方便後續聚合。
4. **`merge_sources_v3(calendar_df, raw_tables)`**
   - 僅合併實際觀測值（不再在此階段外插未來月份）；即把 v2 的 `_extend_monthly_tables` 功能移除。
   - 合併順序：六個月度表 → `sector_poi` → `city_indexes`（年資料轉月、forward fill）→ `city_search_index`（維持原始觀測，不做變異數篩選）。
   - 合併後保留 `sector` 文字欄位供稽核，於最終輸出前刪除；新增 `city_id` 整數欄位。
5. **`attach_target_v3(panel_df)`**
   - `target_raw = amount_new_house_transactions`。
   - `target_available_flag = (month <= TARGET_CUTOFF).astype(np.int8)`。
   - `target = np.where(target_available_flag == 1, target_raw, np.nan)`.
   - `target_filled = target_raw.fillna(0.0)`（僅供歷史 lag；未來月份維持 0 以便預測後更新）。
   - `target_filled_was_missing = target_raw.isna().astype(np.int8)`。
   - `is_future = (month > TARGET_CUTOFF).astype(np.int8)`。
   - 生成 `id = month.strftime('%Y %b') + '_sector ' + sector_id.astype(str)`，`city_id` 與 `sector_id` 於資料框中常駐供後續使用。
6. **`save_panel_v3(panel_df, panel_path)`**
   - 加入 `year = month.dt.year`，輸出至 `data/processed/panel_v3.parquet`（依年份分割）。
   - 另寫 `reports_v3/panel_build_summary.json`：包含 `month_min`, `month_max`, `sector_count`, `labeled_rows`, `future_rows`, `missing_target_ratio`。
7. **清理**
   - 在返回特徵階段前即刪除 `sector` 字串欄位，但保留 `sector_id` 與 `city_id`。

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
5. **同城月度共享**（資料量 < 3）：使用同月份（`month.dt.month`）且 `month < forecast_start`、`city_id` 相同的中位數。若該城市在該月
歷史樣本亦不足 3 筆，直接跳到第 6 步。
6. **全域備援**：使用整體歷史中位數；若仍為 NaN 則填 0。
7. 任何負值經 `np.clip(value, 0, None)` 截為 0。
8. 寫入 `*_proj_source` 欄位：`0` 表示觀測值、`1` 表示第 2 步（EWMA）、`2` 表示平均/中位數、`3` 表示同城月度共享、`4` 表示全域備援
（含保底 0）。
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
- `sector_id`、`city_id`（整數）。
- 城市指標：僅保留以下 12 欄（及對應 `was_interpolated` 旗標）：  
  `city_gdp_100m`, `city_secondary_industry_100m`, `city_tertiary_industry_100m`, `city_gdp_per_capita_yuan`, `city_total_households_10k`, `city_year_end_resident_population_10k`, `city_total_retail_sales_of_consumer_goods_100m`, `city_per_capita_disposable_income_absolute_yuan`, `city_annual_average_wage_urban_non_private_employees_yuan`, `city_number_of_universities`, `city_hospital_beds_10k`, `city_number_of_operating_bus_lines`。
- Search 原始欄位：投影後再選出 12 個 slug（第 4.6 節）。

### 4.3 缺值旗標與補值
1. 對第 4.2 中所有月度欄位建立 `*_was_missing` (`np.int8`)。
2. 補值策略（先依 `sector_id` 分組再排序），僅在 `history_mask = month < forecast_start` 範圍內進行；未來區段保持 NaN 供模型推論時再
    個別計算：
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
   - 以 `history_mask = month < forecast_start`、`groupby(['sector_id'])` 後排序，對 `target_filled` 建立 `target_lag_1`, `target_lag_3`, `target_lag_6`, `target_lag_12`（均使用 `shift(1)`，缺值以 0 填補），並標記 `target_lag_*_was_imputed`。
   - `lag_1`, `lag_3`, `lag_6` for `METRIC_SET_FULL`；`lag_12` for `METRIC_SET_LONG`，同樣使用 `shift(1)` 並僅在 `history_mask` 內計算。
   - `rolling_mean_3`, `rolling_mean_6` for `METRIC_SET_FULL`；`rolling_mean_12` for `METRIC_SET_LONG`。所有 rolling 統一 `shift(1)` 後再計算，`min_periods=2`。
   - `rolling_std_3`, `rolling_std_6` for `METRIC_SET_FULL`，`ddof=0`。
   - 任何 lag/rolling 在 `history_mask` 內若出現缺值，僅於該遮罩內 `ffill()`，不得把驗證/未來區段的值向前傳遞；對 `history_mask` 外的列維持 NaN，待模型推論時計算對應月度再填。
2. **Growth**
   - `growth_1`、`growth_3` 對 `GROWTH_METRICS`：`(value - lag) / (abs(lag) + 1e-6)`；lag 缺值時填 0，並加 `*_growth_*_was_missing`.
3. **Share / Weighted Mean**
   - 使用 `population_weight = np.where(resident_population>0, resident_population, np.where(population_scale>0, population_scale, 1.0))`。
   - `metric_city_weighted_mean` = 加權平均（權重和為 0 則使用全體平均）。
   - `metric_share = value / (metric_city_weighted_mean + 1e-6)`。
4. **Log1p**
   - 對 `METRIC_SET_FULL` 及其 lag/rolling/growth/weighted_mean/share 統一建立 `_log1p` 版本（避免重覆計算；使用 `np.log1p(np.clip(value, 0, None))`）。
5. **Search 關鍵字**
   - 在 `apply_projection` 後的資料上，先以 `history_mask` 過濾訓練列，再依 `month` 做 3 折時間序交叉擬合：每折以前序折數 fit `mutual_info_regression`（`random_state=42`, `discrete_features=False`），評估該折保留集的 MI。若 keyword 在某折完全為常數，該折貢獻 0。
   - 將三折 MI 取平均後排序，選取前 12 個 slug，寫入 `reports_v3/search_keywords_v3.json`（包含每折 MI、平均 MI、使用的 `history_mask` 範圍與版本號）。
   - 對每個保留 slug：保留原始值與 `lag_1`, `rolling_mean_3`, `pct_change_1`, `zscore_3m`（以 `rolling(window=3, min_periods=1).mean()` 與標準差計算 z-score）；並保留 `_was_missing`。
6. **POI PCA**
   - 將 `sector_POI` 中除核心欄位外的 `_dense`/`number_of_` 欄位以 0 補值後取 `np.log1p`。
   - 僅以 `history_mask` 內的列 fit `sklearn.decomposition.PCA(n_components=2, random_state=42)`，再對全體 transform。若第二成分解釋率不足 5% 仍保留以維持固定欄位數。輸出 `poi_pca_1`, `poi_pca_2`，並寫 `reports_v3/poi_pca_summary.json`（包含原始欄位數、解釋率、使用的遮罩區間與 `random_state`）。
7. **欄位清單與配額**
   - 彙整所有欄位分類，格式：
     ```json
     {
       "total_features": 556,
       "min_features": 450,
       "max_features": 650,
       "delta_vs_v30": -8,
       "by_category": {
         "raw": [...],
         "missing_flags": [...],
         "target_lag": ["target_lag_1", "target_lag_3", ...],
         "target_lag_flags": ["target_lag_1_was_imputed", ...],
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
   - 若 `total_features` 不在 `[min_features, max_features]`，直接 `raise ValueError(f"feature count {n_features} outside expected range")`，同時將錯誤寫入 `reports_v3/feature_inventory_v3.json` 中的 `status="failed"`。
8. **投影漂移防護**
   - 於 `collect_projection_drift` 產出的報告中，對每個欄位抓取最近可用月份（優先 2024-07）的 `mape`, `mae`, `proj_source_3_ratio`, `proj_source_4_ratio`。
   - 若 `mape` 高於歷史 P80（以 2019-01~2024-06 的 `mape` 分布計算）或 `proj_source_3_ratio + proj_source_4_ratio > 0.85`，則對該欄位停用高階派生：移除其 `rolling_*`, `growth_*`, `share`, `log1p` 以及所有以其為基底的組合特徵。基礎原始欄位與 `lag_*` 可保留。
   - 移除紀錄寫入 `reports_v3/feature_inventory_v3.json` 的 `projection_suppressed_features`，格式包含欄位名稱、觸發指標與處置結果。

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
PURGE_PERIOD = pd.DateOffset(months=1)
```
- 交叉驗證僅涵蓋 2023-12 ~ 2024-06，以對齊測試月前的季節性。
- 留存月專門評估 2024-07。

### 5.2 `generate_time_series_folds_v3(features_df)`
1. 依 `month`、`sector_id` 排序後，建立索引。
2. 針對每個 fold：
   - `train_mask = (month >= train_start) & (month <= train_end)`。
   - `valid_mask = (month >= valid_start) & (month <= valid_end)`。
   - `purge_start = pd.Timestamp(valid_start) - PURGE_PERIOD`，`purge_mask = (month >= purge_start) & (month < valid_start)`；最終訓練列為 `train_mask & ~purge_mask`。
   - `forecast_start = pd.Timestamp(valid_start)`。
   - 呼叫 `apply_projection` 產生 `projected_df`，並僅保留 `forecast_start` 前的列做為訓練、`forecast_start` 起的列做為驗證。
3. 返回 `list[tuple[np.ndarray, np.ndarray]]`，並寫入 `reports_v3/folds_definition_v3.json`（含 `forecast_start`、`purge_period_months=1` 及每折實際留空筆數）。

---

## 6. AutoML 與模型訓練 (`src/automl_runner_v3.py`)

### 6.1 設定檔 `config/automl_v3.yaml`
```yaml
seed: 42
time_budget_per_fold: 900
min_trials_per_fold: 25
metric: competition_score
task: regression
estimator_list: ["xgboost", "lgbm"]
n_jobs: -1
gpu_per_trial: 1
fit_kwargs_by_estimator:
  xgboost:
    tree_method: gpu_hist
    predictor: gpu_predictor
    gpu_id: 0
    objective: reg:tweedie
    tweedie_variance_power: 1.3
  lgbm:
    device_type: cpu
    boosting_type: gbdt
    objective: tweedie
    tweedie_variance_power: 1.3
zero_inflation:
  enabled: true
  method: tweedie
search_space:
  xgboost:
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
  lgbm:
    num_leaves:
      _type: randint
      _value: [31, 255]
    max_depth:
      _type: randint
      _value: [-1, 12]
    learning_rate:
      _type: loguniform
      _value: [0.01, 0.12]
    min_child_samples:
      _type: randint
      _value: [20, 120]
    subsample:
      _type: uniform
      _value: [0.6, 1.0]
    colsample_bytree:
      _type: uniform
      _value: [0.5, 1.0]
    reg_alpha:
      _type: loguniform
      _value: [0.0001, 5]
    reg_lambda:
      _type: loguniform
      _value: [0.5, 10]
```

### 6.2 流程

1. **資料準備與樣本權重**
   - 僅使用 `is_future == 0` 且 `month <= train_end` 的列作訓練資料，目標維持原始尺度（不再 `log1p`）。
   - 特徵欄位：排除 `['month', 'id']`。保留 `sector_id` 與 `city_id` 以捕捉層級差異。
   - 建立 `sample_weight = np.where(month.between('2024-04-01', '2024-06-30'), 1.8, 1.0)`；於 `reports_v3/fold_metrics_v3.csv` 附加 `avg_sample_weight`、`weighted_rows` 等統計。
2. **特徵處理**
   - 樹模型不進行標準化或均值中心化；維持補值後的原尺度。
   - 建立單調性約束映射：`monotone_positive = ['resident_population', 'population_scale', 'surrounding_housing_average_price', 'surrounding_shop_average_rent', 'city_gdp_100m', 'city_total_retail_sales_of_consumer_goods_100m']`。依照訓練設計矩陣的欄序生成與特徵數等長的向量，對上述欄位填 `+1`，其餘填 `0`。XGBoost 使用 `monotone_constraints="(0,0,...,1,...)"` 字串，LGBM 使用 `monotone_constraints` list。
3. **FLAML 設定與搜尋**
   - 使用 `automl = AutoML()`，透過 `automl.add_metric("competition_score", metrics.make_flaml_metric())` 將 `metrics/competition_score.py` 的回傳函式封裝成最小化損失（`1 - competition_score`）。
   - 對 `estimator_list` 中每個模型分別呼叫 `automl.fit(estimator_list=[estimator], ...)`，傳入 `sample_weight`, `time_budget=config['time_budget_per_fold']` 與相對應的 `monotone_constraints`/`fit_kwargs_by_estimator`。若 `automl.best_iteration < config['min_trials_per_fold']` 則額外延長至累計 1,200 秒。
4. **再訓練**
   - 取得每個估計器的 `best_config`，合併 `fit_kwargs_by_estimator` 與單調性設定後重新建模：
     - XGBoost：`xgboost.XGBRegressor(**params, random_state=42, monotone_constraints=..., objective='reg:tweedie', tweedie_variance_power=1.3)`。
     - LightGBM：`lightgbm.LGBMRegressor(**params, random_state=42, monotone_constraints=..., objective='tweedie', tweedie_variance_power=1.3)`。
   - `fit` 時傳入 `sample_weight`、`eval_set=[(X_valid, y_valid)]`、`eval_sample_weight`、`early_stopping_rounds=100`、`verbose=False`。
   - 模型另存：`models_v3/fold_<k>_xgb.json`、`models_v3/fold_<k>_lgbm.txt`；同時輸出特徵重要度至 `reports_v3/feature_importance_fold_<k>_<model>.csv`。
5. **預測與評估**
   - 兩模型皆輸出非負預測：`np.clip(pred, 0, None)`。
   - 計算 XGBoost、LGBM 以及簡單平均 `ensemble_pred = 0.5 * pred_xgb + 0.5 * pred_lgbm` 的 `competition_score`, `SMAPE`, `MAPE`, `MAE`, `RMSE`, `R2`。`fold_metrics_v3.csv` 需包含三種模型的成績並標註 `model_type`。
   - 將各模型與集成的驗證預測寫入 `reports_v3/predictions_fold_<k>_<model>.parquet`（含 `id`, `month`, `prediction`, `sample_weight`）。
6. **AutoML 審計與摘要**
   - 將每次試驗資訊（config、score、wall_time、estimator）追加到 `logs_v3/automl_trials.csv`。
   - `reports_v3/automl_summary.json`：分別記錄 XGBoost 與 LGBM 的 `best_config`, `best_iteration`, `fold_scores`, `avg_score`, `sample_weight_strategy`, `monotone_constraints`，以及 `ensemble_score`。

### 6.3 留存月與全量訓練
1. 依折均 `competition_score` 選出最佳 XGBoost 與 LGBM 組合，組成 `final_config = {'xgb': ..., 'lgbm': ...}`。
2. **留存月模型**
   - 特徵來源：`forecast_start = 2024-07-01`。
   - 訓練資料：`month <= 2024-06-30`，沿用相同 `sample_weight` 與單調性設定。
   - 分別保存 `models_v3/best_model_holdout_xgb.json`、`models_v3/best_model_holdout_lgbm.txt` 與 `reports_v3/holdout_feature_importance_<model>.csv`。
   - 評估結果寫入 `reports_v3/holdout_metrics_v3.json`，所有指標均使用 `metrics/competition_score.py`，需同時列出兩模型與 0.5/0.5 集成的分數，並附上 `projection_drift_202407.json`。
   - 若 `competition_score < 0.70` 或 `MAPE > 0.45`，記錄於報告中 `status="warning"` 且發送告警，但流程不終止。
3. **全量模型**
   - 特徵來源：`forecast_start = 2024-08-01`。
   - 訓練資料：`month <= 2024-07-31`（標籤僅到 2024-07）。
   - 保存 `models_v3/best_model_full_xgb.json`、`models_v3/best_model_full_lgbm.txt` 與 `reports_v3/feature_importance_full_<model>.csv`；再生成 `reports_v3/top_features.txt`（集成重要度前 50 名）。

### 6.4 官方評分模組 (`metrics/competition_score.py`)
1. 實作 `def competition_score(y_true: np.ndarray, y_pred: np.ndarray) -> float`：
   - 先將 `y_true`, `y_pred` 轉為 `np.float64` 並裁剪 `y_pred = np.clip(y_pred, 0, None)`。
   - 計算 `ape = np.abs(y_true - y_pred) / np.maximum(y_true, 1.0)`，其中 `y_true == 0` 時分母固定為 1。
   - 將 `ape > 1` 的樣本視為 0 貢獻（官方規則），其餘取 `1 - ape`，最後回傳平均值。
2. 提供 `def make_flaml_metric() -> Callable`，回傳可供 FLAML 登記的函式：
   ```python
   def make_flaml_metric():
       def flaml_metric(y_pred, dtrain):
           y_true = dtrain.get_label()
           score = competition_score(y_true, y_pred)
           return 1 - score
       return flaml_metric
   ```
3. 若需評估 DataFrame，提供輔助函式 `competition_score_df(df: pd.DataFrame, pred_col: str, target_col: str) -> float`，方便於報告階段重用。
4. 所有訓練、驗證、留存月、偽測試與最終提交統一呼叫本模組，不得自製指標版本。

### 6.5 零膨脹管線（可選）
1. 於 `config/automl_v3.yaml` 增列 `zero_inflation: {enabled: true, method: "hurdle"}`；支援 `"hurdle"` 與 `"tweedie"` 兩種策略，預設 `tweedie`。
2. **Hurdle 流程**（`method == "hurdle"` 時啟用）：
   - Stage-A：以 `target_binary = (target > 0).astype(int)` 訓練分類模型（可使用 `lightgbm.LGBMClassifier`，超參數固定，採與 Stage-B 相同的樣本權重）。
     - 模型另存為 `models_v3/hurdle_classifier_<phase>.txt`，並於 `reports_v3/zero_inflation_summary.json` 記錄 AUC、PR-AUC。
   - Stage-B：沿用前述 Tweedie 回歸模型，僅於 `target > 0` 的列上訓練，預測時輸出期望值 `E[target | target>0]`。
     - XGBoost 與 LGBM 各自訓練一個 Stage-B 模型，儲存為 `hurdle_regressor_<phase>_<model>.txt`；評估 MAPE 與 RMSE。
   - 最終預測：`pred = prob_positive * regression_pred`，於每個 fold 與 holdout/全量階段均輸出。
3. **Tweedie 流程**（`method == "tweedie"` 或 Stage-B 回歸）：
   - 直接使用 6.2 的 Tweedie 參數，另報告 `tweedie_variance_power` 與 `objective`。
4. 於 `reports_v3/zero_inflation_summary.json` 記錄是否啟用、分類模型評分（AUC、PR-AUC）、Stage-B MAPE 以及最終集成得分。
5. 預測端需能同時支援 `method='tweedie'`（單段回歸）與 `method='hurdle'`（兩段組合），並在 `test_window_summary.json` 中標註使用的策略。

### 6.6 量化預測
1. 以 LightGBM 另外訓練 P50、P80 兩個分位模型：
   - 使用與主模型相同的特徵集與 `sample_weight`，但設定 `objective='quantile'`、`alpha ∈ {0.5, 0.8}`，並共用 `monotone_constraints`。
   - 模型儲存為 `models_v3/quantile_full_alpha{50,80}.txt`，驗證預測寫入 `reports_v3/predictions_fold_<k>_lgbm_q{50,80}.parquet`。
2. 於報告新增 `reports_v3/uncertainty_summary.json`：包含分位預測與實際的覆蓋率（P80 覆蓋率應 ≥ 75%）。
3. 預測時同步輸出分位數，以供後續風險護欄使用。

---

## 7. 殘差修正 (`scripts/residual_correction_v3.py`)
1. 載入所有 fold 與留存月預測，計算 `residual = target - prediction`，僅使用 `target_available_flag == 1`。
2. 特徵集：`prediction`, `month_index`, `sector_id`, `city_id`, `resident_population`, `population_scale`, `amount_new_house_transactions`, `num_new_house_transactions`, `amount_pre_owned_house_transactions`, `num_pre_owned_house_transactions`, `transaction_amount`, `transaction_amount_nearby_sectors`、12 個保留的 `search_kw_<slug>`。
3. 使用時間序 KFold（5 折，按 `month` 排序），每折評估 `MAPE`、`RMSE`。若平均 `MAPE >= 0.40` 則不寫模型、直接輸出 `reports_v3/residual_training_metrics.json` 並標記 `status = "rejected"`。
4. 通過門檻時：
   - `StandardScaler` → `sklearn.linear_model.Ridge(alpha=0.5, random_state=42)`。
   - 儲存 `models_v3/residual_scaler.pkl`, `models_v3/residual_model.pkl`, `reports_v3/residual_features.json`（排序後的特徵清單）與 `residual_training_metrics.json`（含每折分數）。

---

## 8. 預測與提交 (`src/predict_v3.py`)
1. 入口 `generate_submission(panel_path, features_path, model_paths, sample_submission, output_path, reports_dir, *, zero_inflation=None, residual_bundle=None, quantile_paths=None)`。
   - `model_paths` 為 `{"xgb": Path, "lgbm": Path}`，分別指向全量訓練後的模型；若啟用 hurdle，`zero_inflation` 需再提供 `classifier_path` 與 `regressor_path`。
   - `residual_bundle`（可選）包含 `model`, `scaler`, `features` 三檔案路徑；`quantile_paths` 為 `{0.5: Path, 0.8: Path}`。
2. 預測流程：
   - 起始 `working_panel = panel_v3`，並初始化 `results = []`。
   - 對 `forecast_month` 依序為 `2024-08` → `2024-12`：
     1. 呼叫 `apply_projection(working_panel, month)` 取得當前月（含歷史）投影特徵。
     2. `features_df = build_feature_matrix_v3(..., forecast_start=month)`，僅取 `month == forecast_month` 的列。
     3. 取得 `X_features`（去除 `target` 等欄位）後：
        - Tweedie 途徑：分別以 XGBoost 與 LGBM 模型推論，得 `pred_xgb`, `pred_lgbm`。
        - Hurdle 途徑：先以分類器得到 `prob_pos`，再以回歸器得到 `reg_pred`，輸出 `pred_model = prob_pos * reg_pred`，並對兩模型各自執行。
     4. 形成 `pred_ensemble = 0.5 * pred_xgb + 0.5 * pred_lgbm`，若提供殘差模型則依 `residual_bundle` 內的 scaler、特徵清單做校正後再 `np.clip(pred, 0, None)`。
     5. 若提供分位模型，產出 `pred_p50`, `pred_p80` 並與主預測一同保存；否則以 `np.nan` 佔位以維持欄位一致。
     6. 以 `mask = features_df['month'] == forecast_month` 鎖定當月列，將 `pred_ensemble` 寫回 `working_panel.loc[mask, 'target_filled']` 供下一月目標 lag 使用；同時更新 `results.append({"id": ..., "prediction": pred_ensemble, "prediction_xgb": pred_xgb, "prediction_lgbm": pred_lgbm, "prediction_p50": pred_p50, "prediction_p80": pred_p80})`。
   - 迴圈後，將 `results` 轉為 DataFrame 與 `sample_submission` merge，確保 `id` 順序一致。
3. 寫出 `submission_v3/submission.csv`、`reports_v3/test_window_summary.json`：除原本月別統計外，新增 `model_breakdown`（各月 XGB/LGBM/ensemble 的平均與標準差）、`zero_inflation_method`、`quantile_coverage`（以歷史 P50/P80 與推論值估計覆蓋率）以及 `projection_source_ratio`（含 `proj_source==4`）。
4. 輸出診斷圖：
   - `reports_v3/plots/residual_vs_target_2024-07.png`
   - `reports_v3/plots/prediction_distribution_train.png`
   - `reports_v3/plots/prediction_distribution_test.png`（集成 vs. 分位數箱型圖）。
5. 若任何月份投影來源比例 `proj_source in {1,2,3,4}` 超過 85%，需於 `test_window_summary.json` 標註 `high_projection_dependence = true`，並列出受影響欄位。

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
python scripts/run_pipeline_v3.py run_automl --features-path data/processed/features_v3.parquet --config config/automl_v3.yaml --reports-dir reports_v3
python scripts/run_pipeline_v3.py evaluate_holdout --panel-path data/processed/panel_v3.parquet --features-path data/processed/features_v3.parquet --model-xgb models_v3/best_model_holdout_xgb.json --model-lgbm models_v3/best_model_holdout_lgbm.txt --reports-dir reports_v3 [--hurdle-classifier models_v3/hurdle_classifier_holdout.txt --hurdle-regressor models_v3/hurdle_regressor_holdout.txt]
python scripts/run_pipeline_v3.py predict --panel-path data/processed/panel_v3.parquet --features-path data/processed/features_v3.parquet --model-xgb models_v3/best_model_full_xgb.json --model-lgbm models_v3/best_model_full_lgbm.txt --sample-submission ../sample_submission.csv --output-path submission_v3/submission.csv --reports-dir reports_v3 [--residual-model models_v3/residual_model.pkl --residual-scaler models_v3/residual_scaler.pkl --residual-features reports_v3/residual_features.json --quantile-p50 models_v3/quantile_full_alpha50.txt --quantile-p80 models_v3/quantile_full_alpha80.txt]
```

---

## 10. 產出清單
- `data/processed/panel_v3.parquet`
- `data/processed/features_v3.parquet`（依年份分割）
- `config/data_paths_v3.yaml`, `config/automl_v3.yaml`, `config/sector_city_map_v3.csv`
- `models_v3/`：
  - `best_model_holdout_xgb.json`, `best_model_holdout_lgbm.txt`, `best_model_full_xgb.json`, `best_model_full_lgbm.txt`
  - 若啟用 hurdle：`hurdle_classifier_holdout.txt`, `hurdle_classifier_full.txt`, `hurdle_regressor_holdout_xgb.json`, `hurdle_regressor_holdout_lgbm.txt`, `hurdle_regressor_full_xgb.json`, `hurdle_regressor_full_lgbm.txt`
  - `quantile_full_alpha50.txt`, `quantile_full_alpha80.txt`
  - 殘差相關：`residual_model.pkl`, `residual_scaler.pkl`
- `reports_v3/`：
  `panel_build_summary.json`, `missing_value_report.json`, `projection_drift_*.json`, `feature_inventory_v3.json`, `search_keywords_v3.json`, `poi_pca_summary.json`, `fold_metrics_v3.csv`, `automl_summary.json`, `holdout_metrics_v3.json`, `holdout_feature_importance_{xgb,lgbm}.csv`, `feature_importance_full_{xgb,lgbm}.csv`, `top_features.txt`, `test_window_summary.json`, `zero_inflation_summary.json`, `uncertainty_summary.json`, `residual_training_metrics.json`, `residual_features.json`, `plots/` 目錄
- `logs_v3/automl.log`, `logs_v3/automl_trials.csv`
- `submission_v3/submission.csv`

---

### Testing
⚠️ 未執行實際程式與測試（本次工作僅為文件審查與規格撰寫）。
