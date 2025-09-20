# Real Estate Demand Prediction – AutoML Specification V3 (Feature Reduction Phase)

## 1. 核心目標
- 將可用特徵固定在 540–580 欄之間，避免冗餘並縮短 AutoML 搜尋時間。
- 嚴格限制訓練資料為 2019-01 至 2024-07 的標註月份，確保任何模型或殘差修正都不接觸 2024-08 以後的真實值。
- 建立可重現的特徵治理與選擇流程，輸出正式 allowlist 供後續管線與報表共用。
- 建立涵蓋交叉驗證、獨立留存（月 2024-07）以及 Kaggle 測試月（2024-08～2024-12）的評估與紀錄流程。

## 2. 資料範圍與面板建構要求
1. 沿用 v2 的原始資料來源與欄位定義；所有 CSV 仍以 UTF-8 讀取並保持欄位為 snake_case。
2. 新增 `src/panel_builder_v3.py`，內容可由 v2 版本複製後調整：
   - 月份索引採 `calendar = pd.date_range('2019-01-01', '2024-12-01', freq='MS')`，確保推論期 (2024-08～2024-12) 仍建表。
   - 定義常數 `TARGET_CUTOFF = pd.Timestamp('2024-07-31')`。在 `attach_target` 時僅對 `month <= TARGET_CUTOFF` 產生 `target_filled`；超過此日期的列強制設定 `target_filled = np.nan` 並將 `target_available_flag = 0`。
   - 為每列新增 `is_future = (month > TARGET_CUTOFF).astype(np.int8)`，供後續特徵選擇與評估。
   - 其他缺失值與欄位清理沿用 v2 策略，不得擴充至 2024-08 之後。
3. 面板輸出：
   - `data/processed/panel_v3.parquet`（依年份分割），包含以上新增欄位。
   - `reports_v3/panel_build_summary.json`，記錄月份範圍、sector 數、`target_filled` 的非空筆數與 `is_future` 計數。
4. `config/data_paths_v3.yaml` 需更新為指向 v3 專用輸出目錄 (`panel_v3.parquet`, `features_v3.parquet`, `reports_v3/` 等)。

## 3. 特徵治理策略
### 3.1 原始欄位保留
建構 `features_v3.py` 時必須完整保留下列欄位（含缺失旗標），任何其他原始欄位不得外流至最終矩陣：
- 新建案本地：`amount_new_house_transactions`, `num_new_house_transactions`, `area_new_house_transactions`, `price_new_house_transactions`, `area_per_unit_new_house_transactions`, `total_price_per_unit_new_house_transactions`, `num_new_house_available_for_sale`, `area_new_house_available_for_sale`, `period_new_house_sell_through`。
- 新建案鄰近：上述欄位加 `_nearby_sectors` 後綴。
- 二手房本地：`amount_pre_owned_house_transactions`, `num_pre_owned_house_transactions`, `area_pre_owned_house_transactions`, `price_pre_owned_house_transactions`。
- 二手房鄰近：上述欄位加 `_nearby_sectors` 後綴。
- 土地本地：`transaction_amount`, `num_land_transactions`, `construction_area`, `planned_building_area`。
- 土地鄰近：上述欄位加 `_nearby_sectors` 後綴。
- 城市層指標：所有 `city_` 開頭欄位與 `city_<col>_was_interpolated` 旗標。
- POI 核心欄位：`resident_population`, `resident_population_dense`, `population_scale`, `population_scale_dense`, `office_population`, `office_population_dense`, `surrounding_housing_average_price`, `surrounding_shop_average_rent`, `sector_coverage`。
- 時間欄位：`year`, `month_num`, `quarter`, `month_index`, `days_in_month`, `is_year_start`, `is_year_end`。
- 目標與旗標：`target_filled`, `target_filled_was_missing`, `target_available_flag`, `is_future`, 以及上述所有欄位的 `_was_missing` 版本。
- 僅保留 `sector_id`（int64）與 `month`（datetime64）作為鍵；不要使用文字的 `sector`。

任何原始欄位若在 v2 中產生 `_synthetic` 版本，在 v3 全部移除。

### 3.2 缺失值旗標政策
- 對 3.1 所列的數量、面積、金額、價格欄位建立 `_was_missing`（int8），並於缺值填補前先建立旗標。
- `panel_builder_v3` 完成缺值填補後，須在 `reports_v3/missing_value_report.json` 中記錄：欄位名稱、策略、各步驟填補數量。
- Search 與 city 指標如出現缺值同樣建立 `_was_missing`；POI PCA 輸出不需要旗標。

### 3.3 派生特徵產生規則
#### 3.3.1 Lag 與 Rolling
- 定義 `METRIC_SET_FULL` 為 38 欄：所有新建案、本地/鄰近二手房、土地資料中的金額、數量、價格、面積與 `period_new_house_sell_through`。
- 對 `METRIC_SET_FULL` 產生：
  - `lag_1`, `lag_3`, `lag_6`。
  - `rolling_mean_3`, `rolling_mean_6`（先 `shift(1)` 再 rolling）。
  - `rolling_std_3`, `rolling_std_6`。
- 定義 `METRIC_SET_LONG = ['amount_new_house_transactions', 'num_new_house_transactions', 'price_new_house_transactions', 'amount_new_house_transactions_nearby_sectors', 'num_new_house_transactions_nearby_sectors', 'price_new_house_transactions_nearby_sectors', 'amount_pre_owned_house_transactions', 'num_pre_owned_house_transactions', 'amount_pre_owned_house_transactions_nearby_sectors', 'num_pre_owned_house_transactions_nearby_sectors', 'transaction_amount', 'transaction_amount_nearby_sectors']`。
  - 僅對 `METRIC_SET_LONG` 另外產生 `lag_12` 與 `rolling_mean_12`。
- 禁止生成 `rolling_min_*`, `rolling_max_*`, `rolling_count_*`。
- Growth 比率只對 `GROWTH_METRICS = ['amount_new_house_transactions', 'num_new_house_transactions', 'price_new_house_transactions', 'amount_pre_owned_house_transactions', 'num_pre_owned_house_transactions', 'transaction_amount']` 產生 `growth_1` 與 `growth_3`。
- 生成後立即將 `lag_*`、`rolling_*`、`growth_*` 的缺值以同 sector 前向填補；不得再使用城市或全域統計。

#### 3.3.2 比例與權重特徵
- 保留人口權重 share 特徵僅限以下 10 欄：`amount_new_house_transactions`, `num_new_house_transactions`, `price_new_house_transactions`, `amount_new_house_transactions_nearby_sectors`, `num_new_house_transactions_nearby_sectors`, `amount_pre_owned_house_transactions`, `num_pre_owned_house_transactions`, `transaction_amount`, `transaction_amount_nearby_sectors`, `num_land_transactions`。輸出欄位命名沿用 v2 (`<col>_share`, `<col>_weighted_mean`)，共 20 欄。
- 移除 v2 的 `interaction_features` 產生流程；v3 不建立任意乘積或比值。

#### 3.3.3 Search Keyword 特徵
- 在 `panel_builder_v3` 中完成 keyword 正規化後，於 `features_v3` 執行以下步驟：
  1. 以 2019-01～2024-07 的 `search_volume` 為基礎，針對每個 `keyword_slug` 計算變異數，按降冪排序。
  2. 選取變異數最高的 12 個 keywords，保存為 `reports_v3/search_keywords_v3.json`（含 slug 及變異數）。
  3. 僅對這 12 個欄位產出特徵：原始值、`_was_missing`, `lag_1`, `rolling_mean_3`, `pct_change` （對前一期的比率）, `zscore_3m`。共 72 欄。
  4. 禁止產出 `lag_3`, `lag_6`, `density_adj`。

#### 3.3.4 POI 處理
- 保留 3.1 所列核心 POI 欄位。除了核心欄位外，所有 `_dense` 與 `number_of_` 欄位僅用於 PCA：
  1. 對 `sector_POI` 中除核心欄位外的全部 `_dense` 欄位先做 `log1p`（缺值視為 0）。
  2. 對上述矩陣執行 PCA，保留前兩個主成分 `poi_pca_1`, `poi_pca_2`，覆蓋 95% 以上的變異。
  3. PCA 輸出保存到 `reports_v3/poi_pca_summary.json`（成分數、累積解釋率）。
  4. 原始 `_dense` 與 `number_of_` 欄位在寫出特徵矩陣前全部丟棄。

#### 3.3.5 Log 變換
- 僅對 `METRIC_SET_FULL` 加上 `log1p` 版本；禁止對 share 或 search 欄位取 log。
- `target_filled_log1p` 同樣保留。

#### 3.3.6 特徵清單與計數
- `features_v3.py` 在輸出前需組裝最終欄位清單，存成 `reports_v3/feature_inventory_v3.json`，格式：
  ```
  {
    "total_features": <int>,
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
      "poi_pca": [...],
      "time": [...]
    }
  }
  ```
- 若 `total_features` 不在 540–580 之間，`build_feature_matrix_v3` 必須 raise `ValueError`，要求工程師調整設定後重跑。

## 4. 流水線修改
1. 新增 `src/features_v3.py`，遵照第 3 節規則實作；保留與 v2 相同的 `build_feature_matrix_v3` 函式介面。
2. 新增 `src/splits_v3.py`，定義時間切分：
   ```
   FOLD_DEFINITIONS_V3 = [
       ("2019-01-01", "2023-12-31", "2024-01-01", "2024-02-29"),
       ("2019-01-01", "2024-02-29", "2024-03-01", "2024-04-30"),
       ("2019-01-01", "2024-04-30", "2024-05-01", "2024-06-30")
   ]
   HOLDOUT_WINDOW = ("2019-01-01", "2024-06-30", "2024-07-01", "2024-07-31")
   ```
   - `generate_time_series_folds_v3` 要輸出 `(train_idx, valid_idx)`，並記錄於 `reports_v3/folds_definition_v3.json`。
3. 新增 `config/automl_v3.yaml`，指定：
   ```yaml
   automl:
     metric: competition_score
     time_budget: 5400  # 秒
     estimator_list: ["xgboost"]
     task: regression
     n_jobs: -1
     seed: 42
     gpu_per_trial: 1
     eval_method: cv
     split_type: precomputed
   xgboost_params:
     tree_method: gpu_hist
     predictor: gpu_predictor
     max_depth: [3, 6]
     min_child_weight: [0.5, 8]
     subsample: [0.6, 0.95]
     colsample_bytree: [0.5, 0.9]
     gamma: [0, 5]
     reg_alpha: [0.0001, 5]
     reg_lambda: [0.5, 10]
     learning_rate: [0.02, 0.12]
     n_estimators: [400, 1600]
     max_leaves: [0, 256]
   ```
   - 若使用 FLAML，需透過 `automl.add_learner` 將上述搜尋空間註冊進 `xgboost`，不得啟用其他演算法。
4. `scripts/run_pipeline_v3.py` 需提供下列子命令：
   - `build_panel`: 讀 `config/data_paths_v3.yaml`，呼叫 `panel_builder_v3.build_panel`。
   - `build_features`: 讀 panel，呼叫 `features_v3.build_feature_matrix_v3`，寫入 `data/processed/features_v3.parquet` 與報表。
   - `run_xgb`: 使用 `automl_runner_v3` 執行（詳見第 6 節）。
   - `evaluate_holdout`: 使用最佳模型設定對 `HOLDOUT_WINDOW` 推論並輸出評估。
   - `predict`: 產生 submission；支援殘差修正參數。
5. 所有輸出與 log 改寫至 `logs_v3/`, `reports_v3/`, `models_v3/`, `submission_v3/` 目錄，不得覆寫 v2 artifacts。

## 5. 評估流程
1. **交叉驗證**：針對 `FOLD_DEFINITIONS_V3` 逐一訓練並記錄：
   - 指標：`competition_score`, `SMAPE`, `MAPE`, `MAE`, `RMSE`。
   - 每個 fold 的指標寫入 `reports_v3/fold_metrics_v3.csv`。
   - 保留每 fold 的預測與實際值至 `reports_v3/predictions_fold_<k>.parquet`。
2. **獨立留存月 (2024-07)**：
   - 使用第 6 节所述最佳設定，訓練於 2019-01～2024-06，對 2024-07 產生預測。
   - 指標同上，輸出 `reports_v3/holdout_metrics_v3.json`。
   - 若 `competition_score` < 0.7 或 `MAPE` > 0.45，流程須終止並回報。
3. **Kaggle 測試月模擬**：
   - 以所有可用標註 (<=2024-07) 重新訓練終極模型，對 2024-08～2024-12 預測，結果寫入 `submission_v3/submission.csv`。
   - 另輸出 `reports_v3/test_window_summary.json`（含月別平均、標準差、最值）。
4. **特徵重要度盤點**：
   - 針對終極模型呼叫 `booster.get_score(importance_type='gain')`。
   - 排序後寫入 `reports_v3/feature_importance_gain.csv`，前 50 名另存 `reports_v3/top_features.txt`。
5. **診斷圖表**：
   - 產生 `reports_v3/plots/residual_vs_target_2024-07.png` 與 `reports_v3/plots/prediction_distribution_train.png`（使用 matplotlib），提供殘差與預測分佈檢查。

## 6. 模型與超參設定
1. `src/automl_runner_v3.py`：
   - 僅支援 XGBoost GPU。每個 fold 先以 `StandardScaler` 處理數值特徵（dtype float），旗標欄位 (`*_was_missing`, `is_future`, `target_available_flag`) 保持原狀。
   - 每個 fold 的 AutoML 搜尋時間上限 900 秒；若時間不足以完成 20 次試驗則延長至 1200 秒。
   - AutoML 訓練資料僅使用 `is_future == 0`，並排除目前 fold 的驗證區間。
   - 對於 AutoML 回傳的最佳組合，再以 `early_stopping_rounds=100` 在該 fold 的驗證集上重新訓練一次，取得最終模型與預測。
   - 儲存 `models_v3/fold_<k>_model.json`（XGBoost 原生存檔）、`models_v3/fold_<k>_scaler.pkl` 及 `reports_v3/fold_<k>_summary.json`。
2. **整合訓練**：
   - 選取平均 `competition_score` 最佳的折數設定。
   - 以該設定訓練 2019-01～2024-06，保存為 `models_v3/best_model_holdout.json`，並在留存月上評估。
   - 評估完成後，以 2019-01～2024-07 全量資料重訓，保存 `models_v3/best_model_full.json` 與 `models_v3/scaler_full.pkl`，供最終預測。
3. **配置保存**：
   - 將最佳超參輸出為 `reports_v3/best_config_v3.json`。
   - 保存 AutoML 試驗紀錄於 `logs_v3/automl_trials.csv`。

## 7. 殘差修正流程
1. 新增 `scripts/residual_correction_v3.py`，輸入：
   - fold 預測 (`reports_v3/predictions_fold_*.parquet`)。
   - holdout 預測 (`reports_v3/predictions_holdout.parquet`)。
2. 特徵：`prediction`, `month_index`, `sector_id`, `resident_population`, `population_scale`, `amount_new_house_transactions`, `num_new_house_transactions`, `amount_pre_owned_house_transactions`, `num_pre_owned_house_transactions`, `transaction_amount`, `transaction_amount_nearby_sectors`, 以及 12 個 `search_kw_<slug>` 欄位，共 23 欄。
3. 流程：
   - 將 fold 與 holdout 殘差 (`target - prediction`) 合併，僅使用 `is_future == 0`。
   - `StandardScaler` -> `Ridge(alpha=0.5)`，使用時間序 5 折驗證（依月份遞增切分），需達 `MAPE` < 0.40 才允許保存。
   - 輸出 `models_v3/residual_scaler.pkl`, `models_v3/residual_model.pkl`, `reports_v3/residual_training_metrics.json`, `reports_v3/residual_features.json`（列出使用的欄位）。
4. `predict` 子命令如提供殘差模型，則於主模型結果後套用校正：`prediction += residual_model.predict(residual_features)`。

## 8. 測試、紀錄與交付
- **靜態檢查**：`python -m py_compile src/**/*.py scripts/**/*.py`。
- **單元測試**：更新或新增以下測試檔案：
  - `tests/test_features_v3.py`：驗證特徵計數、禁止欄位、PCA 成分數。
  - `tests/test_splits_v3.py`：確認折數與 holdout 覆蓋範圍正確。
  - `tests/test_metrics_v3.py`：確保 `competition_score` 在 MAPE>100% 時回傳 0。
- **煙霧流程**：
  ```
  python scripts/run_pipeline_v3.py build_panel --data-config config/data_paths_v3.yaml --panel-path data/processed/panel_v3.parquet
  python scripts/run_pipeline_v3.py build_features --panel-path data/processed/panel_v3.parquet --features-path data/processed/features_v3.parquet --reports-dir reports_v3
  python scripts/run_pipeline_v3.py run_xgb --features-path data/processed/features_v3.parquet --config config/automl_v3.yaml
  python scripts/run_pipeline_v3.py evaluate_holdout --features-path data/processed/features_v3.parquet --model-path models_v3/best_model_holdout.json --scaler-path models_v3/scaler_full.pkl --reports-dir reports_v3
  python scripts/run_pipeline_v3.py predict --panel-path data/processed/panel_v3.parquet --features-path data/processed/features_v3.parquet --model-path models_v3/best_model_full.json --scaler-path models_v3/scaler_full.pkl --sample-submission sample_submission.csv --output-path submission_v3/submission.csv --residual-model models_v3/residual_model.pkl --residual-scaler models_v3/residual_scaler.pkl --residual-features reports_v3/residual_features.json
  ```
- **交付清單**：
  - `data/processed/panel_v3.parquet`, `data/processed/features_v3.parquet`
  - `config/data_paths_v3.yaml`, `config/automl_v3.yaml`
  - `models_v3/` 下所有模型與 scaler 檔案
  - `reports_v3/`：`panel_build_summary.json`, `missing_value_report.json`, `feature_inventory_v3.json`, `search_keywords_v3.json`, `fold_metrics_v3.csv`, `holdout_metrics_v3.json`, `feature_importance_gain.csv`, `test_window_summary.json`, `residual_training_metrics.json`, `residual_features.json`, `plots/`
  - `logs_v3/automl.log`, `logs_v3/automl_trials.csv`
  - `submission_v3/submission.csv`

