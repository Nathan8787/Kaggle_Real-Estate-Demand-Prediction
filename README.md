# 中國房地產需求預測 AutoML v3.1 使用手冊

本手冊說明如何依照 `v3/docs/AUTOML_SPEC_v3.md` 的規格建構資料流程、訓練模型並產出提交檔。所有內容皆以繁體中文撰寫，並完整呼應規格要求，確保每一步都有審計追蹤與一致的評分標準。

## 1. 版本核心目標
- **校正訓練/推論落差**：交叉驗證、留存月與最終推論全程共用同一套前瞻特徵投影，避免 v2 曾發生的分數落差。
- **控制特徵維度**：最終特徵數必須介於 450 與 650 之間；若超出範圍流程會立即中止，並在報告中標示失敗狀態。
- **資料安全界線**：任何模型、殘差修正與投影步驟僅能使用 2024-07（含）以前的真實標籤，嚴禁洩漏未來資訊。
- **完整審計鏈**：每個步驟都輸出報表或指標（投影飄移、特徵清單、模型設定等），可追溯「投影 → 特徵 → 訓練 → 推論」。
- **官方指標一致性**：所有評估均呼叫 `metrics/competition_score.py`，確保與競賽規則（APE>1 置零）完全一致。

## 2. 環境與資料來源
### 2.1 執行環境
- 建議使用 Python 3.10 以上版本。
- 需安裝 pandas、numpy、scikit-learn、xgboost、lightgbm、flaml、matplotlib、joblib、pyyaml 等套件。
- 建議在專案根目錄建立虛擬環境後安裝需求套件。

### 2.2 檔案對照
在專案根目錄確保以下資料與設定檔存在：

| 檔案 | 鍵值 | 主要欄位 | 備註 |
| --- | --- | --- | --- |
| `train/new_house_transactions.csv` | `month`, `sector` | 新建案本地 9 欄 | 亦提供目標欄位 |
| `train/new_house_transactions_nearby_sectors.csv` | 同上 | 新建案鄰近 9 欄 |  |
| `train/pre_owned_house_transactions.csv` | 同上 | 二手房本地 4 欄 |  |
| `train/pre_owned_house_transactions_nearby_sectors.csv` | 同上 | 二手房鄰近 4 欄 |  |
| `train/land_transactions.csv` | 同上 | 土地本地 4 欄 |  |
| `train/land_transactions_nearby_sectors.csv` | 同上 | 土地鄰近 4 欄 |  |
| `train/sector_POI.csv` | `sector` | 9 個核心 POI + `_dense` 欄位 | `_dense` 僅用於 PCA |
| `v3/config/sector_city_map_v3.csv` | `sector` | `city_id` | 僅含歷史存在的 sector |
| `train/city_search_index.csv` | `month`,`keyword`,`source` | `search_volume` | 僅使用至 2024-07 |
| `train/city_indexes.csv` | `city_indicator_data_year` | 48 項年度指標 | 最終保留 12 欄 |
| `test.csv` | `id` | 2024-08 ~ 2024-12 | 測試集 |
| `sample_submission.csv` | `id` | Kaggle 格式 | 最終比對順序 |

所有 CSV 皆以 UTF-8 讀取、欄名轉為 snake_case、字串欄位做 NFKC 正規化並移除空白。如在測試集中遇到 `sector` 沒出現在 `sector_city_map_v3.csv`，流程會立即拋錯提醒補齊。

## 3. 主要模組與責任劃分
以下模組皆位於 `v3/src/` 或 `v3/scripts/` 目錄，並按規格實作。

### 3.1 面板建構：`panel_builder_v3.py`
- 定義常數：
  ```python
  MONTH_RANGE = pd.date_range('2019-01-01', '2024-12-01', freq='MS')
  TARGET_CUTOFF = pd.Timestamp('2024-07-31')
  FORECAST_MONTHS = pd.date_range('2024-08-01', '2024-12-01', freq='MS')
  ```
- `load_raw_tables_v3` 讀取 `config/data_paths_v3.yaml` 與 `sector_city_map_v3.csv`，缺少對應 `city_id` 時立即 raise。
- `build_calendar_v3` 建立 2019-01 至 2024-12 的 `month × sector` MultiIndex，並解析 `sector_id`（純數字）與整合 `city_id`（`np.int16`）。
- `merge_sources_v3` 僅合併實際觀測值（不延伸未來月份），依序整併六個交易表、POI、年度指標（轉月後 forward fill）與搜尋指數。
- `attach_target_v3` 建立 `target_raw`, `target_available_flag`, `target`, `target_filled`, `target_filled_was_missing`, `is_future`，並組成 `id = "YYYY Mon_sector <sector_id>"`。
- `save_panel_v3` 輸出 `data/processed/panel_v3.parquet`（依年份分割）與 `reports_v3/panel_build_summary.json`，總結時間範圍、標籤筆數、未來筆數與缺標比例。面板會移除字串型 `sector`，僅保留 `sector_id` 與 `city_id`。

### 3.2 前瞻投影：`projection_v3.py`
- 主要 API：
  - `apply_projection(panel_df, forecast_start)`：針對 `PROJECTION_COLUMNS` 在 `month >= forecast_start` 的列產生投影值，並新增 `*_proj_source`、`*_proj_overridden`。
  - `collect_projection_drift(original_df, projected_df, forecast_start)`：計算各欄位的 MAE、MAPE、RMSE、P10、P90，輸出 `reports_v3/projection_drift_<YYYYMM>.json`。
- `PROJECTION_COLUMNS` 包含所有 34 個交易欄位（本地與鄰近）、所有搜尋欄位與精選的 12 個城市指標；POI 與 `target_*` 不投影。
- 演算法：
  1. 僅用 `month < forecast_start` 的非 NaN 歷史資料。
  2. >=12 筆以 EWMA（衰減係數 0.7）計算。
  3. 6–11 筆採簡單平均；3–5 筆採中位數。
  4. 少於 3 筆時，以同城市、同月份的中位數支援；再不足則用全域中位數，最後保底 0。
  5. 所有投影值以 `np.clip(..., 0, None)` 限制非負，並標記來源代碼（0=觀測、1=EWMA、2=平均/中位數、3=同城共享、4=全域備援）。
- 在交叉驗證、留存月及最終推論皆以對應 `forecast_start` 套用投影，確保流程一致。

### 3.3 特徵工程：`features_v3.py`
- 入口 `build_feature_matrix_v3(panel_path, forecast_start, features_path, reports_dir)`：
  1. 載入 panel 後套用 `apply_projection`。
  2. 僅保留 `month <= TARGET_CUTOFF` 的真實標籤；預測月份 `target` 維持 NaN。
  3. 建立缺值旗標、依規格補值，再生成衍生特徵。
  4. 輸出分年 Parquet 與 `reports_v3/feature_inventory_v3.json`。
- 允許欄位：
  - 34 個月度交易欄位（本地/鄰近新建案、二手房、土地）。
  - 目標衍生欄位：`target`, `target_filled`, `target_available_flag`, `target_filled_was_missing`, `is_future`。
  - POI 核心欄位、時間欄位（`year`, `month_num`, `quarter`, `month_index`, `days_in_month`, `is_year_start`, `is_year_end`）、`sector_id`, `city_id`。
  - 城市指標 12 欄與對應 `was_interpolated`。
  - 搜尋欄位則於投影後根據互資訊挑選 12 個 slug。
- 缺值處理：
  - 對所有月度欄位建立 `*_was_missing` 旗標。
  - 依規格於 `history_mask = month < forecast_start` 範圍內進行 forward fill、滾動中位數或整體中位數補值，最後填 0；城市指標與搜尋欄位依規範補齊。
  - 缺值報告寫入 `reports_v3/missing_value_report.json`，含 `projection_stage`（觀測或投影）。
- 指標集合：
  - `METRIC_SET_FULL`、`METRIC_SET_LONG`、`GROWTH_METRICS`、`SHARE_METRICS` 依表列定義。
- 衍生特徵：
  - `target_filled` 與各指標的 lag、rolling（3/6/12 月）、標準差、成長率、share/加權平均、`log1p` 版本。所有計算僅針對 `history_mask`，未來月份保持 NaN 待推論時計算。
  - 搜尋關鍵字透過 3 折時間序 `mutual_info_regression` 評估後挑選 12 個 slug，保留原始值、`lag_1`、`rolling_mean_3`、`pct_change_1`、`zscore_3m` 及缺值旗標，相關資訊寫入 `reports_v3/search_keywords_v3.json`。
  - POI `_dense` 欄位先 `log1p`，在歷史資料上以 PCA (n=2, random_state=42) 轉換，產出 `poi_pca_1`, `poi_pca_2` 並於 `reports_v3/poi_pca_summary.json` 紀錄解釋率。
- 特徵配額：`feature_inventory_v3.json` 包含 `total_features`, `min_features=450`, `max_features=650`, `delta_vs_v30`, 各分類清單與 `projection_suppressed_features`。若數量超界即 raise 並在報告標記 `status="failed"`。
- 投影飄移防護：若某欄位在最新可用月份出現高 MAPE 或投影來源 3+4 佔比 > 85%，系統會移除其高階衍生（rolling/growth/share/log1p），並記錄在報告中。

### 3.4 時序切分：`splits_v3.py`
- Fold 定義：
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
- `generate_time_series_folds_v3`：依 month、sector 排序，對每折計算 train/valid mask，並在驗證前 1 個月設置 purge。每折會以對應的 `forecast_start` 套用投影並返回索引陣列，同時寫入 `reports_v3/folds_definition_v3.json`。

### 3.5 AutoML 與模型訓練：`automl_runner_v3.py`
- `config/automl_v3.yaml` 指定 FLAML 設定：
  - `time_budget_per_fold=900` 秒、每折至少 25 次試驗、模型僅含 XGBoost 與 LightGBM。
  - 預設 `zero_inflation: {enabled: true, method: tweedie}`，若改為 `hurdle` 會啟用兩段式流程。
  - 搜尋空間詳見規格（含樹數、深度、learning_rate、正則化等上下限）。
- 訓練流程：
  - 僅使用 `is_future == 0` 且 `month <= train_end` 的資料，目標為原始金額。
  - 特徵排除 `month`, `id`；保留 `sector_id`, `city_id`。
  - 样本權重：`month` 在 2024-04~2024-06 的列權重 1.8，其餘 1.0，並於報告記錄統計。
  - 單調性約束：對 `resident_population`, `population_scale`, `surrounding_housing_average_price`, `surrounding_shop_average_rent`, `city_gdp_100m`, `city_total_retail_sales_of_consumer_goods_100m` 設為正向；XGBoost 使用字串格式，LightGBM 使用 list。
  - 透過 FLAML 逐個估計器搜尋，必要時延長時間以達到最小試驗數。所有試驗寫入 `logs_v3/automl_trials.csv`。
  - 取得最佳組態後重新以原始框架訓練，並保存每折模型（`models_v3/fold_<k>_{xgb,lgbm}`）與特徵重要度。
  - 評估時比較 XGB、LGBM 與 0.5/0.5 集成，分數寫入 `reports_v3/fold_metrics_v3.csv`，預測另存 `reports_v3/predictions_fold_<k>_<model>.parquet`。
  - `reports_v3/automl_summary.json` 統整兩模型的最佳設定、迭代次數、fold 分數、集成成績、樣本權重策略與單調性約束。
- 留存月訓練：
  - `forecast_start = 2024-07-01`，資料到 2024-06-30。輸出 `models_v3/best_model_holdout_xgb.json`, `models_v3/best_model_holdout_lgbm.txt`, `reports_v3/holdout_feature_importance_{model}.csv` 以及 `reports_v3/holdout_metrics_v3.json`。若分數低於門檻會於報告標註警示。
- 全量訓練：
  - `forecast_start = 2024-08-01`，資料到 2024-07-31。輸出 `models_v3/best_model_full_{xgb,lgbm}`、`reports_v3/feature_importance_full_{model}.csv` 與 `reports_v3/top_features.txt`。

### 3.6 零膨脹與量化預測
- `zero_inflation`：
  - 預設 Tweedie 單段；改為 `method: hurdle` 時，Stage-A 使用 `LGBMClassifier` 預測是否為正值（模型檔 `models_v3/hurdle_classifier_<phase>.txt`）；Stage-B 以 Tweedie 迴歸估計條件期望（依模型命名區分 XGB/LGBM）。
  - 所有結果寫入 `reports_v3/zero_inflation_summary.json`，包含是否啟用、AUC、PR-AUC、Stage-B MAPE 及集成得分。
- 量化預測：
  - 另外以 LightGBM 訓練 P50、P80 模型，儲存於 `models_v3/quantile_full_alpha50.txt`, `models_v3/quantile_full_alpha80.txt`，並在報告中統計覆蓋率（`reports_v3/uncertainty_summary.json`）。

### 3.7 殘差修正：`scripts/residual_correction_v3.py`
- 收集所有折數與留存月預測，僅使用 `target_available_flag == 1` 的列。
- 特徵包含主模型預測、月份索引、`sector_id`, `city_id`, 核心 POI、人口欄位、主要交易指標以及 12 個搜尋 slug。
- 以時間序 KFold（5 折）評估 `MAPE` 與 `RMSE`。平均 `MAPE >= 0.40` 則標記 `status="rejected"` 並不輸出模型；反之儲存 `residual_scaler.pkl`, `residual_model.pkl`, `reports_v3/residual_features.json` 與 `residual_training_metrics.json`。

### 3.8 推論與提交：`predict_v3.py`
- 介面：
  ```python
  generate_submission(
      panel_path,
      features_path,
      model_paths,
      sample_submission,
      output_path,
      reports_dir,
      *,
      zero_inflation=None,
      residual_bundle=None,
      quantile_paths=None,
  )
  ```
- 推論流程：
  1. 以 panel 為基準，每月（2024-08 至 2024-12）重覆：套用投影 → 建立當月特徵 → 分別以 XGB/LGBM 推論 → 取均值集成。
  2. 若設定 `hurdle`，會使用分類器與回歸器計算 `prob_positive * regression_pred`。
  3. 若提供殘差模型則先標準化後加總修正，再以 `np.clip` 保證非負。
  4. 若提供分位模型則同步輸出 `prediction_p50`, `prediction_p80`；未提供時填 `NaN` 以維持欄位。
  5. 每輪將預測寫回 `target_filled` 供下一月 lag 使用。
- 結果會與 `sample_submission.csv` 合併後輸出 `submission_v3/submission.csv`，並寫入 `reports_v3/test_window_summary.json`：
  - 包含月度統計、XGB/LGBM/ensemble 分解、零膨脹策略、分位覆蓋率、`projection_source_ratio`（含 `proj_source==4`），若有欄位在某月的投影來源 (1–4) 超過 85% 會標記 `high_projection_dependence=true` 並列出欄位。
- 另外產生診斷圖：`reports_v3/plots/residual_vs_target_2024-07.png`, `reports_v3/plots/prediction_distribution_train.png`, `reports_v3/plots/prediction_distribution_test.png`。

## 4. Pipeline 操作流程
建議於專案根目錄執行以下步驟，或先 `cd v3` 後依序操作。除非另有說明，所有指令皆需使用真實資料（禁止合成資料）。

1. **建立面板**
   ```bash
   python scripts/run_pipeline_v3.py build_panel \
       --data-config config/data_paths_v3.yaml \
       --panel-path data/processed/panel_v3.parquet \
       --reports-dir reports_v3
   ```
2. **建立特徵矩陣**（需指定投影起點，示例為 2024-06-01 用於交叉驗證）
   ```bash
   python scripts/run_pipeline_v3.py build_features \
       --panel-path data/processed/panel_v3.parquet \
       --features-path data/processed/features_v3.parquet \
       --reports-dir reports_v3 \
       --forecast-start 2024-06-01
   ```
3. **執行 AutoML 搜尋與再訓練**
   ```bash
   python scripts/run_pipeline_v3.py run_automl \
       --features-path data/processed/features_v3.parquet \
       --config config/automl_v3.yaml \
       --reports-dir reports_v3
   ```
   - 可額外指定 `--models-dir` 與 `--logs-dir`（預設分別為 `models_v3`、`logs_v3`）。
4. **留存月評估**
   ```bash
   python scripts/run_pipeline_v3.py evaluate_holdout \
       --panel-path data/processed/panel_v3.parquet \
       --features-path data/processed/features_v3.parquet \
       --model-xgb models_v3/best_model_holdout_xgb.json \
       --model-lgbm models_v3/best_model_holdout_lgbm.txt \
       --reports-dir reports_v3 \
       [--hurdle-classifier models_v3/hurdle_classifier_holdout.txt \
        --hurdle-regressor models_v3/hurdle_regressor_holdout.txt]
   ```
5. **產出最終提交檔**
   ```bash
   python scripts/run_pipeline_v3.py predict \
       --panel-path data/processed/panel_v3.parquet \
       --features-path data/processed/features_v3.parquet \
       --model-xgb models_v3/best_model_full_xgb.json \
       --model-lgbm models_v3/best_model_full_lgbm.txt \
       --sample-submission ../sample_submission.csv \
       --output-path submission_v3/submission.csv \
       --reports-dir reports_v3 \
       [--residual-model models_v3/residual_model.pkl \
        --residual-scaler models_v3/residual_scaler.pkl \
        --residual-features reports_v3/residual_features.json \
        --quantile-p50 models_v3/quantile_full_alpha50.txt \
        --quantile-p80 models_v3/quantile_full_alpha80.txt]
   ```

上述步驟亦可作為規格要求的煙霧測試（須按順序執行）。

## 5. 產出與報表清單
- `data/processed/panel_v3.parquet`
- `data/processed/features_v3.parquet`（依年份分割）
- `config/data_paths_v3.yaml`, `config/automl_v3.yaml`, `config/sector_city_map_v3.csv`
- `models_v3/`：
  - `best_model_holdout_xgb.json`, `best_model_holdout_lgbm.txt`
  - `best_model_full_xgb.json`, `best_model_full_lgbm.txt`
  - 若啟用 hurdle：`hurdle_classifier_holdout.txt`, `hurdle_classifier_full.txt`, `hurdle_regressor_holdout_xgb.json`, `hurdle_regressor_holdout_lgbm.txt`, `hurdle_regressor_full_xgb.json`, `hurdle_regressor_full_lgbm.txt`
  - 量化模型：`quantile_full_alpha50.txt`, `quantile_full_alpha80.txt`
  - 殘差相關：`residual_model.pkl`, `residual_scaler.pkl`
- `reports_v3/`：`panel_build_summary.json`, `missing_value_report.json`, `projection_drift_*.json`, `feature_inventory_v3.json`, `search_keywords_v3.json`, `poi_pca_summary.json`, `fold_metrics_v3.csv`, `automl_summary.json`, `holdout_metrics_v3.json`, `holdout_feature_importance_{xgb,lgbm}.csv`, `feature_importance_full_{xgb,lgbm}.csv`, `top_features.txt`, `test_window_summary.json`, `zero_inflation_summary.json`, `uncertainty_summary.json`, `residual_training_metrics.json`, `residual_features.json`, `plots/` 目錄。
- `logs_v3/automl.log`, `logs_v3/automl_trials.csv`
- `submission_v3/submission.csv`

## 6. 測試與驗證要求
- 靜態檢查：
  ```bash
  python -m py_compile src/**/*.py scripts/**/*.py
  ```
- 單元測試：
  ```bash
  pytest tests/test_projection_v3.py::test_projection_respects_cutoff
  pytest tests/test_features_v3.py::test_feature_counts_and_categories
  pytest tests/test_splits_v3.py::test_fold_alignment
  pytest tests/test_metrics_v3.py::test_competition_metric_cap
  ```
- 煙霧流程：依照第 4 節的五個指令順序執行，確保 `build_panel → build_features → run_automl → evaluate_holdout → predict` 全流程通過。

## 7. 重要注意事項
1. 嚴禁使用 2024-08 以後的真實標籤訓練或投影，確保無資料洩漏。
2. `feature_inventory_v3.json` 會即時檢查特徵總數是否落在 450–650，若要新增或刪除特徵，需同步更新衍生邏輯與報告。
3. 所有評估一律使用 `metrics/competition_score.py`；請勿自行實作替代版本，以免造成分數偏移。
4. 投影飄移報告與 `projection_suppressed_features` 是追蹤風險的關鍵，當高階特徵被停用時需重新檢視模型表現。
5. 在重新訓練前，請先備份或清空 `models_v3/`、`reports_v3/`、`logs_v3/` 與 `submission_v3/`，避免舊檔案干擾。
6. 若啟用零膨脹或殘差修正，請確認在最終推論時提供對應模型與特徵清單，否則將回落為基本集成預測。

依上述說明執行，即可在符合規格 v3.1 的前提下，重現從資料整備到最終 Kaggle 提交的完整流程。祝實驗順利！
