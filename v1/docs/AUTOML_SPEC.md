# Real Estate Demand Prediction – Data Prep & AutoML Specification

## 1. Objectives
- Build a deterministic month–sector panel that merges every training source and mirrors the Kaggle submission layout.
- Engineer time-aware, leakage-free features, including sector-level weights and population-adjusted aggregates.
- Run a standardized AutoML experiment (FLAML) with a custom competition metric and a fixed estimator list.
- Deliver reproducible tooling (Python scripts, configs, logs, artifacts) ready for hand-off to execution agents.

## 2. Data Inventory & Assumptions
| File | Description | Keys | Integration Rules |
| --- | --- | --- | --- |
| `train/new_house_transactions.csv` | Target + core new-house metrics | `month`, `sector` | `amount_new_house_transactions` (10k CNY) is the prediction target. Absence of a month–sector row implies a true target value of 0. |
| `train/new_house_transactions_nearby_sectors.csv` | Neighbor-sector new-house metrics | `month`, `sector` | Join on `month`, `sector` and suffix `_nearby`. |
| `train/pre_owned_house_transactions.csv` | Local secondary market metrics | `month`, `sector` | Join on `month`, `sector`. |
| `train/pre_owned_house_transactions_nearby_sectors.csv` | Neighbor secondary metrics | `month`, `sector` | Join on `month`, `sector` and suffix `_nearby`. |
| `train/land_transactions.csv` | Local land market metrics | `month`, `sector` | Join on `month`, `sector`. |
| `train/land_transactions_nearby_sectors.csv` | Neighbor land metrics | `month`, `sector` | Join on `month`, `sector` and suffix `_nearby`. |
| `train/sector_POI.csv` | Static POI & density descriptors | `sector` | Broadcast to every month in the panel. Supplies `resident_population` for weighting. |
| `train/city_search_index.csv` | City-level monthly search volumes | `month`, `keyword`, `source` | After UTF‑8 ingestion and normalization, aggregate to a monthly vector; replicate across all sectors for the same month. |
| `train/city_indexes.csv` | Annual macro indicators | `city_indicator_data_year` | Assume all sectors belong to one city. Expand yearly values to every month of that year; forward-fill into future months once annual data ends. |
| `test.csv` | Submission ids | `id` | Months: 2024-08 through 2024-12. |
| `sample_submission.csv` | Submission template | `id`, `new_house_transaction_amount` | Used to enforce final output order. |

## 3. Environment & Tooling
- Python 3.13.2 (confirm via `python --version`).
- Core libraries: `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `category_encoders`, `scipy`, `unidecode`, `pyyaml`.
- AutoML stack: `flaml`, with backends `lightgbm`, `xgboost`, `catboost`, `scikit-learn` (for random forest, extra trees, linear models, k-NN).
- Logging: `loguru` for structured logs. Config management: `omegaconf/hydra` optionality removed; use static YAML configs.
- All scripts run with UTF‑8 locale (`PYTHONUTF8=1`) to preserve Chinese text.

## 4. Panel Construction
1. **Load tables (`data_loading.load_raw_tables`)**
   - Read CSVs with `encoding='utf-8'`, `dtype` maps per column family.
   - Normalize column names to snake_case.
   - Normalize text columns (keywords) using Unicode NFKC and store both original text and an ASCII slug (`slugify` via `unidecode`).
2. **Create base calendar (`panel_builder.build_calendar`)**
   - Parse `month` using `pd.to_datetime` with format `%Y-%b`.
   - Determine month range from 2019-01 to 2024-12 (train min to submission max).
   - Build full cartesian product of unique sectors (95) and every month in range.
3. **Merge data sources (`panel_builder.merge_sources`)**
   - Left join monthly datasets on `['month', 'sector']`.
   - Broadcast `sector_POI` to every month.
   - Expand `city_indexes`: map each record to every month of the corresponding year; for months after the last available year (2023 if applicable), forward-fill values.
   - Process `city_search_index`:
     - Aggregate `search_volume` per (`month`, `keyword`) across sources.
     - Rank keywords by variance over time; select the top 30 keywords.
     - Pivot to `month` rows with columns `search_kw_<ascii_slug>` using normalized keyword slugs.
     - Merge on `month` and replicate to every sector.
4. **Target creation (`panel_builder.attach_target`)**
   - Copy `amount_new_house_transactions` into new `target` column.
   - For month–sector combinations absent in the raw file, set `target = 0`.
   - Drop original `amount_new_house_transactions` after copy.
5. **Submission ID alignment**
   - Create canonical `id` string per row: `f"{month:%Y %b}_sector {sector_id:02d}"` (see Section 6 for sector parsing).
   - Ensure `test.csv` rows align exactly with generated IDs.
6. **Persist panel**
   - Save merged panel as `data/processed/panel.parquet` (partitioned by year) for reuse.

## 5. Missing Value Policy
- **缺失旗標**：對所有會進入補值流程的欄位新增 `*_was_missing`（0/1），在補值前記錄原始缺失狀態，供模型辨識。
- **計數/金額/面積/密度欄位**（欄名以 `num_`/`amount_`/`area_` 開頭或包含 `number_of_`、`_dense`）：依下列順序處理：
  1. 依 `sector_id`, `month` 排序後，針對每個 `sector_id` 進行 `ffill()`。
  2. 仍為 NaN 者，以該 `sector_id` 的歷史中位數（排除 NaN）補值。
  3. 若該 sector 完全沒有歷史資料，再以全體（所有 sector）中位數補值。
  4. 仍遺留的少數缺值最後補 0。
  5. `target_filled` 維持 0 補值，以貼合官方定義。
- **價格/比例/平均欄位**（含 `price` 但非 `city_` 開頭）：先以 `ffill()` 延伸，再呼叫 sector 歷史中位數補值，最後回落全體中位數；即 `_fill_with_sector_history` 新增 `ffill` 前置步驟。
- **期間欄位**（`period_` 前綴）：依 sector `ffill()` → sector 中位數 → 全體中位數。
- **城市層級欄位**（`city_` 前綴）：先於 sector 內 `ffill()`，再對整體 `ffill()`，最後以全體中位數補完。
- **搜尋量 pivot 欄位**：無資料代表無搜尋量，維持 0。
- **補值紀錄**：所有補值步驟統一寫入 `reports/missing_value_report.json`，紀錄各策略填補筆數。

## 6. Feature Engineering (`features.build_feature_matrix`)
1. **Sector parsing**
   - Extract numeric portion via regex `r"(\d+)"`; cast to int. Reject malformed labels with explicit error.
   - Store both `sector_id` (int) and `sector_label` (original string).
2. **Time features**
   - Derive `year`, `month_num`, `quarter`, `month_index` (= months since 2019-01), `is_year_start`, `is_year_end`, `days_in_month`.
3. **Lag & rolling features**
   - For each sector and metric in the set `{target, num_*, area_*, amount_*, price_*}`:
     - Compute lag values at steps 1, 3, 6, 12 months.
     - For rolling stats, dynamically shorten the window to the length of available history (minimum 2 months) and use `min_periods=1`.
     - Keep track of effective window length per observation (`rolling_window_used`).
4. **Growth rates**
   - Compute `(current - lag_n) / (abs(lag_n) + 1e-6)` for lag steps 1 and 3.
5. **Population-weighted city aggregates**
   - For each month and metric above, compute city weighted mean using `resident_population` as weights (fallback to `population_scale` when `resident_population`=0).
   - Create share feature `metric / (weighted_mean + 1e-6)`.
6. **Interaction terms**
   - Compute differences between sector metrics and their nearby-sector counterparts (`metric - metric_nearby`).
   - Create density-adjusted metrics by multiplying POI densities with monthly search-volume z-scores.
7. **Search index features**
   - For each of the 30 selected keywords, compute `month_over_month_pct_change` and a 3-month rolling z-score (using shortened windows as above).
8. **Static transformations**
   - Apply `np.log1p` to skewed counts (`num_*`, `area_*`, POI counts/densities) and store with `_log1p` suffix.
   - Run PCA on POI density columns after log transform; retain components covering ≥95% variance and add as `poi_pca_i`.
9. **Feature output**
   - Return tuple `(features_df, feature_metadata)` where `features_df` contains all engineered features plus `target`, `month`, `sector_id`, `id`.
   - Persist to `data/processed/features.parquet` (partitioned by year).

## 7. Target Transformation & Scaling
- 教育任何模型前，將目標 `target` 轉成 `log1p_target = np.log1p(target)`；AutoML 與自訂訓練都以此轉換後的 y 進行。
- `competition_metric` 會對模型輸出的 log 預測與對應的 log 標籤做 `np.expm1` 還原，再套用競賽評分，確保評估在原始尺度上。
- `predict.generate_submission` 會在每次月度推論時將模型輸出 `np.expm1` 還原、再回寫到 `target_filled` 以支援滾動預測，同時保留 `np.clip(..., 0, None)` 當防呆。
- 保存於 `reports/automl_summary.json` 的模型摘要需記錄 `target_transform: "log1p_expm1"`。
- Fit a `StandardScaler` on the training portion of each CV fold within `automl_runner`.
- Store `scaler_mean_` and `scaler_scale_` per fold for audit.
- During final training (all data up to 2024-07), fit a global `StandardScaler` and persist to `models/scaler.pkl`.

## 8. Dataset Splitting (`splits.generate_time_series_folds`)
- Available training months: 2019-01 to 2024-07 (67 months).
- Generate five expanding-window folds; each validation span is three consecutive months.

| Fold | Train Months (inclusive) | Validation Months |
| --- | --- | --- |
| 1 | 2019-01 – 2022-12 | 2023-01 – 2023-03 |
| 2 | 2019-01 – 2023-03 | 2023-04 – 2023-06 |
| 3 | 2019-01 – 2023-06 | 2023-07 – 2023-09 |
| 4 | 2019-01 – 2023-09 | 2023-10 – 2023-12 |
| 5 | 2019-01 – 2023-12 | 2024-01 – 2024-03 |

- Hold out 2024-04 – 2024-07 as a final internal check before training on full data.
- `splits.generate_time_series_folds` returns a list of `(train_idx, valid_idx)` using positional indices over the chronologically sorted feature frame.

## 9. Custom Competition Metric (`metrics.competition_score`)
```python
import numpy as np

def competition_score(y_true, y_pred):
    eps = 1e-9
    ape = np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))
    if (ape > 1.0).mean() > 0.30:
        return 0.0
    mask = ape <= 1.0
    if mask.sum() == 0:
        return 0.0
    scaled_mape = ape[mask].mean() / mask.mean()
    return 1.0 - scaled_mape
```
- Integration with FLAML:
```python
from flaml import AutoML

def flaml_competition_metric(X_val, y_val, estimator, labels, data, X_train, y_train):
    y_pred = estimator.predict(X_val)
    return competition_score(y_val, y_pred)

automl = AutoML()
automl.add_metric(name="competition_score",
                   metric=flaml_competition_metric,
                   greater_is_better=True)
```
- Use `metric="competition_score"` in `automl.fit` and log raw metric per fold.

## 10. AutoML Configuration (`automl_runner.run`)
- Input: scaled training features, target array, fold index list, config YAML.
- Estimator list: `['lgbm', 'xgboost', 'rf', 'catboost', 'extra_tree', 'lrl1', 'lrl2', 'kneighbor']`.
- Global parameters:
  - `task='regression'`
  - `time_budget=7200` seconds (2 hours)
  - `log_file_name='logs/automl.log'`
  - `seed=42`
  - `metric='competition_score'`
  - `n_jobs=-1`
- For each fold:
  1. Fit `StandardScaler` on training slice, transform train/valid features.
  2. Call `automl.fit` with `eval_method='cv'` and precomputed `(train_idx, valid_idx)` using `sample_weight=None`.
  3. Capture leaderboard via `automl.best_iteration`, `automl.model_history_`.
  4. Store validation predictions and metric in `reports/fold_results.csv`.
- Final model:
  - Refit best estimator on all scaled data up to 2024-07.
  - Persist model as `models/best_model.pkl` using `automl.save_model`.

## 11. Inference (`predict.generate_submission`)
- Reload global `StandardScaler` and `best_model`.
- Regenerate features for months up to 2024-12 using `features.build_feature_matrix`.
- Transform with scaler, predict for rows where `month >= 2024-08`.
- Assemble submission using `sample_submission.csv` order and write to `submission/submission.csv`.
- Log summary statistics (mean, std, min, max) of predictions.

## 12. Handling Non-ASCII Text
- Maintain original Chinese keywords for traceability.
- Store ASCII-safe slug using `unidecode(keyword).replace(' ', '_').lower()` for column names.
- Ensure all logging/writing uses UTF‑8 to avoid `?` substitutions.

## 13. Module Interfaces
| Module | Input | Output |
| --- | --- | --- |
| `data_loading.load_raw_tables(config)` | YAML config with raw paths | Dict of DataFrames keyed by filename stem |
| `panel_builder.build_calendar(raw_tables)` | Dict from loader | DataFrame of month×sector cartesian base |
| `panel_builder.merge_sources(calendar_df, raw_tables)` | Base calendar + tables | Unified panel DataFrame |
| `panel_builder.attach_target(panel_df)` | Unified panel | Panel with `target` + persisted parquet |
| `features.build_feature_matrix(panel_df)` | Panel DataFrame | Tuple `(features_df, metadata)` saved to parquet |
| `splits.generate_time_series_folds(features_df)` | Feature DataFrame | List of `(train_idx, valid_idx)` |
| `metrics.competition_score(y_true, y_pred)` | NumPy arrays | Float score |
| `metrics.flaml_competition_metric(...)` | FLAML callback args | Float score |
| `automl_runner.run(features_df, folds, config)` | Features, fold list, YAML | Saved scaler, model, fold logs |
| `predict.generate_submission(model_path, scaler_path, features_df)` | Paths + features | Submission CSV |

## 14. Interaction Ordering
- Execute feature engineering steps sequentially inside `features.build_feature_matrix`.
- Share features and interaction terms are computed immediately after lag/rolling calculations to ensure they use already imputed monthly metrics.
- PCA runs after log transforms and before scaling.

## 15. Deliverables
- Source files under `src/` matching module definitions.
- Config files (`config/data_paths.yaml`, `config/automl.yaml`).
- Processed datasets: `data/processed/panel.parquet`, `data/processed/features.parquet`.
- Logs: `logs/automl.log`, `reports/fold_results.csv`, `reports/missing_value_report.json`.
- Models: `models/best_model.pkl`, `models/scaler.pkl`.
- Submission: `submission/submission.csv`.
- README detailing command sequence for data prep, training, validation, inference.

---
Save this document as `AUTOML_SPEC.md` for development reference.
