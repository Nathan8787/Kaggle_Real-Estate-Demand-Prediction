# Real Estate Demand Prediction ¡V AutoML Specification V2 (XGBoost GPU)

## 1. Objectives
1. Build a deterministic month¡Vsector panel covering January 2019 through July 2025 by merging all official training sources and generating forward-looking values for future months.
2. Construct a complete feature matrix using fixed imputation and extrapolation policies so that future months never collapse to all-zero or NaN feature vectors.
3. Train GPU-accelerated XGBoost models on `log1p` transformed targets, evaluate with the official competition score after reversing the transform, and use FLAML to search the hyperparameter space within a six-hour budget.
4. Use fixed time-series validation folds that include 2024 Q2 and Q3, and support walk-forward retraining plus residual-based prediction adjustments.

## 2. Data Inventory & Integration Rules
| File | Description | Keys | Pre-processing & Integration |
| --- | --- | --- | --- |
| `../train/new_house_transactions.csv` | Target and primary new-house metrics | `month`, `sector` | Read as UTF-8. Treat every missing month¡Ñsector row as true zero and add it during panel construction. Columns: `num_new_house_transactions`, `area_new_house_transactions`, `price_new_house_transactions`, `amount_new_house_transactions`, `area_per_unit_new_house_transactions`, `total_price_per_unit_new_house_transactions`, `num_new_house_available_for_sale`, `area_new_house_available_for_sale`, `period_new_house_sell_through`. |
| `../train/new_house_transactions_nearby_sectors.csv` | Neighbor-sector new-house metrics | `month`, `sector` | Same columns as above; append `_nearby` suffix before joining. |
| `../train/pre_owned_house_transactions.csv` | Local secondary-market metrics | `month`, `sector` | Columns: `num_pre_owned_house_transactions`, `area_pre_owned_house_transactions`, `price_pre_owned_house_transactions`, `amount_pre_owned_house_transactions`. |
| `../train/pre_owned_house_transactions_nearby_sectors.csv` | Neighbor secondary metrics | `month`, `sector` | Append `_nearby` suffix and join. |
| `../train/land_transactions.csv` | Local land metrics | `month`, `sector` | Columns: `num_land_transactions`, `construction_area`, `planned_building_area`, `transaction_amount`. |
| `../train/land_transactions_nearby_sectors.csv` | Neighbor land metrics | `month`, `sector` | Append `_nearby` suffix and join. |
| `../train/sector_POI.csv` | Static POI/density attributes | `sector` | Broadcast all fields to every month. |
| `../train/city_search_index.csv` | City-level monthly search volume | `month`, `keyword`, `source` | Normalize keywords with NFKC, create ASCII slug via `unidecode`, and store `keyword_original`. Aggregate to monthly sums per slug. For each slug: forward-fill, back-fill, then apply `rolling(window=3, min_periods=1).mean()`. Select the 30 slugs with highest variance (use all slugs if fewer). Pivot into `month` ¡Ñ `search_kw_<slug>` matrix, fill missing values with 0, and set `search_kw_<slug>_was_missing = 1` for months without original observations. |
| `../train/city_indexes.csv` | Annual city indicators | `city_indicator_data_year` | For each column, sort by year, compute CAGR using the latest three years (use all available years if fewer). When values are ?0, replace CAGR with average annual delta. Extrapolate 2023¡V2025 using the derived growth, then forward-fill as needed. Mark extrapolated months with `city_<col>_was_interpolated = 1`. |
| `../test.csv` | Kaggle test IDs | `id` | Contains months 2024-08 through 2025-07. Used only for submission ordering. |
| `../sample_submission.csv` | Kaggle submission template | `id`, `new_house_transaction_amount` | Preserve row order when writing predictions. |

Notes:
- Read all CSV files with UTF-8 encoding. Parse `month` as the first day of the month using `%Y-%b` or `dateutil` parsing.
- Preserve `sector` strings and extract `sector_id = int(re.search(r"(\\d+)", sector).group(1))`; raise `ValueError` if parsing fails.

## 3. Environment & Tooling
- Python 3.13.2 with environment variable `PYTHONUTF8=1`.
- Required pip packages: `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `scipy`, `Unidecode`, `PyYAML`, `joblib`, `xgboost>=2.0.0`, `flaml>=2.1.2`, `catboost`, `lightgbm`, `pytest`.
- GPU requirement: `nvidia-smi` must detect at least one CUDA-capable GPU. Install the appropriate CUDA toolkit and drivers before running training commands.
- All logs are stored as JSON files; no additional logging frameworks are used.

## 4. Panel Builder (`v2/src/panel_builder_v2.py`)
1. `load_raw_tables_v2(config_path)`
   - Read paths defined in `config/data_paths_v2.yaml` (see Section 14). Load each CSV with UTF-8 encoding, specifying string dtype for `month` and float dtype for numeric columns.
   - Convert column names to snake_case. Strip leading/trailing whitespace from string columns.
   - For `city_search_index`, add `keyword_original` and `keyword_slug` columns and normalize `source` with NFKC.
2. `_expand_city_indexes_with_growth(city_index_df, months)`
   - For each column, compute CAGR from the most recent three years (or all available years if fewer). When any value is ?0, compute the mean yearly delta instead.
   - Use the derived growth to extrapolate values for 2023, 2024, and 2025. Apply forward fill if extrapolated values extend beyond the available range. Set `city_<col>_was_interpolated = 1` for extrapolated months; otherwise 0.
3. `_build_search_features_v2(search_df, months)`
   - Aggregate monthly totals by `keyword_slug`.
   - For each slug, apply `ffill`, `bfill`, and `rolling(window=3, min_periods=1).mean()` to create a smoothed series.
   - Select the 30 slugs with highest variance (use all slugs if fewer). Pivot into `month` rows with columns `search_kw_<slug>`.
   - Fill missing values with 0 and create `search_kw_<slug>_was_missing` flag for months without original observations.
4. `_extend_monthly_tables(raw_tables, months)`
   - For each of the six monthly source tables (including `_nearby` variants) and each sector:
     1. Compute a 6-month rolling mean (requiring at least 4 valid observations) to forecast values for months beyond July 2024.
     2. If unavailable, compute a 3-month rolling mean (requiring at least 2 valid observations).
     3. If still unavailable, use the median across all sectors for the same month.
     4. If still unavailable, use the global median across all historical months.
     5. If no history exists, fill with 0.
     6. Mark generated values with `<column>_synthetic = 1`; otherwise 0.
5. `build_calendar(raw_tables)`
   - Create a monthly calendar from 2019-01-01 through 2025-07-01.
   - Extract `sector_id` from `sector`; raise an error on failure.
6. `merge_sources(calendar_df, raw_tables)`
   - Merge in order: new_house ¡÷ new_house_nearby ¡÷ pre_owned ¡÷ pre_owned_nearby ¡÷ land ¡÷ land_nearby ¡÷ sector_POI ¡÷ city indexes ¡÷ search features.
   - Preserve all `_synthetic`, `_was_missing`, and `city_*_was_interpolated` flags.
7. `attach_target_v2(panel_df)`
   - Set `target = amount_new_house_transactions`; missing entries become 0.
   - Set `target_filled = target` (with zero fill) for feature engineering.
   - Set `is_future = month > '2024-07-01'`.
8. `save_panel(panel_df, panel_path)`
   - Remove any existing parquet directory and save the panel partitioned by `year` to `v2/data/processed/panel_v2.parquet`.

## 5. Missing Value Policy (`v2/src/features_v2.py`)
- For every column to be imputed, add `column_was_missing = df[column].isna().astype(np.int8)` and retain the flag in the final dataset.
- `target_filled_was_missing` is created before filling `target_filled` with zero.
- Count/amount/area/density columns (`num_`, `area_`, `amount_`, `construction_`, `transaction_`, `planned_building_`, `number_of_`, `_dense`):
  1. Forward fill per sector.
  2. Fill remaining NaNs with sector rolling median (`window=12`, `min_periods=3`).
  3. Fill remaining NaNs with same-month city median.
  4. Fill remaining NaNs with global median (across all sectors and months).
  5. Fill remaining NaNs with 0.
- Price/ratio columns (names containing `price` but not starting with `city_`): sector forward fill ¡÷ `_fill_with_sector_history` (sorted-sector historical median) ¡÷ global median ¡÷ final fallback 0.
- Period columns (`period_*`): sector forward fill ¡÷ sector median ¡÷ global median.
- City columns (`city_*`): sector forward fill ¡÷ month-level forward fill ¡÷ global median.
- Search columns (`search_kw_*`): after smoothing, fill remaining NaNs with 0; `search_kw_*_was_missing` indicates the original absence of data.
- Write imputation statistics (strategy and fill counts) to `v2/reports/missing_value_report.json`.

## 6. Feature Engineering (`v2/src/features_v2.py`)
1. Convert `month` to `datetime64[ns]`, sort by `sector_id` and `month`.
2. Add time features: `year`, `month_num`, `quarter`, `month_index = 12*(year-2019) + (month_num-1)`, `is_year_start`, `is_year_end`, `days_in_month`.
3. For each column in `{target_filled} ¡å columns starting with num_, area_, amount_, price_, total_price_`:
   - Create lag features for 1, 3, 6, and 12 months.
   - Create rolling mean, std, min, max, and count for windows 3, 6, and 12 (`min_periods=1`). Keep `rolling_count_<window>` to record the number of valid observations.
4. Growth rates: `growth_1 = (value - lag_1) / (abs(lag_1) + 1e-6)`; `growth_3` analogous. When lag is NaN, set growth to 0 and `growth_*_was_missing = 1`.
5. City-weighted share: compute weights per row as `resident_population` (if ?0 or NaN, replace with `population_scale`; if still NaN, use 1.0). For each month and metric, compute weighted mean = sum(value*weight)/sum(weight) (if denominator is 0, use the simple average). Add `metric_city_weighted_mean` and `metric_share = value / (metric_city_weighted_mean + 1e-6)`.
6. Interaction terms: for each base column with `_nearby`, add `base_vs_nearby_diff = base - base_nearby`.
7. Search-derived features: for every `search_kw_*` column, compute sector-level `pct_change` and 3-month rolling z-score (fill NaN with 0). Compute `density_adj = dense_value * mean(zscores)` for every `_dense` column.
8. Apply `np.log1p` to all count/density-style columns (including `target_filled` and POI densities) and append `_log1p` suffix.
9. POI PCA: if there are at least two log density columns with non-zero variance, run PCA to retain components covering ?95% cumulative variance (`poi_pca_1`, `poi_pca_2`, ...). Otherwise skip PCA.
10. Save the resulting dataframe partitioned by year to `v2/data/processed/features_v2.parquet` and record metadata (lag, rolling, share, interaction, search, log, PCA columns) in `v2/reports/feature_metadata.json`.

## 7. Target Transformation & Scaling
- Use `y_log = np.log1p(y_raw)` for every training and validation target.
- During evaluation and inference, restore predictions via `y_pred = np.clip(np.expm1(y_pred_log), 0.0, None)` and restore true labels similarly before scoring.
- For each fold, fit a `StandardScaler` on the training subset of features, save mean and scale to `v2/reports/scaler_fold_<k>.json`.
- Fit a global `StandardScaler` on all training data (`month <= 2024-07-01`) and save to `v2/models/scaler.pkl`.
- `v2/reports/automl_summary.json` must include: `best_estimator`, `best_fold`, `target_transform: "log1p_expm1"`, `best_config`, and fold-level metrics.

## 8. Dataset Splitting (`v2/src/splits_v2.py`)
Define:
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
`generate_time_series_folds_v2(features_df)` must sort by `month` and `sector_id`, capture original row indices, and return seven `(train_idx, valid_idx)` pairs. Each `train_idx` contains all months up to the validation start minus one month.

## 9. Custom Metric (`v2/src/metrics_v2.py`)
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
All validation and reporting must rely on this metric implementation.

## 10. Training (`v2/src/automl_runner_v2.py`)
1. Load `config/automl_v2.yaml`:
```
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
      _value: [400, 4000]
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
      _value: [0.0001, 50]
    reg_lambda:
      _type: loguniform
      _value: [0.0001, 50]
```
2. Filter features to `month <= '2024-07-01'` and compute `y_full_log = np.log1p(y_raw)`.
3. For each fold:
   - Build `fold_X`, `fold_y`, and `fold_y_log`.
   - Fit `StandardScaler` on training positions; store to `v2/reports/scaler_fold_<k>.json`.
   - Instantiate `SingleFoldSplitter(train_positions, valid_positions)`.
   - Configure FLAML:
     ```python
     automl_settings = {
         'time_budget': config['time_budget'],
         'metric': competition_metric,
         'task': 'regression',
         'estimator_list': config['estimator_list'],
         'log_file_name': str(Path('logs') / 'automl.log'),
         'seed': config['seed'],
         'n_jobs': config['n_jobs'],
         'eval_method': 'cv',
         'split_type': splitter,
         'fit_kwargs_by_estimator': config['fit_kwargs_by_estimator'],
         'custom_hp': config['search_space'],
         'verbose': 1,
     }
     automl = AutoML()
     automl.fit(X_train=fold_X_scaled, y_train=fold_y_log, **automl_settings)
     ```
   - Predict validation rows, restore via `np.expm1`, and compute `competition_score`.
   - Append results to `v2/reports/fold_results.csv` with columns `fold,id,month,sector_id,target,prediction,fold_score`.
4. Save `automl.best_config` (filtering out non-parameter entries) to `v2/models/best_config.json`.
5. Fit final model: `final_params = {**best_config, **config['fit_kwargs_by_estimator']['xgboost']}`; train on scaled full data and save as `v2/models/best_model.pkl`. Save scaler to `v2/models/scaler.pkl`.
6. Write summary to `v2/reports/automl_summary.json`.

## 11. Prediction (`v2/src/predict_v2.py`)
1. Load `v2/data/processed/panel_v2.parquet`, `v2/data/processed/features_v2.parquet`, `v2/models/scaler.pkl`, `v2/models/best_model.pkl`, and optional residual correction files.
2. Determine `forecast_months` from `panel_df['is_future']` sorted ascending.
3. For each forecast month:
   - Recompute features using `build_feature_matrix_v2` on the current working panel.
   - Scale features and obtain log predictions, then convert with `np.expm1` and clip at zero.
   - If residual correction is requested, load `models/residual_model.pkl`, `models/residual_scaler.pkl`, and `reports/residual_features.json`, transform the specified features, add residual predictions, and clip again.
   - Update `target_filled` and `target` for the forecast month within the working panel.
   - Collect predictions with their `id` values.
4. Merge predictions with `../sample_submission.csv` based on `id`. If any prediction is missing, raise an error.
5. Save `submission/submission.csv` and summary statistics to `v2/reports/prediction_summary.json` (fields: `prediction_count`, `prediction_mean`, `prediction_std`, `prediction_min`, `prediction_max`).

## 12. Walk-forward Retraining (`v2/scripts/walk_forward_train.py`)
- Disabled by default. When executed:
  1. Load `v2/models/best_config.json` for fixed hyperparameters.
  2. For each month in `forecast_months`:
     - Filter features to months strictly earlier than the forecast month.
     - Fit a fresh scaler and XGBoost model on `np.log1p(target)` using the stored best config and GPU parameters.
     - Save `models/scaler_<YYYY-MM>.pkl` and `models/best_model_<YYYY-MM>.pkl`.
     - Generate predictions for that month with residual adjustment if configured, update the panel, and log summary results.
  3. Write overall stats to `v2/reports/walk_forward_summary.json`.

## 13. Residual Correction (`v2/scripts/residual_correction.py`)
1. Load `v2/reports/fold_results.csv` and compute residuals (`target - prediction`).
2. Select correction features (fixed list defined in script) and fit `StandardScaler`; save to `v2/models/residual_scaler.pkl`.
3. Train `sklearn.linear_model.Ridge(alpha=1.0)` on the scaled features and residuals; save to `v2/models/residual_model.pkl`.
4. Store feature names in `v2/reports/residual_features.json`.
5. During prediction, if residual correction flags are provided, apply the scaler and model before clipping predictions.

## 14. Command Line Interface (`v2/scripts/run_pipeline_v2.py`)
Commands must be executed from the `v2/` directory.

1. Build panel:
   ```bash
   python scripts/run_pipeline_v2.py build_panel \
       --data-config config/data_paths_v2.yaml \
       --panel-path data/processed/panel_v2.parquet
   ```
2. Build features:
   ```bash
   python scripts/run_pipeline_v2.py build_features \
       --panel-path data/processed/panel_v2.parquet \
       --features-path data/processed/features_v2.parquet \
       --reports-dir reports
   ```
3. Execute AutoML search:
   ```bash
   python scripts/run_pipeline_v2.py run_xgb \
       --features-path data/processed/features_v2.parquet \
       --config config/automl_v2.yaml
   ```
4. Generate predictions:
   ```bash
   python scripts/run_pipeline_v2.py predict \
       --panel-path data/processed/panel_v2.parquet \
       --features-path data/processed/features_v2.parquet \
       --model-path models/best_model.pkl \
       --scaler-path models/scaler.pkl \
       --sample-submission ../sample_submission.csv \
       --output-path submission/submission.csv \
       [--residual-model models/residual_model.pkl \
        --residual-scaler models/residual_scaler.pkl \
        --residual-features reports/residual_features.json]
   ```
5. Walk-forward training (optional):
   ```bash
   python scripts/run_pipeline_v2.py walk_forward_train \
       --panel-path data/processed/panel_v2.parquet \
       --features-path data/processed/features_v2.parquet \
       --config config/automl_v2.yaml
   ```

## 15. Artifacts & Directory Structure
```
v2/
¢u¢w¢w artifacts/
¢u¢w¢w config/
¢x   ¢u¢w¢w automl_v2.yaml
¢x   ¢|¢w¢w data_paths_v2.yaml
¢u¢w¢w data/
¢x   ¢|¢w¢w processed/
¢x       ¢u¢w¢w panel_v2.parquet
¢x       ¢|¢w¢w features_v2.parquet
¢u¢w¢w docs/
¢x   ¢|¢w¢w AUTOML_SPEC.md
¢u¢w¢w logs/
¢x   ¢|¢w¢w automl.log
¢u¢w¢w models/
¢x   ¢u¢w¢w scaler.pkl
¢x   ¢u¢w¢w best_model.pkl
¢x   ¢u¢w¢w best_config.json
¢x   ¢u¢w¢w residual_model.pkl (if residual correction is trained)
¢x   ¢u¢w¢w residual_scaler.pkl (if residual correction is trained)
¢x   ¢|¢w¢w best_model_<YYYY-MM>.pkl (if walk-forward training is executed)
¢u¢w¢w reports/
¢x   ¢u¢w¢w automl_summary.json
¢x   ¢u¢w¢w fold_results.csv
¢x   ¢u¢w¢w missing_value_report.json
¢x   ¢u¢w¢w feature_metadata.json
¢x   ¢u¢w¢w prediction_summary.json
¢x   ¢|¢w¢w residual_features.json (if residual correction is trained)
¢u¢w¢w scripts/
¢x   ¢u¢w¢w run_pipeline_v2.py
¢x   ¢u¢w¢w walk_forward_train.py
¢x   ¢|¢w¢w residual_correction.py
¢u¢w¢w src/
¢x   ¢|¢w¢w __init__.py (plus implementation modules)
¢u¢w¢w submission/
¢x   ¢|¢w¢w submission.csv
¢|¢w¢w ../train/, ../test.csv, ../sample_submission.csv (shared raw inputs)
```

## 16. Testing & Verification
1. Syntax check: `python -m py_compile src/**/*.py scripts/**/*.py`.
2. Unit tests (must be implemented):
   - `pytest tests/test_panel_builder_v2.py::test_extend_monthly_tables_fills_future_months`
   - `pytest tests/test_features_v2.py::test_missing_value_policy`
   - `pytest tests/test_metrics_v2.py::test_competition_metric_log_handling`
3. Smoke test:
   ```bash
   python scripts/run_pipeline_v2.py build_panel --data-config config/data_paths_v2.yaml
   python scripts/run_pipeline_v2.py build_features --panel-path data/processed/panel_v2.parquet --features-path data/processed/features_v2.parquet --reports-dir reports
   python scripts/run_pipeline_v2.py run_xgb --features-path data/processed/features_v2.parquet --config config/automl_v2.yaml
   python scripts/run_pipeline_v2.py predict --panel-path data/processed/panel_v2.parquet --features-path data/processed/features_v2.parquet --model-path models/best_model.pkl --scaler-path models/scaler.pkl --sample-submission ../sample_submission.csv --output-path submission/submission.csv
   ```
4. If residual correction is trained, run `python scripts/residual_correction.py` before the final predict command and include residual arguments in the prediction step.

> Reminder: execute all v2 commands inside the `v2/` directory. Raw inputs (`train/`, `test.csv`, `sample_submission.csv`) live one level up; reference them in configs using `../train`, `../test.csv`, and `../sample_submission.csv`.
