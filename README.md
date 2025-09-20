# 房地產需求預測 AutoML v3 導覽

本專案提供依照 `AUTOML_SPEC_v3` 所設計的資料處理、特徵工程、AutoML 訓練與推論流程。以下說明如何使用原始資料集建置完整管線、執行測試，並產出提交檔案。所有內容均以繁體中文撰寫，方便快速上手。

## 1. 專案概覽

- **資料來源**：`train/` 目錄包含歷史交易、POI、搜尋指數等表格；`test.csv` 為測試資料；`sample_submission.csv` 為 Kaggle 格式。
- **目標**：預測 2024-08 至 2024-12 各行政區新建案成交金額，並維持規格中要求的投影、特徵數量與審計報表。
- **主要模組**：
  - `v3/src/panel_builder_v3.py`：彙整原始表格並產出 panel。
  - `v3/src/projection_v3.py`：前瞻特徵投影與飄移報告。
  - `v3/src/features_v3.py`：特徵工程與欄位控管。
  - `v3/src/automl_runner_v3.py`：交叉驗證、留存月與全量模型訓練。
  - `v3/src/predict_v3.py`：逐月迭代推論與提交檔案生成。
  - `v3/scripts/run_pipeline_v3.py`：統一執行所有步驟的指令列工具。

## 2. 前置準備

1. 於專案根目錄確保資料夾結構如下：
   ```text
   Kaggle_Real-Estate-Demand-Prediction/
   ├─ train/
   ├─ test.csv
   ├─ sample_submission.csv
   └─ v3/
   ```
2. 建議使用 Python 3.10 以上版本並安裝對應套件（pandas、numpy、scikit-learn、xgboost、flaml、matplotlib 等）。
3. 如需調整資料或輸出路徑，可修改 `v3/config/data_paths_v3.yaml` 與 `v3/config/automl_v3.yaml`。

## 3. Pipeline 執行流程

請務必按照規格先建立 panel，再產生特徵、訓練模型，最後執行推論。所有測試皆需使用真實資料，禁止生成任何合成資料。

### 3.1 建立面板資料

```bash
python v3/scripts/run_pipeline_v3.py build_panel \
  --data-config v3/config/data_paths_v3.yaml \
  --panel-path data/processed/panel_v3.parquet \
  --reports-dir reports_v3
```

- 讀取 `train/` 目錄的各類交易、POI、搜尋與指標表格。
- 生成 2019-01 至 2024-12 的 `month × sector` 全表，並附加目標欄位與 `id`。
- Panel 以年份分割儲存為 Parquet，同時輸出 `reports_v3/panel_build_summary.json`。

### 3.2 建立特徵矩陣

```bash
python v3/scripts/run_pipeline_v3.py build_features \
  --panel-path data/processed/panel_v3.parquet \
  --features-path data/processed/features_v3.parquet \
  --reports-dir reports_v3 \
  --forecast-start 2024-06-01
```

- 套用前瞻特徵投影、缺值補齊、lag/rolling/growth/share/log1p 等衍生欄位。
- 自動挑選 12 個搜尋關鍵字並執行 POI PCA，確保最終特徵數量落在 537–567。
- 輸出 `reports_v3/feature_inventory_v3.json`、`search_keywords_v3.json`、`poi_pca_summary.json` 與缺值報告。

### 3.3 AutoML 訓練

```bash
python v3/scripts/run_pipeline_v3.py run_xgb \
  --features-path data/processed/features_v3.parquet \
  --config v3/config/automl_v3.yaml \
  --reports-dir reports_v3 \
  --models-dir models_v3 \
  --logs-dir logs_v3
```

- 依 `AUTOML_SPEC_v3` 進行四折時間序交叉驗證，並記錄每次 trial 的組態。
- 以最佳設定分別訓練留存月與全量模型，輸出對應 scaler、模型檔及 `holdout_metrics_v3.json`。
- 同步產生 `reports_v3/fold_metrics_v3.csv`、`logs_v3/automl_trials.csv`、`reports_v3/automl_summary.json`、`reports_v3/feature_importance_gain.csv` 與 `top_features.txt`。

### 3.4 留存月驗證（可選）

```bash
python v3/scripts/run_pipeline_v3.py evaluate_holdout \
  --features-path data/processed/features_v3.parquet \
  --model-path models_v3/best_model_holdout.json \
  --scaler-path models_v3/scaler_holdout.pkl \
  --reports-dir reports_v3
```

- 載入既有模型與 scaler，重新計算 2024-07 的所有評分指標，並覆寫 `reports_v3/holdout_metrics_v3.json`。
- 若需檢視原始飄移報告，可查看 `reports_v3/projection_drift_202407.json`。

### 3.5 逐月推論與提交檔

```bash
python v3/scripts/run_pipeline_v3.py predict \
  --panel-path data/processed/panel_v3.parquet \
  --features-path data/processed/features_v3.parquet \
  --model-path models_v3/best_model_full.json \
  --scaler-path models_v3/scaler_full.pkl \
  --sample-submission ../sample_submission.csv \
  --output-path submission_v3/submission.csv \
  --reports-dir reports_v3
```

- 依序投影 2024-08 至 2024-12，每月迭代更新 `target_filled` 後生成特徵再推論。
- 自動整併預測與 `sample_submission.csv`，輸出 Kaggle 上傳檔案。
- 產生 `reports_v3/test_window_summary.json` 與 `reports_v3/plots/` 下的診斷圖。
- 若有殘差修正模型，可加上 `--residual-model`、`--residual-scaler`、`--residual-features` 參數。

## 4. 設定檔重點

- `v3/config/data_paths_v3.yaml`：定義資料來源、處理後輸出與報表位置。若修改目錄，記得保持 `reports_dir` 與 pipeline 參數一致。
- `v3/config/automl_v3.yaml`：記錄 FLAML 設定、搜尋空間與 XGBoost 參數。可視硬體資源調整 `time_budget_per_fold` 或 `gpu_id`。

## 5. 測試與驗證

完成程式修改後，請至少執行以下檢查：

```bash
# 靜態語法檢查
python -m py_compile v3/src/*.py v3/src/**/*.py v3/scripts/*.py

# 單元測試
pytest v3/tests/test_projection_v3.py::test_projection_respects_cutoff
pytest v3/tests/test_features_v3.py::test_feature_counts_and_categories
pytest v3/tests/test_splits_v3.py::test_fold_alignment
pytest v3/tests/test_metrics_v3.py::test_competition_metric_cap
```

若要進行煙霧測試，可依序執行第 3 節列出的五個指令，確認 panel → features → AutoML → holdout → predict 的流程皆能順利運作。

## 6. 注意事項

- 僅能使用官方提供的歷史資料與測試集，禁止生成或注入任何合成資料。
- 所有報表與模型會被寫入 `reports_v3/`、`models_v3/`、`logs_v3/` 與 `submission_v3/`。若需要重新執行，請預先備份或清除舊檔案。
- 投影流程嚴格限制僅能使用 2024-07（含）以前的真實觀測；請勿變更此界線，以免造成資料洩漏。
- `build_feature_matrix_v3` 會在特徵數量不符 537–567 時直接拋錯，若新增/刪除特徵請同步更新相關邏輯及報告。

若在規格或流程上遇到問題，可重新檢視 `v3/docs/AUTOML_SPEC_v3.md`，或記錄異常情況再行討論修訂。

