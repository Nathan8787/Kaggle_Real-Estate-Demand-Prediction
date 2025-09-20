# Kaggle Real Estate Demand Prediction – v2 專案說明

本專案以 `v2/` 目錄為核心，實作依照 `v2/docs/AUTOML_SPEC.md` 制定的流程。使用者可透過下列步驟完成資料處理、特徵工程、模型訓練與推論：

1. **建立 Panel** – 於 `v2/` 目錄執行：
   ```bash
   python scripts/run_pipeline_v2.py build_panel \
       --data-config config/data_paths_v2.yaml \
       --panel-path data/processed/panel_v2.parquet
   ```
2. **建立特徵**
   ```bash
   python scripts/run_pipeline_v2.py build_features \
       --panel-path data/processed/panel_v2.parquet \
       --features-path data/processed/features_v2.parquet \
       --reports-dir reports
   ```
3. **啟動 GPU XGBoost 自動化搜尋**
   ```bash
   python scripts/run_pipeline_v2.py run_xgb \
       --features-path data/processed/features_v2.parquet \
       --config config/automl_v2.yaml
   ```
4. **產出提交檔**
   ```bash
   python scripts/run_pipeline_v2.py predict \
       --panel-path data/processed/panel_v2.parquet \
       --features-path data/processed/features_v2.parquet \
       --model-path models/best_model.pkl \
       --scaler-path models/scaler.pkl \
       --sample-submission ../sample_submission.csv \
       --output-path submission/submission.csv
   ```

需要進一步調整模型或進行 walk-forward retraining、殘差修正等進階操作時，可使用 `scripts/walk_forward_train.py`、`scripts/residual_correction.py`，所有細節請參考 `v2/docs/AUTOML_SPEC.md`。

> **注意**：原始資料 (`train/`、`test.csv`、`sample_submission.csv`) 置於專案根目錄；舊版流程與成果已完整移至 `v1/` 目錄。
