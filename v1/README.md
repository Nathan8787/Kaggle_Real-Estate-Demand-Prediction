# 中國房地產需求預測 Pipeline 使用指南

本文說明依照 `AUTOML_SPEC.md` 規格所實作的資料處理與 AutoML 流程，協助你快速重現整個模型訓練與提交產出。

## 1. 環境需求
- Python 3.13.2（已於執行環境確認）
- 建議先建立虛擬環境，再安裝所需套件：

```powershell
pip install -r requirements.txt
```

主要套件：pandas、numpy、pyarrow、scikit-learn、FLAML、LightGBM、XGBoost、CatBoost、Unidecode、PyYAML、joblib 等。

## 2. 專案結構
```
project/
├─ config/
│  ├─ data_paths.yaml          # 原始資料與輸出路徑設定
│  └─ automl.yaml              # 正式 AutoML 參數（7200 秒）
├─ data/
│  └─ processed/               # 合併後 panel 與特徵檔（Parquet，依年份分區）
├─ logs/
│  └─ automl.log               # AutoML 執行日誌（JSON line）
├─ models/
│  ├─ best_model.pkl           # 重新訓練後的最佳模型
│  └─ scaler.pkl               # 對應的 StandardScaler
├─ reports/
│  ├─ missing_value_report.json     # 缺值填補紀錄
│  ├─ fold_results.csv              # 各折驗證預測與分數
│  ├─ automl_summary.json           # AutoML 摘要（含每折最佳模型）
│  ├─ prediction_summary.json       # 產生 submission 後的統計
│  └─ scaler_fold_*.json            # 每折 scaler 平均值與標準差
├─ src/
│  ├─ data_loading.py          # 讀檔與正規化
│  ├─ panel_builder.py         # 建立 month×sector panel 並合併資料
│  ├─ features.py              # 特徵工程與缺值處理
│  ├─ splits.py                # 時序交叉驗證切分
│  ├─ metrics.py               # 競賽自訂評分函式
│  ├─ automl_runner.py         # 嵌入 FLAML 的 AutoML 流程
│  └─ predict.py               # 產出 submission
├─ submission/
│  └─ submission.csv           # 目前最新一次推論結果
├─ run_pipeline.py             # CLI 入口，整合整個流程
├─ requirements.txt            # 套件需求
└─ README.md                   # 本說明
```

## 3. 快速開始（正式流程）
建議依序執行以下四個指令。以下以 PowerShell 為例，若使用其他 shell 請自行調整。

1. **整併原始資料，建立 panel**
   ```powershell
   python run_pipeline.py build_panel \
       --data-config config/data_paths.yaml \
       --panel-path data/processed/panel.parquet
   ```

2. **產生特徵矩陣**
   ```powershell
   python run_pipeline.py build_features \
       --panel-path data/processed/panel.parquet \
       --features-path data/processed/features.parquet \
       --reports-dir reports
   ```

3. **執行 AutoML（正式訓練）**
   ```powershell
   python run_pipeline.py run_automl \
       --features-path data/processed/features.parquet \
       --automl-config config/automl.yaml
   ```
   - `config/automl.yaml` 設定 7,200 秒搜尋預算，請預留足夠時間與硬體資源。
   - 若僅需驗證流程是否順利，先用 `config/automl_fast.yaml`（10 秒）跑 smoke test，確認無誤後再切換正式設定。

4. **產出 Kaggle submission**
   ```powershell
   python run_pipeline.py predict \
       --model-path models/best_model.pkl \
       --scaler-path models/scaler.pkl \
       --features-path data/processed/features.parquet \
       --sample-submission sample_submission.csv \
       --output-path submission/submission.csv
   ```

## 4. 主要輸出說明
- `data/processed/panel.parquet`：整併後的 month×sector panel，含 `target` 與自訂 `id`（格式：`2024 Aug_sector 1`）。
- `data/processed/features.parquet`：特徵工程結果，包含 lag/rolling、人口加權 share、搜尋變化、POI PCA 等欄位。
- `reports/missing_value_report.json`：各欄位的缺值補值策略與填補數量。
- `reports/fold_results.csv`：五個時間序列折的驗證預測與評分。
- `reports/automl_summary.json`：每折最佳模型與分數、最終最佳 estimator、被排除的 estimator。
- `logs/automl.log`：AutoML 執行摘要（一行一筆 JSON）。
- `models/best_model.pkl`、`models/scaler.pkl`：以全部可訓練月份（截至 2024-07）重訓後的模型與 scaler。
- `reports/prediction_summary.json`：submission 產出後的統計摘要（筆數、平均、標準差、最小值、最大值）。
- `submission/submission.csv`：符合 Kaggle `sample_submission.csv` 順序的最終預測檔。

## 5. 注意事項與規格差異
- **自訂評分函式整合**：FLAML 2.3.6 尚未提供 `add_metric` API，因此改以 `metric=<callable>` 傳入，使搜尋目標最小化 `1 - competition_score`，同時在 log 中紀錄分數。
- **Estimator 清單**：規格中列出的 `lrl1`、`lrl2` 為 Lasso/Ridge baseline，但 FLAML 僅支援分類任務版本，回歸任務會報錯。本實作自動排除並於 `automl_summary.json` 標註。
- **輸出 ID 格式**：對齊 Kaggle 提供的樣本檔，不對 sector 號碼補零，避免 submission 與官方格式不一致。
- **特徵工程效能警示**：`features.py` 插入大量欄位（lag/rolling/差值等）時，pandas 會顯示 DataFrame fragment 警示，僅為效能提醒，可視需求再優化。
- **快速測試 vs. 正式訓練**：目前 `submission/submission.csv` 來自 10 秒預算的 smoke test，用於驗證流程。若要提交競賽，請務必以 `config/automl.yaml` 重新訓練並再執行 `predict` 產出正式結果。

## 6. 常見問題
1. **預測結果缺少某些 `id`**：確認已執行正式的 `run_automl` 並重新跑 `predict`，系統會檢查是否有缺漏。
2. **FLAML 提示 estimator 不支援**：若手動修改 `estimator_list`，請確保僅保留 FLAML 支援的回歸器。
3. **調整資料路徑**：修改 `config/data_paths.yaml` 中的 `raw_dir`、`train_dir` 等設定即可。

## 7. 延伸建議
- 調整 `config/automl.yaml` 的 `time_budget`、`estimator_list` 或自訂搜尋空間，追求更高分數。
- 在 `src/features.py` 中加入更多領域知識特徵（例如節慶、宏觀政策事件）。
- 拓展 `run_pipeline.py` 的子命令或引入設定檔管理工具，以利團隊協作與多組實驗。

如需調整規格，請先更新 `AUTOML_SPEC.md`，再依上述流程重新執行。祝模型訓練順利！
