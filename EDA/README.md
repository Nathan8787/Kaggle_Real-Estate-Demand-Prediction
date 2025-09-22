# 中國房地產需求競賽：EDA 腳本使用說明

本文件說明如何依照 `EDA_spec.md` 執行 `run_eda.py` 腳本。腳本會從 Kaggle 公開資料集中載入所有表格、產出完整的資料品質報表、搜尋熱度降維、土地與房市互動分析、結構斷點偵測，以及基準模型回測結果。所有輸出皆儲存在 `EDA/reports/` 目錄，依規格區分子資料夾。

## 環境需求

1. **Python 版本**：必須使用 `3.10.14`。若透過 `pyenv` 管理，可執行：
   ```bash
   pyenv install 3.10.14
   PYENV_VERSION=3.10.14 python --version
   ```
2. **套件安裝**：進入專案根目錄後安裝規格指定版本。
   ```bash
   PYENV_VERSION=3.10.14 pip install \
       pandas==2.1.4 numpy==1.26.4 scikit-learn==1.4.2 \
       statsmodels==0.14.2 lightgbm==4.3.0 ruptures==1.1.9 \
       matplotlib==3.8.4 seaborn==0.13.2 pyarrow==17.0.0 \
       openpyxl==3.1.5 typer==0.12.5 shap==0.44.1
   ```
3. **字型安裝**：所有圖表使用 `Source Han Sans`。若在 Linux，可下載 OTF/TTF 後放入 `~/.local/share/fonts/` 並執行 `fc-cache -fv` 更新快取。

## 執行步驟

1. 於專案根目錄執行腳本，預設會自動清除舊有報表並使用 headless 模式：
   ```bash
   PYENV_VERSION=3.10.14 python EDA/run_eda.py
   ```
   若需顯示互動視窗，可加入 `--no-headless`：
   ```bash
   PYENV_VERSION=3.10.14 python EDA/run_eda.py --no-headless
   ```
2. 腳本流程摘要：
   - 檢查所有原始檔案的大小與 SHA256。
   - 讀取並轉換各資料表欄位型態，輸出缺值／異常報表。
   - 生成目標趨勢、二手市場互動、土地供給分析、搜尋熱度降維、POI PCA 與 KMeans、宏觀指標、結構斷點等成果。
   - 依擴張窗口與留存月實施 LightGBM、Lag1、SARIMA 基準模型回測。
   - 鎖定環境需求到 `EDA/reports/environment/requirements_lock.txt`。

## 測試

規格要求提供評分函數單元測試，可於專案根目錄執行：
```bash
PYENV_VERSION=3.10.14 pytest
```

## 輸出目錄結構

執行完畢後 `EDA/reports/` 會包含下列子資料夾：
- `quality/`：缺值統計、異常 Parquet、月份缺口等資料品質報表。
- `figures/`：所有 Matplotlib/Seaborn 圖檔（字軸為英文）。
- `eda_tables/`：各類分析數據表的 CSV 與 Markdown 版本。
- `search/`：搜尋降維特徵及 PCA 解釋率。
- `baseline/`：基準模型得分、預測與圖表。
- `backtest/`：擴張窗口回測分數記錄。
- `structural/`：結構斷點偵測結果。
- `environment/`：`pip freeze` 鎖定清單。

所有圖表與數據僅作為後續特徵工程參考，提交時請勿附帶任何具體 EDA 結果。
