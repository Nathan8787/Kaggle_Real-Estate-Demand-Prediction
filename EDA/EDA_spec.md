# 中國房地產需求競賽：初始 EDA 規劃書

> 目的：針對 Kaggle《Real Estate Demand Prediction》競賽之訓練資料，建立一套可重複的探索性資料分析（EDA）藍圖，協助後續特徵工程與機器學習建模。依照使用者要求，本規劃僅產出說明文件，不產生任何額外檔案或程式。

## 1. 競賽背景與分析目標
- 競賽網址：<https://www.kaggle.com/competitions/china-real-estate-demand-prediction>。官方提供 2019-01 至 2024-07 的多表格歷史資料，預測 2024-08 至 2024-12 各 `sector` 的 **`new_house_transaction_amount`**。
- 評分方式：根據競賽頁面公布的規則（需再次確認最新公告），以預測值與真值的誤差計算官方評分函數。初次 EDA 時需同步紀錄該評分函數與任何裁切（如 APE>1 設零）的細節，以便後續模型一致性。
- 資料型態：屬於「長表」面板資料（`month × sector`），並搭配年度城市指標、搜尋指數、POI 靜態屬性等多來源資料。EDA 必須釐清各表的粒度與可連結鍵值，避免合併時產生重複列或資料遺失。
- 核心分析問題：
  1. 新建案成交金額的時間趨勢、季節性與疫情後復甦狀態。
  2. 供給端（土地拍賣、新屋庫存）、需求端（搜尋量、二手市場）、區位屬性（POI、城市宏觀指標）與目標之關聯性。
  3. 2024 年最新月份（4–7 月）是否出現異常值或結構性斷點，避免訓練資料外推失真。

## 2. 原始資料盤點與鍵值關係
| 檔案 | 粒度/鍵值 | 時間範圍 | 主要欄位 | 備註 |
| --- | --- | --- | --- | --- |
| `train/new_house_transactions.csv` | `month`, `sector` | 2019-01 ~ 2024-07 | `num_*`, `area_*`, `price_*`, `amount_*`, `period_new_house_sell_through` | 目標欄位 `amount_new_house_transactions` 與其他成交指標皆在此表 |
| `train/new_house_transactions_nearby_sectors.csv` | `month`, `sector` | 同上 | 對應鄰近區段的 9 個成交欄位 | 需確認鄰近區段是何種加權或簡單平均 |
| `train/pre_owned_house_transactions.csv` | `month`, `sector` | 同上 | 二手房成交量、金額、單價 | 只有 4 個度量，注意 0 是否代表停滯 |
| `train/pre_owned_house_transactions_nearby_sectors.csv` | `month`, `sector` | 同上 | 鄰近二手房 4 指標 | 與本區比較差距、比率 |
| `train/land_transactions.csv` | `month`, `sector` | 同上 | 土地拍賣量、建築面積、成交額 | 常見大量 0，需區分缺值與無交易 |
| `train/land_transactions_nearby_sectors.csv` | `month`, `sector` | 同上 | 鄰近土地交易指標 | 供給面指標的重要補充 |
| `train/city_search_index.csv` | `month`, `keyword`, `source` | 2019-01 ~ 2024-07 | `search_volume` | 包含中文字詞，來源含 PC / 移動端；需要轉為寬表以利分析 |
| `train/city_indexes.csv` | `city_indicator_data_year` | 2017 ~ 2020 | 人口、就業、GDP、財政、教育、醫療等 60+ 指標 | 年度資料需映射至月份（例如 forward fill） |
| `train/sector_POI.csv` | `sector` | 靜態 | 142 個 POI 與人口類欄位 | 包含房租、均價、設施密度等；欄位名稱長且含下劃線 |
| `test.csv` | `id` (`YYYY Mon_sector X`) | 2024-08 ~ 2024-12 | 需預測 `new_house_transaction_amount` | 用於驗證 `month × sector` 是否與訓練集對齊 |
| `sample_submission.csv` | 同上 | 2024-08 ~ 2024-12 | 比賽提交格式 | 做排序與欄位參考 |

> 注意：所有 CSV 皆帶有 UTF-8 BOM，讀取時需使用 `encoding="utf-8-sig"`；欄名包含長字串與中文字，未來腳本需統一轉換大小寫與空白。

## 3. 讀取與資料清理規範
1. **欄位型別與標準化**
   - `month` 解析為 pandas Period 或 datetime（`MS`），另外建立 `year`, `month_num`, `quarter`, `month_index`。
   - `sector` 去除字串前綴後保留數值與原始標籤，必要時建立 `sector_id`（int）。
   - 貨幣金額、面積、交易量皆轉為 `float64`，保留 2–4 位小數；確認是否需要 `int64`（例如交易戶數）。
   - 將中文欄位（如搜尋關鍵字、來源）做 Unicode NFKC 正規化，以免同義字不同碼。
2. **資料品質檢查**
   - 缺失值檢視：統計每欄缺值數、比例，並辨識是否以 `0`、空字串代表缺失。
   - 重複鍵檢查：確認 `month × sector`（或 `month × sector × keyword × source`）唯一。若有重複需記錄處理策略。
   - 合法值篩檢：
     - 金額、面積、交易量不應為負；如出現負值需列入異常清單。
     - `period_new_house_sell_through` 需確認單位（以月或週），並偵測極端值。
     - `city_indexes` 中的空白欄位（例如 2017 年的缺值）需判斷是否為真正缺漏。
   - 時間覆蓋：檢查各表最早與最晚月份，確保 2019-01~2024-07 都有資料；若特定月份缺列需記錄。
3. **資料整併準備**
   - 建立完整的 `month × sector` 主表（Calendar），作為各表合併的基準。
   - 年度城市指標映射：以年為粒度，透過 `month.dt.year` join 後 forward fill；同時計算 `was_interpolated` 旗標。
   - 搜尋指數轉換：從長表（keyword/source） pivot 成寬表，保留原始值與合併欄位（如總搜尋量、PC/移動占比）。
   - 確認 `test.csv` 中所有 `sector` 均存在於訓練資料；若有新 sector，需在 EDA 報告中高亮。

## 4. 探索分析模組設計
### 4.1 基礎輪廓與資料摘要
- 每個表計算列數、欄位數、缺失比例、零值比例、基本統計量（平均、標準差、分位數）。
- 產生欄位資料型別對照表與欄位說明（如單位、計算方式）；若官方未提供，需在報告中標註「待確認」。
- 建立「資料時序覆蓋圖」，展示各表在每個月的資料完整度。

### 4.2 新建案成交主體分析（Target 導向）
- **時間序列**：總體與各 sector 月度 `amount_new_house_transactions`、`num_new_house_transactions`、`price_new_house_transactions`。
- **季節性與成長率**：計算 MoM、YoY、rolling mean（3/6/12 月）與標準差。
- **極端值檢測**：利用箱型圖或 Z-score 找出價格/金額異常尖峰；逐月列出 top/bottom 5 sector。
- **當月最新觀測**：聚焦 2024-04~2024-07，比較其與 2023 同期的差異。

### 4.3 二手市場互動
- 新屋 vs 二手屋成交量、平均單價的散佈圖與相關係數（同月與 lag 1–3 月）。
- 建立「新屋成交量 / 二手成交量」比值，評估市場轉換行為。
- 比較鄰近區段與本區段的差距（差值、比率），檢查是否存在空間溢出效應。

### 4.4 土地供給與庫存
- 土地成交金額/面積與新屋銷售的時間對齊分析（cross-correlation、lead/lag 0–6 月）。
- 土地交易為零的月份是否對應新屋供給萎縮，並標記長期為零的 sector。
- 研究 `period_new_house_sell_through` 與待售庫存 (`num_new_house_available_for_sale`, `area_new_house_available_for_sale`) 的變化，估算吸收率與庫存週期。

### 4.5 搜尋熱度與需求先行指標
- 將各關鍵字在 PC/移動端的搜尋量求和、占比，分析與成交金額的相關性。
- 對主要關鍵字（如「買房」、「二手房市場」等）建立滯後相關（lag 1–3 月），檢驗是否具領先性。
- 分析 2023-2024 年關鍵字排行變化，找出新興或衰退的需求訊號。

### 4.6 城市宏觀指標
- 將年度城市指標聚合成月度面板後，檢視與成交金額的趨勢（例如人口、GDP、財政收入）。
- 計算長期趨勢（2017–2020）與 2024 年銷售表現的相關性，評估哪些指標值得納入長期特徵。
- 若年限不足以涵蓋 2021 之後，需在報告中紀錄缺口並提出插值或外部資料需求。

### 4.7 區段 POI 與空間屬性
- 基礎統計：各 POI 指標的平均、標準差、缺值比例；針對極端密度值做 Winsorization 建議。
- 降維：建議使用 PCA 或因子分析找出主要區位因子，並檢查與成交金額的相關性。
- 聚類：依 POI 特徵將 sector 分群（KMeans 或層次式），比對不同集群的成交水準差異。
- 空間參考：結合鄰近表計算空間加權平均或差異，確認 POI 是否具有空間自相關。

## 5. 衍生指標與特徵工程方向（EDA 輸出）
- **時間衍生**：lag 1/3/6/12 月、rolling mean/std、成長率（MoM、YoY）、指數平滑、假期旗標（春節、十一）。
- **比率與效率**：
  - `sell_through_speed = num_new_house_transactions / num_new_house_available_for_sale`。
  - `inventory_months = num_new_house_available_for_sale / max(num_new_house_transactions, 1)`。
  - `new_vs_preowned_ratio`, `new_vs_land_amount_ratio` 等交叉比值。
- **搜尋與需求融合**：建立總搜尋量、PC 比例、top 10 關鍵字滯後值、移動平均、z-score。
- **宏觀與 POI 指標**：
  - 人均 GDP、財政收入成長率、醫療/教育資源密度。
  - POI PCA 分數、商業/交通指標的標準化分位數。
- **空間差異**：本區 vs 鄰近區段的差值、比率、z-score；必要時納入同城市 median 作為基準。
- **品質控管**：針對缺值補值策略建立 `*_was_missing`、`*_imputed_by` 等旗標，以便模型辨識不確定性。

## 6. 後續腳本設計建議（僅規劃）
1. `EDA/run_eda_pipeline.py`（尚未實作）：
   - 模組化流程：`load_data` → `clean_standardize` → `merge_panel` → `generate_profiles` → `produce_visuals` → `compose_report`。
   - 以 `conf/eda_config.yaml` 定義關鍵字分組、滯後窗口、輸出圖表清單（可於未來任務建立）。
2. 輸出規劃（未產生）：
   - `EDA/outputs/tables/`：資料品質、相關矩陣、聚類摘要等表格。
   - `EDA/outputs/figures/`：趨勢圖、熱力圖、箱型圖、聚類視覺化。
   - `EDA/reports/initial_eda_summary.md`：結論與特徵建議。
3. 可重現性：
   - 設定隨機種子、紀錄環境需求（pandas、numpy、matplotlib、seaborn、scipy、scikit-learn、statsmodels、plotly 選配）。
   - 若需中文字體供圖表顯示，請在未來 README 註明安裝方式。

## 7. 時程與交付驗證
- **第 1 階段：資料讀取與品質盤點**（預估 1~1.5 日）
  - 完成欄位型別、缺失值、重複鍵、時間覆蓋檢查。
  - 產出資料字典草稿與讀取腳本雛形。
- **第 2 階段：主體分析與交叉關聯**（預估 2 日）
  - 建立 `month × sector` 面板，完成新屋/二手/土地/搜尋/宏觀/POI 交叉分析。
  - 蒐集候選特徵清單與初步重要性假設。
- **第 3 階段：報告撰寫與後續計畫**（預估 0.5 日）
  - 彙整圖表、指標結論，確認與競賽評分一致。
  - 設定後續自動化腳本的需求工單與資源清單。
- 每個階段皆需保留檢查結果的摘要（可於後續報告中呈現），並再次核對與 `test.csv` 的對齊狀態。

---
此規劃書為後續 EDA 與特徵工程的工作藍圖；在未來開發腳本與報告時應以此為依據，並依實際分析結果迭代更新。
