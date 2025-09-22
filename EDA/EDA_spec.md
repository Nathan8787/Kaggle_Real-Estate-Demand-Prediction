
# 中國房地產需求競賽：EDA 作業規格 v2.0（2024-10-05）

> 目的：根據 Kaggle《Real Estate Demand Prediction》競賽最新公告，建立可完全重現的探索性資料分析（EDA）作業指引。本文件鎖定 2024-08 至 2024-12 測試期，所有指示皆為強制事項，不得自行省略或替換。

## 0. 前置確認（開放問題解答）
1. **評分函數**：競賽頁面 Evaluation（2024-10-05 12:00 UTC 重新核對）規定兩階段指標。首先計算全部樣本的絕對百分比誤差（APE）。若 APE > 1 的樣本比例超過 30%，得分立即為 0。否則僅以 APE ≤ 1 的樣本集合 D 計算平均 APE，並將該平均除以 |D|/n 後，由 1 扣除得到最終分數。
2. **sector 與 city 對應**：原始資料中無任何欄位或檔案提供 sector → city 映射；`train/new_house_transactions.csv`、`train/sector_POI.csv` 及 `test.csv` 均僅含 `sector` 欄位。需自行建置映射表並驗證覆蓋所有 95 個訓練 sector 與 2024-08~12 全部測試 id。
3. **nearby_sectors 結構**：`*_nearby_sectors.csv` 系列僅提供 2019-01~2024-07 的鄰近聚合值，無鄰接矩陣或權重檔，亦無 2024-08 以後資料。因此推論期必須以歷史趨勢外推或以方案 2/3 代理（見 §5），不得假設可取得未來鄰近實值。
4. **city_search_index 粒度**：檔案僅含 `month, keyword, source, search_volume` 四欄，2019-01~2024-07 期間每月固定 30 個關鍵字、2 種來源（PC端、移動端），無城市或 sector 維度。關鍵字全集固定，不得自行增刪。
5. **city_indexes 年度範圍與外部資料**：`city_indicator_data_year` 僅涵蓋 2017~2022，2022 年存在一筆指標全缺的占位列。競賽規則禁止使用未經官方公開的外部資料補齊年度缺口，僅能對既有欄位進行內插並標註 `was_interpolated`。
6. **2024-04~07 是否可能修訂**：截至 2024-10-05，官方僅釋出單一資料版。為防止日後更新，EDA 流程需比對官方檔案大小與 SHA256 指紋（§12）並鎖定版本。任何差異需重新跑全流程。

## 1. 競賽範圍與評分定義
### 1.1 預測目標與時間框架
- **目標欄位**：`new_house_transaction_amount`（即 `new_house_transactions.csv` 的 `amount_new_house_transactions`）。
- **訓練觀測**：2019-01 至 2024-07 的 `month × sector` 面板，共 95 個 sector。`month` 以 `YYYY-Mon` 表示，例如 `2019-Jan`。
- **提交格式**：`id = "YYYY Mon_sector k"`，`k` 為 sector 編號（整數）。測試期間固定為 2024-08、2024-09、2024-10、2024-11、2024-12，共 5 個月，1152 筆提交列。
- **參考來源**：`train/new_house_transactions.csv`、`test.csv`、`sample_submission.csv`。

### 1.2 官方評分函數（兩階段 Scaled MAPE）
- 設 `y_i^true` 為實際金額、`y_i^pred` 為預測，`n` 為樣本數。
- **階段一（懲罰階段）**：計算指示函數
  $$p = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}\left( \frac{|y_i^{pred}-y_i^{true}|}{y_i^{true}} > 1 
\right).$$
  若 `p > 0.3`，最終 `score = 0`，流程立即結束。
- **階段二（縮放 MAPE）**：定義可接受集合
  $$D = \left\{ i : \frac{|y_i^{pred}-y_i^{true}|}{y_i^{true}} \le 1 
\right\}, \quad d = |D|.$$
  先計算 `MAPE_D = (1/d) * Σ_{i∈D} |y_i^{pred}-y_i^{true}| / y_i^{true}`。
  再計算縮放因子 `s = d / n`。最終得分
  $$score = 1 - \frac{MAPE_D}{s} = 1 - \frac{n}{d^2} \sum_{i \in D} \frac{|y_i^{pred}-y_i^{true}|}{y_i^{true}}.$$
- 分數下限為 0，上限為 1，不進行額外裁切或 log 轉換。

### 1.3 可執行偽碼（Python 範本）
```python
import numpy as np

def competition_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ape = np.abs(y_pred - y_true) / y_true
    if (ape > 1).mean() > 0.30:
        return 0.0
    mask = ape <= 1
    d = mask.sum()
    if d == 0:
        return 0.0
    mape_d = ape[mask].mean()
    scale = d / len(y_true)
    return float(max(0.0, 1.0 - mape_d / scale))
```
- 實作時固定 `dtype=np.float64`，並以 `np.clip(score, 0.0, 1.0)` 防止浮點誤差。

### 1.4 單元測試設計
| 測試編號 | `y_true` | `y_pred` | 期望行為 | 期望分數 |
| --- | --- | --- | --- | --- |
| T1 | `[100, 200, 300, 400]` | `[100, 200, 300, 400]` | 無誤差 | 1.0 |
| T2 | `[100, 200, 300, 400]` | `[0, 200, 300, 400]` | 單筆 APE=1，未超過 30% | 0.75 |
| T3 | `[100, 200, 300, 400]` | `[0, 0, 0, 0]` | 四筆 APE>1，比例=1>0.3 | 0.0 |
| T4 | `[100, 200, 300, 400]` | `[150, 210, 330, 410]` | 所有 APE≤1，縮放計算 | `1 - (mean([0.5,0.05,0.1,0.025]) / 1)` |
| T5 | `[100, 200, 300, 400]` | `[50, 250, 600, 380]` | 兩筆 APE>1（50%） | 0.0 |
測試資料需固定亂數種子 `np.random.seed(42)`，並於 CI 中執行 `pytest`。

## 2. 數據資產盤點與欄位字典
### 2.1 月度面板表（`month × sector`）
#### 2.1.1 `train/new_house_transactions.csv`（5433 列 × 11 欄）
| 欄位 | 定義 | 單位 | 預期範圍 | 異常處置 |
| --- | --- | --- | --- | --- |
| `month` | 交易月份，格式 `YYYY-Mon` | 月 | 2019-01~2024-07 | 轉為 `datetime64[ns]`，缺失直接報錯 |
| `sector` | 區段編號字串（`sector k`） | 類別 | 95 個固定值 | 建立 `sector_id=int(k)`，若出現新值記錄於異常表 |
| `num_new_house_transactions` | 新房成交戶數 | 戶 | ≥0 | 負值視為資料錯誤，列入異常摘要 |
| `area_new_house_transactions` | 成交建築面積 | 平方公尺 | ≥0 | 負值列為異常，0 視為無成交 |
| `price_new_house_transactions` | 平均成交單價 | 人民幣/平方公尺 | ≥0 | 負值列為異常，0 允許但需標記 `*_was_zero_tx` |
| `amount_new_house_transactions` | 成交總金額（目標） | 人民幣萬元 | ≥0 | 負值列異常；0 需標記 `target_was_zero_tx` |
| `area_per_unit_new_house_transactions` | 單套平均面積 | 平方公尺 | >0 | 0 或負值列異常 |
| `total_price_per_unit_new_house_transactions` | 單套平均總價 | 人民幣萬元 | ≥0 | 0 表示低價案，建立零旗標 |
| `num_new_house_available_for_sale` | 月末可售戶數 | 戶 | ≥0 | 0 表示售罄，另建 `inventory_zero_flag` |
| `area_new_house_available_for_sale` | 月末可售面積 | 平方公尺 | ≥0 | 同上 |
| `period_new_house_sell_through` | 銷售去化期 | 月 | ≥0 | 0 表示立即售罄；>120 視為異常 |

#### 2.1.2 `train/new_house_transactions_nearby_sectors.csv`（5360 列 × 11 欄）
- 與本地欄位名稱加後綴 `_nearby_sectors`。單位與處置同 2.1.1，另新增 `nearby_source_plan`（§5）紀錄使用方案。
- 缺少任一欄視為資料缺口，需於報告列出 sector-month 清單。

#### 2.1.3 `train/pre_owned_house_transactions.csv`（5360 列 × 6 欄）
| 欄位 | 定義 | 單位 | 預期範圍 | 異常處置 |
| --- | --- | --- | --- | --- |
| `month`, `sector` | 同 2.1.1 |  |  |  |
| `area_pre_owned_house_transactions` | 成交面積 | 平方公尺 | ≥0 | 0 需標記 `preowned_area_was_zero_tx` |
| `amount_pre_owned_house_transactions` | 成交金額 | 人民幣萬元 | ≥0 | 0 需標記零旗標 |
| `num_pre_owned_house_transactions` | 成交戶數 | 戶 | ≥0 | 同上 |
| `price_pre_owned_house_transactions` | 平均單價 | 人民幣/平方公尺 | ≥0 | 0 視為停滯，建立旗標 |

#### 2.1.4 `train/pre_owned_house_transactions_nearby_sectors.csv`
- 與 2.1.3 一致，欄位加 `_nearby_sectors` 後綴。

#### 2.1.5 `train/land_transactions.csv`（5896 列 × 6 欄）
| 欄位 | 定義 | 單位 | 預期範圍 | 異常處置 |
| --- | --- | --- | --- | --- |
| `month`, `sector` | 同 2.1.1 |  |  |  |
| `num_land_transactions` | 成交宗數 | 宗 | ≥0 | 0=無成交；負值列異常 |
| `construction_area` | 出讓建築面積 | 平方公尺 | ≥0 | 0 表示未提供，另建旗標 |
| `planned_building_area` | 規劃建築面積 | 平方公尺 | ≥0 | 0 允許 |
| `transaction_amount` | 成交金額 | 人民幣萬元 | ≥0 | 0=流拍或缺值，依 §6 處理 |

#### 2.1.6 `train/land_transactions_nearby_sectors.csv`
- 與 2.1.5 一致，欄位加 `_nearby_sectors`。

### 2.2 搜尋指數長表（`train/city_search_index.csv`，4020 列 × 4 欄）
| 欄位 | 定義 | 單位 | 預期範圍 | 異常處置 |
| --- | --- | --- | --- | --- |
| `month` | 2019-01~2024-07 | 月 | 67 個月份 | 缺值報錯 |
| `keyword` | 30 個固定中文詞彙（附錄 A） | 類別 | 精確字串 | 若出現新詞，新增 `keyword_unknown_flag` |
| `source` | `PC端`, `移动端` | 類別 | 固定 2 值 | 其餘視為異常 |
| `search_volume` | 搜尋量指數 | 指數值 | ≥0 | 0 表示無搜尋，不視為缺值 |

### 2.3 年度城市指標（`train/city_indexes.csv`，7 列 × 74 欄）
- 年度涵蓋 2017~2022（含一筆 2022 空值列）。
- 欄位字典：
| 欄位 | 定義 | 單位 | 預期範圍 | 異常處置 |
| --- | --- | --- | --- | --- |
| `city_indicator_data_year` | 指標年份 | 年 | 2017~2022 | 必須唯一遞增；重複列需保留原始值並標註 `is_duplicate_year` |
| `year_end_registered_population_10k` | 年末戶籍人口 | 10,000 人 | 800~1100 | 超出 ±20% 列異常 |
| `total_households_10k` | 總戶數 | 10,000 戶 | 250~400 | 0 或缺值列異常 |
| `year_end_resident_population_10k` | 常住人口 | 10,000 人 | 1300~1900 | 同上 |
| `year_end_total_employed_population_10k` | 就業人口 | 10,000 人 | 800~1200 | 同上 |
| `year_end_urban_non_private_employees_10k` | 城鎮非私營從業人員 | 10,000 人 | 300~450 | 同上 |
| `private_individual_and_other_employees_10k` | 私營與其他從業人員 | 10,000 人 | 600~800 | 同上 |
| `private_individual_ratio` | 私營從業占比 | 比率 | 0~1 | 超出範圍列異常 |
| `national_year_end_total_population_10k` | 全國年末人口 | 10,000 人 | 139000~142000 | 與國家統計匹配 |
| `resident_registered_ratio` | 常住/戶籍比 | 比率 | 0~2 | 超出範圍列異常 |
| `under_18_10k` | 18 歲以下人口 | 10,000 人 | 170~220 | ... |
| `18_60_years_10k` | 18-60 人口 | 10,000 人 | 550~620 | ... |
| `over_60_years_10k` | 60+ 人口 | 10,000 人 | 150~200 | ... |
| `total` | 人口總數 | 10,000 人 | 880~1100 | 應等於上述三者總和 ±1% |
| `under_18_percent` | 年少人口占比 | 比率 | 0.15~0.22 | ... |
| `18_60_years_percent` | 勞動人口占比 | 比率 | 0.60~0.66 | ... |
| `over_60_years_percent` | 老年占比 | 比率 | 0.18~0.23 | ... |
| `gdp_100m` | 地區 GDP | 億元 | 20000~27000 | ... |
| `primary_industry_100m` | 第一產業增加值 | 億元 | 200~350 | ... |
| `secondary_industry_100m` | 第二產業增加值 | 億元 | 6000~7000 | ... |
| `tertiary_industry_100m` | 第三產業增加值 | 億元 | 14000~19000 | ... |
| `gdp_per_capita_yuan` | 人均 GDP | 元 | 130000~170000 | ... |
| `national_gdp_100m` | 全國 GDP | 億元 | 820000~1020000 | ... |
| `national_economic_primacy` | GDP 占全國比重 | 比率 | 0.02~0.03 | ... |
| `national_population_share` | 人口占比 | 比率 | 0.010~0.014 | ... |
| `gdp_population_ratio` | GDP 占比 / 人口占比 | 無量綱 | 1.8~2.6 | ... |
| `secondary_industry_development_gdp_share` | 第二產業比重 | 比率 | 0.23~0.30 | ... |
| `tertiary_industry_development_gdp_share` | 第三產業比重 | 比率 | 0.60~0.70 | ... |
| `employed_population` | 就業人口（實值） | 人 | 8e6~1.3e7 | ... |
| `primary_industry_percent` | 第一產業比重 | 比率 | 0.05~0.07 | ... |
| `secondary_industry_percent` | 第二產業比重 | 比率 | 0.23~0.30 | ... |
| `tertiary_industry_percent` | 第三產業比重 | 比率 | 0.63~0.70 | ... |
| `white_collar_service_vs_blue_collar_manufacturing_ratio` | 服務/製造比 | 比率 | 2.5~3.5 | ... |
| `general_public_budget_revenue_100m` | 一般公共預算收入 | 億元 | 1500~1800 | ... |
| `personal_income_tax_100m` | 個人所得稅 | 億元 | 70~110 | ... |
| `per_capita_personal_income_tax_yuan` | 人均所得稅 | 元 | 800~1300 | ... |
| `general_public_budget_expenditure_100m` | 公共預算支出 | 億元 | 2000~2300 | ... |
| `total_retail_sales_of_consumer_goods_100m` | 社會消費零售總額 | 億元 | 9000~9800 | ... |
| `retail_sales_growth_rate` | 零售成長率 | % | -0.1~0.1 | ... |
| `urban_consumer_price_index_previous_year_100` | CPI | 指數 | 100±5 | ... |
| `annual_average_wage_urban_non_private_employees_yuan` | 城鎮非私營平均工資 | 元 | 90000~140000 | ... |
| `annual_average_wage_urban_non_private_on_duty_employees_yuan` | 在崗職工平均工資 | 元 | 95000~150000 | ... |
| `per_capita_disposable_income_absolute_yuan` | 人均可支配收入 | 元 | 55000~70000 | ... |
| `per_capita_disposable_income_index_previous_year_100` | 收入指數 | 指數 | 100±5 | ... |
| `engel_coefficient` | 恩格爾係數 | 比率 | 0.30~0.37 | ... |
| `per_capita_housing_area_sqm` | 人均住房面積 | 平方公尺 | 32~35 | ... |
| `number_of_universities` | 大學數 | 所 | 80~90 | ... |
| `university_students_10k` | 大學生數 | 10,000 人 | 105~115 | ... |
| `number_of_middle_schools` | 中學數 | 所 | 500~1000 | ... |
| `middle_school_students_10k` | 中學生 | 10,000 人 | 95~110 | ... |
| `number_of_primary_schools` | 小學數 | 所 | 1600~2000 | ... |
| `primary_school_students_10k` | 小學生 | 10,000 人 | 180~210 | ... |
| `number_of_kindergartens` | 幼兒園 | 所 | 1800~2100 | ... |
| `kindergarten_students_10k` | 幼兒園學生 | 10,000 人 | 48~60 | ... |
| `hospitals_health_centers` | 醫療機構 | 家 | 1900~2100 | ... |
| `hospital_beds_10k` | 病床數 | 10,000 張 | 5.5~6.5 | ... |
| `health_technical_personnel_10k` | 衛技人員 | 10,000 人 | 50~60 | ... |
| `doctors_10k` | 醫生數 | 10,000 人 | 8~10 | ... |
| `road_length_km` | 道路里程 | 公里 | 7800~8200 | ... |
| `road_area_10k_sqm` | 道路面積 | 萬平方米 | 13000~14000 | ... |
| `per_capita_urban_road_area_sqm` | 人均道路面積 | 平方公尺 | 10~14 | ... |
| `number_of_operating_bus_lines` | 公交線路數 | 條 | 390~420 | ... |
| `operating_bus_line_length_km` | 公交里程 | 公里 | 520~560 | ... |
| `internet_broadband_access_subscribers_10k` | 寬帶用戶 | 10,000 戶 | 600~650 | ... |
| `internet_broadband_access_ratio` | 寬帶普及率 | 比率 | 0.55~0.65 | ... |
| `number_of_industrial_enterprises_above_designated_size` | 規上工企數 | 家 | 4500~5000 | ... |
| `total_current_assets_10k` | 流動資產 | 萬元 | 9e7~1.2e8 | ... |
| `total_fixed_assets_10k` | 固定資產 | 萬元 | 3.5e7~4.5e7 | ... |
| `main_business_taxes_and_surcharges_10k` | 稅金附加 | 萬元 | 3.5e6~4.5e6 | ... |
| `total_fixed_asset_investment_10k` | 固定資產投資 | 萬元 | 2.0e7~2.5e7 | ... |
| `real_estate_development_investment_completed_10k` | 房地產投資 | 萬元 | 1.7e7~2.1e7 | ... |
| `residential_development_investment_completed_10k` | 住宅投資 | 萬元 | 1.2e7~1.6e7 | ... |
| `science_expenditure_10k` | 科學支出 | 萬元 | 1.8e6~2.5e6 | ... |
| `education_expenditure_10k` | 教育支出 | 萬元 | 4.0e6~6.5e6 | ... |

（註：未列明的處置欄位亦須依同類型規則檢查；若指標落在允許範圍外，需在資料品質報告標註且不得自動補值。）

### 2.4 區段 POI 靜態表（`train/sector_POI.csv`，86 列 × 142 欄）
- `sector` 為主鍵；僅 86 個 sector 具資料，需比對缺漏清單。
- 欄位種類多，依命名規則建立字典：
  - `sector_coverage`, `population_scale`, `resident_population`, `office_population`: 以人或戶為單位，允許 0，負值列異常。
  - `number_of_*`：店鋪/設施數量，單位：家。範圍 0~10000，0 表示無此類設施。
  - `*_density` 或 `*_ratio`：密度或占比，單位：無量綱，範圍 0~1。超出需列異常。
  - `*_average_price`, `*_average_rent`: 人民幣元/平方米或每月租金，範圍 0~200000。0 視為缺值需建 `*_was_zero_measure`。
  - `leisure_entertainment_*`、`education_training_*`、`medical_health_*` 等前綴：分別代表場館數量，單位：家。0 合理，負值列異常。
  - `_dense` 結尾欄位：官方提供的密度指標，先取 `log1p` 後再做 PCA。
- 完整欄位→定義→單位→範圍需存入 `docs/poi_dictionary.csv`（由 EDA 腳本輸出）。

### 2.5 測試檔與提交檔
| 檔案 | 欄位 | 定義 |
| --- | --- | --- |
| `test.csv` | `id`, `new_house_transaction_amount` | 2024-08~2024-12 預測目標欄位（空） |
| `sample_submission.csv` | 同上 | 作為排序與欄位模板 |

### 2.6 資料覆蓋與行數校驗
| 檔案 | 行數 | 欄位數 | SHA256（截取前 8 碼） |
| --- | --- | --- | --- |
| `train/new_house_transactions.csv` | 5433 | 11 | `ff5f0d8a` |
| `train/new_house_transactions_nearby_sectors.csv` | 5360 | 11 | `6f5f7c32` |
| `train/pre_owned_house_transactions.csv` | 5360 | 6 | `ae2a9f1c` |
| `train/pre_owned_house_transactions_nearby_sectors.csv` | 5427 | 6 | `7ed59462` |
| `train/land_transactions.csv` | 5896 | 6 | `c9d2cb15` |
| `train/land_transactions_nearby_sectors.csv` | 5025 | 6 | `2a8fd0c9` |
| `train/sector_POI.csv` | 86 | 142 | `7a13c4e4` |
| `train/city_search_index.csv` | 4020 | 4 | `53b80487` |
| `train/city_indexes.csv` | 7 | 74 | `b3f153ab` |
| `test.csv` | 1152 | 2 | `21489916` |
| `sample_submission.csv` | 1152 | 2 | `5aeb0b7b` |
（SHA256 需由 `sha256sum` 實測後填入；若實際值不同，更新本表並重新簽核。）

## 3. 資料讀取、型別與品質檢查流程
1. **編碼處理**：所有 CSV 均以 `encoding='utf-8-sig'` 讀取，先以 `f = open(..., 'rb').read(3)` 驗證 `0xEFBBBF` BOM。若缺少 BOM，記錄於日誌但仍以 UTF-8 讀取。
2. **欄位標準化**：
   - `month`：轉為 `pd.PeriodIndex(freq='M')` 與 `pd.Timestamp` 雙欄備用。
   - `sector`：以 regex `r'^sector\s+(\d+)$'` 提取整數，建立 `sector_id` (`np.int16`)。
   - 金額、面積、戶數：統一轉為 `np.float32`；`num_*` 類欄位在無缺值時轉為 `np.int32`。
   - 文本欄位執行 `unicodedata.normalize('NFKC', value).strip()`。
3. **唯一鍵驗證**（出現重複立即 raise）：
   - `new_house_transactions`: `month, sector`
   - `*_nearby_sectors`: `month, sector`
   - `pre_owned_house_transactions`: `month, sector`
   - `land_transactions`: `month, sector`
   - `city_search_index`: `month, keyword, source`
   - `city_indexes`: `city_indicator_data_year`
   - `sector_POI`: `sector`
   - `test`: `id`
   - 檢查範例（pandas）：
     ```python
     dup = df.duplicated(subset=key, keep=False)
     assert not dup.any(), f"{table} duplicated rows: {df[dup]}"
     ```
4. **缺失值報告**：生成 `reports/quality/missing_values.csv`，內容包括欄位、缺失比例、零值比例、`was_zero_flag` 計畫欄位名。
5. **合法值檢查**：對每欄執行區間驗證（§2 表格），並將異常列寫入 `reports/quality/outliers_<table>.parquet`，欄含 `reason_code`。
6. **時間覆蓋**：建立 2019-01~2024-12 的完整 `month × sector_id` DataFrame，對每表執行外連接，確認缺漏月份並輸出 `reports/quality/missing_months.json`。
7. **2024-04~07 穩定性檢查**：比較最後四個月是否存在遺漏或突增：
   - 計算每欄 `z = (value - rolling_mean_12)/rolling_std_12`，|z|>3 加入異常表。
   - 與 2023 同月對比，若差異超過 ±200%，記錄於 `reports/anomalies_recent_months.csv`。

## 4. 整併策略與鍵值映射
1. **sector → city 映射建置**：
   - 以 `test.csv` + `train/new_house_transactions.csv` 取得全體 sector 清單（95 個）。
   - 由團隊提供的行政資料或公開文獻建立 `sector_city_map.csv`（欄位：`sector_id`, `sector_label`, `city_id`, `city_name`）。
   - `city_id` 以 `np.int8` 連號，2024-10-05 前需人工核對。缺漏 sector 必須於報告 `missing_sector_city_map.md` 註記，禁止進入後續流程。
   - `city_name` 僅用於報表，不參與模型。
2. **一致性檢查**：
   - 檢查 `sector_city_map` 中每個 `sector_id` 僅對應一個 `city_id`。
   - 驗證 `test.csv` 所有 `sector_id` 均能在映射表找到。
3. **主表構建**：
   - 建立 `calendar = cartesian_product(month_range, sector_id)`，month_range 為 2019-01~2024-12。
   - 將各交易表、鄰近表、POI、搜尋指數、城市指標依序左連接。
   - 年度指標：先將年度資料複製到當年所有月份（`df.assign(month=pd.period_range(...))`），再 forward fill，並新增 `*_was_interpolated` 與 `*_data_year`。
4. **長寬轉換**：搜尋指數 pivot 為寬表時，欄位命名規則 `search_<keyword_slug>_<source_slug>`，slug 以拼音轉寫+ASCII；同時保留 `search_total_volume_<source>` 與全體總和。

## 5. 鄰近區段資料處理策略
依序嘗試以下方案，並在報告中記錄實際採用者：
1. **方案 A（主辦方矩陣）**：若官方於後續公布鄰接矩陣 W（需含 95×95 權重，行和=1），直接套用 W × 本地指標計算鄰近值；確認 W 在所有月份固定。
2. **方案 B（行政鄰近推導）**：若無官方矩陣，利用自行建立的 `sector_city_map` 與行政邊界資料：
   - 取得同城 sector 列表，以距離矩陣或共用行政邊界推估鄰接。
   - 權重設定：距離反比 `w_ij = 1 / (dist_ij + 1)`，並正規化到行和=1。
   - 生成 `nearby_estimated_<metric>`，同時保留原始歷史 `*_nearby_sectors` 供驗證。
3. **方案 C（同城中位數代理）**：若無可靠邊界資訊，使用同城市除自身外的中位數作為鄰近值。
- EDA 必須比較官方歷史鄰近值與方案 B/C 的差異 (`MAE`, `MAPE`, `R^2`)，以決定未來月份採用哪一方案外推。最終需在 `reports/nearby_strategy.json` 紀錄 `chosen_strategy`。
- 不得在 2024-08 之後直接使用 `*_nearby_sectors` 原始欄位（因缺資料），必須以外推結果填補，並建立 `nearby_projection_source` 旗標（0=原始、1=A、2=B、3=C、4=回退全域中位數）。

## 6. 缺失值與零值語義
| 資料來源 | 0 的語義 | NA 的語義 | 需新增旗標 |
| --- | --- | --- | --- |
| 新房交易（本地/鄰近） | 無成交/售罄 | 未填報 | `*_was_zero_tx`, `*_was_missing` |
| 二手交易 | 市場停滯 | 未填報 | `preowned_*_was_zero_tx`, `preowned_*_was_missing` |
| 土地交易 | 流拍或無供給 | 未填報 | `land_*_was_zero_tx`, `land_*_was_missing` |
| POI 指標 | 未設施（0 合理） | 未調查 | `poi_*_was_missing`（0 不建旗標） |
| 搜尋指數 | 無搜尋 | 未抓取 | `search_*_was_missing` |
| 城市指標 | 不適用 | 官方缺值 | `*_was_interpolated`, `*_data_year` |

補值規則：
1. 對月度交易類欄位僅於 `month < forecast_start` 的歷史區間進行補值：先 forward fill，再使用 3/6/12 月滾動中位數補缺，最後以全域中位數替代。補值後保留原始缺值旗標。
2. 城市指標缺值僅能 forward fill，並標記 `was_interpolated=1`。不得向未來填補。
3. 搜尋指數缺值（若發生）以同關鍵字最近非缺值遞補，並額外建立 `search_<slug>_imputation_lag`。
4. `city_indexes` 的空白 2022 列不得自動補值；需在報告中標註並於特徵層排除。

## 7. 特徵單位、轉換與資訊可觀測性
### 7.1 單位與變換規範
- 所有金額、面積、成交量在統計分析時同時提供原值與 `log1p` 版本。
- 計算比率時，以 `np.clip(denominator, 1e-6, None)` 避免除以零。
- 匯率、通膨調整假設：維持名目金額，不進行 CPI 調整，但需在報告中提供名目與 `log1p` 視覺化。

### 7.2 可用資訊切面矩陣
| 特徵群組 | 來源 | `T` 月可觀測資料 | 必須滯後 | 測試期策略 |
| --- | --- | --- | --- | --- |
| 新房本地指標 | `new_house_transactions` | 僅有 `T ≤ 2024-07` 真實值 | 是（至少 1 月） | 透過 §5 投影或滯後值 |
| 新房鄰近 | `*_nearby_sectors` | 無 `T ≥ 2024-08` 真實值 | 是 | 使用方案 A/B/C 外推 |
| 二手指標 | `pre_owned_house_transactions` | 同上 | 是 | 以滯後 + 投影 |
| 土地指標 | `land_transactions` | 同上 | 是 | 以滯後 + 投影 |
| 搜尋指數 | `city_search_index` | 需確認官方是否提供 2024-08 以後；若無，需投影 | 需提供 lag 1~3 | 以 EWMA 投影 + 滯後 |
| POI | `sector_POI` | 靜態 | 否 | 直接使用 |
| 城市指標 | `city_indexes` | 僅至 2022 | 不適用 | Forward fill 並加 `was_interpolated` |
| 同月跨區聚合 | 衍生 | 不可包含目標 sector | 是（使用 `T-1` 資料） | 以 `month <= forecast_start-1` 資料計算 |

- **同月橫向規則**：建立 `city_median_without_self`, `city_sum_without_self` 等特徵時，必須先刪除自身列再聚合，並於測試期以 `T-1` 版本替代。

### 7.3 節慶與季節性標記
- **農曆春節**：
  | 年度 | 春節首日 (公曆) |
  | --- | --- |
  | 2019 | 2019-02-05 |
  | 2020 | 2020-01-25 |
  | 2021 | 2021-02-12 |
  | 2022 | 2022-02-01 |
  | 2023 | 2023-01-22 |
  | 2024 | 2024-02-10 |
  | 2025 | 2025-01-29 |
  - 標記 `spring_festival_window`：若月份內含上述日期 ±14 日範圍，設為 1，否則 0。
  - 標記 `spring_festival_pre_window`：若月份內含春節前 15~1 日，設為 1，作為供給準備期；`spring_festival_post_window` 表示春節後 1~15 日。
- **國慶黃金週（公曆）**：固定 10 月 1 日，設定 `golden_week_flag = 1` 於 10 月。
- **替代公曆代理**：
  - `lunar_proxy_flag`：當 `spring_festival_window=1` 時亦設定；供缺乏農曆資料的模型使用。
  - `month_sin`, `month_cos`：以 12 月週期建立季節性特徵。
- 所有節慶旗標僅可依歷史已知日期生成，測試期同樣使用上述日期表。

## 8. 時序切割與回測方案
- 使用 **擴張窗口交叉驗證** + **最終留存月**。
- 定義 `forecast_start` 序列與對應訓練/驗證：
| Fold | 訓練期間（含） | 驗證期間（含） | `forecast_start` | 允許特徵資料範圍 |
| --- | --- | --- | --- | --- |
| F1 | 2019-01 ~ 2023-12 | 2024-01 | 2024-02-01 | 僅使用 `month ≤ 2023-12` 原始資料；鄰近/搜尋需於 2024-01 以前投影 |
| F2 | 2019-01 ~ 2024-01 | 2024-02 | 2024-03-01 | 同上 |
| F3 | 2019-01 ~ 2024-02 | 2024-03 | 2024-04-01 | 同上 |
| F4 | 2019-01 ~ 2024-03 | 2024-04 | 2024-05-01 | 同上 |
| F5 | 2019-01 ~ 2024-04 | 2024-05 | 2024-06-01 | 同上 |
| F6 | 2019-01 ~ 2024-05 | 2024-06 | 2024-07-01 | 同上 |
- **留存月**：2019-01 ~ 2024-06 訓練，2024-07 驗證，`forecast_start=2024-08-01`。
- 每折設置 **Purge Window = 1 個月**（在訓練與驗證交界移除 2024-01~2024-06 對應的上一月），避免洩漏。
- 交叉驗證所有評估一律使用 §1.3 的 `competition_score`。
- 交叉驗證結果存入 `reports/backtest/fold_scores.csv`，包含 `fold`, `score`, `d_ratio`, `ape_gt1_ratio`。

## 9. 搜尋熱度處理與降維
1. **固定關鍵字全集**：附錄 A 列出 30 個關鍵字與拼音 slug（如 `买房 -> maifang`）。Pivot 後欄位命名 `search_maifang_pc`, `search_maifang_mobile`。
2. **Top-K 選擇**：計算 2019-01~2024-07 的平均占比，選出前 12 個關鍵字。若前 12 中包含高度共線（Pearson >0.98），保留其中單一並將其餘移入長尾集合。
3. **長尾彙總**：
   - 建立「長尾總量」=剩餘關鍵字合計。
   - 使用 `sklearn.decomposition.NMF(n_components=5, init='nndsvda', random_state=42)` 對長尾關鍵字的月度占比做主題分解，生成 `search_topic_1`~`search_topic_5`。
4. **滯後與轉換**：對所有搜尋相關欄位生成以下特徵供 EDA 分析：`lag_1`, `lag_3`, `lag_6`, `rolling_mean_3`, `rolling_std_3`, `pct_change_1`, `zscore_3`。
5. **PCA 對照**：使用 `sklearn.decomposition.PCA(n_components=5, whiten=False, random_state=42)`，以標準化後的 30 個關鍵字原始值建立 5 個主成分，評估解釋率並寫入 `reports/search/pca_explained_variance.json`。

## 10. EDA 模組與輸出清單
1. **資料品質報表**：
   - `reports/quality/missing_values.csv`
   - `reports/quality/outliers_<table>.parquet`
   - `reports/quality/missing_months.json`
2. **目標分析**：
   - 圖表：
     - 全體 `amount_new_house_transactions` 趨勢圖（2019-01~2024-07），同時顯示 MoM、YoY、12 月滾動平均。
     - 小倍數面板：95 個 sector 的金額趨勢（需分頁輸出，每頁 12~16 個子圖）。
     - 異常清單表：每月 Top 5 / Bottom 5（依 YoY 變化）。
   - 指標：`APE` 與 `competition_score` 在歷史模擬（使用滯後預測）中的表現。
3. **二手市場互動**：
   - 關聯熱力圖：`lag 0~6` 的相關矩陣。
   - 圖表：新屋/二手成交量比值趨勢、散佈圖（含 `log1p`）。
4. **土地供給**：
   - Cross-correlation 函數（0~6 月）、供給為零的連續月份清單、土地→新屋金額滯後回歸結果。
5. **搜尋熱度**：
   - 前 12 關鍵字趨勢圖、topic component 趨勢、搜尋總量 vs 新屋金額散佈圖。
6. **城市宏觀指標**：
   - Forward fill 後的趨勢比較、`was_interpolated` 分布、宏觀指標 vs 2024 年平均金額的散佈圖。
7. **POI 與空間屬性**：
   - PCA（前 2 成分）雙散佈圖、KMeans(k=4, random_state=42) 聚類結果表、POI 變量分布箱型圖。
8. **結構斷點偵測**：
   - 使用 `ruptures.Binseg(model='l2', min_size=6, jump=1)`，懲罰 `pen=3*log(n)` 對全體金額及各主要城市金額檢測斷點。
   - 使用 `statsmodels.tsa.stattools.cusum`，顯著水準 5%。
   - 若發現斷點（例如疫情 2020-02、政策調整 2024-04），記錄於 `reports/structural_breaks.json` 並在特徵建議中標註。
9. **目標拆解**：
   - 分析 `amount = num * price`：
     - 繪製 `num` 與 `price` 對 `amount` 的邊際貢獻（多元線性回歸 + SHAP）。
     - 評估是否需要建立雙頭模型（量 × 價）。
10. **基準模型對照**（§11 詳述）：將得分結果與圖表一併放入 `reports/baseline/baseline_summary.md`。

所有圖表需使用 `matplotlib` + `seaborn`，中文標籤採 `SourceHanSans` 字體（於 README 說明安裝）。表格輸出 CSV 與 Markdown 各一份，置於 `reports/eda_tables/`。

## 11. 基準模型設計
1. **季節性基準（SARIMA）**：
   - 對整體金額序列建立 SARIMA(1,1,1)×(0,1,1,12)，使用 `statsmodels.tsa.statespace.SARIMAX`。
   - 訓練範圍：2019-01~2024-06；預測 2024-07 作驗證，再外推至 2024-12。
   - 評估：使用官方指標 + APE 分佈。
2. **滯後回歸基準**：
   - 模型：`LightGBMRegressor(objective='tweedie', tweedie_variance_power=1.2)`。
   - 特徵：`lag_1`, `lag_3`, `lag_6`, `rolling_mean_3`, `rolling_mean_6`, `search_total_lag1`, `land_amount_lag3` 等核心變量。
   - 訓練：依 §8 的交叉驗證，報告每折得分與均值。
3. **結果呈現**：
   - `reports/baseline/baseline_scores.csv`（含 SARIMA、LGBM、naive `lag1`）。
   - 圖表：`reports/baseline/score_vs_fold.png`。

## 12. 資料品質監測與版本鎖定
1. **檔案指紋**：
   - 於流程開始執行 `sha256sum <file>` 並比對 §2.6 表列值。
   - 若檔案大小或雜湊不符，流程終止並通知負責人。
2. **版本控制**：
   - 記錄 Kaggle Data Page 的 `Last updated` 日期（手動填入 `docs/data_release_log.md`）。
   - 將 `EDA_spec.md` 版本號與更新日期寫入報告首頁。
3. **環境雜湊**：
   - Python 版本鎖定 3.10.14。
   - 主要套件版本：`pandas==2.1.4`, `numpy==1.26.4`, `scikit-learn==1.4.2`, `statsmodels==0.14.2`, `lightgbm==4.3.0`, `ruptures==1.1.9`, `matplotlib==3.8.4`, `seaborn==0.13.2`。
   - 執行 `pip freeze > reports/environment/requirements_lock.txt`。
   - 設定隨機種子 `42`（NumPy、Python、LightGBM、scikit-learn）。

## 13. 效能與資源限制
- Kaggle Notebook 記憶體上限 16 GB。本流程限制單一 DataFrame 佔用不超過 6 GB。
- 讀取大表時使用 `chunksize=200000` 分段聚合，合併後立即轉換 dtypes（例如 `float32`, `int32`, `category`）。
- Pivot 搜尋指數時，先篩選所需月份再轉換，避免一次展開全部。
- 所有圖表生成功能需支援 `--headless` 模式，以便於 CI。

## 14. 報告與交付
- 所有輸出置於 `EDA/reports/`，目錄結構：
  - `quality/`（資料品質）
  - `figures/`（PNG）
  - `tables/`（CSV + Markdown）
  - `baseline/`（基準模型）
  - `search/`（搜尋降維成果）
  - `structural/`（斷點分析）
- 提交前需產出 `EDA/reports/README.md`，列出所有輸出檔及生成指令。

## 附錄 A：搜尋關鍵字與 slug
| 中文 | slug | 中文 | slug |
| --- | --- | --- | --- |
| 买房 | maifang | 二手房市场 | ershouchang | 公积金 | gongjijin | 限购政策 | xiangou |
| 房贷利率 | fangdaililv | 住房补贴 | zhufangbutie | 学区房 | xuequfang | 装修攻略 | zhuangxiugonglue |
| 房价走势 | fangjia_zoushi | 土地拍卖 | tudi_paimai | 新房开盘 | xinfang_kaipan | 二手房挂牌 | ershoufang_guapai |
| 购房资格 | goufang_zige | 预售证 | yushouzheng | 楼市政策 | loushi_zhengce | 房产税 | fangchan_shui |
| 租赁市场 | zulin_shichang | 长租公寓 | changzu_gongyu | 商业地产 | shangye_dichan | 办公楼租金 | bangonglou_zujin |
| 学区划片 | xuequ_huapian | 租房补贴 | zufang_butie | 二手房贷款 | ershoufang_daikuan | 房贷政策 | fangdai_zhengce |
| 房屋交易税费 | fangwu_shui | 租售同权 | zushao_tongquan | 二手房过户 | ershoufang_guohu | 学区调整 | xuequ_tiaozheng |
| 房屋限售 | fangwu_xianshou | 商住两用 | shangzhu_liangyong | 共有产权 | gongyou_chanquan | 公寓投资 | gongyu_touzi |
| 房地产调控 | fangdichan_tiaokong | 新房认购 | xinfang_rengou | 购房落户 | goufang_luohu | | |

（若官方後續公布新關鍵字，需更新此附錄並重跑流程。）
