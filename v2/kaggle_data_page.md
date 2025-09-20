Title: Real Estate Demand Prediction

URL Source: https://www.kaggle.com/competitions/china-real-estate-demand-prediction/data

Markdown Content:
Participants are expected to use data in the `train` folder which contains all files for train set to predict monthly price of new house transactions for each sector.

### Predict Target

The target is in `train/new_house_transactions.csv` file under column `amount_new_house_transaction`. Please note that not all sectors and months are included in the file. For example, there is no data in the file for `2019 Jan_sector 3`, which means participants are expected to predict a value of 0.

```
month,sector,num_new_house_transactions,area_new_house_transactions,amount_new_house_transaction,...
2019 Jan_sector 1,52,4906,28184,...
2019 Jan_sector 2,145,15933,17747,...
2019 Jan_sector 4,6,725,28004,...
2019 Jan_sector 5,2,212,37432,...
2019 Jan_sector 6,5,773,15992,...
```

### Avoid Data Leakage

When predicting for a certain month, participants are _not_ supposed to use any data corresponding to a future time period relative to said month.

Files
-----

*   **train/*.csv** - the training set
*   **test.csv** - the test set
*   **sample_submission.csv** - a sample submission file in the correct format

Columns
-------

_Unless specified otherwise, all appearances of `yuan` in this section stand for the Chinese yuan, the official currency of China._

**train/pre_owned_house_transactions.csv**

*   **month**: The month of the transaction.
*   **sector**: The specific geographic sector where the transaction occurred.
*   **area_pre_owned_house_transactions**: The total area of pre-owned house transactions in square meters.
*   **amount_pre_owned_house_transactions**: The total monetary value of pre-owned house transactions in 10,000 yuan.
*   **num_pre_owned_house_transactions**: The total number of pre-owned house transactions.
*   **price_pre_owned_house_transactions**: The average price per square meter of pre-owned house transactions in yuan.

**train/pre_owned_house_transactions_nearby_sectors.csv**

*   **month**: The month of the transaction.
*   **sector**: The specific geographic sector of interest.
*   **area_pre_owned_house_transactions_nearby_sectors**: The total area of pre-owned house transactions in nearby sectors in square meters.
*   **amount_pre_owned_house_transactions_nearby_sectors**: The total monetary value of pre-owned house transactions in nearby sectors in 10,000 yuan.
*   **num_pre_owned_house_transactions_nearby_sectors**: The total number of pre-owned house transactions in nearby sectors.
*   **price_pre_owned_house_transactions_nearby_sectors**: The average price per square meter of pre-owned house transactions in nearby sectors in yuan.

**train/land_transactions.csv**

*   **month**: The month of the transaction.
*   **sector**: The specific geographic sector where the land transaction occurred.
*   **num_land_transactions**: The total number of land transactions.
*   **construction_area**: The total area of land designated for construction in square meters.
*   **planned_building_area**: The total planned building area on the transacted land in square meters.
*   **transaction_amount**: The total monetary value of land transactions in 10,000 yuan.

**train/land_transactions_nearby_sectors.csv**

*   **month**: The month of the transaction.
*   **sector**: The specific geographic sector of interest.
*   **num_land_transactions_nearby_sectors**: The total number of land transactions in nearby sectors.
*   **construction_area_nearby_sectors**: The total area of land designated for construction in nearby sectors in square meters.
*   **planned_building_area_nearby_sectors**: The total planned building area on transacted land in nearby sectors in square meters.
*   **transaction_amount_nearby_sectors**: The total monetary value of land transactions in nearby sectors in 10,000 yuan.

**train/new_house_transactions.csv**

*   **month**: The month of the transaction.
*   **sector**: The specific geographic sector where the new house transaction occurred.
*   **num_new_house_transactions**: The total number of new house transactions.
*   **area_new_house_transactions**: The total area of new house transactions in square meters.
*   **price_new_house_transactions**: The average price per square meter of new house transactions in yuan.
*   **amount_new_house_transactions**: The total monetary value of new house transactions in 10,000 yuan.
*   **area_per_unit_new_house_transactions**: The average area per new house transaction unit in square meters per unit.
*   **total_price_per_unit_new_house_transactions**: The average total price per new house transaction unit in 10,000 yuan per unit.
*   **num_new_house_available_for_sale**: The total number of new houses available for sale.
*   **area_new_house_available_for_sale**: The total area of new houses available for sale in square meters.
*   **period_new_house_sell_through**: The estimated time in months to sell all available new houses.

**train/new_house_transactions_nearby_sectors.csv**

*   **month**: The month of the transaction.
*   **sector**: The specific geographic sector of interest.
*   **num_new_house_transactions_nearby_sectors**: The total number of new house transactions in nearby sectors.
*   **area_new_house_transactions_nearby_sectors**: The total area of new house transactions in nearby sectors in square meters.
*   **price_new_house_transactions_nearby_sectors**: The average price per square meter of new house transactions in nearby sectors in yuan.
*   **amount_new_house_transactions_nearby_sectors**: The total monetary value of new house transactions in nearby sectors in 10,000 yuan.
*   **area_per_unit_new_house_transactions_nearby_sectors**: The average area per new house transaction unit in nearby sectors in square meters per unit.
*   **total_price_per_unit_new_house_transactions_nearby_sectors**: The average total price per new house transaction unit in nearby sectors in 10,000 yuan per unit.
*   **num_new_house_available_for_sale_nearby_sectors**: The total number of new houses available for sale in nearby sectors.
*   **area_new_house_available_for_sale_nearby_sectors**: The total area of new houses available for sale in nearby sectors in square meters.
*   **period_new_house_sell_through_nearby_sectors**: The estimated time in months to sell all available new houses in nearby sectors.

**train/sector_POI.csv**

*   **sector**: The specific geographic sector.
*   **sector_coverage**: The geographical extent or area covered by the sector.
*   **population_scale**: The general size of the population within the sector.
*   **residential_area**: The presence or extent of residential zones within the sector.
*   **office_building**: The presence or extent of office buildings within the sector.
*   **commercial_area**: The presence or extent of commercial zones within the sector.
*   **resident_population**: The number of people residing in the sector.
*   **office_population**: The number of people working in offices within the sector.
*   **number_of_shops**: The total count of shops in the sector.
*   **catering**: The number or density of catering establishments.
*   **retail**: The number or density of retail establishments.
*   **hotel**: The number or density of hotel establishments.
*   **transportation_station**: The number or density of transportation stations.
*   **education**: The number or density of educational facilities.
*   **leisure_and_entertainment**: The number or density of leisure and entertainment venues.
*   **bus_station_cnt**: The count of bus stations.
*   **subway_station_cnt**: The count of subway stations.
*   **rentable_shops**: The number of shops available for rent.
*   **surrounding_housing_average_price**: The average price of housing in the surrounding area.
*   **surrounding_shop_average_rent**: The average rent of shops in the surrounding area.
*   **leisure_entertainment_entertainment_venue_game_arcade**: The number or density of game arcades.
*   **leisure_entertainment_entertainment_venue_party_house**: The number or density of party houses.
*   **leisure_entertainment_cultural_venue_cultural_palace**: The number or density of cultural palaces.
*   **office_building_industrial_building_industrial_building**: The number or density of industrial buildings used as office spaces.
*   **education_training_school_education_middle_school**: The number or density of middle schools.
*   **education_training_school_education_primary_school**: The number or density of primary schools.
*   **education_training_school_education_kindergarten**: The number or density of kindergartens.
*   **education_training_school_education_research_institution**: The number or density of research institutions.
*   **medical_health**: The number or density of general medical and health facilities.
*   **medical_health_specialty_hospital**: The number or density of specialty hospitals.
*   **medical_health_tcm_hospital**: The number or density of Traditional Chinese Medicine (TCM) hospitals.
*   **medical_health_physical_examination_institution**: The number or density of physical examination institutions.
*   **medical_health_veterinary_station**: The number or density of veterinary stations.
*   **medical_health_pharmaceutical_healthcare**: The number or density of pharmaceutical healthcare providers.
*   **medical_health_rehabilitation_institution**: The number or density of rehabilitation institutions.
*   **medical_health_first_aid_center**: The number or density of first aid centers.
*   **medical_health_blood_donation_station**: The number or density of blood donation stations.
*   **medical_health_disease_prevention_institution**: The number or density of disease prevention institutions.
*   **medical_health_general_hospital**: The number or density of general hospitals.
*   **medical_health_clinic**: The number or density of clinics.
*   **transportation_facilities_service_bus_station**: The presence or density of bus stations.
*   **transportation_facilities_service_subway_station**: The presence or density of subway stations.
*   **transportation_facilities_service_airport_related**: The presence or density of airport-related facilities.
*   **transportation_facilities_service_port_terminal**: The presence or density of port or terminal facilities.
*   **transportation_facilities_service_train_station**: The presence or density of train stations.
*   **transportation_facilities_service_light_rail_station**: The presence or density of light rail stations.
*   **transportation_facilities_service_long_distance_bus_station**: The presence or density of long-distance bus stations.
*   **number_of_leisure_and_entertainment_stores**: The count of leisure and entertainment stores.
*   **number_of_other_stores**: The count of miscellaneous other stores.
*   **number_of_other_anchor_stores**: The count of other major or anchor stores.
*   **number_of_home_appliance_stores**: The count of home appliance stores.
*   **number_of_skincare_cosmetics_stores**: The count of skincare and cosmetics stores.
*   **number_of_fashion_stores**: The count of fashion stores.
*   **number_of_service_stores**: The count of service-oriented stores.
*   **number_of_jewelry_stores**: The count of jewelry stores.
*   **number_of_lifestyle_leisure_stores**: The count of lifestyle and leisure stores.
*   **number_of_supermarket_convenience_stores**: The count of supermarkets and convenience stores.
*   **number_of_catering_food_stores**: The count of catering and food stores.
*   **number_of_residential_commercial**: The count of commercial establishments within residential areas.
*   **number_of_office_building_commercial**: The count of commercial establishments within office buildings.
*   **number_of_commercial_buildings**: The count of dedicated commercial buildings.
*   **number_of_hypermarkets**: The count of hypermarkets.
*   **number_of_department_stores**: The count of department stores.
*   **number_of_shopping_centers**: The count of shopping centers.
*   **number_of_hotel_commercial**: The count of commercial establishments within hotels.
*   **number_of_third_tier_shopping_malls_in_business_district**: The count of third-tier shopping malls within the business district.
*   **number_of_second_tier_shopping_malls_in_business_district**: The count of second-tier shopping malls within the business district.
*   **number_of_city_winner_malls**: The count of high-performing "city winner" malls.
*   **number_of_shopping_malls_with_street_facing_shops**: The count of shopping malls featuring street-facing shops.
*   **number_of_unranked_malls**: The count of shopping malls without a specific ranking.
*   **number_of_community_malls**: The count of community-focused malls.
*   **number_of_community_winner_malls**: The count of high-performing "community winner" malls.
*   **number_of_key_focus_malls**: The count of shopping malls identified for key focus.
*   **population_scale_dense**: The density of the population scale within the sector.
*   **residential_area_dense**: The density of residential areas within the sector.
*   **office_building_dense**: The density of office buildings within the sector.
*   **commercial_area_dense**: The density of commercial areas within the sector.
*   **resident_population_dense**: The density of the resident population within the sector.
*   **office_population_dense**: The density of the office population within the sector.
*   **number_of_shops_dense**: The density of shops within the sector.
*   **catering_dense**: The density of catering establishments within the sector.
*   **retail_dense**: The density of retail establishments within the sector.
*   **hotel_dense**: The density of hotel establishments within the sector.
*   **transportation_station_dense**: The density of transportation stations within the sector.
*   **education_dense**: The density of educational facilities within the sector.
*   **leisure_and_entertainment_dense**: The density of leisure and entertainment venues within the sector.
*   **bus_station_cnt_dense**: The density of bus stations.
*   **subway_station_cnt_dense**: The density of subway stations.
*   **rentable_shops_dense**: The density of rentable shops.
*   **leisure_entertainment_stores_dense**: The density of leisure and entertainment stores.
*   **other_stores_dense**: The density of miscellaneous other stores.
*   **other_anchor_stores_dense**: The density of other major or anchor stores.
*   **home_appliance_stores_dense**: The density of home appliance stores.
*   **skincare_cosmetics_stores_dense**: The density of skincare and cosmetics stores.
*   **fashion_stores_dense**: The density of fashion stores.
*   **service_stores_dense**: The density of service-oriented stores.
*   **jewelry_stores_dense**: The density of jewelry stores.
*   **lifestyle_leisure_stores_dense**: The density of lifestyle and leisure stores.
*   **supermarket_convenience_stores_dense**: The density of supermarkets and convenience stores.
*   **catering_food_stores_dense**: The density of catering and food stores.
*   **residential_commercial_dense**: The density of commercial establishments within residential areas.
*   **office_building_commercial_dense**: The density of commercial establishments within office buildings.
*   **commercial_buildings_dense**: The density of dedicated commercial buildings.
*   **hypermarkets_dense**: The density of hypermarkets.
*   **department_stores_dense**: The density of department stores.
*   **shopping_centers_dense**: The density of shopping centers.
*   **hotel_commercial_dense**: The density of commercial establishments within hotels.
*   **third_tier_shopping_malls_in_business_district_dense**: The density of third-tier shopping malls within the business district.
*   **second_tier_shopping_malls_in_business_district_dense**: The density of second-tier shopping malls within the business district.
*   **city_winner_malls_dense**: The density of high-performing "city winner" malls.
*   **shopping_malls_with_street_facing_shops_dense**: The density of shopping malls featuring street-facing shops.
*   **unranked_malls_dense**: The density of shopping malls without a specific ranking.
*   **community_malls_dense**: The density of community-focused malls.
*   **community_winner_malls_dense**: The density of high-performing "community winner" malls.
*   **key_focus_malls_dense**: The density of shopping malls identified for key focus.
*   **transportation_facilities_service_bus_station_dense**: The density of bus stations.
*   **transportation_facilities_service_subway_station_dense**: The density of subway stations.
*   **transportation_facilities_service_airport_related_dense**: The density of airport-related facilities.
*   **transportation_facilities_service_port_terminal_dense**: The density of port or terminal facilities.
*   **transportation_facilities_service_train_station_dense**: The density of train stations.
*   **transportation_facilities_service_light_rail_station_dense**: The density of light rail stations.
*   **transportation_facilities_service_long_distance_bus_station_dense**: The density of long-distance bus stations.
*   **leisure_entertainment_entertainment_venue_game_arcade_dense**: The density of game arcades.
*   **leisure_entertainment_entertainment_venue_party_house_dense**: The density of party houses.
*   **leisure_entertainment_cultural_venue_cultural_palace_dense**: The density of cultural palaces.
*   **office_building_industrial_building_industrial_building_dense**: The density of industrial buildings used as office spaces.
*   **medical_health_dense**: The density of general medical and health facilities.
*   **medical_health_specialty_hospital_dense**: The density of specialty hospitals.
*   **medical_health_tcm_hospital_dense**: The density of Traditional Chinese Medicine (TCM) hospitals.
*   **medical_health_physical_examination_institution_dense**: The density of physical examination institutions.
*   **medical_health_veterinary_station_dense**: The density of veterinary stations.
*   **medical_health_pharmaceutical_healthcare_dense**: The density of pharmaceutical healthcare providers.
*   **medical_health_rehabilitation_institution_dense**: The density of rehabilitation institutions.
*   **medical_health_first_aid_center_dense**: The density of first aid centers.
*   **medical_health_blood_donation_station_dense**: The density of blood donation stations.
*   **medical_health_disease_prevention_institution_dense**: The density of disease prevention institutions.
*   **medical_health_general_hospital_dense**: The density of general hospitals.
*   **medical_health_clinic_dense**: The density of clinics.
*   **education_training_school_education_middle_school_dense**: The density of middle schools.
*   **education_training_school_education_primary_school_dense**: The density of primary schools.
*   **education_training_school_education_kindergarten_dense**: The density of kindergartens.
*   **education_training_school_education_research_institution_dense**: The density of research institutions.

**train/city_search_index.csv**

*   **month**: The month the search data was recorded.
*   **keyword**: The specific search term.
*   **source**: The origin or platform of the search data.
*   **search_volume**: The total number of searches for the keyword.

**train/city_indexes.csv**

*   **city_indicator_data_year**: The year to which the city indicator data pertains.
*   **year_end_registered_population_10k**: The registered population at year-end in tens of thousands.
*   **total_households_10k**: The total number of households in tens of thousands.
*   **year_end_resident_population_10k**: The permanent resident population at year-end in tens of thousands.
*   **year_end_total_employed_population_10k**: The total employed population at year-end in tens of thousands.
*   **year_end_urban_non_private_employees_10k**: The number of urban non-private unit employees at year-end in tens of thousands.
*   **private_individual_and_other_employees_10k**: The number of private, individual, and other employees in tens of thousands.
*   **private_individual_ratio**: The proportion of private and individual employees.
*   **national_year_end_total_population_10k**: The national total population at year-end in tens of thousands.
*   **resident_registered_ratio**: The ratio of permanent residents to registered population.
*   **under_18_10k**: The population under 18 years old in tens of thousands.
*   **18_60_years_10k**: The population aged 18 to 60 years old in tens of thousands.
*   **over_60_years_10k**: The population over 60 years old in tens of thousands.
*   **total**: The total population count.
*   **under_18_percent**: The percentage of the population under 18 years old.
*   **18_60_years_percent**: The percentage of the population aged 18 to 60 years old.
*   **over_60_years_percent**: The percentage of the population over 60 years old.
*   **gdp_100m**: The Gross Domestic Product (GDP) in 100 million yuan.
*   **primary_industry_100m**: The output value of the primary industry in 100 million yuan.
*   **secondary_industry_100m**: The output value of the secondary industry in 100 million yuan.
*   **tertiary_industry_100m**: The output value of the tertiary industry in 100 million yuan.
*   **gdp_per_capita_yuan**: The GDP per capita in yuan.
*   **national_gdp_100m**: The national GDP in 100 million yuan.
*   **national_economic_primacy**: An indicator of the city's economic dominance compared to the nation.
*   **national_population_share**: The city's share of the national population.
*   **gdp_population_ratio**: The ratio of the city's GDP primacy to its national population share.
*   **secondary_industry_development_gdp_share**: The share of the secondary industry in the GDP, indicating its development.
*   **tertiary_industry_development_gdp_share**: The share of the tertiary industry in the GDP, indicating its development.
*   **employed_population**: The total number of employed individuals.
*   **primary_industry_percent**: The percentage of the employed population in the primary industry.
*   **secondary_industry_percent**: The percentage of the employed population in the secondary industry.
*   **tertiary_industry_percent**: The percentage of the employed population in the tertiary industry.
*   **white_collar_service_vs_blue_collar_manufacturing_ratio**: The ratio of white-collar (service industry) to blue-collar (manufacturing industry) population.
*   **general_public_budget_revenue_100m**: The general public budget revenue in 100 million yuan.
*   **personal_income_tax_100m**: The personal income tax collected in 100 million yuan.
*   **per_capita_personal_income_tax_yuan**: The per capita personal income tax in yuan.
*   **general_public_budget_expenditure_100m**: The general public budget expenditure in 100 million yuan.
*   **total_retail_sales_of_consumer_goods_100m**: The total retail sales of consumer goods in 100 million yuan.
*   **retail_sales_growth_rate**: The growth rate of retail sales.
*   **urban_consumer_price_index_previous_year_100**: The urban consumer price index, with the previous year set as 100.
*   **annual_average_wage_urban_non_private_employees_yuan**: The annual average wage of urban non-private unit employees in yuan.
*   **annual_average_wage_urban_non_private_on_duty_employees_yuan**: The annual average wage of urban non-private on-duty employees in yuan.
*   **per_capita_disposable_income_absolute_yuan**: The absolute value of per capita disposable income in yuan.
*   **per_capita_disposable_income_index_previous_year_100**: The per capita disposable income index, with the previous year set as 100.
*   **engel_coefficient**: The Engel coefficient, indicating the proportion of income spent on food.
*   **per_capita_housing_area_sqm**: The per capita housing area in square meters.
*   **number_of_universities**: The total count of universities and colleges.
*   **university_students_10k**: The number of university students in tens of thousands.
*   **number_of_middle_schools**: The total count of middle schools.
*   **middle_school_students_10k**: The number of middle school students in tens of thousands.
*   **number_of_primary_schools**: The total count of primary schools.
*   **primary_school_students_10k**: The number of primary school students in tens of thousands.
*   **number_of_kindergartens**: The total count of kindergartens.
*   **kindergarten_students_10k**: The number of kindergarten students in tens of thousands.
*   **hospitals_health_centers**: The total count of hospitals and health centers.
*   **hospital_beds_10k**: The number of hospital beds in tens of thousands.
*   **health_technical_personnel_10k**: The number of health technical personnel in tens of thousands.
*   **doctors_10k**: The number of doctors in tens of thousands.
*   **road_length_km**: The total length of roads in kilometers.
*   **road_area_10k_sqm**: The total area of roads in 10,000 square meters.
*   **per_capita_urban_road_area_sqm**: The per capita urban road area in square meters.
*   **number_of_operating_bus_lines**: The total count of operating bus lines.
*   **operating_bus_line_length_km**: The total length of operating bus lines in kilometers.
*   **internet_broadband_access_subscribers_10k**: The number of internet broadband access subscribers in tens of thousands.
*   **internet_broadband_access_ratio**: The ratio of internet broadband access.
*   **number_of_industrial_enterprises_above_designated_size**: The total count of industrial enterprises above a designated size.
*   **total_current_assets_10k**: The total current assets in 10,000 yuan.
*   **total_fixed_assets_10k**: The total fixed assets in 10,000 yuan.
*   **main_business_taxes_and_surcharges_10k**: The main business taxes and surcharges in 10,000 yuan.
*   **total_fixed_asset_investment_10k**: The total fixed asset investment in 10,000 yuan.
*   **real_estate_development_investment_completed_10k**: The completed real estate development investment in 10,000 yuan.
*   **residential_development_investment_completed_10k**: The completed residential development investment in 10,000 yuan.
*   **science_expenditure_10k**: The expenditure on science in 10,000 yuan.
*   **education_expenditure_10k**: The expenditure on education in 10,000 yuan.
