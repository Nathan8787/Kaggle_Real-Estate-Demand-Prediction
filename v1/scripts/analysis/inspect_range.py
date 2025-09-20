import pandas as pd
from pathlib import Path

files = {
    'new_house_transactions': 'new_house_transactions.csv',
    'new_house_transactions_nearby_sectors': 'new_house_transactions_nearby_sectors.csv',
    'pre_owned_house_transactions': 'pre_owned_house_transactions.csv',
    'pre_owned_house_transactions_nearby_sectors': 'pre_owned_house_transactions_nearby_sectors.csv',
    'land_transactions': 'land_transactions.csv',
    'land_transactions_nearby_sectors': 'land_transactions_nearby_sectors.csv',
    'sector_POI': 'sector_POI.csv',
    'city_search_index': 'city_search_index.csv',
    'city_indexes': 'city_indexes.csv',
}

for name, fname in files.items():
    path = Path(fname)
    df = pd.read_csv(path)
    if 'month' in df.columns:
        months = pd.to_datetime(df['month'], errors='coerce')
        print(f"{name}: min={months.min()}, max={months.max()}, count={months.notna().sum()}")
    elif 'city_indicator_data_year' in df.columns:
        years = pd.to_numeric(df['city_indicator_data_year'], errors='coerce')
        print(f"{name}: year_min={years.min()}, year_max={years.max()}, count={years.notna().sum()}")
    else:
        print(f"{name}: no month/year column")
