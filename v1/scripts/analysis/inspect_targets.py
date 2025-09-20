import pandas as pd

panel = pd.read_parquet('data/processed/panel.parquet')
train = panel[panel['month'] <= pd.Timestamp('2024-07-01')]
print(train['target'].describe())
future = panel[panel['month'] >= pd.Timestamp('2024-08-01')]
print('Future target unique values:', future['target'].dropna().unique()[:5])
