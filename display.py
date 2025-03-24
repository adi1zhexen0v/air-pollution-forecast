import pandas as pd

df = pd.read_csv('data/processed/2025-03-24_15-07-23.csv')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print(df)
print("Shape:" + str(df.shape))

