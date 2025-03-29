import pandas as pd

# df = pd.read_excel('data.xlsx')
# df = df.drop(columns=['source_id', 'engine_id'])
# df.to_csv('data.csv')

df = pd.read_csv('data.csv')
df = df[df['asset'] == 'BTCUSDT']
df.to_csv("data_btc.csv")