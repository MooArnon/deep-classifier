from deep_classifier.fe import ClassifierFE
import datetime

import polars as pl


def huge_minutely_df():
    """
    A small Polars DataFrame of minute-granularity data
    over 10 minutes, with prices that increase by 1 each minute
    so we can easily verify calculations of lag, rolling mean, etc.
    """
    start = datetime.datetime(2023, 1, 1, 0, 0, 0)
    rows = []
    
    # We'll have 10 minutes of data, from 00:00 to 00:09
    for i in range(30):
        minute_time = start + datetime.timedelta(minutes=i)
        rows.append({
            "datetime": minute_time.isoformat(),  # or str
            "price": float(100 + i*10),  # 100, 101, 102, ... for easy checking
        })
    
    return pl.DataFrame(rows)

df = pl.read_excel("data.xlsx")

fe = ClassifierFE(
    control_column="scraped_timestamp",
    target_column="close",
    fe_name_list=[],      # We'll set this in each test
    unused_feature=[]
)
"""
fe.fe_name_list = ["rsi_df", "percent_change_df", "macd_df", "percent_price_ema_df"]


df = df.drop(
    [
        'engine_id',
        'source_id'
    ]
)
# For demonstration, assume your rsi_df uses period=3 internally.
df_transformed = fe.transform_df(df, keep_target=True)
df_transformed.to_pandas().to_csv('result.csv')
"""

df = fe.add_binary_label(df)
print(df)
