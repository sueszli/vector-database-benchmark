import bigdl.orca.data.pandas
from bigdl.orca.data.transformer import *
import numpy as np
path = '/home/ding/data/house_price/train.csv'
data_shard = bigdl.orca.data.pandas.read_csv(path, nullValue='NA')
null_cnt_pdf = data_shard.get_null_sum()
print(null_cnt_pdf)
sort_pdf = null_cnt_pdf.sort_values(by='total', ascending=False)
print(sort_pdf)
new_shards = data_shard.drop_missing_value()
new_cnt_pdf = new_shards.get_null_sum()
max_value = new_cnt_pdf['total'].max()
print(max_value)

def drop_data(df):
    if False:
        while True:
            i = 10
    df = df.drop(df[df['Id'] == 0].index)
    df = df.drop(df[df['Id'] == 1].index)
    return df
new_shards3 = new_shards.transform_shard(drop_data)

def generate_new_sale_price(df):
    if False:
        for i in range(10):
            print('nop')
    df['SalePrice'] = np.log(df['SalePrice'])
    return df
new_shards4 = new_shards3.transform_shard(generate_new_sale_price)

def generate_HasBsmt(df):
    if False:
        return 10
    df['HasBsmt'] = 0
    df.loc[df['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
    return df
new_shards5 = new_shards4.transform_shard(generate_HasBsmt)