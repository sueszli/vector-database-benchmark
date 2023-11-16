"""
Created on Sat Dec 16 17:12:06 2017

@author: Administrator
"""
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from scipy.stats import boxcox
from sklearn.linear_model import Ridge
import warnings
import os.path
warnings.filterwarnings('ignore')
data_dir = '/opt/data/kaggle/getting-started/house-prices'
mapper = {'LandSlope': {'Gtl': 'Gtl', 'Mod': 'unGtl', 'Sev': 'unGtl'}, 'LotShape': {'Reg': 'Reg', 'IR1': 'IR1', 'IR2': 'other', 'IR3': 'other'}, 'RoofMatl': {'ClyTile': 'other', 'CompShg': 'CompShg', 'Membran': 'other', 'Metal': 'other', 'Roll': 'other', 'Tar&Grv': 'Tar&Grv', 'WdShake': 'WdShake', 'WdShngl': 'WdShngl'}, 'Heating': {'GasA': 'GasA', 'GasW': 'GasW', 'Grav': 'Grav', 'Floor': 'other', 'OthW': 'other', 'Wall': 'Wall'}, 'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'KitchenQual': {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}}
to_drop = ['Id', 'Street', 'Utilities', 'Condition2', 'PoolArea', 'PoolQC', 'Fence', 'YrSold', 'MoSold', 'BsmtHalfBath', 'BsmtFinSF2', 'GarageQual', 'MiscVal', 'EnclosedPorch', '3SsnPorch', 'GarageArea', 'TotRmsAbvGrd', 'GarageYrBlt', 'BsmtFinType2', 'BsmtUnfSF', 'GarageCond', 'GarageFinish', 'FireplaceQu', 'BsmtCond', 'BsmtQual', 'Alley']
"\ndata['house_remod']:  重新装修的年份与房建年份的差值\ndata['livingRate']:   LotArea查了下是地块面积,这个特征是居住面积/地块面积*总体评价\ndata['lot_area']:    LotFrontage是与房子相连的街道大小,现在想了下把GrLivArea换成LotArea会不会好点?\ndata['room_area']:   房间数/居住面积\ndata['fu_room']:    带有浴室的房间占总房间数的比例\ndata['gr_room']:    卧室与房间数的占比\n"

def create_feature(data):
    if False:
        print('Hello World!')
    hBsmt_index = data.index[data['TotalBsmtSF'] > 0]
    data['HaveBsmt'] = 0
    data.loc[hBsmt_index, 'HaveBsmt'] = 1
    data['house_remod'] = data['YearRemodAdd'] - data['YearBuilt']
    data['livingRate'] = data['GrLivArea'] / data['LotArea'] * data['OverallCond']
    data['lot_area'] = data['LotFrontage'] / data['GrLivArea']
    data['room_area'] = data['TotRmsAbvGrd'] / data['GrLivArea']
    data['fu_room'] = data['FullBath'] / data['TotRmsAbvGrd']
    data['gr_room'] = data['BedroomAbvGr'] / data['TotRmsAbvGrd']

def processing(data):
    if False:
        i = 10
        return i + 15
    create_feature(data)
    data.drop(to_drop, axis=1, inplace=True)
    fill_none = ['MasVnrType', 'BsmtExposure', 'GarageType', 'MiscFeature']
    for col in fill_none:
        data[col].fillna('None', inplace=True)
    na_col = data.dtypes[data.isnull().any()]
    for col in na_col.index:
        if na_col[col] != 'object':
            med = data[col].median()
            data[col].fillna(med, inplace=True)
        else:
            mode = data[col].mode()[0]
            data[col].fillna(mode, inplace=True)
    numeric_col = data.skew().index
    zero_col = data.columns[data.isin([0]).any()]
    for col in numeric_col:
        if len(pd.value_counts(data[col])) <= 10:
            continue
        if col in zero_col:
            trans_data = data[data > 0][col]
            before = abs(trans_data.skew())
            (cox, _) = boxcox(trans_data)
            log_after = abs(Series(cox).skew())
            if log_after < before:
                data.loc[trans_data.index, col] = cox
        else:
            before = abs(data[col].skew())
            (cox, _) = boxcox(data[col])
            log_after = abs(Series(cox).skew())
            if log_after < before:
                data.loc[:, col] = cox
    for (col, mapp) in mapper.items():
        data.loc[:, col] = data[col].map(mapp)
df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
test_ID = df_test['Id']
GrLivArea_outlier = set(df_train.index[(df_train['SalePrice'] < 200000) & (df_train['GrLivArea'] > 4000)])
LotFrontage_outlier = set(df_train.index[df_train['LotFrontage'] > 300])
df_train.drop(LotFrontage_outlier | GrLivArea_outlier, inplace=True)
df_train.reset_index(drop=True, inplace=True)
prices = np.log1p(df_train.loc[:, 'SalePrice'])
df_train.drop(['SalePrice'], axis=1, inplace=True)
all_data = pd.concat([df_train, df_test])
all_data.reset_index(drop=True, inplace=True)
processing(all_data)
dummy = pd.get_dummies(all_data, drop_first=True)
ridge = Ridge(6)
ridge.fit(dummy.iloc[:prices.shape[0], :], prices)
result = np.expm1(ridge.predict(dummy.iloc[prices.shape[0]:, :]))
pre = DataFrame(result, columns=['SalePrice'])
prediction = pd.concat([test_ID, pre], axis=1)
prediction.to_csv(os.path.join(data_dir, 'submission_1.csv'), index=False)