"""
Created on 2017-12-11
Update  on 2017-12-11
Author: Usernametwo
Github: https://github.com/apachecn/kaggle
"""
import time
import pandas as pd
from sklearn.linear_model import Ridge
import os.path
data_dir = '/opt/data/kaggle/getting-started/house-prices'

def opencsv():
    if False:
        return 10
    df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    return (df_train, df_test)

def saveResult(result):
    if False:
        return 10
    result.to_csv(os.path.join(data_dir, 'submission.csv'), sep=',', encoding='utf-8')

def ridgeRegression(trainData, trainLabel, df_test):
    if False:
        for i in range(10):
            print('nop')
    ridge = Ridge(alpha=10.0)
    ridge.fit(trainData, trainLabel)
    predict = ridge.predict(df_test)
    pred_df = pd.DataFrame(predict, index=df_test['Id'], columns=['SalePrice'])
    return pred_df

def dataProcess(df_train, df_test):
    if False:
        while True:
            i = 10
    trainLabel = df_train['SalePrice']
    df = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    df.dropna(axis=1, inplace=True)
    df = pd.get_dummies(df)
    trainData = df[:df_train.shape[0]]
    test = df[df_train.shape[0]:]
    return (trainData, trainLabel, test)

def Regression_ridge():
    if False:
        return 10
    start_time = time.time()
    (df_train, df_test) = opencsv()
    print('load data finish')
    stop_time_l = time.time()
    print('load data time used:%f' % (stop_time_l - start_time))
    (train_data, trainLabel, df_test) = dataProcess(df_train, df_test)
    result = ridgeRegression(train_data, trainLabel, df_test)
    saveResult(result)
    print('finish!')
    stop_time_r = time.time()
    print('classify time used:%f' % (stop_time_r - start_time))
if __name__ == '__main__':
    Regression_ridge()