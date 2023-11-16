from bigdl.ppml.fl import *
from bigdl.ppml.fl.algorithms.fgboost_regression import FGBoostRegression
import pandas as pd
from bigdl.ppml.fl.algorithms.psi import PSI
import click

def preprocess(train_dataset, test_dataset):
    if False:
        return 10
    categorical_features_all = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
    categorical_features_party = list(set(train_dataset.columns) & set(categorical_features_all))
    every_column_non_categorical = [col for col in train_dataset.columns if col not in categorical_features_party and col not in ['Id']]
    numeric_feats = train_dataset[every_column_non_categorical].dtypes[train_dataset.dtypes != 'object'].index
    train_dataset[numeric_feats] = np.log1p(train_dataset[numeric_feats])
    every_column_non_categorical = [col for col in test_dataset.columns if col not in categorical_features_party and col not in ['Id']]
    numeric_feats = test_dataset[every_column_non_categorical].dtypes[test_dataset.dtypes != 'object'].index
    test_dataset[numeric_feats] = np.log1p(test_dataset[numeric_feats])
    categorical_features_with_nan_all = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish']
    categorical_features_with_nan_party = list(set(train_dataset.columns) & set(categorical_features_with_nan_all))
    numeric_features_with_nan_all = ['LotFrontage', 'GarageYrBlt']
    numeric_features_with_nan_party = list(set(train_dataset.columns) & set(numeric_features_with_nan_all))

    def ConvertNaNToNAString(data, columnList):
        if False:
            while True:
                i = 10
        for x in columnList:
            data[x] = str(data[x])

    def FillNaWithMean(data, columnList):
        if False:
            return 10
        for x in columnList:
            data[x] = data[x].fillna(data[x].mean())
    ConvertNaNToNAString(train_dataset, categorical_features_with_nan_party)
    ConvertNaNToNAString(test_dataset, categorical_features_with_nan_party)
    FillNaWithMean(train_dataset, numeric_features_with_nan_party)
    FillNaWithMean(test_dataset, numeric_features_with_nan_party)
    train_dataset = pd.get_dummies(train_dataset, columns=categorical_features_party)
    test_dataset = pd.get_dummies(test_dataset, columns=categorical_features_party)
    every_column_except_y = [col for col in train_dataset.columns if col not in ['SalePrice', 'Id']]
    y = train_dataset[['SalePrice']] if 'SalePrice' in train_dataset else None
    return (train_dataset[every_column_except_y], y, test_dataset)

@click.command()
@click.option('--load_model', default=False)
def run_client(load_model):
    if False:
        print('Hello World!')
    client_id = 2
    init_fl_context(client_id)
    df_train = pd.read_csv('./data/house-prices-train-2.csv')
    df_train['Id'] = df_train['Id'].astype(str)
    df_test = pd.read_csv('./data/house-prices-test-2.csv')
    df_test['Id'] = df_test['Id'].astype(str)
    psi = PSI()
    intersection = psi.get_intersection(list(df_train['Id']))
    df_train = df_train[df_train['Id'].isin(intersection)]
    (x, y, x_test) = preprocess(df_train, df_test)
    if load_model:
        loaded = FGBoostRegression.load_model('/tmp/fgboost_model_2.json')
        loaded.fit(x, y, feature_columns=x.columns, label_columns=y.columns, num_round=10)
    else:
        fgboost_regression = FGBoostRegression()
        fgboost_regression.fit(x, y, feature_columns=x.columns, label_columns=y.columns, num_round=10)
        fgboost_regression.save_model('/tmp/fgboost_model_2.json')
        loaded = FGBoostRegression.load_model('/tmp/fgboost_model_2.json')
    result = loaded.predict(x_test, feature_columns=x_test.columns)
    for i in range(5):
        print(f'{i}-th result of FGBoost predict: {result[i]}')
if __name__ == '__main__':
    run_client()