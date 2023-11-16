from flaml import tune
from flaml.automl.model import LGBMEstimator
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
data = fetch_california_housing(return_X_y=False, as_frame=True)
(df, X, y) = (data.frame, data.data, data.target)
(df_train, _, X_train, X_test, _, y_test) = train_test_split(df, X, y, test_size=0.33, random_state=42)
csv_file_name = 'test/housing.csv'
df_train.to_csv(csv_file_name, index=False)

def train_lgbm(config: dict) -> dict:
    if False:
        return 10
    params = LGBMEstimator(**config).params
    train_set = lightgbm.Dataset(csv_file_name, params={'label_column': 'name:MedHouseVal', 'header': True})
    model = lightgbm.train(params, train_set)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    return {'mse': mse}

def test_tune_lgbm_csv():
    if False:
        while True:
            i = 10
    flaml_lgbm_search_space = LGBMEstimator.search_space(X_train.shape)
    config_search_space = {hp: space['domain'] for (hp, space) in flaml_lgbm_search_space.items()}
    low_cost_partial_config = {hp: space['low_cost_init_value'] for (hp, space) in flaml_lgbm_search_space.items() if 'low_cost_init_value' in space}
    points_to_evaluate = [{hp: space['init_value'] for (hp, space) in flaml_lgbm_search_space.items() if 'init_value' in space}]
    analysis = tune.run(train_lgbm, metric='mse', mode='min', config=config_search_space, low_cost_partial_config=low_cost_partial_config, points_to_evaluate=points_to_evaluate, time_budget_s=3, num_samples=-1)
    print(analysis.best_result)
if __name__ == '__main__':
    test_tune_lgbm_csv()