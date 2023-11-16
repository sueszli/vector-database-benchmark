from flaml import tune
from flaml.automl.model import LGBMEstimator
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
import ray
data = fetch_california_housing(return_X_y=False, as_frame=True)
(X, y) = (data.data, data.target)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, random_state=42)
X_train_ref = ray.put(X_train)
print(isinstance(X_train_ref, ray.ObjectRef))

def train_lgbm(config: dict) -> dict:
    if False:
        while True:
            i = 10
    params = LGBMEstimator(**config).params
    X_train = ray.get(X_train_ref)
    train_set = lightgbm.Dataset(X_train, y_train)
    model = lightgbm.train(params, train_set)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    return {'mse': mse}
flaml_lgbm_search_space = LGBMEstimator.search_space(X_train.shape)
config_search_space = {hp: space['domain'] for (hp, space) in flaml_lgbm_search_space.items()}
low_cost_partial_config = {hp: space['low_cost_init_value'] for (hp, space) in flaml_lgbm_search_space.items() if 'low_cost_init_value' in space}
points_to_evaluate = [{hp: space['init_value'] for (hp, space) in flaml_lgbm_search_space.items() if 'init_value' in space}]
analysis = tune.run(train_lgbm, metric='mse', mode='min', config=config_search_space, low_cost_partial_config=low_cost_partial_config, points_to_evaluate=points_to_evaluate, time_budget_s=3, num_samples=-1)
print(analysis.best_result)