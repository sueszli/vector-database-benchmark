from sklearn import datasets
from sklearn.model_selection import train_test_split
from xgboost_ray import RayDMatrix, RayParams, train
num_cpus_per_actor = 1
num_actors = 1

def train_model(config):
    if False:
        i = 10
        return i + 15
    (data, labels) = datasets.load_breast_cancer(return_X_y=True)
    (train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.25)
    train_set = RayDMatrix(train_x, train_y)
    test_set = RayDMatrix(test_x, test_y)
    evals_result = {}
    bst = train(params=config, dtrain=train_set, evals=[(test_set, 'eval')], evals_result=evals_result, verbose_eval=False, ray_params=RayParams(num_actors=num_actors, cpus_per_actor=num_cpus_per_actor))
    bst.save_model('model.xgb')

def load_best_model(best_logdir):
    if False:
        for i in range(10):
            print('nop')
    import xgboost as xgb
    import os
    best_bst = xgb.Booster()
    best_bst.load_model(os.path.join(best_logdir, 'model.xgb'))
    return best_bst

def main():
    if False:
        for i in range(10):
            print('nop')
    from ray import tune
    config = {'tree_method': 'approx', 'objective': 'binary:logistic', 'eval_metric': ['logloss', 'error'], 'eta': tune.loguniform(0.0001, 0.1), 'subsample': tune.uniform(0.5, 1.0), 'max_depth': tune.randint(1, 9)}
    analysis = tune.run(train_model, config=config, metric='eval-error', mode='min', num_samples=4, resources_per_trial=RayParams(num_actors=num_actors, cpus_per_actor=num_cpus_per_actor).get_tune_resources())
    best_bst = load_best_model(analysis.best_trial.local_path)
    _ = best_bst
    accuracy = 1.0 - analysis.best_result['eval-error']
    print(f'Best model parameters: {analysis.best_config}')
    print(f'Best model total accuracy: {accuracy:.4f}')
if __name__ == '__main__':
    main()