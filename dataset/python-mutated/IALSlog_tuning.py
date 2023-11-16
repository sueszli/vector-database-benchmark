from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from src.Utils.load_ICM import load_ICM
from src.Utils.load_URM import load_URM
URM_all = load_URM('../../../in/data_train.csv')
ICM_all = load_ICM('../../../in/data_ICM_title_abstract.csv')
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
URMs_train = []
URMs_validation = []
for k in range(5):
    (URM_train, URM_validation) = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.8)
    URMs_train.append(URM_train)
    URMs_validation.append(URM_validation)
evaluator_validation = K_Fold_Evaluator_MAP(URMs_validation, cutoff_list=[10], verbose=False)
recommenders = []
for index in range(len(URMs_train)):
    recommenders.append(IALSRecommender(URM_train=URMs_train[index]))
tuning_params = {'num_factors': (50, 500), 'alpha': (1.0, 10.0), 'epsilon': (1.0, 10.0), 'reg': (0.5, 3.0)}
results = []

def BO_func(num_factors, alpha, epsilon, reg):
    if False:
        i = 10
        return i + 15
    for index in range(len(recommenders)):
        recommenders[index].fit(epochs=200, num_factors=int(num_factors), confidence_scaling='log', alpha=alpha, epsilon=epsilon, reg=reg, **{'epochs_min': 0, 'evaluator_object': evaluator_validation.evaluator_list[index], 'stop_on_validation': True, 'validation_every_n': 1, 'validation_metric': 'MAP', 'lower_validations_allowed': 3})
    result = evaluator_validation.evaluateRecommender(recommenders)
    results.append(result)
    return sum(result) / len(result)
from bayes_opt import BayesianOptimization
optimizer = BayesianOptimization(f=BO_func, pbounds=tuning_params, verbose=5, random_state=5)
optimizer.maximize(init_points=10, n_iter=8)
import json
with open('logs/' + recommenders[0].RECOMMENDER_NAME + '_log_logs.json', 'w') as json_file:
    json.dump(optimizer.max, json_file)