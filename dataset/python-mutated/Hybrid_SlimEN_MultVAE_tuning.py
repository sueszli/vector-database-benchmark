import os
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from Recommenders.Neural.MultVAERecommender import MultVAERecommender
if __name__ == '__main__':
    import json
    from Utils.load_ICM import load_ICM
    from Utils.load_URM import load_URM
    from Recommenders.Hybrid.GeneralizedMergedHybridRecommender import GeneralizedMergedHybridRecommender
    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
    from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
    from bayes_opt import BayesianOptimization
    URM_all = load_URM('../../../kaggle-data/data_train.csv')
    ICM_all = load_ICM('../../../kaggle-data/data_ICM_subgenre.csv')
    URMs_train = []
    URMs_validation = []
    for k in range(1):
        (URM_train, URM_validation) = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.8)
        URMs_train.append(URM_train)
        URMs_validation.append(URM_validation)
    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10], verbose=False)
    MultVAE_recommenders = []
    SLIM_recommenders = []
    for index in range(len(URMs_train)):
        MultVAE_recommenders.append(MultVAERecommender(URM_train=URMs_train[index]))
        MultVAE_recommenders[index].fit(epochs=23, learning_rate=0.0015961592417506301, l2_reg=0.0006490776439334211, dropout=0.5791277104923426, total_anneal_steps=146901, anneal_cap=0.3054246574668649, batch_size=256)
        SLIM_recommenders.append(MultiThreadSLIM_SLIMElasticNetRecommender(URM_train=URMs_train[index], verbose=False))
        SLIM_recommenders[index].fit(topK=1139, l1_ratio=6.276359878274636e-05, alpha=0.12289267654724283)
        SLIM_recommenders[index].URM_train = URMs_train[index]
    tuning_params = {'alpha': (0, 1)}
    results = []

    def BO_func(alpha):
        if False:
            while True:
                i = 10
        recommenders = []
        for index in range(len(URMs_train)):
            recommender = GeneralizedMergedHybridRecommender(URM_train=URMs_train[index], recommenders=[MultVAE_recommenders[index], SLIM_recommenders[index]], verbose=False)
            recommender.fit(alphas=[alpha, 1 - alpha])
            recommenders.append(recommender)
        (result, _) = evaluator_validation.evaluateRecommender(recommender)
        results.append(result['MAP'])
        return sum(result['MAP']) / len(result['MAP'])
    optimizer = BayesianOptimization(f=BO_func, pbounds=tuning_params, verbose=5, random_state=5)
    optimizer.maximize(init_points=100, n_iter=50)
    recommender = GeneralizedMergedHybridRecommender(URM_train=URMs_train[0], recommenders=[MultVAE_recommenders[0], SLIM_recommenders[0]], verbose=False)
    recommender.fit()
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    with open('logs/FeatureCombined' + recommender.RECOMMENDER_NAME + '_logs.json', 'w') as json_file:
        json.dump(optimizer.max, json_file)