import os

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender

if __name__ == '__main__':

    import json

    from Utils.load_ICM import load_ICM
    from Utils.load_URM import load_URM

    from Recommenders.Hybrid.GeneralizedMergedHybridRecommender import GeneralizedMergedHybridRecommender
    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
    from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender

    from Data_manager.split_functions.split_train_validation_random_holdout import \
        split_train_in_two_percentage_global_sample

    from bayes_opt import BayesianOptimization

    URM_all = load_URM("../../../kaggle-data/data_train.csv")
    ICM_all = load_ICM("../../../kaggle-data/data_ICM_subgenre.csv")

    URMs_train = []
    URMs_validation = []

    # URMs appended more than once produce more accurate evaluations with K_Fold_Evaluator
    for k in range(1):
        URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)
        URMs_train.append(URM_train)
        URMs_validation.append(URM_validation)

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10], verbose=False)

    # append ICMs the same number of times as the URMs
    # ICMs_combined = []
    # for URM in URMs_train:
    #     ICMs_combined.append(combine(ICM=ICM_all, URM=URM))

    # you can append as many recommenders as you wish in the appropriate list
    UserKNN_recommenders = []
    SLIM_recommenders = []
    # rp3betaCBF_recommenders = []

    # append and fit your recommender to its list a number of times equal to the number of times you've appended your URM
    for index in range(len(URMs_train)):
        UserKNN_recommenders.append(
            UserKNNCFRecommender(
                URM_train=URMs_train[index],
                verbose=True
            )
        )

        UserKNN_recommenders[index].fit(
            topK=497,
            shrink=0,
            similarity='tversky',
            normalize=True,
            tversky_alpha=2.0,
            tversky_beta=1.0492068509506025
        )

        SLIM_recommenders.append(
            MultiThreadSLIM_SLIMElasticNetRecommender(
                URM_train=URMs_train[index],
                verbose=False
            )
        )

        SLIM_recommenders[index].fit(
            topK=1139,
            l1_ratio=6.276359878274636e-05,
            alpha=0.12289267654724283
        )

        SLIM_recommenders[index].URM_train = URMs_train[index]

    tuning_params = {
        "alpha": (0, 1),
        # "beta": (0, 1),
    }

    results = []


    def BO_func(
            alpha,
            # beta
    ):
        recommenders = []

        #
        for index in range(len(URMs_train)):
            recommender = GeneralizedMergedHybridRecommender(
                URM_train=URMs_train[index],
                recommenders=[
                    UserKNN_recommenders[index],
                    SLIM_recommenders[index]
                ],
                verbose=False
            )

            recommender.fit(
                alphas=[
                    alpha,  # originally alpha * beta,
                    # alpha*(1-beta),
                    1-alpha
                ]
            )

            recommenders.append(recommender)

        result, _ = evaluator_validation.evaluateRecommender(recommender)
        results.append(result["MAP"])
        return sum(result["MAP"]) / len(result["MAP"])


    optimizer = BayesianOptimization(
        f=BO_func,
        pbounds=tuning_params,
        verbose=5,
        random_state=5,
    )

    optimizer.maximize(
        init_points=100,  # default 100
        n_iter=50,  # default 50
    )

    recommender = GeneralizedMergedHybridRecommender(
        URM_train=URMs_train[0],
        recommenders=[
            UserKNN_recommenders[0],
            SLIM_recommenders[0]
        ],
        verbose=False
    )
    recommender.fit()

    # If directory does not exist, create
    if not os.path.exists("./logs"):
        os.makedirs("./logs")

    with open("logs/FeatureCombined" + recommender.RECOMMENDER_NAME + "_logs.json", 'w') as json_file:
        json.dump(optimizer.max, json_file)
