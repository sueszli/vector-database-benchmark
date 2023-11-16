import numpy as np
import scipy.sparse as sp
from matplotlib import pyplot, pyplot as plt

from Evaluation.Evaluator import EvaluatorHoldout
from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.GraphBased.SideInfoRP3betaRecommender import SideInfoRP3betaRecommender
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender, \
    MultiThreadSLIM_SLIMElasticNetRecommender
from Recommenders.SLIM.SLIM_BPR_Python import SLIM_BPR_Python
from Recommenders.SLIM.S_SLIMElasticNet_Hybrid_Recommender import S_SLIMElasticNet_Hybrid_Recommender
from src.Utils.dataset_splits import dataset_splits
from src.Utils.load_data import load_data


def hyperparameter_range_selection():
    ######################### DATA PREPARATION ###########################################

    # generate the dataframes for URM and ICMs
    URM, ICM_channel, ICM_event, ICM_genre, ICM_subgenre = load_data("kaggle-data")

    # turn the ICMs into matrices
    icm_event_train = sp.csr_matrix((ICM_event['data'], (ICM_event['row'], ICM_event['col'])))
    icm_channel_train = sp.csr_matrix((ICM_channel['data'], (ICM_channel['row'], ICM_channel['col'])))
    icm_genre_train = sp.csr_matrix((ICM_genre['data'], (ICM_genre['row'], ICM_genre['col'])))
    icm_subgenre_train = sp.csr_matrix((ICM_subgenre['data'], (ICM_subgenre['row'], ICM_subgenre['col'])))

    # concatenate the ICMs
    icm_mixed_train = sp.hstack([icm_channel_train, icm_genre_train, icm_subgenre_train])

    # split the URM df in training, validation and testing and turn it into a matrix
    urm_train_validation, urm_train, urm_validation, urm_test = dataset_splits(URM,
                                                                               validation_percentage=0.20,
                                                                               testing_percentage=0.20)

    # combine
    icm_subgenre_train *= 0.25122021
    urm_stacked_train = sp.vstack([urm_train, icm_subgenre_train.T])
    urm_stacked_train_validation = sp.vstack([urm_train_validation, icm_subgenre_train.T])

    ####################################### EVALUATORS #################################################

    # evaluator_validation = EvaluatorHoldout(urm_validation, [10])
    evaluator_test = EvaluatorHoldout(urm_test, [10])

    ########################### TRY DIFFERENT HYPERPARAMETERS ##########################################

    ################ TOPK ##################
    # x_tick = range(1, 20, 2)
    # y_tick = [True, False]
    # MAP_per_tick = []
    #
    # for tick in x_tick:
    #     icm_subgenre_train=icm_subgenre_train*x_tick
    #     urm_stacked_train = sp.vstack([urm_train, icm_subgenre_train.T])
    #
    #     recommender = ItemKNNCFRecommender(urm_stacked_train)
    #     recommender.fit(topK=601, shrink=89, similarity='tversky', normalize=True, tversky_alpha=0.0, tversky_beta=2.0)
    #     result_df, _ = evaluator_validation.evaluateRecommender(recommender)
    #
    #     tick_map = result_df.loc[10]["MAP"]
    #     MAP_per_tick.append(tick_map)
    #     print("MAP for tick {}: {}".format(tick, tick_map))
    #
    # pyplot.plot(x_tick, MAP_per_tick)
    # pyplot.ylabel('MAP')
    # pyplot.xlabel('TopK')
    # pyplot.show()
    #
    # max_MAP = max(MAP_per_tick)
    # max_index = MAP_per_tick.index(max_MAP)
    # max_tick = x_tick[max_index]
    #
    # print("Best num factors: {}".format(max_tick))

    ###################################### FINAL RESULTS ###############################################
    output_folder_path = "Models/"

    recommender = S_SLIMElasticNet_Hybrid_Recommender(urm_train_validation, icm_mixed_train)
    recommender.fit(topK=1827, l1_ratio=2.9307571870179977e-05, alpha=0.08093238323432947, ICM_weight=0.3574805669644016)

    result_df, _ = evaluator_test.evaluateRecommender(recommender)

    print("FINAL MAP: {}".format(result_df.loc[10]["MAP"]))


if __name__ == '__main__':
    hyperparameter_range_selection()
