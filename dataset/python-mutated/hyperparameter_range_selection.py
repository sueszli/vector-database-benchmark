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
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender, MultiThreadSLIM_SLIMElasticNetRecommender
from Recommenders.SLIM.SLIM_BPR_Python import SLIM_BPR_Python
from Recommenders.SLIM.S_SLIMElasticNet_Hybrid_Recommender import S_SLIMElasticNet_Hybrid_Recommender
from src.Utils.dataset_splits import dataset_splits
from src.Utils.load_data import load_data

def hyperparameter_range_selection():
    if False:
        return 10
    (URM, ICM_channel, ICM_event, ICM_genre, ICM_subgenre) = load_data('kaggle-data')
    icm_event_train = sp.csr_matrix((ICM_event['data'], (ICM_event['row'], ICM_event['col'])))
    icm_channel_train = sp.csr_matrix((ICM_channel['data'], (ICM_channel['row'], ICM_channel['col'])))
    icm_genre_train = sp.csr_matrix((ICM_genre['data'], (ICM_genre['row'], ICM_genre['col'])))
    icm_subgenre_train = sp.csr_matrix((ICM_subgenre['data'], (ICM_subgenre['row'], ICM_subgenre['col'])))
    icm_mixed_train = sp.hstack([icm_channel_train, icm_genre_train, icm_subgenre_train])
    (urm_train_validation, urm_train, urm_validation, urm_test) = dataset_splits(URM, validation_percentage=0.2, testing_percentage=0.2)
    icm_subgenre_train *= 0.25122021
    urm_stacked_train = sp.vstack([urm_train, icm_subgenre_train.T])
    urm_stacked_train_validation = sp.vstack([urm_train_validation, icm_subgenre_train.T])
    evaluator_test = EvaluatorHoldout(urm_test, [10])
    output_folder_path = 'Models/'
    recommender = S_SLIMElasticNet_Hybrid_Recommender(urm_train_validation, icm_mixed_train)
    recommender.fit(topK=1827, l1_ratio=2.9307571870179977e-05, alpha=0.08093238323432947, ICM_weight=0.3574805669644016)
    (result_df, _) = evaluator_test.evaluateRecommender(recommender)
    print('FINAL MAP: {}'.format(result_df.loc[10]['MAP']))
if __name__ == '__main__':
    hyperparameter_range_selection()