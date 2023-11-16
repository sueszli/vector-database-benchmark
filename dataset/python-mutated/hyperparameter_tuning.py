import os
import scipy.sparse as sp
from skopt.space import Integer, Categorical, Real
from IPython.display import display
from Evaluation.Evaluator import EvaluatorHoldout
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Content, runHyperparameterSearch_Collaborative, runHyperparameterSearch_Hybrid
from Recommenders.DataIO import DataIO
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.FactorizationMachines.LightFMRecommender import LightFMCFRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.GraphBased.SideInfoRP3betaRecommender import SideInfoRP3betaRecommender
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender, PureSVDItemRecommender
from Recommenders.MatrixFactorization.NMFRecommender import NMFRecommender
from Recommenders.MatrixFactorization.SideInfoIALSRecommender import SideInfoIALSRecommender
from Recommenders.Neural.MultVAERecommender import MultVAERecommender, MultVAERecommender_OptimizerMask
from Recommenders.NonPersonalizedRecommender import TopPop
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
from Recommenders.SLIM.S_SLIMElasticNet_Hybrid_Recommender import S_SLIMElasticNet_Hybrid_Recommender
from src.Utils.dataset_splits import dataset_splits
from src.Utils.load_data import load_data
from src.Utils.write_submission import write_submission

def hyperparameter_tuning():
    if False:
        for i in range(10):
            print('nop')
    (URM, ICM_channel, ICM_event, ICM_genre, ICM_subgenre) = load_data('kaggle-data')
    icm_channel_train = sp.csr_matrix((ICM_channel['data'], (ICM_channel['row'], ICM_channel['col'])))
    icm_genre_train = sp.csr_matrix((ICM_genre['data'], (ICM_genre['row'], ICM_genre['col'])))
    icm_subgenre_train = sp.csr_matrix((ICM_subgenre['data'], (ICM_subgenre['row'], ICM_subgenre['col'])))
    icm_mixed_train = sp.hstack([icm_channel_train, icm_genre_train, icm_subgenre_train])
    (urm_train_validation, urm_train, urm_validation, urm_test) = dataset_splits(URM, validation_percentage=0.2, testing_percentage=0.2)
    icm_mixed_train *= 0.25122021
    urm_stacked_train = sp.vstack([urm_train, icm_mixed_train.T])
    urm_stacked_train_validation = sp.vstack([urm_train_validation, icm_mixed_train.T])
    evaluator_validation = EvaluatorHoldout(urm_validation, [10])
    evaluator_test = EvaluatorHoldout(urm_test, [10])
    recommender_class = S_SLIMElasticNet_Hybrid_Recommender
    hyperparameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)
    output_folder_path = 'result_experiments/'
    n_cases = 50
    n_random_starts = int(n_cases * 0.3)
    metric_to_optimize = 'MAP'
    cutoff_to_optimize = 10
    runHyperparameterSearch_Hybrid(recommender_class, URM_train=urm_train, ICM_object=icm_mixed_train, ICM_name='mixed', URM_train_last_test=urm_train_validation, n_cases=n_cases, n_random_starts=n_random_starts, resume_from_saved=True, save_model='best', evaluator_validation_earlystopping=evaluator_validation, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test, metric_to_optimize=metric_to_optimize, cutoff_to_optimize=cutoff_to_optimize, output_folder_path=output_folder_path, parallelizeKNN=False, allow_weighting=True)
if __name__ == '__main__':
    hyperparameter_tuning()