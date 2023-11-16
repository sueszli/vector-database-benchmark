import os

import scipy.sparse as sp
from skopt.space import Integer, Categorical, Real
from IPython.display import display
from Evaluation.Evaluator import EvaluatorHoldout
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Content, \
    runHyperparameterSearch_Collaborative, runHyperparameterSearch_Hybrid
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
    ######################### DATA PREPARATION ###########################################

    # generate the dataframes for URM and ICMs
    URM, ICM_channel, ICM_event, ICM_genre, ICM_subgenre = load_data("kaggle-data")

    # turn the ICMs into matrices
    icm_channel_train = sp.csr_matrix((ICM_channel['data'], (ICM_channel['row'], ICM_channel['col'])))
    # icm_event_train = sp.csr_matrix((ICM_event['data'], (ICM_event['row'], ICM_event['col'])))
    icm_genre_train = sp.csr_matrix((ICM_genre['data'], (ICM_genre['row'], ICM_genre['col'])))
    icm_subgenre_train = sp.csr_matrix((ICM_subgenre['data'], (ICM_subgenre['row'], ICM_subgenre['col'])))

    # concatenate the ICMs
    icm_mixed_train = sp.hstack([icm_channel_train, icm_genre_train, icm_subgenre_train])

    # split the URM df in training, validation and testing and turn it into a matrix
    urm_train_validation, urm_train, urm_validation, urm_test = dataset_splits(URM,
                                                                               validation_percentage=0.20,
                                                                               testing_percentage=0.20)

    icm_mixed_train *= 0.25122021
    urm_stacked_train = sp.vstack([urm_train, icm_mixed_train.T])
    urm_stacked_train_validation = sp.vstack([urm_train_validation, icm_mixed_train.T])

    ############ TUNE THOSE HYPERPARAMETERS BABEH ##########################################

    # Step 1: Split the data and create the evaluator objects
    evaluator_validation = EvaluatorHoldout(urm_validation, [10])
    evaluator_test = EvaluatorHoldout(urm_test, [10])

    # Step 2: Define hyperparameter set for the desired model
    # hyperparameters_range_dictionary = {
    #     "topK": Integer(900, 1500),
    #     "l1_ratio": Real(low=1e-5, high=1e-2, prior='log-uniform'),
    #     "alpha": Real(low=1e-2, high=1.0, prior='uniform'),
    #     "ICM_weight": Real(low=1e-1, high=1e4, prior='log-uniform'),
    # }

    # Step 3: Create SearchBayesianSkopt object, providing the desired recommender class and evaluator objects
    recommender_class = S_SLIMElasticNet_Hybrid_Recommender

    hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                               evaluator_validation=evaluator_validation,
                                               evaluator_test=evaluator_test)

    # Step 4: Provide the data needed to create an instance of the model, one trained only on URM_train, the other on URM_train_validation
    # recommender_input_args = SearchInputRecommenderArgs(
    #     CONSTRUCTOR_POSITIONAL_ARGS=[urm_train, icm_mixed_train],  # For a CBF model simply put [URM_train, ICM_train]
    #     CONSTRUCTOR_KEYWORD_ARGS={},
    #     FIT_POSITIONAL_ARGS=[],
    #     FIT_KEYWORD_ARGS={}
    # )
    # recommender_input_args_last_test = SearchInputRecommenderArgs(
    #     CONSTRUCTOR_POSITIONAL_ARGS=[urm_train_validation, icm_mixed_train],  # CBF: [URM, ICM], CF: [URM]
    #     CONSTRUCTOR_KEYWORD_ARGS={},
    #     FIT_POSITIONAL_ARGS=[],
    #     FIT_KEYWORD_ARGS={}
    # )

    # Step 5: Create a result folder and select the number of cases (50 with 30% random is a good number)
    output_folder_path = "result_experiments/"
    n_cases = 50
    n_random_starts = int(n_cases * 0.3)
    metric_to_optimize = "MAP"
    cutoff_to_optimize = 10

    # # general?

    # hyperparameterSearch.search(recommender_input_args,
    #                             recommender_input_args_last_test=recommender_input_args_last_test,
    #                             hyperparameter_search_space=hyperparameters_range_dictionary,
    #                             n_cases=n_cases,
    #                             n_random_starts=n_random_starts,
    #                             save_model="best",
    #                             output_folder_path=output_folder_path,  # Where to save the results
    #                             output_file_name_root=recommender_class.RECOMMENDER_NAME,  # How to call the files
    #                             metric_to_optimize=metric_to_optimize,
    #                             cutoff_to_optimize=cutoff_to_optimize,
    #                             )

    # runHyperparameterSearch_Collaborative(recommender_class,
    #                                       URM_train=urm_stacked_train,
    #                                       URM_train_last_test=urm_stacked_train_validation,
    #                                       metric_to_optimize=metric_to_optimize,
    #                                       cutoff_to_optimize=cutoff_to_optimize,
    #                                       evaluator_validation=evaluator_validation,
    #                                       evaluator_validation_earlystopping=evaluator_validation,
    #                                       evaluator_test=evaluator_test,
    #                                       output_folder_path=output_folder_path,
    #                                       allow_bias_URM=True,
    #                                       resume_from_saved=True,
    #                                       save_model="best",
    #                                       n_cases=n_cases,
    #                                       n_random_starts=n_random_starts)

    # runHyperparameterSearch_Content(recommender_class,
    #                                 URM_train=urm_train,
    #                                 ICM_object=icm_mixed_train,
    #                                 ICM_name="mixed",
    #                                 URM_train_last_test=urm_train_validation,
    #                                 metric_to_optimize=metric_to_optimize,
    #                                 cutoff_to_optimize=cutoff_to_optimize,
    #                                 evaluator_validation=evaluator_validation,
    #                                 evaluator_test=evaluator_test,
    #                                 output_folder_path=output_folder_path,
    #                                 resume_from_saved=True,
    #                                 save_model="best",
    #                                 n_cases=n_cases,
    #                                 n_random_starts=n_random_starts,
    #                                 evaluate_on_test="best",
    #                                 max_total_time=None,
    #                                 parallelizeKNN=True,
    #                                 allow_weighting=True, allow_bias_ICM=False,
    #                                 similarity_type_list=None
    #                                 )

    runHyperparameterSearch_Hybrid(recommender_class,
                                   URM_train=urm_train,
                                   ICM_object=icm_mixed_train,
                                   ICM_name="mixed",
                                   URM_train_last_test=urm_train_validation,
                                   n_cases=n_cases,
                                   n_random_starts=n_random_starts,
                                   resume_from_saved=True,
                                   save_model="best",
                                   evaluator_validation_earlystopping=evaluator_validation,
                                   evaluator_validation=evaluator_validation,
                                   evaluator_test=evaluator_test,
                                   metric_to_optimize=metric_to_optimize,
                                   cutoff_to_optimize=cutoff_to_optimize,
                                   output_folder_path=output_folder_path,
                                   parallelizeKNN=False,
                                   allow_weighting=True)

    ################### VISUALIZE RESULTS FOR HYPERPARAMETER SEARCH ########################

    # data_loader = DataIO(folder_path=output_folder_path)
    # search_metadata = data_loader.load_data(
    #     recommender_class.RECOMMENDER_NAME + "_metadata.zip")
    #
    # search_metadata.keys()
    #
    # hyperparameters_df = search_metadata["hyperparameters_df"]
    # display(hyperparameters_df)
    #
    # result_on_validation_df = search_metadata["result_on_validation_df"]
    # display(result_on_validation_df)
    #
    # result_best_on_test = search_metadata["result_on_last"]
    # display(result_best_on_test)
    #
    # print("FINAL MAP: {}".format(result_best_on_test.loc[10]["MAP"]))
    #
    # best_hyperparameters = search_metadata["hyperparameters_best"]
    # display(best_hyperparameters)


if __name__ == '__main__':
    hyperparameter_tuning()
