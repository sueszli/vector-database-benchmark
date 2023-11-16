import os
import scipy.sparse as sp
from skopt.space import Integer, Categorical, Real
from IPython.display import display
from Evaluation.Evaluator import EvaluatorHoldout
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Content, runHyperparameterSearch_Collaborative, runHyperparameterSearch_Hybrid
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.Hybrids.ScoresHybridRecommender import ScoresHybridRecommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.MatrixFactorization.SideInfoIALSRecommender import SideInfoIALSRecommender
from Recommenders.Neural.MultVAERecommender import MultVAERecommender, MultVAERecommender_OptimizerMask
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
from Recommenders.SLIM.S_SLIMElasticNet_Hybrid_Recommender import S_SLIMElasticNet_Hybrid_Recommender
from src.Utils.dataset_splits import dataset_splits
from src.Utils.load_data import load_data
from src.Utils.write_submission import write_submission

def linear_combination():
    if False:
        print('Hello World!')
    (URM, ICM_channel, ICM_event, ICM_genre, ICM_subgenre) = load_data('kaggle-data')
    icm_channel_train = sp.csr_matrix((ICM_channel['data'], (ICM_channel['row'], ICM_channel['col'])))
    icm_genre_train = sp.csr_matrix((ICM_genre['data'], (ICM_genre['row'], ICM_genre['col'])))
    icm_subgenre_train = sp.csr_matrix((ICM_subgenre['data'], (ICM_subgenre['row'], ICM_subgenre['col'])))
    icm_mixed_train = sp.hstack([icm_channel_train, icm_genre_train, icm_subgenre_train])
    (urm_train_validation, urm_train, urm_validation, urm_test) = dataset_splits(URM, validation_percentage=0.2, testing_percentage=0.2)
    icm_mixed_train *= 0.25122021
    urm_stacked_mix_train = sp.vstack([urm_train, icm_mixed_train.T])
    urm_stacked_mix_train_validation = sp.vstack([urm_train_validation, icm_mixed_train.T])
    icm_subgenre_train *= 100.0
    urm_stacked_sub_train = sp.vstack([urm_train, icm_subgenre_train.T])
    urm_stacked_sub_train_validation = sp.vstack([urm_train_validation, icm_subgenre_train.T])
    output_folder_path = 'Models/'
    rec1 = MultVAERecommender_OptimizerMask(urm_stacked_mix_train_validation)
    rec1.load_model(output_folder_path + 'MultVAE(0.2313)/', file_name=rec1.RECOMMENDER_NAME + '_best_model_last.zip')
    print('Rec 1 is ready!')
    rec2 = MultiThreadSLIM_SLIMElasticNetRecommender(urm_stacked_mix_train_validation)
    rec2.load_model(output_folder_path + 'S_SLIM_ElasticNet/', file_name=rec2.RECOMMENDER_NAME + '_best_model_last.zip')
    print('Rec 2 is ready!')
    evaluator_validation = EvaluatorHoldout(urm_validation, [10])
    evaluator_test = EvaluatorHoldout(urm_test, [10])
    (result_df, _) = evaluator_test.evaluateRecommender(rec1)
    print('{} FINAL MAP: {}'.format(rec1.RECOMMENDER_NAME, result_df.loc[10]['MAP']))
    (result_df, _) = evaluator_test.evaluateRecommender(rec2)
    print('{} FINAL MAP: {}'.format(rec2.RECOMMENDER_NAME, result_df.loc[10]['MAP']))
    hyperparameters_range_dictionary = {'alpha': Real(low=0.0, high=1.0, prior='uniform')}
    recommender_class = ScoresHybridRecommender
    hyperparameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_test, evaluator_test=evaluator_test)
    recommender_input_args = SearchInputRecommenderArgs(CONSTRUCTOR_POSITIONAL_ARGS=[urm_train_validation, rec1, rec2], CONSTRUCTOR_KEYWORD_ARGS={}, FIT_POSITIONAL_ARGS=[], FIT_KEYWORD_ARGS={})
    recommender_input_args_last_test = SearchInputRecommenderArgs(CONSTRUCTOR_POSITIONAL_ARGS=[urm_train_validation, rec1, rec2], CONSTRUCTOR_KEYWORD_ARGS={}, FIT_POSITIONAL_ARGS=[], FIT_KEYWORD_ARGS={})
    output_folder_path = 'result_experiments/'
    n_cases = 100
    n_random_starts = int(n_cases * 0.3)
    metric_to_optimize = 'MAP'
    cutoff_to_optimize = 10
    hyperparameterSearch.search(recommender_input_args, recommender_input_args_last_test=recommender_input_args_last_test, hyperparameter_search_space=hyperparameters_range_dictionary, n_cases=n_cases, n_random_starts=n_random_starts, save_model='best', resume_from_saved=False, output_folder_path=output_folder_path, output_file_name_root=recommender_class.RECOMMENDER_NAME, metric_to_optimize=metric_to_optimize, cutoff_to_optimize=cutoff_to_optimize)
if __name__ == '__main__':
    linear_combination()