import sys
import random
import os
import numpy as np
import scipy
import math
from scipy import stats
from builtins import range
import time
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.grid.grid_search import H2OGridSearch

class TestGLMGaussian:
    """
    This class is created to test the GLM algo with Gaussian family.  In this case, the relationship
    between the response Y and the predictor vector X is assumed to be Y = W^T * X + E where E is
    unknown Gaussian noise.  We generate random data set using the exact formula.  Since we know
    what the W is and there are theoretical solutions to calculating W, p-values, we know the solution
     to W/p-values/MSE for test and training data set for each randomly generated
    data set.  Hence, we are able to evaluate the H2O GLM Model generated using the same random data
    sets.  When regularization and other parameters are enabled, theoretical solutions are no longer
    available.  However, we can still evaluate H2O GLM model performance by comparing the MSE from H2O model
    and to the theoretical limits since they all are using the same data sets.  As long as they do not
    deviate too much, we consider the H2O model performance satisfactory.  In particular, I have
    written 8 tests in the hope to exercise as many parameters settings of the GLM algo with Gaussian
    distribution as possible.  Tomas has requested 2 tests to be added to test his new feature of
    missing_values_handling for predictors with both categorical/real columns.  Here is a list of
    all tests:

    test1_glm_and_theory(): theoretical values for weights, p-values and MSE are calculated.
        H2O GLM model is built Gaussian family with the same random data set.  We compare
        the weights, p-values, MSEs generated from H2O with theory.
    test2_glm_lambda_search(): test lambda search with alpha set to 0.5 per Tomas's
        suggestion.  Make sure MSEs generated here is comparable in value to H2O
        GLM model with no regularization.
    test3_glm_grid_search_over_params(): test grid search with over
        various alpha values while lambda is set to be the best value obtained
        from test 2.  The best model performance hopefully will generate MSEs
        close to H2O with no regularization in test 1.
    test4_glm_remove_collinear_columns(): test parameter remove_collinear_columns=True
        with lambda set to best lambda from test 2, alpha set to best alpha from Gridsearch
        and solver set to the one which generate the smallest test MSEs.  The same data set
        is used here except that we randomly choose predictor columns to repeat and scale.
        Make sure MSEs generated here is comparable in value to H2O GLM with no
        regularization.
    test5_missing_values(): Test parameter missing_values_handling="MeanImputation" with
        only real value predictors.  The same data sets as before is used.  However, we
        go into the predictor matrix and randomly decide to change a value to be
        nan and create missing values.  Since no regularization is enabled in this case,
        we are able to calculate a theoretical weight/p-values/MSEs where we can
        compare our H2O models with.
    test6_enum_missing_values(): Test parameter missing_values_handling="MeanImputation" with
        mixed predictors (categorical/real value columns).  We first generate a data set that
        contains a random number of columns of categorical and real value columns.  Next, we
        encode the categorical columns.  Then, we generate the random data set using the formula
        Y = W^T * X+ E as before.  Next, we go into the predictor matrix (before encoding) and randomly
        decide to change a value to be nan and create missing values.  Since no regularization
        is enabled in this case, we are able to calculate a theoretical weight/p-values/MSEs
        where we can compare our H2O models with.
    test7_missing_enum_values_lambda_search(): Test parameter
        missing_values_handling="MeanImputation" with mixed predictors (categorical/real value columns).
        We first generate a data set that contains a random number of columns of categorical and real
        value columns.  Next, we encode the categorical columns using true one hot encoding.  Then,
        we generate the random data set using the formula Y = W^T * X+ E as before.  Next, we go into
        the predictor matrix (before encoding) and randomly  decide to change a value to be nan and
        create missing values.  Lambda-search will be enabled with alpha set to 0.5.  Since the
        encoding is different in this case than in test6, we will compute a theoretical weights/MSEs
        and compare the best H2O model MSEs with theoretical calculations and hope that they are close.
    """
    max_col_count = 100
    max_col_count_ratio = 200
    min_col_count_ratio = 50
    max_p_value = 50
    min_p_value = -50
    max_w_value = 50
    min_w_value = -50
    enum_levels = 5
    family = 'gaussian'
    curr_time = str(round(time.time()))
    training_filename = family + '_' + curr_time + '_training_set.csv'
    training_filename_duplicate = family + '_' + curr_time + '_training_set_duplicate.csv'
    training_filename_nans = family + '_' + curr_time + '_training_set_NA.csv'
    training_filename_enum = family + '_' + curr_time + '_training_set_enum.csv'
    training_filename_enum_true_one_hot = family + '_' + curr_time + '_training_set_enum_trueOneHot.csv'
    training_filename_enum_nans = family + '_' + curr_time + '_training_set_enum_NAs.csv'
    training_filename_enum_nans_true_one_hot = family + '_' + curr_time + '_training_set_enum_NAs_trueOneHot.csv'
    validation_filename = family + '_' + curr_time + '_validation_set.csv'
    validation_filename_enum = family + '_' + curr_time + '_validation_set_enum.csv'
    validation_filename_enum_true_one_hot = family + '_' + curr_time + '_validation_set_enum_trueOneHot.csv'
    validation_filename_enum_nans = family + '_' + curr_time + '_validation_set_enum_NAs.csv'
    validation_filename_enum_nans_true_one_hot = family + '_' + curr_time + '_validation_set_enum_NAs_trueOneHot.csv'
    test_filename = family + '_' + curr_time + '_test_set.csv'
    test_filename_duplicate = family + '_' + curr_time + '_test_set_duplicate.csv'
    test_filename_nans = family + '_' + curr_time + '_test_set_NA.csv'
    test_filename_enum = family + '_' + curr_time + '_test_set_enum.csv'
    test_filename_enum_true_one_hot = family + '_' + curr_time + '_test_set_enum_trueOneHot.csv'
    test_filename_enum_nans = family + '_' + curr_time + '_test_set_enum_NAs.csv'
    test_filename_enum_nans_true_one_hot = family + '_' + curr_time + '_test_set_enum_NAs_trueOneHot.csv'
    weight_filename = family + '_' + curr_time + '_weight.csv'
    weight_filename_enum = family + '_' + curr_time + '_weight_enum.csv'
    total_test_number = 7
    ignored_eps = 1e-15
    allowed_diff = 1e-05
    duplicate_col_counts = 5
    duplicate_threshold = 0.8
    duplicate_max_scale = 2
    nan_fraction = 0.2
    current_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    enum_col = 0
    enum_level_vec = []
    noise_std = 0.01
    noise_var = noise_std * noise_std
    train_row_count = 0
    train_col_count = 0
    data_type = 2
    training_data_file = os.path.join(current_dir, training_filename)
    training_data_file_duplicate = os.path.join(current_dir, training_filename_duplicate)
    training_data_file_nans = os.path.join(current_dir, training_filename_nans)
    training_data_file_enum = os.path.join(current_dir, training_filename_enum)
    training_data_file_enum_true_one_hot = os.path.join(current_dir, training_filename_enum_true_one_hot)
    training_data_file_enum_nans = os.path.join(current_dir, training_filename_enum_nans)
    training_data_file_enum_nans_true_one_hot = os.path.join(current_dir, training_filename_enum_nans_true_one_hot)
    validation_data_file = os.path.join(current_dir, validation_filename)
    validation_data_file_enum = os.path.join(current_dir, validation_filename_enum)
    validation_data_file_enum_true_one_hot = os.path.join(current_dir, validation_filename_enum_true_one_hot)
    validation_data_file_enum_nans = os.path.join(current_dir, validation_filename_enum_nans)
    validation_data_file_enum_nans_true_one_hot = os.path.join(current_dir, validation_filename_enum_nans_true_one_hot)
    test_data_file = os.path.join(current_dir, test_filename)
    test_data_file_duplicate = os.path.join(current_dir, test_filename_duplicate)
    test_data_file_nans = os.path.join(current_dir, test_filename_nans)
    test_data_file_enum = os.path.join(current_dir, test_filename_enum)
    test_data_file_enum_true_one_hot = os.path.join(current_dir, test_filename_enum_true_one_hot)
    test_data_file_enum_nans = os.path.join(current_dir, test_filename_enum_nans)
    test_data_file_enum_nans_true_one_hot = os.path.join(current_dir, test_filename_enum_nans_true_one_hot)
    weight_data_file = os.path.join(current_dir, weight_filename)
    weight_data_file_enum = os.path.join(current_dir, weight_filename_enum)
    test_failed = 0
    test_failed_array = [0] * total_test_number
    test_num = 0
    duplicate_col_indices = []
    duplicate_col_scales = []
    test1_r2_train = 0
    test1_mse_train = 0
    test1_weight = []
    test1_p_values = []
    test1_r2_test = 0
    test1_mse_test = 0
    test1_mse_train_theory = 0
    test1_weight_theory = []
    test1_p_values_theory = []
    test1_mse_test_theory = 0
    best_lambda = 0.0
    test_name = 'pyunit_glm_gaussian.py'
    sandbox_dir = ''
    x_indices = []
    y_index = []
    training_data = []
    test_data = []
    valid_data = []
    training_data_grid = []
    best_alpha = -1
    best_grid_mse = -1

    def __init__(self):
        if False:
            return 10
        self.setup()

    def setup(self):
        if False:
            while True:
                i = 10
        '\n        This function performs all initializations necessary to test the GLM algo for Gaussian family:\n        1. generates all the random values for our dynamic tests like the Gaussian\n        noise std, column count and row count for training/validation/test data sets;\n        2. generate the training/validation/test data sets with only real values;\n        3. insert missing values into training/valid/test data sets.\n        4. taken the training/valid/test data sets, duplicate random certain columns,\n            a random number of times and randomly scale each duplicated column;\n        5. generate the training/validation/test data sets with predictors containing enum\n            and real values as well***.\n        6. insert missing values into the training/validation/test data sets with predictors\n            containing enum and real values as well\n\n        *** according to Tomas, when working with mixed predictors (contains both enum/real\n        value columns), the encoding used is different when regularization is enabled or disabled.\n        When regularization is enabled, true one hot encoding is enabled to encode the enum\n        values to binary bits.  When regularization is disabled, a reference level plus one hot encoding\n        is enabled when encoding the enum values to binary bits.  Hence, two data sets are generated\n        when we work with mixed predictors.  One with true-one-hot set to False for no regularization\n        and one with true-one-hot set to True when regularization is enabled.\n        '
        self.sandbox_dir = pyunit_utils.make_Rsandbox_dir(self.current_dir, self.test_name, True)
        self.noise_std = random.uniform(0, math.sqrt(pow(self.max_p_value - self.min_p_value, 2) / 12))
        self.noise_var = self.noise_std * self.noise_std
        self.train_col_count = random.randint(3, self.max_col_count)
        self.train_row_count = int(round(self.train_col_count * random.uniform(self.min_col_count_ratio, self.max_col_count_ratio)))
        self.enum_col = random.randint(1, self.train_col_count - 1)
        self.enum_level_vec = np.random.random_integers(2, self.enum_levels - 1, [self.enum_col, 1])
        pyunit_utils.write_syn_floating_point_dataset_glm(self.training_data_file, self.validation_data_file, self.test_data_file, self.weight_data_file, self.train_row_count, self.train_col_count, self.data_type, self.max_p_value, self.min_p_value, self.max_w_value, self.min_w_value, self.noise_std, self.family, self.train_row_count, self.train_row_count)
        (self.duplicate_col_indices, self.duplicate_col_scales) = pyunit_utils.random_col_duplication(self.train_col_count, self.duplicate_threshold, self.duplicate_col_counts, True, self.duplicate_max_scale)
        dup_col_indices = self.duplicate_col_indices
        dup_col_indices.append(self.train_col_count)
        dup_col_scale = self.duplicate_col_scales
        dup_col_scale.append(1.0)
        print('duplication column and duplication scales are: ')
        print(dup_col_indices)
        print(dup_col_scale)
        pyunit_utils.duplicate_scale_cols(dup_col_indices, dup_col_scale, self.training_data_file, self.training_data_file_duplicate)
        pyunit_utils.duplicate_scale_cols(dup_col_indices, dup_col_scale, self.test_data_file, self.test_data_file_duplicate)
        pyunit_utils.insert_nan_in_data(self.training_data_file, self.training_data_file_nans, self.nan_fraction)
        pyunit_utils.insert_nan_in_data(self.test_data_file, self.test_data_file_nans, self.nan_fraction)
        pyunit_utils.write_syn_mixed_dataset_glm(self.training_data_file_enum, self.training_data_file_enum_true_one_hot, self.validation_data_file_enum, self.validation_data_file_enum_true_one_hot, self.test_data_file_enum, self.test_data_file_enum_true_one_hot, self.weight_data_file_enum, self.train_row_count, self.train_col_count, self.max_p_value, self.min_p_value, self.max_w_value, self.min_w_value, self.noise_std, self.family, self.train_row_count, self.train_row_count, self.enum_col, self.enum_level_vec)
        pyunit_utils.insert_nan_in_data(self.training_data_file_enum, self.training_data_file_enum_nans, self.nan_fraction)
        pyunit_utils.insert_nan_in_data(self.validation_data_file_enum, self.validation_data_file_enum_nans, self.nan_fraction)
        pyunit_utils.insert_nan_in_data(self.test_data_file_enum, self.test_data_file_enum_nans, self.nan_fraction)
        pyunit_utils.insert_nan_in_data(self.training_data_file_enum_true_one_hot, self.training_data_file_enum_nans_true_one_hot, self.nan_fraction)
        pyunit_utils.insert_nan_in_data(self.validation_data_file_enum_true_one_hot, self.validation_data_file_enum_nans_true_one_hot, self.nan_fraction)
        pyunit_utils.insert_nan_in_data(self.test_data_file_enum_true_one_hot, self.test_data_file_enum_nans_true_one_hot, self.nan_fraction)
        self.training_data = h2o.import_file(pyunit_utils.locate(self.training_data_file))
        self.y_index = self.training_data.ncol - 1
        self.x_indices = list(range(self.y_index))
        self.valid_data = h2o.import_file(pyunit_utils.locate(self.validation_data_file))
        self.test_data = h2o.import_file(pyunit_utils.locate(self.test_data_file))
        self.training_data_grid = self.training_data.rbind(self.valid_data)
        pyunit_utils.remove_csv_files(self.current_dir, '.csv', action='copy', new_dir_path=self.sandbox_dir)

    def teardown(self):
        if False:
            while True:
                i = 10
        '\n        This function performs teardown after the dynamic test is completed.  If all tests\n        passed, it will delete all data sets generated since they can be quite large.  It\n        will move the training/validation/test data sets into a Rsandbox directory so that\n        we can re-run the failed test.\n        '
        remove_files = []
        self.sandbox_dir = pyunit_utils.make_Rsandbox_dir(self.current_dir, self.test_name, True)
        if sum(self.test_failed_array[0:4]):
            pyunit_utils.move_files(self.sandbox_dir, self.training_data_file, self.training_filename)
            pyunit_utils.move_files(self.sandbox_dir, self.validation_data_file, self.validation_filename)
            pyunit_utils.move_files(self.sandbox_dir, self.test_data_file, self.test_filename)
        else:
            remove_files.append(self.training_data_file)
            remove_files.append(self.validation_data_file)
            remove_files.append(self.test_data_file)
        if sum(self.test_failed_array[0:6]):
            pyunit_utils.move_files(self.sandbox_dir, self.weight_data_file, self.weight_filename)
        else:
            remove_files.append(self.weight_data_file)
        if self.test_failed_array[3]:
            pyunit_utils.move_files(self.sandbox_dir, self.training_data_file, self.training_filename)
            pyunit_utils.move_files(self.sandbox_dir, self.test_data_file, self.test_filename)
            pyunit_utils.move_files(self.sandbox_dir, self.test_data_file_duplicate, self.test_filename_duplicate)
            pyunit_utils.move_files(self.sandbox_dir, self.training_data_file_duplicate, self.training_filename_duplicate)
        else:
            remove_files.append(self.training_data_file_duplicate)
            remove_files.append(self.test_data_file_duplicate)
        if self.test_failed_array[4]:
            pyunit_utils.move_files(self.sandbox_dir, self.training_data_file, self.training_filename)
            pyunit_utils.move_files(self.sandbox_dir, self.test_data_file, self.test_filename)
            pyunit_utils.move_files(self.sandbox_dir, self.training_data_file_nans, self.training_filename_nans)
            pyunit_utils.move_files(self.sandbox_dir, self.test_data_file_nans, self.test_filename_nans)
        else:
            remove_files.append(self.training_data_file_nans)
            remove_files.append(self.test_data_file_nans)
        if self.test_failed_array[5]:
            pyunit_utils.move_files(self.sandbox_dir, self.training_data_file_enum_nans, self.training_filename_enum_nans)
            pyunit_utils.move_files(self.sandbox_dir, self.test_data_file_enum_nans, self.test_filename_enum_nans)
            pyunit_utils.move_files(self.sandbox_dir, self.weight_data_file_enum, self.weight_filename_enum)
        else:
            remove_files.append(self.training_data_file_enum_nans)
            remove_files.append(self.training_data_file_enum)
            remove_files.append(self.test_data_file_enum_nans)
            remove_files.append(self.test_data_file_enum)
            remove_files.append(self.validation_data_file_enum_nans)
            remove_files.append(self.validation_data_file_enum)
            remove_files.append(self.weight_data_file_enum)
        if self.test_failed_array[6]:
            pyunit_utils.move_files(self.sandbox_dir, self.training_data_file_enum_nans_true_one_hot, self.training_filename_enum_nans_true_one_hot)
            pyunit_utils.move_files(self.sandbox_dir, self.validation_data_file_enum_nans_true_one_hot, self.validation_filename_enum_nans_true_one_hot)
            pyunit_utils.move_files(self.sandbox_dir, self.test_data_file_enum_nans_true_one_hot, self.test_filename_enum_nans_true_one_hot)
            pyunit_utils.move_files(self.sandbox_dir, self.weight_data_file_enum, self.weight_filename_enum)
        else:
            remove_files.append(self.training_data_file_enum_nans_true_one_hot)
            remove_files.append(self.training_data_file_enum_true_one_hot)
            remove_files.append(self.validation_data_file_enum_nans_true_one_hot)
            remove_files.append(self.validation_data_file_enum_true_one_hot)
            remove_files.append(self.test_data_file_enum_nans_true_one_hot)
            remove_files.append(self.test_data_file_enum_true_one_hot)
        if not self.test_failed:
            pyunit_utils.make_Rsandbox_dir(self.current_dir, self.test_name, False)
        if len(remove_files) > 0:
            for file in remove_files:
                pyunit_utils.remove_files(file)

    def test1_glm_and_theory(self):
        if False:
            i = 10
            return i + 15
        '\n        This test is used to test the p-value/linear intercept weight calculation of our GLM\n        when family is set to Gaussian.  Since theoretical values are available, we will compare\n        our GLM output with the theoretical outputs.  This will provide assurance that our GLM\n        is implemented correctly.\n        '
        print('*******************************************************************************************')
        print('Test1: compares the linear regression weights/p-values computed from theory and H2O GLM.')
        try:
            (self.test1_weight_theory, self.test1_p_values_theory, self.test1_mse_train_theory, self.test1_mse_test_theory) = self.theoretical_glm(self.training_data_file, self.test_data_file, False, False)
        except:
            print('problems with lin-alg.  Got bad data set.')
            sys.exit(0)
        model_h2o = H2OGeneralizedLinearEstimator(family=self.family, Lambda=0, compute_p_values=True, standardize=False)
        model_h2o.train(x=self.x_indices, y=self.y_index, training_frame=self.training_data)
        h2o_model_test_metrics = model_h2o.model_performance(test_data=self.test_data)
        num_test_failed = self.test_failed
        (self.test1_weight, self.test1_p_values, self.test1_mse_train, self.test1_r2_train, self.test1_mse_test, self.test1_r2_test, self.test_failed) = pyunit_utils.extract_comparison_attributes_and_print(model_h2o, h2o_model_test_metrics, '\nTest1 Done!', True, True, True, self.test1_weight_theory, self.test1_p_values_theory, self.test1_mse_train_theory, self.test1_mse_test_theory, 'Comparing intercept and weights ....', 'H2O intercept and weights: ', 'Theoretical intercept and weights: ', 'Intercept and weights are not equal!', 'Intercept and weights are close enough!', 'Comparing p-values ....', 'H2O p-values: ', 'Theoretical p-values: ', 'P-values are not equal!', 'P-values are close enough!', 'Comparing training MSEs ....', 'H2O training MSE: ', 'Theoretical training MSE: ', 'Training MSEs are not equal!', 'Training MSEs are close enough!', 'Comparing test MSEs ....', 'H2O test MSE: ', 'Theoretical test MSE: ', 'Test MSEs are not equal!', 'Test MSEs are close enough!', self.test_failed, self.ignored_eps, self.allowed_diff, self.noise_var, False)
        self.test_failed_array[self.test_num] += pyunit_utils.show_test_results('test1_glm_and_theory', num_test_failed, self.test_failed)
        self.test_num += 1

    def test2_glm_lambda_search(self):
        if False:
            return 10
        '\n        This test is used to test the lambda search.  Recall that lambda search enables efficient and\n        automatic search for the optimal value of the lambda parameter.  When lambda search is enabled,\n        GLM will first fit a model with maximum regularization and then keep decreasing it until\n        over fitting occurs.  The resulting model is based on the best lambda value.  According to Tomas,\n        set alpha = 0.5 and enable validation but not cross-validation.\n        '
        print('*******************************************************************************************')
        print('Test2: tests the lambda search.')
        model_h2o_0p5 = H2OGeneralizedLinearEstimator(family=self.family, lambda_search=True, alpha=0.5, lambda_min_ratio=1e-20)
        model_h2o_0p5.train(x=self.x_indices, y=self.y_index, training_frame=self.training_data, validation_frame=self.valid_data)
        self.best_lambda = pyunit_utils.get_train_glm_params(model_h2o_0p5, 'best_lambda')
        h2o_model_0p5_test_metrics = model_h2o_0p5.model_performance(test_data=self.test_data)
        num_test_failed = self.test_failed
        (_, _, _, _, _, _, self.test_failed) = pyunit_utils.extract_comparison_attributes_and_print(model_h2o_0p5, h2o_model_0p5_test_metrics, '\nTest2 Done!', False, False, False, self.test1_weight, None, self.test1_mse_train, self.test1_mse_test, 'Comparing intercept and weights ....', 'H2O lambda search intercept and weights: ', 'H2O test1 template intercept and weights: ', 'Intercept and weights are not equal!', 'Intercept and weights are close enough!', '', '', '', '', '', 'Comparing training MSEs ....', 'H2O lambda search training MSE: ', 'H2O Test1 template training MSE: ', 'Training MSEs are not equal!', 'Training MSEs are close enough!', 'Comparing test MSEs ....', 'H2O lambda search test MSE: ', 'H2O Test1 template test MSE: ', 'Test MSEs are not equal!', 'Test MSEs are close enough!', self.test_failed, self.ignored_eps, self.allowed_diff, self.noise_var, True)
        self.test_failed_array[self.test_num] += pyunit_utils.show_test_results('test2_glm_lambda_search', num_test_failed, self.test_failed)
        self.test_num += 1

    def test3_glm_grid_search(self):
        if False:
            i = 10
            return i + 15
        '\n        This test is used to test GridSearch with the following parameters:\n\n        1. Lambda = best_lambda value from test2\n        2. alpha = [0 0.5 0.99]\n        3. cross-validation with nfolds = 5, fold_assignment = "Random"\n\n        We will look at the best results from the grid search and compare it with test 1\n        results.\n\n        :return: None\n        '
        print('*******************************************************************************************')
        print('Test3: explores various parameter settings in training the GLM using GridSearch using solver ')
        hyper_parameters = {'alpha': [0, 0.5, 0.99]}
        model_h2o_grid_search = H2OGridSearch(H2OGeneralizedLinearEstimator(family=self.family, Lambda=self.best_lambda, nfolds=5, fold_assignment='Random'), hyper_parameters)
        model_h2o_grid_search.train(x=self.x_indices, y=self.y_index, training_frame=self.training_data_grid)
        temp_model = model_h2o_grid_search.sort_by('mse(xval=True)')
        best_model_id = temp_model['Model Id'][0]
        self.best_grid_mse = temp_model['mse(xval=True)'][0]
        self.best_alpha = model_h2o_grid_search.get_hyperparams(best_model_id)
        best_model = h2o.get_model(best_model_id)
        best_model_test_metrics = best_model.model_performance(test_data=self.test_data)
        num_test_failed = self.test_failed
        (_, _, _, _, _, _, self.test_failed) = pyunit_utils.extract_comparison_attributes_and_print(best_model, best_model_test_metrics, '\nTest3 Done!', False, False, False, self.test1_weight, None, self.test1_mse_train, self.test1_mse_test, 'Comparing intercept and weights ....', 'H2O best model from gridsearch intercept and weights: ', 'H2O test1 template intercept and weights: ', 'Intercept and weights are not equal!', 'Intercept and weights are close enough!', '', '', '', '', '', 'Comparing training MSEs ....', 'H2O best model from gridsearch training MSE: ', 'H2O Test1 template training MSE: ', 'Training MSEs are not equal!', 'Training MSEs are close enough!', 'Comparing test MSEs ....', 'H2O best model from gridsearch test MSE: ', 'H2O Test1 template test MSE: ', 'Test MSEs are not equal!', 'Test MSEs are close enough!', self.test_failed, self.ignored_eps, self.allowed_diff, self.noise_var, False)
        self.test_failed_array[self.test_num] += pyunit_utils.show_test_results('test_glm_grid_search_over_params', num_test_failed, self.test_failed)
        self.test_num += 1

    def test4_glm_remove_collinear_columns(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        With the best parameters obtained from test 3 grid search, we will trained GLM\n        with duplicated columns and enable remove_collinear_columns and see if the\n        algorithm catches the duplicated columns.  We will compare the results with test\n        1 results.\n        '
        print('*******************************************************************************************')
        print('Test4: test the GLM remove_collinear_columns.')
        training_data = h2o.import_file(pyunit_utils.locate(self.training_data_file_duplicate))
        test_data = h2o.import_file(pyunit_utils.locate(self.test_data_file_duplicate))
        y_index = training_data.ncol - 1
        x_indices = list(range(y_index))
        print('Best lambda is {0}, best alpha is {1}'.format(self.best_lambda, self.best_alpha))
        model_h2o = H2OGeneralizedLinearEstimator(family=self.family, Lambda=self.best_lambda, alpha=self.best_alpha, remove_collinear_columns=True)
        model_h2o.train(x=x_indices, y=y_index, training_frame=training_data)
        model_h2o_metrics = model_h2o.model_performance(test_data=test_data)
        num_test_failed = self.test_failed
        (_, _, _, _, _, _, self.test_failed) = pyunit_utils.extract_comparison_attributes_and_print(model_h2o, model_h2o_metrics, '\nTest4 Done!', False, False, False, self.test1_weight, None, self.test1_mse_train, self.test1_mse_test, 'Comparing intercept and weights ....', 'H2O remove_collinear_columns intercept and weights: ', 'H2O test1 template intercept and weights: ', 'Intercept and weights are not equal!', 'Intercept and weights are close enough!', '', '', '', '', '', 'Comparing training MSEs ....', 'H2O remove_collinear_columns training MSE: ', 'H2O Test1 template training MSE: ', 'Training MSEs are not equal!', 'Training MSEs are close enough!', 'Comparing test MSEs ....', 'H2O remove_collinear_columns test MSE: ', 'H2O Test1 template test MSE: ', 'Test MSEs are not equal!', 'Test MSEs are close enough!', self.test_failed, self.ignored_eps, self.allowed_diff, self.noise_var, False)
        self.test_failed_array[self.test_num] += pyunit_utils.show_test_results('test4_glm_remove_collinear_columns', num_test_failed, self.test_failed)
        self.test_num += 1

    def test5_missing_values(self):
        if False:
            print('Hello World!')
        '\n        Test parameter missing_values_handling="MeanImputation" with\n        only real value predictors.  The same data sets as before are used.  However, we\n        go into the predictor matrix and randomly decide to change a value to be\n        nan and create missing values.  Since no regularization is enabled in this case,\n        we are able to calculate a theoretical weight/p-values/MSEs where we can\n        compare our H2O models with.\n        '
        print('*******************************************************************************************')
        print('Test5: test the GLM with imputation of missing values with column averages.')
        try:
            (weight_theory, p_values_theory, mse_train_theory, mse_test_theory) = self.theoretical_glm(self.training_data_file_nans, self.test_data_file_nans, False, False)
        except:
            print('Bad dataset, lin-alg package problem.')
            sys.exit(0)
        training_data = h2o.import_file(pyunit_utils.locate(self.training_data_file_nans))
        test_data = h2o.import_file(pyunit_utils.locate(self.test_data_file_nans))
        model_h2o = H2OGeneralizedLinearEstimator(family=self.family, Lambda=0, compute_p_values=True, missing_values_handling='MeanImputation', standardize=False)
        model_h2o.train(x=self.x_indices, y=self.y_index, training_frame=training_data)
        h2o_model_test_metrics = model_h2o.model_performance(test_data=test_data)
        num_test_failed = self.test_failed
        (_, _, _, _, _, _, self.test_failed) = pyunit_utils.extract_comparison_attributes_and_print(model_h2o, h2o_model_test_metrics, '\nTest5 Done!', True, True, True, weight_theory, p_values_theory, mse_train_theory, mse_test_theory, 'Comparing intercept and weights ....', 'H2O missing values intercept and weights: ', 'Theoretical intercept and weights: ', 'Intercept and weights are not equal!', 'Intercept and weights are close enough!', 'Comparing p-values ....', 'H2O missing values p-values: ', 'Theoretical p-values: ', 'P-values are not equal!', 'P-values are close enough!', 'Comparing training MSEs ....', 'H2O missing values training MSE: ', 'Theoretical training MSE: ', 'Training MSEs are not equal!', 'Training MSEs are close enough!', 'Comparing test MSEs ....', 'H2O missing values test MSE: ', 'Theoretical test MSE: ', 'Test MSEs are not equal!', 'Test MSEs are close enough!', self.test_failed, self.ignored_eps, self.allowed_diff, self.noise_var, False)
        self.test_failed_array[self.test_num] += pyunit_utils.show_test_results('test5_missing_values', num_test_failed, self.test_failed)
        self.test_num += 1

    def test6_enum_missing_values(self):
        if False:
            return 10
        '\n        Test parameter missing_values_handling="MeanImputation" with\n        mixed predictors (categorical/real value columns).  We first generate a data set that\n        contains a random number of columns of categorical and real value columns.  Next, we\n        encode the categorical columns.  Then, we generate the random data set using the formula\n        Y = W^T * X+ E as before.  Next, we go into the predictor matrix and randomly\n        decide to change a value to be nan and create missing values.  Since no regularization\n        is enabled in this case, we are able to calculate a theoretical weight/p-values/MSEs\n        where we can compare our H2O models with.\n        '
        print('*******************************************************************************************')
        print('Test6: test the GLM with enum/real values.')
        try:
            (weight_theory, p_values_theory, mse_train_theory, mse_test_theory) = self.theoretical_glm(self.training_data_file_enum_nans, self.test_data_file_enum_nans, True, False)
        except:
            print('Bad data set.  Problem with lin-alg.')
            sys.exit(0)
        training_data = h2o.import_file(pyunit_utils.locate(self.training_data_file_enum_nans))
        test_data = h2o.import_file(pyunit_utils.locate(self.test_data_file_enum_nans))
        for ind in range(self.enum_col):
            training_data[ind] = training_data[ind].round().asfactor()
            test_data[ind] = test_data[ind].round().asfactor()
        num_col = training_data.ncol
        y_index = num_col - 1
        x_indices = list(range(y_index))
        model_h2o = H2OGeneralizedLinearEstimator(family=self.family, Lambda=0, compute_p_values=True, missing_values_handling='MeanImputation')
        model_h2o.train(x=x_indices, y=y_index, training_frame=training_data)
        h2o_model_test_metrics = model_h2o.model_performance(test_data=test_data)
        num_test_failed = self.test_failed
        (_, _, _, _, _, _, self.test_failed) = pyunit_utils.extract_comparison_attributes_and_print(model_h2o, h2o_model_test_metrics, '\nTest6 Done!', True, False, False, weight_theory, p_values_theory, mse_train_theory, mse_test_theory, 'Comparing intercept and weights with enum and missing values....', 'H2O enum missing values no regularization intercept and weights: ', 'Theoretical intercept and weights: ', 'Intercept and weights are not equal!', 'Intercept and weights are close enough!', 'Comparing p-values ....', 'H2O enum missing values no regularization p-values: ', 'Theoretical p-values: ', 'P-values are not equal!', 'P-values are close enough!', 'Comparing training MSEs ....', 'H2O enum missing values no regularization training MSE: ', 'Theoretical training MSE: ', 'Training MSEs are not equal!', 'Training MSEs are close enough!', 'Comparing test MSEs ....', 'H2O enum missing values no regularization test MSE: ', 'Theoretical test MSE: ', 'Test MSEs are not equal!', 'Test MSEs are close enough!', self.test_failed, self.ignored_eps, self.allowed_diff, self.noise_var, False, attr3_bool=False)
        self.test_failed_array[self.test_num] += pyunit_utils.show_test_results('test6_enum_missing_values', num_test_failed, self.test_failed)
        self.test_num += 1

    def test7_missing_enum_values_lambda_search(self):
        if False:
            print('Hello World!')
        '\n        Test parameter missing_values_handling="MeanImputation" with mixed predictors (categorical/real value columns).\n        Test parameter missing_values_handling="MeanImputation" with\n        mixed predictors (categorical/real value columns).  We first generate a data set that\n        contains a random number of columns of categorical and real value columns.  Next, we\n        encode the categorical columns.  Then, we generate the random data set using the formula\n        Y = W^T * X+ E as before.  Next, we go into the predictor matrix and randomly\n        decide to change a value to be nan and create missing values.  Lambda-search will be\n        enabled with alpha set to 0.5.  Since the encoding is different in this case\n        than in test6, we will compute a theoretical weights/MSEs and compare the best H2O\n        model MSEs with theoretical calculations and hope that they are close.\n        '
        print('*******************************************************************************************')
        print('Test7: test the GLM with imputation of missing enum/real values under lambda search.')
        try:
            (weight_theory, p_values_theory, mse_train_theory, mse_test_theory) = self.theoretical_glm(self.training_data_file_enum_nans_true_one_hot, self.test_data_file_enum_nans_true_one_hot, True, True, validation_data_file=self.validation_data_file_enum_nans_true_one_hot)
        except:
            print('Bad data set.  Problem with lin-alg.')
            sys.exit(0)
        training_data = h2o.import_file(pyunit_utils.locate(self.training_data_file_enum_nans_true_one_hot))
        validation_data = h2o.import_file(pyunit_utils.locate(self.validation_data_file_enum_nans_true_one_hot))
        test_data = h2o.import_file(pyunit_utils.locate(self.test_data_file_enum_nans_true_one_hot))
        for ind in range(self.enum_col):
            training_data[ind] = training_data[ind].round().asfactor()
            validation_data[ind] = validation_data[ind].round().asfactor()
            test_data[ind] = test_data[ind].round().asfactor()
        num_col = training_data.ncol
        y_index = num_col - 1
        x_indices = list(range(y_index))
        model_h2o_0p5 = H2OGeneralizedLinearEstimator(family=self.family, lambda_search=True, alpha=0.5, lambda_min_ratio=1e-20, missing_values_handling='MeanImputation')
        model_h2o_0p5.train(x=x_indices, y=y_index, training_frame=training_data, validation_frame=validation_data)
        h2o_model_0p5_test_metrics = model_h2o_0p5.model_performance(test_data=test_data)
        num_test_failed = self.test_failed
        (_, _, _, _, _, _, self.test_failed) = pyunit_utils.extract_comparison_attributes_and_print(model_h2o_0p5, h2o_model_0p5_test_metrics, '\nTest7 Done!', False, False, True, weight_theory, None, mse_train_theory, mse_test_theory, 'Comparing intercept and weights with categorical columns, missing values and lambda search....', 'H2O enum missing values and lambda search intercept and weights: ', 'Theoretical intercept and weights: ', 'Intercept and weights are not equal!', 'Intercept and weights are close enough!', 'Comparing p-values ....', 'H2O enum missing valuesand lambda search p-values: ', 'Theoretical p-values: ', 'P-values are not equal!', 'P-values are close enough!', 'Comparing training MSEs ....', 'H2O enum missing values and lambda search training MSE: ', 'Theoretical training MSE: ', 'Training MSEs are not equal!', 'Training MSEs are close enough!', 'Comparing test MSEs ....', 'H2O enum missing values and lambda search test MSE: ', 'Theoretical test MSE: ', 'Test MSEs are not equal!', 'Test MSEs are close enough!', self.test_failed, self.ignored_eps, self.allowed_diff, self.noise_var, False, attr3_bool=False)
        self.test_failed_array[self.test_num] += pyunit_utils.show_test_results('test7_missing_enum_values_lambda_search', num_test_failed, self.test_failed)
        self.test_num += 1

    def theoretical_glm(self, training_data_file, test_data_file, has_categorical, true_one_hot, validation_data_file=''):
        if False:
            print('Hello World!')
        '\n        This function is written to load in a training/test data sets with predictors followed by the response\n        as the last column.  We then calculate the weights/bias and the p-values using derived formulae\n        off the web.\n\n        :param training_data_file: string representing the training data set filename\n        :param test_data_file:  string representing the test data set filename\n        :param has_categorical: bool indicating if the data set contains mixed predictors (both enum and real)\n        :param true_one_hot:  bool True: true one hot encoding is used.  False: reference level plus one hot\n        encoding is used\n        :param validation_data_file: optional string, denoting validation file so that we can concatenate\n         training and validation data sets into a big training set since H2O model is using a training\n         and a validation data set.\n\n        :return: a tuple containing weights, p-values, training data set MSE and test data set MSE\n\n        '
        training_data_xy = np.asmatrix(np.genfromtxt(training_data_file, delimiter=',', dtype=None))
        test_data_xy = np.asmatrix(np.genfromtxt(test_data_file, delimiter=',', dtype=None))
        if len(validation_data_file) > 0:
            temp_data_xy = np.asmatrix(np.genfromtxt(validation_data_file, delimiter=',', dtype=None))
            training_data_xy = np.concatenate((training_data_xy, temp_data_xy), axis=0)
        if has_categorical:
            training_data_xy = pyunit_utils.encode_enum_dataset(training_data_xy, self.enum_level_vec, self.enum_col, true_one_hot, np.any(training_data_xy))
            test_data_xy = pyunit_utils.encode_enum_dataset(test_data_xy, self.enum_level_vec, self.enum_col, true_one_hot, np.any(training_data_xy))
        if np.isnan(training_data_xy).any():
            inds = np.where(np.isnan(training_data_xy))
            col_means = np.asarray(np.nanmean(training_data_xy, axis=0))[0]
            training_data_xy[inds] = np.take(col_means, inds[1])
            if np.isnan(test_data_xy).any():
                inds = np.where(np.isnan(test_data_xy))
                test_data_xy = pyunit_utils.replace_nan_with_mean(test_data_xy, inds, col_means)
        (num_row, num_col) = training_data_xy.shape
        dof = num_row - num_col
        response_y = training_data_xy[:, num_col - 1]
        training_data = training_data_xy[:, range(0, num_col - 1)]
        temp_ones = np.asmatrix(np.ones(num_row)).transpose()
        x_mat = np.concatenate((temp_ones, training_data), axis=1)
        mat_inv = np.linalg.pinv(x_mat.transpose() * x_mat)
        t_weights = mat_inv * x_mat.transpose() * response_y
        t_predict_y = x_mat * t_weights
        delta = t_predict_y - response_y
        mse_train = delta.transpose() * delta
        mysd = mse_train / dof
        se = np.sqrt(mysd * np.diag(mat_inv))
        tval = abs(t_weights.transpose() / se)
        p_values = scipy.stats.t.sf(tval, dof) * 2
        test_response_y = test_data_xy[:, num_col - 1]
        test_data = test_data_xy[:, range(0, num_col - 1)]
        t_predict = pyunit_utils.generate_response_glm(t_weights, test_data, 0, self.family)
        (num_row_t, num_col_t) = test_data.shape
        temp = t_predict - test_response_y
        mse_test = temp.transpose() * temp / num_row_t
        return (np.array(t_weights.transpose())[0].tolist(), np.array(p_values)[0].tolist(), mse_train[0, 0] / num_row, mse_test[0, 0])

def test_glm_gaussian():
    if False:
        i = 10
        return i + 15
    '\n    Create and instantiate TestGLMGaussian class and perform tests specified for GLM\n    Gaussian family.\n\n    :return: None\n    '
    test_glm_gaussian = TestGLMGaussian()
    test_glm_gaussian.test1_glm_and_theory()
    test_glm_gaussian.test2_glm_lambda_search()
    test_glm_gaussian.test3_glm_grid_search()
    test_glm_gaussian.test4_glm_remove_collinear_columns()
    test_glm_gaussian.test5_missing_values()
    test_glm_gaussian.test6_enum_missing_values()
    test_glm_gaussian.test7_missing_enum_values_lambda_search()
    test_glm_gaussian.teardown()
    sys.stdout.flush()
    if test_glm_gaussian.test_failed:
        sys.exit(1)
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_glm_gaussian)
else:
    test_glm_gaussian()