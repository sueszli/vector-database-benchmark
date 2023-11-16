import sys
import random
import os
import numpy as np
from builtins import range
import time
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.grid.grid_search import H2OGridSearch

class Test_glm_grid_search:
    """
    PUBDEV-1843: Grid testing.  Subtask 7,8.

    This class is created to test the gridsearch with the GLM algo using Guassian, Binonmial or
    Multinomial family.  Three tests are written to test the following conditions:
    1. For hyper-parameters containing all legal parameter names and parameter value lists with legal
    and illegal values, grid-models should be built for all combinations of legal parameter values.  For
    illegal parameter values, a warning/error message should be printed out to warn the user but the
    program should not throw an exception;
    2. For hyper-parameters with illegal names, an exception should be thrown and no models should be built;
    3. For parameters that are specified both in the hyper-parameters and model parameters, unless the values
    specified in the model parameters are set to default values, an exception will be thrown since parameters are
    not supposed to be specified in both places.

    Test Descriptions:
    test1_glm_grid_search_over_params: test for condition 1 and performs the following:
        a. grab all truely griddable parameters and randomly or manually set the parameter values.
        b. Next, build H2O GLM models using grid search.  Count and make sure models
           are only built for hyper-parameters set to legal values.  No model is built for bad hyper-parameters
           values.  We should instead get a warning/error message printed out.
        c. For each model built using grid search, we will extract the parameters used in building
           that model and manually build a H2O GLM model.  Training metrics are calculated from the
           gridsearch model and the manually built model.  If their metrics
           differ by too much, print a warning message but don't fail the test.
        d. we will check and make sure the models are built within the max_runtime_secs time limit that was set
           for it as well.  If max_runtime_secs was exceeded, declare test failure.

    test2_illegal_name_value: test for condition 1 and 2.  Randomly go into the hyper_parameters that we
    have specified, either
        a. randomly alter the name of a hyper-parameter name (fatal, exception will be thrown)
        b. randomly choose a hyper-parameter and remove all elements in its list (fatal)
        c. add randomly generated new hyper-parameter names with random list (fatal)
        d: randomly choose a hyper-parameter and insert an illegal type into it (non fatal, model built with
           legal hyper-parameters settings only and error messages printed out for illegal hyper-parameters
           settings)

    test3_duplicated_parameter_specification: test for condition 3.  Go into our hyper_parameters list, randomly
    choose some hyper-parameters to specify and specify it as part of the model parameters.  Hence, the same
    parameter is specified both in the model parameters and hyper-parameters.  Make sure the test failed with
    error messages when the parameter values are not set to default if they are specified in the model parameters
    as well as in the hyper-parameters.
    """
    max_grid_model = 200
    curr_time = str(round(time.time()))
    training1_filename = ['smalldata/gridsearch/gaussian_training1_set.csv', 'smalldata/gridsearch/binomial_training1_set.csv', 'smalldata/gridsearch/multinomial_training1_set.csv']
    training2_filename = ['smalldata/gridsearch/gaussian_training2_set.csv', 'smalldata/gridsearch/binomial_training2_set.csv', 'smalldata/gridsearch/multinomial_training2_set.csv']
    json_filename = 'gridsearch_hyper_parameter_' + curr_time + '.json'
    json_filename_bad = 'gridsearch_hyper_parameter_bad_' + curr_time + '.json'
    weight_filename = 'gridsearch_' + curr_time + '_weight.csv'
    allowed_diff = 1e-05
    allowed_runtime_diff = 0.15
    current_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    train_row_count = 0
    train_col_count = 0
    max_int_val = 10
    min_int_val = -10
    max_int_number = 3
    max_real_val = 1
    min_real_val = -0.1
    max_real_number = 3
    lambda_scale = 50
    alpha_scale = 1.2
    time_scale = 3
    extra_time_fraction = 0.5
    min_runtime_per_epoch = 0
    families = ['gaussian', 'binomial', 'multinomial']
    family = 'gaussian'
    test_name = 'pyunit_glm_gridsearch_over_all_params_large.py'
    sandbox_dir = ''
    x_indices = []
    y_index = 0
    total_test_number = 3
    test_failed = 0
    test_failed_array = [0] * total_test_number
    test_num = 0
    hyper_params_bad = dict()
    hyper_params_bad['fold_assignment'] = ['AUTO', 'Random', 'Modulo', 'Stratified']
    hyper_params_bad['missing_values_handling'] = ['MeanImputation', 'Skip']
    hyper_params = dict()
    hyper_params['fold_assignment'] = ['AUTO', 'Random', 'Modulo', 'Stratified']
    hyper_params['missing_values_handling'] = ['MeanImputation', 'Skip']
    final_hyper_params_bad = dict()
    final_hyper_params = dict()
    scale_model = 1
    exclude_parameter_lists = ['tweedie_link_power', 'tweedie_variance_power', 'seed']
    exclude_parameter_lists.extend(['fold_column', 'weights_column', 'offset_column'])
    exclude_parameter_lists.extend(['model_id'])
    gridable_parameters = []
    gridable_types = []
    gridable_defaults = []
    possible_number_models = 0
    correct_model_number = 0
    true_correct_model_number = 0
    nfolds = 5
    params_zero_one = ['alpha', 'stopping_tolerance']
    params_more_than_zero = []
    params_more_than_one = []
    params_zero_positive = ['max_runtime_secs', 'stopping_rounds', 'lambda']

    def __init__(self):
        if False:
            while True:
                i = 10
        self.setup_data()
        self.setup_model()

    def setup_data(self):
        if False:
            i = 10
            return i + 15
        '\n        This function performs all initializations necessary:\n        1. Randomly choose which distribution family to use\n        2. load the correct data sets and set the training set indices and response column index\n        '
        self.sandbox_dir = pyunit_utils.make_Rsandbox_dir(self.current_dir, self.test_name, True)
        self.family = self.families[random.randint(0, len(self.families) - 1)]
        if 'binomial' in self.family:
            self.training1_data = h2o.import_file(path=pyunit_utils.locate(self.training1_filename[1]))
            self.training2_data = h2o.import_file(path=pyunit_utils.locate(self.training2_filename[1]))
        elif 'multinomial' in self.family:
            self.training1_data = h2o.import_file(path=pyunit_utils.locate(self.training1_filename[2]))
            self.training2_data = h2o.import_file(path=pyunit_utils.locate(self.training2_filename[2]))
        else:
            self.training1_data = h2o.import_file(path=pyunit_utils.locate(self.training1_filename[0]))
            self.training2_data = h2o.import_file(path=pyunit_utils.locate(self.training2_filename[0]))
            self.scale_model = 0.75
            self.hyper_params['fold_assignment'] = ['AUTO', 'Random', 'Modulo']
        self.y_index = self.training1_data.ncol - 1
        self.x_indices = list(range(self.y_index))
        if 'binomial' in self.family or 'multinomial' in self.family:
            self.training1_data[self.y_index] = self.training1_data[self.y_index].round().asfactor()
            self.training2_data[self.y_index] = self.training2_data[self.y_index].round().asfactor()
        pyunit_utils.remove_csv_files(self.current_dir, '.csv', action='copy', new_dir_path=self.sandbox_dir)

    def setup_model(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This function setup the gridsearch hyper-parameters that will be used later on:\n\n        1. It will first try to grab all the parameters that are griddable and parameters used by GLM.\n        2. It will find the intersection of parameters that are both griddable and used by GLM.\n        3. There are several extra parameters that are used by GLM that are denoted as griddable but actually is not.\n        These parameters have to be discovered manually and they These are captured in self.exclude_parameter_lists.\n        4. We generate the gridsearch hyper-parameter.  For numerical parameters, we will generate those randomly.\n        For enums, we will include all of them.\n\n        :return: None\n        '
        model = H2OGeneralizedLinearEstimator(family=self.family, nfolds=self.nfolds)
        model.train(x=self.x_indices, y=self.y_index, training_frame=self.training1_data)
        run_time = pyunit_utils.find_grid_runtime([model])
        print('Time taken to build a base barebone model is {0}'.format(run_time))
        summary_list = model._model_json['output']['model_summary']
        num_iteration = summary_list.cell_values[0][summary_list.col_header.index('number_of_iterations')]
        if num_iteration == 0:
            self.min_runtime_per_epoch = run_time
        else:
            self.min_runtime_per_epoch = run_time / num_iteration
        (self.gridable_parameters, self.gridable_types, self.gridable_defaults) = pyunit_utils.get_gridables(model._model_json['parameters'])
        (self.hyper_params_bad, self.gridable_parameters, self.gridable_types, self.gridable_defaults) = pyunit_utils.gen_grid_search(model.full_parameters.keys(), self.hyper_params_bad, self.exclude_parameter_lists, self.gridable_parameters, self.gridable_types, self.gridable_defaults, random.randint(1, self.max_int_number), self.max_int_val, self.min_int_val, random.randint(1, self.max_real_number), self.max_real_val * self.alpha_scale, self.min_real_val * self.alpha_scale)
        if 'lambda' in list(self.hyper_params_bad):
            self.hyper_params_bad['lambda'] = [self.lambda_scale * x for x in self.hyper_params_bad['lambda']]
        time_scale = self.time_scale * run_time
        if 'max_runtime_secs' in list(self.hyper_params_bad):
            self.hyper_params_bad['max_runtime_secs'] = [time_scale * x for x in self.hyper_params_bad['max_runtime_secs']]
        [self.possible_number_models, self.final_hyper_params_bad] = pyunit_utils.check_and_count_models(self.hyper_params_bad, self.params_zero_one, self.params_more_than_zero, self.params_more_than_one, self.params_zero_positive, self.max_grid_model)
        if 'max_runtime_secs' not in list(self.final_hyper_params_bad) and 'max_runtime_secs' in list(self.hyper_params_bad):
            self.final_hyper_params_bad['max_runtime_secs'] = self.hyper_params_bad['max_runtime_secs']
            len_good_time = len([x for x in self.hyper_params_bad['max_runtime_secs'] if x >= 0])
            self.possible_number_models = self.possible_number_models * len_good_time
        self.possible_number_models = self.possible_number_models * self.scale_model
        (self.hyper_params, self.gridable_parameters, self.gridable_types, self.gridable_defaults) = pyunit_utils.gen_grid_search(model.full_parameters.keys(), self.hyper_params, self.exclude_parameter_lists, self.gridable_parameters, self.gridable_types, self.gridable_defaults, random.randint(1, self.max_int_number), self.max_int_val, 0, random.randint(1, self.max_real_number), self.max_real_val, 0)
        if 'lambda' in list(self.hyper_params):
            self.hyper_params['lambda'] = [self.lambda_scale * x for x in self.hyper_params['lambda']]
        if 'max_runtime_secs' in list(self.hyper_params):
            self.hyper_params['max_runtime_secs'] = [time_scale * x for x in self.hyper_params['max_runtime_secs']]
        [self.true_correct_model_number, self.final_hyper_params] = pyunit_utils.check_and_count_models(self.hyper_params, self.params_zero_one, self.params_more_than_zero, self.params_more_than_one, self.params_zero_positive, self.max_grid_model)
        if 'max_runtime_secs' not in list(self.final_hyper_params) and 'max_runtime_secs' in list(self.hyper_params):
            self.final_hyper_params['max_runtime_secs'] = self.hyper_params['max_runtime_secs']
            self.true_correct_model_number = self.true_correct_model_number * len(self.final_hyper_params['max_runtime_secs'])
        pyunit_utils.write_hyper_parameters_json(self.current_dir, self.sandbox_dir, self.json_filename_bad, self.final_hyper_params_bad)
        pyunit_utils.write_hyper_parameters_json(self.current_dir, self.sandbox_dir, self.json_filename, self.final_hyper_params)

    def tear_down(self):
        if False:
            for i in range(10):
                print('nop')
        pyunit_utils.remove_files(os.path.join(self.current_dir, self.json_filename))
        pyunit_utils.remove_files(os.path.join(self.current_dir, self.json_filename_bad))

    def test1_glm_grid_search_over_params(self):
        if False:
            i = 10
            return i + 15
        "\n        test1_glm_grid_search_over_params: test for condition 1 and performs the following:\n        a. grab all truely griddable parameters and randomly or manually set the parameter values.\n        b. Next, build H2O GLM models using grid search.  Count and make sure models\n           are only built for hyper-parameters set to legal values.  No model is built for bad hyper-parameters\n           values.  We should instead get a warning/error message printed out.\n        c. For each model built using grid search, we will extract the parameters used in building\n           that model and manually build a H2O GLM model.  Training metrics are calculated from the\n           gridsearch model and the manually built model.  If their metrics\n           differ by too much, print a warning message but don't fail the test.\n        d. we will check and make sure the models are built within the max_runtime_secs time limit that was set\n           for it as well.  If max_runtime_secs was exceeded, declare test failure.\n        "
        print('*******************************************************************************************')
        print('test1_glm_grid_search_over_params for GLM ' + self.family)
        h2o.cluster_info()
        try:
            print('Hyper-parameters used here is {0}'.format(self.final_hyper_params_bad))
            grid_model = H2OGridSearch(H2OGeneralizedLinearEstimator(family=self.family, nfolds=self.nfolds), hyper_params=self.final_hyper_params_bad)
            grid_model.train(x=self.x_indices, y=self.y_index, training_frame=self.training1_data)
            self.correct_model_number = len(grid_model)
            if self.correct_model_number - self.possible_number_models > 0.9:
                self.test_failed += 1
                self.test_failed_array[self.test_num] = 1
                print('test_glm_search_over_params for GLM failed: number of models built by gridsearch: {0} does not equal to all possible combinations of hyper-parameters: {1}'.format(self.correct_model_number, self.possible_models))
            else:
                params_dict = dict()
                params_dict['family'] = self.family
                params_dict['nfolds'] = self.nfolds
                total_run_time_limits = 0.0
                true_run_time_limits = 0.0
                manual_run_runtime = 0.0
                for each_model in grid_model:
                    params_list = grid_model.get_hyperparams_dict(each_model._id)
                    params_list.update(params_dict)
                    model_params = dict()
                    if 'lambda' in list(params_list):
                        params_list['Lambda'] = params_list['lambda']
                        del params_list['lambda']
                    if 'max_runtime_secs' in params_list:
                        model_params['max_runtime_secs'] = params_list['max_runtime_secs']
                        del params_list['max_runtime_secs']
                    if 'stopping_rounds' in params_list:
                        model_params['stopping_rounds'] = params_list['stopping_rounds']
                        del params_list['stopping_rounds']
                    if 'stopping_tolerance' in params_list:
                        model_params['stopping_tolerance'] = params_list['stopping_tolerance']
                        del params_list['stopping_tolerance']
                    manual_model = H2OGeneralizedLinearEstimator(**params_list)
                    manual_model.train(x=self.x_indices, y=self.y_index, training_frame=self.training1_data, **model_params)
                    model_runtime = pyunit_utils.find_grid_runtime([manual_model])
                    manual_run_runtime += model_runtime
                    summary_list = manual_model._model_json['output']['model_summary']
                    iteration_num = summary_list.cell_values['number_of_iterations'][0]
                    if model_params['max_runtime_secs'] > 0:
                        if model_params['max_runtime_secs'] < self.min_runtime_per_epoch or iteration_num <= 1:
                            total_run_time_limits += model_runtime
                        else:
                            total_run_time_limits += model_params['max_runtime_secs']
                    true_run_time_limits += model_params['max_runtime_secs']
                    grid_model_metrics = each_model.model_performance(test_data=self.training2_data)
                    manual_model_metrics = manual_model.model_performance(test_data=self.training2_data)
                    if not (type(grid_model_metrics.mse()) == str or type(manual_model_metrics.mse()) == str):
                        mse = grid_model_metrics.mse()
                        if abs(mse) > 0 and abs(mse - manual_model_metrics.mse()) / mse > self.allowed_diff:
                            print('test1_glm_grid_search_over_params for GLM warning: grid search model metric ({0}) and manually built H2O model metric ({1}) differ too much!'.format(grid_model_metrics.mse(), manual_model_metrics.mse()))
                total_run_time_limits = max(total_run_time_limits, true_run_time_limits) * (1 + self.extra_time_fraction)
            if not self.correct_model_number == self.possible_number_models:
                self.test_failed += 1
                self.test_failed_array[self.test_num] = 1
                print('test1_glm_grid_search_over_params for GLM failed: number of models built by gridsearch does not equal to all possible combinations of hyper-parameters')
            if not manual_run_runtime <= total_run_time_limits:
                print('test1_glm_grid_search_over_params for GLM warning: allow time to build models: {0}, actual time taken: {1}'.format(total_run_time_limits, manual_run_runtime))
            self.test_num += 1
            if self.test_failed == 0:
                print('test1_glm_grid_search_over_params for GLM has passed!')
        except:
            if self.possible_number_models > 0:
                print('test1_glm_grid_search_over_params for GLM failed: exception was thrown for no reason.')

    def test2_illegal_name_value(self):
        if False:
            while True:
                i = 10
        '\n        test2_illegal_name_value: test for condition 1 and 2.  Randomly go into the hyper_parameters that we\n        have specified, either\n        a. randomly alter the name of a hyper-parameter name (fatal, exception will be thrown)\n        b. randomly choose a hyper-parameter and remove all elements in its list (fatal)\n        c. add randomly generated new hyper-parameter names with random list (fatal)\n        d: randomly choose a hyper-parameter and insert an illegal type into it (non fatal, model built with\n           legal hyper-parameters settings only and error messages printed out for illegal hyper-parameters\n           settings)\n\n        The following error conditions will be created depending on the error_number generated:\n\n        error_number = 0: randomly alter the name of a hyper-parameter name;\n        error_number = 1: randomly choose a hyper-parameter and remove all elements in its list\n        error_number = 2: add randomly generated new hyper-parameter names with random list\n        error_number = 3: randomly choose a hyper-parameter and insert an illegal type into it\n\n        :return: None\n        '
        print('*******************************************************************************************')
        print('test2_illegal_name_value for GLM ' + self.family)
        h2o.cluster_info()
        error_number = np.random.random_integers(0, 3, 1)
        error_hyper_params = pyunit_utils.insert_error_grid_search(self.final_hyper_params, self.gridable_parameters, self.gridable_types, error_number[0])
        print('test2_illegal_name_value: the bad hyper-parameters are: ')
        print(error_hyper_params)
        try:
            grid_model = H2OGridSearch(H2OGeneralizedLinearEstimator(family=self.family, nfolds=self.nfolds), hyper_params=error_hyper_params)
            grid_model.train(x=self.x_indices, y=self.y_index, training_frame=self.training1_data)
            if error_number[0] > 2:
                if not len(grid_model) == self.true_correct_model_number:
                    self.test_failed += 1
                    self.test_failed_array[self.test_num] = 1
                    print('test2_illegal_name_value failed. Number of model generated is incorrect.')
                else:
                    print('test2_illegal_name_value passed.')
            else:
                self.test_failed += 1
                self.test_failed_array[self.test_num] = 1
                print('test2_illegal_name_value failed: exception should have been thrown for illegalparameter name or empty hyper-parameter parameter list but did not!')
        except:
            if error_number[0] <= 2 and error_number[0] >= 0:
                print('test2_illegal_name_value passed: exception is thrown for illegal parameter name or emptyhyper-parameter parameter list.')
            else:
                self.test_failed += 1
                self.test_failed_array[self.test_num] = 1
                print('test2_illegal_name_value failed: exception should not have been thrown but did!')
        self.test_num += 1

    def test3_duplicated_parameter_specification(self):
        if False:
            i = 10
            return i + 15
        '\n        This function will randomly choose a parameter in hyper_parameters and specify it as a parameter in the\n        model.  Depending on the random error_number generated, the following is being done to the model parameter\n        and hyper-parameter:\n\n        error_number = 0: set model parameter to be  a value in the hyper-parameter value list, should\n        generate error;\n        error_number = 1: set model parameter to be default value, should not generate error in this case;\n        error_number = 2: make sure model parameter is not set to default and choose a value not in the\n        hyper-parameter value list.\n\n        :return: None\n        '
        print('*******************************************************************************************')
        print('test3_duplicated_parameter_specification for GLM ' + self.family)
        error_number = np.random.random_integers(0, 2, 1)
        print('error_number is {0}'.format(error_number[0]))
        (params_dict, error_hyper_params) = pyunit_utils.generate_redundant_parameters(self.final_hyper_params, self.gridable_parameters, self.gridable_defaults, error_number[0])
        params_dict['family'] = self.family
        params_dict['nfolds'] = self.nfolds
        if 'stopping_rounds' in list(params_dict):
            del params_dict['stopping_rounds']
        if 'stopping_tolerance' in list(params_dict):
            del params_dict['stopping_tolerance']
        print('Your hyper-parameter dict is: ')
        print(error_hyper_params)
        print('Your model parameters are: ')
        print(params_dict)
        try:
            grid_model = H2OGridSearch(H2OGeneralizedLinearEstimator(**params_dict), hyper_params=error_hyper_params)
            grid_model.train(x=self.x_indices, y=self.y_index, training_frame=self.training1_data)
            if not error_number[0] == 1:
                self.test_failed += 1
                self.test_failed_array[self.test_num] = 1
                print('test3_duplicated_parameter_specification failed: Java error exception should have been thrown but did not!')
            else:
                print('test3_duplicated_parameter_specification passed: Java error exception should not have been thrown and did not!')
        except Exception as e:
            if error_number[0] == 1:
                self.test_failed += 1
                self.test_failed_array[self.test_num] = 1
                print('test3_duplicated_parameter_specification failed: Java error exception ({0}) should not have been thrown! '.format(e))
            else:
                print('test3_duplicated_parameter_specification passed: Java error exception ({0}) should have been thrown and did.'.format(e))

def test_grid_search_for_glm_over_all_params():
    if False:
        i = 10
        return i + 15
    '\n    Create and instantiate class and perform tests specified for GLM\n\n    :return: None\n    '
    test_glm_grid = Test_glm_grid_search()
    test_glm_grid.test1_glm_grid_search_over_params()
    test_glm_grid.test2_illegal_name_value()
    test_glm_grid.test3_duplicated_parameter_specification()
    sys.stdout.flush()
    if test_glm_grid.test_failed:
        sys.exit(1)
    else:
        test_glm_grid.tear_down()
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_grid_search_for_glm_over_all_params)
else:
    test_grid_search_for_glm_over_all_params()