import sys
import random
import os
from builtins import range
import time
import json
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.naive_bayes import H2ONaiveBayesEstimator
from h2o.grid.grid_search import H2OGridSearch

class Test_naivebayes_grid_search:
    """
    PUBDEV-1843: Grid testing.  Subtask 2.

    This class is created to test the gridsearch for naivebayes algo and make sure it runs.  Only one test is
    performed here.

    Test Descriptions:
        a. grab all truely griddable parameters and randomly or manually set the parameter values.
        b. Next, build H2O naivebayes models using grid search.  Count and make sure models
           are only built for hyper-parameters set to legal values.  No model is built for bad hyper-parameters
           values.  We should instead get a warning/error message printed out.
        c. For each model built using grid search, we will extract the parameters used in building
           that model and manually build a H2O naivebayes model.  Training metrics are calculated from the
           gridsearch model and the manually built model.  If their metrics
           differ by too much, print a warning message but don't fail the test.
        d. we will check and make sure the models are built within the max_runtime_secs time limit that was set
           for it as well.  If max_runtime_secs was exceeded, declare test failure.

    Note that for hyper-parameters containing all legal parameter names and parameter value lists with legal
    and illegal values, grid-models should be built for all combinations of legal parameter values.  For
    illegal parameter values, a warning/error message should be printed out to warn the user but the
    program should not throw an exception;

    We will re-use the dataset generation methods for GLM.  There will be only one data set for classification.
    """
    max_grid_model = 100
    curr_time = str(round(time.time()))
    seed = int(round(time.time()))
    training1_filename = 'smalldata/gridsearch/multinomial_training1_set.csv'
    json_filename = 'gridsearch_naivebayes_hyper_parameter_' + curr_time + '.json'
    allowed_diff = 0.01
    current_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    train_row_count = 0
    train_col_count = 0
    max_int_val = 10
    min_int_val = -2
    max_int_number = 5
    max_real_val = 1
    min_real_val = -0.1
    max_real_number = 5
    time_scale = 2
    extra_time_fraction = 0.5
    min_runtime_per_tree = 0
    model_run_time = 0.0
    allowed_runtime_diff = 0.1
    laplace_scale = max_int_val
    family = 'multinomial'
    training_metric = 'logloss'
    test_name = 'pyunit_naivebayes_gridsearch_over_all_params_large.py'
    sandbox_dir = ''
    x_indices = []
    y_index = 0
    training1_data = []
    test_failed = 0
    hyper_params = dict()
    hyper_params['fold_assignment'] = ['AUTO', 'Random', 'Modulo', 'Stratified']
    hyper_params['compute_metrics'] = [False, True]
    exclude_parameter_lists = ['validation_frame', 'response_column', 'fold_column', 'offset_column', 'min_sdev', 'eps_sdev', 'seed', 'class_sampling_factors', 'balance_classes', 'max_after_balance_size']
    params_zero_one = ['min_prob', 'eps_prob']
    params_more_than_zero = []
    params_more_than_one = []
    params_zero_positive = ['max_runtime_secs', 'laplace']
    final_hyper_params = dict()
    gridable_parameters = []
    gridable_types = []
    gridable_defaults = []
    possible_number_models = 0
    correct_model_number = 0
    true_correct_model_number = 0
    nfolds = 5

    def __init__(self):
        if False:
            print('Hello World!')
        self.setup_data()
        self.setup_model()

    def setup_data(self):
        if False:
            while True:
                i = 10
        '\n        This function performs all initializations necessary:\n        1. generates all the random parameter values for our dynamic tests like the Gaussian\n        noise std, column count and row count for training/test data sets.\n        2. with the chosen distribution family, generate the appropriate data sets\n        4. load the data sets and set the training set indices and response column index\n        '
        self.sandbox_dir = pyunit_utils.make_Rsandbox_dir(self.current_dir, self.test_name, True)
        self.training1_data = h2o.import_file(path=pyunit_utils.locate(self.training1_filename))
        self.y_index = self.training1_data.ncol - 1
        self.x_indices = list(range(self.y_index))
        self.training1_data[self.y_index] = self.training1_data[self.y_index].round().asfactor()
        pyunit_utils.remove_csv_files(self.current_dir, '.csv', action='copy', new_dir_path=self.sandbox_dir)

    def setup_model(self):
        if False:
            while True:
                i = 10
        '\n        This function setup the gridsearch hyper-parameters that will be used later on:\n\n        1. It will first try to grab all the parameters that are griddable and parameters used by naivebayes.\n        2. It will find the intersection of parameters that are both griddable and used by naivebayes.\n        3. There are several extra parameters that are used by naivebayes that are denoted as griddable but actually\n        are not.  These parameters have to be discovered manually and they are captured in\n        self.exclude_parameter_lists.\n        4. We generate the gridsearch hyper-parameter.  For numerical parameters, we will generate those randomly.\n        For enums, we will include all of them.\n\n        :return: None\n        '
        model = H2ONaiveBayesEstimator(nfolds=self.nfolds, compute_metrics=True)
        model.train(x=self.x_indices, y=self.y_index, training_frame=self.training1_data)
        self.model_run_time = pyunit_utils.find_grid_runtime([model])
        print('Time taken to build a base barebone model is {0}'.format(self.model_run_time))
        (self.gridable_parameters, self.gridable_types, self.gridable_defaults) = pyunit_utils.get_gridables(model._model_json['parameters'])
        (self.hyper_params, self.gridable_parameters, self.gridable_types, self.gridable_defaults) = pyunit_utils.gen_grid_search(model.full_parameters.keys(), self.hyper_params, self.exclude_parameter_lists, self.gridable_parameters, self.gridable_types, self.gridable_defaults, random.randint(1, self.max_int_number), self.max_int_val, self.min_int_val, random.randint(1, self.max_real_number), self.max_real_val, self.min_real_val)
        time_scale = self.time_scale * self.model_run_time
        if 'max_runtime_secs' in list(self.hyper_params):
            self.hyper_params['max_runtime_secs'] = [time_scale * x for x in self.hyper_params['max_runtime_secs']]
        [self.possible_number_models, self.final_hyper_params] = pyunit_utils.check_and_count_models(self.hyper_params, self.params_zero_one, self.params_more_than_zero, self.params_more_than_one, self.params_zero_positive, self.max_grid_model)
        final_hyper_params_keys = list(self.final_hyper_params)
        if 'max_runtime_secs' not in final_hyper_params_keys and 'max_runtime_secs' in list(self.hyper_params):
            self.final_hyper_params['max_runtime_secs'] = self.hyper_params['max_runtime_secs']
            len_good_time = len([x for x in self.hyper_params['max_runtime_secs'] if x >= 0])
            self.possible_number_models = self.possible_number_models * len_good_time
        if 'min_prob' in final_hyper_params_keys:
            old_len_prob = len([x for x in self.final_hyper_params['max_runtime_secs'] if x >= 0])
            good_len_prob = len([x for x in self.final_hyper_params['max_runtime_secs'] if x >= 1e-10])
            if old_len_prob > 0:
                self.possible_number_models = self.possible_number_models * good_len_prob / old_len_prob
            else:
                self.possible_number_models = 0
        if 'laplace' in final_hyper_params_keys:
            self.final_hyper_params['laplace'] = [self.laplace_scale * x for x in self.hyper_params['laplace']]
        pyunit_utils.write_hyper_parameters_json(self.current_dir, self.sandbox_dir, self.json_filename, self.final_hyper_params)

    def tear_down(self):
        if False:
            while True:
                i = 10
        '\n        This function performs teardown after the dynamic test is completed.  If all tests\n        passed, it will delete all data sets generated since they can be quite large.  It\n        will move the training/validation/test data sets into a Rsandbox directory so that\n        we can re-run the failed test.\n        '
        if self.test_failed:
            self.sandbox_dir = pyunit_utils.make_Rsandbox_dir(self.current_dir, self.test_name, True)
            pyunit_utils.move_files(self.sandbox_dir, self.training1_data_file, self.training1_filename)
            json_file = os.path.join(self.sandbox_dir, self.json_filename)
            with open(json_file, 'wb') as test_file:
                json.dump(self.hyper_params, test_file)
        else:
            pyunit_utils.make_Rsandbox_dir(self.current_dir, self.test_name, False)
        pyunit_utils.remove_csv_files(self.current_dir, '.csv')
        pyunit_utils.remove_csv_files(self.current_dir, '.json')

    def test_naivebayes_grid_search_over_params(self):
        if False:
            print('Hello World!')
        "\n        test_naivebayes_grid_search_over_params the following:\n        a. grab all truely griddable parameters and randomly or manually set the parameter values.\n        b. Next, build H2O naivebayes models using grid search.  Count and make sure models\n           are only built for hyper-parameters set to legal values.  No model is built for bad hyper-parameters\n           values.  We should instead get a warning/error message printed out.\n        c. For each model built using grid search, we will extract the parameters used in building\n           that model and manually build a H2O naivebayes model.  Training metrics are calculated from the\n           gridsearch model and the manually built model.  If their metrics\n           differ by too much, print a warning message but don't fail the test.\n        d. we will check and make sure the models are built within the max_runtime_secs time limit that was set\n           for it as well.  If max_runtime_secs was exceeded, declare test failure.\n        "
        print('*******************************************************************************************')
        print('test_naivebayes_grid_search_over_params for naivebayes ')
        h2o.cluster_info()
        try:
            print('Hyper-parameters used here is {0}'.format(self.final_hyper_params))
            grid_model = H2OGridSearch(H2ONaiveBayesEstimator(nfolds=self.nfolds), hyper_params=self.final_hyper_params)
            grid_model.train(x=self.x_indices, y=self.y_index, training_frame=self.training1_data)
            self.correct_model_number = len(grid_model)
            if not self.correct_model_number == self.possible_number_models:
                self.test_failed += 1
                print('test_naivebayes_grid_search_over_params for naivebayes failed: number of models built by gridsearch {0} does not equal to all possible combinations of hyper-parameters {1}'.format(self.correct_model_number, self.possible_number_models))
            else:
                params_dict = dict()
                params_dict['nfolds'] = self.nfolds
                total_run_time_limits = 0.0
                true_run_time_limits = 0.0
                manual_run_runtime = 0.0
                gridsearch_runtime = 0.0
                for each_model in grid_model:
                    params_list = grid_model.get_hyperparams_dict(each_model._id)
                    params_list.update(params_dict)
                    model_params = dict()
                    if 'max_runtime_secs' in params_list:
                        model_params['max_runtime_secs'] = params_list['max_runtime_secs']
                        max_runtime = params_list['max_runtime_secs']
                        del params_list['max_runtime_secs']
                    else:
                        max_runtime = 0
                    if 'validation_frame' in params_list:
                        model_params['validation_frame'] = params_list['validation_frame']
                        del params_list['validation_frame']
                    if 'eps_prob' in params_list:
                        model_params['eps_prob'] = params_list['eps_prob']
                        del params_list['eps_prob']
                    if 'min_prob' in params_list:
                        model_params['min_prob'] = params_list['min_prob']
                        del params_list['min_prob']
                    each_model_runtime = pyunit_utils.find_grid_runtime([each_model])
                    gridsearch_runtime += each_model_runtime
                    manual_model = H2ONaiveBayesEstimator(**params_list)
                    manual_model.train(x=self.x_indices, y=self.y_index, training_frame=self.training1_data, **model_params)
                    model_runtime = pyunit_utils.find_grid_runtime([manual_model])
                    manual_run_runtime += model_runtime
                    if max_runtime > 0:
                        if max_runtime < self.model_run_time:
                            total_run_time_limits += model_runtime
                        else:
                            total_run_time_limits += max_runtime
                    true_run_time_limits += max_runtime
                    grid_model_metrics = each_model.model_performance(test_data=self.training1_data)._metric_json[self.training_metric]
                    manual_model_metrics = manual_model.model_performance(test_data=self.training1_data)._metric_json[self.training_metric]
                    if not (type(grid_model_metrics) == str or type(manual_model_metrics) == str):
                        if abs(grid_model_metrics) > 0 and abs(grid_model_metrics - manual_model_metrics) / grid_model_metrics > self.allowed_diff:
                            print('test_naivebayes_grid_search_over_params for naivebayes WARNING\ngrid search model {0}: {1}, time taken to build (secs): {2}\n and manually built H2O model {3}: {4}, time taken to build (secs): {5}\ndiffer too much!'.format(self.training_metric, grid_model_metrics, each_model_runtime, self.training_metric, manual_model_metrics, model_runtime))
                print('Time taken for gridsearch to build all models (sec): {0}\n Time taken to manually build all models (sec): {1}, total run time limits (sec): {2}'.format(gridsearch_runtime, manual_run_runtime, total_run_time_limits))
                total_run_time_limits = max(total_run_time_limits, true_run_time_limits) * (1 + self.extra_time_fraction)
                if not manual_run_runtime <= total_run_time_limits:
                    self.test_failed += 1
                    print('test_naivebayes_grid_search_over_params for naivebayes failed: time taken to manually build models is {0}.  Maximum allowed time is {1}'.format(manual_run_runtime, total_run_time_limits))
                if self.test_failed == 0:
                    print('test_naivebayes_grid_search_over_params for naivebayes has passed!')
        except Exception as e:
            if self.possible_number_models > 0:
                print('test_naivebayes_grid_search_over_params for naivebayes failed: exception ({0}) was thrown for no reason.'.format(e))
                self.test_failed += 1

def test_grid_search_for_naivebayes_over_all_params():
    if False:
        return 10
    '\n    Create and instantiate class and perform tests specified for naive bayes\n\n    :return: None\n    '
    test_naivebayes_grid = Test_naivebayes_grid_search()
    test_naivebayes_grid.test_naivebayes_grid_search_over_params()
    sys.stdout.flush()
    if test_naivebayes_grid.test_failed:
        sys.exit(1)
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_grid_search_for_naivebayes_over_all_params)
else:
    test_grid_search_for_naivebayes_over_all_params()