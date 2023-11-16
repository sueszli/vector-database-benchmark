import sys
import random
import os
from builtins import range
import time
import json
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.grid.grid_search import H2OGridSearch

class Test_rf_gridsearch_sorting_metrics:
    """
    PUBDEV-2967: gridsearch sorting metric with cross-validation.

    This class is created to test that when cross-validation is enabled, the gridsearch models are returned sorted
    according to the cross-validation metrics.

    Test Descriptions:
        a. grab all truely griddable parameters and randomly or manually set the parameter values.
        b. Next, build H2O random forest models using grid search.  No model is built for bad hyper-parameters
           values.  We should instead get a warning/error message printed out.
        c. Check and make sure that the models are returned sorted with the correct cross-validation metrics.

    Note that for hyper-parameters containing all legal parameter names and parameter value lists with legal
    and illegal values, grid-models should be built for all combinations of legal parameter values.  For
    illegal parameter values, a warning/error message should be printed out to warn the user but the
    program should not throw an exception;

    We will re-use the dataset generation methods for GLM.  There will be only one data set for classification.
    """
    max_grid_model = 25
    diff = 1e-10
    curr_time = str(round(time.time()))
    seed = int(round(time.time()))
    training1_filename = 'smalldata/gridsearch/multinomial_training1_set.csv'
    json_filename = 'gridsearch_rf_hyper_parameter_' + curr_time + '.json'
    current_dir = os.path.dirname(os.path.realpath(sys.argv[1]))
    train_row_count = 0
    train_col_count = 0
    max_int_val = 10
    min_int_val = 0
    max_int_number = 2
    max_real_val = 1
    min_real_val = 0
    max_real_number = 2
    time_scale = 2
    family = 'multinomial'
    training_metric = 'logloss'
    test_name = 'pyunit_rf_gridsearch_sorting_metrics.py'
    sandbox_dir = ''
    x_indices = []
    y_index = 0
    training1_data = []
    test_failed = 0
    hyper_params = dict()
    hyper_params['balance_classes'] = [True, False]
    hyper_params['fold_assignment'] = ['AUTO', 'Random', 'Modulo', 'Stratified']
    hyper_params['stopping_metric'] = ['logloss']
    exclude_parameter_lists = ['validation_frame', 'response_column', 'fold_column', 'offset_column', 'col_sample_rate_change_per_level', 'sample_rate_per_class', 'col_sample_rate_per_tree', 'nbins', 'nbins_top_level', 'nbins_cats', 'seed', 'class_sampling_factors', 'max_after_balance_size', 'min_split_improvement', 'histogram_type', 'mtries', 'weights_column', 'min_rows', 'r2_stopping', 'score_tree_interval']
    params_zero_one = ['sample_rate']
    params_more_than_zero = ['ntrees', 'max_depth']
    params_more_than_one = []
    params_zero_positive = ['max_runtime_secs', 'stopping_rounds', 'stopping_tolerance']
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
        '\n        This function performs all initializations necessary:\n        load the data sets and set the training set indices and response column index\n        '
        self.sandbox_dir = pyunit_utils.make_Rsandbox_dir(self.current_dir, self.test_name, True)
        self.training1_data = h2o.import_file(path=pyunit_utils.locate(self.training1_filename))
        self.y_index = self.training1_data.ncol - 1
        self.x_indices = list(range(self.y_index))
        self.training1_data[self.y_index] = self.training1_data[self.y_index].round().asfactor()
        pyunit_utils.remove_csv_files(self.current_dir, '.csv', action='copy', new_dir_path=self.sandbox_dir)

    def setup_model(self):
        if False:
            return 10
        '\n        This function setup the gridsearch hyper-parameters that will be used later on:\n\n        1. It will first try to grab all the parameters that are griddable and parameters used by random forest.\n        2. It will find the intersection of parameters that are both griddable and used by random forest.\n        3. There are several extra parameters that are used by random forest that are denoted as griddable but actually\n        are not.  These parameters have to be discovered manually and they are captured in\n        self.exclude_parameter_lists.\n        4. We generate the gridsearch hyper-parameter.  For numerical parameters, we will generate those randomly.\n        For enums, we will include all of them.\n\n        :return: None\n        '
        model = H2ORandomForestEstimator(ntrees=self.max_int_val, nfolds=self.nfolds, score_tree_interval=0)
        model.train(x=self.x_indices, y=self.y_index, training_frame=self.training1_data)
        self.model_run_time = pyunit_utils.find_grid_runtime([model])
        print('Time taken to build a base barebone model is {0}'.format(self.model_run_time))
        summary_list = model._model_json['output']['model_summary']
        num_trees = summary_list['number_of_trees'][0]
        if num_trees == 0:
            self.min_runtime_per_tree = self.model_run_time
        else:
            self.min_runtime_per_tree = self.model_run_time / num_trees
        (self.gridable_parameters, self.gridable_types, self.gridable_defaults) = pyunit_utils.get_gridables(model._model_json['parameters'])
        (self.hyper_params, self.gridable_parameters, self.gridable_types, self.gridable_defaults) = pyunit_utils.gen_grid_search(model.full_parameters.keys(), self.hyper_params, self.exclude_parameter_lists, self.gridable_parameters, self.gridable_types, self.gridable_defaults, random.randint(1, self.max_int_number), self.max_int_val, self.min_int_val, random.randint(1, self.max_real_number), self.max_real_val, self.min_real_val)
        time_scale = self.time_scale * self.model_run_time
        if 'max_runtime_secs' in list(self.hyper_params):
            self.hyper_params['max_runtime_secs'] = [time_scale * x for x in self.hyper_params['max_runtime_secs']]
        [self.possible_number_models, self.final_hyper_params] = pyunit_utils.check_and_count_models(self.hyper_params, self.params_zero_one, self.params_more_than_zero, self.params_more_than_one, self.params_zero_positive, self.max_grid_model)
        if 'max_runtime_secs' not in list(self.final_hyper_params) and 'max_runtime_secs' in list(self.hyper_params):
            self.final_hyper_params['max_runtime_secs'] = self.hyper_params['max_runtime_secs']
            len_good_time = len([x for x in self.hyper_params['max_runtime_secs'] if x >= 0])
            self.possible_number_models = self.possible_number_models * len_good_time
        pyunit_utils.write_hyper_parameters_json(self.current_dir, self.sandbox_dir, self.json_filename, self.final_hyper_params)

    def test_rf_gridsearch_sorting_metrics(self):
        if False:
            i = 10
            return i + 15
        '\n        test_rf_gridsearch_sorting_metrics performs the following:\n        b. build H2O random forest models using grid search.  No model is built for bad hyper-parameters\n           values.  We should instead get a warning/error message printed out.\n        c. Check and make sure that the models are returned sorted with the correct cross-validation metrics.\n        '
        if self.possible_number_models > 0:
            print('*******************************************************************************************')
            print('test_rf_gridsearch_sorting_metrics for random forest ')
            h2o.cluster_info()
            print('Hyper-parameters used here is {0}'.format(self.final_hyper_params))
            grid_model = H2OGridSearch(H2ORandomForestEstimator(nfolds=self.nfolds, seed=self.seed, score_tree_interval=0), hyper_params=self.final_hyper_params)
            grid_model.train(x=self.x_indices, y=self.y_index, training_frame=self.training1_data)
            result_table = grid_model._grid_json['summary_table']
            model_index = 0
            grid_model_metrics = []
            diff = 0
            diff_train = 0
            for each_model in grid_model:
                grid_model_metric = float(result_table[self.training_metric][model_index])
                grid_model_metrics.append(grid_model_metric)
                manual_metric = each_model._model_json['output']['cross_validation_metrics']._metric_json['logloss']
                if not type(grid_model_metrics) == unicode and (not type(manual_metric) == unicode):
                    diff += abs(grid_model_metric - manual_metric)
                manual_training_metric = each_model._model_json['output']['training_metrics']._metric_json['logloss']
                if not type(grid_model_metrics) == unicode and (not type(manual_training_metric) == unicode):
                    diff_train += abs(grid_model_metric - manual_training_metric)
                print('grid model logloss: {0}, grid model training logloss: {1}'.format(grid_model_metric, manual_training_metric))
                model_index += 1
            if diff > self.diff or not grid_model_metrics == sorted(grid_model_metrics) or diff_train < self.diff:
                self.test_failed = 1
                print('test_rf_gridsearch_sorting_metrics for random forest has failed!')
            if self.test_failed == 0:
                print('test_rf_gridsearch_sorting_metrics for random forest has passed!')

def test_gridsearch_sorting_metrics():
    if False:
        print('Hello World!')
    '\n    Create and instantiate class and perform tests specified for random forest\n\n    :return: None\n    '
    test_rf_grid = Test_rf_gridsearch_sorting_metrics()
    test_rf_grid.test_rf_gridsearch_sorting_metrics()
    sys.stdout.flush()
    if test_rf_grid.test_failed:
        sys.exit(1)
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_gridsearch_sorting_metrics)
else:
    test_gridsearch_sorting_metrics()