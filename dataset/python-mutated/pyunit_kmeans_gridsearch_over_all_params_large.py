import sys
import random
import os
from builtins import range
import time
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.kmeans import H2OKMeansEstimator
from h2o.grid.grid_search import H2OGridSearch

class Test_kmeans_grid_search:
    """
    PUBDEV-1843: Grid testing.  Subtask 2.

    This class is created to test the gridsearch for kmeans algo and make sure it runs.  Only one test is performed
    here.

    Test Descriptions:
        a. grab all truely griddable parameters and randomly or manually set the parameter values.
        b. Next, build H2O kmeans models using grid search.  Count and make sure models
           are only built for hyper-parameters set to legal values.  No model is built for bad hyper-parameters
           values.  We should instead get a warning/error message printed out.
        c. For each model built using grid search, we will extract the parameters used in building
           that model and manually build a H2O kmeans model.  Training metrics are calculated from the
           gridsearch model and the manually built model.  If their metrics
           differ by too much, print a warning message but don't fail the test.
        d. we will check and make sure the models are built within the max_runtime_secs time limit that was set
           for it as well.  If max_runtime_secs was exceeded, declare test failure.

    Note that for hyper-parameters containing all legal parameter names and parameter value lists with legal
    and illegal values, grid-models should be built for all combinations of legal parameter values.  For
    illegal parameter values, a warning/error message should be printed out to warn the user but the
    program should not throw an exception.
    """
    max_grid_model = 100
    curr_time = str(round(time.time()))
    seed = int(round(time.time()))
    training1_filenames = 'smalldata/gridsearch/kmeans_8_centers_3_coords.csv'
    json_filename = 'gridsearch_kmeans_hyper_parameter_' + curr_time + '.json'
    allowed_diff = 0.01
    current_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    max_int_val = 10
    min_int_val = -2
    max_int_number = 3
    max_real_val = 1
    min_real_val = -0.1
    max_real_number = 3
    time_scale = 2
    max_iter_scale = 10
    extra_time_fraction = 0.5
    model_run_time = 0.0
    allowed_runtime_diff = 0.05
    test_name = 'pyunit_kmeans_gridsearch_over_all_params_large.py'
    sandbox_dir = ''
    x_indices = []
    training1_data = []
    test_failed = 0
    hyper_params = dict()
    hyper_params['init'] = ['Random', 'PlusPlus', 'Furthest']
    exclude_parameter_lists = ['model_id', 'seed', 'keep_cross_validation_fold_assignment', 'fold_assignmentkeep_cross_validation_predictions', 'fold_column', 'validation_frame', 'standardize']
    params_zero_one = ['col_sample_rate', 'learn_rate_annealing', 'learn_rate', 'col_sample_rate_per_tree', 'sample_rate']
    params_more_than_zero = ['k']
    params_more_than_one = ['nbins_cats', 'nbins']
    params_zero_positive = ['max_runtime_secs', 'stopping_rounds', 'stopping_tolerance', 'max_iterations']
    final_hyper_params = dict()
    gridable_parameters = []
    gridable_types = []
    gridable_defaults = []
    possible_number_models = 0
    correct_model_number = 0
    true_correct_model_number = 0

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_data()
        self.setup_model()

    def setup_data(self):
        if False:
            i = 10
            return i + 15
        '\n        This function performs all initializations necessary:\n        load the data sets and set the training set indices\n        '
        self.sandbox_dir = pyunit_utils.make_Rsandbox_dir(self.current_dir, self.test_name, True)
        self.training1_data = h2o.import_file(path=pyunit_utils.locate(self.training1_filenames))
        self.x_indices = list(range(self.training1_data.ncol))
        pyunit_utils.remove_csv_files(self.current_dir, '.csv', action='copy', new_dir_path=self.sandbox_dir)

    def setup_model(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This function setup the gridsearch hyper-parameters that will be used later on:\n\n        1. It will first try to grab all the parameters that are griddable and parameters used by kmeans.\n        2. It will find the intersection of parameters that are both griddable and used by kmeans.\n        3. There are several extra parameters that are used by kmeans that are denoted as griddable but actually is not.\n        These parameters have to be discovered manually and they These are captured in self.exclude_parameter_lists.\n        4. We generate the gridsearch hyper-parameter.  For numerical parameters, we will generate those randomly.\n        For enums, we will include all of them.\n\n        :return: None\n        '
        model = H2OKMeansEstimator(k=10)
        model.train(x=self.x_indices, training_frame=self.training1_data)
        self.model_run_time = pyunit_utils.find_grid_runtime([model])
        print('Time taken to build a base barebone model is {0}'.format(self.model_run_time))
        summary_list = model._model_json['output']['model_summary']
        num_iter = summary_list['number_of_iterations'][0]
        self.min_runtime_per_iter = self.model_run_time / num_iter
        (self.gridable_parameters, self.gridable_types, self.gridable_defaults) = pyunit_utils.get_gridables(model._model_json['parameters'])
        (self.hyper_params, self.gridable_parameters, self.gridable_types, self.gridable_defaults) = pyunit_utils.gen_grid_search(model.full_parameters.keys(), self.hyper_params, self.exclude_parameter_lists, self.gridable_parameters, self.gridable_types, self.gridable_defaults, random.randint(1, self.max_int_number), self.max_int_val, self.min_int_val, random.randint(1, self.max_real_number), self.max_real_val, self.min_real_val)
        time_scale = self.time_scale * self.model_run_time
        if 'max_runtime_secs' in list(self.hyper_params):
            self.hyper_params['max_runtime_secs'] = [time_scale * x for x in self.hyper_params['max_runtime_secs']]
        if 'max_iterations' in list(self.hyper_params):
            self.hyper_params['max_iterations'] = [self.max_iter_scale * x for x in self.hyper_params['max_iterations']]
        [self.possible_number_models, self.final_hyper_params] = pyunit_utils.check_and_count_models(self.hyper_params, self.params_zero_one, self.params_more_than_zero, self.params_more_than_one, self.params_zero_positive, self.max_grid_model)
        if 'max_runtime_secs' not in list(self.final_hyper_params) and 'max_runtime_secs' in list(self.hyper_params):
            self.final_hyper_params['max_runtime_secs'] = self.hyper_params['max_runtime_secs']
            len_good_time = len([x for x in self.hyper_params['max_runtime_secs'] if x >= 0])
            self.possible_number_models = self.possible_number_models * len_good_time
        if 'k' not in list(self.final_hyper_params) and 'k' in list(self.hyper_params):
            self.final_hyper_params['k'] = self.hyper_params['k']
            len_good_k = len([x for x in self.hyper_params['k'] if x > 0])
            self.possible_number_models = self.possible_number_models * len_good_k
        self.final_hyper_params['seed'] = [self.seed]
        pyunit_utils.write_hyper_parameters_json(self.current_dir, self.sandbox_dir, self.json_filename, self.final_hyper_params)

    def test_kmeans_grid_search_over_params(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        test_kmeans_grid_search_over_params performs the following:\n        a. build H2O kmeans models using grid search.  Count and make sure models\n           are only built for hyper-parameters set to legal values.  No model is built for bad hyper-parameters\n           values.  We should instead get a warning/error message printed out.\n        b. For each model built using grid search, we will extract the parameters used in building\n           that model and manually build a H2O kmeans model.  Training metrics are calculated from the\n           gridsearch model and the manually built model.  If their metrics\n           differ by too much, print a warning message but don't fail the test.\n        c. we will check and make sure the models are built within the max_runtime_secs time limit that was set\n           for it as well.  If max_runtime_secs was exceeded, declare test failure.\n        "
        print('*******************************************************************************************')
        print('test_kmeans_grid_search_over_params for kmeans ')
        h2o.cluster_info()
        try:
            print('Hyper-parameters used here is {0}'.format(self.final_hyper_params))
            grid_model = H2OGridSearch(H2OKMeansEstimator(), hyper_params=self.final_hyper_params)
            grid_model.train(x=self.x_indices, training_frame=self.training1_data)
            self.correct_model_number = len(grid_model)
            if self.correct_model_number - self.possible_number_models > 0.9:
                self.test_failed += 1
                print('test_kmeans_grid_search_over_params for kmeans failed: number of models built by gridsearch: {0} does not equal to all possible combinations of hyper-parameters: {1}'.format(self.correct_model_number, self.possible_number_models))
            else:
                params_dict = dict()
                total_run_time_limits = 0.0
                true_run_time_limits = 0.0
                manual_run_runtime = 0.0
                for each_model in grid_model:
                    params_list = grid_model.get_hyperparams_dict(each_model._id)
                    params_list.update(params_dict)
                    model_params = dict()
                    num_iter = 0
                    if 'max_runtime_secs' in params_list:
                        model_params['max_runtime_secs'] = params_list['max_runtime_secs']
                        max_runtime = params_list['max_runtime_secs']
                        del params_list['max_runtime_secs']
                    else:
                        max_runtime = 0
                    each_model_runtime = pyunit_utils.find_grid_runtime([each_model])
                    manual_model = H2OKMeansEstimator(**params_list)
                    manual_model.train(x=self.x_indices, training_frame=self.training1_data, **model_params)
                    model_runtime = pyunit_utils.find_grid_runtime([manual_model])
                    manual_run_runtime += model_runtime
                    summary_list = manual_model._model_json['output']['model_summary']
                    if summary_list is not None:
                        num_iter = summary_list['number_of_iterations'][0]
                        if not each_model._model_json['output']['model_summary'] is None:
                            grid_model_metrics = each_model._model_json['output']['model_summary']['total_sum_of_squares'][0]
                            manual_model_metrics = manual_model._model_json['output']['model_summary']['total_sum_of_squares'][0]
                            if not (type(grid_model_metrics) == str or type(manual_model_metrics) == str):
                                if abs(grid_model_metrics) > 0 and abs(grid_model_metrics - manual_model_metrics) / grid_model_metrics > self.allowed_diff:
                                    print('test_kmeans_grid_search_over_params for kmeans warning: grid search model metric ({0}) and manually built H2O model metric ({1}) differ too much!'.format(grid_model_metrics, manual_model_metrics))
                    if max_runtime > 0:
                        if max_runtime < self.min_runtime_per_iter or num_iter <= 1:
                            total_run_time_limits += model_runtime
                        else:
                            total_run_time_limits += max_runtime
                    true_run_time_limits += max_runtime
                total_run_time_limits = max(total_run_time_limits, true_run_time_limits) * (1 + self.extra_time_fraction)
                if not manual_run_runtime <= total_run_time_limits:
                    self.test_failed += 1
                    print('test_kmeans_grid_search_over_params for kmeans failed: time taken to manually build models is {0}.  Maximum allowed time is {1}'.format(manual_run_runtime, total_run_time_limits))
                else:
                    print('time taken to manually build all models is {0}. Maximum allowed time is {1}'.format(manual_run_runtime, total_run_time_limits))
                if self.test_failed == 0:
                    print('test_kmeans_grid_search_over_params for kmeans has passed!')
        except Exception as e:
            if self.possible_number_models > 0:
                print('test_kmeans_grid_search_over_params for kmeans failed: exception ({0}) was thrown for no reason.'.format(e))
                self.test_failed += 1

def test_grid_search_for_kmeans_over_all_params():
    if False:
        while True:
            i = 10
    '\n    Create and instantiate class and perform tests specified for kmeans\n\n    :return: None\n    '
    test_kmeans_grid = Test_kmeans_grid_search()
    test_kmeans_grid.test_kmeans_grid_search_over_params()
    sys.stdout.flush()
    if test_kmeans_grid.test_failed:
        sys.exit(1)
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_grid_search_for_kmeans_over_all_params)
else:
    test_grid_search_for_kmeans_over_all_params()