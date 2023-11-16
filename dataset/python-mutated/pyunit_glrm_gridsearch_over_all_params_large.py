import sys
import random
import os
from builtins import range
import time
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.glrm import H2OGeneralizedLowRankEstimator
from h2o.grid.grid_search import H2OGridSearch

class Test_glrm_grid_search:
    """
    PUBDEV-1843: Grid testing.  Subtask 2.

    This class is created to test the gridsearch for GLRM algo and make sure it runs.  Only one test is performed
    here.

    Test Descriptions:
        a. grab all truely griddable parameters and randomly or manually set the parameter values.
        b. Next, build H2O GLRM models using grid search.  Count and make sure models
           are only built for hyper-parameters set to legal values.  No model is built for bad hyper-parameters
           values.  We should instead get a warning/error message printed out.
        c. For each model built using grid search, we will extract the parameters used in building
           that model and manually build a H2O GLRM model.  Training metrics are calculated from the
           gridsearch model and the manually built model.  If their metrics
           differ by too much, print a warning message but don't fail the test.
        d. we will check and make sure the models are built within the max_runtime_secs time limit that was set
           for it as well.  If max_runtime_secs was exceeded, declare test failure as well.

    Note that for hyper-parameters containing all legal parameter names and parameter value lists with legal
    and illegal values, grid-models should be built for all combinations of legal parameter values.  For
    illegal parameter values, a warning/error message should be printed out to warn the user but the
    program should not throw an exception;

    A fixed dataset has been generated for GLRM.  Assume the original matrix is 1000 rows by 25 columns.  It is
    formed by multiplying two random 1000 x 5 and 5 x 25 matrices.  This final matrix will be our dataset.

    Note: GLRM does not yet support cross-validation.  However, Arno is going to work on it.  I need to come back
    and fixed this once it is done.
    """
    max_grid_model = 100
    curr_time = str(round(time.time()))
    seed = int(round(time.time()))
    training1_filenames = 'smalldata/gridsearch/glrmdata1000x25.csv'
    json_filename = 'gridsearch_glrm_hyper_parameter_' + curr_time + '.json'
    allowed_diff = 0.01
    current_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    train_row_count = 0
    train_col_count = 0
    max_int_val = 10
    min_int_val = -2
    max_int_number = 3
    max_real_val = 1
    min_real_val = -0.1
    max_real_number = 3
    time_scale = 2
    extra_time_fraction = 0.5
    model_run_time = 0.0
    iter_scale = 10
    test_name = 'pyunit_glrm_gridsearch_over_all_params_large.py'
    sandbox_dir = ''
    x_indices = []
    y_index = 0
    training1_data = []
    test_failed = 0
    hyper_params = dict()
    hyper_params['transform'] = ['NONE', 'DEMEAN', 'DESCALE', 'STANDARDIZE', 'NORMALIZE']
    hyper_params['loss'] = ['Quadratic', 'Absolute', 'Huber', 'Hinge', 'Logistic']
    hyper_params['regularization_x'] = ['None', 'Quadratic', 'L2', 'L1', 'NonNegative', 'OneSparse', 'UnitOneSparse', 'Simplex']
    hyper_params['regularization_y'] = ['None', 'Quadratic', 'L2', 'L1', 'NonNegative', 'OneSparse', 'UnitOneSparse', 'Simplex']
    hyper_params['init'] = ['Random', 'PlusPlus']
    hyper_params['svd_method'] = ['GramSVD', 'Power', 'Randomized']
    exclude_parameter_lists = ['validation_frame', 'multi-loss', 'max_updates', 'seed', 'period', 'min_step_size', 'fold_assignment']
    exclude_parameter_lists.extend(['fold_column', 'weights_column', 'offset_column'])
    params_zero_one = ['col_sample_rate', 'learn_rate_annealing', 'learn_rate', 'col_sample_rate_per_tree', 'sample_rate']
    params_more_than_zero = ['k', 'max_iterations', 'init_step_size']
    params_more_than_one = []
    params_zero_positive = ['max_runtime_secs', 'stopping_rounds', 'stopping_tolerance', 'period', 'gamma_x', 'gamma_y']
    final_hyper_params = dict()
    gridable_parameters = []
    gridable_types = []
    gridable_defaults = []
    possible_number_models = 0
    correct_model_number = 0

    def __init__(self):
        if False:
            return 10
        self.setup_data()
        self.setup_model()

    def setup_data(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This function performs all initializations necessary:\n        1. load the dataset\n        2. set the x_indices and y_index if applicable.\n        '
        self.sandbox_dir = pyunit_utils.make_Rsandbox_dir(self.current_dir, self.test_name, True)
        self.training1_data = h2o.import_file(path=pyunit_utils.locate(self.training1_filenames))
        self.x_indices = list(range(self.training1_data.ncol))
        pyunit_utils.remove_csv_files(self.current_dir, '.csv', action='copy', new_dir_path=self.sandbox_dir)

    def setup_model(self):
        if False:
            return 10
        '\n        This function setup the gridsearch hyper-parameters that will be used later on:\n\n        1. It will first try to grab all the parameters that are griddable and parameters used by GLRM.\n        2. It will find the intersection of parameters that are both griddable and used by GLRM.\n        3. There are several extra parameters that are used by GLRM that are denoted as griddable but actually is not.\n        These parameters have to be discovered manually and they These are captured in self.exclude_parameter_lists.\n        4. We generate the gridsearch hyper-parameter.  For numerical parameters, we will generate those randomly.\n        For enums, we will include all of them.\n\n        :return: None\n        '
        model = H2OGeneralizedLowRankEstimator(k=10, loss='Quadratic', gamma_x=random.uniform(0, 1), gamma_y=random.uniform(0, 1), transform='DEMEAN')
        model.train(x=self.training1_data.names, training_frame=self.training1_data)
        self.model_run_time = pyunit_utils.find_grid_runtime([model])
        print('Time taken to build a base barebone model is {0}'.format(self.model_run_time))
        summary_list = model._model_json['output']['model_summary']
        num_iter = summary_list['number_of_iterations'][0]
        self.min_runtime_per_iter = self.model_run_time / num_iter
        (self.gridable_parameters, self.gridable_types, self.gridable_defaults) = pyunit_utils.get_gridables(model._model_json['parameters'])
        (self.hyper_params, self.gridable_parameters, self.gridable_types, self.gridable_defaults) = pyunit_utils.gen_grid_search(model.full_parameters.keys(), self.hyper_params, self.exclude_parameter_lists, self.gridable_parameters, self.gridable_types, self.gridable_defaults, random.randint(1, self.max_int_number), self.max_int_val, self.min_int_val, random.randint(1, self.max_real_number), self.max_real_val, self.min_real_val)
        hyper_params_list = list(self.hyper_params)
        time_scale = self.time_scale * self.model_run_time
        if 'max_runtime_secs' in hyper_params_list:
            self.hyper_params['max_runtime_secs'] = [time_scale * x for x in self.hyper_params['max_runtime_secs']]
        if 'max_iterations' in hyper_params_list:
            self.hyper_params['max_iterations'] = [self.iter_scale * x for x in self.hyper_params['max_iterations']]
        [self.possible_number_models, self.final_hyper_params] = pyunit_utils.check_and_count_models(self.hyper_params, self.params_zero_one, self.params_more_than_zero, self.params_more_than_one, self.params_zero_positive, self.max_grid_model)
        if 'max_runtime_secs' not in list(self.final_hyper_params) and 'max_runtime_secs' in list(self.hyper_params):
            self.final_hyper_params['max_runtime_secs'] = self.hyper_params['max_runtime_secs']
            len_good_time = len([x for x in self.hyper_params['max_runtime_secs'] if x >= 0])
            self.possible_number_models = self.possible_number_models * len_good_time
        if 'k' not in list(self.final_hyper_params):
            self.final_hyper_params['k'] = self.hyper_params['k']
            len_good_k = len([x for x in self.final_hyper_params['k'] if x >= 1])
            self.possible_number_models = self.possible_number_models * len_good_k
        self.final_hyper_params['seed'] = [self.seed]
        pyunit_utils.write_hyper_parameters_json(self.current_dir, self.sandbox_dir, self.json_filename, self.final_hyper_params)

    def test_glrm_grid_search_over_params(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        test_glrm_grid_search_over_params performs the following:\n        a. build H2O GLRM models using grid search.  Count and make sure models\n           are only built for hyper-parameters set to legal values.  No model is built for bad hyper-parameters\n           values.  We should instead get a warning/error message printed out.\n        b. For each model built using grid search, we will extract the parameters used in building\n           that model and manually build a H2O GLRM model.  Training metrics are calculated from the\n           gridsearch model and the manually built model.  If their metrics\n           differ by too much, print a warning message but don't fail the test.\n        c. we will check and make sure the models are built within the max_runtime_secs time limit that was set\n           for it as well.  If max_runtime_secs was exceeded, declare test failure.\n        "
        print('*******************************************************************************************')
        print('test_glrm_grid_search_over_params for GLRM ')
        h2o.cluster_info()
        if self.possible_number_models > 0:
            print('Hyper-parameters used here is {0}'.format(self.final_hyper_params))
            grid_model = H2OGridSearch(H2OGeneralizedLowRankEstimator(), hyper_params=self.final_hyper_params)
            grid_model.train(x=self.x_indices, training_frame=self.training1_data)
            self.correct_model_number = len(grid_model)
            if not self.correct_model_number == self.possible_number_models:
                self.test_failed += 1
                print('test_glrm_grid_search_over_params for GLRM failed: number of models built by gridsearch: {1} does not equal to all possible combinations of hyper-parameters: {1}'.format(self.correct_model_number, self.possible_number_models))
            else:
                params_dict = dict()
                total_run_time_limits = 0.0
                true_run_time_limits = 0.0
                manual_run_runtime = 0.0
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
                    manual_model = H2OGeneralizedLowRankEstimator(**params_list)
                    manual_model.train(x=self.training1_data.names, training_frame=self.training1_data, **model_params)
                    model_runtime = pyunit_utils.find_grid_runtime([manual_model])
                    manual_run_runtime += model_runtime
                    summary_list = manual_model._model_json['output']['model_summary']
                    num_iter = summary_list['number_of_iterations'][0]
                    if max_runtime > 0:
                        if max_runtime < self.min_runtime_per_iter or num_iter <= 1:
                            total_run_time_limits += model_runtime
                        else:
                            total_run_time_limits += max_runtime
                    true_run_time_limits += max_runtime
                    grid_model_metrics = each_model._model_json['output']['objective']
                    manual_model_metrics = manual_model._model_json['output']['objective']
                    if not (type(grid_model_metrics) == unicode or type(manual_model_metrics) == unicode):
                        if abs(grid_model_metrics) > 0 and abs(grid_model_metrics - manual_model_metrics) / grid_model_metrics > self.allowed_diff:
                            print('test_glrm_grid_search_over_params for GLRM warning: grid search model mdetric ({0}) and manually built H2O model metric ({1}) differ too much!'.format(grid_model_metrics, manual_model_metrics))
                total_run_time_limits = max(total_run_time_limits, true_run_time_limits) * (1 + self.extra_time_fraction)
                if not manual_run_runtime <= total_run_time_limits:
                    self.test_failed += 1
                    print('test_glrm_grid_search_over_params for GLRM failed: time taken to manually build models is {0}.  Maximum allowed time is {1}'.format(manual_run_runtime, total_run_time_limits))
                else:
                    print('time taken to manually build all models is {0}. Maximum allowed time is {1}'.format(manual_run_runtime, total_run_time_limits))
                if self.test_failed == 0:
                    print('test_glrm_grid_search_over_params for GLRM has passed!')

def test_grid_search_for_glrm_over_all_params():
    if False:
        print('Hello World!')
    '\n    Create and instantiate class and perform tests specified for GLRM\n\n    :return: None\n    '
    test_glrm_grid = Test_glrm_grid_search()
    test_glrm_grid.test_glrm_grid_search_over_params()
    sys.stdout.flush()
    if test_glrm_grid.test_failed:
        sys.exit(1)
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_grid_search_for_glrm_over_all_params)
else:
    test_grid_search_for_glrm_over_all_params()