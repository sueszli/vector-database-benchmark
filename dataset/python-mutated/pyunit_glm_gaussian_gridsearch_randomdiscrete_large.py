import sys
import random
import os
from builtins import range
import time
import json
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.grid.grid_search import H2OGridSearch

class Test_glm_random_grid_search:
    """
    This class is created to test the three stopping conditions for randomized gridsearch using
    GLM Binomial family.  The three stopping conditions are :

    1. max_runtime_secs:
    2. max_models:
    3. metrics.  We will be picking 2 stopping metrics to test this stopping condition with.  One metric
    will be optimized if it increases and the other one should be optimized if it decreases.

    I have written 4 tests:
    1. test1_glm_random_grid_search_model_number: this test will not put any stopping conditions
    on randomized search.  The purpose here is to make sure that randomized search will give us all possible
    hyper-parameter combinations.
    2. test2_glm_random_grid_search_max_model: this test the stopping condition of setting the max_model in
    search criteria;
    3. test3_glm_random_grid_search_max_runtime_secs: this test the stopping condition max_runtime_secs
    in search criteria;
    4. test4_glm_random_grid_search_metric: this test the stopping condition of using a metric which can be
    increasing or decreasing.
    """
    curr_time = str(round(time.time()))
    training1_filename = 'smalldata/gridsearch/gaussian_training1_set.csv'
    json_filename = 'random_gridsearch_GLM_Gaussian_hyper_parameter_' + curr_time + '.json'
    allowed_diff = 0.5
    allowed_time_diff = 0.1
    current_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    train_row_count = 0
    train_col_count = 0
    max_int_val = 1000
    min_int_val = 0
    max_int_number = 3
    max_real_val = 1
    min_real_val = 0.0
    max_real_number = 3
    lambda_scale = 100
    max_runtime_scale = 3
    one_model_time = 0
    possible_number_models = 0
    max_model_number = 0
    max_grid_runtime = 1
    allowed_scaled_overtime = 1
    allowed_scaled_time = 1
    allowed_scaled_model_number = 1.5
    max_stopping_rounds = 5
    max_tolerance = 0.01
    family = 'gaussian'
    test_name = 'pyunit_glm_gaussian_gridsearch_randomdiscrete_large.py'
    sandbox_dir = ''
    x_indices = []
    y_index = 0
    training1_data = []
    total_test_number = 5
    test_failed = 0
    test_failed_array = [0] * total_test_number
    test_num = 0
    hyper_params = {}
    exclude_parameter_lists = ['tweedie_link_power', 'tweedie_variance_power']
    exclude_parameter_lists.extend(['fold_column', 'weights_column', 'offset_column'])
    exclude_parameter_lists.extend(['model_id'])
    gridable_parameters = []
    gridable_types = []
    gridable_defaults = []
    correct_model_number = 0
    nfolds = 5

    def __init__(self, family):
        if False:
            return 10
        '\n        Constructor.\n\n        :param family: distribution family for tests\n        :return: None\n        '
        self.setup_data()
        self.setup_grid_params()

    def setup_data(self):
        if False:
            return 10
        '\n        This function performs all initializations necessary:\n        load the data sets and set the training set indices and response column index\n        '
        self.sandbox_dir = pyunit_utils.make_Rsandbox_dir(self.current_dir, self.test_name, True)
        self.training1_data = h2o.import_file(path=pyunit_utils.locate(self.training1_filename))
        self.y_index = self.training1_data.ncol - 1
        self.x_indices = list(range(self.y_index))
        pyunit_utils.remove_csv_files(self.current_dir, '.csv', action='copy', new_dir_path=self.sandbox_dir)

    def setup_grid_params(self):
        if False:
            return 10
        '\n        This function setup the randomized gridsearch parameters that will be used later on:\n\n        1. It will first try to grab all the parameters that are griddable and parameters used by GLM.\n        2. It will find the intersection of parameters that are both griddable and used by GLM.\n        3. There are several extra parameters that are used by GLM that are denoted as griddable but actually is not.\n        These parameters have to be discovered manually and they These are captured in self.exclude_parameter_lists.\n        4. We generate the gridsearch hyper-parameter.  For numerical parameters, we will generate those randomly.\n        For enums, we will include all of them.\n\n        :return: None\n        '
        model = H2OGeneralizedLinearEstimator(family=self.family, nfolds=self.nfolds)
        model.train(x=self.x_indices, y=self.y_index, training_frame=self.training1_data)
        self.one_model_time = pyunit_utils.find_grid_runtime([model])
        print('Time taken to build a base barebone model is {0}'.format(self.one_model_time))
        (self.gridable_parameters, self.gridable_types, self.gridable_defaults) = pyunit_utils.get_gridables(model._model_json['parameters'])
        self.hyper_params = {}
        self.hyper_params['fold_assignment'] = ['AUTO', 'Random', 'Modulo']
        self.hyper_params['missing_values_handling'] = ['MeanImputation', 'Skip']
        (self.hyper_params, self.gridable_parameters, self.gridable_types, self.gridable_defaults) = pyunit_utils.gen_grid_search(model.full_parameters.keys(), self.hyper_params, self.exclude_parameter_lists, self.gridable_parameters, self.gridable_types, self.gridable_defaults, random.randint(1, self.max_int_number), self.max_int_val, self.min_int_val, random.randint(1, self.max_real_number), self.max_real_val, self.min_real_val)
        if 'lambda' in list(self.hyper_params):
            self.hyper_params['lambda'] = [self.lambda_scale * x for x in self.hyper_params['lambda']]
        time_scale = self.max_runtime_scale * self.one_model_time
        if 'max_runtime_secs' in list(self.hyper_params):
            self.hyper_params['max_runtime_secs'] = [time_scale * x for x in self.hyper_params['max_runtime_secs']]
        self.possible_number_models = pyunit_utils.count_models(self.hyper_params)
        pyunit_utils.write_hyper_parameters_json(self.current_dir, self.sandbox_dir, self.json_filename, self.hyper_params)

    def tear_down(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This function performs teardown after the dynamic test is completed.  If all tests\n        passed, it will delete all data sets generated since they can be quite large.  It\n        will move the training/validation/test data sets into a Rsandbox directory so that\n        we can re-run the failed test.\n        '
        if self.test_failed:
            self.sandbox_dir = pyunit_utils.make_Rsandbox_dir(self.current_dir, self.test_name, True)
            pyunit_utils.move_files(self.sandbox_dir, self.training1_data_file, self.training1_filename)
            json_file = os.path.join(self.sandbox_dir, self.json_filename)
            with open(json_file, 'wb') as test_file:
                json.dump(self.hyper_params, test_file)
        else:
            pyunit_utils.make_Rsandbox_dir(self.current_dir, self.test_name, False)

    def test1_glm_random_grid_search_model_number(self, metric_name):
        if False:
            i = 10
            return i + 15
        '\n        This test is used to make sure the randomized gridsearch will generate all models specified in the\n        hyperparameters if no stopping condition is given in the search criterion.\n\n        :param metric_name: string to denote what grid search model should be sort by\n\n        :return: None\n        '
        print('*******************************************************************************************')
        print('test1_glm_random_grid_search_model_number for GLM ' + self.family)
        h2o.cluster_info()
        search_criteria = {'strategy': 'RandomDiscrete', 'stopping_rounds': 0, 'seed': round(time.time())}
        print('GLM Gaussian grid search_criteria: {0}'.format(search_criteria))
        random_grid_model = H2OGridSearch(H2OGeneralizedLinearEstimator(family=self.family, nfolds=self.nfolds), hyper_params=self.hyper_params, search_criteria=search_criteria)
        random_grid_model.train(x=self.x_indices, y=self.y_index, training_frame=self.training1_data)
        if not len(random_grid_model) == self.possible_number_models:
            self.test_failed += 1
            self.test_failed_array[self.test_num] = 1
            print('test1_glm_random_grid_search_model_number for GLM: failed, number of models generatedpossible model number {0} and randomized gridsearch model number {1} are not equal.'.format(self.possible_number_models, len(random_grid_model)))
        else:
            self.max_grid_runtime = pyunit_utils.find_grid_runtime(random_grid_model)
        if self.test_failed_array[self.test_num] == 0:
            print('test1_glm_random_grid_search_model_number for GLM: passed!')
        self.test_num += 1
        sys.stdout.flush()

    def test2_glm_random_grid_search_max_model(self):
        if False:
            while True:
                i = 10
        '\n        This test is used to test the stopping condition max_model_number in the randomized gridsearch.  The\n        max_models parameter is randomly generated.  If it is higher than the actual possible number of models\n        that can be generated with the current hyper-space parameters, randomized grid search should generate\n        all the models.  Otherwise, grid search shall return a model that equals to the max_model setting.\n        '
        print('*******************************************************************************************')
        print('test2_glm_random_grid_search_max_model for GLM ' + self.family)
        h2o.cluster_info()
        self.max_model_number = random.randint(1, int(self.allowed_scaled_model_number * self.possible_number_models))
        search_criteria = {'strategy': 'RandomDiscrete', 'max_models': self.max_model_number, 'seed': round(time.time())}
        print('GLM Gaussian grid search_criteria: {0}'.format(search_criteria))
        print('Possible number of models built is {0}'.format(self.possible_number_models))
        grid_model = H2OGridSearch(H2OGeneralizedLinearEstimator(family=self.family, nfolds=self.nfolds), hyper_params=self.hyper_params, search_criteria=search_criteria)
        grid_model.train(x=self.x_indices, y=self.y_index, training_frame=self.training1_data)
        number_model_built = len(grid_model)
        print('Maximum model limit is {0}.  Number of models built is {1}'.format(search_criteria['max_models'], number_model_built))
        if self.possible_number_models >= self.max_model_number:
            if not number_model_built == self.max_model_number:
                print('test2_glm_random_grid_search_max_model: failed.  Number of model built {0} does not match stopping condition number{1}.'.format(number_model_built, self.max_model_number))
                self.test_failed += 1
                self.test_failed_array[self.test_num] = 1
            else:
                print('test2_glm_random_grid_search_max_model for GLM: passed.')
        elif not number_model_built == self.possible_number_models:
            self.test_failed += 1
            self.test_failed_array[self.test_num] = 1
            print('test2_glm_random_grid_search_max_model: failed. Number of model built {0} does not equal to possible model number {1}.'.format(number_model_built, self.possible_number_models))
        else:
            print('test2_glm_random_grid_search_max_model for GLM: passed.')
        self.test_num += 1
        sys.stdout.flush()

    def test3_glm_random_grid_search_max_runtime_secs(self):
        if False:
            return 10
        '\n        This function will test the stopping criteria max_runtime_secs.  For each model built, the field\n        run_time actually denote the time in ms used to build the model.  We will add up the run_time from all\n        models and check against the stopping criteria max_runtime_secs.  Since each model will check its run time\n        differently, there is some inaccuracies in the actual run time.  For example, if we give a model 10 ms to\n        build.  The GLM may check and see if it has used up all the time for every 10 epochs that it has run.  On\n        the other hand, deeplearning may check the time it has spent after every epoch of training.\n\n        If we are able to restrict the runtime to not exceed the specified max_runtime_secs by a certain\n        percentage, we will consider the test a success.\n\n        :return: None\n        '
        print('*******************************************************************************************')
        print('test3_glm_random_grid_search_max_runtime_secs for GLM ' + self.family)
        h2o.cluster_info()
        if 'max_runtime_secs' in list(self.hyper_params):
            del self.hyper_params['max_runtime_secs']
            self.possible_number_models = pyunit_utils.count_models(self.hyper_params)
        max_run_time_secs = random.uniform(self.one_model_time, self.allowed_scaled_time * self.max_grid_runtime)
        search_criteria = {'strategy': 'RandomDiscrete', 'max_runtime_secs': max_run_time_secs, 'seed': round(time.time())}
        print('GLM Gaussian grid search_criteria: {0}'.format(search_criteria))
        grid_model = H2OGridSearch(H2OGeneralizedLinearEstimator(family=self.family, nfolds=self.nfolds), hyper_params=self.hyper_params, search_criteria=search_criteria)
        grid_model.train(x=self.x_indices, y=self.y_index, training_frame=self.training1_data)
        actual_run_time_secs = pyunit_utils.find_grid_runtime(grid_model)
        print('Maximum time limit is {0}.  Time taken to build all model is {1}'.format(search_criteria['max_runtime_secs'], actual_run_time_secs))
        print('Maximum model number is {0}.  Actual number of models built is {1}'.format(self.possible_number_models, len(grid_model)))
        if actual_run_time_secs <= search_criteria['max_runtime_secs'] * (1 + self.allowed_diff):
            print('test3_glm_random_grid_search_max_runtime_secs: passed!')
            if len(grid_model) > self.possible_number_models:
                self.test_failed += 1
                self.test_failed_array[self.test_num] = 1
                print('test3_glm_random_grid_search_max_runtime_secs: failed.  Generated {0} models  which exceeds maximum possible model number {1}'.format(len(grid_model), self.possible_number_models))
        elif len(grid_model) == 1:
            print('test3_glm_random_grid_search_max_runtime_secs: passed!')
        else:
            self.test_failed += 1
            self.test_failed_array[self.test_num] = 1
            print('test3_glm_random_grid_search_max_runtime_secs: failed.  Model takes time {0} seconds which exceeds allowed time {1}'.format(actual_run_time_secs, max_run_time_secs * (1 + self.allowed_diff)))
        self.test_num += 1
        sys.stdout.flush()

    def test4_glm_random_grid_search_metric(self, metric_name, bigger_is_better):
        if False:
            for i in range(10):
                print('nop')
        '\n        This function will test the last stopping condition using metrics.\n\n        :param metric_name: metric we want to use to test the last stopping condition\n        :param bigger_is_better: higher metric value indicates better model performance\n\n        :return: None\n        '
        print('*******************************************************************************************')
        print('test4_glm_random_grid_search_metric using ' + metric_name + ' for family ' + self.family)
        h2o.cluster_info()
        search_criteria = {'strategy': 'RandomDiscrete', 'stopping_metric': metric_name, 'stopping_tolerance': random.uniform(1e-08, self.max_tolerance), 'stopping_rounds': random.randint(1, self.max_stopping_rounds), 'seed': round(time.time())}
        print('GLM Gaussian grid search_criteria: {0}'.format(search_criteria))
        self.hyper_params['max_runtime_secs'] = [0.3]
        grid_model = H2OGridSearch(H2OGeneralizedLinearEstimator(family=self.family, nfolds=self.nfolds), hyper_params=self.hyper_params, search_criteria=search_criteria)
        grid_model.train(x=self.x_indices, y=self.y_index, training_frame=self.training1_data)
        stopped_correctly = pyunit_utils.evaluate_metrics_stopping(grid_model.models, metric_name, bigger_is_better, search_criteria, self.possible_number_models)
        if stopped_correctly:
            print('test4_glm_random_grid_search_metric ' + metric_name + ': passed. ')
        else:
            self.test_failed += 1
            self.test_failed_array[self.test_num] = 1
            print('test4_glm_random_grid_search_metric ' + metric_name + ': failed. ')
        self.test_num += 1

def test_random_grid_search_for_glm():
    if False:
        return 10
    '\n    Create and instantiate classes, call test methods to test randomize grid search for GLM Gaussian\n    or Binomial families.\n\n    :return: None\n    '
    test_glm_gaussian_random_grid = Test_glm_random_grid_search('gaussian')
    test_glm_gaussian_random_grid.test1_glm_random_grid_search_model_number('mse(xval=True)')
    test_glm_gaussian_random_grid.test2_glm_random_grid_search_max_model()
    test_glm_gaussian_random_grid.test3_glm_random_grid_search_max_runtime_secs()
    test_glm_gaussian_random_grid.test4_glm_random_grid_search_metric('MSE', False)
    if test_glm_gaussian_random_grid.test_failed > 0:
        sys.exit(1)
    else:
        pyunit_utils.remove_files(os.path.join(test_glm_gaussian_random_grid.current_dir, test_glm_gaussian_random_grid.json_filename))
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_random_grid_search_for_glm)
else:
    test_random_grid_search_for_glm()