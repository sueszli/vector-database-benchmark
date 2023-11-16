import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.gam import H2OGeneralizedAdditiveEstimator
from h2o.grid.grid_search import H2OGridSearch

class test_random_gam_gridsearch_specific:
    h2o_data = []
    myX = []
    myY = []
    h2o_model = []
    search_criteria = {'strategy': 'RandomDiscrete', 'max_models': 2, 'seed': 1}
    hyper_parameters = {'scale': [[1, 1, 1], [2, 2, 2]], 'gam_columns': [['C6', 'C7', 'C8']]}
    manual_gam_models = []
    num_grid_models = 0
    num_expected_models = 2

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_data()

    def setup_data(self):
        if False:
            print('Hello World!')
        '\n        This function performs all initializations necessary:\n        load the data sets and set the training set indices and response column index\n        '
        self.h2o_data = h2o.import_file(path=pyunit_utils.locate('smalldata/glm_test/multinomial_10_classes_10_cols_10000_Rows_train.csv'))
        self.h2o_data['C1'] = self.h2o_data['C1'].asfactor()
        self.h2o_data['C2'] = self.h2o_data['C2'].asfactor()
        self.myX = ['C1', 'C2']
        self.myY = 'C11'
        self.h2o_data['C11'] = self.h2o_data['C11'].asfactor()
        for scale in self.hyper_parameters['scale']:
            for gam_columns in self.hyper_parameters['gam_columns']:
                if len(scale) != len(gam_columns):
                    continue
                self.manual_gam_models.append(H2OGeneralizedAdditiveEstimator(family='multinomial', gam_columns=gam_columns, keep_gam_cols=True, scale=scale, seed=1234, bs=[3, 0, 3]))

    def train_models(self):
        if False:
            print('Hello World!')
        self.h2o_model = H2OGridSearch(H2OGeneralizedAdditiveEstimator(family='multinomial', seed=1234, bs=[3, 0, 3], keep_gam_cols=True), hyper_params=self.hyper_parameters, search_criteria=self.search_criteria)
        self.h2o_model.train(x=self.myX, y=self.myY, training_frame=self.h2o_data)
        for model in self.manual_gam_models:
            model.train(x=self.myX, y=self.myY, training_frame=self.h2o_data)
        print('done')

    def match_models(self):
        if False:
            while True:
                i = 10
        for model in self.manual_gam_models:
            scale = model.actual_params['scale']
            gam_columns = model.actual_params['gam_columns']
            for grid_search_model in self.h2o_model.models:
                if grid_search_model.actual_params['gam_columns'] == gam_columns and grid_search_model.actual_params['scale'] == scale:
                    self.num_grid_models += 1
                    assert grid_search_model.coef() == model.coef(), 'coefficients should be equal'
                    break
        assert self.num_grid_models == self.num_expected_models, 'Grid search model parameters incorrect or incorrect number of models generated'

def test_gridsearch():
    if False:
        for i in range(10):
            print('nop')
    test_gam_grid = test_random_gam_gridsearch_specific()
    test_gam_grid.train_models()
    test_gam_grid.match_models()
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_gridsearch)
else:
    test_gridsearch()