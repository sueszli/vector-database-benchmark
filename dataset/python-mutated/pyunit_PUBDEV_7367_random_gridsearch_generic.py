from past.utils import old_div
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.gam import H2OGeneralizedAdditiveEstimator
from h2o.grid.grid_search import H2OGridSearch

class test_random_gam_gridsearch_generic:
    h2o_data = []
    myX = []
    myY = []
    h2o_model = []
    search_criteria = {'strategy': 'RandomDiscrete', 'max_models': 8, 'seed': 1}
    hyper_parameters = {'scale': [[1, 1], [2, 2]], 'gam_columns': [['C11', 'C12'], ['C12', 'C13']], 'lambda': [0.01]}
    manual_gam_models = []
    num_grid_models = 0
    num_expected_models = 4

    def __init__(self):
        if False:
            return 10
        self.setup_data()

    def setup_data(self):
        if False:
            i = 10
            return i + 15
        '\n        This function performs all initializations necessary:\n        load the data sets and set the training set indices and response column index\n        '
        self.h2o_data = h2o.import_file(path=pyunit_utils.locate('smalldata/glm_test/gaussian_20cols_10000Rows.csv'))
        self.h2o_data['C1'] = self.h2o_data['C1'].asfactor()
        self.h2o_data['C2'] = self.h2o_data['C2'].asfactor()
        self.myX = ['C1', 'C2']
        self.myY = 'C21'
        for scale in self.hyper_parameters['scale']:
            for gam_columns in self.hyper_parameters['gam_columns']:
                for lambda_ in self.hyper_parameters['lambda']:
                    self.manual_gam_models.append(H2OGeneralizedAdditiveEstimator(family='gaussian', gam_columns=gam_columns, keep_gam_cols=True, scale=scale, lambda_=lambda_))

    def train_models(self):
        if False:
            for i in range(10):
                print('nop')
        self.h2o_model = H2OGridSearch(H2OGeneralizedAdditiveEstimator(family='gaussian', keep_gam_cols=True), hyper_params=self.hyper_parameters, search_criteria=self.search_criteria)
        self.h2o_model.train(x=self.myX, y=self.myY, training_frame=self.h2o_data)
        for model in self.manual_gam_models:
            model.train(x=self.myX, y=self.myY, training_frame=self.h2o_data)
        print('done')

    def match_models(self):
        if False:
            i = 10
            return i + 15
        for model in self.manual_gam_models:
            scale = model.actual_params['scale']
            gam_columns = model.actual_params['gam_columns']
            lambda_ = model.actual_params['lambda']
            for grid_search_model in self.h2o_model.models:
                if grid_search_model.actual_params['gam_columns'] == gam_columns and grid_search_model.actual_params['scale'] == scale and (grid_search_model.actual_params['lambda'] == lambda_):
                    self.num_grid_models += 1
                    assert grid_search_model.coef() == model.coef(), 'coefficients should be equal'
                    break
        assert self.num_grid_models == self.num_expected_models, 'Grid search model parameters incorrect or incorrect number of models generated'

def test_gridsearch():
    if False:
        i = 10
        return i + 15
    test_gam_grid = test_random_gam_gridsearch_generic()
    test_gam_grid.train_models()
    test_gam_grid.match_models()
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_gridsearch)
else:
    test_gridsearch()