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

class Test_PUBDEV_2980_kmeans:
    """
    PUBDEV-2980: kmeans return the field model._model_json['output']['summary']
    as null sometimes.  Normally, it is twoDimensionTable.  Sometimes, it is None.


    This class is created to train a kmeans model with different parameters settings and re-create two
    models with different field type for debugging purposes.
    """
    training1_filenames = 'smalldata/gridsearch/kmeans_8_centers_3_coords.csv'
    test_name = 'pyunit_PUBDEV_2980_kmeans.py'
    x_indices = []
    training1_data = []
    test_failed = 0

    def __init__(self):
        if False:
            return 10
        self.setup_data()

    def setup_data(self):
        if False:
            return 10
        '\n        This function performs all initializations necessary:\n        load the data sets and set the training set indices\n        '
        self.training1_data = h2o.import_file(path=pyunit_utils.locate(self.training1_filenames))
        self.x_indices = list(range(self.training1_data.ncol))

    def test_kmeans_fields(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        test_kmeans_grid_search_over_validation_datasets performs the following:\n        a. build H2O kmeans models using grid search.  Count and make sure models\n           are only built for hyper-parameters set to legal values.  No model is built for bad hyper-parameters\n           values.  We should instead get a warning/error message printed out.\n        b. For each model built using grid search, we will extract the parameters used in building\n           that model and manually build a H2O kmeans model.  Training metrics are calculated from the\n           gridsearch model and the manually built model.  If their metrics\n           differ by too much, print a warning message but don't fail the test.\n        c. we will check and make sure the models are built within the max_runtime_secs time limit that was set\n           for it as well.  If max_runtime_secs was exceeded, declare test failure.\n        "
        print('*******************************************************************************************')
        h2o.cluster_info()
        good_params_list = {'max_iterations': 20, 'k': 6, 'init': 'Furthest', 'seed': 1464891169}
        good_model_params = {'max_runtime_secs': 0.014673351}
        good_model = H2OKMeansEstimator(**good_params_list)
        good_model.train(x=self.x_indices, training_frame=self.training1_data, **good_model_params)
        bad_params_list = {'init': 'Random', 'seed': 1464888628, 'k': 6, 'max_iterations': 0}
        bad_model_params = {'max_runtime_secs': 0.007948218600000001}
        bad_model = H2OKMeansEstimator(**bad_params_list)
        bad_model.train(x=self.x_indices, training_frame=self.training1_data, **bad_model_params)
        good_model_type = type(good_model._model_json['output']['model_summary'])
        bad_model_type = type(bad_model._model_json['output']['model_summary'])
        print("good_model._model_json['output']['model_summary'] type is {0}.  bad_model._model_json['output']['model_summary'] type is {1}".format(good_model_type, bad_model_type))
        if not good_model_type == bad_model_type:
            print('They are not equal for some reason....')
            self.test_failed = 1
        else:
            print('The fields are of the same type.')

def test_PUBDEV_2980_for_kmeans():
    if False:
        print('Hello World!')
    '\n    Create and instantiate class and perform tests specified for kmeans\n\n    :return: None\n    '
    test_kmeans_grid = Test_PUBDEV_2980_kmeans()
    test_kmeans_grid.test_kmeans_fields()
    sys.stdout.flush()
    if test_kmeans_grid.test_failed:
        sys.exit(1)
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_PUBDEV_2980_for_kmeans)
else:
    test_PUBDEV_2980_for_kmeans()