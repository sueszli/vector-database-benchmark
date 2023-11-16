import sys
from builtins import range
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.kmeans import H2OKMeansEstimator

class Test_PUBDEV_2981_kmeans:
    """
    PUBDEV-2981: Sometimes algos just hangs and seem to be doing nothing.
    This class is created to train a kmeans model with different parameters settings and re-create the hangning
     for debugging purposes.
    """
    training1_filenames = 'smalldata/gridsearch/kmeans_8_centers_3_coords.csv'
    test_name = 'pyunit_PUBDEV_2981_kmeans.py'
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

    def test_kmeans_hangup(self):
        if False:
            return 10
        '\n        train a kmeans model with some parameters that will make the system hang.\n        '
        print('*******************************************************************************************')
        h2o.cluster_info()
        good_params_list = {'seed': 1464837706, 'max_iterations': 50, 'init': 'Furthest', 'k': 5}
        good_model_params = {'max_runtime_secs': 0.001}
        good_model = H2OKMeansEstimator(**good_params_list)
        good_model.train(x=self.x_indices, training_frame=self.training1_data, **good_model_params)
        print('Finished.')

def test_PUBDEV_2981_for_kmeans():
    if False:
        i = 10
        return i + 15
    '\n    Create and instantiate class and perform tests specified for kmeans\n    :return: None\n    '
    test_kmeans_grid = Test_PUBDEV_2981_kmeans()
    test_kmeans_grid.test_kmeans_hangup()
    sys.stdout.flush()
    if test_kmeans_grid.test_failed:
        sys.exit(1)
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_PUBDEV_2981_for_kmeans)
else:
    test_PUBDEV_2981_for_kmeans()