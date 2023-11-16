from builtins import range
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.random_forest import H2ORandomForestEstimator
import random
import copy

def weights_vi():
    if False:
        for i in range(10):
            print('nop')
    response = ['a'] * 10000 + ['b'] * 10000
    p1 = [(1 if random.uniform(0, 1) < 0.9 else 0) if y == 'a' else 0 if random.uniform(0, 1) < 0.9 else 1 for y in response]
    p2 = [(1 if random.uniform(0, 1) < 0.7 else 0) if y == 'a' else 0 if random.uniform(0, 1) < 0.7 else 1 for y in response]
    p3 = [(1 if random.uniform(0, 1) < 0.5 else 0) if y == 'a' else 0 if random.uniform(0, 1) < 0.5 else 1 for y in response]
    dataset1_python = [response, p1, p2, p3]
    dataset1_h2o = h2o.H2OFrame(list(zip(*dataset1_python)))
    dataset1_h2o.set_names(['response', 'p1', 'p2', 'p3'])
    p1 = [(1 if random.uniform(0, 1) < 0.7 else 0) if y == 'a' else 0 if random.uniform(0, 1) < 0.7 else 1 for y in response]
    p2 = [(1 if random.uniform(0, 1) < 0.5 else 0) if y == 'a' else 0 if random.uniform(0, 1) < 0.5 else 1 for y in response]
    p3 = [(1 if random.uniform(0, 1) < 0.9 else 0) if y == 'a' else 0 if random.uniform(0, 1) < 0.9 else 1 for y in response]
    dataset2_python = [response, p1, p2, p3]
    dataset2_h2o = h2o.H2OFrame(list(zip(*dataset2_python)))
    dataset2_h2o.set_names(['response', 'p1', 'p2', 'p3'])
    model_dataset1 = H2ORandomForestEstimator()
    model_dataset1.train(x=['p1', 'p2', 'p3'], y='response', training_frame=dataset1_h2o)
    varimp_dataset1 = tuple([p[0] for p in model_dataset1.varimp()])
    assert varimp_dataset1 == ('p1', 'p2', 'p3'), "Expected the following relative variable importance on dataset1: ('p1', 'p2', 'p3'), but got: {0}".format(varimp_dataset1)
    model_dataset2 = H2ORandomForestEstimator()
    model_dataset2.train(x=['p1', 'p2', 'p3'], y='response', training_frame=dataset2_h2o)
    varimp_dataset2 = tuple([p[0] for p in model_dataset2.varimp()])
    assert varimp_dataset2 == ('p3', 'p1', 'p2'), "Expected the following relative variable importance on dataset2: ('p3', 'p1', 'p2'), but got: {0}".format(varimp_dataset2)
    dataset1_python_weighted = copy.deepcopy(dataset1_python) + [[0.8] * 20000]
    dataset2_python_weighted = copy.deepcopy(dataset2_python) + [[0.2] * 20000]
    combined_dataset_python = [dataset1_python_weighted[i] + dataset2_python_weighted[i] for i in range(len(dataset1_python_weighted))]
    combined_dataset_h2o = h2o.H2OFrame(list(zip(*combined_dataset_python)))
    combined_dataset_h2o.set_names(['response', 'p1', 'p2', 'p3', 'weights'])
    model_combined_dataset = H2ORandomForestEstimator()
    model_combined_dataset.train(x=['p1', 'p2', 'p3'], y='response', training_frame=combined_dataset_h2o, weights_column='weights')
    varimp_combined = tuple([p[0] for p in model_combined_dataset.varimp()])
    assert varimp_combined == ('p1', 'p2', 'p3'), "Expected the following relative variable importance on the combined dataset: ('p1', 'p2', 'p3'), but got: {0}".format(varimp_combined)
    dataset1_python_weighted = copy.deepcopy(dataset1_python) + [[0.2] * 20000]
    dataset2_python_weighted = copy.deepcopy(dataset2_python) + [[0.8] * 20000]
    combined_dataset_python = [dataset1_python_weighted[i] + dataset2_python_weighted[i] for i in range(len(dataset1_python_weighted))]
    combined_dataset_h2o = h2o.H2OFrame(list(zip(*combined_dataset_python)))
    combined_dataset_h2o.set_names(['response', 'p1', 'p2', 'p3', 'weights'])
    model_combined_dataset = H2ORandomForestEstimator()
    model_combined_dataset.train(x=['p1', 'p2', 'p3'], y='response', training_frame=combined_dataset_h2o, weights_column='weights')
    varimp_combined = tuple([p[0] for p in model_combined_dataset.varimp()])
    assert varimp_combined == ('p3', 'p1', 'p2'), "Expected the following relative variable importance on the combined dataset: ('p3', 'p1', 'p2'), but got: {0}".format(varimp_combined)
if __name__ == '__main__':
    pyunit_utils.standalone_test(weights_vi)
else:
    weights_vi()