from builtins import zip
from builtins import range
import sys, os
sys.path.insert(1, os.path.join('..', '..', '..'))
import h2o
from tests import pyunit_utils
import random
import copy
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

def weights_vi():
    if False:
        while True:
            i = 10
    random.seed(1234)
    response = ['a' for y in range(10)]
    [response.append('b') for y in range(10)]
    p1 = [(1 if random.uniform(0, 1) < 0.9 else 0) if y == 'a' else 0 if random.uniform(0, 1) < 0.9 else 1 for y in response]
    p2 = [(1 if random.uniform(0, 1) < 0.7 else 0) if y == 'a' else 0 if random.uniform(0, 1) < 0.7 else 1 for y in response]
    p3 = [(1 if random.uniform(0, 1) < 0.5 else 0) if y == 'a' else 0 if random.uniform(0, 1) < 0.5 else 1 for y in response]
    dataset1_python = [response, p1, p2, p3]
    dataset1_h2o = h2o.H2OFrame(list(zip(*dataset1_python)))
    dataset1_h2o.set_names(['response', 'p1', 'p2', 'p3'])
    response = ['a' for y in range(10)]
    [response.append('b') for y in range(10)]
    p1 = [(1 if random.uniform(0, 1) < 0.7 else 0) if y == 'a' else 0 if random.uniform(0, 1) < 0.7 else 1 for y in response]
    p2 = [(1 if random.uniform(0, 1) < 0.5 else 0) if y == 'a' else 0 if random.uniform(0, 1) < 0.5 else 1 for y in response]
    p3 = [(1 if random.uniform(0, 1) < 0.9 else 0) if y == 'a' else 0 if random.uniform(0, 1) < 0.9 else 1 for y in response]
    dataset2_python = [response, p1, p2, p3]
    dataset2_h2o = h2o.H2OFrame(list(zip(*dataset2_python)))
    dataset2_h2o.set_names(['response', 'p1', 'p2', 'p3'])
    model_dataset1 = H2ODeepLearningEstimator(variable_importances=True, reproducible=True, hidden=[1], seed=1234, activation='Tanh')
    model_dataset1.train(x=['p1', 'p2', 'p3'], y='response', training_frame=dataset1_h2o)
    varimp_dataset1 = tuple([p[0] for p in model_dataset1.varimp()])
    assert sorted(varimp_dataset1) == ['p1', 'p2', 'p3'], "Expected the following relative variable importance on dataset1: ('p1', 'p2', 'p3'), but got: {0}".format(varimp_dataset1)
    model_dataset2 = H2ODeepLearningEstimator(variable_importances=True, reproducible=True, hidden=[1], seed=1234, activation='Tanh')
    model_dataset2.train(x=['p1', 'p2', 'p3'], y='response', training_frame=dataset2_h2o)
    varimp_dataset2 = tuple([p[0] for p in model_dataset2.varimp()])
    assert sorted(varimp_dataset1) == ['p1', 'p2', 'p3'], "Expected the following relative variable importance on dataset2: ('p3', 'p1', 'p2'), but got: {0}".format(varimp_dataset2)
    dataset1_python_weighted = copy.deepcopy(dataset1_python)
    dataset1_python_weighted.append([0.8] * len(dataset1_python_weighted[0]))
    dataset2_python_weighted = copy.deepcopy(dataset2_python)
    dataset2_python_weighted.append([0.2] * len(dataset2_python_weighted[0]))
    combined_dataset_python = [d1 + d2 for (d1, d2) in zip(dataset1_python_weighted, dataset2_python_weighted)]
    combined_dataset_h2o = h2o.H2OFrame(list(zip(*combined_dataset_python)))
    combined_dataset_h2o.set_names(['response', 'p1', 'p2', 'p3', 'weights'])
    model_combined_dataset = H2ODeepLearningEstimator(variable_importances=True, reproducible=True, hidden=[1], seed=1234, activation='Tanh')
    model_combined_dataset.train(x=['p1', 'p2', 'p3'], y='response', weights_column='weights', training_frame=combined_dataset_h2o)
    varimp_combined = tuple([p[0] for p in model_combined_dataset.varimp()])
    assert sorted(varimp_dataset1) == ['p1', 'p2', 'p3'], "Expected the following relative variable importance on the combined dataset: ('p1', 'p3', 'p2'), but got: {0}".format(varimp_combined)
    dataset1_python_weighted = copy.deepcopy(dataset1_python)
    dataset1_python_weighted.append([0.2] * len(dataset1_python_weighted[0]))
    dataset2_python_weighted = copy.deepcopy(dataset2_python)
    dataset2_python_weighted.append([0.8] * len(dataset2_python_weighted[0]))
    combined_dataset_python = [d1 + d2 for (d1, d2) in zip(dataset1_python_weighted, dataset2_python_weighted)]
    combined_dataset_h2o = h2o.H2OFrame(list(zip(*combined_dataset_python)))
    combined_dataset_h2o.set_names(['response', 'p1', 'p2', 'p3', 'weights'])
    model_combined_dataset = H2ODeepLearningEstimator(variable_importances=True, reproducible=True, hidden=[1], seed=1234, activation='Tanh')
    model_combined_dataset.train(x=['p1', 'p2', 'p3'], y='response', weights_column='weights', training_frame=combined_dataset_h2o)
    varimp_combined = tuple([p[0] for p in model_combined_dataset.varimp()])
    assert sorted(varimp_dataset1) == ['p1', 'p2', 'p3'], "Expected the following relative variable importance on the combined dataset: ('p1', 'p3', 'p2'), but got: {0}".format(varimp_combined)
if __name__ == '__main__':
    pyunit_utils.standalone_test(weights_vi)
else:
    weights_vi()