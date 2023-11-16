import inspect
import os
import sys
from urllib.parse import urlparse
import deepchecks.tabular.checks as tabular_checks
import deepchecks.vision.checks as vision_checks
from deepchecks.core import BaseCheck
from deepchecks.utils.strings import generate_check_docs_link
from deepchecks.vision.checks.data_integrity.abstract_property_outliers import AbstractPropertyOutliers
checks_dirs = ['deepchecks/tabular/checks', 'deepchecks/vision/checks']
ignored_classes = [AbstractPropertyOutliers, tabular_checks.WholeDatasetDrift, tabular_checks.CategoryMismatchTrainTest, tabular_checks.TrainTestLabelDrift, tabular_checks.TrainTestPredictionDrift, tabular_checks.TrainTestFeatureDrift, vision_checks.TrainTestLabelDrift, vision_checks.TrainTestPredictionDrift]

def test_read_more_link(check_class, compiled_dir: str):
    if False:
        while True:
            i = 10
    link = urlparse(generate_check_docs_link(check_class()))
    relevant_path_parts = link.path.split('/')[2:]
    file_path = os.path.join(*compiled_dir.split('/'), *relevant_path_parts)
    if not os.path.exists(file_path):
        print(f"Check {check_class.__name__} 'read more' link didn't correspond to an html file")
        return False
    return True

def get_check_classes_in_module(module):
    if False:
        print('Hello World!')
    all_classes = dir(module)
    for class_name in all_classes:
        class_ = getattr(module, class_name)
        if hasattr(class_, 'mro') and BaseCheck in class_.mro() and (class_ not in ignored_classes):
            yield class_

def validate_dir(checks_path, examples_path):
    if False:
        print('Hello World!')
    all_valid = True
    for (root, _, files) in os.walk(checks_path):
        for file_name in files:
            if file_name != '__init__.py' and file_name.endswith('.py'):
                check_path = os.path.join(root, file_name)
                if any((inspect.getmodule(cls).__file__.endswith(check_path) for cls in ignored_classes)):
                    continue
                example_file_name = 'plot_' + file_name
                splitted_path = check_path.split('/')
                submodule_name = splitted_path[1]
                check_type = splitted_path[-2]
                example_path = os.path.join(examples_path, submodule_name, check_type, example_file_name)
                if not os.path.exists(example_path):
                    print(f'Check {check_path} does not have a corresponding example file')
                    all_valid = False
                else:
                    pass
    return all_valid
SOURCE_DIR = 'docs/source/checks'
COMPILED_DIR = 'docs/build/html'
valid = True
for x in checks_dirs:
    valid = valid and validate_dir(x, SOURCE_DIR)
for check in get_check_classes_in_module(tabular_checks):
    valid = valid and test_read_more_link(check, COMPILED_DIR)
for check in get_check_classes_in_module(vision_checks):
    valid = valid and test_read_more_link(check, COMPILED_DIR)
sys.exit(0 if valid else 1)