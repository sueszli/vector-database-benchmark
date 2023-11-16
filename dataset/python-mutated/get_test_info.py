import importlib
import os
import sys
sys.path.append('.')
'\nThe argument `test_file` in this file refers to a model test file. This should be a string of the from\n`tests/models/*/test_modeling_*.py`.\n'

def get_module_path(test_file):
    if False:
        print('Hello World!')
    'Return the module path of a model test file.'
    components = test_file.split(os.path.sep)
    if components[0:2] != ['tests', 'models']:
        raise ValueError(f'`test_file` should start with `tests/models/` (with `/` being the OS specific path separator). Got {test_file} instead.')
    test_fn = components[-1]
    if not test_fn.endswith('py'):
        raise ValueError(f'`test_file` should be a python file. Got {test_fn} instead.')
    if not test_fn.startswith('test_modeling_'):
        raise ValueError(f'`test_file` should point to a file name of the form `test_modeling_*.py`. Got {test_fn} instead.')
    components = components[:-1] + [test_fn.replace('.py', '')]
    test_module_path = '.'.join(components)
    return test_module_path

def get_test_module(test_file):
    if False:
        print('Hello World!')
    'Get the module of a model test file.'
    test_module_path = get_module_path(test_file)
    test_module = importlib.import_module(test_module_path)
    return test_module

def get_tester_classes(test_file):
    if False:
        for i in range(10):
            print('nop')
    'Get all classes in a model test file whose names ends with `ModelTester`.'
    tester_classes = []
    test_module = get_test_module(test_file)
    for attr in dir(test_module):
        if attr.endswith('ModelTester'):
            tester_classes.append(getattr(test_module, attr))
    return sorted(tester_classes, key=lambda x: x.__name__)

def get_test_classes(test_file):
    if False:
        i = 10
        return i + 15
    'Get all [test] classes in a model test file with attribute `all_model_classes` that are non-empty.\n\n    These are usually the (model) test classes containing the (non-slow) tests to run and are subclasses of one of the\n    classes `ModelTesterMixin`, `TFModelTesterMixin` or `FlaxModelTesterMixin`, as well as a subclass of\n    `unittest.TestCase`. Exceptions include `RagTestMixin` (and its subclasses).\n    '
    test_classes = []
    test_module = get_test_module(test_file)
    for attr in dir(test_module):
        attr_value = getattr(test_module, attr)
        model_classes = getattr(attr_value, 'all_model_classes', [])
        if len(model_classes) > 0:
            test_classes.append(attr_value)
    return sorted(test_classes, key=lambda x: x.__name__)

def get_model_classes(test_file):
    if False:
        return 10
    'Get all model classes that appear in `all_model_classes` attributes in a model test file.'
    test_classes = get_test_classes(test_file)
    model_classes = set()
    for test_class in test_classes:
        model_classes.update(test_class.all_model_classes)
    return sorted(model_classes, key=lambda x: x.__name__)

def get_model_tester_from_test_class(test_class):
    if False:
        i = 10
        return i + 15
    'Get the model tester class of a model test class.'
    test = test_class()
    if hasattr(test, 'setUp'):
        test.setUp()
    model_tester = None
    if hasattr(test, 'model_tester'):
        if test.model_tester is not None:
            model_tester = test.model_tester.__class__
    return model_tester

def get_test_classes_for_model(test_file, model_class):
    if False:
        i = 10
        return i + 15
    'Get all [test] classes in `test_file` that have `model_class` in their `all_model_classes`.'
    test_classes = get_test_classes(test_file)
    target_test_classes = []
    for test_class in test_classes:
        if model_class in test_class.all_model_classes:
            target_test_classes.append(test_class)
    return sorted(target_test_classes, key=lambda x: x.__name__)

def get_tester_classes_for_model(test_file, model_class):
    if False:
        i = 10
        return i + 15
    'Get all model tester classes in `test_file` that are associated to `model_class`.'
    test_classes = get_test_classes_for_model(test_file, model_class)
    tester_classes = []
    for test_class in test_classes:
        tester_class = get_model_tester_from_test_class(test_class)
        if tester_class is not None:
            tester_classes.append(tester_class)
    return sorted(tester_classes, key=lambda x: x.__name__)

def get_test_to_tester_mapping(test_file):
    if False:
        for i in range(10):
            print('nop')
    'Get a mapping from [test] classes to model tester classes in `test_file`.\n\n    This uses `get_test_classes` which may return classes that are NOT subclasses of `unittest.TestCase`.\n    '
    test_classes = get_test_classes(test_file)
    test_tester_mapping = {test_class: get_model_tester_from_test_class(test_class) for test_class in test_classes}
    return test_tester_mapping

def get_model_to_test_mapping(test_file):
    if False:
        for i in range(10):
            print('nop')
    'Get a mapping from model classes to test classes in `test_file`.'
    model_classes = get_model_classes(test_file)
    model_test_mapping = {model_class: get_test_classes_for_model(test_file, model_class) for model_class in model_classes}
    return model_test_mapping

def get_model_to_tester_mapping(test_file):
    if False:
        i = 10
        return i + 15
    'Get a mapping from model classes to model tester classes in `test_file`.'
    model_classes = get_model_classes(test_file)
    model_to_tester_mapping = {model_class: get_tester_classes_for_model(test_file, model_class) for model_class in model_classes}
    return model_to_tester_mapping

def to_json(o):
    if False:
        i = 10
        return i + 15
    "Make the information succinct and easy to read.\n\n    Avoid the full class representation like `<class 'transformers.models.bert.modeling_bert.BertForMaskedLM'>` when\n    displaying the results. Instead, we use class name (`BertForMaskedLM`) for the readability.\n    "
    if isinstance(o, str):
        return o
    elif isinstance(o, type):
        return o.__name__
    elif isinstance(o, (list, tuple)):
        return [to_json(x) for x in o]
    elif isinstance(o, dict):
        return {to_json(k): to_json(v) for (k, v) in o.items()}
    else:
        return o