"""A script to add and/or update the attribute `pipeline_model_mapping` in model test files.

This script will be (mostly) used in the following 2 situations:

  - run within a (scheduled) CI job to:
    - check if model test files in the library have updated `pipeline_model_mapping`,
    - and/or update test files and (possibly) open a GitHub pull request automatically
  - being run by a `transformers` member to quickly check and update some particular test file(s)

This script is **NOT** intended to be run (manually) by community contributors.
"""
import argparse
import glob
import inspect
import os
import re
import unittest
from get_test_info import get_test_classes
from tests.test_pipeline_mixin import pipeline_test_mapping
PIPELINE_TEST_MAPPING = {}
for (task, _) in pipeline_test_mapping.items():
    PIPELINE_TEST_MAPPING[task] = {'pt': None, 'tf': None}
TEST_FILE_TO_IGNORE = {'tests/models/esm/test_modeling_esmfold.py'}

def get_framework(test_class):
    if False:
        for i in range(10):
            print('nop')
    'Infer the framework from the test class `test_class`.'
    if 'ModelTesterMixin' in [x.__name__ for x in test_class.__bases__]:
        return 'pt'
    elif 'TFModelTesterMixin' in [x.__name__ for x in test_class.__bases__]:
        return 'tf'
    elif 'FlaxModelTesterMixin' in [x.__name__ for x in test_class.__bases__]:
        return 'flax'
    else:
        return None

def get_mapping_for_task(task, framework):
    if False:
        for i in range(10):
            print('nop')
    'Get mappings defined in `XXXPipelineTests` for the task `task`.'
    if PIPELINE_TEST_MAPPING[task].get(framework, None) is not None:
        return PIPELINE_TEST_MAPPING[task][framework]
    pipeline_test_class = pipeline_test_mapping[task]['test']
    mapping = None
    if framework == 'pt':
        mapping = getattr(pipeline_test_class, 'model_mapping', None)
    elif framework == 'tf':
        mapping = getattr(pipeline_test_class, 'tf_model_mapping', None)
    if mapping is not None:
        mapping = dict(mapping.items())
    PIPELINE_TEST_MAPPING[task][framework] = mapping
    return mapping

def get_model_for_pipeline_test(test_class, task):
    if False:
        return 10
    'Get the model architecture(s) related to the test class `test_class` for a pipeline `task`.'
    framework = get_framework(test_class)
    if framework is None:
        return None
    mapping = get_mapping_for_task(task, framework)
    if mapping is None:
        return None
    config_classes = list({model_class.config_class for model_class in test_class.all_model_classes})
    if len(config_classes) != 1:
        raise ValueError('There should be exactly one configuration class from `test_class.all_model_classes`.')
    model_class = mapping.get(config_classes[0], None)
    if isinstance(model_class, (tuple, list)):
        model_class = sorted(model_class, key=lambda x: x.__name__)
    return model_class

def get_pipeline_model_mapping(test_class):
    if False:
        return 10
    'Get `pipeline_model_mapping` for `test_class`.'
    mapping = [(task, get_model_for_pipeline_test(test_class, task)) for task in pipeline_test_mapping]
    mapping = sorted([(task, model) for (task, model) in mapping if model is not None], key=lambda x: x[0])
    return dict(mapping)

def get_pipeline_model_mapping_string(test_class):
    if False:
        return 10
    'Get `pipeline_model_mapping` for `test_class` as a string (to be added to the test file).\n\n    This will be a 1-line string. After this is added to a test file, `make style` will format it beautifully.\n    '
    framework = get_framework(test_class)
    if framework == 'pt':
        framework = 'torch'
    default_value = '{}'
    mapping = get_pipeline_model_mapping(test_class)
    if len(mapping) == 0:
        return ''
    texts = []
    for (task, model_classes) in mapping.items():
        if isinstance(model_classes, (tuple, list)):
            value = '(' + ', '.join([x.__name__ for x in model_classes]) + ')'
        else:
            value = model_classes.__name__
        texts.append(f'"{task}": {value}')
    text = '{' + ', '.join(texts) + '}'
    text = f'pipeline_model_mapping = {text} if is_{framework}_available() else {default_value}'
    return text

def is_valid_test_class(test_class):
    if False:
        print('Hello World!')
    'Restrict to `XXXModelTesterMixin` and should be a subclass of `unittest.TestCase`.'
    base_class_names = {'ModelTesterMixin', 'TFModelTesterMixin', 'FlaxModelTesterMixin'}
    if not issubclass(test_class, unittest.TestCase):
        return False
    return len(base_class_names.intersection([x.__name__ for x in test_class.__bases__])) > 0

def find_test_class(test_file):
    if False:
        for i in range(10):
            print('nop')
    'Find a test class in `test_file` to which we will add `pipeline_model_mapping`.'
    test_classes = [x for x in get_test_classes(test_file) if is_valid_test_class(x)]
    target_test_class = None
    for test_class in test_classes:
        if getattr(test_class, 'pipeline_model_mapping', None) is not None:
            target_test_class = test_class
            break
    if target_test_class is None and len(test_classes) > 0:
        target_test_class = sorted(test_classes, key=lambda x: (len(x.__name__), x.__name__))[0]
    return target_test_class

def find_block_ending(lines, start_idx, indent_level):
    if False:
        for i in range(10):
            print('nop')
    end_idx = start_idx
    for (idx, line) in enumerate(lines[start_idx:]):
        indent = len(line) - len(line.lstrip())
        if idx == 0 or indent > indent_level or (indent == indent_level and line.strip() == ')'):
            end_idx = start_idx + idx
        elif idx > 0 and indent <= indent_level:
            break
    return end_idx

def add_pipeline_model_mapping(test_class, overwrite=False):
    if False:
        i = 10
        return i + 15
    'Add `pipeline_model_mapping` to `test_class`.'
    if getattr(test_class, 'pipeline_model_mapping', None) is not None:
        if not overwrite:
            return ('', -1)
    line_to_add = get_pipeline_model_mapping_string(test_class)
    if len(line_to_add) == 0:
        return ('', -1)
    line_to_add = line_to_add + '\n'
    (class_lines, class_start_line_no) = inspect.getsourcelines(test_class)
    for (idx, line) in enumerate(class_lines):
        if line.lstrip().startswith('class '):
            class_lines = class_lines[idx:]
            class_start_line_no += idx
            break
    class_end_line_no = class_start_line_no + len(class_lines) - 1
    start_idx = None
    indent_level = 0
    def_line = None
    for (idx, line) in enumerate(class_lines):
        if line.strip().startswith('all_model_classes = '):
            indent_level = len(line) - len(line.lstrip())
            start_idx = idx
        elif line.strip().startswith('all_generative_model_classes = '):
            indent_level = len(line) - len(line.lstrip())
            start_idx = idx
        elif line.strip().startswith('pipeline_model_mapping = '):
            indent_level = len(line) - len(line.lstrip())
            start_idx = idx
            def_line = line
            break
    if start_idx is None:
        return ('', -1)
    end_idx = find_block_ending(class_lines, start_idx, indent_level)
    r = re.compile('\\s(is_\\S+?_available\\(\\))\\s')
    for line in class_lines[start_idx:end_idx + 1]:
        backend_condition = r.search(line)
        if backend_condition is not None:
            target = ' ' + backend_condition[0][1:-1] + ' '
            line_to_add = r.sub(target, line_to_add)
            break
    if def_line is None:
        target_idx = end_idx
    else:
        target_idx = start_idx - 1
        for idx in range(start_idx, end_idx + 1):
            class_lines[idx] = None
    parent_classes = [x.__name__ for x in test_class.__bases__]
    if 'PipelineTesterMixin' not in parent_classes:
        _parent_classes = [x for x in parent_classes if x != 'TestCase'] + ['PipelineTesterMixin']
        if 'TestCase' in parent_classes:
            _parent_classes.append('unittest.TestCase')
        parent_classes = ', '.join(_parent_classes)
        for (idx, line) in enumerate(class_lines):
            if line.strip().endswith('):'):
                for _idx in range(idx + 1):
                    class_lines[_idx] = None
                break
        class_lines[0] = f'class {test_class.__name__}({parent_classes}):\n'
    line_to_add = ' ' * indent_level + line_to_add
    class_lines = class_lines[:target_idx + 1] + [line_to_add] + class_lines[target_idx + 1:]
    class_lines = [x for x in class_lines if x is not None]
    module_lines = inspect.getsourcelines(inspect.getmodule(test_class))[0]
    module_lines = module_lines[:class_start_line_no - 1] + class_lines + module_lines[class_end_line_no:]
    code = ''.join(module_lines)
    moddule_file = inspect.getsourcefile(test_class)
    with open(moddule_file, 'w', encoding='UTF-8', newline='\n') as fp:
        fp.write(code)
    return line_to_add

def add_pipeline_model_mapping_to_test_file(test_file, overwrite=False):
    if False:
        print('Hello World!')
    'Add `pipeline_model_mapping` to `test_file`.'
    test_class = find_test_class(test_file)
    if test_class:
        add_pipeline_model_mapping(test_class, overwrite=overwrite)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, help="A path to the test file, starting with the repository's `tests` directory.")
    parser.add_argument('--all', action='store_true', help='If to check and modify all test files.')
    parser.add_argument('--overwrite', action='store_true', help='If to overwrite a test class if it has already defined `pipeline_model_mapping`.')
    args = parser.parse_args()
    if not args.all and (not args.test_file):
        raise ValueError('Please specify either `test_file` or pass `--all` to check/modify all test files.')
    elif args.all and args.test_file:
        raise ValueError('Only one of `--test_file` and `--all` could be specified.')
    test_files = []
    if args.test_file:
        test_files = [args.test_file]
    else:
        pattern = os.path.join('tests', 'models', '**', 'test_modeling_*.py')
        for test_file in glob.glob(pattern):
            if not test_file.startswith('test_modeling_flax_'):
                test_files.append(test_file)
    for test_file in test_files:
        if test_file in TEST_FILE_TO_IGNORE:
            print(f'[SKIPPED] {test_file} is skipped as it is in `TEST_FILE_TO_IGNORE` in the file {__file__}.')
            continue
        add_pipeline_model_mapping_to_test_file(test_file, overwrite=args.overwrite)