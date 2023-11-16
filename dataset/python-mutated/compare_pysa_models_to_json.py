"""
TODO(T132414938) Add a module-level docstring
"""
import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple, TypedDict
LOG: logging.Logger = logging.getLogger(__name__)
ANNOTATION_TO_MODEL_TYPE = {'TaintSource': 'sources', 'TaintSink': 'sinks', 'TaintInTaintOut': 'tito'}
PYSA_CALLABLE_MODEL_PATTERN: Pattern[str] = re.compile('def\\s*(?P<callable_name>.+)\\((?P<parameters>.*)\\)(:?\\s*->\\s*(?P<return_model>[^:]+))*(:?:\\s*...)*')
PYSA_ATTRIBUTE_MODEL_PATTERN: Pattern[str] = re.compile('(?P<attribute_name>.+):\\s*(?P<attribute_model>.*\\[.*\\])(= ...)*')
PARAMETERS_ANNOTATION_PATTERN: Pattern[str] = re.compile('(\\w*):\\s?(\\w*)\\[((\\w+)(,\\s*\\w+)*)\\]')
RETURN_ANNOTATION_PATTERN: Pattern[str] = re.compile('(?P<model_type>.*)\\[(?P<model_leaves>.*)\\]')

class TaintModel(TypedDict):
    sources: Set[str]
    sinks: Set[str]
    tito: Set[str]

class TargetModel(TypedDict):
    parameters: Dict[str, TaintModel]
    return_model: TaintModel

def make_default_taint_model() -> TaintModel:
    if False:
        return 10
    return {'sources': set(), 'sinks': set(), 'tito': set()}

def make_default_target_model() -> TargetModel:
    if False:
        print('Hello World!')
    return {'parameters': defaultdict(make_default_taint_model), 'return_model': make_default_taint_model()}

def parse_kinds(taints: List[Dict[str, Any]]) -> Set[str]:
    if False:
        i = 10
        return i + 15
    '\n    Parse the list of sources/sinks/tito from a Pysa JSON output\n    dump, e.g.\n        [ { "decl": null, "kinds": [ { "kind": "Test" } ]\n    into a set consisting of just the leaf names, i.e.\n        { "Test" }\n    '
    kinds = set()
    for taint in taints:
        for kind in taint.get('kinds', []):
            kinds.add(kind['kind'])
        for kind in taint.get('leaves', []):
            kinds.add(kind['kind'])
    return kinds

def json_to_parsed_model(taint_data: List[Dict[str, Any]]) -> TargetModel:
    if False:
        print('Hello World!')
    '\n    Parse the list of taint models from a Pysa JSON output dump, e.g.\n        [{\n            "kind": "model",\n            "data": {\n                "callable": "foo.bar.some_callable",\n                "sources": [\n                    {\n                        "port": "formal(data)",\n                        "taint": [...]\n                    }\n                ]\n                "sinks": [...]\n                "tito": [...]\n            }\n        }]\n    into the form\n        {\n            \'parameters\': {\'x\': {\'sources\': {\'A\'}, \'sinks\': {}, \'titos\': {} }, ...},\n            \'return_model\': {\'sources\': {}, \'sinks\': {\'B\'}, \'tito\': {}}\n        }\n    '
    result: TargetModel = make_default_target_model()
    for data in taint_data:
        if 'data' not in data:
            continue
        model = data['data']
        for model_type in ANNOTATION_TO_MODEL_TYPE.values():
            if model_type in model:
                for entry in model[model_type]:
                    port = entry['port']
                    taints = parse_kinds(entry['taint'])
                    if port == 'result':
                        result['return_model'][model_type].update(taints)
                    else:
                        port = entry['port'].replace('formal(', '').replace(')', '')
                        result['parameters'][port][model_type].update(taints)
    return result

def get_models_from_json_file(path: str) -> Dict[str, TargetModel]:
    if False:
        for i in range(10):
            print('nop')
    "\n    Process a JSON file and return a dictionary of callables and their models,\n    in the form:\n        {\n            'parameters': {'x': {'TaintSource[A]'}},\n            'return_model': {'TaintSink[B]'}\n        }\n    "
    json_models: Dict[str, TargetModel] = defaultdict(make_default_target_model)
    with Path(path).open() as json_file:
        for entry in json.loads(json_file.read()):
            for (_, models) in entry.items():
                for json_model in models:
                    callable_name = json_model['callable']
                    model = json_to_parsed_model(json_model['model'])
                    json_models[callable_name]['parameters'].update(model['parameters'])
                    json_models[callable_name]['return_model'].update(model['return_model'])
    return json_models

def get_callable_model_from_line(line: str) -> Optional[Tuple[str, TargetModel]]:
    if False:
        i = 10
        return i + 15
    match = PYSA_CALLABLE_MODEL_PATTERN.match(line)
    if not match:
        return None
    result = make_default_target_model()
    callable_name = match.group('callable_name')
    parameters = match.group('parameters')
    return_model = match.group('return_model')
    if not callable_name and (not parameters and (not return_model)):
        return None
    annotated_parameters = PARAMETERS_ANNOTATION_PATTERN.findall(parameters)
    for (parameter_name, model_annotation, leaves, _, _) in annotated_parameters:
        if not parameter_name or not model_annotation or (not leaves):
            continue
        model_type = ANNOTATION_TO_MODEL_TYPE[model_annotation]
        parameter_model = {annotation.strip() for annotation in leaves.split(',')}
        result['parameters'][parameter_name][model_type].update(parameter_model)
    if return_model:
        annotation_match = RETURN_ANNOTATION_PATTERN.match(return_model)
        if not annotation_match or None in annotation_match.groups():
            return None
        model_type = ANNOTATION_TO_MODEL_TYPE[annotation_match.group('model_type').strip()]
        return_model = {annotation.strip() for annotation in annotation_match.group('model_leaves').split(',')}
        result['return_model'][model_type].update(return_model)
    return (callable_name, result)

def get_attribute_model_from_line(line: str) -> Optional[Tuple[str, TargetModel]]:
    if False:
        print('Hello World!')
    match = PYSA_ATTRIBUTE_MODEL_PATTERN.match(line)
    if not match:
        return None
    result = make_default_target_model()
    attribute_name = 'Obj{{{}}}'.format(match.group('attribute_name'))
    attribute_model = match.group('attribute_model')
    if not attribute_name or not attribute_model:
        return None
    annotation_match = RETURN_ANNOTATION_PATTERN.match(attribute_model)
    if not annotation_match or None in annotation_match.groups():
        return None
    model_type = ANNOTATION_TO_MODEL_TYPE[annotation_match.group('model_type').strip()]
    attribute_model_leaves = {annotation.strip() for annotation in annotation_match.group('model_leaves').split(', ')}
    if model_type == 'sources':
        result['return_model'][model_type].update(attribute_model_leaves)
    else:
        result['parameters']['$global'][model_type].update(attribute_model_leaves)
    return (attribute_name, result)

def get_models_from_pysa_file(path: str) -> Dict[str, TargetModel]:
    if False:
        for i in range(10):
            print('nop')
    "\n    Process a .pysa file with models in the form of:\n        def foo.bar(x: TaintSource[A], b) -> TaintSink[B]: ...\n    and return a dictionary of callables and their models in the form:\n        {\n            'parameters': {'x': {'sources': {'A', ...}, 'sinks': ... }, ...},\n            'return_model': {'sources': {}, 'sinks': {'B'}, 'tito': {}}\n        }\n    IMPORTANT: Note that this only works on .pysa files where:\n        1. All the models are self-contained on a single line.\n        2. Models do not contain ViaTag[...], AppliesTo[...] syntax\n\n    This script was originally intended to compare models that were generated\n    by the existing Python model generators, so it should be noted that\n    this will likely not work with most user-defined .pysa files.\n    "
    pysa_models: Dict[str, TargetModel] = defaultdict(make_default_target_model)
    skipped = 0
    with Path(path).open() as pysa_file:
        for line in pysa_file:
            if '[' not in line:
                skipped += 1
                continue
            if 'def ' in line:
                result = get_callable_model_from_line(line)
            else:
                result = get_attribute_model_from_line(line)
            if result:
                (name, model) = result
                for parameter in model['parameters']:
                    for model_type in model['parameters'][parameter]:
                        pysa_models[name]['parameters'][parameter][model_type].update(model['parameters'][parameter][model_type])
                pysa_models[name]['parameters'].update(model['parameters'])
                for model_type in model['return_model']:
                    pysa_models[name]['return_model'][model_type].update(model['return_model'][model_type])
            else:
                skipped += 1
    LOG.warning(f'Skipped {skipped} lines in .pysa (no models found or were invalid).')
    return pysa_models

def main() -> None:
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser(description='A script to compare models in a .pysa file to the JSON model dump generated by Pysa.')
    parser.add_argument('-j', '--json', required=True, type=str, help="Path of the JSON file containing Pysa's taint output dump.")
    parser.add_argument('-p', '--pysa', required=True, type=str, help='Path of the .pysa model file.')
    arguments = parser.parse_args()
    logging.basicConfig(format='[%(asctime)s][%(levelname)s]: %(message)s', level=logging.INFO)
    json_models: Dict[str, TargetModel] = get_models_from_json_file(arguments.json)
    pysa_models: Dict[str, TargetModel] = get_models_from_pysa_file(arguments.pysa)
    diff_json = {k: v for (k, v) in json_models.items() if not (k in pysa_models and json_models[k] == pysa_models[k])}
    diff_pysa = {k: v for (k, v) in pysa_models.items() if not (k in json_models and pysa_models[k] == json_models[k])}
    diff_pysa_test = {k: v for (k, v) in diff_pysa.items() if 'test' in k}
    diff_pysa_non_test = {k: v for (k, v) in diff_pysa.items() if 'test' not in k}
    diff_json_message = '\n'.join(['{}\nIn JSON: {}\nIn .pysa: {}\n'.format(callable_name, json_models[callable_name], pysa_models[callable_name] if callable_name in pysa_models else {}) for callable_name in sorted(diff_json.keys())])
    diff_pysa_test_message = '\n'.join(['{}\nIn .pysa: {}\nIn JSON: {}\n'.format(callable_name, pysa_models[callable_name], json_models[callable_name] if callable_name in json_models else {}) for callable_name in sorted(diff_pysa_test.keys())])
    diff_pysa_non_test_message = '\n'.join(['{}\nIn .pysa: {}\nIn JSON: {}\n'.format(callable_name, pysa_models[callable_name], json_models[callable_name] if callable_name in json_models else {}) for callable_name in sorted(diff_pysa_non_test.keys())])
    LOG.info(f'"\n-- RESULTS --\nTotal models in JSON: {len(json_models)}\nTotal models in .pysa: {len(pysa_models)}\n-------\nModels in JSON but not in .pysa: {len(diff_json)}\n{diff_json_message}\n-------\nModels in .pysa but not in JSON (test): {len(diff_pysa_test)}\n{diff_pysa_test_message}\n-------\nModels in .pysa but not in JSON (non-test): {len(diff_pysa_non_test)}\n{diff_pysa_non_test_message}\n')
if __name__ == '__main__':
    main()