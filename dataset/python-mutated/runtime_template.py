"""
All-in-one metadata about runtimes
"""
import itertools
import os
import pathlib
import re
from typing import List
_init_path = str(pathlib.Path(os.path.dirname(__file__)).parent.parent)
_templates = os.path.join(_init_path, 'lib', 'init', 'templates')
_lambda_images_templates = os.path.join(_init_path, 'lib', 'init', 'image_templates')
RUNTIME_DEP_TEMPLATE_MAPPING = {'python': [{'runtimes': ['python3.11', 'python3.10', 'python3.9', 'python3.8', 'python3.7'], 'dependency_manager': 'pip', 'init_location': os.path.join(_templates, 'cookiecutter-aws-sam-hello-python'), 'build': True}], 'ruby': [{'runtimes': ['ruby3.2', 'ruby2.7'], 'dependency_manager': 'bundler', 'init_location': os.path.join(_templates, 'cookiecutter-aws-sam-hello-ruby'), 'build': True}], 'nodejs': [{'runtimes': ['nodejs20.x', 'nodejs18.x', 'nodejs16.x', 'nodejs14.x', 'nodejs12.x'], 'dependency_manager': 'npm', 'init_location': os.path.join(_templates, 'cookiecutter-aws-sam-hello-nodejs'), 'build': True}], 'dotnet': [{'runtimes': ['dotnet6'], 'dependency_manager': 'cli-package', 'init_location': os.path.join(_templates, 'cookiecutter-aws-sam-hello-dotnet'), 'build': True}], 'go': [{'runtimes': ['go1.x'], 'dependency_manager': 'mod', 'init_location': os.path.join(_templates, 'cookiecutter-aws-sam-hello-golang'), 'build': False}], 'java': [{'runtimes': ['java11', 'java8', 'java8.al2', 'java17'], 'dependency_manager': 'maven', 'init_location': os.path.join(_templates, 'cookiecutter-aws-sam-hello-java-maven'), 'build': True}, {'runtimes': ['java11', 'java8', 'java8.al2', 'java17'], 'dependency_manager': 'gradle', 'init_location': os.path.join(_templates, 'cookiecutter-aws-sam-hello-java-gradle'), 'build': True}]}

def get_local_manifest_path():
    if False:
        print('Hello World!')
    return pathlib.Path(_init_path, 'lib', 'init', 'local_manifest.json')

def get_local_lambda_images_location(mapping, runtime):
    if False:
        while True:
            i = 10
    dir_name = os.path.basename(mapping['init_location'])
    if dir_name.endswith('-lambda-image'):
        return os.path.join(_lambda_images_templates, runtime, dir_name)
    return os.path.join(_lambda_images_templates, runtime, dir_name + '-lambda-image')
SUPPORTED_DEP_MANAGERS: List[str] = sorted(list(set({c.get('dependency_manager') for c in list(itertools.chain(*RUNTIME_DEP_TEMPLATE_MAPPING.values())) if c.get('dependency_manager')})))
INIT_RUNTIMES = ['dotnet6', 'go1.x', 'java17', 'java11', 'java8.al2', 'java8', 'nodejs20.x', 'nodejs18.x', 'nodejs16.x', 'nodejs14.x', 'nodejs12.x', 'provided.al2023', 'provided.al2', 'provided', 'python3.11', 'python3.10', 'python3.9', 'python3.8', 'python3.7', 'ruby3.2', 'ruby2.7']
LAMBDA_IMAGES_RUNTIMES_MAP = {'dotnet6': 'amazon/dotnet6-base', 'go1.x': 'amazon/go1.x-base', 'go (provided.al2)': 'amazon/go-provided.al2-base', 'go (provided.al2023)': 'amazon/go-provided.al2023-base', 'java17': 'amazon/java17-base', 'java11': 'amazon/java11-base', 'java8.al2': 'amazon/java8.al2-base', 'java8': 'amazon/java8-base', 'nodejs20.x': 'amazon/nodejs20.x-base', 'nodejs18.x': 'amazon/nodejs18.x-base', 'nodejs16.x': 'amazon/nodejs16.x-base', 'nodejs14.x': 'amazon/nodejs14.x-base', 'nodejs12.x': 'amazon/nodejs12.x-base', 'python3.11': 'amazon/python3.11-base', 'python3.10': 'amazon/python3.10-base', 'python3.9': 'amazon/python3.9-base', 'python3.8': 'amazon/python3.8-base', 'python3.7': 'amazon/python3.7-base', 'ruby3.2': 'amazon/ruby3.2-base', 'ruby2.7': 'amazon/ruby2.7-base'}
LAMBDA_IMAGES_RUNTIMES: List = sorted(list(set(LAMBDA_IMAGES_RUNTIMES_MAP.values())))
SAM_RUNTIME_TO_SCHEMAS_CODE_LANG_MAPPING = {'java8': 'Java8', 'java8.al2': 'Java8', 'java11': 'Java8', 'java17': 'Java17', 'python3.7': 'Python36', 'python3.8': 'Python36', 'python3.9': 'Python36', 'python3.10': 'Python36', 'python3.11': 'Python36', 'dotnet6': 'dotnet6', 'go1.x': 'Go1'}
PROVIDED_RUNTIMES = ['provided.al2023', 'provided.al2', 'provided']

def is_custom_runtime(runtime):
    if False:
        return 10
    '\n    validated if a runtime is custom or not\n    Parameters\n    ----------\n    runtime : str\n        runtime to be\n    Returns\n    -------\n    _type_\n        _description_\n    '
    if not runtime:
        return False
    provided_runtime = get_provided_runtime_from_custom_runtime(runtime)
    return runtime in PROVIDED_RUNTIMES or bool(provided_runtime in PROVIDED_RUNTIMES)

def get_provided_runtime_from_custom_runtime(runtime):
    if False:
        return 10
    '\n    Gets the base lambda runtime for which a custom runtime is based on\n    Example:\n    rust (provided.al2) --> provided.al2\n    java11 --> None\n\n    Parameters\n    ----------\n    runtime : str\n        Custom runtime or Lambda runtime\n\n    Returns\n    -------\n    str\n        returns the base lambda runtime for which a custom runtime is based on\n    '
    base_runtime_list = re.findall('\\(([^()]+)\\)', runtime)
    return base_runtime_list[0] if base_runtime_list else None