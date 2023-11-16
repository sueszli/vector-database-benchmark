"""
Init flow based helper functions
"""
import functools
import logging
import re
from typing import Optional
from samcli.lib.utils.architecture import X86_64
from samcli.local.common.runtime_template import INIT_RUNTIMES, LAMBDA_IMAGES_RUNTIMES_MAP, is_custom_runtime
LOG = logging.getLogger(__name__)

def get_sorted_runtimes(runtime_option_list):
    if False:
        return 10
    '\n    Return a list of sorted runtimes in ascending order of runtime names and\n    descending order of runtime version.\n\n    Parameters\n    ----------\n    runtime_option_list : list\n        list of possible runtime to be selected\n\n    Returns\n    -------\n    list\n        sorted list of possible runtime to be selected\n    '
    supported_runtime_list = get_supported_runtime(runtime_option_list)
    return sorted(supported_runtime_list, key=functools.cmp_to_key(compare_runtimes))

def get_supported_runtime(runtime_list):
    if False:
        i = 10
        return i + 15
    '\n    Returns a list of only runtimes supported by the current version of SAMCLI.\n    This is the list that is presented to the customer to select from.\n\n    Parameters\n    ----------\n    runtime_list : list\n        List of runtime\n\n    Returns\n    -------\n    list\n        List of supported runtime\n    '
    supported_runtime_list = []
    error_message = ''
    for runtime in runtime_list:
        if runtime not in INIT_RUNTIMES and (not is_custom_runtime(runtime)):
            if not error_message:
                error_message = 'Additional runtimes may be available in the latest SAM CLI version.                     Upgrade your SAM CLI to see the full list.'
                LOG.debug(error_message)
            continue
        supported_runtime_list.append(runtime)
    return supported_runtime_list

def compare_runtimes(first_runtime, second_runtime):
    if False:
        for i in range(10):
            print('nop')
    '\n    Logic to compare supported runtime for sorting.\n\n    Parameters\n    ----------\n    first_runtime : str\n        runtime to be compared\n    second_runtime : str\n        runtime to be compared\n\n    Returns\n    -------\n    int\n        comparison result\n    '
    (first_runtime_name, first_version_number) = _split_runtime(first_runtime)
    (second_runtime_name, second_version_number) = _split_runtime(second_runtime)
    if first_runtime_name == second_runtime_name:
        if first_version_number == second_version_number:
            return -1 if first_runtime.endswith('.al2') else 1
        return second_version_number - first_version_number
    return 1 if first_runtime_name > second_runtime_name else -1

def _split_runtime(runtime):
    if False:
        for i in range(10):
            print('nop')
    '\n    Split a runtime into its name and version number.\n\n    Parameters\n    ----------\n    runtime : str\n        Runtime in the format supported by Lambda\n\n    Returns\n    -------\n    (str, float)\n        Tuple of runtime name and runtime version\n    '
    return (_get_runtime_name(runtime), _get_version_number(runtime))

def _get_runtime_name(runtime):
    if False:
        while True:
            i = 10
    '\n    Return the runtime name without the version\n\n    Parameters\n    ----------\n    runtime : str\n        Runtime in the format supported by Lambda.\n\n    Returns\n    -------\n    str\n        Runtime name, which is obtained as everything before the first number\n    '
    return re.split('\\d', runtime)[0]

def _get_version_number(runtime):
    if False:
        return 10
    '\n    Return the runtime version number\n\n    Parameters\n    ----------\n    runtime_version : str\n        version of a runtime\n\n    Returns\n    -------\n    float\n        Runtime version number\n    '
    if is_custom_runtime(runtime):
        return 1.0
    return float(re.search('\\d+(\\.\\d+)?', runtime).group())

def _get_templates_with_dependency_manager(templates_options, dependency_manager):
    if False:
        for i in range(10):
            print('nop')
    return [t for t in templates_options if t.get('dependencyManager') == dependency_manager]

def _get_runtime_from_image(image: str) -> Optional[str]:
    if False:
        while True:
            i = 10
    "\n    Get corresponding runtime from the base-image parameter\n\n    Expecting 'amazon/{runtime}-base'\n    But might also be like 'amazon/{runtime}-provided.al2-base'\n    "
    match = re.fullmatch('amazon/([a-z0-9.]*)-?([a-z0-9.]*)-base', image)
    if match is None:
        return None
    (runtime, base) = match.groups()
    if base:
        return f'{runtime} ({base})'
    return runtime

def _get_image_from_runtime(runtime):
    if False:
        print('Hello World!')
    '\n    Get corresponding base-image from the runtime parameter\n    '
    return LAMBDA_IMAGES_RUNTIMES_MAP[runtime]

def get_architectures(architecture):
    if False:
        return 10
    '\n    Returns list of architecture value based on the init input value\n    '
    return [X86_64] if architecture is None else [architecture]