"""The service for gating features.

This service provides different interfaces to access the feature flag values
for clients and the backend respectively as they have different context for
evaluation of feature flag values.

For clients, please use 'evaluate_all_feature_flag_values_for_client' from
request handlers with client context.

For the backend, please directly call 'is_feature_enabled' with the name of
the feature.

For more details of the usage of these two methods, please refer their
docstrings in this file.
"""
from __future__ import annotations
import copy
import json
import os
from core import feconf
from core import platform_feature_list
from core import utils
from core.constants import constants
from core.domain import platform_parameter_domain
from core.domain import platform_parameter_registry as registry
from typing import Dict, Final, List, Set
ALL_FEATURE_FLAGS: List[platform_feature_list.ParamNames] = platform_feature_list.DEV_FEATURES_LIST + platform_feature_list.TEST_FEATURES_LIST + platform_feature_list.PROD_FEATURES_LIST
ALL_FEATURES_NAMES_SET: Set[str] = set((feature.value for feature in ALL_FEATURE_FLAGS))
DATA_TYPE_TO_SCHEMA_TYPE: Dict[str, str] = {'number': 'float', 'string': 'unicode', 'bool': 'bool'}
PACKAGE_JSON_FILE_PATH: Final = os.path.join(os.getcwd(), 'package.json')

class FeatureFlagNotFoundException(Exception):
    """Exception thrown when an unknown feature flag is requested."""
    pass

class PlatformParameterNotFoundException(Exception):
    """Exception thrown when an unknown platform parameter is requested."""
    pass

def create_evaluation_context_for_client(client_context_dict: platform_parameter_domain.ClientSideContextDict) -> platform_parameter_domain.EvaluationContext:
    if False:
        return 10
    'Returns context instance for evaluation, using the information\n    provided by clients.\n\n    Args:\n        client_context_dict: dict. The client side context.\n\n    Returns:\n        EvaluationContext. The context for evaluation.\n    '
    return platform_parameter_domain.EvaluationContext.from_dict(client_context_dict, {'server_mode': get_server_mode()})

def get_all_feature_flag_dicts() -> List[platform_parameter_domain.PlatformParameterDict]:
    if False:
        for i in range(10):
            print('nop')
    'Returns dict representations of all feature flags. This method is used\n    for providing detailed feature flags information to the admin panel.\n\n    Returns:\n        list(dict). A list containing the dict mappings of all fields of the\n        feature flags.\n    '
    return [registry.Registry.get_platform_parameter(_feature.value).to_dict() for _feature in ALL_FEATURE_FLAGS]

def get_all_platform_parameters_except_feature_flag_dicts() -> List[platform_parameter_domain.PlatformParameterDict]:
    if False:
        print('Hello World!')
    'Returns dict representations of all platform parameters that do not\n    contains feature flags. This method is used for providing detailed\n    platform parameters information to the release-coordinator page.\n\n    Returns:\n        list(dict). A list containing the dict mappings of all fields of the\n        platform parameters.\n    '
    return [registry.Registry.get_platform_parameter(_plat_param.value).to_dict() for _plat_param in platform_feature_list.ALL_PLATFORM_PARAMS_EXCEPT_FEATURE_FLAGS]

def evaluate_all_feature_flag_values_for_client(context: platform_parameter_domain.EvaluationContext) -> Dict[str, bool]:
    if False:
        return 10
    'Evaluates and returns the values for all feature flags.\n\n    Args:\n        context: EvaluationContext. The context used for evaluation.\n\n    Returns:\n        dict. The keys are the feature names and the values are boolean\n        results of corresponding flags.\n    '
    return _evaluate_feature_flag_values_for_context(ALL_FEATURES_NAMES_SET, context)

def is_feature_enabled(feature_name: str) -> bool:
    if False:
        while True:
            i = 10
    "A short-form method for server-side usage. This method evaluates and\n    returns the values of the feature flag, using context from the server only.\n\n    Args:\n        feature_name: str. The name of the feature flag that needs to\n            be evaluated.\n\n    Returns:\n        bool. The value of the feature flag, True if it's enabled.\n    "
    return _evaluate_feature_flag_value_for_server(feature_name)

def update_feature_flag(feature_name: str, committer_id: str, commit_message: str, new_rules: List[platform_parameter_domain.PlatformParameterRule]) -> None:
    if False:
        while True:
            i = 10
    "Updates the feature flag's rules.\n\n    Args:\n        feature_name: str. The name of the feature to update.\n        committer_id: str. ID of the committer.\n        commit_message: str. The commit message.\n        new_rules: list(PlatformParameterRule). A list of PlatformParameterRule\n            objects to update.\n\n    Raises:\n        FeatureFlagNotFoundException. The feature_name is not registered in\n            core/platform_feature_list.py.\n    "
    if feature_name not in ALL_FEATURES_NAMES_SET:
        raise FeatureFlagNotFoundException('Unknown feature flag: %s.' % feature_name)
    registry.Registry.update_platform_parameter(feature_name, committer_id, commit_message, new_rules, False)

def get_server_mode() -> platform_parameter_domain.ServerMode:
    if False:
        for i in range(10):
            print('nop')
    'Returns the running mode of Oppia.\n\n    Returns:\n        Enum(SERVER_MODES). The server mode of Oppia. This is "dev" if Oppia is\n        running in development mode, "test" if Oppia is running in production\n        mode but not on the main website, and "prod" if Oppia is running in\n        full production mode on the main website.\n    '
    return platform_parameter_domain.ServerMode.DEV if constants.DEV_MODE else platform_parameter_domain.ServerMode.PROD if feconf.ENV_IS_OPPIA_ORG_PRODUCTION_SERVER else platform_parameter_domain.ServerMode.TEST

def _create_evaluation_context_for_server() -> platform_parameter_domain.EvaluationContext:
    if False:
        return 10
    'Returns evaluation context with information of the server.\n\n    Returns:\n        EvaluationContext. The context for evaluation.\n    '
    current_app_version = json.load(utils.open_file(PACKAGE_JSON_FILE_PATH, 'r'))['version']
    if not constants.BRANCH_NAME == '' and 'release' in constants.BRANCH_NAME:
        current_app_version = constants.BRANCH_NAME.split('release-')[1]
        if 'hotfix' in current_app_version:
            split_via_hotfix = current_app_version.split('-hotfix')
            current_app_version = split_via_hotfix[0].replace('-', '.') + '-hotfix' + split_via_hotfix[1]
        else:
            current_app_version = current_app_version.replace('-', '.')
    return platform_parameter_domain.EvaluationContext.from_dict({'platform_type': 'Web', 'app_version': current_app_version}, {'server_mode': get_server_mode()})

def _evaluate_feature_flag_values_for_context(feature_names_set: Set[str], context: platform_parameter_domain.EvaluationContext) -> Dict[str, bool]:
    if False:
        print('Hello World!')
    "Evaluates and returns the values for specified feature flags.\n\n    Args:\n        feature_names_set: set(str). The set of names of feature flags that need\n            to be evaluated.\n        context: EvaluationContext. The context used for evaluation.\n\n    Returns:\n        dict. The keys are the feature names and the values are boolean\n        results of corresponding flags.\n\n    Raises:\n        FeatureFlagNotFoundException. Some names in 'feature_names_set' are not\n            registered in core/platform_feature_list.py.\n    "
    unknown_feature_names = list(feature_names_set - ALL_FEATURES_NAMES_SET)
    if len(unknown_feature_names) > 0:
        raise FeatureFlagNotFoundException('Unknown feature flag(s): %s.' % unknown_feature_names)
    result_dict = {}
    for feature_name in feature_names_set:
        param = registry.Registry.get_platform_parameter(feature_name)
        feature_is_enabled = param.evaluate(context)
        assert isinstance(feature_is_enabled, bool)
        result_dict[feature_name] = feature_is_enabled
    return result_dict

def _evaluate_feature_flag_value_for_server(feature_name: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    "Evaluates and returns the values of the feature flag, using context\n    from the server only.\n\n    Args:\n        feature_name: str. The name of the feature flag that needs to\n            be evaluated.\n\n    Returns:\n        bool. The value of the feature flag, True if it's enabled.\n    "
    context = _create_evaluation_context_for_server()
    values_dict = _evaluate_feature_flag_values_for_context(set([feature_name]), context)
    return values_dict[feature_name]

def get_platform_parameter_value(parameter_name: str) -> platform_parameter_domain.PlatformDataTypes:
    if False:
        i = 10
        return i + 15
    'Returns the value of the platform parameter.\n\n    Args:\n        parameter_name: str. The name of the platform parameter whose\n            value is required.\n\n    Returns:\n        PlatformDataTypes. The value of the platform parameter.\n\n    Raises:\n        PlatformParameterNotFoundException. Platform parameter is not valid.\n    '
    all_platform_params_dicts = get_all_platform_parameters_except_feature_flag_dicts()
    all_platform_params_names_set = set((param['name'] for param in all_platform_params_dicts))
    if parameter_name not in all_platform_params_names_set:
        raise PlatformParameterNotFoundException('Unknown platform parameter: %s.' % parameter_name)
    context = _create_evaluation_context_for_server()
    param = registry.Registry.get_platform_parameter(parameter_name)
    return param.evaluate(context)

def get_platform_parameter_schema(param_name: str) -> Dict[str, str]:
    if False:
        while True:
            i = 10
    'Returns the schema for the platform parameter.\n\n    Args:\n        param_name: str. The name of the platform parameter.\n\n    Returns:\n        Dict[str, str]. The schema of the platform parameter according\n        to the data_type.\n\n    Raises:\n        Exception. The platform parameter does not have valid data type.\n    '
    parameter = registry.Registry.get_platform_parameter(param_name)
    if DATA_TYPE_TO_SCHEMA_TYPE.get(parameter.data_type) is not None:
        schema_type = copy.deepcopy(DATA_TYPE_TO_SCHEMA_TYPE[parameter.data_type])
        return {'type': schema_type}
    else:
        raise Exception('The %s platform parameter has a data type of %s which is not valid. Please use one of these data types instead: %s.' % (parameter.name, parameter.data_type, platform_parameter_domain.PlatformDataTypes))