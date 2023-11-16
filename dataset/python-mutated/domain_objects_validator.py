"""Methods for validating domain objects for schema validation of
handler arguments.
"""
from __future__ import annotations
from core import utils
from core.constants import constants
from core.controllers import base
from core.domain import blog_domain
from core.domain import blog_services
from core.domain import change_domain
from core.domain import config_domain
from core.domain import exp_domain
from core.domain import image_validation_services
from core.domain import improvements_domain
from core.domain import platform_parameter_domain
from core.domain import platform_parameter_list
from core.domain import platform_parameter_registry
from core.domain import question_domain
from core.domain import skill_domain
from core.domain import state_domain
from core.domain import stats_domain
from typing import Dict, Mapping, Optional, Union

def validate_suggestion_change(obj: Mapping[str, change_domain.AcceptableChangeDictTypes]) -> Mapping[str, change_domain.AcceptableChangeDictTypes]:
    if False:
        return 10
    'Validates Exploration or Question change.\n\n    Args:\n        obj: dict. Data that needs to be validated.\n\n    Returns:\n        dict. Returns suggestion change dict after validation.\n    '
    if obj.get('cmd') is None:
        raise base.BaseHandler.InvalidInputException('Missing cmd key in change dict')
    exp_change_commands = [command['name'] for command in exp_domain.ExplorationChange.ALLOWED_COMMANDS]
    question_change_commands = [command['name'] for command in question_domain.QuestionChange.ALLOWED_COMMANDS]
    if obj['cmd'] in exp_change_commands:
        exp_domain.ExplorationChange(obj)
    elif obj['cmd'] in question_change_commands:
        question_domain.QuestionSuggestionChange(obj)
    else:
        raise base.BaseHandler.InvalidInputException('%s cmd is not allowed.' % obj['cmd'])
    return obj

def validate_new_config_property_values(new_config_property: Mapping[str, config_domain.AllowedDefaultValueTypes]) -> Mapping[str, config_domain.AllowedDefaultValueTypes]:
    if False:
        return 10
    "Validates new config property values.\n\n    Args:\n        new_config_property: dict. Data that needs to be validated.\n\n    Returns:\n        dict(str, *). Returns a dict for new config properties.\n\n    Raises:\n        Exception. The config property name is not a string.\n        Exception. The value corresponding to config property name\n            don't have any schema.\n    "
    for (name, value) in new_config_property.items():
        if not isinstance(name, str):
            raise Exception('config property name should be a string, received: %s' % name)
        config_property = config_domain.Registry.get_config_property(name)
        if config_property is None:
            raise Exception('%s do not have any schema.' % name)
        config_property.normalize(value)
    return new_config_property

def validate_platform_params_values_for_blog_admin(new_platform_parameter_values: Mapping[str, platform_parameter_domain.PlatformDataTypes]) -> Mapping[str, platform_parameter_domain.PlatformDataTypes]:
    if False:
        i = 10
        return i + 15
    'Validates new platform parameter values.\n\n    Args:\n        new_platform_parameter_values: dict. Data that needs to be validated.\n\n    Returns:\n        dict(str, PlatformDataTypes). Returns the dict after validation.\n\n    Raises:\n        Exception. The name of the platform parameter is not of type string.\n        Exception. The value of the platform parameter is not of valid type.\n        Exception. The max_number_of_tags_assigned_to_blog_post platform\n            parameter has incoming value less than or equal to 0.\n    '
    for (name, value) in new_platform_parameter_values.items():
        if not isinstance(name, str):
            raise Exception('Platform parameter name should be a string, received: %s' % name)
        if not isinstance(value, (bool, float, int, str)):
            raise Exception('The value of %s platform parameter is not of valid type, it should be one of %s.' % (name, str(platform_parameter_domain.PlatformDataTypes)))
        parameter = platform_parameter_registry.Registry.get_platform_parameter(name)
        if not (isinstance(value, bool) and parameter.data_type == 'bool' or (isinstance(value, str) and parameter.data_type == 'string') or (isinstance(value, float) and parameter.data_type == 'number') or (isinstance(value, int) and parameter.data_type == 'number')):
            raise Exception("The value of platform parameter %s is of type '%s', expected it to be of type '%s'" % (name, value, parameter.data_type))
        if name == platform_parameter_list.ParamNames.MAX_NUMBER_OF_TAGS_ASSIGNED_TO_BLOG_POST.value:
            assert isinstance(value, int)
            if value <= 0:
                raise Exception('The value of %s should be greater than 0, it is %s.' % (name, value))
    return new_platform_parameter_values

def validate_new_default_value_of_platform_parameter(default_value: Mapping[str, platform_parameter_domain.PlatformDataTypes]) -> Mapping[str, platform_parameter_domain.PlatformDataTypes]:
    if False:
        return 10
    'Validates new default value of platform parameter.\n\n    Args:\n        default_value: dict. Data that needs to be validated.\n\n    Returns:\n        dict(str, PlatformDataTypes). Returns the default value dict after\n        validating.\n\n    Raises:\n        Exception. The default_value is not of valid type.\n    '
    if not isinstance(default_value['value'], (bool, float, int, str)):
        raise Exception('Expected type to be %s but received %s' % (platform_parameter_domain.PlatformDataTypes, default_value['value']))
    return default_value

def validate_change_dict_for_blog_post(change_dict: blog_services.BlogPostChangeDict) -> blog_services.BlogPostChangeDict:
    if False:
        print('Hello World!')
    'Validates change_dict required for updating values of blog post.\n\n    Args:\n        change_dict: dict. Data that needs to be validated.\n\n    Returns:\n        dict. Returns the change_dict after validation.\n\n    Raises:\n        Exception. Invalid tags provided.\n    '
    if 'title' in change_dict:
        blog_domain.BlogPost.require_valid_title(change_dict['title'], True)
    if 'thumbnail_filename' in change_dict:
        blog_domain.BlogPost.require_valid_thumbnail_filename(change_dict['thumbnail_filename'])
    if 'tags' in change_dict:
        blog_domain.BlogPost.require_valid_tags(change_dict['tags'], False)
        list_of_default_tags = constants.LIST_OF_DEFAULT_TAGS_FOR_BLOG_POST
        assert list_of_default_tags is not None
        list_of_default_tags_value = list_of_default_tags
        if not all((tag in list_of_default_tags_value for tag in change_dict['tags'])):
            raise Exception('Invalid tags provided. Tags not in default tags list.')
    return change_dict

def validate_state_dict(state_dict: state_domain.StateDict) -> state_domain.StateDict:
    if False:
        print('Hello World!')
    'Validates state dict.\n\n    Args:\n        state_dict: dict. The dict representation of State object.\n\n    Returns:\n        State. The state_dict after validation.\n    '
    state_object = state_domain.State.from_dict(state_dict)
    state_object.validate(exp_param_specs_dict=None, allow_null_interaction=True)
    return state_dict

def validate_email_dashboard_data(data: Dict[str, Optional[Union[bool, int]]]) -> Dict[str, Optional[Union[bool, int]]]:
    if False:
        return 10
    "Validates email dashboard data.\n\n    Args:\n        data: dict. Data that needs to be validated.\n\n    Returns:\n        dict. Returns the dict after validation.\n\n    Raises:\n        Exception. The key in 'data' is not one of the allowed keys.\n    "
    predicates = constants.EMAIL_DASHBOARD_PREDICATE_DEFINITION
    possible_keys = [predicate['backend_attr'] for predicate in predicates]
    for (key, value) in data.items():
        if value is None:
            continue
        if key not in possible_keys:
            raise Exception('400 Invalid input for query.')
    return data

def validate_task_entries(task_entries: improvements_domain.TaskEntryDict) -> improvements_domain.TaskEntryDict:
    if False:
        while True:
            i = 10
    'Validates the task entry dict.\n\n    Args:\n        task_entries: dict. Data that needs to be validated.\n\n    Returns:\n        dict. Returns the task entries dict after validation.\n    '
    entity_version = task_entries.get('entity_version', None)
    if entity_version is None:
        raise base.BaseHandler.InvalidInputException('No entity_version provided')
    task_type = task_entries.get('task_type', None)
    if task_type is None:
        raise base.BaseHandler.InvalidInputException('No task_type provided')
    target_id = task_entries.get('target_id', None)
    if target_id is None:
        raise base.BaseHandler.InvalidInputException('No target_id provided')
    status = task_entries.get('status', None)
    if status is None:
        raise base.BaseHandler.InvalidInputException('No status provided')
    return task_entries

def validate_aggregated_stats(aggregated_stats: stats_domain.AggregatedStatsDict) -> stats_domain.AggregatedStatsDict:
    if False:
        for i in range(10):
            print('nop')
    'Validates the attribute stats dict.\n\n    Args:\n        aggregated_stats: dict. Data that needs to be validated.\n\n    Returns:\n        dict. Data after validation.\n\n    Raises:\n        InvalidInputException. Property not in aggregated stats dict.\n    '
    return stats_domain.SessionStateStats.validate_aggregated_stats_dict(aggregated_stats)

def validate_suggestion_images(files: Dict[str, bytes]) -> Dict[str, bytes]:
    if False:
        i = 10
        return i + 15
    'Validates the files dict.\n\n    Args:\n        files: dict. Data that needs to be validated.\n\n    Returns:\n        dict. Returns the dict after validation.\n    '
    for (filename, raw_image) in files.items():
        image_validation_services.validate_image_and_filename(raw_image, filename)
    return files

def validate_skill_ids(comma_separated_skill_ids: str) -> str:
    if False:
        return 10
    'Checks whether the given skill ids are valid.\n\n    Args:\n        comma_separated_skill_ids: str. Comma separated skill IDs.\n\n    Returns:\n        str. The comma separated skill ids after validation.\n    '
    skill_ids = comma_separated_skill_ids.split(',')
    skill_ids = list(set(skill_ids))
    try:
        for skill_id in skill_ids:
            skill_domain.Skill.require_valid_skill_id(skill_id)
    except utils.ValidationError as e:
        raise base.BaseHandler.InvalidInputException('Invalid skill id') from e
    return comma_separated_skill_ids