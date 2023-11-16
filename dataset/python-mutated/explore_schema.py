import copy
import random
from collections import deque, namedtuple
from typing import Any, Deque, Dict, List, Tuple, Union
import pandas as pd
from ludwig.config_sampling.parameter_sampling import handle_property_type, ParameterBaseTypes
from ludwig.constants import SEQUENCE, TEXT, TIMESERIES
from ludwig.data.dataset_synthesizer import build_synthetic_dataset_df
from ludwig.schema.model_types.base import ModelConfig
from ludwig.types import ModelConfigDict
from ludwig.utils.misc_utils import merge_dict
NUM_SYNTHETIC_EXAMPLES = 10
ConfigOption = namedtuple('ConfigOption', ['config_option', 'fully_explored'])

def explore_properties(jsonschema_properties: Dict[str, Any], parent_parameter_path: str, dq: Deque[ConfigOption], allow_list: List[str]=[]) -> Deque[Tuple[Dict, bool]]:
    if False:
        print('Hello World!')
    'Recursively explores the `properties` part of any subsection of the schema.\n\n    Args:\n        jsonschema_properties: any properties section of the schema.\n        parent_parameter_path: period-delimited list of parent dictionary keys up to the given jsonschema_properties\n            (e.g. defaults.number.preprocessing)\n        dq: dequeue data structure that stores tuples of (config_options, fully_explored).\n            config_options: Dict[str, List], fully_explored: bool is a dictionary is a dictionary of parameter name to\n            list of values to explore.\n            fully_explored is a boolean value indicating whether all subsections of the properties dictionary have been\n            explored.\n        allow_list: list of top level keys of the properties dictionary to skip.\n\n    Returns:\n        A deque of (dict, bool) tuples.\n        - The first element of the tuple contains a dictionary of config options, which maps from a ludwig\n            config parameter to a list of the values to be explored for that parameter. Here\'s an example:\n                trainer.batch_size: ["auto", 2, 43]\n                trainer.learning_rate: ["auto", 0.1, 0.00002, 0.32424]\n                ...\n        - The second element of the tuple is whether we\'ve explored this "config path"\n            fully. This is important to track when recursing into nested structures.\n    '
    processed_dq = deque()
    while dq and (not dq[0].fully_explored):
        for (parameter_name_or_section, jsonschema_property) in jsonschema_properties.items():
            if allow_list and parameter_name_or_section not in allow_list:
                continue
            parameter_path = f'{parent_parameter_path}.{parameter_name_or_section}' if parent_parameter_path else parameter_name_or_section
            (config_options, _) = dq.popleft()
            if 'properties' in jsonschema_property and 'allOf' in jsonschema_property:
                for child_item in jsonschema_property['allOf']:
                    expanded_config_options_dq = explore_from_all_of(config_options=copy.deepcopy(config_options), item=child_item, key_so_far=parameter_path)
                    dq.extend(expanded_config_options_dq)
            elif 'properties' in jsonschema_property and 'allOf' not in jsonschema_property:
                child_properties = jsonschema_property['properties']
                raw_entry = deque([ConfigOption(copy.deepcopy(config_options), False)])
                child_config_options_dq = explore_properties(child_properties, parameter_path, raw_entry)
                merged_config_options_dq = merge_dq(config_options, child_config_options_dq)
                dq.extend(merged_config_options_dq)
            else:
                parameter_samples = get_samples(jsonschema_property)
                if parameter_samples:
                    config_options[parameter_path] = parameter_samples
                dq.appendleft(ConfigOption(config_options, False))
        while dq:
            (config_options, _) = dq.popleft()
            processed_dq.append(ConfigOption(config_options, True))
    return processed_dq

def get_samples(jsonschema_property: Dict[str, Any]) -> List[ParameterBaseTypes]:
    if False:
        i = 10
        return i + 15
    'Get possible values for a leaf property (no sub-properties).\n\n    Args:\n        jsonschema_property: leaf property in the schema. Has no sub-properties.\n    '
    if 'oneOf' in jsonschema_property:
        temp = []
        for elem in jsonschema_property['oneOf']:
            temp += get_potential_values(elem)
        return temp
    else:
        return get_potential_values(jsonschema_property)

def merge_dq(config_options: Dict[str, Any], child_config_options_dq: Deque[ConfigOption]) -> Deque[ConfigOption]:
    if False:
        i = 10
        return i + 15
    'Merge config_options with the child_config_options in the dq.'
    dq = deque()
    while child_config_options_dq:
        (child_config_options, visited) = child_config_options_dq.popleft()
        cfg = merge_dict(child_config_options, config_options)
        dq.append(ConfigOption(cfg, visited))
    return dq

def explore_from_all_of(config_options: Dict[str, Any], item: Dict[str, Any], key_so_far: str) -> Deque[ConfigOption]:
    if False:
        for i in range(10):
            print('nop')
    'Takes a child of `allOf` and calls `explore_properties` on it.'
    for parameter_name_or_section in item['if']['properties']:
        config_options[key_so_far + '.' + parameter_name_or_section] = item['if']['properties'][parameter_name_or_section]['const']
    jsonschema_properties = item['then']['properties']
    raw_entry = deque([ConfigOption(copy.deepcopy(config_options), False)])
    return explore_properties(jsonschema_properties, parent_parameter_path=key_so_far, dq=raw_entry)

def get_potential_values(item: Dict[str, Any]) -> List[Union[ParameterBaseTypes, List[ParameterBaseTypes]]]:
    if False:
        for i in range(10):
            print('nop')
    'Returns a list of values to explore for a config parameter.\n\n    Param:\n        item: config parameter-specific dictionary. Considered as a leaf in the schema. Contains type, default, and\n            parameter metadata, etc.\n    '
    temp = []
    if isinstance(item['type'], list):
        for property_type in item['type']:
            temp += handle_property_type(property_type, item)
    else:
        temp += handle_property_type(item['type'], item)
    unique_temp = []
    for temp_item in temp:
        if temp_item not in unique_temp:
            unique_temp.append(temp_item)
    return unique_temp

def generate_possible_configs(config_options: Dict[str, Any]):
    if False:
        i = 10
        return i + 15
    'Generate exhaustive configs from config_options.\n\n    This function does not take a cross product of all the options for all the config parameters. It selects parameter\n    values independently from each other.\n\n    Args:\n        config_options: dictionary mapping from ludwig config parameter to all values to be explored.\n            Here\'s an example of what it could look like:\n\n                trainer.batch_size: ["auto", 2, 43]\n                trainer.learning_rate: ["auto", 0.1, 0.00002, 0.32424]\n                ...\n    '
    num_configs = 1
    for parameter_name in config_options:
        if isinstance(config_options[parameter_name], list):
            num_configs = max(num_configs, len(config_options[parameter_name]))
            config_options[parameter_name] = deque(config_options[parameter_name])
    for _ in range(num_configs):
        config = {}
        for parameter_name in config_options:
            if config_options[parameter_name] and (not isinstance(config_options[parameter_name], str)):
                config[parameter_name] = config_options[parameter_name].popleft()
            elif isinstance(config_options[parameter_name], str):
                config[parameter_name] = config_options[parameter_name]
        yield create_nested_dict(config)

def create_nested_dict(flat_dict: Dict[str, Union[float, str]]) -> ModelConfigDict:
    if False:
        for i in range(10):
            print('nop')
    "Generate a nested dict out of a flat dict whose keys are delimited by a delimiter character.\n\n    Args:\n        flat_dict: potential generated baseline config. Here's an example of what it could look like:\n\n            trainer.batch_size: 324\n            trainer.learning_rate: 0.0635\n\n        The expected output would be\n\n            trainer:\n                batch_size: 324\n                learning_rate: 0.0635\n    "

    def to_nested_format(parameter_name: str, value: Union[str, int, float], delimiter: str='.') -> Dict[str, Any]:
        if False:
            print('Hello World!')
        split_parameter_name = parameter_name.split(delimiter)
        for parameter_name_or_section in reversed(split_parameter_name):
            value = {parameter_name_or_section: value}
        return value
    config = {}
    for parameter_name_or_section in flat_dict:
        config = merge_dict(config, to_nested_format(parameter_name_or_section, copy.deepcopy(flat_dict[parameter_name_or_section])))
    return config

def combine_configs(explored: Deque[Tuple[Dict, bool]], config: ModelConfigDict) -> List[Tuple[ModelConfigDict, pd.DataFrame]]:
    if False:
        print('Hello World!')
    'Merge base config with explored sections.\n\n    Args:\n        explored: deque containing all the config options.\n        config: base Ludwig config to merge the explored configs with.\n    '
    dataset = build_synthetic_dataset_df(NUM_SYNTHETIC_EXAMPLES, config)
    ret = []
    for (config_options, _) in explored:
        for default_config in generate_possible_configs(config_options=config_options):
            merged_config = merge_dict(copy.deepcopy(config), default_config)
            try:
                ModelConfig.from_dict(merged_config)
                ret.append((merged_config, dataset))
            except Exception:
                pass
    return ret

def combine_configs_for_comparator_combiner(explored: Deque[Tuple], config: ModelConfigDict) -> List[Tuple[ModelConfigDict, pd.DataFrame]]:
    if False:
        for i in range(10):
            print('nop')
    'Merge base config with explored sections.\n\n    Completes the entity_1 and entity_2 paramters of the comparator combiner.\n\n    Args:\n        explored: deque containing all the config options.\n        config: base Ludwig config to merge the explored configs with.\n    '
    dataset = build_synthetic_dataset_df(NUM_SYNTHETIC_EXAMPLES, config)
    ret = []
    for item in explored:
        for default_config in generate_possible_configs(config_options=item[0]):
            merged_config = merge_dict(copy.deepcopy(config), default_config)
            entity_names = [feature['name'] for feature in config['input_features']]
            random.shuffle(entity_names)
            entity_1_size = random.randint(1, len(entity_names) - 1)
            merged_config['combiner']['entity_1'] = entity_names[:entity_1_size]
            merged_config['combiner']['entity_2'] = entity_names[entity_1_size:]
            try:
                ModelConfig.from_dict(merged_config)
                ret.append((merged_config, dataset))
            except Exception:
                pass
    return ret

def combine_configs_for_sequence_combiner(explored: Deque[Tuple], config: ModelConfigDict) -> List[Tuple[ModelConfigDict, pd.DataFrame]]:
    if False:
        print('Hello World!')
    'Merge base config with explored sections.\n\n    Uses the right reduce_output strategy for the sequence and sequence_concat combiners.\n\n    Args:\n        explored: deque containing all the config options.\n        config: base Ludwig config to merge the explored configs with.\n    '
    dataset = build_synthetic_dataset_df(NUM_SYNTHETIC_EXAMPLES, config)
    ret = []
    for item in explored:
        for default_config in generate_possible_configs(config_options=item[0]):
            merged_config = merge_dict(copy.deepcopy(config), default_config)
            for i in range(len(merged_config['input_features'])):
                if merged_config['input_features'][i]['type'] in {SEQUENCE, TEXT, TIMESERIES}:
                    merged_config['input_features'][0]['encoder'] = {'type': 'embed', 'reduce_output': None}
            try:
                ModelConfig.from_dict(merged_config)
                ret.append((merged_config, dataset))
            except Exception:
                pass
    return ret