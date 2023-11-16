"""
YAML helper, sourced from the AWS CLI

https://github.com/aws/aws-cli/blob/develop/awscli/customizations/cloudformation/yamlhelper.py
"""
import json
from typing import Dict, Optional, cast
import yaml
from botocore.compat import OrderedDict
from samtranslator.utils.py27hash_fix import Py27Dict, Py27UniStr
from yaml.nodes import ScalarNode, SequenceNode
TAG_STR = 'tag:yaml.org,2002:str'
TIMESTAMP_TAG = 'tag:yaml.org,2002:timestamp'

def string_representer(dumper, value):
    if False:
        print('Hello World!')
    '\n    Customer Yaml representer that will force the scalar to be quoted in a yaml.dump\n    if it scalar starts with a 0. This is needed to keep account ids a string instead\n    of turning into on int because yaml thinks it an octal.\n\n    Parameters\n    ----------\n    dumper yaml.dumper\n    value str\n        Value in template to resolve\n\n    Returns\n    -------\n\n    '
    if value.startswith('0'):
        return dumper.represent_scalar(TAG_STR, value, style="'")
    return dumper.represent_scalar(TAG_STR, value)

def intrinsics_multi_constructor(loader, tag_prefix, node):
    if False:
        print('Hello World!')
    '\n    YAML constructor to parse CloudFormation intrinsics.\n    This will return a dictionary with key being the instrinsic name\n    '
    tag = node.tag[1:]
    prefix = 'Fn::'
    if tag in ['Ref', 'Condition']:
        prefix = ''
    cfntag = prefix + tag
    if tag == 'GetAtt' and isinstance(node.value, str):
        value = node.value.split('.', 1)
    elif isinstance(node, ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, SequenceNode):
        value = loader.construct_sequence(node)
    else:
        value = loader.construct_mapping(node)
    return {cfntag: value}

def _dict_representer(dumper, data):
    if False:
        return 10
    return dumper.represent_dict(data.items())

def yaml_dump(dict_to_dump):
    if False:
        print('Hello World!')
    '\n    Dumps the dictionary as a YAML document\n    :param dict_to_dump:\n    :return:\n    '
    CfnDumper.add_representer(OrderedDict, _dict_representer)
    CfnDumper.add_representer(str, string_representer)
    CfnDumper.add_representer(Py27Dict, _dict_representer)
    CfnDumper.add_representer(Py27UniStr, string_representer)
    return yaml.dump(dict_to_dump, default_flow_style=False, Dumper=CfnDumper)

def _dict_constructor(loader, node):
    if False:
        i = 10
        return i + 15
    loader.flatten_mapping(node)
    return OrderedDict(loader.construct_pairs(node))

def yaml_parse(yamlstr) -> Dict:
    if False:
        print('Hello World!')
    'Parse a yaml string'
    try:
        return cast(Dict, json.loads(yamlstr, object_pairs_hook=OrderedDict))
    except ValueError:
        yaml.constructor.SafeConstructor.yaml_constructors[TIMESTAMP_TAG] = yaml.constructor.SafeConstructor.yaml_constructors[TAG_STR]
        yaml.SafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _dict_constructor)
        yaml.SafeLoader.add_multi_constructor('!', intrinsics_multi_constructor)
        return cast(Dict, yaml.safe_load(yamlstr))

def parse_yaml_file(file_path, extra_context: Optional[Dict]=None) -> Dict:
    if False:
        print('Hello World!')
    '\n    Read the file, do variable substitution, parse it as JSON/YAML\n\n    Parameters\n    ----------\n    file_path : string\n        Path to the file to read\n    extra_context : Dict\n        if the file contains variable in the format of %(variableName)s i.e. the same format of the string % operator,\n        this parameter provides the values for those variables substitution.\n\n    Returns\n    -------\n    questions data as a dictionary\n    '
    with open(file_path, 'r', encoding='utf-8') as fp:
        content = fp.read()
        if isinstance(extra_context, dict):
            content = content % extra_context
        return yaml_parse(content)

class CfnDumper(yaml.SafeDumper):

    def ignore_aliases(self, data):
        if False:
            i = 10
            return i + 15
        return True