"""
Tools for parsing neon model definition files (YAML formatted) and
generating neon model objects from the definition.
"""
from copy import deepcopy
import numpy as np
import yaml
from neon import NervanaObject
from neon.layers import GeneralizedCost
from neon.models import Model
import neon.optimizers
from neon.layers.container import Sequential

def format_yaml_dict(yamldict, type_prefix):
    if False:
        print('Hello World!')
    '\n    Helper function for format the YAML model config into\n    the proper format for object and layer initialization\n\n    Arguments:\n        yamldict (dict): dictionary with model parameters\n\n        type_prefix (str): module path for this object\n\n    Returns:\n        dict : formatted dict\n    '
    yamldict['type'] = type_prefix + yamldict['type']
    return yamldict

def create_objects(root_yaml, be_type='gpu', batch_size=128, rng_seed=None, device_id=0, default_dtype=np.float32, stochastic_rounding=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Instantiate objects as per the given specifications.\n\n    Arguments:\n        root_yaml (dict): Model definition dictionary parse from YAML file\n\n        be_type (str): backend either 'gpu', 'mgpu' or 'cpu'\n\n        batch_size (int): Batch size.\n        rng_seed (None or int): random number generator seed\n\n        device_id (int): for GPU backends id of device to use\n\n        default_dtype (type): numpy data format for default data types,\n\n        stochastic_rounding (bool or int): number of bits for stochastic rounding\n                                           use False for no rounding\n\n    Returns:\n        tuple: Contains model, cost and optimizer objects.\n    "
    assert NervanaObject.be is not None, 'Must generate a backend before running this function'
    if type(root_yaml) is str:
        with open(root_yaml, 'r') as fid:
            root_yaml = yaml.safe_load(fid.read())
    root_yaml = deepcopy(root_yaml)
    yaml_layers = root_yaml['layers']
    layer_dict = {'layers': yaml_layers}
    layers = Sequential.gen_class(layer_dict)
    model = Model(layers=layers)
    cost_name = root_yaml['cost']
    cost = GeneralizedCost.gen_class({'costfunc': {'type': cost_name}})
    opt = None
    if 'optimizer' in root_yaml:
        yaml_opt = root_yaml['optimizer']
        typ = yaml_opt['type']
        opt = getattr(neon.optimizers, typ).gen_class(yaml_opt['config'])
    return (model, cost, opt)