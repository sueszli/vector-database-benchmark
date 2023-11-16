"""
Print all ops desc in dict:
    {op1_name:
        {INPUTS:
            {input_name1:
                {DISPENSABLE: bool,
                 INTERMEDIATE: bool,
                 DUPLICABLE: bool,
                 EXTRA: bool,
                 QUANT: bool,
                },
            input_name2:{}
            },
         OUTPUTS:{},
         ATTRS:
            {attr_name1:
                {TYPE: int,
                 GENERATED: bool,
                 DEFAULT_VALUE: int/str/etc,
                 EXTRA: bool,
                 QUANT: bool,
                }
            }
        }
     op2_name:{}
    }

Usage:
    python print_op_desc.py > op_desc.spec
"""
import json
from paddle.base import core, framework
INPUTS = 'Inputs'
OUTPUTS = 'Outputs'
ATTRS = 'Attrs'
DUPLICABLE = 'duplicable'
INTERMEDIATE = 'intermediate'
DISPENSABLE = 'dispensable'
TYPE = 'type'
GENERATED = 'generated'
DEFAULT_VALUE = 'default_value'
EXTRA = 'extra'
QUANT = 'quant'

def get_attr_default_value(op_name):
    if False:
        print('Hello World!')
    return core.get_op_attrs_default_value(op_name.encode())

def get_vars_info(op_vars_proto):
    if False:
        while True:
            i = 10
    vars_info = {}
    for var_proto in op_vars_proto:
        name = str(var_proto.name)
        vars_info[name] = {}
        vars_info[name][DUPLICABLE] = var_proto.duplicable
        vars_info[name][DISPENSABLE] = var_proto.dispensable
        vars_info[name][INTERMEDIATE] = var_proto.intermediate
        vars_info[name][EXTRA] = var_proto.extra
        vars_info[name][QUANT] = var_proto.quant
    return vars_info

def get_attrs_info(op_proto, op_attrs_proto):
    if False:
        while True:
            i = 10
    attrs_info = {}
    attrs_default_values = get_attr_default_value(op_proto.type)
    for attr_proto in op_attrs_proto:
        attr_name = str(attr_proto.name)
        attrs_info[attr_name] = {}
        attrs_info[attr_name][TYPE] = attr_proto.type
        attrs_info[attr_name][GENERATED] = attr_proto.generated
        attrs_info[attr_name][DEFAULT_VALUE] = attrs_default_values[attr_name] if attr_name in attrs_default_values else None
        attrs_info[attr_name][EXTRA] = attr_proto.extra
        attrs_info[attr_name][QUANT] = attr_proto.quant
    return attrs_info

def get_op_desc(op_proto):
    if False:
        while True:
            i = 10
    op_info = {}
    op_info[INPUTS] = get_vars_info(op_proto.inputs)
    op_info[OUTPUTS] = get_vars_info(op_proto.outputs)
    op_info[ATTRS] = get_attrs_info(op_proto, op_proto.attrs)
    return op_info

def get_all_ops_desc():
    if False:
        for i in range(10):
            print('nop')
    all_op_protos_dict = {}
    all_op_protos = framework.get_all_op_protos()
    for op_proto in all_op_protos:
        op_type = str(op_proto.type)
        all_op_protos_dict[op_type] = get_op_desc(op_proto)
    return all_op_protos_dict
all_op_protos_dict = get_all_ops_desc()
result = json.dumps(all_op_protos_dict)
print(result)