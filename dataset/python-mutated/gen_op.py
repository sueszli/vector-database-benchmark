import sys
import yaml
import argparse
import os
from copy import deepcopy
from typing import Dict, List, Set
parser = argparse.ArgumentParser()
parser.add_argument('--template_dir', default='.', help='where template.h is')
parser.add_argument('--yaml_dir', default='aten/src/ATen/ATen', help='where ATen yaml files are')
parser.add_argument('--output_prefix', default='', help='')
parser.add_argument('--install_dir', default='.', help='where to put generated file')
parser.add_argument('--aten_root', default='', help='root directory of aten')
(args, _) = parser.parse_known_args()
if args.aten_root:
    if not os.path.exists(args.aten_root):
        raise ValueError('aten_root ({}) does not exist'.format(args.aten_root))
    sys.path.insert(0, os.path.join(args.aten_root, '..'))
    from torchgen.code_template import CodeTemplate as CT
else:
    from torchgen.code_template import CodeTemplate as CT
OP_TEMPLATE = CT.from_file(os.path.join(args.template_dir, 'aten_op_template.h'))
try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader

def write(filename, s):
    if False:
        i = 10
        return i + 15
    with open(filename, 'w') as f:
        f.write(s)

def read(filename):
    if False:
        print('Hello World!')
    with open(filename, 'r') as f:
        return f.read()

def value_has_tensors(v):
    if False:
        print('Hello World!')
    return 'Tensor' in v['dynamic_type'] and 'Sparse' not in v['dynamic_type']

def value_is_tensor_type(v):
    if False:
        print('Hello World!')
    return value_has_tensors(v) and v['dynamic_type'] not in TENSORLIST_TYPE
TENSORLIST_TYPE = ['at::TensorList', 'const at::ITensorListRef &', 'const c10::List<c10::optional<at::Tensor>> &']
RETURN_MAP = {'at::Tensor': 'assignTo(Output(${offset}),${output});', 'at::Scalar': 'assignTo(Output(${offset}),${output}.type(), ${output});', 'bool': 'assignToValue<int64_t>(Output(${offset}),${output});', 'int64_t': 'assignToValue<int64_t>(Output(${offset}),${output});', '::std::vector<at::Tensor>': 'assignListStartingAt(${offset}, ${output});'}
ARGUMENT_MAP = {'const at::Scalar &': 'at::Scalar ${arg} = readScalarAttribute("${arg}");', 'bool': 'bool ${arg} = readAttribute<int64_t>("${arg}");', 'int': 'int ${arg} = readAttribute<int64_t>("${arg}");', 'double': 'double ${arg} = readAttribute<float>("${arg}");', 'int64_t': 'int64_t ${arg} = readAttribute<int64_t>("${arg}");', 'at::IntArrayRef': 'auto ${arg} = readIntArrayRef("${arg}");', '::std::array<bool,2>': 'auto ${arg} = readBoolMask<2>("${arg}");', '::std::array<bool,3>': 'auto ${arg} = readBoolMask<3>("${arg}");'}
SPECIAL_IMPLEMENTATIONS = {'index': 'internal::index_with_uint8_handling'}

def expand(o):
    if False:
        print('Hello World!')
    num_defaults = sum((1 if 'default' in arg else 0 for arg in o['arguments']))
    results = [o]
    for i in range(0, num_defaults):
        assert 'default' in o['arguments'][-(i + 1)]
        v = deepcopy(o)
        v['arguments'] = v['arguments'][:-(i + 1)]
        results.append(v)
    return results

def supports(o, factory_methods):
    if False:
        return 10
    if o['name'] in factory_methods:
        if factory_methods[o['name']] == 0:
            print('Skipping {} because it is a factory method'.format(o['name']))
        factory_methods[o['name']] += 1
        return False
    if o['inplace']:
        return False
    if '_out' in o['name']:
        return False
    if len(o['returns']) == 0:
        return False
    for ret in o['returns']:
        if not value_has_tensors(ret) and ret['type'] not in RETURN_MAP:
            print('Skipping {} Because of Ret: {} ({})'.format(o['name'], ret['type'], ret['dynamic_type']))
            return False
    for arg in o['arguments']:
        if not value_has_tensors(arg) and arg['type'] not in ARGUMENT_MAP:
            print('Skipping {} Because of Arg: {} ({}) '.format(o['name'], arg['type'], arg['dynamic_type']))
            return False
    return True
IMPLEMENTATION_TEMPLATE = CT('C10_NOINLINE void implementation_${key}() { // ${name}\n    ${initialization}\n    run_op = [=] {\n        at::AutoDispatchBelowAutograd guard;\n        ${statements}\n        auto the_result = ${invocation};\n        ${assignments}\n        return true;\n    };\n}\n')
CASE_TEMPLATE = CT('case ${key}: // ${name}\n  implementation_${key}();\n  break;\n')
ASSIGN_CHECK_SIZE_TEMPLATE = CT('  if(OutputSize() > ${offset}) {${assignment}}\n')

def get_output(o, i):
    if False:
        print('Hello World!')
    if len(o['returns']) == 1:
        return 'the_result'
    else:
        return '::std::get<{}>(the_result)'.format(i)

def attribute_names(o):
    if False:
        while True:
            i = 10
    return sorted([a['name'] for a in o['arguments'] if not value_has_tensors(a)])

def required_attribute_names(o):
    if False:
        return 10
    return sorted([a['name'] for a in o['arguments'] if not value_has_tensors(a) and 'default' not in a])

def self_as_first_argument(arguments):
    if False:
        i = 10
        return i + 15
    return [a for a in arguments if a['name'] == 'self'] + [a for a in arguments if a['name'] != 'self']

def get_num_inputs(o):
    if False:
        print('Hello World!')
    args = 0
    for a in o['arguments']:
        if a['type'] in TENSORLIST_TYPE:
            return '*'
        elif value_has_tensors(a):
            args += 1
    return str(args)

def find_factory_methods(decls):
    if False:
        for i in range(10):
            print('nop')
    factory_methods = {}
    for o in decls:
        if any((arg['dynamic_type'] == 'at::TensorOptions' for arg in o['arguments'])):
            factory_methods[o['name']] = 0
    return factory_methods

def emit_assignments(o, env):
    if False:
        while True:
            i = 10
    for (i, r) in enumerate(o['returns']):
        t = RETURN_MAP[r['type'] if not value_is_tensor_type(r) else 'at::Tensor']
        assignment = CT(t).substitute(env, offset=i, output=get_output(o, i))
        check_size_assignment = ASSIGN_CHECK_SIZE_TEMPLATE.substitute(env, offset=i, assignment=assignment)
        env['assignments'].append(check_size_assignment)
if __name__ == '__main__':
    decls = yaml.load(read(os.path.join(args.yaml_dir, 'Declarations.yaml')), Loader=Loader)
    factory_methods = find_factory_methods(decls)
    filtered = [expanded for o in decls for expanded in expand(o) if supports(expanded, factory_methods)]
    top_env: Dict[str, List] = {'mappings': [], 'implementations': [], 'cases': []}
    seen: Set[str] = set()
    key = 0
    for o in filtered:
        attr_names = attribute_names(o)
        num_inputs = get_num_inputs(o)
        descriptor = '-'.join([o['name']] + attr_names + [num_inputs])
        if descriptor in seen:
            continue
        seen.add(descriptor)
        top_env['mappings'].append('{{ "{}", {} }},'.format(descriptor, key))
        env = {'name': o['name'], 'statements': [], 'arguments': [], 'assignments': [], 'initialization': [], 'key': str(key)}
        if 'namespace' not in o['method_of'] and 'Tensor' not in o['method_of']:
            assert 'Type' in o['method_of']
        static_tensor_inputs = sum((arg['type'] not in TENSORLIST_TYPE and value_is_tensor_type(arg) for arg in o['arguments']))
        has_tensorlist = any((arg['type'] in TENSORLIST_TYPE for arg in o['arguments']))
        if has_tensorlist:
            tensorlist_idx = [i for (i, arg) in enumerate(o['arguments']) if arg['type'] in TENSORLIST_TYPE][0]
        real_inputs = 0
        for (i, arg) in enumerate(o['arguments']):
            env['arguments'].append(arg['name'])
            view_length = 'InputSize()' if has_tensorlist and i < tensorlist_idx else static_tensor_inputs
            if arg['type'] == 'at::TensorList' or arg['type'] == 'const at::ITensorListRef &':
                env['statements'].append('auto {} = peekSlice({}, InputSize() - {}, InputSize());'.format(arg['name'], real_inputs, static_tensor_inputs))
            elif arg['type'] == 'const c10::List<c10::optional<at::Tensor>> &':
                env['statements'].append('auto {} = peekSliceOptionals({}, InputSize() - {}, InputSize());'.format(arg['name'], real_inputs, static_tensor_inputs))
            elif value_is_tensor_type(arg):
                env['statements'].append('auto {} = peek({}, {});'.format(arg['name'], real_inputs, view_length))
                real_inputs += 1
            else:
                init = CT(ARGUMENT_MAP[arg['type']]).substitute(env, arg=arg['name'])
                env['initialization'].append(init)
        emit_assignments(o, env)
        if o['name'] in SPECIAL_IMPLEMENTATIONS:
            env['invocation'] = '{}({})'.format(SPECIAL_IMPLEMENTATIONS[o['name']], ','.join(env['arguments']))
        elif 'namespace' in o['method_of']:
            env['invocation'] = CT('at::${name}(${arguments})').substitute(env)
        else:
            assert 'Tensor' in o['method_of']
            env['invocation'] = 'self.{}({})'.format(o['name'], ', '.join(env['arguments'][1:]))
        top_env['implementations'].append(IMPLEMENTATION_TEMPLATE.substitute(env))
        top_env['cases'].append(CASE_TEMPLATE.substitute(env))
        key += 1
    write(os.path.join(args.install_dir, args.output_prefix + 'aten_op.h'), OP_TEMPLATE.substitute(top_env))