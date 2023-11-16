import argparse
import os
from codegen_utils import FunctionGeneratorBase, GeneratorBase, GetForwardFunctionName, GetInplacedFunctionName, IsVectorTensorType
skipped_forward_api_names = set()

def SkipAPIGeneration(forward_api_name):
    if False:
        i = 10
        return i + 15
    return forward_api_name in skipped_forward_api_names
atype_to_parsing_function = {'bool': 'CastPyArg2Boolean', 'int': 'CastPyArg2Int', 'long': 'CastPyArg2Long', 'int64_t': 'CastPyArg2Long', 'float': 'CastPyArg2Float', 'double': 'CastPyArg2Double', 'std::string': 'CastPyArg2String', 'std::vector<bool>': 'CastPyArg2Booleans', 'std::vector<int>': 'CastPyArg2Ints', 'std::vector<long>': 'CastPyArg2Longs', 'std::vector<int64_t>': 'CastPyArg2Longs', 'std::vector<float>': 'CastPyArg2Floats', 'std::vector<double>': 'CastPyArg2Float64s', 'std::vector<std::string>': 'CastPyArg2Strings', 'paddle::experimental::Scalar': 'CastPyArg2Scalar', 'std::vector<phi::Scalar>': 'CastPyArg2ScalarArray', 'paddle::experimental::IntArray': 'CastPyArg2IntArray', 'paddle::Place': 'CastPyArg2Place', 'phi::DataType': 'CastPyArg2DataType'}

def FindParsingFunctionFromAttributeType(atype):
    if False:
        for i in range(10):
            print('nop')
    if atype not in atype_to_parsing_function.keys():
        raise AssertionError(f'Unable to find {atype} in atype_to_parsing_function.')
    return atype_to_parsing_function[atype]
PARSE_PYTHON_C_TENSORS_TEMPLATE = '    auto {} = {}("{}", "{}", args, {}, {});\n'
CONVERT_INPUT_TENSORS_TO_DIST_TENSOR_TEMPLATE = '\n    const phi::distributed::ProcessMesh* mesh = nullptr;\n    if (InputsContainDistTensor(&mesh{inputs})) {{\n      ConvertAllInputsToDistTensor(mesh{inputs});\n    }}\n'
PARSE_PYTHON_C_ARGS_TEMPLATE = '    PyObject* {}_obj = PyTuple_GET_ITEM(args, {});\n    {} {} = {}({}_obj, "{}", {});\n'
RECORD_EVENT_TEMPLATE = 'paddle::platform::RecordEvent {}("{} {}", paddle::platform::TracerEventType::UserDefined, 1);'
RETURN_INPLACE_PYOBJECT_TEMPLATE = '\n    inplace_var_idx_map[{}] = {};\n'
PYTHON_C_FUNCTION_TEMPLATE = '\nPyObject * eager_api_{}(PyObject *self, PyObject *args, PyObject *kwargs) {{\n  {}\n  PyThreadState *tstate = nullptr;\n  try {{\n    VLOG(6) << "Running Eager Final State API: {}";\n\n    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);\n    // Get EagerTensors from args\n{}\n    // Parse Attributes if needed\n{}\n    tstate = PyEval_SaveThread();\n\n    // Set Device ID\n{}\n    // Call dygraph function\n    {}\n\n    PyEval_RestoreThread(tstate);\n    tstate = nullptr;\n{}\n  }} catch(...) {{\n    if (tstate) {{\n      PyEval_RestoreThread(tstate);\n    }}\n    ThrowExceptionToPython(std::current_exception());\n    return nullptr;\n  }}\n}}\n'
NOAMP_DYGRAPH_FUNCTION_TEMPLATE = 'decltype({}({})) out = {}({});'
FUNCTION_SET_DEVICE_TEMPLATE = '{}\n    SetPythonStack();\n    if (paddle::platform::is_gpu_place(place)) {{\n#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)\n      phi::backends::gpu::SetDeviceId(place.device);\n      VLOG(4) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;\n#else\n      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(\n        "PaddlePaddle should compile with GPU if use CUDAPlace."));\n#endif\n    }}\n    if (paddle::platform::is_custom_place(place)) {{\n#if defined(PADDLE_WITH_CUSTOM_DEVICE)\n      phi::DeviceManager::SetDevice(place);\n      VLOG(4) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;\n#else\n      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(\n        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));\n#endif\n    }}\n    if (paddle::platform::is_xpu_place(place)) {{\n#if defined(PADDLE_WITH_XPU)\n      phi::backends::xpu::SetXPUDeviceId(place.device);\n      VLOG(4) <<"CurrentDeviceId: " << phi::backends::xpu::GetXPUCurrentDeviceId() << " from " << (int)place.device;\n#else\n      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(\n        "PaddlePaddle should compile with XPU if use XPUPlace."));\n#endif\n    }}\n'
FUNCTION_NAME_TEMPLATE = '{}{}{}'
PYTHON_C_FUNCTION_REG_TEMPLATE = '  {{"{}{}", (PyCFunction)(void(*)(void)) {}eager_api_{}, METH_VARARGS | METH_KEYWORDS, "C++ interface function for {} in dygraph."}},\n'
PYTHON_C_WRAPPER_TEMPLATE = '\n#include <Python.h>\n#include "paddle/fluid/platform/enforce.h"\n#include "paddle/phi/api/include/strings_api.h"\n#include "paddle/phi/backends/device_manager.h"\n#include "paddle/fluid/pybind/eager_utils.h"\n#include "paddle/fluid/pybind/exception.h"\n#include "paddle/fluid/platform/profiler/event_tracing.h"\n#include "paddle/fluid/pybind/op_function_common.h"\n#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"\n#include "paddle/fluid/eager/api/manual/eager_manual/dygraph_forward_api.h"\n#include "paddle/fluid/pybind/eager_custom_python_api.h"\n#include "paddle/fluid/pybind/eager.h"\n#include "paddle/fluid/eager/amp_utils.h"\n#include "paddle/fluid/eager/eager_amp_auto_cast.h"\n#include "paddle/fluid/pybind/eager_op_function.h"\nnamespace paddle {{\nnamespace pybind {{\n\n{}\n\nstatic PyMethodDef EagerFinalStateMethods[] = {{\n{}\n}};\n\nvoid BindFinalStateEagerOpFunctions(pybind11::module *module) {{\n  if (PyModule_AddFunctions(module->ptr(), EagerFinalStateMethods) < 0) {{\n    PADDLE_THROW(platform::errors::Fatal ("Add functions to core.eager.ops failed!"));\n  }}\n\n  if (PyModule_AddFunctions(module->ptr(), CustomEagerFinalStateMethods) < 0) {{\n    PADDLE_THROW(platform::errors::Fatal ("Add functions to core.eager.ops failed!"));\n  }}\n}}\n\n}} // namespace pybind\n}} // namespace paddle\n'
CORE_OPS_INFO = '\nstatic PyObject * eager_get_core_ops_args_info(PyObject *self) {\n    PyThreadState *tstate = nullptr;\n    try {\n      return ToPyObject(core_ops_args_info);\n    }\n    catch(...) {\n      if (tstate) {\n        PyEval_RestoreThread(tstate);\n      }\n      ThrowExceptionToPython(std::current_exception());\n      return nullptr;\n    }\n}\n\nstatic PyObject * eager_get_core_ops_args_type_info(PyObject *self) {\n    PyThreadState *tstate = nullptr;\n    try {\n      return ToPyObject(core_ops_args_type_info);\n    }\n    catch(...) {\n      if (tstate) {\n        PyEval_RestoreThread(tstate);\n      }\n      ThrowExceptionToPython(std::current_exception());\n      return nullptr;\n    }\n}\n\nstatic PyObject * eager_get_core_ops_returns_info(PyObject *self) {\n    PyThreadState *tstate = nullptr;\n    try {\n      return ToPyObject(core_ops_returns_info);\n    }\n    catch(...) {\n      if (tstate) {\n        PyEval_RestoreThread(tstate);\n      }\n      ThrowExceptionToPython(std::current_exception());\n      return nullptr;\n    }\n}\n'
CORE_OPS_INFO_REGISTRY = '\n  {"get_core_ops_args_info", (PyCFunction)(void(*)(void))eager_get_core_ops_args_info, METH_NOARGS, "C++ interface function for eager_get_core_ops_args_info."},\n  {"get_core_ops_args_type_info", (PyCFunction)(void(*)(void))eager_get_core_ops_args_type_info, METH_NOARGS, "C++ interface function for eager_get_core_ops_args_type_info."},\n  {"get_core_ops_returns_info", (PyCFunction)(void(*)(void))eager_get_core_ops_returns_info, METH_NOARGS, "C++ interface function for eager_get_core_ops_returns_info."},\n'
NAMESPACE_WRAPPER_TEMPLATE = 'namespace {} {{\n    {}\n}}\n'
PYTHON_C_H_TEMPLATE = '\n#pragma once\n\n#include <Python.h>\n\n// Avoid a problem with copysign defined in pyconfig.h on Windows.\n#ifdef copysign\n#undef copysign\n#endif\n\nnamespace paddle {{\nnamespace pybind {{\n\n{body}\n\n}} // namespace pybind\n}} // namespace paddle\n'
PYTHON_C_FUNCTION_DECLARE_TEMPLATE = '\nPyObject *eager_api_{name}(PyObject *self, PyObject *args, PyObject *kwargs);\n'

class PythonCSingleFunctionGenerator(FunctionGeneratorBase):

    def __init__(self, forward_api_contents, namespace):
        if False:
            print('Hello World!')
        FunctionGeneratorBase.__init__(self, forward_api_contents, namespace)
        self.is_forward_only = True
        self.python_c_function_str = ''
        self.python_c_function_reg_str = ''
        self.python_c_funcion_declare_str = ''

    def CollectIsForwardOnly(self):
        if False:
            while True:
                i = 10
        forward_api_contents = self.forward_api_contents
        self.is_forward_only = False if 'backward' in forward_api_contents.keys() else True

    def GeneratePythonCFunction(self):
        if False:
            i = 10
            return i + 15
        namespace = self.namespace
        forward_inplace_map = self.forward_inplace_map
        forward_api_name = self.forward_api_name
        orig_forward_attrs_list = self.orig_forward_attrs_list
        forward_inputs_position_map = self.forward_inputs_position_map
        forward_outputs_position_map = self.forward_outputs_position_map
        optional_inputs = self.optional_inputs
        is_forward_only = self.is_forward_only
        inplace_args_pos_map = {}
        inplace_returns_pos_map = {}
        get_eager_tensor_str = ''
        input_names = ''
        for (name, (ttype, pos)) in forward_inputs_position_map.items():
            input_names = input_names + ', ' + name
            if forward_inplace_map and name in forward_inplace_map.keys():
                inplace_args_pos_map[name] = pos
            is_optional = name in optional_inputs
            if IsVectorTensorType(ttype):
                if is_optional:
                    get_eager_tensor_str += PARSE_PYTHON_C_TENSORS_TEMPLATE.format(name, 'GetOptionalTensorListFromArgs', forward_api_name, name, pos, 'true')
                else:
                    get_eager_tensor_str += PARSE_PYTHON_C_TENSORS_TEMPLATE.format(name, 'GetTensorListFromArgs', forward_api_name, name, pos, 'false')
            elif is_optional:
                get_eager_tensor_str += PARSE_PYTHON_C_TENSORS_TEMPLATE.format(name, 'GetOptionalTensorFromArgs', forward_api_name, name, pos, 'true')
            else:
                get_eager_tensor_str += PARSE_PYTHON_C_TENSORS_TEMPLATE.format(name, 'GetTensorFromArgs', forward_api_name, name, pos, 'false')
        if len(input_names) > 0:
            get_eager_tensor_str += CONVERT_INPUT_TENSORS_TO_DIST_TENSOR_TEMPLATE.format(inputs=input_names)
        if forward_inplace_map:
            for (name, (ttype, pos)) in forward_outputs_position_map.items():
                if name in forward_inplace_map.values():
                    inplace_returns_pos_map[name] = pos
        parse_attributes_str = ''
        expected_place_str = '    auto place = egr::Controller::Instance().GetExpectedPlace();\n'
        for (name, atype, _, pos) in orig_forward_attrs_list:
            parsing_function_name = FindParsingFunctionFromAttributeType(atype)
            if len(expected_place_str) != 0 and parsing_function_name == 'CastPyArg2Place':
                expected_place_str = ''
                assert name == 'place', "Only support 'place' as template argument name in FUNCTION_SET_DEVICE_TEMPLATE."
            parse_attributes_str += PARSE_PYTHON_C_ARGS_TEMPLATE.format(name, pos, atype, name, parsing_function_name, name, forward_api_name, pos)
        set_device_str = FUNCTION_SET_DEVICE_TEMPLATE.format(expected_place_str)
        num_args = len(forward_inputs_position_map.keys()) + len(orig_forward_attrs_list)
        dygraph_function_call_list = ['' for i in range(num_args)]
        for (name, (_, pos)) in forward_inputs_position_map.items():
            dygraph_function_call_list[pos] = f'{name}'
        for (name, _, _, pos) in orig_forward_attrs_list:
            dygraph_function_call_list[pos] = f'{name}'
        dygraph_function_call_str = ','.join(dygraph_function_call_list)
        fwd_function_name = FUNCTION_NAME_TEMPLATE.format('::', namespace, GetForwardFunctionName(forward_api_name))
        return_str = '    return ToPyObject(out);'
        pythonc_record_event_str = RECORD_EVENT_TEMPLATE.format('pythonc_record_event', forward_api_name, 'pybind_imperative_func')
        noamp_dygraph_function_str = NOAMP_DYGRAPH_FUNCTION_TEMPLATE.format(fwd_function_name, dygraph_function_call_str, fwd_function_name, dygraph_function_call_str)
        self.python_c_function_str = PYTHON_C_FUNCTION_TEMPLATE.format(forward_api_name, pythonc_record_event_str, forward_api_name, get_eager_tensor_str, parse_attributes_str, set_device_str, noamp_dygraph_function_str, return_str)
        self.python_c_funcion_declare_str = PYTHON_C_FUNCTION_DECLARE_TEMPLATE.format(name=forward_api_name)
        prefix = self.namespace.strip('::')
        forward_api_name_prefix = '' if prefix == '' else prefix + '_'
        self.python_c_function_reg_str = PYTHON_C_FUNCTION_REG_TEMPLATE.format(forward_api_name_prefix, forward_api_name, namespace, forward_api_name, forward_api_name)
        if forward_inplace_map:
            inplaced_forward_api_name = GetInplacedFunctionName(self.forward_api_name)
            inplaced_fwd_function_name = FUNCTION_NAME_TEMPLATE.format('::', namespace, GetForwardFunctionName(inplaced_forward_api_name))
            inplace_noamp_dygraph_function_str = NOAMP_DYGRAPH_FUNCTION_TEMPLATE.format(inplaced_fwd_function_name, dygraph_function_call_str, inplaced_fwd_function_name, dygraph_function_call_str)
            return_str = '    std::map<ssize_t, ssize_t> inplace_var_idx_map;'
            for (inplace_input, inplace_output) in forward_inplace_map.items():
                return_str += RETURN_INPLACE_PYOBJECT_TEMPLATE.format(inplace_returns_pos_map[inplace_output], inplace_args_pos_map[inplace_input])
            return_str += '    return ToPyObject(out, args, inplace_var_idx_map);'
            python_c_inplace_func_str = PYTHON_C_FUNCTION_TEMPLATE.format(inplaced_forward_api_name, pythonc_record_event_str, inplaced_forward_api_name, get_eager_tensor_str, parse_attributes_str, set_device_str, inplace_noamp_dygraph_function_str, return_str)
            python_c_funcion_declare_str = PYTHON_C_FUNCTION_DECLARE_TEMPLATE.format(name=inplaced_forward_api_name)
            python_c_inplace_func_reg_str = PYTHON_C_FUNCTION_REG_TEMPLATE.format(forward_api_name_prefix, inplaced_forward_api_name, namespace, inplaced_forward_api_name, inplaced_forward_api_name)
            if self.forward_api_name[-1] == '_':
                self.python_c_function_str = python_c_inplace_func_str
                self.python_c_funcion_declare_str = python_c_funcion_declare_str
                self.python_c_function_reg_str = python_c_inplace_func_reg_str
            else:
                self.python_c_function_str += python_c_inplace_func_str
                self.python_c_funcion_declare_str += python_c_funcion_declare_str
                self.python_c_function_reg_str += python_c_inplace_func_reg_str

    def run(self):
        if False:
            return 10
        self.CollectIsForwardOnly()
        self.ParseDispensable()
        self.ParseForwardInplaceInfo()
        self.CollectOriginalForwardInfo()
        if SkipAPIGeneration(self.forward_api_name):
            return False
        self.DetermineForwardPositionMap(self.orig_forward_inputs_list, self.orig_forward_returns_list)
        self.GeneratePythonCFunction()
        return True

class PythonCGenerator(GeneratorBase):

    def __init__(self, path):
        if False:
            print('Hello World!')
        GeneratorBase.__init__(self, api_yaml_path)
        self.python_c_functions_str = ''
        self.python_c_functions_reg_str = ''
        self.python_c_funcion_declare_str = ''

    def GeneratePythonCFunctions(self):
        if False:
            return 10
        namespace = self.namespace
        forward_api_list = self.forward_api_list
        for forward_api_content in forward_api_list:
            f_generator = PythonCSingleFunctionGenerator(forward_api_content, namespace)
            status = f_generator.run()
            if status:
                self.python_c_functions_str += f_generator.python_c_function_str + '\n'
                self.python_c_functions_reg_str += f_generator.python_c_function_reg_str
                self.python_c_funcion_declare_str += f_generator.python_c_funcion_declare_str

    def AttachNamespace(self):
        if False:
            while True:
                i = 10
        namespace = self.namespace
        python_c_functions_str = self.python_c_functions_str
        if namespace != '':
            if namespace.endswith('::'):
                namespace = namespace[:-2]
            self.python_c_functions_str = NAMESPACE_WRAPPER_TEMPLATE.format(namespace, python_c_functions_str)
            self.python_c_funcion_declare_str = NAMESPACE_WRAPPER_TEMPLATE.format(namespace, self.python_c_funcion_declare_str)

    def run(self):
        if False:
            i = 10
            return i + 15
        self.InferNameSpace()
        self.ParseForwardYamlContents()
        self.GeneratePythonCFunctions()
        self.AttachNamespace()

def ParseArguments():
    if False:
        return 10
    parser = argparse.ArgumentParser(description='Eager Code Generator Args Parser')
    parser.add_argument('--api_yaml_path', type=str)
    parser.add_argument('--source_path', type=str)
    parser.add_argument('--header_path', type=str)
    args = parser.parse_args()
    return args

def GenerateCoreOpsInfoMap():
    if False:
        print('Hello World!')
    return (CORE_OPS_INFO, CORE_OPS_INFO_REGISTRY)

def GeneratePythonCWrappers(python_c_function_str, python_c_function_reg_str):
    if False:
        i = 10
        return i + 15
    (core_ops_infos_definition, core_ops_infos_registry) = GenerateCoreOpsInfoMap()
    python_c_function_str += core_ops_infos_definition
    python_c_function_reg_str += core_ops_infos_registry
    python_c_function_reg_str += '  {nullptr,nullptr,0,nullptr}'
    python_c_str = PYTHON_C_WRAPPER_TEMPLATE.format(python_c_function_str, python_c_function_reg_str)
    return python_c_str

def GeneratePythonCFile(filepath, python_c_str):
    if False:
        for i in range(10):
            print('nop')
    with open(filepath, 'a') as f:
        f.write(python_c_str)
if __name__ == '__main__':
    args = ParseArguments()
    api_yaml_paths = args.api_yaml_path.split(',')
    generated_python_c_functions = ''
    generated_python_c_registration = ''
    generated_python_c_functions_header = ''
    for i in range(len(api_yaml_paths)):
        api_yaml_path = api_yaml_paths[i]
        py_c_generator = PythonCGenerator(api_yaml_path)
        py_c_generator.run()
        generated_python_c_functions += py_c_generator.python_c_functions_str + '\n'
        generated_python_c_registration += py_c_generator.python_c_functions_reg_str
        generated_python_c_functions_header += py_c_generator.python_c_funcion_declare_str
    python_c_str = GeneratePythonCWrappers(generated_python_c_functions, generated_python_c_registration)
    soucre_path = args.source_path
    header_path = args.header_path
    for path in [soucre_path, header_path]:
        if os.path.exists(path):
            os.remove(path)
    GeneratePythonCFile(soucre_path, python_c_str)
    GeneratePythonCFile(header_path, PYTHON_C_H_TEMPLATE.format(body=generated_python_c_functions_header))