from typing import Dict, List, Sequence, Tuple
from torchgen.api.autograd import Derivative, DifferentiabilityInfo, SavedAttribute, uses_retain_variables, uses_single_grad
from torchgen.api.types import ArrayRefCType, BaseCppType, BaseCType, Binding, boolT, doubleT, intArrayRefT, iTensorListRefT, ListCType, longT, MutRefCType, OptionalCType, optionalIntArrayRefT, optionalSymIntArrayRefT, scalarT, stringT, symIntArrayRefT, SymIntT, TENSOR_LIST_LIKE_CTYPES, tensorListT, tensorT, VectorCType
from torchgen.code_template import CodeTemplate
from torchgen.model import Argument, FunctionSchema
from torchgen.utils import FileManager
from .gen_inplace_or_view_type import VIEW_FUNCTIONS
FUNCTION_DECLARATION = CodeTemplate('#ifdef _WIN32\nstruct ${op} : public ${superclass} {\n  TORCH_API ${op}() = default;\n#else\nstruct TORCH_API ${op} : public ${superclass} {\n#endif\n  using ${superclass}::${superclass};\n  variable_list apply(variable_list&& grads) override;\n  std::string name() const override { return "${op}"; }\n  void release_variables() override {\n    ${thread_lock}\n    ${release_variables}\n  }\n  ${will_release_variables}\n  void compiled_args(CompiledNodeArgs& args) override;\n  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;\n  ${saved_variables}\n  ${saved_list_sizes}\n};\n')
WILL_RELEASE_VARIABLES = CodeTemplate('bool retain_variables = true;\nvoid will_release_variables() override {\n  retain_variables = false;\n}\n')
FUNCTION_DEFINITION = CodeTemplate('variable_list ${op}::apply(variable_list&& grads) {\n  ${thread_lock}\n  ${asserts}\n  IndexRangeGenerator gen;\n  ${compute_index_ranges}\n  variable_list grad_inputs(gen.size());\n  ${body}\n  return grad_inputs;\n}\nvoid ${op}::compiled_args(CompiledNodeArgs& args) {\n    ${compiled_args}\n}\nvariable_list ${op}::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {\n    ${apply_with_saved_before}\n    variable_list result = apply(variable_list(grads));\n    ${apply_with_saved_after}\n    return result;\n}\n')
GRAD_INPUT_MASK = CodeTemplate('  auto grad_input_mask = std::array<bool, ${n}>{\n    ${masks}\n  };')
DERIVATIVE_SINGLE = CodeTemplate('if (task_should_compute_output({ ${name}_ix })) {\n  auto grad_result = ${derivative};\n  copy_range(grad_inputs, ${name}_ix, grad_result);\n}\n')
DERIVATIVE_SINGLE_FOREACH = CodeTemplate('if (task_should_compute_output({ ${name}_ix })) {\n  std::vector<Tensor> grad_result;\n  grad_result.reserve(grads.size());\n  for (const auto & i : c10::irange(grads.size())) {\n    if (grads[i].defined()) {\n      grad_result.emplace_back(${derivative});\n    } else {\n      grad_result.emplace_back(Tensor());\n    }\n  }\n  copy_range(grad_inputs, ${name}_ix, grad_result);\n}\n')
DERIVATIVE_MULTI_COPY_RANGE = CodeTemplate('  if (task_should_compute_output({ ${name}_ix })) {\n    copy_range(grad_inputs, ${name}_ix, std::get<${i}>(grad_result));\n  }\n')
DERIVATIVE_MULTI = CodeTemplate('if (task_should_compute_output({ ${idx_ranges} })) {\n  ${grad_input_mask}\n  auto grad_result = ${derivative};\n  ${copy_ranges}\n}\n')
PY_FUNCTION_DEFINITION = CodeTemplate('static PyTypeObject ${op}Class;\naddClass<${op}>(module, ${op}Class, "${op}", ${op}_properties);\n')
PY_FUNCTION_PROPS_AND_GETTERS = CodeTemplate('${all_getter_definitions}\n\nstatic struct PyGetSetDef ${op}_properties[] = {\n  THP_FUNCTION_DEFAULT_PROPERTIES,\n  ${all_getsetdef_structs}\n  {nullptr} /* sentinel */\n};\n\n')
PY_GETSETDEF_STRUCT = CodeTemplate('{(char*)"_saved_${name}", (getter)THP${op}_${name}_getter, nullptr, nullptr, nullptr}')
PY_RAW_GETSETDEF_STRUCT = CodeTemplate('{(char*)"_raw_saved_${name}", (getter)THP${op}_${name}_raw_getter, nullptr, nullptr, nullptr}')
GETTER_DEFINITION = CodeTemplate('PyObject* THP${op}_${name}_getter(THPCppFunction *self, void *_unused) {\n  HANDLE_TH_ERRORS\n  auto prop = static_cast<${op}*>(self->cdata.get())->${name};\n  ${body}\n  END_HANDLE_TH_ERRORS\n}\n')
GETTER_DEFINITION_SAVEDVAR = CodeTemplate('PyObject* THP${op}_${name}_getter(THPCppFunction *self, void *_unused) {\n  HANDLE_TH_ERRORS\n  const auto& prop = static_cast<${op}*>(self->cdata.get())->${name}_;\n  ${body}\n  END_HANDLE_TH_ERRORS\n}\n')
GETTER_DEFINITION_RAW_SAVEDVAR = CodeTemplate('PyObject* THP${op}_${name}_raw_getter(THPCppFunction *self, void *_unused) {\n  HANDLE_TH_ERRORS\n  const auto& prop = static_cast<${op}*>(self->cdata.get())->${name}_;\n  ${body}\n  END_HANDLE_TH_ERRORS\n}\n')
GETTER_DEFINITION_VEC_SAVEDVAR = CodeTemplate('PyObject* THP${op}_${name}_getter(THPCppFunction *self, void *_unused) {\n  HANDLE_TH_ERRORS\n  const auto *node = static_cast<${op}*>(self->cdata.get());\n  const auto& prop = node->${name}_;\n  if (node->${name}_released_) {\n    PyErr_SetString(PyExc_RuntimeError, ERR_BACKWARD_TWICE);\n    return nullptr;\n  }\n  ${body}\n  END_HANDLE_TH_ERRORS\n}\n')
GETTER_DEFINITION_RAW_VEC_SAVEDVAR = CodeTemplate('PyObject* THP${op}_${name}_raw_getter(THPCppFunction *self, void *_unused) {\n  HANDLE_TH_ERRORS\n  const auto *node = static_cast<${op}*>(self->cdata.get());\n  const auto& prop = node->${name}_;\n  if (node->${name}_released_) {\n    PyErr_SetString(PyExc_RuntimeError, ERR_BACKWARD_TWICE);\n    return nullptr;\n  }\n  ${body}\n  END_HANDLE_TH_ERRORS\n}\n')
GETTER_DEFINITION_OPT = CodeTemplate('PyObject* THP${op}_${name}_getter(THPCppFunction *self, void *_unused) {\n  HANDLE_TH_ERRORS\n  auto opt_prop = static_cast<${op}*>(self->cdata.get())->${name};\n  if (!opt_prop.has_value()) {\n    Py_RETURN_NONE;\n  }\n  auto prop = opt_prop.value();\n  ${body}\n  END_HANDLE_TH_ERRORS\n}\n')
GETTER_DEFINITION_OPT_ARRAYREF = CodeTemplate('PyObject* THP${op}_${name}_getter(THPCppFunction *self, void *_unused) {\n  HANDLE_TH_ERRORS\n  auto opt_prop = static_cast<${op}*>(self->cdata.get())->${name};\n  if (!opt_prop.list.has_value()) {\n    Py_RETURN_NONE;\n  }\n  auto prop = opt_prop.list.value();\n  ${body}\n  END_HANDLE_TH_ERRORS\n}\n')
GETTER_BODY_SAVEDVAR = 'return THPVariable_Wrap(prop.unpack(self->cdata));\n'
GETTER_BODY_RAW_SAVEDVAR = 'pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);\nreturn obj.release().ptr();\n'
GETTER_BODY_VEC_SAVEDVAR = 'PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());\nfor (auto i: c10::irange(prop.size())) {\n  PyTuple_SetItem(tup, (Py_ssize_t) i, THPVariable_Wrap(prop[i].unpack(self->cdata)));\n}\nreturn tup;\n'
GETTER_BODY_RAW_VEC_SAVEDVAR = 'PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());\nfor (auto i : c10::irange(prop.size())) {\n  pybind11::object obj = pybind11::cast(prop[i], pybind11::return_value_policy::reference);\n  PyTuple_SetItem(tup, (Py_ssize_t) i, obj.release().ptr());\n}\nreturn tup;\n'
GETTER_BODY_ARRAYREF_LONG = 'PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());\nfor (auto i : c10::irange(prop.size())) {\n  PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));\n}\nreturn tup;\n'
GETTER_BODY_ARRAYREF_SYMINT = 'PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());\nfor (auto i : c10::irange(prop.size())) {\n    auto si = prop[i];\n    if (auto m = si.maybe_as_int()) {\n      PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));\n    } else {\n      auto py_symint = py::cast(si).release().ptr();\n      PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);\n    }\n}\nreturn tup;\n'
GETTER_BODY_ARRAYREF_DOUBLE = 'PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());\nfor (auto i : c10::irange(prop.size())) {\n  PyTuple_SetItem(tup, (Py_ssize_t) i, PyFloat_FromDouble((double) prop[i]));\n}\nreturn tup;\n'
GETTER_BODY_INT64_T = 'return PyLong_FromUnsignedLong((int64_t) prop);\n'
GETTER_BODY_SYMINT = 'if (auto m = prop.maybe_as_int()) {\n  return PyLong_FromUnsignedLong(*m);\n} else {\n  return py::cast(prop).release().ptr();\n}\n'
GETTER_BODY_DOUBLE = 'return PyFloat_FromDouble((double) prop);\n'
GETTER_BODY_BOOL = 'if (prop) {\n  Py_RETURN_TRUE;\n} else {\n  Py_RETURN_FALSE;\n}\n'
GETTER_BODY_STRING = 'return PyUnicode_FromStringAndSize(prop.data(), prop.size());\n'
GETTER_BODY_SCALAR = 'if (prop.isComplex()) {\n  auto cprop = prop.to<c10::complex<double>>();\n  return PyComplex_FromDoubles(cprop.real(), cprop.imag());\n} else if (prop.isFloatingPoint()) {\n  return PyFloat_FromDouble(prop.to<double>());\n} else if (prop.isIntegral(/*includeBool=*/false)) {\n  return PyLong_FromLong(prop.to<int64_t>());\n} else if (prop.isBoolean()) {\n  if (prop.to<bool>()) {\n    Py_RETURN_TRUE;\n  } else {\n    Py_RETURN_FALSE;\n  }\n} else {\n  PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");\n  return nullptr;\n}\n'
GETTER_BODY_VEC_SCALAR = 'PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());\nfor (auto i: c10::irange(prop.size())) {\n  if (prop[i].isComplex()) {\n    auto cprop = prop[i].to<c10::complex<double>>();\n    PyTuple_SetItem(tup, (Py_ssize_t) i, PyComplex_FromDoubles(cprop.real(), cprop.imag()));\n  } else if (prop[i].isFloatingPoint()) {\n    auto double_prop = prop[i].to<double>();\n    PyTuple_SetItem(tup, (Py_ssize_t) i, PyFloat_FromDouble(double_prop));\n  } else if (prop[i].isIntegral(/*includeBool=*/false)) {\n    auto long_prop = prop[i].to<int64_t>();\n    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromLong(long_prop));\n  } else if (prop[i].isBoolean()) {\n    if (prop[i].to<bool>()) {\n      PyTuple_SetItem(tup, (Py_ssize_t) i, Py_True);\n    } else {\n      PyTuple_SetItem(tup, (Py_ssize_t) i, Py_False);\n    }\n  } else {\n    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");\n    return nullptr;\n  }\n}\nreturn tup;\n'
MISC_GETTER_DEFS = {OptionalCType(BaseCType(longT)): (GETTER_DEFINITION_OPT, GETTER_BODY_INT64_T), OptionalCType(BaseCType(SymIntT)): (GETTER_DEFINITION_OPT, GETTER_BODY_SYMINT), BaseCType(doubleT): (GETTER_DEFINITION, GETTER_BODY_DOUBLE), OptionalCType(BaseCType(doubleT)): (GETTER_DEFINITION_OPT, GETTER_BODY_DOUBLE), BaseCType(boolT): (GETTER_DEFINITION, GETTER_BODY_BOOL), BaseCType(scalarT): (GETTER_DEFINITION, GETTER_BODY_SCALAR), OptionalCType(BaseCType(scalarT)): (GETTER_DEFINITION_OPT, GETTER_BODY_SCALAR)}
UNTRACEABLE_FUNCTIONS = VIEW_FUNCTIONS

def get_infos_with_derivatives_list(differentiability_infos: Dict[FunctionSchema, Dict[str, DifferentiabilityInfo]]) -> List[DifferentiabilityInfo]:
    if False:
        return 10
    diff_info_list = [info for diffinfo_dict in differentiability_infos.values() for info in diffinfo_dict.values()]
    return list(filter(lambda info: info.args_with_derivatives, diff_info_list))

def gen_autograd_functions_lib(out: str, differentiability_infos: Dict[FunctionSchema, Dict[str, DifferentiabilityInfo]], template_path: str) -> None:
    if False:
        while True:
            i = 10
    'Functions.h and Functions.cpp body\n\n    These contain the auto-generated subclasses of torch::autograd::Node\n    for each every differentiable torch function.\n    '
    infos = get_infos_with_derivatives_list(differentiability_infos)
    declarations = [process_function(f, FUNCTION_DECLARATION) for f in infos]
    definitions = [process_function(f, FUNCTION_DEFINITION) for f in infos]
    file_basename = 'Functions'
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    for suffix in ['.h', '.cpp']:
        fname = file_basename + suffix
        fm.write_with_template(fname, fname, lambda : {'generated_comment': '@' + f'generated from {fm.template_dir_for_comments()}/' + fname, 'autograd_function_declarations': declarations, 'autograd_function_definitions': definitions})

def gen_autograd_functions_python(out: str, differentiability_infos: Dict[FunctionSchema, Dict[str, DifferentiabilityInfo]], template_path: str) -> None:
    if False:
        return 10
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    num_shards = 5
    fm.write('python_functions.h', lambda : {'generated_comment': '@' + f'generated from {fm.template_dir_for_comments()}/python_functions.h', 'shard_forward_declare': [f'void initialize_autogenerated_functions_{i}(PyObject* module);' for i in range(num_shards)], 'shard_call': [f'initialize_autogenerated_functions_{i}(module);' for i in range(num_shards)]})
    infos = get_infos_with_derivatives_list(differentiability_infos)
    fm.write_sharded('python_functions.cpp', infos, key_fn=lambda info: info.name, base_env={'generated_comment': '@' + f'generated from {fm.template_dir_for_comments()}/python_functions.cpp'}, env_callable=lambda info: {'py_function_initializers': [process_function(info, PY_FUNCTION_DEFINITION)], 'py_function_props_and_getters': [process_function(info, PY_FUNCTION_PROPS_AND_GETTERS)]}, num_shards=num_shards, sharded_keys={'py_function_initializers', 'py_function_props_and_getters'})

def process_function(info: DifferentiabilityInfo, template: CodeTemplate) -> str:
    if False:
        return 10
    saved_variables: List[str] = []
    release_variables: List[str] = []
    saved_list_sizes: List[str] = []
    unpack: List[str] = []
    asserts: List[str] = []
    compute_index_ranges: List[str] = []
    getter_definitions: List[str] = []
    py_getsetdef_structs: List[str] = []
    compiled_args: List[str] = []
    apply_with_saved_before: List[str] = []
    apply_with_saved_after: List[str] = []
    for arg in info.args_with_derivatives:
        if arg.type in TENSOR_LIST_LIKE_CTYPES:
            size = f'{arg.name}_size_'
            saved_list_sizes.append(f'size_t {arg.name}_size_;')
        else:
            size = '1'
        compute_index_ranges.append(f'auto {arg.name}_ix = gen.range({size});')

    def save_var(var: SavedAttribute, is_output: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        name = var.nctype.name
        type = var.nctype.type
        should_append_getsetdef = True
        should_append_raw_getsetdef = False
        visit_name = name
        if type == BaseCType(tensorT) or type == OptionalCType(BaseCType(tensorT)) or type == MutRefCType(OptionalCType(BaseCType(tensorT))) or (type == BaseCType(scalarT) and is_output):
            saved_variables.append(f'SavedVariable {name}_;')
            release_variables.append(f'{name}_.reset_data();')
            ptr = 'shared_from_this()' if is_output else ''
            unpack.append(f'auto {name} = {name}_.unpack({ptr});')
            getter_definitions.append(GETTER_DEFINITION_SAVEDVAR.substitute(op=info.op, name=name, body=GETTER_BODY_SAVEDVAR))
            getter_definitions.append(GETTER_DEFINITION_RAW_SAVEDVAR.substitute(op=info.op, name=name, body=GETTER_BODY_RAW_SAVEDVAR))
            should_append_raw_getsetdef = True
            visit_name = f'{name}_'
        elif type == BaseCType(tensorListT) or type == BaseCType(iTensorListRefT) or type == VectorCType(BaseCType(tensorT)):
            if type == VectorCType(BaseCType(tensorT)):
                assert info.func.func.name.name.base.startswith('_foreach') and is_output
            saved_variables.append(f'std::vector<SavedVariable> {name}_;')
            saved_variables.append(f'bool {name}_released_ = false;')
            release_variables.append(f'{name}_.clear();')
            release_variables.append(f'{name}_released_ = true;')
            ptr = 'shared_from_this()' if is_output else 'nullptr'
            unpack.append(f'auto {name} = unpack_list({name}_, {ptr});')
            asserts.append(f'TORCH_CHECK(!{name}_released_, ERR_BACKWARD_TWICE);')
            getter_definitions.append(GETTER_DEFINITION_VEC_SAVEDVAR.substitute(op=info.op, name=name, body=GETTER_BODY_VEC_SAVEDVAR))
            getter_definitions.append(GETTER_DEFINITION_RAW_VEC_SAVEDVAR.substitute(op=info.op, name=name, body=GETTER_BODY_RAW_VEC_SAVEDVAR))
            should_append_raw_getsetdef = True
            visit_name = f'{name}_'
        elif type == ListCType(OptionalCType(BaseCType(tensorT))):
            saved_variables.append(f'std::vector<SavedVariable> {name}_;')
            saved_variables.append(f'bool {name}_released_ = false;')
            release_variables.append(f'{name}_.clear();')
            release_variables.append(f'{name}_released_ = true;')
            unpack.append(f'auto {name} = unpack_opt_list({name}_);')
            asserts.append(f'TORCH_CHECK(!{name}_released_, ERR_BACKWARD_TWICE);')
            getter_definitions.append(GETTER_DEFINITION_VEC_SAVEDVAR.substitute(op=info.op, name=name, body=GETTER_BODY_VEC_SAVEDVAR))
            getter_definitions.append(GETTER_DEFINITION_RAW_VEC_SAVEDVAR.substitute(op=info.op, name=name, body=GETTER_BODY_RAW_VEC_SAVEDVAR))
            should_append_raw_getsetdef = True
            visit_name = f'{name}_'
        elif type == BaseCType(intArrayRefT):
            saved_variables.append(f'std::vector<int64_t> {name};')
            getter_definitions.append(GETTER_DEFINITION.substitute(op=info.op, name=name, body=GETTER_BODY_ARRAYREF_LONG))
        elif type == BaseCType(symIntArrayRefT):
            saved_variables.append(f'std::vector<c10::SymInt> {name};')
            getter_definitions.append(GETTER_DEFINITION.substitute(op=info.op, name=name, body=GETTER_BODY_ARRAYREF_SYMINT))
        elif type == BaseCType(optionalIntArrayRefT):
            saved_variables.append(f'c10::OptionalArray<int64_t> {name};')
            getter_definitions.append(GETTER_DEFINITION_OPT_ARRAYREF.substitute(op=info.op, name=name, body=GETTER_BODY_ARRAYREF_LONG))
        elif type == BaseCType(optionalSymIntArrayRefT):
            saved_variables.append(f'c10::OptionalArray<c10::SymInt> {name};')
            getter_definitions.append(GETTER_DEFINITION_OPT_ARRAYREF.substitute(op=info.op, name=name, body=GETTER_BODY_ARRAYREF_SYMINT))
        elif type == OptionalCType(BaseCType(intArrayRefT)):
            saved_variables.append(f'c10::OptionalArray<int64_t> {name};')
            getter_definitions.append(GETTER_DEFINITION_OPT_ARRAYREF.substitute(op=info.op, name=name, body=GETTER_BODY_ARRAYREF_LONG))
        elif type == OptionalCType(BaseCType(symIntArrayRefT)):
            saved_variables.append(f'c10::OptionalArray<c10::SymInt> {name};')
            getter_definitions.append(GETTER_DEFINITION_OPT_ARRAYREF.substitute(op=info.op, name=name, body=GETTER_BODY_ARRAYREF_SYMINT))
        elif type == OptionalCType(ArrayRefCType(BaseCType(doubleT))):
            saved_variables.append(f'c10::OptionalArray<double> {name};')
            getter_definitions.append(GETTER_DEFINITION_OPT_ARRAYREF.substitute(op=info.op, name=name, body=GETTER_BODY_ARRAYREF_DOUBLE))
        elif type == BaseCType(longT):
            saved_variables.append(f'{type.cpp_type()} {name} = 0;')
            getter_definitions.append(GETTER_DEFINITION.substitute(op=info.op, name=name, body=GETTER_BODY_INT64_T))
        elif type == BaseCType(SymIntT):
            saved_variables.append(f'c10::SymInt {name};')
            getter_definitions.append(GETTER_DEFINITION.substitute(op=info.op, name=name, body=GETTER_BODY_SYMINT))
        elif type == BaseCType(stringT):
            saved_variables.append(f'std::string {name};')
            getter_definitions.append(GETTER_DEFINITION.substitute(op=info.op, name=name, body=GETTER_BODY_STRING))
        elif type == OptionalCType(BaseCType(stringT)):
            saved_variables.append(f'c10::optional<std::string> {name};')
            getter_definitions.append(GETTER_DEFINITION_OPT.substitute(op=info.op, name=name, body=GETTER_BODY_STRING))
        elif type == ArrayRefCType(elem=BaseCType(type=BaseCppType(ns='at', name='Scalar'))):
            saved_variables.append(f'std::vector<at::Scalar> {name};')
            saved_variables.append(f'bool {name}_released_ = false;')
            release_variables.append(f'{name}.clear();')
            getter_definitions.append(CodeTemplate('PyObject* THP${op}_${name}_getter(THPCppFunction *self, void *_unused) {\n  HANDLE_TH_ERRORS\n  const auto *node = static_cast<${op}*>(self->cdata.get());\n  const auto& prop = node->${name};\n  if (node->${name}_released_) {\n    PyErr_SetString(PyExc_RuntimeError, ERR_BACKWARD_TWICE);\n    return nullptr;\n  }\n  ${body}\n  END_HANDLE_TH_ERRORS\n}\n                            ').substitute(op=info.op, name=name, body=GETTER_BODY_VEC_SCALAR))
        else:
            assert 'ref' not in type.cpp_type().lower() and 'view' not in type.cpp_type().lower() and ('*' not in type.cpp_type()) and ('&' not in type.cpp_type()), f'{type.cpp_type()} looks like it contains a non-owning reference'
            saved_variables.append(f'{type.cpp_type()} {name};')
            if type in MISC_GETTER_DEFS:
                (getter_def, body) = MISC_GETTER_DEFS[type]
                getter_definitions.append(getter_def.substitute(op=info.op, name=name, body=body))
            else:
                should_append_getsetdef = False
        if should_append_getsetdef:
            py_getsetdef_structs.append(PY_GETSETDEF_STRUCT.substitute(op=info.op, name=name))
        if should_append_raw_getsetdef:
            py_getsetdef_structs.append(PY_RAW_GETSETDEF_STRUCT.substitute(op=info.op, name=name))
        compiled_args.append(f'args.collect({visit_name});')
        apply_with_saved_before.append(f'saved.before({visit_name});')
        apply_with_saved_after.append(f'saved.after({visit_name});')
    for var in sorted(info.all_saved_inputs, key=lambda sa: str(sa.nctype.name)):
        save_var(var, is_output=False)
    for var in sorted(info.all_saved_outputs, key=lambda sa: str(sa.nctype.name)):
        save_var(var, is_output=True)
    if len(release_variables) > 0:
        thread_lock = 'std::lock_guard<std::mutex> lock(mutex_);'
    else:
        thread_lock = ''
    if uses_retain_variables(info):
        will_release_variables = WILL_RELEASE_VARIABLES.substitute()
    else:
        will_release_variables = ''
    body: List[str] = []
    if uses_single_grad(info):
        body.append('const auto& grad = grads[0];')
    else:
        body.extend((f'const auto& {name} = grads[{info.available_named_gradients.index(name)}];' for name in sorted(info.used_named_gradients)))

    def emit_derivative(derivative: Derivative, args_with_derivatives: Sequence[Binding]) -> Tuple[bool, str]:
        if False:
            for i in range(10):
                print('nop')
        formula = derivative.formula
        var_names = derivative.var_names
        if len(var_names) == 1:
            checks_any_grad_defined = False
            if 'not_implemented' not in formula:
                matching_args = [arg for arg in args_with_derivatives if arg.name == var_names[0]]
                if len(matching_args) == 1:
                    arg = matching_args[0]
                    if isinstance(arg.argument, Argument) and str(arg.argument.type) in ('Tensor', 'Tensor?'):
                        formula = 'any_grad_defined ? (' + formula + ') : Tensor()'
                        checks_any_grad_defined = True
            if info.name.startswith('_foreach_'):
                derivative_template = DERIVATIVE_SINGLE_FOREACH
            else:
                derivative_template = DERIVATIVE_SINGLE
            return (checks_any_grad_defined, derivative_template.substitute(name=var_names[0], derivative=formula))
        else:
            if 'grad_input_mask' in formula:
                masks = [f'task_should_compute_output({{ {n}_ix }}),' for n in var_names]
                grad_input_mask = GRAD_INPUT_MASK.substitute(masks=masks, n=len(var_names))
            else:
                grad_input_mask = ''
            idx_ranges = ', '.join((f'{n}_ix' for n in var_names))
            copy_ranges: List[str] = []
            for (i, n) in enumerate(var_names):
                copy_ranges.append(DERIVATIVE_MULTI_COPY_RANGE.substitute(name=n, i=i))
            return (False, DERIVATIVE_MULTI.substitute(idx_ranges=idx_ranges, copy_ranges=copy_ranges, derivative=formula, grad_input_mask=grad_input_mask))
    body.extend(unpack)
    need_any_grad_defined_var = False
    for derivative in info.derivatives:
        (checks_any_grad_defined, derivative_text) = emit_derivative(derivative, info.args_with_derivatives)
        body.append(derivative_text)
        need_any_grad_defined_var |= checks_any_grad_defined
    if need_any_grad_defined_var:
        body.insert(-len(info.derivatives), 'bool any_grad_defined = any_variable_defined(grads);')
    if info.name in UNTRACEABLE_FUNCTIONS:
        superclass = 'Node'
    else:
        superclass = 'TraceableFunction'
    all_getsetdef_structs = ',\n'.join(py_getsetdef_structs) + ',' if len(py_getsetdef_structs) != 0 else ''
    all_getter_definitions = '\n'.join(getter_definitions)
    return template.substitute(op=info.op, compute_index_ranges=compute_index_ranges, saved_variables=saved_variables, release_variables=release_variables, saved_list_sizes=saved_list_sizes, asserts=asserts, thread_lock=thread_lock, will_release_variables=will_release_variables, body=body, superclass=superclass, all_getter_definitions=all_getter_definitions, all_getsetdef_structs=all_getsetdef_structs, compiled_args=compiled_args, apply_with_saved_before=apply_with_saved_before, apply_with_saved_after=apply_with_saved_after)