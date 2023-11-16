"""Stub generator for C modules.

The public interface is via the mypy.stubgen module.
"""
from __future__ import annotations
import glob
import importlib
import inspect
import keyword
import os.path
from types import FunctionType, ModuleType
from typing import Any, Mapping
from mypy.fastparse import parse_type_comment
from mypy.moduleinspect import is_c_module
from mypy.stubdoc import ArgSig, FunctionSig, Sig, find_unique_signatures, infer_arg_sig_from_anon_docstring, infer_prop_type_from_docstring, infer_ret_type_sig_from_anon_docstring, infer_ret_type_sig_from_docstring, infer_sig_from_docstring, parse_all_signatures
from mypy.stubutil import BaseStubGenerator, ClassInfo, FunctionContext, SignatureGenerator, infer_method_arg_types, infer_method_ret_type

class ExternalSignatureGenerator(SignatureGenerator):

    def __init__(self, func_sigs: dict[str, str] | None=None, class_sigs: dict[str, str] | None=None):
        if False:
            print('Hello World!')
        '\n        Takes a mapping of function/method names to signatures and class name to\n        class signatures (usually corresponds to __init__).\n        '
        self.func_sigs = func_sigs or {}
        self.class_sigs = class_sigs or {}

    @classmethod
    def from_doc_dir(cls, doc_dir: str) -> ExternalSignatureGenerator:
        if False:
            print('Hello World!')
        'Instantiate from a directory of .rst files.'
        all_sigs: list[Sig] = []
        all_class_sigs: list[Sig] = []
        for path in glob.glob(f'{doc_dir}/*.rst'):
            with open(path) as f:
                (loc_sigs, loc_class_sigs) = parse_all_signatures(f.readlines())
            all_sigs += loc_sigs
            all_class_sigs += loc_class_sigs
        sigs = dict(find_unique_signatures(all_sigs))
        class_sigs = dict(find_unique_signatures(all_class_sigs))
        return ExternalSignatureGenerator(sigs, class_sigs)

    def get_function_sig(self, default_sig: FunctionSig, ctx: FunctionContext) -> list[FunctionSig] | None:
        if False:
            return 10
        if ctx.class_info and ctx.name in ('__new__', '__init__') and (ctx.name not in self.func_sigs) and (ctx.class_info.name in self.class_sigs):
            return [FunctionSig(name=ctx.name, args=infer_arg_sig_from_anon_docstring(self.class_sigs[ctx.class_info.name]), ret_type=infer_method_ret_type(ctx.name))]
        if ctx.name not in self.func_sigs:
            return None
        inferred = [FunctionSig(name=ctx.name, args=infer_arg_sig_from_anon_docstring(self.func_sigs[ctx.name]), ret_type=None)]
        if ctx.class_info:
            return self.remove_self_type(inferred, ctx.class_info.self_var)
        else:
            return inferred

    def get_property_type(self, default_type: str | None, ctx: FunctionContext) -> str | None:
        if False:
            i = 10
            return i + 15
        return None

class DocstringSignatureGenerator(SignatureGenerator):

    def get_function_sig(self, default_sig: FunctionSig, ctx: FunctionContext) -> list[FunctionSig] | None:
        if False:
            for i in range(10):
                print('nop')
        inferred = infer_sig_from_docstring(ctx.docstring, ctx.name)
        if inferred:
            assert ctx.docstring is not None
            if is_pybind11_overloaded_function_docstring(ctx.docstring, ctx.name):
                del inferred[-1]
        if ctx.class_info:
            if not inferred and ctx.name == '__init__':
                inferred = infer_sig_from_docstring(ctx.class_info.docstring, ctx.class_info.name)
                if inferred:
                    inferred = [sig._replace(name='__init__') for sig in inferred]
            return self.remove_self_type(inferred, ctx.class_info.self_var)
        else:
            return inferred

    def get_property_type(self, default_type: str | None, ctx: FunctionContext) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        'Infer property type from docstring or docstring signature.'
        if ctx.docstring is not None:
            inferred = infer_ret_type_sig_from_anon_docstring(ctx.docstring)
            if not inferred:
                inferred = infer_ret_type_sig_from_docstring(ctx.docstring, ctx.name)
            if not inferred:
                inferred = infer_prop_type_from_docstring(ctx.docstring)
            return inferred
        else:
            return None

def is_pybind11_overloaded_function_docstring(docstring: str, name: str) -> bool:
    if False:
        print('Hello World!')
    return docstring.startswith(f'{name}(*args, **kwargs)\nOverloaded function.\n\n')

def generate_stub_for_c_module(module_name: str, target: str, known_modules: list[str], doc_dir: str='', *, include_private: bool=False, export_less: bool=False, include_docstrings: bool=False) -> None:
    if False:
        return 10
    'Generate stub for C module.\n\n    Signature generators are called in order until a list of signatures is returned.  The order\n    is:\n    - signatures inferred from .rst documentation (if given)\n    - simple runtime introspection (looking for docstrings and attributes\n      with simple builtin types)\n    - fallback based special method names or "(*args, **kwargs)"\n\n    If directory for target doesn\'t exist it will be created. Existing stub\n    will be overwritten.\n    '
    subdir = os.path.dirname(target)
    if subdir and (not os.path.isdir(subdir)):
        os.makedirs(subdir)
    gen = InspectionStubGenerator(module_name, known_modules, doc_dir, include_private=include_private, export_less=export_less, include_docstrings=include_docstrings)
    gen.generate_module()
    output = gen.output()
    with open(target, 'w') as file:
        file.write(output)

class CFunctionStub:
    """
    Class that mimics a C function in order to provide parseable docstrings.
    """

    def __init__(self, name: str, doc: str, is_abstract: bool=False):
        if False:
            for i in range(10):
                print('nop')
        self.__name__ = name
        self.__doc__ = doc
        self.__abstractmethod__ = is_abstract

    @classmethod
    def _from_sig(cls, sig: FunctionSig, is_abstract: bool=False) -> CFunctionStub:
        if False:
            while True:
                i = 10
        return CFunctionStub(sig.name, sig.format_sig()[:-4], is_abstract)

    @classmethod
    def _from_sigs(cls, sigs: list[FunctionSig], is_abstract: bool=False) -> CFunctionStub:
        if False:
            return 10
        return CFunctionStub(sigs[0].name, '\n'.join((sig.format_sig()[:-4] for sig in sigs)), is_abstract)

    def __get__(self) -> None:
        if False:
            while True:
                i = 10
        '\n        This exists to make this object look like a method descriptor and thus\n        return true for CStubGenerator.ismethod()\n        '
        pass

class InspectionStubGenerator(BaseStubGenerator):
    """Stub generator that does not parse code.

    Generation is performed by inspecting the module's contents, and thus works
    for highly dynamic modules, pyc files, and C modules (via the CStubGenerator
    subclass).
    """

    def __init__(self, module_name: str, known_modules: list[str], doc_dir: str='', _all_: list[str] | None=None, include_private: bool=False, export_less: bool=False, include_docstrings: bool=False, module: ModuleType | None=None) -> None:
        if False:
            i = 10
            return i + 15
        self.doc_dir = doc_dir
        if module is None:
            self.module = importlib.import_module(module_name)
        else:
            self.module = module
        self.is_c_module = is_c_module(self.module)
        self.known_modules = known_modules
        self.resort_members = self.is_c_module
        super().__init__(_all_, include_private, export_less, include_docstrings)
        self.module_name = module_name

    def get_default_function_sig(self, func: object, ctx: FunctionContext) -> FunctionSig:
        if False:
            return 10
        argspec = None
        if not self.is_c_module:
            try:
                argspec = inspect.getfullargspec(func)
            except TypeError:
                pass
        if argspec is None:
            if ctx.class_info is not None:
                return FunctionSig(name=ctx.name, args=infer_c_method_args(ctx.name, ctx.class_info.self_var), ret_type=infer_method_ret_type(ctx.name))
            else:
                return FunctionSig(name=ctx.name, args=[ArgSig(name='*args'), ArgSig(name='**kwargs')], ret_type=None)
        args = argspec.args
        defaults = argspec.defaults
        varargs = argspec.varargs
        kwargs = argspec.varkw
        annotations = argspec.annotations

        def get_annotation(key: str) -> str | None:
            if False:
                return 10
            if key not in annotations:
                return None
            argtype = annotations[key]
            if argtype is None:
                return 'None'
            if not isinstance(argtype, str):
                return self.get_type_fullname(argtype)
            return argtype
        arglist: list[ArgSig] = []
        for (i, arg) in enumerate(args):
            if defaults and i >= len(args) - len(defaults):
                default_value = defaults[i - (len(args) - len(defaults))]
                if arg in annotations:
                    argtype = annotations[arg]
                else:
                    argtype = self.get_type_annotation(default_value)
                    if argtype == 'None':
                        incomplete = self.add_name('_typeshed.Incomplete')
                        argtype = f'{incomplete} | None'
                arglist.append(ArgSig(arg, argtype, default=True))
            else:
                arglist.append(ArgSig(arg, get_annotation(arg), default=False))
        if varargs:
            arglist.append(ArgSig(f'*{varargs}', get_annotation(varargs)))
        if kwargs:
            arglist.append(ArgSig(f'**{kwargs}', get_annotation(kwargs)))
        if ctx.class_info is not None and all((arg.type is None and arg.default is False for arg in arglist)):
            new_args = infer_method_arg_types(ctx.name, ctx.class_info.self_var, [arg.name for arg in arglist if arg.name])
            if new_args is not None:
                arglist = new_args
        ret_type = get_annotation('return') or infer_method_ret_type(ctx.name)
        return FunctionSig(ctx.name, arglist, ret_type)

    def get_sig_generators(self) -> list[SignatureGenerator]:
        if False:
            for i in range(10):
                print('nop')
        if not self.is_c_module:
            return []
        else:
            sig_generators: list[SignatureGenerator] = [DocstringSignatureGenerator()]
            if self.doc_dir:
                sig_generators.insert(0, ExternalSignatureGenerator.from_doc_dir(self.doc_dir))
            return sig_generators

    def strip_or_import(self, type_name: str) -> str:
        if False:
            i = 10
            return i + 15
        'Strips unnecessary module names from typ.\n\n        If typ represents a type that is inside module or is a type coming from builtins, remove\n        module declaration from it. Return stripped name of the type.\n\n        Arguments:\n            typ: name of the type\n        '
        local_modules = ['builtins', self.module_name]
        parsed_type = parse_type_comment(type_name, 0, 0, None)[1]
        assert parsed_type is not None, type_name
        return self.print_annotation(parsed_type, self.known_modules, local_modules)

    def get_obj_module(self, obj: object) -> str | None:
        if False:
            return 10
        'Return module name of the object.'
        return getattr(obj, '__module__', None)

    def is_defined_in_module(self, obj: object) -> bool:
        if False:
            return 10
        'Check if object is considered defined in the current module.'
        module = self.get_obj_module(obj)
        return module is None or module == self.module_name

    def generate_module(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        all_items = self.get_members(self.module)
        if self.resort_members:
            all_items = sorted(all_items, key=lambda x: x[0])
        items = []
        for (name, obj) in all_items:
            if inspect.ismodule(obj) and obj.__name__ in self.known_modules:
                module_name = obj.__name__
                if module_name.startswith(self.module_name + '.'):
                    (pkg_name, mod_name) = module_name.rsplit('.', 1)
                    rel_module = pkg_name[len(self.module_name):] or '.'
                    self.import_tracker.add_import_from(rel_module, [(mod_name, name)])
                    self.import_tracker.reexport(name)
                else:
                    self.import_tracker.add_import(module_name, name)
                    self.import_tracker.reexport(name)
            elif self.is_defined_in_module(obj) and (not inspect.ismodule(obj)):
                items.append((name, obj))
            else:
                obj_module_name = self.get_obj_module(obj)
                if obj_module_name:
                    self.import_tracker.add_import_from(obj_module_name, [(name, None)])
                    if self.should_reexport(name, obj_module_name, name_is_alias=False):
                        self.import_tracker.reexport(name)
        self.set_defined_names(set([name for (name, obj) in all_items if not inspect.ismodule(obj)]))
        if self.resort_members:
            functions: list[str] = []
            types: list[str] = []
            variables: list[str] = []
        else:
            output: list[str] = []
            functions = types = variables = output
        for (name, obj) in items:
            if self.is_function(obj):
                self.generate_function_stub(name, obj, output=functions)
            elif inspect.isclass(obj):
                self.generate_class_stub(name, obj, output=types)
            else:
                self.generate_variable_stub(name, obj, output=variables)
        self._output = []
        if self.resort_members:
            for line in variables:
                self._output.append(line + '\n')
            for line in types:
                if line.startswith('class') and self._output and self._output[-1]:
                    self._output.append('\n')
                self._output.append(line + '\n')
            if self._output and functions:
                self._output.append('\n')
            for line in functions:
                self._output.append(line + '\n')
        else:
            for (i, line) in enumerate(output):
                if self._output and line.startswith('class') and (not self._output[-1].startswith('class') or (len(output) > i + 1 and output[i + 1].startswith('    '))) or (self._output and self._output[-1].startswith('def') and (not line.startswith('def'))):
                    self._output.append('\n')
                self._output.append(line + '\n')
        self.check_undefined_names()

    def is_skipped_attribute(self, attr: str) -> bool:
        if False:
            while True:
                i = 10
        return attr in ('__class__', '__getattribute__', '__str__', '__repr__', '__doc__', '__dict__', '__module__', '__weakref__', '__annotations__') or attr in self.IGNORED_DUNDERS or is_pybind_skipped_attribute(attr) or keyword.iskeyword(attr)

    def get_members(self, obj: object) -> list[tuple[str, Any]]:
        if False:
            i = 10
            return i + 15
        obj_dict: Mapping[str, Any] = getattr(obj, '__dict__')
        results = []
        for name in obj_dict:
            if self.is_skipped_attribute(name):
                continue
            try:
                value = getattr(obj, name)
            except AttributeError:
                continue
            else:
                results.append((name, value))
        return results

    def get_type_annotation(self, obj: object) -> str:
        if False:
            return 10
        '\n        Given an instance, return a string representation of its type that is valid\n        to use as a type annotation.\n        '
        if obj is None or obj is type(None):
            return 'None'
        elif inspect.isclass(obj):
            return 'type[{}]'.format(self.get_type_fullname(obj))
        elif isinstance(obj, FunctionType):
            return self.add_name('typing.Callable')
        elif isinstance(obj, ModuleType):
            return self.add_name('types.ModuleType', require=False)
        else:
            return self.get_type_fullname(type(obj))

    def is_function(self, obj: object) -> bool:
        if False:
            return 10
        if self.is_c_module:
            return inspect.isbuiltin(obj)
        else:
            return inspect.isfunction(obj)

    def is_method(self, class_info: ClassInfo, name: str, obj: object) -> bool:
        if False:
            while True:
                i = 10
        if self.is_c_module:
            return inspect.ismethoddescriptor(obj) or type(obj) in (type(str.index), type(str.__add__), type(str.__new__))
        else:
            return inspect.isfunction(obj)

    def is_classmethod(self, class_info: ClassInfo, name: str, obj: object) -> bool:
        if False:
            while True:
                i = 10
        if self.is_c_module:
            return inspect.isbuiltin(obj) or type(obj).__name__ in ('classmethod', 'classmethod_descriptor')
        else:
            return inspect.ismethod(obj)

    def is_staticmethod(self, class_info: ClassInfo | None, name: str, obj: object) -> bool:
        if False:
            while True:
                i = 10
        if self.is_c_module:
            return False
        else:
            return class_info is not None and isinstance(inspect.getattr_static(class_info.cls, name), staticmethod)

    @staticmethod
    def is_abstract_method(obj: object) -> bool:
        if False:
            i = 10
            return i + 15
        return getattr(obj, '__abstractmethod__', False)

    @staticmethod
    def is_property(class_info: ClassInfo, name: str, obj: object) -> bool:
        if False:
            while True:
                i = 10
        return inspect.isdatadescriptor(obj) or hasattr(obj, 'fget')

    @staticmethod
    def is_property_readonly(prop: Any) -> bool:
        if False:
            while True:
                i = 10
        return hasattr(prop, 'fset') and prop.fset is None

    def is_static_property(self, obj: object) -> bool:
        if False:
            while True:
                i = 10
        'For c-modules, whether the property behaves like an attribute'
        if self.is_c_module:
            return type(obj).__name__ in ('pybind11_static_property', 'StaticProperty')
        else:
            return False

    def process_inferred_sigs(self, inferred: list[FunctionSig]) -> None:
        if False:
            return 10
        for (i, sig) in enumerate(inferred):
            for arg in sig.args:
                if arg.type is not None:
                    arg.type = self.strip_or_import(arg.type)
            if sig.ret_type is not None:
                inferred[i] = sig._replace(ret_type=self.strip_or_import(sig.ret_type))

    def generate_function_stub(self, name: str, obj: object, *, output: list[str], class_info: ClassInfo | None=None) -> None:
        if False:
            while True:
                i = 10
        "Generate stub for a single function or method.\n\n        The result (always a single line) will be appended to 'output'.\n        If necessary, any required names will be added to 'imports'.\n        The 'class_name' is used to find signature of __init__ or __new__ in\n        'class_sigs'.\n        "
        docstring: Any = getattr(obj, '__doc__', None)
        if not isinstance(docstring, str):
            docstring = None
        ctx = FunctionContext(self.module_name, name, docstring=docstring, is_abstract=self.is_abstract_method(obj), class_info=class_info)
        if self.is_private_name(name, ctx.fullname) or self.is_not_in_all(name):
            return
        self.record_name(ctx.name)
        default_sig = self.get_default_function_sig(obj, ctx)
        inferred = self.get_signatures(default_sig, self.sig_generators, ctx)
        self.process_inferred_sigs(inferred)
        decorators = []
        if len(inferred) > 1:
            decorators.append('@{}'.format(self.add_name('typing.overload')))
        if ctx.is_abstract:
            decorators.append('@{}'.format(self.add_name('abc.abstractmethod')))
        if class_info is not None:
            if self.is_staticmethod(class_info, name, obj):
                decorators.append('@staticmethod')
            else:
                for sig in inferred:
                    if not sig.args or sig.args[0].name not in ('self', 'cls'):
                        sig.args.insert(0, ArgSig(name=class_info.self_var))
                if inferred[0].args and inferred[0].args[0].name == 'cls':
                    decorators.append('@classmethod')
        output.extend(self.format_func_def(inferred, decorators=decorators, docstring=docstring))
        self._fix_iter(ctx, inferred, output)

    def _fix_iter(self, ctx: FunctionContext, inferred: list[FunctionSig], output: list[str]) -> None:
        if False:
            print('Hello World!')
        'Ensure that objects which implement old-style iteration via __getitem__\n        are considered iterable.\n        '
        if ctx.class_info and ctx.class_info.cls is not None and (ctx.name == '__getitem__') and ('__iter__' not in ctx.class_info.cls.__dict__):
            item_type: str | None = None
            for sig in inferred:
                if sig.args and sig.args[-1].type == 'int':
                    item_type = sig.ret_type
                    break
            if item_type is None:
                return
            obj = CFunctionStub('__iter__', f'def __iter__(self) -> typing.Iterator[{item_type}]\n')
            self.generate_function_stub('__iter__', obj, output=output, class_info=ctx.class_info)

    def generate_property_stub(self, name: str, raw_obj: object, obj: object, static_properties: list[str], rw_properties: list[str], ro_properties: list[str], class_info: ClassInfo | None=None) -> None:
        if False:
            return 10
        "Generate property stub using introspection of 'obj'.\n\n        Try to infer type from docstring, append resulting lines to 'output'.\n\n        raw_obj : object before evaluation of descriptor (if any)\n        obj : object after evaluation of descriptor\n        "
        docstring = getattr(raw_obj, '__doc__', None)
        fget = getattr(raw_obj, 'fget', None)
        if fget:
            alt_docstr = getattr(fget, '__doc__', None)
            if alt_docstr and docstring:
                docstring += alt_docstr
            elif alt_docstr:
                docstring = alt_docstr
        ctx = FunctionContext(self.module_name, name, docstring=docstring, is_abstract=False, class_info=class_info)
        if self.is_private_name(name, ctx.fullname) or self.is_not_in_all(name):
            return
        self.record_name(ctx.name)
        static = self.is_static_property(raw_obj)
        readonly = self.is_property_readonly(raw_obj)
        if static:
            ret_type: str | None = self.strip_or_import(self.get_type_annotation(obj))
        else:
            default_sig = self.get_default_function_sig(raw_obj, ctx)
            ret_type = default_sig.ret_type
        inferred_type = self.get_property_type(ret_type, self.sig_generators, ctx)
        if inferred_type is not None:
            inferred_type = self.strip_or_import(inferred_type)
        if static:
            classvar = self.add_name('typing.ClassVar')
            trailing_comment = '  # read-only' if readonly else ''
            if inferred_type is None:
                inferred_type = self.add_name('_typeshed.Incomplete')
            static_properties.append(f'{self._indent}{name}: {classvar}[{inferred_type}] = ...{trailing_comment}')
        elif readonly:
            ro_properties.append(f'{self._indent}@property')
            sig = FunctionSig(name, [ArgSig('self')], inferred_type)
            ro_properties.append(sig.format_sig(indent=self._indent))
        else:
            if inferred_type is None:
                inferred_type = self.add_name('_typeshed.Incomplete')
            rw_properties.append(f'{self._indent}{name}: {inferred_type}')

    def get_type_fullname(self, typ: type) -> str:
        if False:
            while True:
                i = 10
        'Given a type, return a string representation'
        if typ is Any:
            return 'Any'
        typename = getattr(typ, '__qualname__', typ.__name__)
        module_name = self.get_obj_module(typ)
        assert module_name is not None, typ
        if module_name != 'builtins':
            typename = f'{module_name}.{typename}'
        return typename

    def get_base_types(self, obj: type) -> list[str]:
        if False:
            return 10
        all_bases = type.mro(obj)
        if all_bases[-1] is object:
            del all_bases[-1]
        if all_bases and all_bases[-1].__name__ == 'pybind11_object':
            del all_bases[-1]
        all_bases = all_bases[1:]
        bases: list[type] = []
        for base in all_bases:
            if not any((issubclass(b, base) for b in bases)):
                bases.append(base)
        return [self.strip_or_import(self.get_type_fullname(base)) for base in bases]

    def generate_class_stub(self, class_name: str, cls: type, output: list[str]) -> None:
        if False:
            while True:
                i = 10
        "Generate stub for a single class using runtime introspection.\n\n        The result lines will be appended to 'output'. If necessary, any\n        required names will be added to 'imports'.\n        "
        raw_lookup = getattr(cls, '__dict__')
        items = self.get_members(cls)
        if self.resort_members:
            items = sorted(items, key=lambda x: method_name_sort_key(x[0]))
        names = set((x[0] for x in items))
        methods: list[str] = []
        types: list[str] = []
        static_properties: list[str] = []
        rw_properties: list[str] = []
        ro_properties: list[str] = []
        attrs: list[tuple[str, Any]] = []
        self.record_name(class_name)
        self.indent()
        class_info = ClassInfo(class_name, '', getattr(cls, '__doc__', None), cls)
        for (attr, value) in items:
            raw_value = raw_lookup.get(attr, value)
            if self.is_method(class_info, attr, value) or self.is_classmethod(class_info, attr, value):
                if attr == '__new__':
                    if '__init__' in names:
                        continue
                    attr = '__init__'
                if self.is_classmethod(class_info, attr, value):
                    class_info.self_var = 'cls'
                else:
                    class_info.self_var = 'self'
                self.generate_function_stub(attr, value, output=methods, class_info=class_info)
            elif self.is_property(class_info, attr, raw_value):
                self.generate_property_stub(attr, raw_value, value, static_properties, rw_properties, ro_properties, class_info)
            elif inspect.isclass(value) and self.is_defined_in_module(value):
                self.generate_class_stub(attr, value, types)
            else:
                attrs.append((attr, value))
        for (attr, value) in attrs:
            if attr == '__hash__' and value is None:
                continue
            prop_type_name = self.strip_or_import(self.get_type_annotation(value))
            classvar = self.add_name('typing.ClassVar')
            static_properties.append(f'{self._indent}{attr}: {classvar}[{prop_type_name}] = ...')
        self.dedent()
        bases = self.get_base_types(cls)
        if bases:
            bases_str = '(%s)' % ', '.join(bases)
        else:
            bases_str = ''
        if types or static_properties or rw_properties or methods or ro_properties:
            output.append(f'{self._indent}class {class_name}{bases_str}:')
            for line in types:
                if output and output[-1] and (not output[-1].strip().startswith('class')) and line.strip().startswith('class'):
                    output.append('')
                output.append(line)
            for line in static_properties:
                output.append(line)
            for line in rw_properties:
                output.append(line)
            for line in methods:
                output.append(line)
            for line in ro_properties:
                output.append(line)
        else:
            output.append(f'{self._indent}class {class_name}{bases_str}: ...')

    def generate_variable_stub(self, name: str, obj: object, output: list[str]) -> None:
        if False:
            print('Hello World!')
        "Generate stub for a single variable using runtime introspection.\n\n        The result lines will be appended to 'output'. If necessary, any\n        required names will be added to 'imports'.\n        "
        if self.is_private_name(name, f'{self.module_name}.{name}') or self.is_not_in_all(name):
            return
        self.record_name(name)
        type_str = self.strip_or_import(self.get_type_annotation(obj))
        output.append(f'{name}: {type_str}')

def method_name_sort_key(name: str) -> tuple[int, str]:
    if False:
        for i in range(10):
            print('nop')
    'Sort methods in classes in a typical order.\n\n    I.e.: constructor, normal methods, special methods.\n    '
    if name in ('__new__', '__init__'):
        return (0, name)
    if name.startswith('__') and name.endswith('__'):
        return (2, name)
    return (1, name)

def is_pybind_skipped_attribute(attr: str) -> bool:
    if False:
        i = 10
        return i + 15
    return attr.startswith('__pybind11_module_local_')

def infer_c_method_args(name: str, self_var: str='self', arg_names: list[str] | None=None) -> list[ArgSig]:
    if False:
        print('Hello World!')
    args: list[ArgSig] | None = None
    if name.startswith('__') and name.endswith('__'):
        name = name[2:-2]
        if name in ('hash', 'iter', 'next', 'sizeof', 'copy', 'deepcopy', 'reduce', 'getinitargs', 'int', 'float', 'trunc', 'complex', 'bool', 'abs', 'bytes', 'dir', 'len', 'reversed', 'round', 'index', 'enter'):
            args = []
        elif name == 'getitem':
            args = [ArgSig(name='index')]
        elif name == 'setitem':
            args = [ArgSig(name='index'), ArgSig(name='object')]
        elif name in ('delattr', 'getattr'):
            args = [ArgSig(name='name')]
        elif name == 'setattr':
            args = [ArgSig(name='name'), ArgSig(name='value')]
        elif name == 'getstate':
            args = []
        elif name == 'setstate':
            args = [ArgSig(name='state')]
        elif name in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            args = [ArgSig(name='other', type='object')]
        elif name in ('add', 'radd', 'sub', 'rsub', 'mul', 'rmul', 'mod', 'rmod', 'floordiv', 'rfloordiv', 'truediv', 'rtruediv', 'divmod', 'rdivmod', 'pow', 'rpow', 'xor', 'rxor', 'or', 'ror', 'and', 'rand', 'lshift', 'rlshift', 'rshift', 'rrshift', 'contains', 'delitem', 'iadd', 'iand', 'ifloordiv', 'ilshift', 'imod', 'imul', 'ior', 'ipow', 'irshift', 'isub', 'itruediv', 'ixor'):
            args = [ArgSig(name='other')]
        elif name in ('neg', 'pos', 'invert'):
            args = []
        elif name == 'get':
            args = [ArgSig(name='instance'), ArgSig(name='owner')]
        elif name == 'set':
            args = [ArgSig(name='instance'), ArgSig(name='value')]
        elif name == 'reduce_ex':
            args = [ArgSig(name='protocol')]
        elif name == 'exit':
            args = [ArgSig(name='type', type='type[BaseException] | None'), ArgSig(name='value', type='BaseException | None'), ArgSig(name='traceback', type='types.TracebackType | None')]
    if args is None:
        args = infer_method_arg_types(name, self_var, arg_names)
    else:
        args = [ArgSig(name=self_var)] + args
    if args is None:
        args = [ArgSig(name='*args'), ArgSig(name='**kwargs')]
    return args