"""Generate C code for a Python C extension module from Python source code."""
from __future__ import annotations
import json
import os
from typing import Iterable, List, Optional, Tuple, TypeVar
from mypy.build import BuildResult, BuildSource, State, build, compute_hash, create_metastore, get_cache_names, sorted_components
from mypy.errors import CompileError
from mypy.fscache import FileSystemCache
from mypy.nodes import MypyFile
from mypy.options import Options
from mypy.plugin import Plugin, ReportConfigContext
from mypy.util import hash_digest
from mypyc.codegen.cstring import c_string_initializer
from mypyc.codegen.emit import Emitter, EmitterContext, HeaderDeclaration, c_array_initializer
from mypyc.codegen.emitclass import generate_class, generate_class_type_decl
from mypyc.codegen.emitfunc import generate_native_function, native_function_header
from mypyc.codegen.emitwrapper import generate_legacy_wrapper_function, generate_wrapper_function, legacy_wrapper_function_header, wrapper_function_header
from mypyc.codegen.literals import Literals
from mypyc.common import MODULE_PREFIX, PREFIX, RUNTIME_C_FILES, TOP_LEVEL_NAME, shared_lib_name, short_id_from_name, use_vectorcall
from mypyc.errors import Errors
from mypyc.ir.class_ir import ClassIR
from mypyc.ir.func_ir import FuncIR
from mypyc.ir.module_ir import ModuleIR, ModuleIRs, deserialize_modules
from mypyc.ir.ops import DeserMaps, LoadLiteral
from mypyc.ir.rtypes import RType
from mypyc.irbuild.main import build_ir
from mypyc.irbuild.mapper import Mapper
from mypyc.irbuild.prepare import load_type_map
from mypyc.namegen import NameGenerator, exported_name
from mypyc.options import CompilerOptions
from mypyc.transform.exceptions import insert_exception_handling
from mypyc.transform.refcount import insert_ref_count_opcodes
from mypyc.transform.uninit import insert_uninit_checks
Group = Tuple[List[BuildSource], Optional[str]]
Groups = List[Group]
FileContents = List[Tuple[str, str]]

class MarkedDeclaration:
    """Add a mark, useful for topological sort."""

    def __init__(self, declaration: HeaderDeclaration, mark: bool) -> None:
        if False:
            while True:
                i = 10
        self.declaration = declaration
        self.mark = False

class MypycPlugin(Plugin):
    """Plugin for making mypyc interoperate properly with mypy incremental mode.

    Basically the point of this plugin is to force mypy to recheck things
    based on the demands of mypyc in a couple situations:
      * Any modules in the same group must be compiled together, so we
        tell mypy that modules depend on all their groupmates.
      * If the IR metadata is missing or stale or any of the generated
        C source files associated missing or stale, then we need to
        recompile the module so we mark it as stale.
    """

    def __init__(self, options: Options, compiler_options: CompilerOptions, groups: Groups) -> None:
        if False:
            return 10
        super().__init__(options)
        self.group_map: dict[str, tuple[str | None, list[str]]] = {}
        for (sources, name) in groups:
            modules = sorted((source.module for source in sources))
            for id in modules:
                self.group_map[id] = (name, modules)
        self.compiler_options = compiler_options
        self.metastore = create_metastore(options)

    def report_config_data(self, ctx: ReportConfigContext) -> tuple[str | None, list[str]] | None:
        if False:
            while True:
                i = 10
        (id, path, is_check) = (ctx.id, ctx.path, ctx.is_check)
        if id not in self.group_map:
            return None
        if not is_check:
            return self.group_map[id]
        (meta_path, _, _) = get_cache_names(id, path, self.options)
        ir_path = get_ir_cache_name(id, path, self.options)
        try:
            meta_json = self.metastore.read(meta_path)
            ir_json = self.metastore.read(ir_path)
        except FileNotFoundError:
            return None
        ir_data = json.loads(ir_json)
        if compute_hash(meta_json) != ir_data['meta_hash']:
            return None
        for (path, hash) in ir_data['src_hashes'].items():
            try:
                with open(os.path.join(self.compiler_options.target_dir, path), 'rb') as f:
                    contents = f.read()
            except FileNotFoundError:
                return None
            real_hash = hash_digest(contents)
            if hash != real_hash:
                return None
        return self.group_map[id]

    def get_additional_deps(self, file: MypyFile) -> list[tuple[int, str, int]]:
        if False:
            for i in range(10):
                print('nop')
        return [(10, id, -1) for id in self.group_map.get(file.fullname, (None, []))[1]]

def parse_and_typecheck(sources: list[BuildSource], options: Options, compiler_options: CompilerOptions, groups: Groups, fscache: FileSystemCache | None=None, alt_lib_path: str | None=None) -> BuildResult:
    if False:
        while True:
            i = 10
    assert options.strict_optional, 'strict_optional must be turned on'
    result = build(sources=sources, options=options, alt_lib_path=alt_lib_path, fscache=fscache, extra_plugins=[MypycPlugin(options, compiler_options, groups)])
    if result.errors:
        raise CompileError(result.errors)
    return result

def compile_scc_to_ir(scc: list[MypyFile], result: BuildResult, mapper: Mapper, compiler_options: CompilerOptions, errors: Errors) -> ModuleIRs:
    if False:
        for i in range(10):
            print('nop')
    'Compile an SCC into ModuleIRs.\n\n    Any modules that this SCC depends on must have either compiled or\n    loaded from a cache into mapper.\n\n    Arguments:\n        scc: The list of MypyFiles to compile\n        result: The BuildResult from the mypy front-end\n        mapper: The Mapper object mapping mypy ASTs to class and func IRs\n        compiler_options: The compilation options\n        errors: Where to report any errors encountered\n\n    Returns the IR of the modules.\n    '
    if compiler_options.verbose:
        print('Compiling {}'.format(', '.join((x.name for x in scc))))
    modules = build_ir(scc, result.graph, result.types, mapper, compiler_options, errors)
    if errors.num_errors > 0:
        return modules
    for module in modules.values():
        for fn in module.functions:
            insert_uninit_checks(fn)
    for module in modules.values():
        for fn in module.functions:
            insert_exception_handling(fn)
    for module in modules.values():
        for fn in module.functions:
            insert_ref_count_opcodes(fn)
    return modules

def compile_modules_to_ir(result: BuildResult, mapper: Mapper, compiler_options: CompilerOptions, errors: Errors) -> ModuleIRs:
    if False:
        i = 10
        return i + 15
    "Compile a collection of modules into ModuleIRs.\n\n    The modules to compile are specified as part of mapper's group_map.\n\n    Returns the IR of the modules.\n    "
    deser_ctx = DeserMaps({}, {})
    modules = {}
    for scc in sorted_components(result.graph):
        scc_states = [result.graph[id] for id in scc]
        trees = [st.tree for st in scc_states if st.id in mapper.group_map and st.tree]
        if not trees:
            continue
        fresh = all((id not in result.manager.rechecked_modules for id in scc))
        if fresh:
            load_scc_from_cache(trees, result, mapper, deser_ctx)
        else:
            scc_ir = compile_scc_to_ir(trees, result, mapper, compiler_options, errors)
            modules.update(scc_ir)
    return modules

def compile_ir_to_c(groups: Groups, modules: ModuleIRs, result: BuildResult, mapper: Mapper, compiler_options: CompilerOptions) -> dict[str | None, list[tuple[str, str]]]:
    if False:
        print('Hello World!')
    'Compile a collection of ModuleIRs to C source text.\n\n    Returns a dictionary mapping group names to a list of (file name,\n    file text) pairs.\n    '
    source_paths = {source.module: result.graph[source.module].xpath for (sources, _) in groups for source in sources}
    names = NameGenerator([[source.module for source in sources] for (sources, _) in groups])
    ctext: dict[str | None, list[tuple[str, str]]] = {}
    for (group_sources, group_name) in groups:
        group_modules = {source.module: modules[source.module] for source in group_sources if source.module in modules}
        if not group_modules:
            ctext[group_name] = []
            continue
        generator = GroupGenerator(group_modules, source_paths, group_name, mapper.group_map, names, compiler_options)
        ctext[group_name] = generator.generate_c_for_modules()
    return ctext

def get_ir_cache_name(id: str, path: str, options: Options) -> str:
    if False:
        print('Hello World!')
    (meta_path, _, _) = get_cache_names(id, path, options)
    return meta_path.replace('.meta.json', '.ir.json')

def get_state_ir_cache_name(state: State) -> str:
    if False:
        i = 10
        return i + 15
    return get_ir_cache_name(state.id, state.xpath, state.options)

def write_cache(modules: ModuleIRs, result: BuildResult, group_map: dict[str, str | None], ctext: dict[str | None, list[tuple[str, str]]]) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Write out the cache information for modules.\n\n    Each module has the following cache information written (which is\n    in addition to the cache information written by mypy itself):\n      * A serialized version of its mypyc IR, minus the bodies of\n        functions. This allows code that depends on it to use\n        these serialized data structures when compiling against it\n        instead of needing to recompile it. (Compiling against a\n        module requires access to both its mypy and mypyc data\n        structures.)\n      * The hash of the mypy metadata cache file for the module.\n        This is used to ensure that the mypyc cache and the mypy\n        cache are in sync and refer to the same version of the code.\n        This is particularly important if mypyc crashes/errors/is\n        stopped after mypy has written its cache but before mypyc has.\n      * The hashes of all of the source file outputs for the group\n        the module is in. This is so that the module will be\n        recompiled if the source outputs are missing.\n    '
    hashes = {}
    for (name, files) in ctext.items():
        hashes[name] = {file: compute_hash(data) for (file, data) in files}
    for (id, module) in modules.items():
        st = result.graph[id]
        (meta_path, _, _) = get_cache_names(id, st.xpath, result.manager.options)
        try:
            meta_data = result.manager.metastore.read(meta_path)
        except OSError:
            continue
        newpath = get_state_ir_cache_name(st)
        ir_data = {'ir': module.serialize(), 'meta_hash': compute_hash(meta_data), 'src_hashes': hashes[group_map[id]]}
        result.manager.metastore.write(newpath, json.dumps(ir_data, separators=(',', ':')))
    result.manager.metastore.commit()

def load_scc_from_cache(scc: list[MypyFile], result: BuildResult, mapper: Mapper, ctx: DeserMaps) -> ModuleIRs:
    if False:
        for i in range(10):
            print('nop')
    'Load IR for an SCC of modules from the cache.\n\n    Arguments and return are as compile_scc_to_ir.\n    '
    cache_data = {k.fullname: json.loads(result.manager.metastore.read(get_state_ir_cache_name(result.graph[k.fullname])))['ir'] for k in scc}
    modules = deserialize_modules(cache_data, ctx)
    load_type_map(mapper, scc, ctx)
    return modules

def compile_modules_to_c(result: BuildResult, compiler_options: CompilerOptions, errors: Errors, groups: Groups) -> tuple[ModuleIRs, list[FileContents]]:
    if False:
        for i in range(10):
            print('nop')
    'Compile Python module(s) to the source of Python C extension modules.\n\n    This generates the source code for the "shared library" module\n    for each group. The shim modules are generated in mypyc.build.\n    Each shared library module provides, for each module in its group,\n    a PyCapsule containing an initialization function.\n    Additionally, it provides a capsule containing an export table of\n    pointers to all of the group\'s functions and static variables.\n\n    Arguments:\n        result: The BuildResult from the mypy front-end\n        compiler_options: The compilation options\n        errors: Where to report any errors encountered\n        groups: The groups that we are compiling. See documentation of Groups type above.\n\n    Returns the IR of the modules and a list containing the generated files for each group.\n    '
    group_map = {source.module: lib_name for (group, lib_name) in groups for source in group}
    mapper = Mapper(group_map)
    result.manager.errors.set_file('<mypyc>', module=None, scope=None, options=result.manager.options)
    modules = compile_modules_to_ir(result, mapper, compiler_options, errors)
    ctext = compile_ir_to_c(groups, modules, result, mapper, compiler_options)
    if errors.num_errors == 0:
        write_cache(modules, result, group_map, ctext)
    return (modules, [ctext[name] for (_, name) in groups])

def generate_function_declaration(fn: FuncIR, emitter: Emitter) -> None:
    if False:
        while True:
            i = 10
    emitter.context.declarations[emitter.native_function_name(fn.decl)] = HeaderDeclaration(f'{native_function_header(fn.decl, emitter)};', needs_export=True)
    if fn.name != TOP_LEVEL_NAME:
        if is_fastcall_supported(fn, emitter.capi_version):
            emitter.context.declarations[PREFIX + fn.cname(emitter.names)] = HeaderDeclaration(f'{wrapper_function_header(fn, emitter.names)};')
        else:
            emitter.context.declarations[PREFIX + fn.cname(emitter.names)] = HeaderDeclaration(f'{legacy_wrapper_function_header(fn, emitter.names)};')

def pointerize(decl: str, name: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Given a C decl and its name, modify it to be a declaration to a pointer.'
    if '(' in decl:
        return decl.replace(name, f'(*{name})')
    else:
        return decl.replace(name, f'*{name}')

def group_dir(group_name: str) -> str:
    if False:
        print('Hello World!')
    'Given a group name, return the relative directory path for it.'
    return os.sep.join(group_name.split('.')[:-1])

class GroupGenerator:

    def __init__(self, modules: dict[str, ModuleIR], source_paths: dict[str, str], group_name: str | None, group_map: dict[str, str | None], names: NameGenerator, compiler_options: CompilerOptions) -> None:
        if False:
            i = 10
            return i + 15
        'Generator for C source for a compilation group.\n\n        The code for a compilation group contains an internal and an\n        external .h file, and then one .c if not in multi_file mode or\n        one .c file per module if in multi_file mode.)\n\n        Arguments:\n            modules: (name, ir) pairs for each module in the group\n            source_paths: Map from module names to source file paths\n            group_name: The name of the group (or None if this is single-module compilation)\n            group_map: A map of modules to their group names\n            names: The name generator for the compilation\n            multi_file: Whether to put each module in its own source file regardless\n                        of group structure.\n        '
        self.modules = modules
        self.source_paths = source_paths
        self.context = EmitterContext(names, group_name, group_map)
        self.names = names
        self.simple_inits: list[tuple[str, str]] = []
        self.group_name = group_name
        self.use_shared_lib = group_name is not None
        self.compiler_options = compiler_options
        self.multi_file = compiler_options.multi_file

    @property
    def group_suffix(self) -> str:
        if False:
            i = 10
            return i + 15
        return '_' + exported_name(self.group_name) if self.group_name else ''

    @property
    def short_group_suffix(self) -> str:
        if False:
            while True:
                i = 10
        return '_' + exported_name(self.group_name.split('.')[-1]) if self.group_name else ''

    def generate_c_for_modules(self) -> list[tuple[str, str]]:
        if False:
            return 10
        file_contents = []
        multi_file = self.use_shared_lib and self.multi_file
        for module in self.modules.values():
            for fn in module.functions:
                collect_literals(fn, self.context.literals)
        base_emitter = Emitter(self.context)
        if self.compiler_options.include_runtime_files:
            for name in RUNTIME_C_FILES:
                base_emitter.emit_line(f'#include "{name}"')
        base_emitter.emit_line(f'#include "__native{self.short_group_suffix}.h"')
        base_emitter.emit_line(f'#include "__native_internal{self.short_group_suffix}.h"')
        emitter = base_emitter
        self.generate_literal_tables()
        for (module_name, module) in self.modules.items():
            if multi_file:
                emitter = Emitter(self.context)
                emitter.emit_line(f'#include "__native{self.short_group_suffix}.h"')
                emitter.emit_line(f'#include "__native_internal{self.short_group_suffix}.h"')
            self.declare_module(module_name, emitter)
            self.declare_internal_globals(module_name, emitter)
            self.declare_imports(module.imports, emitter)
            for cl in module.classes:
                if cl.is_ext_class:
                    generate_class(cl, module_name, emitter)
            self.generate_module_def(emitter, module_name, module)
            for fn in module.functions:
                emitter.emit_line()
                generate_native_function(fn, emitter, self.source_paths[module_name], module_name)
                if fn.name != TOP_LEVEL_NAME:
                    emitter.emit_line()
                    if is_fastcall_supported(fn, emitter.capi_version):
                        generate_wrapper_function(fn, emitter, self.source_paths[module_name], module_name)
                    else:
                        generate_legacy_wrapper_function(fn, emitter, self.source_paths[module_name], module_name)
            if multi_file:
                name = f'__native_{emitter.names.private_name(module_name)}.c'
                file_contents.append((name, ''.join(emitter.fragments)))
        ext_declarations = Emitter(self.context)
        ext_declarations.emit_line(f'#ifndef MYPYC_NATIVE{self.group_suffix}_H')
        ext_declarations.emit_line(f'#define MYPYC_NATIVE{self.group_suffix}_H')
        ext_declarations.emit_line('#include <Python.h>')
        ext_declarations.emit_line('#include <CPy.h>')
        declarations = Emitter(self.context)
        declarations.emit_line(f'#ifndef MYPYC_NATIVE_INTERNAL{self.group_suffix}_H')
        declarations.emit_line(f'#define MYPYC_NATIVE_INTERNAL{self.group_suffix}_H')
        declarations.emit_line('#include <Python.h>')
        declarations.emit_line('#include <CPy.h>')
        declarations.emit_line(f'#include "__native{self.short_group_suffix}.h"')
        declarations.emit_line()
        declarations.emit_line('int CPyGlobalsInit(void);')
        declarations.emit_line()
        for (module_name, module) in self.modules.items():
            self.declare_finals(module_name, module.final_names, declarations)
            for cl in module.classes:
                generate_class_type_decl(cl, emitter, ext_declarations, declarations)
            for fn in module.functions:
                generate_function_declaration(fn, declarations)
        for lib in sorted(self.context.group_deps):
            elib = exported_name(lib)
            short_lib = exported_name(lib.split('.')[-1])
            declarations.emit_lines('#include <{}>'.format(os.path.join(group_dir(lib), f'__native_{short_lib}.h')), f'struct export_table_{elib} exports_{elib};')
        sorted_decls = self.toposort_declarations()
        emitter = base_emitter
        self.generate_globals_init(emitter)
        emitter.emit_line()
        for declaration in sorted_decls:
            decls = ext_declarations if declaration.is_type else declarations
            if not declaration.is_type:
                decls.emit_lines(f'extern {declaration.decl[0]}', *declaration.decl[1:])
                if declaration.defn:
                    emitter.emit_lines(*declaration.defn)
                else:
                    emitter.emit_lines(*declaration.decl)
            else:
                decls.emit_lines(*declaration.decl)
        if self.group_name:
            self.generate_export_table(ext_declarations, emitter)
            self.generate_shared_lib_init(emitter)
        ext_declarations.emit_line('#endif')
        declarations.emit_line('#endif')
        output_dir = group_dir(self.group_name) if self.group_name else ''
        return file_contents + [(os.path.join(output_dir, f'__native{self.short_group_suffix}.c'), ''.join(emitter.fragments)), (os.path.join(output_dir, f'__native_internal{self.short_group_suffix}.h'), ''.join(declarations.fragments)), (os.path.join(output_dir, f'__native{self.short_group_suffix}.h'), ''.join(ext_declarations.fragments))]

    def generate_literal_tables(self) -> None:
        if False:
            i = 10
            return i + 15
        'Generate tables containing descriptions of Python literals to construct.\n\n        We will store the constructed literals in a single array that contains\n        literals of all types. This way we can refer to an arbitrary literal by\n        its index.\n        '
        literals = self.context.literals
        self.declare_global('PyObject *[%d]' % literals.num_literals(), 'CPyStatics')
        init_str = c_string_array_initializer(literals.encoded_str_values())
        self.declare_global('const char * const []', 'CPyLit_Str', initializer=init_str)
        init_bytes = c_string_array_initializer(literals.encoded_bytes_values())
        self.declare_global('const char * const []', 'CPyLit_Bytes', initializer=init_bytes)
        init_int = c_string_array_initializer(literals.encoded_int_values())
        self.declare_global('const char * const []', 'CPyLit_Int', initializer=init_int)
        init_floats = c_array_initializer(literals.encoded_float_values())
        self.declare_global('const double []', 'CPyLit_Float', initializer=init_floats)
        init_complex = c_array_initializer(literals.encoded_complex_values())
        self.declare_global('const double []', 'CPyLit_Complex', initializer=init_complex)
        init_tuple = c_array_initializer(literals.encoded_tuple_values())
        self.declare_global('const int []', 'CPyLit_Tuple', initializer=init_tuple)
        init_frozenset = c_array_initializer(literals.encoded_frozenset_values())
        self.declare_global('const int []', 'CPyLit_FrozenSet', initializer=init_frozenset)

    def generate_export_table(self, decl_emitter: Emitter, code_emitter: Emitter) -> None:
        if False:
            return 10
        'Generate the declaration and definition of the group\'s export struct.\n\n        To avoid needing to deal with deeply platform specific issues\n        involving dynamic library linking (and some possibly\n        insurmountable issues involving cyclic dependencies), compiled\n        code accesses functions and data in other compilation groups\n        via an explicit "export struct".\n\n        Each group declares a struct type that contains a pointer to\n        every function and static variable it exports. It then\n        populates this struct and stores a pointer to it in a capsule\n        stored as an attribute named \'exports\' on the group\'s shared\n        library\'s python module.\n\n        On load, a group\'s init function will import all of its\n        dependencies\' exports tables using the capsule mechanism and\n        copy the contents into a local copy of the table (to eliminate\n        the need for a pointer indirection when accessing it).\n\n        Then, all calls to functions in another group and accesses to statics\n        from another group are done indirectly via the export table.\n\n        For example, a group containing a module b, where b contains a class B\n        and a function bar, would declare an export table like:\n            struct export_table_b {\n                PyTypeObject **CPyType_B;\n                PyObject *(*CPyDef_B)(CPyTagged cpy_r_x);\n                CPyTagged (*CPyDef_B___foo)(PyObject *cpy_r_self, CPyTagged cpy_r_y);\n                tuple_T2OI (*CPyDef_bar)(PyObject *cpy_r_x);\n                char (*CPyDef___top_level__)(void);\n            };\n        that would be initialized with:\n            static struct export_table_b exports = {\n                &CPyType_B,\n                &CPyDef_B,\n                &CPyDef_B___foo,\n                &CPyDef_bar,\n                &CPyDef___top_level__,\n            };\n        To call `b.foo`, then, a function in another group would do\n        `exports_b.CPyDef_bar(...)`.\n        '
        decls = decl_emitter.context.declarations
        decl_emitter.emit_lines('', f'struct export_table{self.group_suffix} {{')
        for (name, decl) in decls.items():
            if decl.needs_export:
                decl_emitter.emit_line(pointerize('\n'.join(decl.decl), name))
        decl_emitter.emit_line('};')
        code_emitter.emit_lines('', f'static struct export_table{self.group_suffix} exports = {{')
        for (name, decl) in decls.items():
            if decl.needs_export:
                code_emitter.emit_line(f'&{name},')
        code_emitter.emit_line('};')

    def generate_shared_lib_init(self, emitter: Emitter) -> None:
        if False:
            return 10
        'Generate the init function for a shared library.\n\n        A shared library contains all of the actual code for a\n        compilation group.\n\n        The init function is responsible for creating Capsules that\n        wrap pointers to the initialization function of all the real\n        init functions for modules in this shared library as well as\n        the export table containing all of the exported functions and\n        values from all the modules.\n\n        These capsules are stored in attributes of the shared library.\n        '
        assert self.group_name is not None
        emitter.emit_line()
        emitter.emit_lines('PyMODINIT_FUNC PyInit_{}(void)'.format(shared_lib_name(self.group_name).split('.')[-1]), '{', 'static PyModuleDef def = {{ PyModuleDef_HEAD_INIT, "{}", NULL, -1, NULL, NULL }};'.format(shared_lib_name(self.group_name)), 'int res;', 'PyObject *capsule;', 'PyObject *tmp;', 'static PyObject *module;', 'if (module) {', 'Py_INCREF(module);', 'return module;', '}', 'module = PyModule_Create(&def);', 'if (!module) {', 'goto fail;', '}', '')
        emitter.emit_lines('capsule = PyCapsule_New(&exports, "{}.exports", NULL);'.format(shared_lib_name(self.group_name)), 'if (!capsule) {', 'goto fail;', '}', 'res = PyObject_SetAttrString(module, "exports", capsule);', 'Py_DECREF(capsule);', 'if (res < 0) {', 'goto fail;', '}', '')
        for mod in self.modules:
            name = exported_name(mod)
            emitter.emit_lines(f'extern PyObject *CPyInit_{name}(void);', 'capsule = PyCapsule_New((void *)CPyInit_{}, "{}.init_{}", NULL);'.format(name, shared_lib_name(self.group_name), name), 'if (!capsule) {', 'goto fail;', '}', f'res = PyObject_SetAttrString(module, "init_{name}", capsule);', 'Py_DECREF(capsule);', 'if (res < 0) {', 'goto fail;', '}', '')
        for group in sorted(self.context.group_deps):
            egroup = exported_name(group)
            emitter.emit_lines('tmp = PyImport_ImportModule("{}"); if (!tmp) goto fail; Py_DECREF(tmp);'.format(shared_lib_name(group)), 'struct export_table_{} *pexports_{} = PyCapsule_Import("{}.exports", 0);'.format(egroup, egroup, shared_lib_name(group)), f'if (!pexports_{egroup}) {{', 'goto fail;', '}', 'memcpy(&exports_{group}, pexports_{group}, sizeof(exports_{group}));'.format(group=egroup), '')
        emitter.emit_lines('return module;', 'fail:', 'Py_XDECREF(module);', 'return NULL;', '}')

    def generate_globals_init(self, emitter: Emitter) -> None:
        if False:
            for i in range(10):
                print('nop')
        emitter.emit_lines('', 'int CPyGlobalsInit(void)', '{', 'static int is_initialized = 0;', 'if (is_initialized) return 0;', '')
        emitter.emit_line('CPy_Init();')
        for (symbol, fixup) in self.simple_inits:
            emitter.emit_line(f'{symbol} = {fixup};')
        values = 'CPyLit_Str, CPyLit_Bytes, CPyLit_Int, CPyLit_Float, CPyLit_Complex, CPyLit_Tuple, CPyLit_FrozenSet'
        emitter.emit_lines(f'if (CPyStatics_Initialize(CPyStatics, {values}) < 0) {{', 'return -1;', '}')
        emitter.emit_lines('is_initialized = 1;', 'return 0;', '}')

    def generate_module_def(self, emitter: Emitter, module_name: str, module: ModuleIR) -> None:
        if False:
            i = 10
            return i + 15
        'Emit the PyModuleDef struct for a module and the module init function.'
        module_prefix = emitter.names.private_name(module_name)
        emitter.emit_line(f'static PyMethodDef {module_prefix}module_methods[] = {{')
        for fn in module.functions:
            if fn.class_name is not None or fn.name == TOP_LEVEL_NAME:
                continue
            name = short_id_from_name(fn.name, fn.decl.shortname, fn.line)
            if is_fastcall_supported(fn, emitter.capi_version):
                flag = 'METH_FASTCALL'
            else:
                flag = 'METH_VARARGS'
            emitter.emit_line('{{"{name}", (PyCFunction){prefix}{cname}, {flag} | METH_KEYWORDS, NULL /* docstring */}},'.format(name=name, cname=fn.cname(emitter.names), prefix=PREFIX, flag=flag))
        emitter.emit_line('{NULL, NULL, 0, NULL}')
        emitter.emit_line('};')
        emitter.emit_line()
        emitter.emit_lines(f'static struct PyModuleDef {module_prefix}module = {{', 'PyModuleDef_HEAD_INIT,', f'"{module_name}",', 'NULL, /* docstring */', '-1,       /* size of per-interpreter state of the module,', '             or -1 if the module keeps state in global variables. */', f'{module_prefix}module_methods', '};')
        emitter.emit_line()
        if not self.use_shared_lib:
            declaration = f'PyMODINIT_FUNC PyInit_{module_name}(void)'
        else:
            declaration = f'PyObject *CPyInit_{exported_name(module_name)}(void)'
        emitter.emit_lines(declaration, '{')
        emitter.emit_line('PyObject* modname = NULL;')
        module_static = self.module_internal_static_name(module_name, emitter)
        emitter.emit_lines(f'if ({module_static}) {{', f'Py_INCREF({module_static});', f'return {module_static};', '}')
        emitter.emit_lines(f'{module_static} = PyModule_Create(&{module_prefix}module);', f'if (unlikely({module_static} == NULL))', '    goto fail;')
        emitter.emit_line(f'modname = PyObject_GetAttrString((PyObject *){module_static}, "__name__");')
        module_globals = emitter.static_name('globals', module_name)
        emitter.emit_lines(f'{module_globals} = PyModule_GetDict({module_static});', f'if (unlikely({module_globals} == NULL))', '    goto fail;')
        type_structs: list[str] = []
        for cl in module.classes:
            type_struct = emitter.type_struct_name(cl)
            type_structs.append(type_struct)
            if cl.is_generated:
                emitter.emit_lines('{t} = (PyTypeObject *)CPyType_FromTemplate((PyObject *){t}_template, NULL, modname);'.format(t=type_struct))
                emitter.emit_lines(f'if (unlikely(!{type_struct}))', '    goto fail;')
        emitter.emit_lines('if (CPyGlobalsInit() < 0)', '    goto fail;')
        self.generate_top_level_call(module, emitter)
        emitter.emit_lines('Py_DECREF(modname);')
        emitter.emit_line(f'return {module_static};')
        emitter.emit_lines('fail:', f'Py_CLEAR({module_static});', 'Py_CLEAR(modname);')
        for (name, typ) in module.final_names:
            static_name = emitter.static_name(name, module_name)
            emitter.emit_dec_ref(static_name, typ, is_xdec=True)
            undef = emitter.c_undefined_value(typ)
            emitter.emit_line(f'{static_name} = {undef};')
        for t in type_structs:
            emitter.emit_line(f'Py_CLEAR({t});')
        emitter.emit_line('return NULL;')
        emitter.emit_line('}')

    def generate_top_level_call(self, module: ModuleIR, emitter: Emitter) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Generate call to function representing module top level.'
        for fn in reversed(module.functions):
            if fn.name == TOP_LEVEL_NAME:
                emitter.emit_lines(f'char result = {emitter.native_function_name(fn.decl)}();', 'if (result == 2)', '    goto fail;')
                break

    def toposort_declarations(self) -> list[HeaderDeclaration]:
        if False:
            print('Hello World!')
        'Topologically sort the declaration dict by dependencies.\n\n        Declarations can require other declarations to come prior in C (such as declaring structs).\n        In order to guarantee that the C output will compile the declarations will thus need to\n        be properly ordered. This simple DFS guarantees that we have a proper ordering.\n\n        This runs in O(V + E).\n        '
        result = []
        marked_declarations: dict[str, MarkedDeclaration] = {}
        for (k, v) in self.context.declarations.items():
            marked_declarations[k] = MarkedDeclaration(v, False)

        def _toposort_visit(name: str) -> None:
            if False:
                return 10
            decl = marked_declarations[name]
            if decl.mark:
                return
            for child in decl.declaration.dependencies:
                _toposort_visit(child)
            result.append(decl.declaration)
            decl.mark = True
        for (name, marked_declaration) in marked_declarations.items():
            _toposort_visit(name)
        return result

    def declare_global(self, type_spaced: str, name: str, *, initializer: str | None=None) -> None:
        if False:
            i = 10
            return i + 15
        if '[' not in type_spaced:
            base = f'{type_spaced}{name}'
        else:
            (a, b) = type_spaced.split('[', 1)
            base = f'{a}{name}[{b}'
        if not initializer:
            defn = None
        else:
            defn = [f'{base} = {initializer};']
        if name not in self.context.declarations:
            self.context.declarations[name] = HeaderDeclaration(f'{base};', defn=defn)

    def declare_internal_globals(self, module_name: str, emitter: Emitter) -> None:
        if False:
            return 10
        static_name = emitter.static_name('globals', module_name)
        self.declare_global('PyObject *', static_name)

    def module_internal_static_name(self, module_name: str, emitter: Emitter) -> str:
        if False:
            return 10
        return emitter.static_name(module_name + '_internal', None, prefix=MODULE_PREFIX)

    def declare_module(self, module_name: str, emitter: Emitter) -> None:
        if False:
            for i in range(10):
                print('nop')
        if module_name in self.modules:
            internal_static_name = self.module_internal_static_name(module_name, emitter)
            self.declare_global('CPyModule *', internal_static_name, initializer='NULL')
        static_name = emitter.static_name(module_name, None, prefix=MODULE_PREFIX)
        self.declare_global('CPyModule *', static_name)
        self.simple_inits.append((static_name, 'Py_None'))

    def declare_imports(self, imps: Iterable[str], emitter: Emitter) -> None:
        if False:
            while True:
                i = 10
        for imp in imps:
            self.declare_module(imp, emitter)

    def declare_finals(self, module: str, final_names: Iterable[tuple[str, RType]], emitter: Emitter) -> None:
        if False:
            for i in range(10):
                print('nop')
        for (name, typ) in final_names:
            static_name = emitter.static_name(name, module)
            emitter.context.declarations[static_name] = HeaderDeclaration(f'{emitter.ctype_spaced(typ)}{static_name};', [self.final_definition(module, name, typ, emitter)], needs_export=True)

    def final_definition(self, module: str, name: str, typ: RType, emitter: Emitter) -> str:
        if False:
            return 10
        static_name = emitter.static_name(name, module)
        undefined = emitter.c_initializer_undefined_value(typ)
        return f'{emitter.ctype_spaced(typ)}{static_name} = {undefined};'

    def declare_static_pyobject(self, identifier: str, emitter: Emitter) -> None:
        if False:
            print('Hello World!')
        symbol = emitter.static_name(identifier, None)
        self.declare_global('PyObject *', symbol)

def sort_classes(classes: list[tuple[str, ClassIR]]) -> list[tuple[str, ClassIR]]:
    if False:
        i = 10
        return i + 15
    mod_name = {ir: name for (name, ir) in classes}
    irs = [ir for (_, ir) in classes]
    deps: dict[ClassIR, set[ClassIR]] = {}
    for ir in irs:
        if ir not in deps:
            deps[ir] = set()
        if ir.base:
            deps[ir].add(ir.base)
        deps[ir].update(ir.traits)
    sorted_irs = toposort(deps)
    return [(mod_name[ir], ir) for ir in sorted_irs]
T = TypeVar('T')

def toposort(deps: dict[T, set[T]]) -> list[T]:
    if False:
        for i in range(10):
            print('nop')
    'Topologically sort a dict from item to dependencies.\n\n    This runs in O(V + E).\n    '
    result = []
    visited: set[T] = set()

    def visit(item: T) -> None:
        if False:
            while True:
                i = 10
        if item in visited:
            return
        for child in deps[item]:
            visit(child)
        result.append(item)
        visited.add(item)
    for item in deps:
        visit(item)
    return result

def is_fastcall_supported(fn: FuncIR, capi_version: tuple[int, int]) -> bool:
    if False:
        while True:
            i = 10
    if fn.class_name is not None:
        if fn.name == '__call__':
            return use_vectorcall(capi_version)
        return fn.name != '__init__'
    return True

def collect_literals(fn: FuncIR, literals: Literals) -> None:
    if False:
        i = 10
        return i + 15
    "Store all Python literal object refs in fn.\n\n    Collecting literals must happen only after we have the final IR.\n    This way we won't include literals that have been optimized away.\n    "
    for block in fn.blocks:
        for op in block.ops:
            if isinstance(op, LoadLiteral):
                literals.record_literal(op.value)

def c_string_array_initializer(components: list[bytes]) -> str:
    if False:
        for i in range(10):
            print('nop')
    result = []
    result.append('{\n')
    for s in components:
        result.append('    ' + c_string_initializer(s) + ',\n')
    result.append('}')
    return ''.join(result)