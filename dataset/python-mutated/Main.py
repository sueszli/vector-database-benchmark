from __future__ import absolute_import, print_function
import os
import re
import sys
import io
if sys.version_info[:2] < (2, 7) or (3, 0) <= sys.version_info[:2] < (3, 3):
    sys.stderr.write('Sorry, Cython requires Python 2.7 or 3.3+, found %d.%d\n' % tuple(sys.version_info[:2]))
    sys.exit(1)
try:
    from __builtin__ import basestring
except ImportError:
    basestring = str
from . import Errors
from .StringEncoding import EncodedString
from .Scanning import PyrexScanner, FileSourceDescriptor
from .Errors import PyrexError, CompileError, error, warning
from .Symtab import ModuleScope
from .. import Utils
from . import Options
from .Options import CompilationOptions, default_options
from .CmdLine import parse_command_line
from .Lexicon import unicode_start_ch_any, unicode_continuation_ch_any, unicode_start_ch_range, unicode_continuation_ch_range

def _make_range_re(chrs):
    if False:
        print('Hello World!')
    out = []
    for i in range(0, len(chrs), 2):
        out.append(u'{0}-{1}'.format(chrs[i], chrs[i + 1]))
    return u''.join(out)
module_name_pattern = u'[{0}{1}][{0}{2}{1}{3}]*'.format(unicode_start_ch_any, _make_range_re(unicode_start_ch_range), unicode_continuation_ch_any, _make_range_re(unicode_continuation_ch_range))
module_name_pattern = re.compile(u'{0}(\\.{0})*$'.format(module_name_pattern))
standard_include_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Includes'))

class Context(object):
    cython_scope = None
    language_level = None

    def __init__(self, include_directories, compiler_directives, cpp=False, language_level=None, options=None):
        if False:
            i = 10
            return i + 15
        from . import Builtin, CythonScope
        self.modules = {'__builtin__': Builtin.builtin_scope}
        self.cython_scope = CythonScope.create_cython_scope(self)
        self.modules['cython'] = self.cython_scope
        self.include_directories = include_directories
        self.future_directives = set()
        self.compiler_directives = compiler_directives
        self.cpp = cpp
        self.options = options
        self.pxds = {}
        self._interned = {}
        if language_level is not None:
            self.set_language_level(language_level)
        self.legacy_implicit_noexcept = self.compiler_directives.get('legacy_implicit_noexcept', False)
        self.gdb_debug_outputwriter = None

    @classmethod
    def from_options(cls, options):
        if False:
            return 10
        return cls(options.include_path, options.compiler_directives, options.cplus, options.language_level, options=options)

    def set_language_level(self, level):
        if False:
            return 10
        from .Future import print_function, unicode_literals, absolute_import, division, generator_stop
        future_directives = set()
        if level == '3str':
            level = 3
        else:
            level = int(level)
            if level >= 3:
                future_directives.add(unicode_literals)
        if level >= 3:
            future_directives.update([print_function, absolute_import, division, generator_stop])
        self.language_level = level
        self.future_directives = future_directives
        if level >= 3:
            self.modules['builtins'] = self.modules['__builtin__']

    def intern_ustring(self, value, encoding=None):
        if False:
            return 10
        key = (EncodedString, value, encoding)
        try:
            return self._interned[key]
        except KeyError:
            pass
        value = EncodedString(value)
        if encoding:
            value.encoding = encoding
        self._interned[key] = value
        return value

    def process_pxd(self, source_desc, scope, module_name):
        if False:
            while True:
                i = 10
        from . import Pipeline
        if isinstance(source_desc, FileSourceDescriptor) and source_desc._file_type == 'pyx':
            source = CompilationSource(source_desc, module_name, os.getcwd())
            result_sink = create_default_resultobj(source, self.options)
            pipeline = Pipeline.create_pyx_as_pxd_pipeline(self, result_sink)
            result = Pipeline.run_pipeline(pipeline, source)
        else:
            pipeline = Pipeline.create_pxd_pipeline(self, scope, module_name)
            result = Pipeline.run_pipeline(pipeline, source_desc)
        return result

    def nonfatal_error(self, exc):
        if False:
            for i in range(10):
                print('nop')
        return Errors.report_error(exc)

    def _split_qualified_name(self, qualified_name, relative_import=False):
        if False:
            return 10
        qualified_name_parts = qualified_name.split('.')
        last_part = qualified_name_parts.pop()
        qualified_name_parts = [(p, True) for p in qualified_name_parts]
        if last_part != '__init__':
            is_package = False
            for suffix in ('.py', '.pyx'):
                path = self.search_include_directories(qualified_name, suffix=suffix, source_pos=None, source_file_path=None, sys_path=not relative_import)
                if path:
                    is_package = self._is_init_file(path)
                    break
            qualified_name_parts.append((last_part, is_package))
        return qualified_name_parts

    @staticmethod
    def _is_init_file(path):
        if False:
            return 10
        return os.path.basename(path) in ('__init__.pyx', '__init__.py', '__init__.pxd') if path else False

    @staticmethod
    def _check_pxd_filename(pos, pxd_pathname, qualified_name):
        if False:
            print('Hello World!')
        if not pxd_pathname:
            return
        pxd_filename = os.path.basename(pxd_pathname)
        if '.' in qualified_name and qualified_name == os.path.splitext(pxd_filename)[0]:
            warning(pos, "Dotted filenames ('%s') are deprecated. Please use the normal Python package directory layout." % pxd_filename, level=1)

    def find_module(self, module_name, from_module=None, pos=None, need_pxd=1, absolute_fallback=True, relative_import=False):
        if False:
            for i in range(10):
                print('nop')
        debug_find_module = 0
        if debug_find_module:
            print('Context.find_module: module_name = %s, from_module = %s, pos = %s, need_pxd = %s' % (module_name, from_module, pos, need_pxd))
        scope = None
        pxd_pathname = None
        if from_module:
            if module_name:
                qualified_name = from_module.qualify_name(module_name)
            else:
                qualified_name = from_module.qualified_name
                scope = from_module
                from_module = None
        else:
            qualified_name = module_name
        if not module_name_pattern.match(qualified_name):
            raise CompileError(pos or (module_name, 0, 0), u"'%s' is not a valid module name" % module_name)
        if from_module:
            if debug_find_module:
                print('...trying relative import')
            scope = from_module.lookup_submodule(module_name)
            if not scope:
                pxd_pathname = self.find_pxd_file(qualified_name, pos, sys_path=not relative_import)
                self._check_pxd_filename(pos, pxd_pathname, qualified_name)
                if pxd_pathname:
                    is_package = self._is_init_file(pxd_pathname)
                    scope = from_module.find_submodule(module_name, as_package=is_package)
        if not scope:
            if debug_find_module:
                print('...trying absolute import')
            if absolute_fallback:
                qualified_name = module_name
            scope = self
            for (name, is_package) in self._split_qualified_name(qualified_name, relative_import=relative_import):
                scope = scope.find_submodule(name, as_package=is_package)
        if debug_find_module:
            print('...scope = %s' % scope)
        if not scope.pxd_file_loaded:
            if debug_find_module:
                print('...pxd not loaded')
            if not pxd_pathname:
                if debug_find_module:
                    print('...looking for pxd file')
                pxd_pathname = self.find_pxd_file(qualified_name, pos, sys_path=need_pxd and (not relative_import))
                self._check_pxd_filename(pos, pxd_pathname, qualified_name)
                if debug_find_module:
                    print('......found %s' % pxd_pathname)
                if not pxd_pathname and need_pxd:
                    scope.pxd_file_loaded = True
                    package_pathname = self.search_include_directories(qualified_name, suffix='.py', source_pos=pos, sys_path=not relative_import)
                    if package_pathname and package_pathname.endswith(Utils.PACKAGE_FILES):
                        pass
                    else:
                        error(pos, "'%s.pxd' not found" % qualified_name.replace('.', os.sep))
            if pxd_pathname:
                scope.pxd_file_loaded = True
                try:
                    if debug_find_module:
                        print('Context.find_module: Parsing %s' % pxd_pathname)
                    rel_path = module_name.replace('.', os.sep) + os.path.splitext(pxd_pathname)[1]
                    if not pxd_pathname.endswith(rel_path):
                        rel_path = pxd_pathname
                    source_desc = FileSourceDescriptor(pxd_pathname, rel_path)
                    (err, result) = self.process_pxd(source_desc, scope, qualified_name)
                    if err:
                        raise err
                    (pxd_codenodes, pxd_scope) = result
                    self.pxds[module_name] = (pxd_codenodes, pxd_scope)
                except CompileError:
                    pass
        return scope

    def find_pxd_file(self, qualified_name, pos=None, sys_path=True, source_file_path=None):
        if False:
            for i in range(10):
                print('nop')
        pxd = self.search_include_directories(qualified_name, suffix='.pxd', source_pos=pos, sys_path=sys_path, source_file_path=source_file_path)
        if pxd is None and Options.cimport_from_pyx:
            return self.find_pyx_file(qualified_name, pos, sys_path=sys_path)
        return pxd

    def find_pyx_file(self, qualified_name, pos=None, sys_path=True, source_file_path=None):
        if False:
            i = 10
            return i + 15
        return self.search_include_directories(qualified_name, suffix='.pyx', source_pos=pos, sys_path=sys_path, source_file_path=source_file_path)

    def find_include_file(self, filename, pos=None, source_file_path=None):
        if False:
            return 10
        path = self.search_include_directories(filename, source_pos=pos, include=True, source_file_path=source_file_path)
        if not path:
            error(pos, "'%s' not found" % filename)
        return path

    def search_include_directories(self, qualified_name, suffix=None, source_pos=None, include=False, sys_path=False, source_file_path=None):
        if False:
            for i in range(10):
                print('nop')
        include_dirs = self.include_directories
        if sys_path:
            include_dirs = include_dirs + sys.path
        include_dirs = tuple(include_dirs + [standard_include_path])
        return search_include_directories(include_dirs, qualified_name, suffix or '', source_pos, include, source_file_path)

    def find_root_package_dir(self, file_path):
        if False:
            for i in range(10):
                print('nop')
        return Utils.find_root_package_dir(file_path)

    def check_package_dir(self, dir, package_names):
        if False:
            print('Hello World!')
        return Utils.check_package_dir(dir, tuple(package_names))

    def c_file_out_of_date(self, source_path, output_path):
        if False:
            for i in range(10):
                print('nop')
        if not os.path.exists(output_path):
            return 1
        c_time = Utils.modification_time(output_path)
        if Utils.file_newer_than(source_path, c_time):
            return 1
        pxd_path = Utils.replace_suffix(source_path, '.pxd')
        if os.path.exists(pxd_path) and Utils.file_newer_than(pxd_path, c_time):
            return 1
        for (kind, name) in self.read_dependency_file(source_path):
            if kind == 'cimport':
                dep_path = self.find_pxd_file(name, source_file_path=source_path)
            elif kind == 'include':
                dep_path = self.search_include_directories(name, source_file_path=source_path)
            else:
                continue
            if dep_path and Utils.file_newer_than(dep_path, c_time):
                return 1
        return 0

    def find_cimported_module_names(self, source_path):
        if False:
            return 10
        return [name for (kind, name) in self.read_dependency_file(source_path) if kind == 'cimport']

    def is_package_dir(self, dir_path):
        if False:
            while True:
                i = 10
        return Utils.is_package_dir(dir_path)

    def read_dependency_file(self, source_path):
        if False:
            while True:
                i = 10
        dep_path = Utils.replace_suffix(source_path, '.dep')
        if os.path.exists(dep_path):
            with open(dep_path, 'rU') as f:
                chunks = [line.split(' ', 1) for line in (l.strip() for l in f) if ' ' in line]
            return chunks
        else:
            return ()

    def lookup_submodule(self, name):
        if False:
            i = 10
            return i + 15
        return self.modules.get(name, None)

    def find_submodule(self, name, as_package=False):
        if False:
            for i in range(10):
                print('nop')
        scope = self.lookup_submodule(name)
        if not scope:
            scope = ModuleScope(name, parent_module=None, context=self, is_package=as_package)
            self.modules[name] = scope
        return scope

    def parse(self, source_desc, scope, pxd, full_module_name):
        if False:
            while True:
                i = 10
        if not isinstance(source_desc, FileSourceDescriptor):
            raise RuntimeError('Only file sources for code supported')
        source_filename = source_desc.filename
        scope.cpp = self.cpp
        num_errors = Errors.get_errors_count()
        try:
            with Utils.open_source_file(source_filename) as f:
                from . import Parsing
                s = PyrexScanner(f, source_desc, source_encoding=f.encoding, scope=scope, context=self)
                tree = Parsing.p_module(s, pxd, full_module_name)
                if self.options.formal_grammar:
                    try:
                        from ..Parser import ConcreteSyntaxTree
                    except ImportError:
                        raise RuntimeError('Formal grammar can only be used with compiled Cython with an available pgen.')
                    ConcreteSyntaxTree.p_module(source_filename)
        except UnicodeDecodeError as e:
            raise self._report_decode_error(source_desc, e)
        if Errors.get_errors_count() > num_errors:
            raise CompileError()
        return tree

    def _report_decode_error(self, source_desc, exc):
        if False:
            print('Hello World!')
        msg = exc.args[-1]
        position = exc.args[2]
        encoding = exc.args[0]
        line = 1
        column = idx = 0
        with io.open(source_desc.filename, 'r', encoding='iso8859-1', newline='') as f:
            for (line, data) in enumerate(f, 1):
                idx += len(data)
                if idx >= position:
                    column = position - (idx - len(data)) + 1
                    break
        return error((source_desc, line, column), 'Decoding error, missing or incorrect coding=<encoding-name> at top of source (cannot decode with encoding %r: %s)' % (encoding, msg))

    def extract_module_name(self, path, options):
        if False:
            i = 10
            return i + 15
        (dir, filename) = os.path.split(path)
        (module_name, _) = os.path.splitext(filename)
        if '.' in module_name:
            return module_name
        names = [module_name]
        while self.is_package_dir(dir):
            (parent, package_name) = os.path.split(dir)
            if parent == dir:
                break
            names.append(package_name)
            dir = parent
        names.reverse()
        return '.'.join(names)

    def setup_errors(self, options, result):
        if False:
            i = 10
            return i + 15
        Errors.init_thread()
        if options.use_listing_file:
            path = result.listing_file = Utils.replace_suffix(result.main_source_file, '.lis')
        else:
            path = None
        Errors.open_listing_file(path=path, echo_to_stderr=options.errors_to_stderr)

    def teardown_errors(self, err, options, result):
        if False:
            i = 10
            return i + 15
        source_desc = result.compilation_source.source_desc
        if not isinstance(source_desc, FileSourceDescriptor):
            raise RuntimeError('Only file sources for code supported')
        Errors.close_listing_file()
        result.num_errors = Errors.get_errors_count()
        if result.num_errors > 0:
            err = True
        if err and result.c_file:
            try:
                Utils.castrate_file(result.c_file, os.stat(source_desc.filename))
            except EnvironmentError:
                pass
            result.c_file = None

def get_output_filename(source_filename, cwd, options):
    if False:
        while True:
            i = 10
    if options.cplus:
        c_suffix = '.cpp'
    else:
        c_suffix = '.c'
    suggested_file_name = Utils.replace_suffix(source_filename, c_suffix)
    if options.output_file:
        out_path = os.path.join(cwd, options.output_file)
        if os.path.isdir(out_path):
            return os.path.join(out_path, os.path.basename(suggested_file_name))
        else:
            return out_path
    else:
        return suggested_file_name

def create_default_resultobj(compilation_source, options):
    if False:
        for i in range(10):
            print('nop')
    result = CompilationResult()
    result.main_source_file = compilation_source.source_desc.filename
    result.compilation_source = compilation_source
    source_desc = compilation_source.source_desc
    result.c_file = get_output_filename(source_desc.filename, compilation_source.cwd, options)
    result.embedded_metadata = options.embedded_metadata
    return result

def run_pipeline(source, options, full_module_name=None, context=None):
    if False:
        for i in range(10):
            print('nop')
    from . import Pipeline
    if sys.version_info[0] == 2:
        source = Utils.decode_filename(source)
        if full_module_name:
            full_module_name = Utils.decode_filename(full_module_name)
    source_ext = os.path.splitext(source)[1]
    options.configure_language_defaults(source_ext[1:])
    if context is None:
        context = Context.from_options(options)
    cwd = os.getcwd()
    abs_path = os.path.abspath(source)
    full_module_name = full_module_name or context.extract_module_name(source, options)
    full_module_name = EncodedString(full_module_name)
    Utils.raise_error_if_module_name_forbidden(full_module_name)
    if options.relative_path_in_code_position_comments:
        rel_path = full_module_name.replace('.', os.sep) + source_ext
        if not abs_path.endswith(rel_path):
            rel_path = source
    else:
        rel_path = abs_path
    source_desc = FileSourceDescriptor(abs_path, rel_path)
    source = CompilationSource(source_desc, full_module_name, cwd)
    result = create_default_resultobj(source, options)
    if options.annotate is None:
        html_filename = os.path.splitext(result.c_file)[0] + '.html'
        if os.path.exists(html_filename):
            with io.open(html_filename, 'r', encoding='UTF-8') as html_file:
                if u'<!-- Generated by Cython' in html_file.read(100):
                    options.annotate = True
    if source_ext.lower() == '.py' or not source_ext:
        pipeline = Pipeline.create_py_pipeline(context, options, result)
    else:
        pipeline = Pipeline.create_pyx_pipeline(context, options, result)
    context.setup_errors(options, result)
    if '.' in full_module_name and '.' in os.path.splitext(os.path.basename(abs_path))[0]:
        warning((source_desc, 1, 0), "Dotted filenames ('%s') are deprecated. Please use the normal Python package directory layout." % os.path.basename(abs_path), level=1)
    (err, enddata) = Pipeline.run_pipeline(pipeline, source)
    context.teardown_errors(err, options, result)
    if err is None and options.depfile:
        from ..Build.Dependencies import create_dependency_tree
        dependencies = create_dependency_tree(context).all_dependencies(result.main_source_file)
        Utils.write_depfile(result.c_file, result.main_source_file, dependencies)
    return result

class CompilationSource(object):
    """
    Contains the data necessary to start up a compilation pipeline for
    a single compilation unit.
    """

    def __init__(self, source_desc, full_module_name, cwd):
        if False:
            for i in range(10):
                print('nop')
        self.source_desc = source_desc
        self.full_module_name = full_module_name
        self.cwd = cwd

class CompilationResult(object):
    """
    Results from the Cython compiler:

    c_file           string or None   The generated C source file
    h_file           string or None   The generated C header file
    i_file           string or None   The generated .pxi file
    api_file         string or None   The generated C API .h file
    listing_file     string or None   File of error messages
    object_file      string or None   Result of compiling the C file
    extension_file   string or None   Result of linking the object file
    num_errors       integer          Number of compilation errors
    compilation_source CompilationSource
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self.c_file = None
        self.h_file = None
        self.i_file = None
        self.api_file = None
        self.listing_file = None
        self.object_file = None
        self.extension_file = None
        self.main_source_file = None

class CompilationResultSet(dict):
    """
    Results from compiling multiple Pyrex source files. A mapping
    from source file paths to CompilationResult instances. Also
    has the following attributes:

    num_errors   integer   Total number of compilation errors
    """
    num_errors = 0

    def add(self, source, result):
        if False:
            while True:
                i = 10
        self[source] = result
        self.num_errors += result.num_errors

def compile_single(source, options, full_module_name=None):
    if False:
        i = 10
        return i + 15
    '\n    compile_single(source, options, full_module_name)\n\n    Compile the given Pyrex implementation file and return a CompilationResult.\n    Always compiles a single file; does not perform timestamp checking or\n    recursion.\n    '
    return run_pipeline(source, options, full_module_name)

def compile_multiple(sources, options):
    if False:
        i = 10
        return i + 15
    '\n    compile_multiple(sources, options)\n\n    Compiles the given sequence of Pyrex implementation files and returns\n    a CompilationResultSet. Performs timestamp checking and/or recursion\n    if these are specified in the options.\n    '
    if len(sources) > 1 and options.module_name:
        raise RuntimeError('Full module name can only be set for single source compilation')
    sources = [os.path.abspath(source) for source in sources]
    processed = set()
    results = CompilationResultSet()
    timestamps = options.timestamps
    verbose = options.verbose
    context = None
    cwd = os.getcwd()
    for source in sources:
        if source not in processed:
            if context is None:
                context = Context.from_options(options)
            output_filename = get_output_filename(source, cwd, options)
            out_of_date = context.c_file_out_of_date(source, output_filename)
            if not timestamps or out_of_date:
                if verbose:
                    sys.stderr.write('Compiling %s\n' % source)
                result = run_pipeline(source, options, full_module_name=options.module_name, context=context)
                results.add(source, result)
                context = None
            processed.add(source)
    return results

def compile(source, options=None, full_module_name=None, **kwds):
    if False:
        return 10
    '\n    compile(source [, options], [, <option> = <value>]...)\n\n    Compile one or more Pyrex implementation files, with optional timestamp\n    checking and recursing on dependencies.  The source argument may be a string\n    or a sequence of strings.  If it is a string and no recursion or timestamp\n    checking is requested, a CompilationResult is returned, otherwise a\n    CompilationResultSet is returned.\n    '
    options = CompilationOptions(defaults=options, **kwds)
    if isinstance(source, basestring):
        if not options.timestamps:
            return compile_single(source, options, full_module_name)
        source = [source]
    return compile_multiple(source, options)

@Utils.cached_function
def search_include_directories(dirs, qualified_name, suffix='', pos=None, include=False, source_file_path=None):
    if False:
        while True:
            i = 10
    "\n    Search the list of include directories for the given file name.\n\n    If a source file path or position is given, first searches the directory\n    containing that file.  Returns None if not found, but does not report an error.\n\n    The 'include' option will disable package dereferencing.\n    "
    if pos and (not source_file_path):
        file_desc = pos[0]
        if not isinstance(file_desc, FileSourceDescriptor):
            raise RuntimeError('Only file sources for code supported')
        source_file_path = file_desc.filename
    if source_file_path:
        if include:
            dirs = (os.path.dirname(source_file_path),) + dirs
        else:
            dirs = (Utils.find_root_package_dir(source_file_path),) + dirs
    dotted_filename = qualified_name
    if suffix:
        dotted_filename += suffix
    for dirname in dirs:
        path = os.path.join(dirname, dotted_filename)
        if os.path.exists(path):
            return path
    if not include:
        names = qualified_name.split('.')
        package_names = tuple(names[:-1])
        module_name = names[-1]
        namespace_dirs = []
        for dirname in dirs:
            (package_dir, is_namespace) = Utils.check_package_dir(dirname, package_names)
            if package_dir is not None:
                if is_namespace:
                    namespace_dirs.append(package_dir)
                    continue
                path = search_module_in_dir(package_dir, module_name, suffix)
                if path:
                    return path
        for package_dir in namespace_dirs:
            path = search_module_in_dir(package_dir, module_name, suffix)
            if path:
                return path
    return None

@Utils.cached_function
def search_module_in_dir(package_dir, module_name, suffix):
    if False:
        i = 10
        return i + 15
    path = Utils.find_versioned_file(package_dir, module_name, suffix)
    if not path and suffix:
        path = Utils.find_versioned_file(os.path.join(package_dir, module_name), '__init__', suffix)
    return path

def setuptools_main():
    if False:
        for i in range(10):
            print('nop')
    return main(command_line=1)

def main(command_line=0):
    if False:
        i = 10
        return i + 15
    args = sys.argv[1:]
    any_failures = 0
    if command_line:
        try:
            (options, sources) = parse_command_line(args)
        except IOError as e:
            import errno
            if errno.ENOENT != e.errno:
                raise
            print("{}: No such file or directory: '{}'".format(sys.argv[0], e.filename), file=sys.stderr)
            sys.exit(1)
    else:
        options = CompilationOptions(default_options)
        sources = args
    if options.show_version:
        Utils.print_version()
    if options.working_path != '':
        os.chdir(options.working_path)
    try:
        result = compile(sources, options)
        if result.num_errors > 0:
            any_failures = 1
    except (EnvironmentError, PyrexError) as e:
        sys.stderr.write(str(e) + '\n')
        any_failures = 1
    if any_failures:
        sys.exit(1)