"""Use pytype to analyze and infer types for an entire project."""
import collections
import importlib
import itertools
import logging
import subprocess
import sys
from typing import Iterable, Sequence, Tuple
from pytype import file_utils
from pytype import module_utils
from pytype import utils
from pytype.platform_utils import path_utils
from pytype.tools.analyze_project import config
DEFAULT_PYI = '\nfrom typing import Any\ndef __getattr__(name) -> Any: ...\n'

class Action:
    CHECK = 'check'
    INFER = 'infer'
    GENERATE_DEFAULT = 'generate default'

class Stage:
    SINGLE_PASS = 'single pass'
    FIRST_PASS = 'first pass'
    SECOND_PASS = 'second pass'
FIRST_PASS_SUFFIX = '-1'

def _get_executable(binary, module=None):
    if False:
        for i in range(10):
            print('nop')
    'Get the path to the executable with the given name.'
    if binary == 'pytype-single':
        custom_bin = path_utils.join('out', 'bin', 'pytype')
        if sys.argv[0] == custom_bin:
            return ([] if sys.platform != 'win32' else [sys.executable]) + [path_utils.join(path_utils.abspath(path_utils.dirname(custom_bin)), 'pytype-single')]
    importable = importlib.util.find_spec(module or binary)
    if sys.executable is not None and importable:
        return [sys.executable, '-m', module or binary]
    else:
        return [binary]
PYTYPE_SINGLE = _get_executable('pytype-single', 'pytype.single')

def resolved_file_to_module(f):
    if False:
        print('Hello World!')
    'Turn an importlab ResolvedFile into a pytype Module.'
    full_path = f.path
    target = f.short_path
    path = full_path[:-len(target)]
    name = f.module_name
    if path_utils.basename(full_path) == '__init__.py':
        name += '.__init__'
    return module_utils.Module(path=path, target=target, name=name, kind=f.__class__.__name__)

def _get_filenames(node):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(node, str):
        return (node,)
    else:
        return tuple(sorted(node.nodes))

def deps_from_import_graph(import_graph):
    if False:
        return 10
    'Construct PytypeRunner args from an importlab.ImportGraph instance.\n\n  Kept as a separate function so PytypeRunner can be tested independently of\n  importlab.\n\n  Args:\n    import_graph: An importlab.ImportGraph instance.\n\n  Returns:\n    List of (tuple of source modules, tuple of direct deps) in dependency order.\n  '

    def make_module(filename):
        if False:
            return 10
        return resolved_file_to_module(import_graph.provenance[filename])

    def split_files(filenames):
        if False:
            i = 10
            return i + 15
        stubs = []
        sources = []
        for f in filenames:
            if _is_type_stub(f):
                stubs.append(f)
            else:
                sources.append(make_module(f))
        return (stubs, sources)
    stubs_to_source_deps = collections.defaultdict(list)
    modules = []
    for (node, deps) in reversed(import_graph.deps_list()):
        (stubs, sources) = split_files(_get_filenames(node))
        flat_deps = utils.unique_list(itertools.chain.from_iterable((_get_filenames(d) for d in deps)))
        (stub_deps, source_deps) = split_files(flat_deps)
        for stub in stubs:
            stubs_to_source_deps[stub].extend(source_deps)
            for stub_dep in stub_deps:
                stubs_to_source_deps[stub].extend(stubs_to_source_deps[stub_dep])
        if sources:
            for stub in stub_deps:
                source_deps.extend(stubs_to_source_deps[stub])
            modules.append((tuple(sources), tuple(source_deps)))
    return modules

def _is_type_stub(f):
    if False:
        i = 10
        return i + 15
    (_, ext) = path_utils.splitext(f)
    return ext in ('.pyi', '.pytd')

def _module_to_output_path(mod):
    if False:
        print('Hello World!')
    'Convert a module to an output path.'
    (path, _) = path_utils.splitext(mod.target)
    if path.replace(path_utils.sep, '.').endswith(mod.name):
        return path[-len(mod.name):]
    else:
        return mod.name[0] + mod.name[1:].replace('.', path_utils.sep)

def escape_ninja_path(path: str):
    if False:
        for i in range(10):
            print('nop')
    'escape `:` in absolute path on windows.'
    if sys.platform == 'win32':
        new_path = ''
        last_char = None
        for ch in path:
            if last_char != '$' and ch == ':':
                new_path += '$:'
            else:
                new_path += ch
            last_char = ch
        return new_path
    else:
        return path

def get_imports_map(deps, module_to_imports_map, module_to_output):
    if False:
        print('Hello World!')
    'Get a short path -> full path map for the given deps.'
    imports_map = {}
    for m in deps:
        if m in module_to_imports_map:
            imports_map.update(module_to_imports_map[m])
        imports_map[_module_to_output_path(m)] = module_to_output[m]
    return imports_map

class PytypeRunner:
    """Runs pytype over an import graph."""

    def __init__(self, conf, sorted_sources):
        if False:
            return 10
        self.filenames = set(conf.inputs)
        self.sorted_sources = sorted_sources
        self.python_version = conf.python_version
        self.platform = conf.platform
        self.pyi_dir = path_utils.join(conf.output, 'pyi')
        self.imports_dir = path_utils.join(conf.output, 'imports')
        self.ninja_file = path_utils.join(conf.output, 'build.ninja')
        self.custom_options = [(k, getattr(conf, k)) for k in set(conf.__slots__) - set(config.ITEMS)]
        self.keep_going = conf.keep_going
        self.jobs = conf.jobs

    def set_custom_options(self, flags_with_values, binary_flags, report_errors):
        if False:
            while True:
                i = 10
        'Merge self.custom_options into flags_with_values and binary_flags.'
        for (dest, value) in self.custom_options:
            if not report_errors and dest in config.REPORT_ERRORS_ITEMS:
                continue
            arg_info = config.get_pytype_single_item(dest).arg_info
            assert arg_info is not None
            if arg_info.to_command_line:
                value = arg_info.to_command_line(value)
            if isinstance(value, bool):
                if value:
                    binary_flags.add(arg_info.flag)
                else:
                    binary_flags.discard(arg_info.flag)
            elif value:
                flags_with_values[arg_info.flag] = str(value)

    def get_pytype_command_for_ninja(self, report_errors):
        if False:
            while True:
                i = 10
        'Get the command line for running pytype.'
        exe = PYTYPE_SINGLE
        flags_with_values = {'--imports_info': '$imports', '-V': self.python_version, '-o': '$out', '--module-name': '$module', '--platform': self.platform}
        binary_flags = {'--quick', '--analyze-annotated' if report_errors else '--no-report-errors', '--nofail'}
        self.set_custom_options(flags_with_values, binary_flags, report_errors)
        return exe + list(sum(sorted(flags_with_values.items()), ())) + sorted(binary_flags) + ['$in']

    def make_imports_dir(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            file_utils.makedirs(self.imports_dir)
        except OSError:
            logging.error('Could not create imports directory: %s', self.imports_dir)
            return False
        return True

    def write_default_pyi(self):
        if False:
            print('Hello World!')
        'Write a default pyi file.'
        output = path_utils.join(self.imports_dir, 'default.pyi')
        with open(output, 'w') as f:
            f.write(DEFAULT_PYI)
        return output

    def write_imports(self, module_name, imports_map, suffix):
        if False:
            while True:
                i = 10
        'Write a .imports file.'
        output = path_utils.join(self.imports_dir, module_name + '.imports' + suffix)
        with open(output, 'w') as f:
            for item in imports_map.items():
                f.write('%s %s\n' % item)
        return output

    def get_module_action(self, module):
        if False:
            while True:
                i = 10
        'Get the action for the given module.\n\n    Args:\n      module: A module_utils.Module object.\n\n    Returns:\n      An Action object, or None for a non-Python file.\n    '
        f = module.full_path
        if f in self.filenames:
            action = Action.CHECK
            report = logging.warning
        else:
            action = Action.INFER
            report = logging.info
        if not module.name.startswith('pytype_extensions.') and module.kind in ('Builtin', 'System'):
            action = Action.GENERATE_DEFAULT
            report('%s: %s module %s', action, module.kind, module.name)
        return action

    def yield_sorted_modules(self) -> Iterable[Tuple[module_utils.Module, str, Sequence[module_utils.Module], str]]:
        if False:
            return 10
        'Yield modules from our sorted source files.'
        for (group, deps) in self.sorted_sources:
            modules = []
            for module in group:
                action = self.get_module_action(module)
                if action:
                    modules.append((module, action))
            if len(modules) == 1:
                yield (modules[0] + (deps, Stage.SINGLE_PASS))
            else:
                second_pass_deps = []
                for (module, action) in modules:
                    second_pass_deps.append(module)
                    if action == Action.CHECK:
                        action = Action.INFER
                    yield (module, action, deps, Stage.FIRST_PASS)
                deps += tuple(second_pass_deps)
                for (module, action) in modules:
                    if action != Action.GENERATE_DEFAULT:
                        yield (module, action, deps, Stage.SECOND_PASS)

    def write_ninja_preamble(self):
        if False:
            print('Hello World!')
        'Write out the pytype-single commands that the build will call.'
        with open(self.ninja_file, 'w') as f:
            for (action, report_errors) in ((Action.INFER, False), (Action.CHECK, True)):
                command = ' '.join(self.get_pytype_command_for_ninja(report_errors=report_errors))
                logging.info('%s command: %s', action, command)
                f.write('rule {action}\n  command = {command}\n  description = {action} $module\n'.format(action=action, command=command))

    def write_build_statement(self, module, action, deps, imports, suffix):
        if False:
            i = 10
            return i + 15
        "Write a build statement for the given module.\n\n    Args:\n      module: A module_utils.Module object.\n      action: An Action object.\n      deps: The module's dependencies.\n      imports: An imports file.\n      suffix: An output file suffix.\n\n    Returns:\n      The expected output of the build statement.\n    "
        output = path_utils.join(self.pyi_dir, _module_to_output_path(module) + '.pyi' + suffix)
        logging.info('%s %s\n  imports: %s\n  deps: %s\n  output: %s', action, module.name, imports, deps, output)
        if deps:
            deps = ' | ' + escape_ninja_path(' '.join(deps))
        else:
            deps = ''
        with open(self.ninja_file, 'a') as f:
            f.write('build {output}: {action} {input}{deps}\n  imports = {imports}\n  module = {module}\n'.format(output=escape_ninja_path(output), action=action, input=escape_ninja_path(module.full_path), deps=deps, imports=escape_ninja_path(imports), module=module.name))
        return output

    def setup_build(self):
        if False:
            for i in range(10):
                print('nop')
        'Write out the full build.ninja file.\n\n    Returns:\n      All files with build statements.\n    '
        if not self.make_imports_dir():
            return set()
        default_output = self.write_default_pyi()
        self.write_ninja_preamble()
        files = set()
        module_to_imports_map = {}
        module_to_output = {}
        for (module, action, deps, stage) in self.yield_sorted_modules():
            if files >= self.filenames:
                logging.info('skipped: %s %s (%s)', action, module.name, stage)
                continue
            if action == Action.GENERATE_DEFAULT:
                module_to_output[module] = default_output
                continue
            if stage == Stage.SINGLE_PASS:
                files.add(module.full_path)
                suffix = ''
            elif stage == Stage.FIRST_PASS:
                suffix = FIRST_PASS_SUFFIX
            else:
                assert stage == Stage.SECOND_PASS
                files.add(module.full_path)
                suffix = ''
            imports_map = module_to_imports_map[module] = get_imports_map(deps, module_to_imports_map, module_to_output)
            imports = self.write_imports(module.name, imports_map, suffix)
            deps = tuple((module_to_output[m] for m in deps if module_to_output[m] != default_output))
            module_to_output[module] = self.write_build_statement(module, action, deps, imports, suffix)
        return files

    def build(self):
        if False:
            print('Hello World!')
        'Execute the build.ninja file.'
        k = '0' if self.keep_going else '1'
        c = path_utils.relpath(path_utils.dirname(self.ninja_file))
        command = _get_executable('ninja') + ['-k', k, '-C', c, '-j', str(self.jobs)]
        if logging.getLogger().isEnabledFor(logging.INFO):
            command.append('-v')
        ret = subprocess.call(command)
        print(f'Leaving directory {c!r}')
        return ret

    def run(self):
        if False:
            while True:
                i = 10
        'Run pytype over the project.'
        logging.info('------------- Starting pytype run. -------------')
        files_to_analyze = self.setup_build()
        num_sources = len(self.filenames & files_to_analyze)
        print('Analyzing %d sources with %d local dependencies' % (num_sources, len(files_to_analyze) - num_sources))
        ret = self.build()
        if not ret:
            print('Success: no errors found')
        return ret