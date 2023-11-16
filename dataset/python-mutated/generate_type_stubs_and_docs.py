import os
import re
import sys
import argparse
import inspect
import typing
import shutil
import difflib
import importlib
import pathlib
import psutil
import subprocess
import traceback
from tempfile import TemporaryDirectory
from contextlib import contextmanager
from typing import Dict, List
from pybind11_stubgen import ClassStubsGenerator, StubsGenerator
from pybind11_stubgen import main as pybind11_stubgen_main
import mypy.stubtest
from mypy.stubtest import test_stubs, parse_options as mypy_parse_options
import sphinx.ext.autodoc.importer
from sphinx.cmd.build import main as sphinx_build_main
from scripts.postprocess_type_hints import main as postprocess_type_hints_main
MAX_DIFF_LINE_LENGTH = 150
SPHINX_REPLACEMENTS = {'pedalboard_native\\.': 'pedalboard.', '<em class="sig-param"><span class="n"><span class="pre">self</span></span><span class="p"><span class="pre">:.*?</em>, ': '', '<em class="sig-param"><span class="n"><span class="pre">self</span></span><span class="p"><span class="pre">:.*?</em><span class="sig-paren">\\)</span>': '<span class="sig-paren">)</span>', '<span class="pre">pedalboard\\.Plugin</span>': '<span class="pre">Plugin</span>'}

def patch_mypy_stubtest():
    if False:
        return 10

    def patched_verify_metaclass(*args, **kwargs):
        if False:
            while True:
                i = 10
        return []
    mypy.stubtest._verify_metaclass = patched_verify_metaclass

def patch_pybind11_stubgen():
    if False:
        i = 10
        return i + 15
    '\n    Patch ``pybind11_stubgen`` to generate more ergonomic code for Enum-like classes.\n    This generates a subclass of :class:``Enum`` for each Pybind11-generated Enum,\n    which is not strictly correct, but produces much nicer documentation and allows\n    for a much more Pythonic API.\n    '
    original_class_stubs_generator_new = ClassStubsGenerator.__new__

    class EnumClassStubsGenerator(StubsGenerator):

        def __init__(self, klass):
            if False:
                return 10
            self.klass = klass
            assert inspect.isclass(klass)
            assert klass.__name__.isidentifier()
            assert hasattr(klass, '__entries')
            self.doc_string = None
            self.enum_names = []
            self.enum_values = []
            self.enum_docstrings = []

        def get_involved_modules_names(self):
            if False:
                for i in range(10):
                    print('nop')
            return []

        def parse(self):
            if False:
                print('Hello World!')
            self.doc_string = self.klass.__doc__ or ''
            self.doc_string = self.doc_string.split('Members:')[0]
            for (name, (value_object, docstring)) in getattr(self.klass, '__entries').items():
                self.enum_names.append(name)
                self.enum_values.append(value_object.value)
                self.enum_docstrings.append(docstring)

        def to_lines(self):
            if False:
                for i in range(10):
                    print('nop')
            result = ['class {class_name}(Enum):{doc_string}'.format(class_name=self.klass.__name__, doc_string='\n' + self.format_docstring(self.doc_string) if self.doc_string else '')]
            for (name, value, docstring) in sorted(list(zip(self.enum_names, self.enum_values, self.enum_docstrings)), key=lambda x: x[1]):
                result.append(f'    {name} = {value}  # fmt: skip')
                result.append(f'{self.format_docstring(docstring)}')
            if not self.enum_names:
                result.append(self.indent('pass'))
            return result

    def patched_class_stubs_generator_new(cls, klass, *args, **kwargs):
        if False:
            return 10
        if hasattr(klass, '__entries'):
            return EnumClassStubsGenerator(klass, *args, **kwargs)
        else:
            return original_class_stubs_generator_new(cls)
    ClassStubsGenerator.__new__ = patched_class_stubs_generator_new

def import_stub(stubs_path: str, module_name: str) -> typing.Any:
    if False:
        while True:
            i = 10
    '\n    Import a stub file (.pyi) as a regular Python module.\n    Note that two modules of the same name cannot (usually) be imported,\n    so additional care may need to be taken after using this method to\n    change ``sys.modules`` to avoid clobbering existing modules.\n    '
    sys.path_hooks.insert(0, importlib.machinery.FileFinder.path_hook((importlib.machinery.SourceFileLoader, ['.pyi'])))
    sys.path.insert(0, stubs_path)
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path.pop(0)
        sys.path_hooks.pop(0)

def patch_sphinx_to_read_pyi():
    if False:
        i = 10
        return i + 15
    '\n    Sphinx doesn\'t know how to read .pyi files, but we use .pyi files as our\n    "source of truth" for the public API that we expose to IDEs and our documentation.\n    This patch tells Sphinx how to read .pyi files, using them to replace their .py\n    counterparts.\n    '
    old_import_module = sphinx.ext.autodoc.importer.import_module

    def patch_import_module(modname: str, *args, **kwargs) -> typing.Any:
        if False:
            print('Hello World!')
        if modname in sys.modules:
            return sys.modules[modname]
        try:
            return import_stub('.', modname)
        except ImportError:
            return old_import_module(modname, *args, **kwargs)
        except Exception as e:
            print(f'Failed to import stub module: {e}')
            traceback.print_exc()
            raise
    sphinx.ext.autodoc.importer.import_module = patch_import_module

@contextmanager
def isolated_imports(only: typing.Set[str]={}):
    if False:
        i = 10
        return i + 15
    "\n    When used as a context manager, this function scopes all imports\n    that happen within it as local to the scope.\n\n    Put another way: if you import something inside a\n    ``with isolated_imports()`` block, it won't be imported after\n    the block is done.\n    "
    before = list(sys.modules.keys())
    yield
    for module_name in list(sys.modules.keys()):
        if module_name not in before and module_name in only:
            del sys.modules[module_name]

def remove_non_public_files(output_dir: str):
    if False:
        i = 10
        return i + 15
    try:
        shutil.rmtree(os.path.join(output_dir, '.doctrees'))
    except Exception:
        pass
    try:
        os.unlink(os.path.join(output_dir, '.buildinfo'))
    except Exception:
        pass

def trim_diff_line(x: str) -> str:
    if False:
        return 10
    x = x.strip()
    if len(x) > MAX_DIFF_LINE_LENGTH:
        suffix = f' [plus {len(x) - MAX_DIFF_LINE_LENGTH:,} more characters]'
        return x[:MAX_DIFF_LINE_LENGTH - len(suffix)] + suffix
    else:
        return x

def glob_matches(filename: str, globs: List[str]) -> bool:
    if False:
        while True:
            i = 10
    for glob in globs:
        if glob.startswith('*') and filename.lower().endswith(glob[1:].lower()):
            return True
        if glob in filename:
            return True
    return False

def postprocess_sphinx_output(directory: str, renames: Dict[str, str]):
    if False:
        i = 10
        return i + 15
    '\n    I\'ve spent 7 hours of my time this weekend fighting with Sphinx.\n    Rather than find the "correct" way to fix this, I\'m just going to\n    overwrite the HTML output with good old find-and-replace.\n    '
    for html_path in pathlib.Path(directory).rglob('*.html'):
        html_contents = html_path.read_text()
        for (find, replace) in renames.items():
            results = re.findall(find, html_contents)
            if results:
                html_contents = re.sub(find, replace, html_contents)
        with open(html_path, 'w') as f:
            f.write(html_contents)

def main():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(description='Generate type stub files (.pyi) and Sphinx documentation for Pedalboard.')
    parser.add_argument('--docs-output-dir', default='docs', help='Output directory for documentation HTML files.')
    parser.add_argument('--docs-input-dir', default=os.path.join('docs', 'source'), help='Input directory for Sphinx.')
    parser.add_argument('--skip-regenerating-type-hints', action='store_true', help="If set, don't bother regenerating or reprocessing type hint files.")
    parser.add_argument('--check', action='store_true', help='If set, compare the existing files with those that would have been generated if this script were re-run.')
    parser.add_argument('--skip-comparing', nargs='*', default=['*.js', '*.css'], help="If set and if --check is passed, the provided filenames (including '*' globs) will be ignored when comparing expected file contents against actual file contents.")
    args = parser.parse_args()
    patch_mypy_stubtest()
    patch_pybind11_stubgen()
    patch_sphinx_to_read_pyi()
    if not args.skip_regenerating_type_hints:
        with isolated_imports({'pedalboard', 'pedalboard.io', 'pedalboard_native', 'pedalboard_native.io', 'pedalboard_native.utils'}):
            print('Generating type stubs from native code...')
            pybind11_stubgen_main(['-o', 'pedalboard_native', 'pedalboard_native', '--no-setup-py'])
            native_dir = pathlib.Path('pedalboard_native')
            native_subdir = [f for f in native_dir.glob('*') if 'stubs' in f.name][0]
            shutil.copytree(native_subdir, native_dir, dirs_exist_ok=True)
            shutil.rmtree(native_subdir)
            print('Postprocessing generated type hints...')
            postprocess_type_hints_main(['pedalboard_native', 'pedalboard_native'] + (['--check'] if args.check else []))
            if sys.version_info > (3, 6):
                print('Running `mypy.stubtest` to validate stubs match...')
                test_stubs(mypy_parse_options(['pedalboard', '--allowlist', 'stubtest.allowlist', '--ignore-missing-stub', '--ignore-unused-allowlist']))
        subprocess.check_call([psutil.Process(os.getpid()).exe()] + sys.argv + ['--skip-regenerating-type-hints'])
        return
    print('Importing numpy to ensure a successful Pedalboard stub import...')
    import numpy
    print('Importing .pyi files for our native modules...')
    for modname in ['pedalboard_native', 'pedalboard_native.io', 'pedalboard_native.utils']:
        import_stub('.', modname)
    print('Running Sphinx...')
    if args.check:
        missing_files = []
        mismatched_files = []
        with TemporaryDirectory() as tempdir:
            sphinx_build_main(['-b', 'html', args.docs_input_dir, tempdir, '-v', '-v', '-v'])
            postprocess_sphinx_output(tempdir, SPHINX_REPLACEMENTS)
            remove_non_public_files(tempdir)
            for (dirpath, _dirnames, filenames) in os.walk(tempdir):
                prefix = dirpath.replace(tempdir, '').lstrip(os.path.sep)
                for filename in filenames:
                    if glob_matches(filename, args.skip_comparing):
                        print(f'Skipping comparison of file: {filename}')
                        continue
                    expected_path = os.path.join(tempdir, prefix, filename)
                    actual_path = os.path.join(args.docs_output_dir, prefix, filename)
                    if not os.path.isfile(actual_path):
                        missing_files.append(os.path.join(prefix, filename))
                    else:
                        with open(expected_path, 'rb') as e, open(actual_path, 'rb') as a:
                            if e.read() != a.read():
                                mismatched_files.append(os.path.join(prefix, filename))
            if missing_files or mismatched_files:
                error_lines = []
                if missing_files:
                    error_lines.append(f'{len(missing_files):,} file(s) were expected in {args.docs_output_dir}, but not found:')
                    for missing_file in missing_files:
                        error_lines.append(f'\t{missing_file}')
                if mismatched_files:
                    error_lines.append(f'{len(mismatched_files):,} file(s) in {args.docs_output_dir} did not match expected values:')
                    for mismatched_file in mismatched_files:
                        expected_path = os.path.join(tempdir, mismatched_file)
                        actual_path = os.path.join(args.docs_output_dir, mismatched_file)
                        try:
                            with open(expected_path) as e, open(actual_path) as a:
                                diff = difflib.context_diff(e.readlines(), a.readlines(), os.path.join('expected', mismatched_file), os.path.join('actual', mismatched_file))
                            error_lines.append('\n'.join([trim_diff_line(x) for x in diff]))
                        except UnicodeDecodeError:
                            error_lines.append(f'Binary file {mismatched_file} does not match expected contents.')
                raise ValueError('\n'.join(error_lines))
        print('Done! Generated type stubs and documentation are valid.')
    else:
        sphinx_build_main(['-b', 'html', args.docs_input_dir, args.docs_output_dir])
        postprocess_sphinx_output(args.docs_output_dir, SPHINX_REPLACEMENTS)
        remove_non_public_files(args.docs_output_dir)
        print(f'Done! Commit the contents of `{args.docs_output_dir}` to Git.')
if __name__ == '__main__':
    main()