""" Nuitka distutils integration.

"""
import distutils.command.build
import distutils.command.install
import os
import sys
import wheel.bdist_wheel
from nuitka.__past__ import Iterable, unicode
from nuitka.containers.OrderedSets import OrderedSet
from nuitka.importing.Importing import addMainScriptDirectory, decideModuleSourceRef, flushImportCache, locateModule
from nuitka.Tracing import wheel_logger
from nuitka.utils.Execution import check_call
from nuitka.utils.FileOperations import deleteFile, getFileList, renameFile
from nuitka.utils.ModuleNames import ModuleName

def setupNuitkaDistutilsCommands(dist, keyword, value):
    if False:
        return 10
    if not value:
        return
    dist.cmdclass = dist.cmdclass or {}
    dist.cmdclass['build'] = build
    dist.cmdclass['install'] = install
    dist.cmdclass['bdist_wheel'] = bdist_nuitka

def addToPythonPath(python_path, in_front=False):
    if False:
        return 10
    if type(python_path) in (tuple, list):
        python_path = os.pathsep.join(python_path)
    if python_path:
        if 'PYTHONPATH' in os.environ:
            if in_front:
                os.environ['PYTHONPATH'] = python_path + os.pathsep + os.environ['PYTHONPATH']
            else:
                os.environ['PYTHONPATH'] += os.pathsep + python_path
        else:
            os.environ['PYTHONPATH'] = python_path

class build(distutils.command.build.build):

    def run(self):
        if False:
            i = 10
            return i + 15
        wheel_logger.info('Specified packages: %s.' % self.distribution.packages)
        wheel_logger.info('Specified modules: %s.' % self.distribution.py_modules)
        self.compile_packages = self.distribution.packages or []
        self.py_modules = self.distribution.py_modules or []
        self.script_module_names = OrderedSet()
        if self.distribution.entry_points is not None:
            for (group, script_specs) in self.distribution.entry_points.items():
                for script_spec in script_specs:
                    try:
                        script_module_name = script_spec.split('=', 1)[1].strip().split(':')[0]
                    except Exception as e:
                        wheel_logger.warning("Problem parsing '%s' script specification in '%s' due to %s" % (script_spec, group, e))
                    self.script_module_names.add(ModuleName(script_module_name))
        if not self.compile_packages and (not self.py_modules):
            wheel_logger.sysexit("No modules or packages specified, aborting. Did you provide packages in 'setup.cfg' or 'setup.py'?")
        distutils.command.build.build.run(self)
        self._build(os.path.abspath(self.build_lib))

    def _findBuildTasks2(self):
        if False:
            i = 10
            return i + 15
        '\n        Helper for _build\n        Returns list containing bool (is_package) and module_names\n\n        Algorithm for finding distinct packages:\n        1) Take minimum package\n        2) Find related packages that start with this name\n        3) Add this to the list to return, then repeat steps 1 & 2\n           until no more packages exist\n\n        '
        builds = []
        py_packages = [ModuleName(m.replace('/', '.')) for m in sorted(set(self.compile_packages))]
        py_modules = [ModuleName(m) for m in sorted(set(self.py_modules))]
        for script_module_name in self.script_module_names:
            script_module_filename = locateModule(module_name=script_module_name, parent_package=None, level=0)[1]
            if script_module_filename is None:
                wheel_logger.sysexit("Error, failed to locate script containing module '%s'" % script_module_name)
            (_main_added, is_package, _is_namespace, _source_ref, _source_filename) = decideModuleSourceRef(filename=script_module_filename, module_name=script_module_name, is_main=False, is_fake=False, logger=wheel_logger)
            if is_package:
                py_packages.append(script_module_name)
            else:
                py_modules.append(script_module_name)
        builds.extend(((False, current_module) for current_module in py_modules if not current_module.hasOneOfNamespaces(py_packages)))
        while py_packages:
            current_package = min(py_packages)
            py_packages = [p for p in py_packages if not p.hasNamespace(current_package)]
            builds.append((True, current_package))
        return builds

    def _findBuildTasks(self):
        if False:
            return 10
        builds = self._findBuildTasks2()
        result = []
        for (_is_package, module_name_orig) in builds:
            (_module_name, main_filename, module_kind, finding) = locateModule(module_name=module_name_orig, parent_package=None, level=0)
            if os.path.isdir(main_filename):
                if not getFileList(main_filename, only_suffixes=('.py',)):
                    wheel_logger.info("Skipping '%s' from Nuitka compilation due to containing no Python code." % module_name_orig)
                    continue
            if module_kind == 'extension':
                main_filename_away = main_filename + '.away'
                renameFile(main_filename, main_filename_away)
                flushImportCache()
                (_module_name, main_filename, module_kind, finding) = locateModule(module_name=module_name_orig, parent_package=None, level=0)
                if finding != 'not-found':
                    deleteFile(main_filename_away, must_exist=True)
                else:
                    renameFile(main_filename_away, main_filename)
                    continue
            result.append((_is_package, module_name_orig))
        return result

    @staticmethod
    def _parseOptionsEntry(option, value):
        if False:
            for i in range(10):
                print('nop')
        option = '--' + option.lstrip('-')
        if type(value) is tuple and len(value) == 2 and (value[0] == 'setup.py'):
            value = value[1]
        if value is None or value == '':
            yield option
        elif isinstance(value, bool):
            yield ('--' + ('no' if not value else '') + option.lstrip('-'))
        elif isinstance(value, Iterable) and (not isinstance(value, (unicode, bytes, str))):
            for val in value:
                yield ('%s=%s' % (option, val))
        else:
            yield ('%s=%s' % (option, value))

    def _build(self, build_lib):
        if False:
            print('Hello World!')
        old_dir = os.getcwd()
        os.chdir(build_lib)
        if self.distribution.package_dir and '' in self.distribution.package_dir:
            main_package_dir = os.path.join(build_lib, self.distribution.package_dir.get(''))
        else:
            main_package_dir = os.path.abspath(build_lib)
        addMainScriptDirectory(main_package_dir)
        for (is_package, module_name) in self._findBuildTasks():
            (module_name, main_filename, _module_kind, finding) = locateModule(module_name=module_name, parent_package=None, level=0)
            package_name = module_name.getPackageName()
            assert finding == 'absolute', finding
            if package_name is not None:
                output_dir = os.path.join(build_lib, package_name.asPath())
            else:
                output_dir = build_lib
            command = [sys.executable, '-m', 'nuitka', '--module', '--enable-plugin=pylint-warnings', '--output-dir=%s' % output_dir, '--nofollow-import-to=*.tests', '--remove-output']
            if package_name is not None:
                (package_part, include_package_name) = module_name.splitModuleBasename()
                addToPythonPath(os.path.join(main_package_dir, package_part.asPath()), in_front=True)
            else:
                include_package_name = module_name
            if is_package:
                command.append('--include-package=%s' % include_package_name)
            else:
                command.append('--include-module=%s' % include_package_name)
            toml_filename = os.environ.get('NUITKA_TOML_FILE')
            if toml_filename:
                try:
                    from tomli import loads as toml_loads
                except ImportError:
                    from toml import loads as toml_loads
                with open(toml_filename) as toml_file:
                    toml_options = toml_loads(toml_file.read())
                for (option, value) in toml_options.get('nuitka', {}).items():
                    command.extend(self._parseOptionsEntry(option, value))
            if 'nuitka' in self.distribution.command_options:
                for (option, value) in self.distribution.command_options['nuitka'].items():
                    command.extend(self._parseOptionsEntry(option, value))
            command.append(main_filename)
            wheel_logger.info("Building: '%s' with command '%s'" % (module_name.asString(), command))
            check_call(command, cwd=build_lib)
            wheel_logger.info("Finished compilation of '%s'." % module_name.asString(), style='green')
        self.build_lib = build_lib
        os.chdir(old_dir)
        for (root, _, filenames) in os.walk(build_lib):
            for filename in filenames:
                fullpath = os.path.join(root, filename)
                if fullpath.lower().endswith(('.py', '.pyw', '.pyc', '.pyo')):
                    os.unlink(fullpath)

class install(distutils.command.install.install):

    def finalize_options(self):
        if False:
            for i in range(10):
                print('nop')
        distutils.command.install.install.finalize_options(self)
        self.install_lib = self.install_platlib

class bdist_nuitka(wheel.bdist_wheel.bdist_wheel):

    def initialize_options(self):
        if False:
            return 10
        dist = self.distribution
        dist.cmdclass = dist.cmdclass or {}
        dist.cmdclass['build'] = build
        dist.cmdclass['install'] = install
        wheel.bdist_wheel.bdist_wheel.initialize_options(self)

    def finalize_options(self):
        if False:
            i = 10
            return i + 15
        wheel.bdist_wheel.bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def write_wheelfile(self, wheelfile_base, generator=None):
        if False:
            print('Hello World!')
        if generator is None:
            from nuitka.Version import getNuitkaVersion
            generator = 'Nuitka (%s)' % getNuitkaVersion()
        wheel.bdist_wheel.bdist_wheel.write_wheelfile(self, wheelfile_base=wheelfile_base, generator=generator)