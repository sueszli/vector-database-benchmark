""" This test runner compiles all extension modules for standalone mode.

This is a test to reveal hidden dependencies on a system.

"""
import os
import sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
import shutil
from nuitka.tools.testing.Common import check_output, checkLoadedFileAccesses, checkSucceedsWithCPython, compileLibraryTest, createSearchMode, displayFileContents, displayFolderContents, displayRuntimeTraces, getTempDir, my_print, setup, test_logger
from nuitka.tools.testing.RuntimeTracing import getRuntimeTraceOfLoadedFiles
from nuitka.utils.Execution import NuitkaCalledProcessError
from nuitka.utils.FileOperations import getFileContents, openTextFile
from nuitka.utils.ModuleNames import ModuleName

def displayError(dirname, filename):
    if False:
        for i in range(10):
            print('nop')
    assert dirname is None
    dist_path = filename[:-3] + '.dist'
    displayFolderContents('dist folder', dist_path)
    inclusion_log_path = filename[:-3] + '.py.inclusion.log'
    displayFileContents('inclusion log', inclusion_log_path)

def main():
    if False:
        i = 10
        return i + 15
    setup(suite='extension_modules', needs_io_encoding=True)
    search_mode = createSearchMode()
    tmp_dir = getTempDir()
    done = set()

    def decide(root, filename):
        if False:
            return 10
        if os.path.sep + 'Cython' + os.path.sep in root:
            return False
        if root.endswith(os.path.sep + 'matplotlib') or os.path.sep + 'matplotlib' + os.path.sep in root:
            return False
        if filename.endswith('linux-gnu_d.so'):
            return False
        if root.endswith(os.path.sep + 'msgpack'):
            return False
        first_part = filename.split('.')[0]
        if first_part in done:
            return False
        done.add(first_part)
        return filename.endswith(('.so', '.pyd')) and (not filename.startswith('libpython'))
    current_dir = os.path.normpath(os.getcwd())
    current_dir = os.path.normcase(current_dir)

    def action(stage_dir, root, path):
        if False:
            i = 10
            return i + 15
        command = [sys.executable, os.path.join('..', '..', 'bin', 'nuitka'), '--stand', '--run', '--output-dir=%s' % stage_dir, '--remove-output', '--no-progressbar']
        filename = os.path.join(stage_dir, 'importer.py')
        assert path.startswith(root)
        module_name = path[len(root) + 1:]
        module_name = module_name.split('.')[0]
        module_name = module_name.replace(os.path.sep, '.')
        module_name = ModuleName(module_name)
        with openTextFile(filename, 'w') as output:
            plugin_names = set(['pylint-warnings'])
            if module_name.hasNamespace('PySide2'):
                plugin_names.add('pyside2')
            elif module_name.hasNamespace('PySide6'):
                plugin_names.add('pyside6')
            elif module_name.hasNamespace('PyQt5'):
                plugin_names.add('pyqt5')
            elif module_name.hasNamespace('PyQt6'):
                plugin_names.add('pyqt6')
            else:
                plugin_names.add('no-qt')
            for plugin_name in plugin_names:
                output.write('# nuitka-project: --enable-plugin=%s\n' % plugin_name)
            output.write('# nuitka-project: --noinclude-default-mode=error\n')
            output.write('# nuitka-project: --standalone\n')
            output.write('import ' + module_name.asString() + '\n')
            output.write("print('OK.')")
        command += os.environ.get('NUITKA_EXTRA_OPTIONS', '').split()
        command.append(filename)
        if checkSucceedsWithCPython(filename):
            try:
                output = check_output(command).splitlines()
            except NuitkaCalledProcessError as e:
                my_print('SCRIPT:', filename, style='blue')
                my_print(getFileContents(filename))
                test_logger.sysexit('Error with compilation: %s' % e)
            except Exception:
                raise
            else:
                assert os.path.exists(filename[:-3] + '.dist')
                binary_filename = os.path.join(filename[:-3] + '.dist', 'importer.exe' if os.name == 'nt' else 'importer')
                loaded_filenames = getRuntimeTraceOfLoadedFiles(logger=test_logger, command=[binary_filename])
                outside_accesses = checkLoadedFileAccesses(loaded_filenames=loaded_filenames, current_dir=os.getcwd())
                if outside_accesses:
                    displayError(None, filename)
                    displayRuntimeTraces(test_logger, binary_filename)
                    test_logger.warning("Should not access these file(s): '%r'." % outside_accesses)
                    search_mode.onErrorDetected(1)
                if output[-1] != b'OK.':
                    my_print(' '.join(command))
                    my_print(filename)
                    my_print(output)
                    test_logger.sysexit('FAIL.')
                my_print('OK.')
                assert not outside_accesses, outside_accesses
                shutil.rmtree(filename[:-3] + '.dist')
        else:
            my_print('SKIP (does not work with CPython)')
    compileLibraryTest(search_mode=search_mode, stage_dir=os.path.join(tmp_dir, 'compile_extensions'), decide=decide, action=action)
    my_print('FINISHED, all extension modules compiled.')
if __name__ == '__main__':
    main()