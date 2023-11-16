""" This test runner compiles all Python files as a module.

This is a test to achieve some coverage, it will only find assertions of
within Nuitka or warnings from the C compiler. Code will not be run
normally.

"""
import os
import sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from nuitka import Options
from nuitka.__past__ import iter_modules
from nuitka.importing.Importing import addMainScriptDirectory, decideModuleSourceRef, locateModule
from nuitka.tools.testing.Common import my_print, setup, test_logger
from nuitka.Tracing import plugins_logger
from nuitka.tree.SourceHandling import getSourceCodeDiff, readSourceCodeFromFilenameWithInformation
from nuitka.utils.ModuleNames import ModuleName
python_version = setup(suite='python_modules', needs_io_encoding=True)
addMainScriptDirectory('/doesnotexist')
Options.is_full_compat = False

def scanModule(name_space, module_iterator):
    if False:
        i = 10
        return i + 15
    from nuitka.tree.TreeHelpers import parseSourceCodeToAst
    for module_desc in module_iterator:
        if name_space is None:
            module_name = ModuleName(module_desc.name)
        else:
            module_name = name_space.getChildNamed(module_desc.name)
        try:
            (_module_name, module_filename, finding) = locateModule(module_name=module_name, parent_package=None, level=0)
        except AssertionError:
            continue
        assert _module_name == module_name, module_desc
        if module_filename is None:
            continue
        (_main_added, _is_package, _is_namespace, _source_ref, source_filename) = decideModuleSourceRef(filename=module_filename, module_name=module_name, is_main=False, is_fake=False, logger=test_logger)
        try:
            (source_code, original_source_code, contributing_plugins) = readSourceCodeFromFilenameWithInformation(module_name=module_name, source_filename=source_filename)
        except SyntaxError:
            continue
        try:
            parseSourceCodeToAst(source_code=source_code, module_name=module_name, filename=source_filename, line_offset=0)
        except (SyntaxError, IndentationError) as e:
            try:
                parseSourceCodeToAst(source_code=original_source_code, module_name=module_name, filename=source_filename, line_offset=0)
            except (SyntaxError, IndentationError):
                pass
            else:
                source_diff = getSourceCodeDiff(original_source_code, source_code)
                for line in source_diff:
                    plugins_logger.warning(line)
                if len(contributing_plugins) == 1:
                    contributing_plugins[0].sysexit("Making changes to '%s' that cause SyntaxError '%s'" % (module_name, e))
                else:
                    test_logger.sysexit("One of the plugins '%s' is making changes to '%s' that cause SyntaxError '%s'" % (','.join(contributing_plugins), module_name, e))
        my_print(module_name, ':', finding, 'OK')
        if module_desc.ispkg:
            scanModule(module_name, iter_modules([module_filename]))

def main():
    if False:
        return 10
    scanModule(None, iter_modules())
if __name__ == '__main__':
    main()