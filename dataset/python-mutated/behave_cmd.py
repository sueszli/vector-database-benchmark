from __future__ import absolute_import
import sys
import os.path
NAME = 'behave'
HERE = os.path.dirname(__file__)
TOP = os.path.join(HERE, '..')
if os.path.isdir(os.path.join(TOP, NAME)):
    sys.path.insert(0, os.path.abspath(TOP))

def setup_behave():
    if False:
        return 10
    '\n    Apply tweaks, extensions and patches to "behave".\n    '
    from behave.configuration import Configuration
    Configuration.defaults['show_timings'] = False

def behave_main0():
    if False:
        print('Hello World!')
    from behave.__main__ import main as behave_main
    setup_behave()
    return behave_main()
if __name__ == '__main__':
    if 'COVERAGE_PROCESS_START' in os.environ:
        import coverage
        coverage.process_startup()
    sys.exit(behave_main0())