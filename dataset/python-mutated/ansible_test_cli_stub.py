"""Command line entry point for ansible-test."""
from __future__ import annotations
import os
import sys

def main(args=None):
    if False:
        while True:
            i = 10
    'Main program entry point.'
    ansible_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_root = os.path.join(ansible_root, 'test', 'lib')
    if os.path.exists(os.path.join(source_root, 'ansible_test', '_internal', '__init__.py')):
        sys.path.insert(0, source_root)
    from ansible_test._util.target.common.constants import CONTROLLER_PYTHON_VERSIONS
    if version_to_str(sys.version_info[:2]) not in CONTROLLER_PYTHON_VERSIONS:
        raise SystemExit('This version of ansible-test cannot be executed with Python version %s. Supported Python versions are: %s' % (version_to_str(sys.version_info[:3]), ', '.join(CONTROLLER_PYTHON_VERSIONS)))
    if any((not os.get_blocking(handle.fileno()) for handle in (sys.stdin, sys.stdout, sys.stderr))):
        raise SystemExit('Standard input, output and error file handles must be blocking to run ansible-test.')
    from ansible_test._internal import main as cli_main
    cli_main(args)

def version_to_str(version):
    if False:
        return 10
    'Return a version string from a version tuple.'
    return '.'.join((str(n) for n in version))
if __name__ == '__main__':
    main()