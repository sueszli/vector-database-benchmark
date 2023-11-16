"""
NOTE: This file is only used for backward compatibility tests with new dist_utils.py in
st2common/tests/unit/test_dist_utils.py.

DO NOT USE THIS FILE ANYWHERE ELSE!
"""
from __future__ import absolute_import
import os
import re
import sys
PY3 = sys.version_info[0] == 3
if PY3:
    text_type = str
else:
    text_type = unicode
GET_PIP = 'curl https://bootstrap.pypa.io/get-pip.py | python'
try:
    from pip import __version__ as pip_version
except ImportError as e:
    print('Failed to import pip: %s' % text_type(e))
    print('')
    print('Download pip:\n%s' % GET_PIP)
    sys.exit(1)
try:
    from pip.req import parse_requirements
except ImportError:
    try:
        from pip._internal.req.req_file import parse_requirements
    except ImportError as e:
        print('Failed to import parse_requirements from pip: %s' % text_type(e))
        print('Using pip: %s' % str(pip_version))
        sys.exit(1)
__all__ = ['fetch_requirements', 'apply_vagrant_workaround', 'get_version_string', 'parse_version_string']

def fetch_requirements(requirements_file_path):
    if False:
        return 10
    '\n    Return a list of requirements and links by parsing the provided requirements file.\n    '
    links = []
    reqs = []
    for req in parse_requirements(requirements_file_path, session=False):
        link = getattr(req, 'link', getattr(req, 'url', None))
        if link:
            links.append(str(link))
        reqs.append(str(req.req))
    return (reqs, links)

def apply_vagrant_workaround():
    if False:
        return 10
    '\n    Function which detects if the script is being executed inside vagrant and if it is, it deletes\n    "os.link" attribute.\n    Note: Without this workaround, setup.py sdist will fail when running inside a shared directory\n    (nfs / virtualbox shared folders).\n    '
    if os.environ.get('USER', None) == 'vagrant':
        del os.link

def get_version_string(init_file):
    if False:
        for i in range(10):
            print('nop')
    '\n    Read __version__ string for an init file.\n    '
    with open(init_file, 'r') as fp:
        content = fp.read()
        version_match = re.search('^__version__ = [\'\\"]([^\'\\"]*)[\'\\"]', content, re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError('Unable to find version string in %s.' % init_file)
parse_version_string = get_version_string