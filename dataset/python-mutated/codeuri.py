"""
Contains CodeUri Related methods
"""
import logging
import os
LOG = logging.getLogger(__name__)
PRESENT_DIR = '.'

def resolve_code_path(cwd, codeuri):
    if False:
        return 10
    '\n    Returns path to the function code resolved based on current working directory.\n\n    Parameters\n    ----------\n    cwd : str\n        Current working directory\n    codeuri : str\n        CodeURI of the function. This should contain the path to the function code\n\n    Returns\n    -------\n    str\n        Absolute path to the function code\n\n    '
    LOG.debug('Resolving code path. Cwd=%s, CodeUri=%s', cwd, codeuri)
    if not cwd or cwd == PRESENT_DIR:
        cwd = os.getcwd()
    cwd = os.path.abspath(cwd)
    if not os.path.isabs(codeuri):
        codeuri = os.path.normpath(os.path.join(cwd, codeuri))
    return codeuri