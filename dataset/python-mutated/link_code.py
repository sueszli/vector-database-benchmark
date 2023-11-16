"""Functionality in this file is used for getting the [source] links on the classes, methods etc
to link to the correct files & lines on github. Can be simplified once
https://github.com/sphinx-doc/sphinx/issues/1556 is closed
"""
import subprocess
from sphinx.util import logging
sphinx_logger = logging.getLogger(__name__)
LINE_NUMBERS = {}

def _git_branch() -> str:
    if False:
        print('Hello World!')
    "Get's the current git sha if available or fall back to `master`"
    try:
        output = subprocess.check_output(['git', 'describe', '--tags', '--always'], stderr=subprocess.STDOUT)
        return output.decode().strip()
    except Exception as exc:
        sphinx_logger.exception('Failed to get a description of the current commit. Falling back to `master`.', exc_info=exc)
        return 'master'
git_branch = _git_branch()
base_url = 'https://github.com/python-telegram-bot/python-telegram-bot/blob/'

def linkcode_resolve(_, info):
    if False:
        i = 10
        return i + 15
    'See www.sphinx-doc.org/en/master/usage/extensions/linkcode.html'
    combined = '.'.join((info['module'], info['fullname']))
    combined = combined.replace('ExtBot.ExtBot', 'ExtBot')
    line_info = LINE_NUMBERS.get(combined)
    if not line_info:
        line_info = LINE_NUMBERS.get(f"{combined.rsplit('.', 1)[0]}.__init__")
    if not line_info:
        line_info = LINE_NUMBERS.get(f"{combined.rsplit('.', 1)[0]}")
    if not line_info:
        line_info = LINE_NUMBERS.get(info['module'])
    if not line_info:
        return
    (file, start_line, end_line) = line_info
    return f'{base_url}{git_branch}/{file}#L{start_line}-L{end_line}'