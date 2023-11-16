"""
Bugs in sphinx-autoapi using metaclasses prevent us from upgrading to 1.3
which has implicit namespace support. Until that time, we make it look
like a real package for building docs
"""
from __future__ import annotations
import os
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sphinx.application import Sphinx
ROOT_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
PROVIDER_INIT_FILE = os.path.join(ROOT_PROJECT_DIR, 'airflow', 'providers', '__init__.py')

def _create_init_py(app, config):
    if False:
        i = 10
        return i + 15
    del app
    del config
    with open(PROVIDER_INIT_FILE, 'w'):
        pass

def setup(app: Sphinx):
    if False:
        for i in range(10):
            print('nop')
    '\n    Sets the plugin up and returns configuration of the plugin.\n\n    :param app: application.\n    :return json description of the configuration that is needed by the plugin.\n    '
    app.connect('config-inited', _create_init_py)
    return {'version': 'builtin', 'parallel_read_safe': True, 'parallel_write_safe': True}