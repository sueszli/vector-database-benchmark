"""Serialization and deserialization of a Context object."""
import io
import os
from . import xdg
from .context.transform import format_to_http_prompt
from .execution import execute
EXCLUDED_OPTIONS = ['--style']
CONTEXT_FILENAME = 'context.hp'

def _get_context_filepath():
    if False:
        print('Hello World!')
    dir_path = xdg.get_data_dir()
    return os.path.join(dir_path, CONTEXT_FILENAME)

def load_context(context, file_path=None):
    if False:
        print('Hello World!')
    'Load a Context object in place from user data directory.'
    if not file_path:
        file_path = _get_context_filepath()
    if os.path.exists(file_path):
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                execute(line, context)

def save_context(context):
    if False:
        for i in range(10):
            print('nop')
    'Save a Context object to user data directory.'
    file_path = _get_context_filepath()
    content = format_to_http_prompt(context, excluded_options=EXCLUDED_OPTIONS)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)