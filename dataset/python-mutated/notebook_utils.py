import os

def is_jupyter():
    if False:
        for i in range(10):
            print('nop')
    'Check if the module is running on Jupyter notebook/console.\n\n    Returns:\n        bool: True if the module is running on Jupyter notebook or Jupyter console,\n        False otherwise.\n    '
    try:
        shell_name = get_ipython().__class__.__name__
        if shell_name == 'ZMQInteractiveShell':
            return True
        else:
            return False
    except NameError:
        return False

def is_databricks():
    if False:
        for i in range(10):
            print('nop')
    'Check if the module is running on Databricks.\n\n    Returns:\n        bool: True if the module is running on Databricks notebook,\n        False otherwise.\n    '
    try:
        if os.path.realpath('.') == '/databricks/driver':
            return True
        else:
            return False
    except NameError:
        return False