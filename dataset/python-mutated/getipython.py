"""Simple function to call to get the current InteractiveShell instance
"""

def get_ipython():
    if False:
        print('Hello World!')
    'Get the global InteractiveShell instance.\n\n    Returns None if no InteractiveShell instance is registered.\n    '
    from IPython.core.interactiveshell import InteractiveShell
    if InteractiveShell.initialized():
        return InteractiveShell.instance()