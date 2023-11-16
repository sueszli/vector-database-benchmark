from __future__ import annotations
import sys
from ansible.module_utils.common.warnings import deprecate

def get_exception():
    if False:
        print('Hello World!')
    'Get the current exception.\n\n    This code needs to work on Python 2.4 through 3.x, so we cannot use\n    "except Exception, e:" (SyntaxError on Python 3.x) nor\n    "except Exception as e:" (SyntaxError on Python 2.4-2.5).\n    Instead we must use ::\n\n        except Exception:\n            e = get_exception()\n\n    '
    deprecate(msg='The `ansible.module_utils.pycompat24.get_exception` function is deprecated.', version='2.19')
    return sys.exc_info()[1]

def __getattr__(importable_name):
    if False:
        print('Hello World!')
    'Inject import-time deprecation warning for ``literal_eval()``.'
    if importable_name == 'literal_eval':
        deprecate(msg=f'The `ansible.module_utils.pycompat24.{importable_name}` function is deprecated.', version='2.19')
        from ast import literal_eval
        return literal_eval
    raise AttributeError(f'cannot import name {importable_name!r} has no attribute ({__file__!s})')
__all__ = ('get_exception', 'literal_eval')