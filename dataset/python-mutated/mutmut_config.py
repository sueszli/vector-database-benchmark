"""
Configuration for mutmut.

See https://mutmut.readthedocs.io/en/latest/
"""
from mutmut import Context

def pre_mutation(context: Context) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Filter what to mutate.\n\n    Args:\n        context: A mutmut Context object\n    '
    line = context.current_source_line.strip()
    if '_codecs' in context.filename or 'pragma: no cover' in line or 'deprecate' in line or line.startswith('logger'):
        context.skip = True