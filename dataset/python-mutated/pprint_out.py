"""
Python pretty-print (pprint)
============================

The python pretty-print system was once the default outputter. It simply
passes the return data through to ``pprint.pformat`` and prints the results.

CLI Example:

.. code-block:: bash

    salt '*' foo.bar --out=pprint

Example output:

.. code-block:: python

    {'saltmine': {'foo': {'bar': 'baz',
                          'dictionary': {'abc': 123, 'def': 456},
                          'list': ['Hello', 'World']}}}
"""
import pprint
__virtualname__ = 'pprint'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Change the name to pprint\n    '
    return __virtualname__

def output(data, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Print out via pretty print\n    '
    if isinstance(data, Exception):
        data = str(data)
    if 'output_indent' in __opts__ and __opts__['output_indent'] >= 0:
        return pprint.pformat(data, indent=__opts__['output_indent'])
    return pprint.pformat(data)