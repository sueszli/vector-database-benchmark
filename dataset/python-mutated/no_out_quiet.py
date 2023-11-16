"""
Display no output
=================

No output is produced when this outputter is selected

CLI Example:

.. code-block:: bash

    salt '*' foo.bar --out=quiet
"""
__virtualname__ = 'quiet'

def __virtual__():
    if False:
        print('Hello World!')
    return __virtualname__

def output(ret, **kwargs):
    if False:
        print('Hello World!')
    "\n    Don't display data. Used when you only are interested in the\n    return.\n    "
    return ''