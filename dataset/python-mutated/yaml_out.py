"""
Display return data in YAML format
==================================

This outputter defaults to printing in YAML block mode for better readability.

CLI Example:

.. code-block:: bash

    salt '*' foo.bar --out=yaml

Example output:

CLI Example:

.. code-block:: python

    saltmine:
      foo:
        bar: baz
        dictionary:
          abc: 123
          def: 456
        list:
          - Hello
          - World
"""
import logging
import salt.utils.yaml
__virtualname__ = 'yaml'
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        print('Hello World!')
    return __virtualname__

def output(data, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Print out YAML using the block mode\n    '
    params = {}
    if 'output_indent' not in __opts__:
        params.update(default_flow_style=False)
    elif __opts__['output_indent'] >= 0:
        params.update(default_flow_style=False, indent=__opts__['output_indent'])
    else:
        params.update(default_flow_style=True, indent=0)
    try:
        return salt.utils.yaml.safe_dump(data, **params)
    except Exception as exc:
        import pprint
        log.exception('Exception %s encountered when trying to serialize %s', exc, pprint.pformat(data))