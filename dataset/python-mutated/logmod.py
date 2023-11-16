"""
On-demand logging
=================

.. versionadded:: 2017.7.0

The sole purpose of this module is logging messages in the (proxy) minion.
It comes very handy when debugging complex Jinja templates, for example:

.. code-block:: jinja

    {%- for var in range(10) %}
      {%- do salt["log.info"](var) -%}
    {%- endfor %}

CLI Example:

.. code-block:: bash

    salt '*' log.error "Please don't do that, this module is not for CLI use!"
"""
import logging
log = logging.getLogger(__name__)
__virtualname__ = 'log'
__proxyenabled__ = ['*']

def __virtual__():
    if False:
        while True:
            i = 10
    return __virtualname__

def debug(message):
    if False:
        return 10
    '\n    Log message at level DEBUG.\n    '
    log.debug(message)
    return True

def info(message):
    if False:
        print('Hello World!')
    '\n    Log message at level INFO.\n    '
    log.info(message)
    return True

def warning(message):
    if False:
        i = 10
        return i + 15
    '\n    Log message at level WARNING.\n    '
    log.warning(message)
    return True

def error(message):
    if False:
        print('Hello World!')
    '\n    Log message at level ERROR.\n    '
    log.error(message)
    return True

def critical(message):
    if False:
        i = 10
        return i + 15
    '\n    Log message at level CRITICAL.\n    '
    log.critical(message)
    return True

def exception(message):
    if False:
        return 10
    '\n    Log message at level EXCEPTION.\n    '
    log.exception(message)
    return True