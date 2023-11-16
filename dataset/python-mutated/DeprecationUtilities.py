import logging

def check_deprecation(param_list):
    if False:
        print('Hello World!')
    "\n    Shows a deprecation warning message if the parameters\n    passed are not ``None``.\n\n    :param param_list:\n        A dictionary of parameters with their names mapped\n        to their values being checked for deprecation.\n\n    >>> from testfixtures import LogCapture\n    >>> from collections import OrderedDict\n    >>> param_list = OrderedDict([('foo', None),\n    ...                           ('bar', 'Random'),\n    ...                           ('baz', 1773)])\n    >>> with LogCapture() as capture:\n    ...     check_deprecation(param_list)\n    ...     print(capture)\n    root WARNING\n      bar parameter is deprecated\n    root WARNING\n      baz parameter is deprecated\n    "
    for (param_name, param_value) in param_list.items():
        if param_value is not None:
            logging.warning(f'{param_name} parameter is deprecated')