"""
Utility functions for the rest_sample
"""
__proxyenabled__ = ['rest_sample']

def fix_outage():
    if False:
        i = 10
        return i + 15
    '\n    "Fix" the outage\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'rest-sample-proxy\' rest_sample.fix_outage\n\n    '
    return __proxy__['rest_sample.fix_outage']()

def get_test_string():
    if False:
        i = 10
        return i + 15
    "\n    Helper function to test cross-calling to the __proxy__ dunder.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'rest-sample-proxy' rest_sample.get_test_string\n    "
    return __proxy__['rest_sample.test_from_state']()