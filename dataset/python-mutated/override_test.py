"""
Module for running arbitrary tests
"""
__virtualname__ = 'test'

def __virtual__():
    if False:
        return 10
    return __virtualname__

def recho(text):
    if False:
        i = 10
        return i + 15
    "\n    Return a reversed string\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' test.recho 'foo bar baz quo qux'\n    "
    return text[::-1]