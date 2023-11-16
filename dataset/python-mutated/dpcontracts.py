"""
-----------------------
hypothesis[dpcontracts]
-----------------------

This module provides tools for working with the :pypi:`dpcontracts` library,
because `combining contracts and property-based testing works really well
<https://hillelwayne.com/talks/beyond-unit-tests/>`_.

It requires ``dpcontracts >= 0.4``.
"""
from dpcontracts import PreconditionError
from hypothesis import reject
from hypothesis.errors import InvalidArgument
from hypothesis.internal.reflection import proxies

def fulfill(contract_func):
    if False:
        print('Hello World!')
    'Decorate ``contract_func`` to reject calls which violate preconditions,\n    and retry them with different arguments.\n\n    This is a convenience function for testing internal code that uses\n    :pypi:`dpcontracts`, to automatically filter out arguments that would be\n    rejected by the public interface before triggering a contract error.\n\n    This can be used as ``builds(fulfill(func), ...)`` or in the body of the\n    test e.g. ``assert fulfill(func)(*args)``.\n    '
    if not hasattr(contract_func, '__contract_wrapped_func__'):
        raise InvalidArgument(f'{contract_func.__name__} has no dpcontracts preconditions')

    @proxies(contract_func)
    def inner(*args, **kwargs):
        if False:
            while True:
                i = 10
        try:
            return contract_func(*args, **kwargs)
        except PreconditionError:
            reject()
    return inner