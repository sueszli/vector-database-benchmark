"""
This module contains a mock resolver that returns mock functions for operations it cannot resolve.
"""
import functools
import logging
from connexion.resolver import Resolution, Resolver, ResolverError
logger = logging.getLogger(__name__)

class MockResolver(Resolver):

    def __init__(self, mock_all):
        if False:
            return 10
        super().__init__()
        self.mock_all = mock_all
        self._operation_id_counter = 1

    def resolve(self, operation):
        if False:
            for i in range(10):
                print('nop')
        '\n        Mock operation resolver\n\n        :type operation: connexion.operations.AbstractOperation\n        '
        operation_id = self.resolve_operation_id(operation)
        if not operation_id:
            operation_id = f'mock-{self._operation_id_counter}'
            self._operation_id_counter += 1
        mock_func = functools.partial(self.mock_operation, operation=operation)
        if self.mock_all:
            func = mock_func
        else:
            try:
                func = self.resolve_function_from_operation_id(operation_id)
                msg = "... Successfully resolved operationId '{}'! Mock is *not* used for this operation.".format(operation_id)
                logger.debug(msg)
            except ResolverError as resolution_error:
                logger.debug('... {}! Mock function is used for this operation.'.format(resolution_error.args[0].capitalize()))
                func = mock_func
        return Resolution(func, operation_id)

    def mock_operation(self, operation, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        (resp, code) = operation.example_response()
        if resp is not None:
            return (resp, code)
        return ('No example response was defined.', code)