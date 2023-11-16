"""
The reference implementation of usage logger using the Python standard logging library.
"""
from inspect import Signature
import logging
from typing import Any, Optional

def get_logger() -> Any:
    if False:
        for i in range(10):
            print('nop')
    'An entry point of the plug-in and return the usage logger.'
    return PandasOnSparkUsageLogger()

def _format_signature(signature):
    if False:
        i = 10
        return i + 15
    return '({})'.format(', '.join([p.name for p in signature.parameters.values()])) if signature is not None else ''

class PandasOnSparkUsageLogger:
    """
    The reference implementation of usage logger.

    The usage logger needs to provide the following methods:

        - log_success(self, class_name, name, duration, signature=None)
        - log_failure(self, class_name, name, ex, duration, signature=None)
        - log_missing(self, class_name, name, is_deprecated=False, signature=None)
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.logger = logging.getLogger('pyspark.pandas.usage_logger')

    def log_success(self, class_name: str, name: str, duration: float, signature: Optional[Signature]=None) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Log the function or property call is successfully finished.\n\n        :param class_name: the target class name\n        :param name: the target function or property name\n        :param duration: the duration to finish the function or property call\n        :param signature: the signature if the target is a function, else None\n        '
        if self.logger.isEnabledFor(logging.INFO):
            msg = 'A {function} `{class_name}.{name}{signature}` was successfully finished after {duration:.3f} ms.'.format(class_name=class_name, name=name, signature=_format_signature(signature), duration=duration * 1000, function='function' if signature is not None else 'property')
            self.logger.info(msg)

    def log_failure(self, class_name: str, name: str, ex: Exception, duration: float, signature: Optional[Signature]=None) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Log the function or property call failed.\n\n        :param class_name: the target class name\n        :param name: the target function or property name\n        :param ex: the exception causing the failure\n        :param duration: the duration until the function or property call fails\n        :param signature: the signature if the target is a function, else None\n        '
        if self.logger.isEnabledFor(logging.WARNING):
            msg = 'A {function} `{class_name}.{name}{signature}` was failed after {duration:.3f} ms: {msg}'.format(class_name=class_name, name=name, signature=_format_signature(signature), msg=str(ex), duration=duration * 1000, function='function' if signature is not None else 'property')
            self.logger.warning(msg)

    def log_missing(self, class_name: str, name: str, is_deprecated: bool=False, signature: Optional[Signature]=None) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Log the missing or deprecated function or property is called.\n\n        :param class_name: the target class name\n        :param name: the target function or property name\n        :param is_deprecated: True if the function or property is marked as deprecated\n        :param signature: the original function signature if the target is a function, else None\n        '
        if self.logger.isEnabledFor(logging.INFO):
            msg = 'A {deprecated} {function} `{class_name}.{name}{signature}` was called.'.format(class_name=class_name, name=name, signature=_format_signature(signature), function='function' if signature is not None else 'property', deprecated='deprecated' if is_deprecated else 'missing')
            self.logger.info(msg)