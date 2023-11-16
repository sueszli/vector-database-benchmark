"""
Contains method decorator which can be used to convert common exceptions into click exceptions
which will end execution gracefully
"""
from functools import wraps
from typing import Any, Callable, Dict, Optional
from botocore.exceptions import BotoCoreError, ClientError, NoRegionError
from samcli.commands._utils.parameterized_option import parameterized_option
from samcli.commands.exceptions import AWSServiceClientError, RegionError, SDKError

class CustomExceptionHandler:

    def __init__(self, custom_exception_handler_mapping):
        if False:
            while True:
                i = 10
        self.custom_exception_handler_mapping = custom_exception_handler_mapping

    def get_handler(self, exception_type: type):
        if False:
            return 10
        return self.custom_exception_handler_mapping.get(exception_type)

class GenericExceptionHandler:

    def __init__(self, generic_exception_handler_mapping):
        if False:
            print('Hello World!')
        self.generic_exception_handler_mapping = generic_exception_handler_mapping

    def get_handler(self, exception_type: type):
        if False:
            i = 10
            return i + 15
        for (common_exception, common_exception_handler) in self.generic_exception_handler_mapping.items():
            if issubclass(exception_type, common_exception):
                return common_exception_handler

def _handle_no_region_error(ex: NoRegionError) -> None:
    if False:
        i = 10
        return i + 15
    raise RegionError('No region information found. Please provide --region parameter or configure default region settings.')

def _handle_client_errors(ex: ClientError) -> None:
    if False:
        i = 10
        return i + 15
    additional_exception_message = '\n\nFor more information please visit: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html'
    raise AWSServiceClientError(str(ex) + additional_exception_message) from ex

def _catch_all_boto_errors(ex: BotoCoreError) -> None:
    if False:
        for i in range(10):
            print('nop')
    raise SDKError(str(ex)) from ex
CUSTOM_EXCEPTION_HANDLER_MAPPING: Dict[Any, Callable] = {NoRegionError: _handle_no_region_error, ClientError: _handle_client_errors}
GENERIC_EXCEPTION_HANDLER_MAPPING: Dict[Any, Callable] = {BotoCoreError: _catch_all_boto_errors}

@parameterized_option
def command_exception_handler(f, additional_mapping: Optional[Dict[Any, Callable[[Any], None]]]=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    This function returns a wrapped function definition, which handles configured exceptions gracefully\n    '

    def decorator_command_exception_handler(func):
        if False:
            while True:
                i = 10

        @wraps(func)
        def wrapper_command_exception_handler(*args, **kwargs):
            if False:
                print('Hello World!')
            try:
                return func(*args, **kwargs)
            except Exception as ex:
                exception_type = type(ex)
                exception_handler = (additional_mapping or {}).get(exception_type)
                if exception_handler:
                    exception_handler(ex)
                for exception_handler in [CustomExceptionHandler(CUSTOM_EXCEPTION_HANDLER_MAPPING), GenericExceptionHandler(GENERIC_EXCEPTION_HANDLER_MAPPING)]:
                    handler = exception_handler.get_handler(exception_type)
                    if handler:
                        handler(ex)
                raise ex
        return wrapper_command_exception_handler
    return decorator_command_exception_handler(f)