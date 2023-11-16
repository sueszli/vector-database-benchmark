import logging
import sys
from airbyte_cdk.utils.traced_exception import AirbyteTracedException

def init_uncaught_exception_handler(logger: logging.Logger) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Handles uncaught exceptions by emitting an AirbyteTraceMessage and making sure they are not\n    printed to the console without having secrets removed.\n    '

    def hook_fn(exception_type, exception_value, traceback_):
        if False:
            print('Hello World!')
        if issubclass(exception_type, KeyboardInterrupt):
            sys.__excepthook__(exception_type, exception_value, traceback_)
            return
        logger.fatal(exception_value, exc_info=exception_value)
        traced_exc = exception_value if issubclass(exception_type, AirbyteTracedException) else AirbyteTracedException.from_exception(exception_value)
        traced_exc.emit_message()
    sys.excepthook = hook_fn