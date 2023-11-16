import typing
from tenacity import _utils
if typing.TYPE_CHECKING:
    import logging
    from tenacity import RetryCallState

def after_nothing(retry_state: 'RetryCallState') -> None:
    if False:
        print('Hello World!')
    'After call strategy that does nothing.'

def after_log(logger: 'logging.Logger', log_level: int, sec_format: str='%0.3f') -> typing.Callable[['RetryCallState'], None]:
    if False:
        i = 10
        return i + 15
    'After call strategy that logs to some logger the finished attempt.'

    def log_it(retry_state: 'RetryCallState') -> None:
        if False:
            print('Hello World!')
        if retry_state.fn is None:
            fn_name = '<unknown>'
        else:
            fn_name = _utils.get_callback_name(retry_state.fn)
        logger.log(log_level, f"Finished call to '{fn_name}' after {sec_format % retry_state.seconds_since_start}(s), this was the {_utils.to_ordinal(retry_state.attempt_number)} time calling it.")
    return log_it