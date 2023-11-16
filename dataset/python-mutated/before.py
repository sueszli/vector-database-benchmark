import typing
from tenacity import _utils
if typing.TYPE_CHECKING:
    import logging
    from tenacity import RetryCallState

def before_nothing(retry_state: 'RetryCallState') -> None:
    if False:
        i = 10
        return i + 15
    'Before call strategy that does nothing.'

def before_log(logger: 'logging.Logger', log_level: int) -> typing.Callable[['RetryCallState'], None]:
    if False:
        return 10
    'Before call strategy that logs to some logger the attempt.'

    def log_it(retry_state: 'RetryCallState') -> None:
        if False:
            print('Hello World!')
        if retry_state.fn is None:
            fn_name = '<unknown>'
        else:
            fn_name = _utils.get_callback_name(retry_state.fn)
        logger.log(log_level, f"Starting call to '{fn_name}', this is the {_utils.to_ordinal(retry_state.attempt_number)} time calling it.")
    return log_it