import typing
from tenacity import _utils
if typing.TYPE_CHECKING:
    import logging
    from tenacity import RetryCallState

def before_sleep_nothing(retry_state: 'RetryCallState') -> None:
    if False:
        return 10
    'Before call strategy that does nothing.'

def before_sleep_log(logger: 'logging.Logger', log_level: int, exc_info: bool=False) -> typing.Callable[['RetryCallState'], None]:
    if False:
        while True:
            i = 10
    'Before call strategy that logs to some logger the attempt.'

    def log_it(retry_state: 'RetryCallState') -> None:
        if False:
            while True:
                i = 10
        local_exc_info: BaseException | bool | None
        if retry_state.outcome is None:
            raise RuntimeError('log_it() called before outcome was set')
        if retry_state.next_action is None:
            raise RuntimeError('log_it() called before next_action was set')
        if retry_state.outcome.failed:
            ex = retry_state.outcome.exception()
            (verb, value) = ('raised', f'{ex.__class__.__name__}: {ex}')
            if exc_info:
                local_exc_info = retry_state.outcome.exception()
            else:
                local_exc_info = False
        else:
            (verb, value) = ('returned', retry_state.outcome.result())
            local_exc_info = False
        if retry_state.fn is None:
            fn_name = '<unknown>'
        else:
            fn_name = _utils.get_callback_name(retry_state.fn)
        logger.log(log_level, f'Retrying {fn_name} in {retry_state.next_action.sleep} seconds as it {verb} {value}.', exc_info=local_exc_info)
    return log_it