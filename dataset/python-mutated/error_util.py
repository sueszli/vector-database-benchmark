import streamlit as st
from streamlit import config
from streamlit.errors import UncaughtAppException
from streamlit.logger import get_logger
_LOGGER = get_logger(__name__)

def _print_rich_exception(e: BaseException):
    if False:
        return 10
    from rich import box, panel

    class ConfigurablePanel(panel.Panel):

        def __init__(self, renderable, box=box.Box('────\n    \n────\n    \n────\n────\n    \n────\n'), **kwargs):
            if False:
                print('Hello World!')
            super(ConfigurablePanel, self).__init__(renderable, box, **kwargs)
    from rich import traceback as rich_traceback
    rich_traceback.Panel = ConfigurablePanel
    from rich.console import Console
    console = Console(color_system='256', force_terminal=True, width=88, no_color=False, tab_size=8)
    import streamlit.runtime.scriptrunner.script_runner as script_runner
    console.print(rich_traceback.Traceback.from_exception(type(e), e, e.__traceback__, width=88, show_locals=False, max_frames=100, word_wrap=False, extra_lines=3, suppress=[script_runner]))

def handle_uncaught_app_exception(ex: BaseException) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Handle an exception that originated from a user app.\n\n    By default, we show exceptions directly in the browser. However,\n    if the user has disabled client error details, we display a generic\n    warning in the frontend instead.\n    '
    error_logged = False
    if config.get_option('logger.enableRich'):
        try:
            _print_rich_exception(ex)
            error_logged = True
        except Exception:
            error_logged = False
    if config.get_option('client.showErrorDetails'):
        if not error_logged:
            _LOGGER.warning('Uncaught app exception', exc_info=ex)
        st.exception(ex)
    else:
        if not error_logged:
            _LOGGER.error('Uncaught app exception', exc_info=ex)
        st.exception(UncaughtAppException(ex))