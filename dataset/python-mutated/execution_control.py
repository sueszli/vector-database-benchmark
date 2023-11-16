from typing import NoReturn
import streamlit as st
from streamlit.deprecation_util import make_deprecated_name_warning
from streamlit.logger import get_logger
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import RerunData, get_script_run_ctx
_LOGGER = get_logger(__name__)

@gather_metrics('stop')
def stop() -> NoReturn:
    if False:
        while True:
            i = 10
    "Stops execution immediately.\n\n    Streamlit will not run any statements after `st.stop()`.\n    We recommend rendering a message to explain why the script has stopped.\n\n    Example\n    -------\n    >>> import streamlit as st\n    >>>\n    >>> name = st.text_input('Name')\n    >>> if not name:\n    >>>   st.warning('Please input a name.')\n    >>>   st.stop()\n    >>> st.success('Thank you for inputting a name.')\n\n    "
    ctx = get_script_run_ctx()
    if ctx and ctx.script_requests:
        ctx.script_requests.request_stop()
        st.empty()

@gather_metrics('rerun')
def rerun() -> NoReturn:
    if False:
        i = 10
        return i + 15
    'Rerun the script immediately.\n\n    When `st.rerun()` is called, the script is halted - no more statements will\n    be run, and the script will be queued to re-run from the top.\n    '
    ctx = get_script_run_ctx()
    if ctx and ctx.script_requests:
        query_string = ctx.query_string
        page_script_hash = ctx.page_script_hash
        ctx.script_requests.request_rerun(RerunData(query_string=query_string, page_script_hash=page_script_hash))
        st.empty()

@gather_metrics('experimental_rerun')
def experimental_rerun() -> NoReturn:
    if False:
        return 10
    'Rerun the script immediately.\n\n    When `st.experimental_rerun()` is called, the script is halted - no\n    more statements will be run, and the script will be queued to re-run\n    from the top.\n    '
    msg = make_deprecated_name_warning('experimental_rerun', 'rerun', '2024-04-01')
    _LOGGER.warning(msg)
    rerun()