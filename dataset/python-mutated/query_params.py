import urllib.parse as parse
from typing import Any, Dict, List
from streamlit import util
from streamlit.errors import StreamlitAPIException
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import get_script_run_ctx
EMBED_QUERY_PARAM = 'embed'
EMBED_OPTIONS_QUERY_PARAM = 'embed_options'
EMBED_QUERY_PARAMS_KEYS = [EMBED_QUERY_PARAM, EMBED_OPTIONS_QUERY_PARAM]

@gather_metrics('experimental_get_query_params')
def get_query_params() -> Dict[str, List[str]]:
    if False:
        return 10
    'Return the query parameters that is currently showing in the browser\'s URL bar.\n\n    Returns\n    -------\n    dict\n      The current query parameters as a dict. "Query parameters" are the part of the URL that comes\n      after the first "?".\n\n    Example\n    -------\n    Let\'s say the user\'s web browser is at\n    `http://localhost:8501/?show_map=True&selected=asia&selected=america`.\n    Then, you can get the query parameters using the following:\n\n    >>> import streamlit as st\n    >>>\n    >>> st.experimental_get_query_params()\n    {"show_map": ["True"], "selected": ["asia", "america"]}\n\n    Note that the values in the returned dict are *always* lists. This is\n    because we internally use Python\'s urllib.parse.parse_qs(), which behaves\n    this way. And this behavior makes sense when you consider that every item\n    in a query string is potentially a 1-element array.\n\n    '
    ctx = get_script_run_ctx()
    if ctx is None:
        return {}
    return util.exclude_key_query_params(parse.parse_qs(ctx.query_string, keep_blank_values=True), keys_to_exclude=EMBED_QUERY_PARAMS_KEYS)

@gather_metrics('experimental_set_query_params')
def set_query_params(**query_params: Any) -> None:
    if False:
        return 10
    'Set the query parameters that are shown in the browser\'s URL bar.\n\n    .. warning::\n        Query param `embed` cannot be set using this method.\n\n    Parameters\n    ----------\n    **query_params : dict\n        The query parameters to set, as key-value pairs.\n\n    Example\n    -------\n\n    To point the user\'s web browser to something like\n    "http://localhost:8501/?show_map=True&selected=asia&selected=america",\n    you would do the following:\n\n    >>> import streamlit as st\n    >>>\n    >>> st.experimental_set_query_params(\n    ...     show_map=True,\n    ...     selected=["asia", "america"],\n    ... )\n\n    '
    ctx = get_script_run_ctx()
    if ctx is None:
        return
    msg = ForwardMsg()
    msg.page_info_changed.query_string = _ensure_no_embed_params(query_params, ctx.query_string)
    ctx.query_string = msg.page_info_changed.query_string
    ctx.enqueue(msg)

def _ensure_no_embed_params(query_params: Dict[str, List[str]], query_string: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Ensures there are no embed params set (raises StreamlitAPIException) if there is a try,\n    also makes sure old param values in query_string are preserved. Returns query_string : str.'
    query_params_without_embed = util.exclude_key_query_params(query_params, keys_to_exclude=EMBED_QUERY_PARAMS_KEYS)
    if query_params != query_params_without_embed:
        raise StreamlitAPIException('Query param embed and embed_options (case-insensitive) cannot be set using set_query_params method.')
    all_current_params = parse.parse_qs(query_string, keep_blank_values=True)
    current_embed_params = parse.urlencode({EMBED_QUERY_PARAM: [param for param in util.extract_key_query_params(all_current_params, param_key=EMBED_QUERY_PARAM)], EMBED_OPTIONS_QUERY_PARAM: [param for param in util.extract_key_query_params(all_current_params, param_key=EMBED_OPTIONS_QUERY_PARAM)]}, doseq=True)
    query_string = parse.urlencode(query_params, doseq=True)
    if query_string:
        separator = '&' if current_embed_params else ''
        return separator.join([query_string, current_embed_params])
    return current_embed_params