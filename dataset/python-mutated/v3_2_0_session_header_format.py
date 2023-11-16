from typing import Any, Type, List, Dict, TYPE_CHECKING
if TYPE_CHECKING:
    from httpie.sessions import Session
OLD_HEADER_STORE_WARNING = 'Outdated layout detected for the current session. Please consider updating it,\nin order to use the latest features regarding the header layout.\n\nFor fixing the current session:\n\n    $ httpie cli sessions upgrade {hostname} {session_id}\n'
OLD_HEADER_STORE_WARNING_FOR_NAMED_SESSIONS = '\nFor fixing all named sessions:\n\n    $ httpie cli sessions upgrade-all\n'
OLD_HEADER_STORE_LINK = '\nSee $INSERT_LINK for more information.'

def pre_process(session: 'Session', headers: Any) -> List[Dict[str, Any]]:
    if False:
        while True:
            i = 10
    'Serialize the headers into a unified form and issue a warning if\n    the session file is using the old layout.'
    is_old_style = isinstance(headers, dict)
    if is_old_style:
        normalized_headers = list(headers.items())
    else:
        normalized_headers = [(item['name'], item['value']) for item in headers]
    if is_old_style:
        warning = OLD_HEADER_STORE_WARNING.format(hostname=session.bound_host, session_id=session.session_id)
        if not session.is_anonymous:
            warning += OLD_HEADER_STORE_WARNING_FOR_NAMED_SESSIONS
        warning += OLD_HEADER_STORE_LINK
        session.warn_legacy_usage(warning)
    return normalized_headers

def post_process(normalized_headers: List[Dict[str, Any]], *, original_type: Type[Any]) -> Any:
    if False:
        while True:
            i = 10
    'Deserialize given header store into the original form it was\n    used in.'
    if issubclass(original_type, dict):
        return {item['name']: item['value'] for item in normalized_headers}
    else:
        return normalized_headers

def fix_layout(session: 'Session', *args, **kwargs) -> None:
    if False:
        return 10
    from httpie.sessions import materialize_headers
    if not isinstance(session['headers'], dict):
        return None
    session['headers'] = materialize_headers(session['headers'])