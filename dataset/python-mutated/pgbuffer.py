import logging
from prompt_toolkit.enums import DEFAULT_BUFFER
from prompt_toolkit.filters import Condition
from prompt_toolkit.application import get_app
from .packages.parseutils.utils import is_open_quote
_logger = logging.getLogger(__name__)

def _is_complete(sql):
    if False:
        print('Hello World!')
    return sql.endswith(';') and (not is_open_quote(sql))
'\nReturns True if the buffer contents should be handled (i.e. the query/command\nexecuted) immediately. This is necessary as we use prompt_toolkit in multiline\nmode, which by default will insert new lines on Enter.\n'

def safe_multi_line_mode(pgcli):
    if False:
        return 10

    @Condition
    def cond():
        if False:
            print('Hello World!')
        _logger.debug('Multi-line mode state: "%s" / "%s"', pgcli.multi_line, pgcli.multiline_mode)
        return pgcli.multi_line and pgcli.multiline_mode == 'safe'
    return cond

def buffer_should_be_handled(pgcli):
    if False:
        while True:
            i = 10

    @Condition
    def cond():
        if False:
            while True:
                i = 10
        if not pgcli.multi_line:
            _logger.debug('Not in multi-line mode. Handle the buffer.')
            return True
        if pgcli.multiline_mode == 'safe':
            _logger.debug("Multi-line mode is set to 'safe'. Do NOT handle the buffer.")
            return False
        doc = get_app().layout.get_buffer_by_name(DEFAULT_BUFFER).document
        text = doc.text.strip()
        return text.startswith('\\') or text.endswith('\\e') or text.endswith('\\G') or _is_complete(text) or (text == 'exit') or (text == 'quit') or (text == ':q') or (text == '')
    return cond