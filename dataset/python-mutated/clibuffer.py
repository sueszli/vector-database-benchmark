from prompt_toolkit.enums import DEFAULT_BUFFER
from prompt_toolkit.filters import Condition
from prompt_toolkit.application import get_app
from .packages import special

def cli_is_multiline(mycli):
    if False:
        print('Hello World!')

    @Condition
    def cond():
        if False:
            while True:
                i = 10
        doc = get_app().layout.get_buffer_by_name(DEFAULT_BUFFER).document
        if not mycli.multi_line:
            return False
        else:
            return not _multiline_exception(doc.text)
    return cond

def _multiline_exception(text):
    if False:
        for i in range(10):
            print('nop')
    orig = text
    text = text.strip()
    if text.startswith('\\fs'):
        return orig.endswith('\n')
    return text.startswith('\\') or text.lower().startswith('delimiter') or text.endswith(special.get_current_delimiter()) or text.endswith('\\g') or text.endswith('\\G') or text.endswith('\\e') or text.endswith('\\clip') or (text == 'exit') or (text == 'quit') or (text == ':q') or (text == '')