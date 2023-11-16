import base64
import sys
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union
from kitty.fast_data_types import KeyEvent as WindowSystemKeyEvent
from kitty.fast_data_types import get_boss
from kitty.key_encoding import decode_key_event_as_window_system_key
from kitty.options.utils import parse_send_text_bytes
from .base import MATCH_TAB_OPTION, MATCH_WINDOW_OPTION, ArgsType, Boss, CmdGenerator, MatchError, PayloadGetType, PayloadType, RCOptions, RemoteCommand, ResponseType, Window
if TYPE_CHECKING:
    from kitty.cli_stub import SendTextRCOptions as CLIOptions

class Session:
    id: str
    window_ids: Set[int]

    def __init__(self, id: str):
        if False:
            print('Hello World!')
        self.id = id
        self.window_ids = set()
sessions_map: Dict[str, Session] = {}

class SessionAction:

    def __init__(self, sid: str):
        if False:
            return 10
        self.sid = sid

class ClearSession(SessionAction):

    def __call__(self, *a: Any) -> None:
        if False:
            print('Hello World!')
        s = sessions_map.pop(self.sid, None)
        if s is not None:
            boss = get_boss()
            for wid in s.window_ids:
                qw = boss.window_id_map.get(wid)
                if qw is not None:
                    qw.screen.render_unfocused_cursor = 0

class FocusChangedSession(SessionAction):

    def __call__(self, window: Window, focused: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        s = sessions_map.get(self.sid)
        if s is not None:
            boss = get_boss()
            val = int(focused)
            for wid in s.window_ids:
                qw = boss.window_id_map.get(wid)
                if qw is not None:
                    qw.screen.render_unfocused_cursor = val

class SendText(RemoteCommand):
    protocol_spec = __doc__ = '\n    data+/str: The data being sent. Can be either: text: followed by text or base64: followed by standard base64 encoded bytes\n    match/str: A string indicating the window to send text to\n    match_tab/str: A string indicating the tab to send text to\n    all/bool: A boolean indicating all windows should be matched.\n    exclude_active/bool: A boolean that prevents sending text to the active window\n    session_id/str: A string that identifies a "broadcast session"\n    '
    short_desc = 'Send arbitrary text to specified windows'
    desc = "Send arbitrary text to specified windows. The text follows Python escaping rules. So you can use :link:`escapes <https://www.gnu.org/software/bash/manual/html_node/ANSI_002dC-Quoting.html>` like :code:`'\\\\e'` to send control codes and :code:`'\\\\u21fa'` to send Unicode characters. Remember to use single-quotes otherwise the backslash is interpreted as a shell escape character. If you use the :option:`kitten @ send-text --match` option the text will be sent to all matched windows. By default, text is sent to only the currently active window."
    options_spec = MATCH_WINDOW_OPTION + '\n\n' + MATCH_TAB_OPTION.replace('--match -m', '--match-tab -t') + '\n\n--all\ntype=bool-set\nMatch all windows.\n\n\n--exclude-active\ntype=bool-set\nDo not send text to the active window, even if it is one of the matched windows.\n\n\n--stdin\ntype=bool-set\nRead the text to be sent from :italic:`stdin`. Note that in this case the text is sent as is,\nnot interpreted for escapes. If stdin is a terminal, you can press :kbd:`Ctrl+D` to end reading.\n\n\n--from-file\nPath to a file whose contents you wish to send. Note that in this case the file contents\nare sent as is, not interpreted for escapes.\n'
    args = RemoteCommand.Args(spec='[TEXT TO SEND]', json_field='data', special_parse='+session_id:parse_send_text(io_data, args)')

    def message_to_kitty(self, global_opts: RCOptions, opts: 'CLIOptions', args: ArgsType) -> PayloadType:
        if False:
            print('Hello World!')
        limit = 1024
        ret = {'match': opts.match, 'data': '', 'match_tab': opts.match_tab, 'all': opts.all, 'exclude_active': opts.exclude_active}

        def pipe() -> CmdGenerator:
            if False:
                for i in range(10):
                    print('nop')
            if sys.stdin.isatty():
                ret['exclude_active'] = True
                keep_going = True
                from kitty.utils import TTYIO
                with TTYIO(read_with_timeout=False) as tty:
                    while keep_going:
                        if not tty.wait_till_read_available():
                            break
                        data = tty.read(limit)
                        if not data:
                            break
                        decoded_data = data.decode('utf-8')
                        if '\x04' in decoded_data:
                            decoded_data = decoded_data[:decoded_data.index('\x04')]
                            keep_going = False
                        ret['data'] = f'text:{decoded_data}'
                        yield ret
            else:
                while True:
                    data = sys.stdin.buffer.read(limit)
                    if not data:
                        break
                    ret['data'] = f"base64:{base64.standard_b64encode(data).decode('ascii')}"
                    yield ret

        def chunks(text: str) -> CmdGenerator:
            if False:
                for i in range(10):
                    print('nop')
            data = parse_send_text_bytes(text)
            while data:
                b = base64.standard_b64encode(data[:limit]).decode('ascii')
                ret['data'] = f'base64:{b}'
                yield ret
                data = data[limit:]

        def file_pipe(path: str) -> CmdGenerator:
            if False:
                return 10
            with open(path, 'rb') as f:
                while True:
                    data = f.read(limit)
                    if not data:
                        break
                    ret['data'] = f"base64:{base64.standard_b64encode(data).decode('ascii')}"
                    yield ret
        sources = []
        if opts.stdin:
            sources.append(pipe())
        if opts.from_file:
            sources.append(file_pipe(opts.from_file))
        text = ' '.join(args)
        sources.append(chunks(text))

        def chain() -> CmdGenerator:
            if False:
                i = 10
                return i + 15
            for src in sources:
                yield from src
        return chain()

    def response_from_kitty(self, boss: Boss, window: Optional[Window], payload_get: PayloadGetType) -> ResponseType:
        if False:
            while True:
                i = 10
        sid = payload_get('session_id', '')
        if payload_get('all'):
            windows: List[Optional[Window]] = list(boss.all_windows)
        else:
            windows = [boss.active_window]
            match = payload_get('match')
            if match:
                windows = [w for w in self.windows_for_match_payload(boss, window, payload_get)]
                if not windows and (not sid):
                    raise MatchError(payload_get('match'))
            mt = payload_get('match_tab')
            if mt:
                windows = []
                tabs = self.tabs_for_match_payload(boss, window, payload_get)
                if not tabs and (not sid):
                    raise MatchError(payload_get('match_tab'), 'tabs')
                for tab in tabs:
                    if tab:
                        windows += tuple(tab)
        pdata: str = payload_get('data')
        (encoding, _, q) = pdata.partition(':')
        session = ''
        if encoding == 'text':
            data: Union[bytes, WindowSystemKeyEvent] = q.encode('utf-8')
        elif encoding == 'base64':
            data = base64.standard_b64decode(q)
        elif encoding == 'kitty-key':
            bdata = base64.standard_b64decode(q)
            candidate = decode_key_event_as_window_system_key(bdata.decode('ascii'))
            if candidate is None:
                raise ValueError(f'Could not decode window system key: {q}')
            data = candidate
        elif encoding == 'session':
            session = q
        else:
            raise TypeError(f'Invalid encoding for send-text data: {encoding}')
        exclude_active = payload_get('exclude_active')
        actual_windows = (w for w in windows if w is not None and (not exclude_active or w is not boss.active_window))

        def create_or_update_session() -> Session:
            if False:
                while True:
                    i = 10
            s = sessions_map.setdefault(sid, Session(sid))
            return s
        if session == 'end':
            s = create_or_update_session()
            for w in actual_windows:
                w.screen.render_unfocused_cursor = 0
                s.window_ids.discard(w.id)
            ClearSession(sid)()
        elif session == 'start':
            s = create_or_update_session()
            if window is not None:

                def is_ok(x: Any) -> bool:
                    if False:
                        while True:
                            i = 10
                    return not isinstance(x, SessionAction) or x.sid != sid
                window.actions_on_removal = list(filter(is_ok, window.actions_on_removal))
                window.actions_on_focus_change = list(filter(is_ok, window.actions_on_focus_change))
                window.actions_on_removal.append(ClearSession(sid))
                window.actions_on_focus_change.append(FocusChangedSession(sid))
            for w in actual_windows:
                w.screen.render_unfocused_cursor = 1
                s.window_ids.add(w.id)
        else:
            if sid:
                s = create_or_update_session()
            for w in actual_windows:
                if sid:
                    w.screen.render_unfocused_cursor = 1
                    s.window_ids.add(w.id)
                if isinstance(data, WindowSystemKeyEvent):
                    kdata = w.encoded_key(data)
                    if kdata:
                        w.write_to_child(kdata)
                else:
                    w.write_to_child(data)
        return None
send_text = SendText()