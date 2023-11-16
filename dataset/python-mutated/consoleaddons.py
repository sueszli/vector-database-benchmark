import csv
import logging
from collections.abc import Sequence
import mitmproxy.types
from mitmproxy import command
from mitmproxy import command_lexer
from mitmproxy import contentviews
from mitmproxy import ctx
from mitmproxy import dns
from mitmproxy import exceptions
from mitmproxy import flow
from mitmproxy import http
from mitmproxy import log
from mitmproxy import tcp
from mitmproxy import udp
from mitmproxy.log import ALERT
from mitmproxy.tools.console import keymap
from mitmproxy.tools.console import overlay
from mitmproxy.tools.console import signals
from mitmproxy.utils import strutils
logger = logging.getLogger(__name__)
console_palettes = ['lowlight', 'lowdark', 'light', 'dark', 'solarized_light', 'solarized_dark']
view_orders = ['time', 'method', 'url', 'size']
console_layouts = ['single', 'vertical', 'horizontal']
console_flowlist_layout = ['default', 'table', 'list']

class ConsoleAddon:
    """
    An addon that exposes console-specific commands, and hooks into required
    events.
    """

    def __init__(self, master):
        if False:
            while True:
                i = 10
        self.master = master
        self.started = False

    def load(self, loader):
        if False:
            return 10
        loader.add_option('console_default_contentview', str, 'auto', 'The default content view mode.', choices=[i.name.lower() for i in contentviews.views])
        loader.add_option('console_eventlog_verbosity', str, 'info', 'EventLog verbosity.', choices=log.LogLevels)
        loader.add_option('console_layout', str, 'single', 'Console layout.', choices=sorted(console_layouts))
        loader.add_option('console_layout_headers', bool, True, 'Show layout component headers')
        loader.add_option('console_focus_follow', bool, False, 'Focus follows new flows.')
        loader.add_option('console_palette', str, 'solarized_dark', 'Color palette.', choices=sorted(console_palettes))
        loader.add_option('console_palette_transparent', bool, True, 'Set transparent background for palette.')
        loader.add_option('console_mouse', bool, True, 'Console mouse interaction.')
        loader.add_option('console_flowlist_layout', str, 'default', 'Set the flowlist layout', choices=sorted(console_flowlist_layout))
        loader.add_option('console_strip_trailing_newlines', bool, False, 'Strip trailing newlines from edited request/response bodies.')

    @command.command('console.layout.options')
    def layout_options(self) -> Sequence[str]:
        if False:
            print('Hello World!')
        '\n        Returns the available options for the console_layout option.\n        '
        return ['single', 'vertical', 'horizontal']

    @command.command('console.layout.cycle')
    def layout_cycle(self) -> None:
        if False:
            print('Hello World!')
        '\n        Cycle through the console layout options.\n        '
        opts = self.layout_options()
        off = self.layout_options().index(ctx.options.console_layout)
        ctx.options.update(console_layout=opts[(off + 1) % len(opts)])

    @command.command('console.panes.next')
    def panes_next(self) -> None:
        if False:
            print('Hello World!')
        '\n        Go to the next layout pane.\n        '
        self.master.window.switch()

    @command.command('console.panes.prev')
    def panes_prev(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Go to the previous layout pane.\n        '
        return self.panes_next()

    @command.command('console.options.reset.focus')
    def options_reset_current(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Reset the current option in the options editor.\n        '
        fv = self.master.window.current('options')
        if not fv:
            raise exceptions.CommandError('Not viewing options.')
        self.master.commands.call_strings('options.reset.one', [fv.current_name()])

    @command.command('console.nav.start')
    def nav_start(self) -> None:
        if False:
            return 10
        '\n        Go to the start of a list or scrollable.\n        '
        self.master.inject_key('m_start')

    @command.command('console.nav.end')
    def nav_end(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Go to the end of a list or scrollable.\n        '
        self.master.inject_key('m_end')

    @command.command('console.nav.next')
    def nav_next(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Go to the next navigatable item.\n        '
        self.master.inject_key('m_next')

    @command.command('console.nav.select')
    def nav_select(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Select a navigable item for viewing or editing.\n        '
        self.master.inject_key('m_select')

    @command.command('console.nav.up')
    def nav_up(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Go up.\n        '
        self.master.inject_key('up')

    @command.command('console.nav.down')
    def nav_down(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Go down.\n        '
        self.master.inject_key('down')

    @command.command('console.nav.pageup')
    def nav_pageup(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Go up.\n        '
        self.master.inject_key('page up')

    @command.command('console.nav.pagedown')
    def nav_pagedown(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Go down.\n        '
        self.master.inject_key('page down')

    @command.command('console.nav.left')
    def nav_left(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Go left.\n        '
        self.master.inject_key('left')

    @command.command('console.nav.right')
    def nav_right(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Go right.\n        '
        self.master.inject_key('right')

    @command.command('console.choose')
    def console_choose(self, prompt: str, choices: Sequence[str], cmd: mitmproxy.types.Cmd, *args: mitmproxy.types.CmdArgs) -> None:
        if False:
            while True:
                i = 10
        '\n        Prompt the user to choose from a specified list of strings, then\n        invoke another command with all occurrences of {choice} replaced by\n        the choice the user made.\n        '

        def callback(opt):
            if False:
                for i in range(10):
                    print('nop')
            repl = [arg.replace('{choice}', opt) for arg in args]
            try:
                self.master.commands.call_strings(cmd, repl)
            except exceptions.CommandError as e:
                logger.error(str(e))
        self.master.overlay(overlay.Chooser(self.master, prompt, choices, '', callback))

    @command.command('console.choose.cmd')
    def console_choose_cmd(self, prompt: str, choicecmd: mitmproxy.types.Cmd, subcmd: mitmproxy.types.Cmd, *args: mitmproxy.types.CmdArgs) -> None:
        if False:
            while True:
                i = 10
        '\n        Prompt the user to choose from a list of strings returned by a\n        command, then invoke another command with all occurrences of {choice}\n        replaced by the choice the user made.\n        '
        choices = ctx.master.commands.execute(choicecmd)

        def callback(opt):
            if False:
                while True:
                    i = 10
            repl = [arg.replace('{choice}', opt) for arg in args]
            try:
                self.master.commands.call_strings(subcmd, repl)
            except exceptions.CommandError as e:
                logger.error(str(e))
        self.master.overlay(overlay.Chooser(self.master, prompt, choices, '', callback))

    @command.command('console.command')
    def console_command(self, *command_str: str) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Prompt the user to edit a command with a (possibly empty) starting value.\n        '
        quoted = ' '.join((command_lexer.quote(x) for x in command_str))
        if quoted:
            quoted += ' '
        signals.status_prompt_command.send(partial=quoted)

    @command.command('console.command.set')
    def console_command_set(self, option_name: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Prompt the user to set an option.\n        '
        option_value = getattr(self.master.options, option_name, None) or ''
        set_command = f'set {option_name} {option_value!r}'
        cursor = len(set_command) - 1
        signals.status_prompt_command.send(partial=set_command, cursor=cursor)

    @command.command('console.view.keybindings')
    def view_keybindings(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'View the commands list.'
        self.master.switch_view('keybindings')

    @command.command('console.view.commands')
    def view_commands(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'View the commands list.'
        self.master.switch_view('commands')

    @command.command('console.view.options')
    def view_options(self) -> None:
        if False:
            print('Hello World!')
        'View the options editor.'
        self.master.switch_view('options')

    @command.command('console.view.eventlog')
    def view_eventlog(self) -> None:
        if False:
            return 10
        'View the event log.'
        self.master.switch_view('eventlog')

    @command.command('console.view.help')
    def view_help(self) -> None:
        if False:
            return 10
        'View help.'
        self.master.switch_view('help')

    @command.command('console.view.flow')
    def view_flow(self, flow: flow.Flow) -> None:
        if False:
            for i in range(10):
                print('nop')
        'View a flow.'
        if isinstance(flow, (http.HTTPFlow, tcp.TCPFlow, udp.UDPFlow, dns.DNSFlow)):
            self.master.switch_view('flowview')
        else:
            logger.warning(f'No detail view for {type(flow).__name__}.')

    @command.command('console.exit')
    def exit(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Exit mitmproxy.'
        self.master.shutdown()

    @command.command('console.view.pop')
    def view_pop(self) -> None:
        if False:
            print('Hello World!')
        '\n        Pop a view off the console stack. At the top level, this prompts the\n        user to exit mitmproxy.\n        '
        signals.pop_view_state.send()

    @command.command('console.bodyview')
    @command.argument('part', type=mitmproxy.types.Choice('console.bodyview.options'))
    def bodyview(self, flow: flow.Flow, part: str) -> None:
        if False:
            print('Hello World!')
        '\n        Spawn an external viewer for a flow request or response body based\n        on the detected MIME type. We use the mailcap system to find the\n        correct viewer, and fall back to the programs in $PAGER or $EDITOR\n        if necessary.\n        '
        fpart = getattr(flow, part, None)
        if not fpart:
            raise exceptions.CommandError('Part must be either request or response, not %s.' % part)
        t = fpart.headers.get('content-type')
        content = fpart.get_content(strict=False)
        if not content:
            raise exceptions.CommandError('No content to view.')
        self.master.spawn_external_viewer(content, t)

    @command.command('console.bodyview.options')
    def bodyview_options(self) -> Sequence[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Possible parts for console.bodyview.\n        '
        return ['request', 'response']

    @command.command('console.edit.focus.options')
    def edit_focus_options(self) -> Sequence[str]:
        if False:
            return 10
        '\n        Possible components for console.edit.focus.\n        '
        flow = self.master.view.focus.flow
        focus_options = []
        if isinstance(flow, tcp.TCPFlow):
            focus_options = ['tcp-message']
        elif isinstance(flow, udp.UDPFlow):
            focus_options = ['udp-message']
        elif isinstance(flow, http.HTTPFlow):
            focus_options = ['cookies', 'urlencoded form', 'multipart form', 'path', 'method', 'query', 'reason', 'request-headers', 'response-headers', 'request-body', 'response-body', 'status_code', 'set-cookies', 'url']
        elif isinstance(flow, dns.DNSFlow):
            raise exceptions.CommandError('Cannot edit DNS flows yet, please submit a patch.')
        return focus_options

    @command.command('console.edit.focus')
    @command.argument('flow_part', type=mitmproxy.types.Choice('console.edit.focus.options'))
    def edit_focus(self, flow_part: str) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Edit a component of the currently focused flow.\n        '
        flow = self.master.view.focus.flow
        if not flow:
            raise exceptions.CommandError('No flow selected.')
        flow.backup()
        require_dummy_response = flow_part in ('response-headers', 'response-body', 'set-cookies') and flow.response is None
        if require_dummy_response:
            flow.response = http.Response.make()
        if flow_part == 'cookies':
            self.master.switch_view('edit_focus_cookies')
        elif flow_part == 'urlencoded form':
            self.master.switch_view('edit_focus_urlencoded_form')
        elif flow_part == 'multipart form':
            self.master.switch_view('edit_focus_multipart_form')
        elif flow_part == 'path':
            self.master.switch_view('edit_focus_path')
        elif flow_part == 'query':
            self.master.switch_view('edit_focus_query')
        elif flow_part == 'request-headers':
            self.master.switch_view('edit_focus_request_headers')
        elif flow_part == 'response-headers':
            self.master.switch_view('edit_focus_response_headers')
        elif flow_part in ('request-body', 'response-body'):
            if flow_part == 'request-body':
                message = flow.request
            else:
                message = flow.response
            c = self.master.spawn_editor(message.get_content(strict=False) or b'')
            if self.master.options.console_strip_trailing_newlines:
                message.content = c.rstrip(b'\n')
            else:
                message.content = c
        elif flow_part == 'set-cookies':
            self.master.switch_view('edit_focus_setcookies')
        elif flow_part == 'url':
            url = flow.request.url.encode()
            edited_url = self.master.spawn_editor(url)
            url = edited_url.rstrip(b'\n')
            flow.request.url = url.decode()
        elif flow_part in ['method', 'status_code', 'reason']:
            self.master.commands.call_strings('console.command', ['flow.set', '@focus', flow_part])
        elif flow_part in ['tcp-message', 'udp-message']:
            message = flow.messages[-1]
            c = self.master.spawn_editor(message.content or b'')
            message.content = c.rstrip(b'\n')

    def _grideditor(self):
        if False:
            print('Hello World!')
        gewidget = self.master.window.current('grideditor')
        if not gewidget:
            raise exceptions.CommandError('Not in a grideditor.')
        return gewidget.key_responder()

    @command.command('console.grideditor.add')
    def grideditor_add(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Add a row after the cursor.\n        '
        self._grideditor().cmd_add()

    @command.command('console.grideditor.insert')
    def grideditor_insert(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Insert a row before the cursor.\n        '
        self._grideditor().cmd_insert()

    @command.command('console.grideditor.delete')
    def grideditor_delete(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Delete row\n        '
        self._grideditor().cmd_delete()

    @command.command('console.grideditor.load')
    def grideditor_load(self, path: mitmproxy.types.Path) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Read a file into the currrent cell.\n        '
        self._grideditor().cmd_read_file(path)

    @command.command('console.grideditor.load_escaped')
    def grideditor_load_escaped(self, path: mitmproxy.types.Path) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Read a file containing a Python-style escaped string into the\n        currrent cell.\n        '
        self._grideditor().cmd_read_file_escaped(path)

    @command.command('console.grideditor.save')
    def grideditor_save(self, path: mitmproxy.types.Path) -> None:
        if False:
            return 10
        '\n        Save data to file as a CSV.\n        '
        rows = self._grideditor().value
        try:
            with open(path, 'w', newline='', encoding='utf8') as fp:
                writer = csv.writer(fp)
                for row in rows:
                    writer.writerow([strutils.always_str(x) or '' for x in row])
            logger.log(ALERT, 'Saved %s rows as CSV.' % len(rows))
        except OSError as e:
            logger.error(str(e))

    @command.command('console.grideditor.editor')
    def grideditor_editor(self) -> None:
        if False:
            return 10
        '\n        Spawn an external editor on the current cell.\n        '
        self._grideditor().cmd_spawn_editor()

    @command.command('console.flowview.mode.set')
    @command.argument('mode', type=mitmproxy.types.Choice('console.flowview.mode.options'))
    def flowview_mode_set(self, mode: str) -> None:
        if False:
            return 10
        '\n        Set the display mode for the current flow view.\n        '
        fv = self.master.window.current_window('flowview')
        if not fv:
            raise exceptions.CommandError('Not viewing a flow.')
        idx = fv.body.tab_offset
        if mode not in [i.name.lower() for i in contentviews.views]:
            raise exceptions.CommandError('Invalid flowview mode.')
        try:
            self.master.commands.call_strings('view.settings.setval', ['@focus', f'flowview_mode_{idx}', mode])
        except exceptions.CommandError as e:
            logger.error(str(e))

    @command.command('console.flowview.mode.options')
    def flowview_mode_options(self) -> Sequence[str]:
        if False:
            print('Hello World!')
        '\n        Returns the valid options for the flowview mode.\n        '
        return [i.name.lower() for i in contentviews.views]

    @command.command('console.flowview.mode')
    def flowview_mode(self) -> str:
        if False:
            print('Hello World!')
        '\n        Get the display mode for the current flow view.\n        '
        fv = self.master.window.current_window('flowview')
        if not fv:
            raise exceptions.CommandError('Not viewing a flow.')
        idx = fv.body.tab_offset
        return self.master.commands.call_strings('view.settings.getval', ['@focus', f'flowview_mode_{idx}', self.master.options.console_default_contentview])

    @command.command('console.key.contexts')
    def key_contexts(self) -> Sequence[str]:
        if False:
            i = 10
            return i + 15
        '\n        The available contexts for key binding.\n        '
        return list(sorted(keymap.Contexts))

    @command.command('console.key.bind')
    def key_bind(self, contexts: Sequence[str], key: str, cmd: mitmproxy.types.Cmd, *args: mitmproxy.types.CmdArgs) -> None:
        if False:
            print('Hello World!')
        '\n        Bind a shortcut key.\n        '
        try:
            self.master.keymap.add(key, cmd + ' ' + ' '.join(args), contexts, '')
        except ValueError as v:
            raise exceptions.CommandError(v)

    @command.command('console.key.unbind')
    def key_unbind(self, contexts: Sequence[str], key: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Un-bind a shortcut key.\n        '
        try:
            self.master.keymap.remove(key, contexts)
        except ValueError as v:
            raise exceptions.CommandError(v)

    def _keyfocus(self):
        if False:
            for i in range(10):
                print('nop')
        kwidget = self.master.window.current('keybindings')
        if not kwidget:
            raise exceptions.CommandError('Not viewing key bindings.')
        f = kwidget.get_focused_binding()
        if not f:
            raise exceptions.CommandError('No key binding focused')
        return f

    @command.command('console.key.unbind.focus')
    def key_unbind_focus(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Un-bind the shortcut key currently focused in the key binding viewer.\n        '
        b = self._keyfocus()
        try:
            self.master.keymap.remove(b.key, b.contexts)
        except ValueError as v:
            raise exceptions.CommandError(v)

    @command.command('console.key.execute.focus')
    def key_execute_focus(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Execute the currently focused key binding.\n        '
        b = self._keyfocus()
        self.console_command(b.command)

    @command.command('console.key.edit.focus')
    def key_edit_focus(self) -> None:
        if False:
            print('Hello World!')
        '\n        Execute the currently focused key binding.\n        '
        b = self._keyfocus()
        self.console_command('console.key.bind', ','.join(b.contexts), b.key, b.command)

    def running(self):
        if False:
            print('Hello World!')
        self.started = True

    def update(self, flows) -> None:
        if False:
            while True:
                i = 10
        if not flows:
            signals.update_settings.send()
        for f in flows:
            signals.flow_change.send(flow=f)