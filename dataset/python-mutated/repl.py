import re
import textwrap
import yaml
import os
import sys
import traceback
import signal
import frontmatter
import pyperclip
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import NestedCompleter, PathCompleter
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.styles import Style
import prompt_toolkit.document as document
import lwe.core.constants as constants
import lwe.core.util as util
from lwe.core.config import Config
from lwe.core.logger import Logger
from lwe.core.error import NoInputError
from lwe.core.editor import file_editor, pipe_editor
document._FIND_WORD_RE = re.compile('([a-zA-Z0-9-' + constants.COMMAND_LEADER + ']+|[^a-zA-Z0-9_\\.\\s]+)')

class Repl:
    """
    A shell interpreter that serves as a front end to the backend classes
    """
    intro = 'Provide a prompt, or type %shelp or ? to list commands.' % constants.COMMAND_LEADER
    prompt = '> '
    prompt_prefix = ''
    doc_header = 'Documented commands type %shelp [command without %s] (e.g. /help ask) for detailed help' % (constants.COMMAND_LEADER, constants.COMMAND_LEADER)
    prompt_number = 0
    message_map = {}
    logfile = None

    def __init__(self, config=None):
        if False:
            return 10
        self.initialize_repl(config)
        self.history = self.get_shell_history()
        self.style = self.get_styles()
        self.prompt_session = PromptSession(history=self.history, style=self.style)
        self._setup_signal_handlers()

    def initialize_repl(self, config=None):
        if False:
            print('Hello World!')
        self.config = config or Config()
        self.log = Logger(self.__class__.__name__, self.config)
        self.debug = self.config.get('log.console.level').lower() == 'debug'
        self._set_logging()

    def reload_repl(self):
        if False:
            return 10
        util.print_status_message(True, 'Reloading configuration...')
        self.config.load_from_file()
        self.initialize_repl(self.config)
        self.backend.initialize_backend(self.config)
        self.setup()

    def terminate_stream(self, _signal, _frame):
        if False:
            for i in range(10):
                print('nop')
        self.backend.terminate_stream(_signal, _frame)

    def catch_ctrl_c(self, signum, _frame):
        if False:
            return 10
        self.log.debug(f'Ctrl-c hit: {signum}')
        sig = util.is_windows and signal.SIGBREAK or signal.SIGUSR1
        os.kill(os.getpid(), sig)

    def _setup_signal_handlers(self):
        if False:
            while True:
                i = 10
        sig = util.is_windows and signal.SIGBREAK or signal.SIGUSR1
        signal.signal(sig, self.terminate_stream)

    def exec_prompt_pre(self, _command, _arg):
        if False:
            for i in range(10):
                print('nop')
        pass

    def configure_shell_commands(self):
        if False:
            return 10
        self.commands = util.introspect_commands(__class__)

    def get_plugin_commands(self):
        if False:
            print('Hello World!')
        commands = []
        for plugin in self.plugins.values():
            plugin_commands = util.introspect_commands(plugin.__class__)
            commands.extend(plugin_commands)
        return commands

    def configure_commands(self):
        if False:
            for i in range(10):
                print('nop')
        self.commands.extend(self.get_plugin_commands())
        self.dashed_commands = [util.underscore_to_dash(command) for command in self.commands]
        self.dashed_commands.sort()
        self.all_commands = self.dashed_commands + ['help']
        self.all_commands.sort()

    def get_custom_shell_completions(self):
        if False:
            i = 10
            return i + 15
        return {}

    def get_plugin_shell_completions(self, completions):
        if False:
            return 10
        for plugin in self.plugins.values():
            plugin_completions = plugin.get_shell_completions(self.base_shell_completions)
            if plugin_completions:
                completions = util.merge_dicts(completions, plugin_completions)
        return completions

    def set_base_shell_completions(self):
        if False:
            while True:
                i = 10
        commands_with_leader = {}
        for command in self.all_commands:
            commands_with_leader[util.command_with_leader(command)] = None
        config_args = sorted(['edit', 'files', 'profile', 'runtime'] + list(self.config.get().keys()) + self.config.properties)
        commands_with_leader[util.command_with_leader('config')] = util.list_to_completion_hash(config_args)
        commands_with_leader[util.command_with_leader('help')] = util.list_to_completion_hash(self.dashed_commands)
        for command in ['file', 'log']:
            commands_with_leader[util.command_with_leader(command)] = PathCompleter()
        template_completions = util.list_to_completion_hash(self.backend.template_manager.templates)
        commands_with_leader[util.command_with_leader('template')] = {c: template_completions for c in self.get_command_actions('template', dashed=True)}
        self.base_shell_completions = commands_with_leader

    def rebuild_completions(self):
        if False:
            return 10
        self.set_base_shell_completions()
        completions = util.merge_dicts(self.base_shell_completions, self.get_custom_shell_completions())
        completions = self.get_plugin_shell_completions(completions)
        self.command_completer = NestedCompleter.from_nested_dict(completions)

    def get_shell_history(self):
        if False:
            return 10
        history_file = self.config.get('shell.history_file')
        if history_file:
            return FileHistory(history_file)

    def get_styles(self):
        if False:
            return 10
        style = Style.from_dict({'prompt': 'bold', 'completion-menu.completion': 'bg:#008888 #ffffff', 'completion-menu.completion.current': 'bg:#00aaaa #000000', 'scrollbar.background': 'bg:#88aaaa', 'scrollbar.button': 'bg:#222222'})
        return style

    def run_template(self, template_name, substitutions=None):
        if False:
            for i in range(10):
                print('nop')
        (success, response, user_message) = self.backend.run_template_setup(template_name, substitutions)
        if not success:
            return (success, response, user_message)
        (message, overrides) = response
        print('')
        print(message)
        self.log.info('Running template')
        response = self.default(message, **overrides)
        return response

    def edit_run_template(self, template_content, suffix='md'):
        if False:
            print('Hello World!')
        (template_name, filepath) = self.backend.template_manager.make_temp_template(template_content, suffix)
        file_editor(filepath)
        response = self.run_template(template_name)
        self.backend.template_manager.remove_temp_template(template_name)
        return response

    def collect_template_variable_values(self, template_name, variables=None):
        if False:
            i = 10
            return i + 15
        variables = variables or []
        substitutions = {}
        builtin_variables = self.backend.template_manager.template_builtin_variables()
        user_variables = list(set([v for v in variables if v not in builtin_variables]))
        if user_variables:
            self.command_template(template_name)
            util.print_markdown('##### Enter variables:\n')
            self.log.debug(f'Collecting variable values for: {template_name}')
            for variable in user_variables:
                substitutions[variable] = input(f'    {variable}: ').strip()
                self.log.debug(f'Collected variable {variable} for template {template_name}: {substitutions[variable]}')
        substitutions = util.merge_dicts(substitutions, self.backend.template_manager.process_template_builtin_variables(template_name, variables))
        return substitutions

    def get_command_help_brief(self, command):
        if False:
            print('Hello World!')
        help_brief = '    %s%s' % (constants.COMMAND_LEADER, command)
        help_doc = self.get_command_help(command)
        if help_doc:
            first_line = next(filter(lambda x: x.strip(), help_doc.splitlines()), '')
            help_brief += ': %s' % first_line
        return help_brief

    def get_command_help(self, command):
        if False:
            return 10
        command = util.dash_to_underscore(command)
        if command in self.commands:
            (method, _obj) = self.get_command_method(command)
            doc = method.__doc__
            if doc:
                doc = doc.replace('{COMMAND}', '%s%s' % (constants.COMMAND_LEADER, util.underscore_to_dash(command)))
                for sub in constants.HELP_TOKEN_VARIABLE_SUBSTITUTIONS:
                    try:
                        const_value = getattr(constants, sub)
                    except AttributeError as err:
                        raise AttributeError(f'{sub!r} in HELP_TOKEN_VARIABLE_SUBSTITUTIONS is not a valid constant') from err
                    doc = doc.replace('{%s}' % sub, str(const_value))
                return textwrap.dedent(doc)

    def help_commands(self):
        if False:
            for i in range(10):
                print('nop')
        print('')
        util.print_markdown(f'#### {self.doc_header}')
        print('')
        for command in self.dashed_commands:
            print(self.get_command_help_brief(command))
        print('')

    def help(self, command=''):
        if False:
            i = 10
            return i + 15
        if command:
            help_doc = self.get_command_help(command)
            if help_doc:
                print(help_doc)
            else:
                print("\nNo help for '%s'\n\nAvailable commands: %s" % (command, ', '.join(self.dashed_commands)))
        else:
            self.help_commands()

    def _set_logging(self):
        if False:
            return 10
        if self.config.get('chat.log.enabled'):
            log_file = self.config.get('chat.log.filepath')
            if log_file:
                if not self._open_log(log_file):
                    print('\nERROR: could not open log file: %s' % log_file)
                    sys.exit(0)

    def _set_prompt(self, prefix=''):
        if False:
            i = 10
            return i + 15
        self.prompt = f'{self.prompt_prefix}{self.prompt_number}> '

    def _set_prompt_prefix(self, prefix=''):
        if False:
            i = 10
            return i + 15
        self.prompt_prefix = prefix

    def _update_message_map(self):
        if False:
            return 10
        self.prompt_number += 1
        self.message_map[self.prompt_number] = (self.backend.conversation_id,)
        self._set_prompt()

    def _write_log(self, prompt, response):
        if False:
            while True:
                i = 10
        if self.logfile is not None:
            self.logfile.write(f'{self.prompt_number}> {prompt}\n\n{response}\n\n')
            self._write_log_context()

    def _write_log_context(self):
        if False:
            print('Hello World!')
        if self.logfile is not None:
            self.logfile.write(f'## context {self.backend.conversation_id}\n')
            self.logfile.flush()

    def build_shell_user_prefix(self):
        if False:
            i = 10
            return i + 15
        return ''

    def set_user_prompt(self, user=None):
        if False:
            for i in range(10):
                print('nop')
        prefix = self.build_shell_user_prefix()
        self._set_prompt_prefix(prefix)
        self._set_prompt()

    def configure_plugins(self):
        if False:
            for i in range(10):
                print('nop')
        self.plugin_manager = self.backend.plugin_manager
        self.plugins = self.plugin_manager.get_plugins()
        for plugin in self.plugins.values():
            plugin.set_shell(self)

    def configure_backend(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def launch_backend(self, interactive=True):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def setup(self):
        if False:
            return 10
        self.configure_backend()
        self.configure_plugins()
        self.stream = self.config.get('shell.streaming')
        self.backend.template_manager.load_templates()
        self.configure_shell_commands()
        self.configure_commands()
        self.rebuild_completions()
        self._update_message_map()

    def cleanup(self):
        if False:
            i = 10
            return i + 15
        pass

    def _fetch_history(self, limit=constants.DEFAULT_HISTORY_LIMIT, offset=0):
        if False:
            print('Hello World!')
        util.print_markdown('* Fetching conversation history...')
        (success, history, message) = self.backend.get_history(limit=limit, offset=offset)
        return (success, history, message)

    def _set_title(self, title, conversation=None):
        if False:
            print('Hello World!')
        util.print_markdown('* Setting title...')
        (success, _, message) = self.backend.set_title(title, conversation['id'])
        if success:
            return (success, conversation, f"Title set to: {conversation['title']}")
        else:
            return (success, conversation, message)

    def _delete_conversation(self, id, label=None):
        if False:
            while True:
                i = 10
        if id == self.backend.conversation_id:
            self._delete_current_conversation()
        else:
            label = label or id
            util.print_markdown('* Deleting conversation: %s' % label)
            (success, conversation, message) = self.backend.delete_conversation(id)
            if success:
                util.print_status_message(True, f'Deleted conversation: {label}')
            else:
                util.print_status_message(False, f'Failed to deleted conversation: {label}, {message}')

    def _delete_current_conversation(self):
        if False:
            return 10
        util.print_markdown('* Deleting current conversation')
        (success, conversation, message) = self.backend.delete_conversation()
        if success:
            util.print_status_message(True, 'Deleted current conversation')
            self.command_new(None)
        else:
            util.print_status_message(False, 'Failed to delete current conversation')

    def dispatch_command_action(self, command, args):
        if False:
            return 10
        try:
            (action, *action_args) = args.split()
        except ValueError:
            return (False, None, f'Action required for {constants.COMMAND_LEADER}{command} command')
        try:
            (method, klass) = self.get_command_action_method(command, action)
        except AttributeError:
            return (False, None, f'Invalid action {action} for {constants.COMMAND_LEADER}{command} command')
        action_args.insert(0, klass)
        return method(*action_args)

    def get_command_actions(self, command, dashed=False):
        if False:
            print('Hello World!')
        command_actions = util.introspect_command_actions(self.__class__, command)
        for plugin in self.plugins.values():
            plugin_command_actions = util.introspect_command_actions(plugin.__class__, command)
            command_actions.extend(plugin_command_actions)
        if dashed:
            command_actions = list(map(util.underscore_to_dash, command_actions))
        return command_actions

    def command_stream(self, _):
        if False:
            print('Hello World!')
        '\n        Toggle streaming mode\n\n        Streaming mode: streams the raw response (no markdown rendering)\n        Non-streaming mode: Returns full response at completion (markdown rendering supported).\n\n        Examples:\n            {COMMAND}\n        '
        self.stream = not self.stream
        util.print_markdown(f"* Streaming mode is now {('enabled' if self.stream else 'disabled')}.")

    def command_new(self, _):
        if False:
            for i in range(10):
                print('nop')
        '\n        Start a new conversation\n\n        Examples:\n            {COMMAND}\n        '
        self.backend.new_conversation()
        util.print_markdown('* New conversation started.')
        self._update_message_map()
        self._write_log_context()

    def command_delete(self, arg):
        if False:
            return 10
        '\n        Delete one or more conversations\n\n        Can delete by conversation ID, history ID, or current conversation.\n\n        Arguments:\n            history_id : The history ID\n\n        Arguments can be mixed and matched as in the examples below.\n\n        Examples:\n            Current conversation: {COMMAND}\n            Delete one: {COMMAND} 3\n            Multiple IDs: {COMMAND} 1,5\n            Ranges: {COMMAND} 1-5\n            Complex: {COMMAND} 1,3-5\n        '
        if arg:
            result = util.parse_conversation_ids(arg)
            if isinstance(result, list):
                (success, conversations, message) = self._fetch_history()
                if success:
                    history_list = list(conversations.values())
                    for item in result:
                        if isinstance(item, str) and len(item) == 36:
                            self._delete_conversation(item)
                        elif item <= len(history_list):
                            conversation = history_list[item - 1]
                            self._delete_conversation(conversation['id'], conversation['title'])
                        else:
                            util.print_status_message(False, f'Cannont delete history item {item}, does not exist')
                else:
                    return (success, conversations, message)
            else:
                return (False, None, result)
        else:
            self._delete_current_conversation()

    def command_copy(self, _):
        if False:
            while True:
                i = 10
        '\n        Copy last conversation message to clipboard\n\n        Examples:\n            {COMMAND}\n        '
        clipboard = self.backend.message_clipboard
        if clipboard:
            pyperclip.copy(clipboard)
            return (True, clipboard, 'Copied last message to clipboard')
        return (False, None, 'No message to copy')

    def command_history(self, arg):
        if False:
            print('Hello World!')
        '\n        Show recent conversation history\n\n        Arguments;\n            limit: limit the number of messages to show (default {DEFAULT_HISTORY_LIMIT})\n            offset: offset the list of messages by this number\n\n        Examples:\n            {COMMAND}\n            {COMMAND} 10\n            {COMMAND} 10 5\n        '
        limit = constants.DEFAULT_HISTORY_LIMIT
        offset = 0
        if arg:
            args = arg.split(' ')
            if len(args) > 2:
                util.print_markdown('* Invalid number of arguments, must be limit [offest]')
                return
            else:
                try:
                    limit = int(args[0])
                except ValueError:
                    util.print_markdown('* Invalid limit, must be an integer')
                    return
                if len(args) == 2:
                    try:
                        offset = int(args[1])
                    except ValueError:
                        util.print_markdown('* Invalid offset, must be an integer')
                        return
        (success, history, message) = self._fetch_history(limit=limit, offset=offset)
        if success:
            history_list = [h for h in history.values()]
            util.print_markdown('## Recent history:\n\n%s' % '\n'.join(['1. %s: %s (%s)%s' % (h['created_time'].strftime('%Y-%m-%d %H:%M'), h['title'] or constants.NO_TITLE_TEXT, h['id'], f' {constants.ACTIVE_ITEM_INDICATOR}' if h['id'] == self.backend.conversation_id else '') for h in history_list]))
        else:
            return (success, history, message)

    def command_title(self, arg):
        if False:
            for i in range(10):
                print('nop')
        '\n        Show or set title\n\n        Arguments:\n            title: title of the current conversation\n            ...or...\n            history_id: history ID of conversation\n\n        Examples:\n            Get current conversation title: {COMMAND}\n            Set current conversation title: {COMMAND} new title\n            Set conversation title using history ID: {COMMAND} 1\n        '
        if arg:
            id = None
            try:
                id = int(arg)
            except Exception:
                pass
            kwargs = {}
            if id:
                kwargs['limit'] = id
            (success, conversations, message) = self._fetch_history(**kwargs)
            if success:
                history_list = list(conversations.values())
                conversation = None
                if id:
                    if id <= len(history_list):
                        conversation = history_list[id - 1]
                    else:
                        return (False, conversations, 'Cannot set title on history item %d, does not exist' % id)
                    new_title = input("Enter new title for '%s': " % conversation['title'] or constants.NO_TITLE_TEXT)
                elif self.backend.conversation_id:
                    if self.backend.conversation_id in conversations:
                        conversation = conversations[self.backend.conversation_id]
                    else:
                        (success, conversation_data, message) = self.backend.get_conversation(self.backend.conversation_id)
                        if not success:
                            return (success, conversation_data, message)
                        conversation = conversation_data['conversation']
                    new_title = arg
                else:
                    return (False, None, 'Current conversation has no title, you must send information first')
                conversation['title'] = new_title
                return self._set_title(new_title, conversation)
            else:
                return (success, conversations, message)
        elif self.backend.conversation_id:
            (success, conversation_data, message) = self.backend.get_conversation()
            if success:
                util.print_markdown('* Title: %s' % conversation_data['conversation']['title'] or constants.NO_TITLE_TEXT)
            else:
                return (success, conversation_data, message)
        else:
            return (False, None, 'Current conversation has no title, you must send information first')

    def command_chat(self, arg):
        if False:
            print('Hello World!')
        '\n        Retrieve chat content\n\n        Arguments:\n            history_id: The history ID\n            With no arguments, show content of the current conversation.\n\n        Examples:\n            Current conversation: {COMMAND}\n            Older conversation: {COMMAND} 2\n        '
        conversation = None
        conversation_id = None
        title = None
        if arg:
            if len(arg) == 36:
                conversation_id = arg
                title = arg
            else:
                id = None
                try:
                    id = int(arg)
                except Exception:
                    return (False, None, f'Invalid chat history item {arg}, must be in integer')
                kwargs = {}
                if id:
                    kwargs['limit'] = id
                (success, conversations, message) = self._fetch_history(**kwargs)
                if success:
                    history_list = list(conversations.values())
                    if id <= len(history_list):
                        conversation = history_list[id - 1]
                        title = conversation['title'] or constants.NO_TITLE_TEXT
                    else:
                        return (False, conversations, f'Cannot retrieve chat content on history item {id}, does not exist')
                else:
                    return (success, conversations, message)
        elif self.backend.conversation_id:
            conversation_id = self.backend.conversation_id
        else:
            return (False, None, 'Current conversation is empty, you must send information first')
        if conversation:
            conversation_id = conversation['id']
        (success, conversation_data, message) = self.backend.get_conversation(conversation_id)
        if success:
            if conversation_data:
                messages = self.backend.conversation_data_to_messages(conversation_data)
                if title:
                    util.print_markdown(f'## {title}')
                conversation_parts = util.conversation_from_messages(messages)
                for part in conversation_parts:
                    print('\n')
                    style = 'bold red3' if part['role'] == 'user' else 'bold green3'
                    util.print_markdown(part['display_role'], style=style)
                    util.print_markdown(part['message'])
            else:
                return (False, conversation_data, 'Could not load chat content')
        else:
            return (success, conversation_data, message)

    def command_switch(self, arg):
        if False:
            for i in range(10):
                print('nop')
        '\n        Switch to chat\n\n        Arguments:\n            history_id: The history ID of the conversation\n\n        Examples:\n            {COMMAND} 2\n        '
        conversation = None
        conversation_id = None
        title = None
        if arg:
            if len(arg) == 36:
                conversation_id = arg
                title = arg
            else:
                id = None
                try:
                    id = int(arg)
                except Exception:
                    return (False, None, f'Invalid chat history item {arg}, must be in integer')
                kwargs = {}
                if id:
                    kwargs['limit'] = id
                (success, conversations, message) = self._fetch_history(**kwargs)
                if success:
                    history_list = list(conversations.values())
                    if id <= len(history_list):
                        conversation = history_list[id - 1]
                        title = conversation['title'] or constants.NO_TITLE_TEXT
                    else:
                        return (False, conversations, f'Cannot retrieve chat content on history item {id}, does not exist')
                else:
                    return (success, conversations, message)
        else:
            return (False, None, 'Argument required, ID or history ID')
        if conversation:
            conversation_id = conversation['id']
        if conversation_id == self.backend.conversation_id:
            return (True, conversation, f'You are already in chat: {title}')
        (success, conversation_data, message) = self.backend.get_conversation(conversation_id)
        if success:
            if conversation_data:
                self.backend.switch_to_conversation(conversation_id)
                self._update_message_map()
                self._write_log_context()
                if title:
                    util.print_markdown(f'### Switched to: {title}')
            else:
                return (False, conversation_data, 'Could not switch to chat')
        else:
            return (success, conversation_data, message)

    def command_ask(self, input):
        if False:
            i = 10
            return i + 15
        "\n        Ask a question\n\n        It is purely optional.\n\n        Examples:\n            {COMMAND} what is 6+6 (is the same as 'what is 6+6')\n        "
        return self.default(input)

    def default(self, input, request_overrides=None):
        if False:
            return 10
        signal.signal(signal.SIGINT, self.catch_ctrl_c)
        if not input:
            return
        request_overrides = request_overrides or {}
        if self.stream:
            request_overrides['print_stream'] = True
            print('')
            (success, response, user_message) = self.backend.ask_stream(input, request_overrides=request_overrides)
            print('\n')
            if not success:
                return (success, response, user_message)
        else:
            (success, response, user_message) = self.backend.ask(input, request_overrides=request_overrides)
            if success:
                print('')
                util.print_markdown(response)
            else:
                return (success, response, user_message)
        self._write_log(input, response)
        self._update_message_map()

    def command_read(self, _):
        if False:
            print('Hello World!')
        '\n        Begin reading multi-line input\n\n        Allows for entering more complex multi-line input prior to sending it.\n\n        Examples:\n            {COMMAND}\n        '
        ctrl_sequence = '^z' if util.is_windows else '^d'
        util.print_markdown(f'* Reading prompt, hit {ctrl_sequence} when done, or write line with /end.')
        prompt = ''
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line == '':
                print('')
            if line == '/end':
                break
            prompt += line + '\n'
        self.default(prompt)

    def command_editor(self, args):
        if False:
            i = 10
            return i + 15
        '\n        Open an editor for entering a command\n\n        When the editor is closed, the content is sent.\n\n        Arguments:\n            default_text: The default text to open the editor with\n\n        Examples:\n            {COMMAND}\n            {COMMAND} some text to start with\n        '
        output = pipe_editor(args, suffix='md')
        print(output)
        self.default(output)

    def command_file(self, arg):
        if False:
            i = 10
            return i + 15
        '\n        Send a prompt read from the named file\n\n        Arguments:\n            file_name: The name of the file to read from\n\n        Examples:\n            {COMMAND} myprompt.txt\n        '
        try:
            fileprompt = open(arg, encoding='utf-8').read()
        except Exception:
            util.print_markdown(f'Failed to read file {arg!r}')
            return
        self.default(fileprompt)

    def _open_log(self, filename):
        if False:
            for i in range(10):
                print('nop')
        try:
            if os.path.isabs(filename):
                self.logfile = open(filename, 'a', encoding='utf-8')
            else:
                self.logfile = open(os.path.join(os.getcwd(), filename), 'a', encoding='utf-8')
        except Exception:
            util.print_markdown(f'Failed to open log file {filename!r}.')
            return False
        return True

    def command_log(self, arg):
        if False:
            for i in range(10):
                print('nop')
        '\n        Enable/disable logging to a file\n\n        Arguments:\n            file_name: The name of the file to write to\n\n        Examples:\n            Log to file: {COMMAND} mylog.txt\n            Disable logging: {COMMAND}\n        '
        if arg:
            if self._open_log(arg):
                util.print_markdown(f'* Logging enabled, appending to {arg!r}.')
        else:
            self.logfile = None
            util.print_markdown('* Logging is now disabled.')

    def command_model(self, arg):
        if False:
            print('Hello World!')
        '\n        View or set attributes on the current LLM model\n\n        Arguments:\n            path: The attribute path to view or set\n            value: The value to set the attribute to\n            With no arguments, view current set model attributes\n\n        Examples:\n            {COMMAND}\n            {COMMAND} temperature\n            {COMMAND} temperature 1.1\n        '
        if arg:
            try:
                (path, value, *rest) = arg.split()
                if rest:
                    return (False, arg, "Too many parameters, should be 'path value'")
                if path == self.backend.provider.model_property_name:
                    (success, value, user_message) = self.backend.set_model(value)
                else:
                    (success, value, user_message) = self.backend.provider.set_customization_value(path, value)
                if success:
                    model_name = value.get(self.backend.provider.model_property_name, 'unknown')
                    self.backend.model = model_name
                return (success, value, user_message)
            except ValueError:
                (success, value, user_message) = self.backend.provider.get_customization_value(arg)
                if success:
                    if isinstance(value, dict):
                        util.print_markdown('\n```yaml\n%s\n```' % yaml.dump(value, default_flow_style=False))
                    else:
                        util.print_markdown(f'* {arg} = {value}')
                else:
                    return (success, value, user_message)
        else:
            customizations = self.backend.provider.get_customizations()
            model_name = customizations.pop(self.backend.provider.model_property_name, 'unknown')
            provider_name = self.backend.provider.display_name()
            customizations_data = '\n\n```yaml\n%s\n```' % yaml.dump(customizations, default_flow_style=False) if customizations else ''
            util.print_markdown('## Provider: %s, model: %s%s' % (provider_name, model_name, customizations_data))

    def command_templates(self, arg):
        if False:
            for i in range(10):
                print('nop')
        "\n        List available templates\n\n        Templates are pre-configured text content that can be customized before sending a message to the model.\n\n        They are located in the 'templates' directory in the following locations:\n\n            - The main configuration directory\n            - The profile configuration directory\n\n        See {COMMAND_LEADER}config for current locations.\n\n        Arguments:\n            filter_string: Optional. If provided, only templates with a name or description containing the filter string will be shown.\n\n        Examples:\n            {COMMAND}\n            {COMMAND} filterstring\n        "
        self.backend.template_manager.make_temp_template_dir()
        self.backend.template_manager.load_templates()
        self.rebuild_completions()
        templates = []
        for template_name in self.backend.template_manager.templates:
            content = f'* **{template_name}**'
            (template, _) = self.backend.template_manager.get_template_and_variables(template_name)
            try:
                source = frontmatter.load(template.filename)
            except yaml.parser.ParserError:
                util.print_status_message(False, f'Failed to parse template: {template_name}')
                continue
            if 'description' in source.metadata:
                content += f": *{source.metadata['description']}*"
            if not arg or arg.lower() in content.lower():
                templates.append(content)
        util.print_markdown('## Templates:\n\n%s' % '\n'.join(sorted(templates)))

    def command_template(self, args):
        if False:
            for i in range(10):
                print('nop')
        "\n        Run actions on available templates\n\n        Templates are pre-configured text content that can be customized before sending a message to the model.\n\n        'Running' a template sends its content (after variable substitutions) to the model as your input.\n\n        Available actions:\n            * copy: Copy a template\n            * delete: Delete a template\n            * edit: Open or create a template for editing\n            * edit-run: Open the template in an editor, then run it on editor save and close.\n            * prompt-edit-run: Collect values for template variables, then open in an editor, then run it on editor save and close\n            * prompt-run: Collect values for template variables, then run it\n            * run: Run a template\n            * show: Show a template\n\n        Arguments:\n            template_name: Required. The name of the template.\n\n            For copy, a new template name is also required.\n\n        Examples:\n            * /template copy mytemplate.md mytemplate_copy.md\n            * /template delete mytemplate.md\n            * /template edit mytemplate.md\n            * /template edit-run mytemplate.md\n            * /template prompt-edit-run mytemplate.md\n            * /template prompt-run mytemplate.md\n            * /template run mytemplate.md\n            * /template show mytemplate.md\n        "
        return self.dispatch_command_action('template', args)

    def action_template_show(self, template_name):
        if False:
            while True:
                i = 10
        '\n        Display a template.\n\n        :param template_name: The name of the template.\n        :type template_name: str\n        '
        (success, source, user_message) = self.backend.template_manager.get_template_source(template_name)
        if not success:
            return (success, source, user_message)
        util.print_markdown(f'\n## Template {template_name!r}')
        if source.metadata:
            util.print_markdown('\n```yaml\n%s\n```' % yaml.dump(source.metadata, default_flow_style=False))
        util.print_markdown(f'\n\n{source.content}')

    def action_template_edit(self, template_name):
        if False:
            print('Hello World!')
        '\n        Create a new template, or edit an existing template.\n\n        :param template_name: The name of the template.\n        :type template_name: str\n        '
        (success, filepath, user_message) = self.backend.template_manager.get_template_editable_filepath(template_name)
        if not success:
            return (success, filepath, user_message)
        file_editor(filepath)
        self.backend.template_manager.load_templates()
        self.rebuild_completions()

    def action_template_copy(self, *template_names):
        if False:
            i = 10
            return i + 15
        '\n        Copies an existing template and saves it as a new template.\n\n        :param template_names: The names of the old and new templates.\n        :type template_names: tuple\n        :return: Success status, new file path, and user message.\n        :rtype: tuple\n        '
        try:
            (old_name, new_name) = template_names
        except ValueError:
            return (False, template_names, 'Old and new template name required')
        (success, new_filepath, user_message) = self.backend.template_manager.copy_template(old_name, new_name)
        if not success:
            return (success, new_filepath, user_message)
        self.rebuild_completions()
        return (True, new_filepath, f'Copied {old_name} to {new_filepath}')

    def action_template_delete(self, template_name):
        if False:
            i = 10
            return i + 15
        '\n        Deletes an existing template.\n\n        :param template_name: The name of the template to delete.\n        :type template_name: str\n        '
        (success, filename, user_message) = self.backend.template_manager.template_can_delete(template_name)
        if not success:
            return (success, filename, user_message)
        confirmation = input(f'Are you sure you want to delete template {template_name}? [y/N] ').strip()
        if confirmation.lower() in ['yes', 'y']:
            return self.backend.template_manager.template_delete(filename)
        else:
            return (False, template_name, 'Deletion aborted')

    def action_template_run(self, template_name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Run a template.\n\n        :param template_name: The name of the template.\n        :type template_name: str\n        '
        (success, response, user_message) = self.backend.template_manager.get_template_variables_substitutions(template_name)
        if not success:
            return (success, template_name, user_message)
        (_template, variables, substitutions) = response
        return self.run_template(template_name, substitutions)

    def action_template_prompt_run(self, template_name):
        if False:
            i = 10
            return i + 15
        '\n        Prompt for template variable values, then run.\n\n        :param template_name: The name of the template.\n        :type template_name: str\n        '
        response = self.action_template_show(template_name)
        if response:
            return response
        (success, response, user_message) = self.backend.template_manager.get_template_variables_substitutions(template_name)
        if not success:
            return (success, template_name, user_message)
        (_template, variables, _substitutions) = response
        substitutions = self.collect_template_variable_values(template_name, variables)
        return self.run_template(template_name, substitutions)

    def action_template_edit_run(self, template_name):
        if False:
            return 10
        '\n        Open a template for final editing, then run it.\n\n        :param template_name: The name of the template.\n        :type template_name: str\n        '
        (success, template_content, user_message) = self.backend.template_manager.render_template(template_name)
        if not success:
            return (success, template_name, user_message)
        return self.edit_run_template(template_content)

    def action_template_prompt_edit_run(self, template_name):
        if False:
            print('Hello World!')
        '\n        Prompts for a value for each variable in the template, sustitutes the values\n        in the template, opens an editor for final edits, and sends the final content\n        to the model as your input.\n\n        :param template_name: The name of the template.\n        :type template_name: str\n        '
        response = self.action_template_show(template_name)
        if response:
            return response
        (success, response, user_message) = self.backend.template_manager.get_template_variables_substitutions(template_name)
        if not success:
            return (success, template_name, user_message)
        (template, variables, _substitutions) = response
        substitutions = self.collect_template_variable_values(template_name, variables)
        template_content = template.render(**substitutions)
        return self.edit_run_template(template_content)

    def command_plugins(self, arg):
        if False:
            i = 10
            return i + 15
        '\n        List installed plugins\n\n        Plugins are enabled by adding their name to the list of enabled plugins\n        in the profile configuration.\n\n        Arguments:\n            filter_string: Optional. String to filter plugins by. Name and description are matched.\n\n        Examples:\n            {COMMAND}\n            {COMMAND} shell\n        '
        plugin_list = []
        provider_plugin_list = []
        for plugin in self.plugins.values():
            content = f'* {plugin.name}'
            if plugin.description:
                content += f': *{plugin.description}*'
            if not arg or arg.lower() in content.lower():
                if plugin.plugin_type == 'provider':
                    provider_plugin_list.append(content)
                else:
                    plugin_list.append(content)
        plugin_list.sort()
        provider_plugin_list.sort()
        util.print_markdown('## Enabled command plugins:\n\n%s' % '\n'.join(plugin_list))
        util.print_markdown('## Enabled provider plugins:\n\n%s' % '\n'.join(provider_plugin_list))

    def show_backend_config(self):
        if False:
            for i in range(10):
                print('nop')
        output = '\n# Backend configuration: %s\n' % (self.backend.name,)
        util.print_markdown(output)

    def show_files_config(self):
        if False:
            return 10
        output = '\n# File configuration\n\n* **Config dir:** %s\n* **Config profile dir:** %s\n* **Config file:** %s\n* **Data dir:** %s\n* **Data profile dir:** %s\n* **Database:** %s\n* **Template dirs:**\n%s\n* **Preset dirs:**\n%s\n* **Workflow dirs:**\n%s\n* **Function dirs:**\n%s\n' % (self.config.config_dir, self.config.config_profile_dir, self.config.config_file or 'None', self.config.data_dir, self.config.data_profile_dir, self.config.get('database'), util.list_to_markdown_list(self.backend.template_manager.user_template_dirs), util.list_to_markdown_list(self.backend.preset_manager.user_preset_dirs), util.list_to_markdown_list(self.backend.workflow_manager.user_workflow_dirs) if getattr(self.backend, 'workflow_manager', None) else '', util.list_to_markdown_list(self.backend.function_manager.user_function_dirs) if getattr(self.backend, 'function_manager', None) else '')
        util.print_markdown(output)

    def show_profile_config(self):
        if False:
            return 10
        output = "\n# Profile '%s' configuration:\n\n```yaml\n%s\n```\n" % (self.config.profile, yaml.dump(self.config.get(), default_flow_style=False))
        util.print_markdown(output)

    def show_runtime_config(self):
        if False:
            for i in range(10):
                print('nop')
        output = '\n# Runtime configuration\n\n* Streaming: %s\n* Logging to: %s\n' % (str(self.stream), self.logfile and self.logfile.name or 'None')
        output += self.backend.get_runtime_config()
        util.print_markdown(output)

    def show_section_config(self, section, section_data):
        if False:
            return 10
        config_data = yaml.dump(section_data, default_flow_style=False) if isinstance(section_data, dict) else section_data
        output = "\n# Configuration section '%s':\n\n```\n%s\n```\n" % (section, config_data)
        util.print_markdown(output)

    def show_full_config(self):
        if False:
            print('Hello World!')
        self.show_backend_config()
        self.show_files_config()
        self.show_profile_config()
        self.show_runtime_config()

    def command_config(self, arg):
        if False:
            while True:
                i = 10
        '\n        Show or edit the current configuration\n\n        Examples:\n            Show all: {COMMAND}\n            Edit config: {COMMAND} edit\n            Show files config: {COMMAND} files\n            Show profile config: {COMMAND} profile\n            Show runtime config: {COMMAND} runtime\n            Show section: {COMMAND} debug\n        '
        if arg:
            if arg in self.config.properties:
                property = getattr(self.config, arg, None)
                print(property)
            if arg == 'edit':
                file_editor(self.config.config_file)
                self.reload_repl()
                return (True, None, 'Reloaded configuration')
            elif arg == 'files':
                return self.show_files_config()
            elif arg == 'profile':
                return self.show_profile_config()
            elif arg == 'runtime':
                return self.show_runtime_config()
            else:
                section_data = self.config.get(arg)
                if section_data:
                    return self.show_section_config(arg, section_data)
                else:
                    return (False, arg, f'Configuration section {arg} does not exist')
        else:
            self.show_full_config()

    def command_exit(self, _):
        if False:
            i = 10
            return i + 15
        '\n        Exit the shell\n\n        Examples:\n            {COMMAND}\n        '
        pass

    def command_quit(self, _):
        if False:
            for i in range(10):
                print('nop')
        '\n        Exit the shell\n\n        Examples:\n            {COMMAND}\n        '
        pass

    def get_command_method(self, command):
        if False:
            for i in range(10):
                print('nop')
        return self.get_shell_method(f'command_{command}')

    def get_command_action_method(self, command, action):
        if False:
            for i in range(10):
                print('nop')
        return self.get_shell_method(util.dash_to_underscore(f'action_{command}_{action}'))

    def get_shell_method(self, method_string):
        if False:
            return 10
        method = util.get_class_method(self.__class__, method_string)
        if method:
            return (method, self)
        for plugin in self.plugins.values():
            method = util.get_class_method(plugin.__class__, method_string)
            if method:
                return (method, plugin)
        raise AttributeError(f'{method_string} method not found in any shell class')

    def run_command(self, command, argument):
        if False:
            while True:
                i = 10
        command = util.dash_to_underscore(command)
        if command == 'help':
            self.help(argument)
        elif command in self.commands:
            (method, obj) = self.get_command_method(command)
            try:
                response = method(obj, argument)
            except Exception as e:
                print(repr(e))
                if self.debug:
                    traceback.print_exc()
            else:
                util.output_response(response)
        else:
            print(f'Unknown command: {command}')

    def cmdloop(self):
        if False:
            print('Hello World!')
        print('')
        util.print_markdown('### %s' % self.intro)
        while True:
            self.set_user_prompt()
            user_input = self.prompt_session.prompt(self.prompt, completer=self.command_completer, complete_style=CompleteStyle.MULTI_COLUMN, reserve_space_for_menu=3)
            try:
                (command, argument) = util.parse_shell_input(user_input)
            except NoInputError:
                continue
            except EOFError:
                break
            exec_prompt_pre_result = self.exec_prompt_pre(command, argument)
            if exec_prompt_pre_result:
                util.output_response(exec_prompt_pre_result)
            else:
                self.run_command(command, argument)
        print('GoodBye!')