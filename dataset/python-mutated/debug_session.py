import glob
import json
import logging
import os
import shlex
import subprocess
import functools
import vim
import importlib
import typing
from vimspector import breakpoints, code, core_utils, debug_adapter_connection, disassembly, install, output, stack_trace, utils, variables, settings, terminal, installer
from vimspector.vendor.json_minify import minify
VIMSPECTOR_HOME = utils.GetVimspectorBase()
USER_CHOICES = {}

class DebugSession(object):
    child_sessions: typing.List['DebugSession']

    def CurrentSession():
        if False:
            for i in range(10):
                print('nop')

        def decorator(fct):
            if False:
                while True:
                    i = 10

            @functools.wraps(fct)
            def wrapper(self: 'DebugSession', *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                active_session = self
                if self._stackTraceView:
                    active_session = self._stackTraceView.GetCurrentSession()
                if active_session is not None:
                    return fct(active_session, *args, **kwargs)
                return fct(self, *args, **kwargs)
            return wrapper
        return decorator

    def ParentOnly(otherwise=None):
        if False:
            i = 10
            return i + 15

        def decorator(fct):
            if False:
                while True:
                    i = 10

            @functools.wraps(fct)
            def wrapper(self: 'DebugSession', *args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                if self.parent_session:
                    return otherwise
                return fct(self, *args, **kwargs)
            return wrapper
        return decorator

    def IfConnected(otherwise=None):
        if False:
            i = 10
            return i + 15

        def decorator(fct):
            if False:
                while True:
                    i = 10
            'Decorator, call fct if self._connected else echo warning'

            @functools.wraps(fct)
            def wrapper(self: 'DebugSession', *args, **kwargs):
                if False:
                    return 10
                if not self._connection:
                    utils.UserMessage('Vimspector not connected, start a debug session first', persist=False, error=True)
                    return otherwise
                return fct(self, *args, **kwargs)
            return wrapper
        return decorator

    def RequiresUI(otherwise=None):
        if False:
            while True:
                i = 10
        'Decorator, call fct if self._connected else echo warning'

        def decorator(fct):
            if False:
                for i in range(10):
                    print('nop')

            @functools.wraps(fct)
            def wrapper(self, *args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                if not self.HasUI():
                    utils.UserMessage('Vimspector is not active', persist=False, error=True)
                    return otherwise
                return fct(self, *args, **kwargs)
            return wrapper
        return decorator

    def __init__(self, session_id, session_manager, api_prefix, session_name=None, parent_session: 'DebugSession'=None):
        if False:
            print('Hello World!')
        self.session_id = session_id
        self.manager = session_manager
        self.name = session_name
        self.parent_session = parent_session
        self.child_sessions = []
        if parent_session:
            parent_session.child_sessions.append(self)
        self._logger = logging.getLogger(__name__ + '.' + str(session_id))
        utils.SetUpLogging(self._logger, session_id)
        self._api_prefix = api_prefix
        self._render_emitter = utils.EventEmitter()
        self._logger.info(f'**** INITIALISING NEW VIMSPECTOR SESSION FOR ID {session_id} ****')
        self._logger.info('API is: {}'.format(api_prefix))
        self._logger.info('VIMSPECTOR_HOME = %s', VIMSPECTOR_HOME)
        self._logger.info('gadgetDir = %s', install.GetGadgetDir(VIMSPECTOR_HOME))
        self._uiTab = None
        self._logView: output.OutputView = None
        self._stackTraceView: stack_trace.StackTraceView = None
        self._variablesView: variables.VariablesView = None
        self._outputView: output.DAPOutputView = None
        self._codeView: code.CodeView = None
        self._disassemblyView: disassembly.DisassemblyView = None
        if parent_session:
            self._breakpoints = parent_session._breakpoints
        else:
            self._breakpoints = breakpoints.ProjectBreakpoints(session_id, self._render_emitter, self._IsPCPresentAt, self._disassemblyView)
            utils.SetSessionWindows({})
        self._saved_variables_data = None
        self._splash_screen = None
        self._remote_term = None
        self._adapter_term = None
        self._run_on_server_exit = None
        self._configuration = None
        self._adapter = None
        self._launch_config = None
        self._ResetServerState()

    def _ResetServerState(self):
        if False:
            while True:
                i = 10
        self._connection = None
        self._init_complete = False
        self._launch_complete = False
        self._on_init_complete_handlers = []
        self._server_capabilities = {}
        self._breakpoints.ClearTemporaryBreakpoints()

    def GetConfigurations(self, adapters):
        if False:
            i = 10
            return i + 15
        current_file = utils.GetBufferFilepath(vim.current.buffer)
        filetypes = utils.GetBufferFiletypes(vim.current.buffer)
        configurations = settings.Dict('configurations')
        for launch_config_file in PathsToAllConfigFiles(VIMSPECTOR_HOME, current_file, filetypes):
            self._logger.debug(f'Reading configurations from: {launch_config_file}')
            if not launch_config_file or not os.path.exists(launch_config_file):
                continue
            with open(launch_config_file, 'r') as f:
                database = json.loads(minify(f.read()))
                configurations.update(database.get('configurations') or {})
                adapters.update(database.get('adapters') or {})
        filetype_configurations = configurations
        if filetypes:
            filetype_configurations = {k: c for (k, c) in configurations.items() if 'filetypes' not in c or any((ft in c['filetypes'] for ft in filetypes))}
        return (launch_config_file, filetype_configurations, configurations)

    def Name(self):
        if False:
            print('Hello World!')
        return self.name if self.name else 'Unnamed-' + str(self.session_id)

    def DisplayName(self):
        if False:
            i = 10
            return i + 15
        return self.Name() + ' (' + str(self.session_id) + ')'

    @ParentOnly()
    def Start(self, force_choose=False, launch_variables=None, adhoc_configurations=None):
        if False:
            return 10
        if launch_variables is None:
            launch_variables = {}
        self._logger.info('User requested start debug session with %s', launch_variables)
        current_file = utils.GetBufferFilepath(vim.current.buffer)
        adapters = settings.Dict('adapters')
        launch_config_file = None
        configurations = None
        if adhoc_configurations:
            configurations = adhoc_configurations
        else:
            (launch_config_file, configurations, all_configurations) = self.GetConfigurations(adapters)
        if not configurations:
            utils.UserMessage('Unable to find any debug configurations. You need to tell vimspector how to launch your application.')
            return
        glob.glob(install.GetGadgetDir(VIMSPECTOR_HOME))
        for gadget_config_file in PathsToAllGadgetConfigs(VIMSPECTOR_HOME, current_file):
            self._logger.debug(f'Reading gadget config: {gadget_config_file}')
            if not gadget_config_file or not os.path.exists(gadget_config_file):
                continue
            with open(gadget_config_file, 'r') as f:
                a = json.loads(minify(f.read())).get('adapters') or {}
                adapters.update(a)
        if 'configuration' in launch_variables:
            configuration_name = launch_variables.pop('configuration')
        elif force_choose:
            configuration_name = utils.SelectFromList('Which launch configuration?', sorted(configurations.keys()))
        elif len(configurations) == 1 and next(iter(configurations.values())).get('autoselect', True):
            configuration_name = next(iter(configurations.keys()))
        else:
            defaults = {n: c for (n, c) in configurations.items() if c.get('default', False) and c.get('autoselect', True)}
            if len(defaults) == 1:
                configuration_name = next(iter(defaults.keys()))
            else:
                configuration_name = utils.SelectFromList('Which launch configuration?', sorted(configurations.keys()))
        if not configuration_name or configuration_name not in configurations:
            return
        if self.name is None:
            self.name = configuration_name
        if launch_config_file:
            self._workspace_root = os.path.dirname(launch_config_file)
        else:
            self._workspace_root = os.path.dirname(current_file)
        try:
            configuration = configurations[configuration_name]
        except KeyError:
            configuration = all_configurations[configuration_name]
        current_configuration_name = configuration_name
        while 'extends' in configuration:
            base_configuration_name = configuration.pop('extends')
            base_configuration = all_configurations.get(base_configuration_name)
            if base_configuration is None:
                raise RuntimeError(f'The adapter {current_configuration_name} extends configuration {base_configuration_name}, but this does not exist')
            core_utils.override(base_configuration, configuration)
            current_configuration_name = base_configuration_name
            configuration = base_configuration
        adapter = configuration.get('adapter')
        if isinstance(adapter, str):
            adapter_dict = adapters.get(adapter)
            if adapter_dict is None:
                suggested_gadgets = installer.FindGadgetForAdapter(adapter)
                if suggested_gadgets:
                    response = utils.AskForInput(f"The specified adapter '{adapter}' is not installed. Would you like to install the following gadgets? ", ' '.join(suggested_gadgets))
                    if response:
                        new_launch_variables = dict(launch_variables)
                        new_launch_variables['configuration'] = configuration_name
                        installer.RunInstaller(self._api_prefix, False, *shlex.split(response), then=lambda : self.Start(new_launch_variables))
                        return
                    elif response is None:
                        return
                utils.UserMessage(f"The specified adapter '{adapter}' is not available. Did you forget to run 'VimspectorInstall'?", persist=True, error=True)
                return
            adapter = adapter_dict
        if not adapter:
            utils.UserMessage('No adapter configured for {}'.format(configuration_name), persist=True)
            return
        while 'extends' in adapter:
            base_adapter_name = adapter.pop('extends')
            base_adapter = adapters.get(base_adapter_name)
            if base_adapter is None:
                suggested_gadgets = installer.FindGadgetForAdapter(base_adapter_name)
                if suggested_gadgets:
                    response = utils.AskForInput(f"The specified base adapter '{base_adapter_name}' is not installed. Would you like to install the following gadgets? ", ' '.join(suggested_gadgets))
                    if response:
                        new_launch_variables = dict(launch_variables)
                        new_launch_variables['configuration'] = configuration_name
                        installer.RunInstaller(self._api_prefix, False, *shlex.split(response), then=lambda : self.Start(new_launch_variables))
                        return
                    elif response is None:
                        return
                utils.UserMessage(f"The specified base adapter '{base_adapter_name}' is not available. Did you forget to run 'VimspectorInstall'?", persist=True, error=True)
                return
            core_utils.override(base_adapter, adapter)
            adapter = base_adapter

        def relpath(p, relative_to):
            if False:
                print('Hello World!')
            if not p:
                return ''
            return os.path.relpath(p, relative_to)

        def splitext(p):
            if False:
                return 10
            if not p:
                return ['', '']
            return os.path.splitext(p)
        variables = {'dollar': '$', 'workspaceRoot': self._workspace_root, 'workspaceFolder': self._workspace_root, 'gadgetDir': install.GetGadgetDir(VIMSPECTOR_HOME), 'file': current_file}
        calculus = {'relativeFileDirname': lambda : os.path.dirname(relpath(current_file, self._workspace_root)), 'relativeFile': lambda : relpath(current_file, self._workspace_root), 'fileBasename': lambda : os.path.basename(current_file), 'fileBasenameNoExtension': lambda : splitext(os.path.basename(current_file))[0], 'fileDirname': lambda : os.path.dirname(current_file), 'fileExtname': lambda : splitext(os.path.basename(current_file))[1], 'cwd': os.getcwd, 'unusedLocalPort': utils.GetUnusedLocalPort, 'SelectProcess': _SelectProcess, 'PickProcess': _SelectProcess}
        USER_CHOICES.update(launch_variables)
        variables.update(launch_variables)
        try:
            variables.update(utils.ParseVariables(adapter.pop('variables', {}), variables, calculus, USER_CHOICES))
            variables.update(utils.ParseVariables(configuration.pop('variables', {}), variables, calculus, USER_CHOICES))
            utils.ExpandReferencesInDict(configuration, variables, calculus, USER_CHOICES)
            utils.ExpandReferencesInDict(adapter, variables, calculus, USER_CHOICES)
        except KeyboardInterrupt:
            self._Reset()
            return
        self._StartWithConfiguration(configuration, adapter)

    def _StartWithConfiguration(self, configuration, adapter):
        if False:
            return 10

        def start():
            if False:
                i = 10
                return i + 15
            self._configuration = configuration
            self._adapter = adapter
            self._launch_config = None
            self._logger.info('Configuration: %s', json.dumps(self._configuration))
            self._logger.info('Adapter: %s', json.dumps(self._adapter))
            if self.parent_session:
                self._uiTab = self.parent_session._uiTab
                self._stackTraceView = self.parent_session._stackTraceView
                self._variablesView = self.parent_session._variablesView
                self._outputView = self.parent_session._outputView
                self._disassemblyView = self.parent_session._disassemblyView
                self._codeView = self.parent_session._codeView
            elif not self._uiTab:
                self._SetUpUI()
            else:
                with utils.NoAutocommands():
                    vim.current.tabpage = self._uiTab
            self._stackTraceView.AddSession(self)
            self._Prepare()
            if not self._StartDebugAdapter():
                self._logger.info('Failed to launch or attach to the debug adapter')
                return
            self._Initialise()
            if self._saved_variables_data:
                self._variablesView.Load(self._saved_variables_data)
        if self._connection:
            self._logger.debug('Stop debug adapter with callback: start')
            self.StopAllSessions(interactive=False, then=start)
            return
        start()

    @ParentOnly()
    def Restart(self):
        if False:
            for i in range(10):
                print('nop')
        if self._configuration is None or self._adapter is None:
            return self.Start()
        self._StartWithConfiguration(self._configuration, self._adapter)

    def Connection(self):
        if False:
            while True:
                i = 10
        return self._connection

    def HasUI(self):
        if False:
            print('Hello World!')
        return self._uiTab and self._uiTab.valid

    def IsUITab(self, tab_number):
        if False:
            for i in range(10):
                print('nop')
        return self.HasUI() and self._uiTab.number == tab_number

    @ParentOnly()
    def SwitchTo(self):
        if False:
            return 10
        if self.HasUI():
            vim.current.tabpage = self._uiTab
        self._breakpoints.UpdateUI()

    @ParentOnly()
    def SwitchFrom(self):
        if False:
            return 10
        self._breakpoints.ClearUI()

    def OnChannelData(self, data):
        if False:
            while True:
                i = 10
        if self._connection is None:
            return
        self._connection.OnData(data)

    def OnServerStderr(self, data):
        if False:
            return 10
        if self._outputView:
            self._outputView.Print('server', data)

    def OnRequestTimeout(self, timer_id):
        if False:
            for i in range(10):
                print('nop')
        self._connection.OnRequestTimeout(timer_id)

    def OnChannelClosed(self):
        if False:
            return 10
        self._connection = None

    def StopAllSessions(self, interactive=False, then=None):
        if False:
            while True:
                i = 10

        def Next():
            if False:
                i = 10
                return i + 15
            if self.child_sessions:
                c = self.child_sessions.pop()
                c.StopAllSessions(interactive=interactive, then=Next)
            elif self._connection:
                self._StopDebugAdapter(interactive=interactive, callback=then)
            else:
                then()
        Next()

    @ParentOnly()
    @IfConnected()
    def Stop(self, interactive=False):
        if False:
            while True:
                i = 10
        self._logger.debug('Stop debug adapter with no callback')
        self.StopAllSessions(interactive=False)

    @ParentOnly()
    def Destroy(self):
        if False:
            while True:
                i = 10
        'Call when the vimspector session will be removed and never used again'
        if self._connection is not None:
            raise RuntimeError("Can't destroy a session with a live connection")
        if self.HasUI():
            raise RuntimeError("Can't destroy a session with an active UI")
        self.ClearBreakpoints()
        self._ResetUI()

    @ParentOnly()
    def Reset(self, interactive=False):
        if False:
            for i in range(10):
                print('nop')
        self._logger.debug('Stop debug adapter with callback: _Reset')
        self.StopAllSessions(interactive, self._Reset)

    def _IsPCPresentAt(self, file_path, line):
        if False:
            i = 10
            return i + 15
        return self._codeView and self._codeView.IsPCPresentAt(file_path, line)

    def _ResetUI(self):
        if False:
            return 10
        if not self.parent_session:
            if self._stackTraceView:
                self._stackTraceView.Reset()
            if self._variablesView:
                self._variablesView.Reset()
            if self._outputView:
                self._outputView.Reset()
            if self._logView:
                self._logView.Reset()
            if self._codeView:
                self._codeView.Reset()
            if self._disassemblyView:
                self._disassemblyView.Reset()
        self._breakpoints.RemoveConnection(self._connection)
        self._stackTraceView = None
        self._variablesView = None
        self._outputView = None
        self._codeView = None
        self._disassemblyView = None
        self._remote_term = None
        self._uiTab = None
        if self.parent_session:
            self.manager.DestroySession(self)

    def _Reset(self):
        if False:
            i = 10
            return i + 15
        if self.parent_session:
            self._ResetUI()
            return
        vim.vars['vimspector_resetting'] = 1
        self._logger.info('Debugging complete.')
        if self.HasUI():
            self._logger.debug('Clearing down UI')
            with utils.NoAutocommands():
                vim.current.tabpage = self._uiTab
            self._splash_screen = utils.HideSplash(self._api_prefix, self._splash_screen)
            self._ResetUI()
            vim.command('tabclose!')
        else:
            self._ResetUI()
        self._breakpoints.SetDisassemblyManager(None)
        utils.SetSessionWindows({'breakpoints': vim.vars['vimspector_session_windows'].get('breakpoints')})
        vim.command('doautocmd <nomodeline> User VimspectorDebugEnded')
        vim.vars['vimspector_resetting'] = 0
        self._breakpoints.UpdateUI()

    @ParentOnly(False)
    def ReadSessionFile(self, session_file: str=None):
        if False:
            return 10
        if session_file is None:
            session_file = self._DetectSessionFile(invent_one_if_not_found=False)
        if session_file is None:
            utils.UserMessage(f"No {settings.Get('session_file_name')} file found. Specify a file with :VimspectorLoadSession <filename>", persist=True, error=True)
            return False
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            USER_CHOICES.update(session_data.get('session', {}).get('user_choices', {}))
            self._breakpoints.Load(session_data.get('breakpoints'))
            variables_data = session_data.get('variables', {})
            if self._variablesView:
                self._variablesView.Load(variables_data)
            else:
                self._saved_variables_data = variables_data
            utils.UserMessage(f'Loaded session file {session_file}', persist=True)
            return True
        except OSError:
            self._logger.exception(f'Invalid session file {session_file}')
            utils.UserMessage(f'Session file {session_file} not found', persist=True, error=True)
            return False
        except json.JSONDecodeError:
            self._logger.exception(f'Invalid session file {session_file}')
            utils.UserMessage('The session file could not be read', persist=True, error=True)
            return False

    @ParentOnly(False)
    def WriteSessionFile(self, session_file: str=None):
        if False:
            i = 10
            return i + 15
        if session_file is None:
            session_file = self._DetectSessionFile(invent_one_if_not_found=True)
        elif os.path.isdir(session_file):
            session_file = self._DetectSessionFile(invent_one_if_not_found=True, in_directory=session_file)
        try:
            with open(session_file, 'w') as f:
                f.write(json.dumps({'breakpoints': self._breakpoints.Save(), 'session': {'user_choices': USER_CHOICES}, 'variables': self._variablesView.Save() if self._variablesView else {}}))
            utils.UserMessage(f'Wrote {session_file}')
            return True
        except OSError:
            self._logger.exception(f'Unable to write session file {session_file}')
            utils.UserMessage('The session file could not be read', persist=True, error=True)
            return False

    def _DetectSessionFile(self, invent_one_if_not_found: bool, in_directory: str=None):
        if False:
            while True:
                i = 10
        session_file_name = settings.Get('session_file_name')
        if in_directory:
            write_directory = in_directory
            file_path = os.path.join(in_directory, session_file_name)
            if not os.path.exists(file_path):
                file_path = None
        else:
            current_file = utils.GetBufferFilepath(vim.current.buffer)
            write_directory = os.getcwd()
            file_path = utils.PathToConfigFile(session_file_name, os.path.dirname(current_file))
        if file_path:
            return file_path
        if invent_one_if_not_found:
            return os.path.join(write_directory, session_file_name)
        return None

    @CurrentSession()
    @IfConnected()
    def StepOver(self, **kwargs):
        if False:
            print('Hello World!')
        if self._stackTraceView.GetCurrentThreadId() is None:
            return
        arguments = {'threadId': self._stackTraceView.GetCurrentThreadId(), 'granularity': self._CurrentSteppingGranularity()}
        arguments.update(kwargs)
        if not self._server_capabilities.get('supportsSteppingGranularity'):
            arguments.pop('granularity')
        self._connection.DoRequest(None, {'command': 'next', 'arguments': arguments})
        self._stackTraceView.OnContinued(self)
        self.ClearCurrentPC()

    @CurrentSession()
    @IfConnected()
    def StepInto(self, **kwargs):
        if False:
            while True:
                i = 10
        threadId = self._stackTraceView.GetCurrentThreadId()
        if threadId is None:
            return

        def handler(*_):
            if False:
                i = 10
                return i + 15
            self._stackTraceView.OnContinued(self, {'threadId': threadId})
            self.ClearCurrentPC()
        arguments = {'threadId': threadId, 'granularity': self._CurrentSteppingGranularity()}
        arguments.update(kwargs)
        self._connection.DoRequest(handler, {'command': 'stepIn', 'arguments': arguments})

    @CurrentSession()
    @IfConnected()
    def StepOut(self, **kwargs):
        if False:
            return 10
        threadId = self._stackTraceView.GetCurrentThreadId()
        if threadId is None:
            return

        def handler(*_):
            if False:
                return 10
            self._stackTraceView.OnContinued(self, {'threadId': threadId})
            self.ClearCurrentPC()
        arguments = {'threadId': threadId, 'granularity': self._CurrentSteppingGranularity()}
        arguments.update(kwargs)
        self._connection.DoRequest(handler, {'command': 'stepOut', 'arguments': arguments})

    def _CurrentSteppingGranularity(self):
        if False:
            for i in range(10):
                print('nop')
        if self._disassemblyView and self._disassemblyView.IsCurrent():
            return 'instruction'
        return 'statement'

    @CurrentSession()
    def Continue(self):
        if False:
            while True:
                i = 10
        if not self._connection:
            self.Start()
            return
        threadId = self._stackTraceView.GetCurrentThreadId()
        if threadId is None:
            utils.UserMessage('No current thread', persist=True)
            return

        def handler(msg):
            if False:
                while True:
                    i = 10
            self._stackTraceView.OnContinued(self, {'threadId': threadId, 'allThreadsContinued': (msg.get('body') or {}).get('allThreadsContinued', True)})
            self.ClearCurrentPC()
        self._connection.DoRequest(handler, {'command': 'continue', 'arguments': {'threadId': threadId}})

    @CurrentSession()
    @IfConnected()
    def Pause(self):
        if False:
            for i in range(10):
                print('nop')
        if self._stackTraceView.GetCurrentThreadId() is None:
            utils.UserMessage('No current thread', persist=True)
            return
        self._connection.DoRequest(None, {'command': 'pause', 'arguments': {'threadId': self._stackTraceView.GetCurrentThreadId()}})

    @IfConnected()
    def PauseContinueThread(self):
        if False:
            while True:
                i = 10
        self._stackTraceView.PauseContinueThread()

    @CurrentSession()
    @IfConnected()
    def SetCurrentThread(self):
        if False:
            print('Hello World!')
        self._stackTraceView.SetCurrentThread()

    @CurrentSession()
    @IfConnected()
    def ExpandVariable(self, buf=None, line_num=None):
        if False:
            i = 10
            return i + 15
        self._variablesView.ExpandVariable(buf, line_num)

    @CurrentSession()
    @IfConnected()
    def SetVariableValue(self, new_value=None, buf=None, line_num=None):
        if False:
            while True:
                i = 10
        if not self._server_capabilities.get('supportsSetVariable'):
            return
        self._variablesView.SetVariableValue(new_value, buf, line_num)

    @ParentOnly()
    def ReadMemory(self, length=None, offset=None):
        if False:
            for i in range(10):
                print('nop')
        if not self._server_capabilities.get('supportsReadMemoryRequest'):
            utils.UserMessage('Server does not support memory request', error=True)
            return
        connection: debug_adapter_connection.DebugAdapterConnection
        (connection, memoryReference) = self._variablesView.GetMemoryReference()
        if memoryReference is None or connection is None:
            utils.UserMessage('Cannot find memory reference for that', error=True)
            return
        if length is None:
            length = utils.AskForInput('How much data to display? ', default_value='1024')
        try:
            length = int(length)
        except ValueError:
            return
        if offset is None:
            offset = utils.AskForInput('Location offset? ', default_value='0')
        try:
            offset = int(offset)
        except ValueError:
            return

        def handler(msg):
            if False:
                for i in range(10):
                    print('nop')
            self._codeView.ShowMemory(connection.GetSessionId(), memoryReference, length, offset, msg)
        connection.DoRequest(handler, {'command': 'readMemory', 'arguments': {'memoryReference': memoryReference, 'count': int(length), 'offset': int(offset)}})

    @CurrentSession()
    @IfConnected()
    @RequiresUI()
    def ShowDisassembly(self):
        if False:
            return 10
        if self._disassemblyView and self._disassemblyView.WindowIsValid():
            return
        if not self._codeView or not self._codeView._window.valid:
            return
        if not self._stackTraceView:
            return
        if not self._server_capabilities.get('supportsDisassembleRequest', False):
            utils.UserMessage("Sorry, server doesn't support that")
            return
        with utils.LetCurrentWindow(self._codeView._window):
            vim.command(f"rightbelow {settings.Int('disassembly_height')}new")
            self._disassemblyView = disassembly.DisassemblyView(vim.current.window, self._api_prefix, self._render_emitter)
            self._breakpoints.SetDisassemblyManager(self._disassemblyView)
            utils.UpdateSessionWindows({'disassembly': utils.WindowID(vim.current.window, self._uiTab)})
            self._disassemblyView.SetCurrentFrame(self._connection, self._stackTraceView.GetCurrentFrame(), True)

    def OnDisassemblyWindowScrolled(self, win_id):
        if False:
            while True:
                i = 10
        if self._disassemblyView:
            self._disassemblyView.OnWindowScrolled(win_id)

    @CurrentSession()
    @IfConnected()
    def AddWatch(self, expression):
        if False:
            while True:
                i = 10
        self._variablesView.AddWatch(self._connection, self._stackTraceView.GetCurrentFrame(), expression)

    @CurrentSession()
    @IfConnected()
    def EvaluateConsole(self, expression, verbose):
        if False:
            while True:
                i = 10
        self._outputView.Evaluate(self._connection, self._stackTraceView.GetCurrentFrame(), expression, verbose)

    @CurrentSession()
    @IfConnected()
    def DeleteWatch(self):
        if False:
            for i in range(10):
                print('nop')
        self._variablesView.DeleteWatch()

    @CurrentSession()
    @IfConnected()
    def HoverEvalTooltip(self, winnr, bufnr, lnum, expression, is_hover):
        if False:
            for i in range(10):
                print('nop')
        frame = self._stackTraceView.GetCurrentFrame()
        if frame is None:
            self._logger.debug('Tooltip: Not in a stack frame')
            return ''
        if winnr == int(self._codeView._window.number):
            return self._variablesView.HoverEvalTooltip(self._connection, frame, expression, is_hover)
        return self._variablesView.HoverVarWinTooltip(bufnr, lnum, is_hover)

    @CurrentSession()
    def CleanUpTooltip(self):
        if False:
            return 10
        return self._variablesView.CleanUpTooltip()

    @IfConnected()
    def ExpandFrameOrThread(self):
        if False:
            for i in range(10):
                print('nop')
        self._stackTraceView.ExpandFrameOrThread()

    @IfConnected()
    def UpFrame(self):
        if False:
            return 10
        self._stackTraceView.UpFrame()

    @IfConnected()
    def DownFrame(self):
        if False:
            return 10
        self._stackTraceView.DownFrame()

    def ToggleLog(self):
        if False:
            i = 10
            return i + 15
        if self.HasUI():
            return self.ShowOutput('Vimspector')
        if self._logView and self._logView.WindowIsValid():
            self._logView.Reset()
            self._logView = None
            return
        if self._logView:
            self._logView.Reset()
        vim.command(f"botright {settings.Int('bottombar_height')}new")
        win = vim.current.window
        self._logView = output.OutputView(win, self._api_prefix)
        self._logView.AddLogFileView()
        self._logView.ShowOutput('Vimspector')

    @RequiresUI()
    def ShowOutput(self, category):
        if False:
            print('Hello World!')
        if not self._outputView.WindowIsValid():
            with utils.LetCurrentTabpage(self._uiTab):
                vim.command(f"botright {settings.Int('bottombar_height')}new")
                self._outputView.UseWindow(vim.current.window)
                utils.UpdateSessionWindows({'output': utils.WindowID(vim.current.window, self._uiTab)})
        self._outputView.ShowOutput(category)

    @RequiresUI(otherwise=[])
    def GetOutputBuffers(self):
        if False:
            print('Hello World!')
        return self._outputView.GetCategories()

    @CurrentSession()
    @IfConnected(otherwise=[])
    def GetCompletionsSync(self, text_line, column_in_bytes):
        if False:
            while True:
                i = 10
        if not self._server_capabilities.get('supportsCompletionsRequest'):
            return []
        response = self._connection.DoRequestSync({'command': 'completions', 'arguments': {'frameId': self._stackTraceView.GetCurrentFrame()['id'], 'text': text_line, 'column': column_in_bytes}})
        return response['body']['targets']

    @CurrentSession()
    @IfConnected(otherwise=[])
    def GetCommandLineCompletions(self, ArgLead, prev_non_keyword_char):
        if False:
            return 10
        items = []
        for candidate in self.GetCompletionsSync(ArgLead, prev_non_keyword_char):
            label = candidate.get('text', candidate['label'])
            start = prev_non_keyword_char - 1
            if 'start' in candidate and 'length' in candidate:
                start = candidate['start']
            items.append(ArgLead[0:start] + label)
        return items

    @ParentOnly()
    def RefreshSigns(self):
        if False:
            print('Hello World!')
        if self._connection:
            self._codeView.Refresh()
        self._breakpoints.Refresh()

    @ParentOnly()
    def _SetUpUI(self):
        if False:
            print('Hello World!')
        vim.command('$tab split')
        utils.Call('vimspector#internal#state#SwitchToSession', self.session_id)
        self._uiTab = vim.current.tabpage
        mode = settings.Get('ui_mode')
        if mode == 'auto':
            min_width = settings.Int('sidebar_width') + 1 + 2 + 3 + settings.Int('code_minwidth') + 1 + settings.Int('terminal_minwidth')
            min_height = settings.Int('code_minheight') + 1 + settings.Int('topbar_height') + 1 + settings.Int('bottombar_height') + 1 + 2
            mode = 'vertical' if vim.options['columns'] < min_width else 'horizontal'
            if vim.options['lines'] < min_height:
                mode = 'horizontal'
            self._logger.debug('min_width/height: %s/%s, actual: %s/%s - result: %s', min_width, min_height, vim.options['columns'], vim.options['lines'], mode)
        if mode == 'vertical':
            self._SetUpUIVertical()
        else:
            self._SetUpUIHorizontal()

    def _SetUpUIHorizontal(self):
        if False:
            print('Hello World!')
        code_window = vim.current.window
        self._codeView = code.CodeView(self.session_id, code_window, self._api_prefix, self._render_emitter, self._breakpoints.IsBreakpointPresentAt)
        vim.command(f"topleft vertical {settings.Int('sidebar_width')}new")
        stack_trace_window = vim.current.window
        one_third = int(vim.eval('winheight( 0 )')) / 3
        self._stackTraceView = stack_trace.StackTraceView(self.session_id, stack_trace_window)
        vim.command('leftabove new')
        watch_window = vim.current.window
        vim.command('leftabove new')
        vars_window = vim.current.window
        with utils.LetCurrentWindow(vars_window):
            vim.command(f'{one_third}wincmd _')
        with utils.LetCurrentWindow(watch_window):
            vim.command(f'{one_third}wincmd _')
        with utils.LetCurrentWindow(stack_trace_window):
            vim.command(f'{one_third}wincmd _')
        self._variablesView = variables.VariablesView(self.session_id, vars_window, watch_window)
        vim.current.window = code_window
        vim.command(f"rightbelow {settings.Int('bottombar_height')}new")
        output_window = vim.current.window
        self._outputView = output.DAPOutputView(output_window, self._api_prefix, session_id=self.session_id)
        utils.SetSessionWindows({'mode': 'horizontal', 'tabpage': self._uiTab.number, 'code': utils.WindowID(code_window, self._uiTab), 'stack_trace': utils.WindowID(stack_trace_window, self._uiTab), 'variables': utils.WindowID(vars_window, self._uiTab), 'watches': utils.WindowID(watch_window, self._uiTab), 'output': utils.WindowID(output_window, self._uiTab), 'eval': None, 'breakpoints': vim.vars['vimspector_session_windows'].get('breakpoints')})
        with utils.RestoreCursorPosition():
            with utils.RestoreCurrentWindow():
                with utils.RestoreCurrentBuffer(vim.current.window):
                    vim.command('doautocmd User VimspectorUICreated')

    def _SetUpUIVertical(self):
        if False:
            i = 10
            return i + 15
        code_window = vim.current.window
        self._codeView = code.CodeView(self.session_id, code_window, self._api_prefix, self._render_emitter, self._breakpoints.IsBreakpointPresentAt)
        vim.command(f"topleft {settings.Int('topbar_height')}new")
        stack_trace_window = vim.current.window
        one_third = int(vim.eval('winwidth( 0 )')) / 3
        self._stackTraceView = stack_trace.StackTraceView(self.session_id, stack_trace_window)
        vim.command('leftabove vertical new')
        watch_window = vim.current.window
        vim.command('leftabove vertical new')
        vars_window = vim.current.window
        with utils.LetCurrentWindow(vars_window):
            vim.command(f'{one_third}wincmd |')
        with utils.LetCurrentWindow(watch_window):
            vim.command(f'{one_third}wincmd |')
        with utils.LetCurrentWindow(stack_trace_window):
            vim.command(f'{one_third}wincmd |')
        self._variablesView = variables.VariablesView(self.session_id, vars_window, watch_window)
        vim.current.window = code_window
        vim.command(f"rightbelow {settings.Int('bottombar_height')}new")
        output_window = vim.current.window
        self._outputView = output.DAPOutputView(output_window, self._api_prefix, session_id=self.session_id)
        utils.SetSessionWindows({'mode': 'vertical', 'tabpage': self._uiTab.number, 'code': utils.WindowID(code_window, self._uiTab), 'stack_trace': utils.WindowID(stack_trace_window, self._uiTab), 'variables': utils.WindowID(vars_window, self._uiTab), 'watches': utils.WindowID(watch_window, self._uiTab), 'output': utils.WindowID(output_window, self._uiTab), 'eval': None, 'breakpoints': vim.vars['vimspector_session_windows'].get('breakpoints')})
        with utils.RestoreCursorPosition():
            with utils.RestoreCurrentWindow():
                with utils.RestoreCurrentBuffer(vim.current.window):
                    vim.command('doautocmd User VimspectorUICreated')

    @RequiresUI()
    def ClearCurrentFrame(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetCurrentFrame(None)

    def ClearCurrentPC(self):
        if False:
            while True:
                i = 10
        self._codeView.SetCurrentFrame(None, False)
        if self._disassemblyView:
            self._disassemblyView.SetCurrentFrame(None, None, False)

    @RequiresUI()
    def SetCurrentFrame(self, frame, reason=''):
        if False:
            while True:
                i = 10
        if not frame:
            self._variablesView.Clear()
        target = self._codeView
        if self._disassemblyView and self._disassemblyView.IsCurrent():
            target = self._disassemblyView
        if not self._codeView.SetCurrentFrame(frame, target == self._codeView):
            return False
        if self._disassemblyView:
            self._disassemblyView.SetCurrentFrame(self._connection, frame, target == self._disassemblyView)
        assert frame
        if self._codeView.current_syntax not in ('ON', 'OFF'):
            self._variablesView.SetSyntax(self._codeView.current_syntax)
            self._stackTraceView.SetSyntax(self._codeView.current_syntax)
        else:
            self._variablesView.SetSyntax(None)
            self._stackTraceView.SetSyntax(None)
        self._variablesView.LoadScopes(self._connection, frame)
        self._variablesView.EvaluateWatches(self._connection, frame)
        if reason == 'stopped':
            self._breakpoints.ClearTemporaryBreakpoint(frame['source']['path'], frame['line'])
        return True

    def _StartDebugAdapter(self):
        if False:
            while True:
                i = 10
        self._splash_screen = utils.DisplaySplash(self._api_prefix, self._splash_screen, f'Starting debug adapter for session {self.DisplayName()}...')
        if self._connection:
            utils.UserMessage('The connection is already created. Please try again', persist=True)
            return False
        self._logger.info('Starting debug adapter with: %s', json.dumps(self._adapter))
        self._init_complete = False
        self._launch_complete = False
        self._run_on_server_exit = None
        self._connection_type = 'job'
        if 'port' in self._adapter:
            self._connection_type = 'channel'
            if self._adapter['port'] == 'ask':
                port = utils.AskForInput('Enter port to connect to: ')
                if port is None:
                    self._Reset()
                    return False
                self._adapter['port'] = port
        self._connection_type = self._api_prefix + self._connection_type
        self._logger.debug(f'Connection Type: {self._connection_type}')
        self._adapter['env'] = self._adapter.get('env', {})
        if 'cwd' in self._configuration:
            self._adapter['cwd'] = self._configuration['cwd']
        elif 'cwd' not in self._adapter:
            self._adapter['cwd'] = os.getcwd()
        vim.vars['_vimspector_adapter_spec'] = self._adapter
        if self._adapter.get('tty', False):
            if 'port' not in self._adapter:
                utils.UserMessage("Invalid adapter configuration. When using a tty, communication must use socket. Add the 'port' to the adapter config.")
                return False
            if 'command' not in self._adapter:
                utils.UserMessage("Invalid adapter configuration. When using a tty, a command must be supplied. Add the 'command' to the adapter config.")
                return False
            command = self._adapter['command']
            if isinstance(command, str):
                command = shlex.split(command)
            self._adapter_term = terminal.LaunchTerminal(self._api_prefix, {'args': command, 'cwd': self._adapter['cwd'], 'env': self._adapter['env']}, self._codeView._window, self._adapter_term)
        if not vim.eval('vimspector#internal#{}#StartDebugSession(   {},  g:_vimspector_adapter_spec )'.format(self._connection_type, self.session_id)):
            self._logger.error('Unable to start debug server')
            self._splash_screen = utils.DisplaySplash(self._api_prefix, self._splash_screen, ['Unable to start or connect to debug adapter', '', 'Check :messages and :VimspectorToggleLog for more information.', '', ':VimspectorReset to close down vimspector'])
            return False
        else:
            handlers = [self]
            if 'custom_handler' in self._adapter:
                spec = self._adapter['custom_handler']
                if isinstance(spec, dict):
                    module = spec['module']
                    cls = spec['class']
                else:
                    (module, cls) = spec.rsplit('.', 1)
                try:
                    CustomHandler = getattr(importlib.import_module(module), cls)
                    handlers = [CustomHandler(self), self]
                except ImportError:
                    self._logger.exception('Unable to load custom adapter %s', spec)
            self._connection = debug_adapter_connection.DebugAdapterConnection(handlers=handlers, session_id=self.session_id, send_func=lambda msg: utils.Call('vimspector#internal#{}#Send'.format(self._connection_type), self.session_id, msg), sync_timeout=self._adapter.get('sync_timeout'), async_timeout=self._adapter.get('async_timeout'))
        self._logger.info('Debug Adapter Started')
        return True

    def _StopDebugAdapter(self, interactive=False, callback=None):
        if False:
            print('Hello World!')
        arguments = {}

        def disconnect():
            if False:
                return 10
            self._splash_screen = utils.DisplaySplash(self._api_prefix, self._splash_screen, f'Shutting down debug adapter for session {self.DisplayName()}...')

            def handler(*args):
                if False:
                    i = 10
                    return i + 15
                self._splash_screen = utils.HideSplash(self._api_prefix, self._splash_screen)
                if callback:
                    self._logger.debug('Setting server exit handler before disconnect')
                    assert not self._run_on_server_exit
                    self._run_on_server_exit = callback
                vim.eval('vimspector#internal#{}#StopDebugSession( {} )'.format(self._connection_type, self.session_id))
            self._connection.DoRequest(handler, {'command': 'disconnect', 'arguments': arguments}, failure_handler=handler, timeout=self._connection.sync_timeout)
        if not interactive:
            disconnect()
        elif not self._server_capabilities.get('supportTerminateDebuggee'):
            disconnect()
        elif not self._stackTraceView.AnyThreadsRunning():
            disconnect()
        else:

            def handle_choice(choice):
                if False:
                    return 10
                if choice == 1:
                    arguments['terminateDebuggee'] = True
                elif choice == 2:
                    arguments['terminateDebuggee'] = False
                elif choice <= 0:
                    return
                disconnect()
            utils.Confirm(self._api_prefix, 'Terminate debuggee?', handle_choice, default_value=3, options=['(Y)es', '(N)o', '(D)efault'], keys=['y', 'n', 'd'])

    def _PrepareAttach(self, adapter_config, launch_config):
        if False:
            for i in range(10):
                print('nop')
        attach_config = adapter_config.get('attach')
        if not attach_config:
            return
        if 'remote' in attach_config:
            remote = attach_config['remote']
            remote_exec_cmd = self._GetRemoteExecCommand(remote)
            pid_cmd = remote_exec_cmd + remote['pidCommand']
            self._logger.debug('Getting PID: %s', pid_cmd)
            pid = subprocess.check_output(pid_cmd).decode('utf-8').strip()
            self._logger.debug('Got PID: %s', pid)
            if not pid:
                utils.UserMessage('Unable to get PID', persist=True)
                return
            if 'initCompleteCommand' in remote:
                initcmd = remote_exec_cmd + remote['initCompleteCommand'][:]
                for (index, item) in enumerate(initcmd):
                    initcmd[index] = item.replace('%PID%', pid)
                self._on_init_complete_handlers.append(lambda : subprocess.check_call(initcmd))
            commands = self._GetCommands(remote, 'attach')
            for command in commands:
                cmd = remote_exec_cmd + command
                for (index, item) in enumerate(cmd):
                    cmd[index] = item.replace('%PID%', pid)
                self._logger.debug('Running remote app: %s', cmd)
                self._remote_term = terminal.LaunchTerminal(self._api_prefix, {'args': cmd, 'cwd': os.getcwd()}, self._codeView._window, self._remote_term)
        else:
            if attach_config['pidSelect'] == 'ask':
                prop = attach_config['pidProperty']
                if prop not in launch_config:
                    pid = _SelectProcess()
                    if pid is None:
                        return
                    launch_config[prop] = pid
                return
            elif attach_config['pidSelect'] == 'none':
                return
            raise ValueError('Unrecognised pidSelect {0}'.format(attach_config['pidSelect']))
        if 'delay' in attach_config:
            utils.UserMessage(f"Waiting ( {attach_config['delay']} )...")
            vim.command(f"sleep {attach_config['delay']}")

    def _PrepareLaunch(self, command_line, adapter_config, launch_config):
        if False:
            return 10
        run_config = adapter_config.get('launch', {})
        if 'remote' in run_config:
            remote = run_config['remote']
            remote_exec_cmd = self._GetRemoteExecCommand(remote)
            commands = self._GetCommands(remote, 'run')
            for (index, command) in enumerate(commands):
                cmd = remote_exec_cmd + command[:]
                full_cmd = []
                for item in cmd:
                    if isinstance(command_line, list):
                        if item == '%CMD%':
                            full_cmd.extend(command_line)
                        else:
                            full_cmd.append(item)
                    else:
                        full_cmd.append(item.replace('%CMD%', command_line))
                self._logger.debug('Running remote app: %s', full_cmd)
                self._remote_term = terminal.LaunchTerminal(self._api_prefix, {'args': full_cmd, 'cwd': os.getcwd()}, self._codeView._window, self._remote_term)
        if 'delay' in run_config:
            utils.UserMessage(f"Waiting ( {run_config['delay']} )...")
            vim.command(f"sleep {run_config['delay']}")

    def _GetSSHCommand(self, remote):
        if False:
            print('Hello World!')
        ssh_config = remote.get('ssh', {})
        ssh = ssh_config.get('cmd', ['ssh']) + ssh_config.get('args', [])
        if 'account' in remote:
            ssh.append(remote['account'] + '@' + remote['host'])
        else:
            ssh.append(remote['host'])
        return ssh

    def _GetShellCommand(self):
        if False:
            return 10
        return []

    def _GetDockerCommand(self, remote):
        if False:
            print('Hello World!')
        docker = ['docker', 'exec', '-t']
        docker.append(remote['container'])
        return docker

    def _GetRemoteExecCommand(self, remote):
        if False:
            return 10
        is_ssh_cmd = any((key in remote for key in ['ssh', 'host', 'account']))
        is_docker_cmd = 'container' in remote
        if is_ssh_cmd:
            return self._GetSSHCommand(remote)
        elif is_docker_cmd:
            return self._GetDockerCommand(remote)
        else:
            return self._GetShellCommand()

    def _GetCommands(self, remote, pfx):
        if False:
            print('Hello World!')
        commands = remote.get(pfx + 'Commands', None)
        if isinstance(commands, list):
            return commands
        elif commands is not None:
            raise ValueError('Invalid commands; must be list')
        command = remote[pfx + 'Command']
        if isinstance(command, str):
            command = shlex.split(command)
        if not isinstance(command, list):
            raise ValueError('Invalid command; must be list/string')
        if not command:
            raise ValueError('Could not determine commands for ' + pfx)
        return [command]

    def _Initialise(self):
        if False:
            print('Hello World!')
        self._splash_screen = utils.DisplaySplash(self._api_prefix, self._splash_screen, f'Initializing debug session {self.DisplayName()}...')

        def handle_initialize_response(msg):
            if False:
                print('Hello World!')
            self._server_capabilities = msg.get('body') or {}
            if not self.parent_session:
                self._breakpoints.SetServerCapabilities(self._server_capabilities)
            self._Launch()
        self._connection.DoRequest(handle_initialize_response, {'command': 'initialize', 'arguments': {'adapterID': self._adapter.get('name', 'adapter'), 'clientID': 'vimspector', 'clientName': 'vimspector', 'linesStartAt1': True, 'columnsStartAt1': True, 'locale': 'en_GB', 'pathFormat': 'path', 'supportsVariableType': True, 'supportsVariablePaging': False, 'supportsRunInTerminalRequest': True, 'supportsMemoryReferences': True, 'supportsStartDebuggingRequest': True}})

    def OnFailure(self, reason, request, message):
        if False:
            return 10
        msg = "Request for '{}' failed: {}\nResponse: {}".format(request, reason, message)
        self._outputView.Print('server', msg)

    def _Prepare(self):
        if False:
            print('Hello World!')
        self._on_init_complete_handlers = []
        self._logger.debug('LAUNCH!')
        if self._launch_config is None:
            self._launch_config = {}
            self._launch_config.update(self._adapter.get('configuration', {}))
            self._launch_config.update(self._configuration['configuration'])
        request = self._configuration.get('remote-request', self._launch_config.get('request', 'launch'))
        if request == 'attach':
            self._splash_screen = utils.DisplaySplash(self._api_prefix, self._splash_screen, f'Attaching to debuggee {self.DisplayName()}...')
            self._PrepareAttach(self._adapter, self._launch_config)
        elif request == 'launch':
            self._splash_screen = utils.DisplaySplash(self._api_prefix, self._splash_screen, f'Launching debuggee {self.DisplayName()}...')
            self._PrepareLaunch(self._configuration.get('remote-cmdLine', []), self._adapter, self._launch_config)
        if 'name' not in self._launch_config:
            self._launch_config['name'] = 'test'

    def _Launch(self):
        if False:
            return 10

        def failure_handler(reason, msg):
            if False:
                for i in range(10):
                    print('nop')
            text = [f'Initialize for session {self.DisplayName()} Failed', ''] + reason.splitlines() + ['', 'Use :VimspectorReset to close']
            self._logger.info('Launch failed: %s', '\n'.join(text))
            self._splash_screen = utils.DisplaySplash(self._api_prefix, self._splash_screen, text)
        self._connection.DoRequest(lambda msg: self._OnLaunchComplete(), {'command': self._launch_config['request'], 'arguments': self._launch_config}, failure_handler)

    def _OnLaunchComplete(self):
        if False:
            while True:
                i = 10
        self._launch_complete = True
        self._LoadThreadsIfReady()

    def _OnInitializeComplete(self):
        if False:
            i = 10
            return i + 15
        self._init_complete = True
        self._LoadThreadsIfReady()

    def _LoadThreadsIfReady(self):
        if False:
            for i in range(10):
                print('nop')
        if self._launch_complete and self._init_complete:
            self._splash_screen = utils.HideSplash(self._api_prefix, self._splash_screen)
            for h in self._on_init_complete_handlers:
                h()
            self._on_init_complete_handlers = []
            self._stackTraceView.LoadThreads(self, True)

    @CurrentSession()
    @IfConnected()
    @RequiresUI()
    def PrintDebugInfo(self):
        if False:
            return 10

        def Line():
            if False:
                return 10
            return '--------------------------------------------------------------------------------'

        def Pretty(obj):
            if False:
                while True:
                    i = 10
            if obj is None:
                return ['None']
            return [Line()] + json.dumps(obj, indent=2).splitlines() + [Line()]
        debugInfo = ['Vimspector Debug Info', Line(), f'ConnectionType: {self._connection_type}', 'Adapter: '] + Pretty(self._adapter) + ['Configuration: '] + Pretty(self._configuration) + [f'API Prefix: {self._api_prefix}', f'Launch/Init: {self._launch_complete} / {self._init_complete}', f'Workspace Root: {self._workspace_root}', 'Launch Config: '] + Pretty(self._launch_config) + ['Server Capabilities: '] + Pretty(self._server_capabilities) + ['Line Breakpoints: '] + Pretty(self._breakpoints._line_breakpoints) + ['Func Breakpoints: '] + Pretty(self._breakpoints._func_breakpoints) + ['Ex Breakpoints: '] + Pretty(self._breakpoints._exception_breakpoints)
        self._outputView.ClearCategory('DebugInfo')
        self._outputView.Print('DebugInfo', debugInfo)
        self.ShowOutput('DebugInfo')

    def OnEvent_loadedSource(self, msg):
        if False:
            while True:
                i = 10
        pass

    def OnEvent_capabilities(self, msg):
        if False:
            print('Hello World!')
        self._server_capabilities.update((msg.get('body') or {}).get('capabilities') or {})

    def OnEvent_initialized(self, message):
        if False:
            return 10

        def OnBreakpointsDone():
            if False:
                while True:
                    i = 10
            self._breakpoints.Refresh()
            if self._server_capabilities.get('supportsConfigurationDoneRequest'):
                self._connection.DoRequest(lambda msg: self._OnInitializeComplete(), {'command': 'configurationDone'})
            else:
                self._OnInitializeComplete()
        self._breakpoints.SetConfiguredBreakpoints(self._configuration.get('breakpoints', {}))
        self._breakpoints.AddConnection(self._connection)
        self._breakpoints.UpdateUI(OnBreakpointsDone)

    def OnEvent_thread(self, message):
        if False:
            i = 10
            return i + 15
        self._stackTraceView.OnThreadEvent(self, message['body'])

    def OnEvent_breakpoint(self, message):
        if False:
            print('Hello World!')
        reason = message['body']['reason']
        bp = message['body']['breakpoint']
        if reason == 'changed':
            self._breakpoints.UpdatePostedBreakpoint(self._connection, bp)
        elif reason == 'new':
            self._breakpoints.AddPostedBreakpoint(self._connection, bp)
        elif reason == 'removed':
            self._breakpoints.DeletePostedBreakpoint(self._connection, bp)
        else:
            utils.UserMessage('Unrecognised breakpoint event (undocumented): {0}'.format(reason), persist=True)

    def OnRequest_runInTerminal(self, message):
        if False:
            i = 10
            return i + 15
        params = message['arguments']
        if not params.get('cwd'):
            params['cwd'] = self._workspace_root
            self._logger.debug('Defaulting working directory to %s', params['cwd'])
        term_id = self._codeView.LaunchTerminal(params)
        response = {'processId': int(utils.Call('vimspector#internal#{}term#GetPID'.format(self._api_prefix), term_id))}
        self._connection.DoResponse(message, None, response)

    def OnEvent_terminated(self, message):
        if False:
            for i in range(10):
                print('nop')
        self.SetCurrentFrame(None)

    def OnEvent_exited(self, message):
        if False:
            while True:
                i = 10
        utils.UserMessage('The debuggee exited with status code: {}'.format(message['body']['exitCode']))
        self._stackTraceView.OnExited(self, message)
        self.ClearCurrentPC()

    def OnRequest_startDebugging(self, message):
        if False:
            for i in range(10):
                print('nop')
        self._DoStartDebuggingRequest(message, message['arguments']['request'], message['arguments']['configuration'], self._adapter)

    def _DoStartDebuggingRequest(self, message, request_type, launch_arguments, adapter, session_name=None):
        if False:
            while True:
                i = 10
        session = self.manager.NewSession(session_name=session_name or launch_arguments.get('name'), parent_session=self)
        session._launch_config = launch_arguments
        session._launch_config['request'] = request_type
        session._StartWithConfiguration({'configuration': launch_arguments}, adapter)
        self._connection.DoResponse(message, None, {})

    def OnEvent_process(self, message):
        if False:
            while True:
                i = 10
        utils.UserMessage('debuggee was started: {}'.format(message['body']['name']))

    def OnEvent_module(self, message):
        if False:
            while True:
                i = 10
        pass

    def OnEvent_continued(self, message):
        if False:
            return 10
        self._stackTraceView.OnContinued(self, message['body'])
        self.ClearCurrentPC()

    @ParentOnly()
    def Clear(self):
        if False:
            for i in range(10):
                print('nop')
        self._codeView.Clear()
        if self._disassemblyView:
            self._disassemblyView.Clear()
        self._stackTraceView.Clear()
        self._variablesView.Clear()

    def OnServerExit(self, status):
        if False:
            return 10
        self._logger.info('The server has terminated with status %s', status)
        if self._connection is not None:
            self._connection.Reset()
        self._stackTraceView.ConnectionClosed(self)
        self._breakpoints.ConnectionClosed(self._connection)
        self._variablesView.ConnectionClosed(self._connection)
        if self._disassemblyView:
            self._disassemblyView.ConnectionClosed(self._connection)
        self.Clear()
        self._ResetServerState()
        if self._run_on_server_exit:
            self._logger.debug('Running server exit handler')
            callback = self._run_on_server_exit
            self._run_on_server_exit = None
            callback()
        else:
            self._logger.debug('No server exit handler')

    def OnEvent_output(self, message):
        if False:
            i = 10
            return i + 15
        if self._outputView:
            self._outputView.OnOutput(message['body'])

    def OnEvent_stopped(self, message):
        if False:
            return 10
        event = message['body']
        reason = event.get('reason') or '<protocol error>'
        description = event.get('description')
        text = event.get('text')
        if description:
            explanation = description + '(' + reason + ')'
        else:
            explanation = reason
        if text:
            explanation += ': ' + text
        msg = 'Paused in thread {0} due to {1}'.format(event.get('threadId', '<unknown>'), explanation)
        utils.UserMessage(msg)
        if self._outputView:
            self._outputView.Print('server', msg)
        self._stackTraceView.OnStopped(self, event)

    def BreakpointsAsQuickFix(self):
        if False:
            i = 10
            return i + 15
        return self._breakpoints.BreakpointsAsQuickFix()

    def ListBreakpoints(self):
        if False:
            print('Hello World!')
        self._breakpoints.ToggleBreakpointsView()

    def ToggleBreakpointViewBreakpoint(self):
        if False:
            print('Hello World!')
        self._breakpoints.ToggleBreakpointViewBreakpoint()

    def ToggleAllBreakpointsViewBreakpoint(self):
        if False:
            i = 10
            return i + 15
        self._breakpoints.ToggleAllBreakpointsViewBreakpoint()

    def DeleteBreakpointViewBreakpoint(self):
        if False:
            i = 10
            return i + 15
        self._breakpoints.ClearBreakpointViewBreakpoint()

    def JumpToBreakpointViewBreakpoint(self):
        if False:
            for i in range(10):
                print('nop')
        self._breakpoints.JumpToBreakpointViewBreakpoint()

    def EditBreakpointOptionsViewBreakpoint(self):
        if False:
            while True:
                i = 10
        self._breakpoints.EditBreakpointOptionsViewBreakpoint()

    def JumpToNextBreakpoint(self):
        if False:
            i = 10
            return i + 15
        self._breakpoints.JumpToNextBreakpoint()

    def JumpToPreviousBreakpoint(self):
        if False:
            while True:
                i = 10
        self._breakpoints.JumpToPreviousBreakpoint()

    def JumpToProgramCounter(self):
        if False:
            print('Hello World!')
        self._stackTraceView.JumpToProgramCounter()

    def ToggleBreakpoint(self, options):
        if False:
            for i in range(10):
                print('nop')
        return self._breakpoints.ToggleBreakpoint(options)

    def RunTo(self, file_name, line):
        if False:
            return 10
        self._breakpoints.ClearTemporaryBreakpoints()
        self._breakpoints.AddTemporaryLineBreakpoint(file_name, line, {'temporary': True}, lambda : self.Continue())

    @CurrentSession()
    @IfConnected()
    def GoTo(self, file_name, line):
        if False:
            while True:
                i = 10

        def failure_handler(reason, *args):
            if False:
                return 10
            utils.UserMessage(f"Can't jump to location: {reason}", error=True)

        def handle_targets(msg):
            if False:
                i = 10
                return i + 15
            targets = msg.get('body', {}).get('targets', [])
            if not targets:
                failure_handler('No targets')
                return
            if len(targets) == 1:
                target_selected = 0
            else:
                target_selected = utils.SelectFromList('Which target?', [t['label'] for t in targets], ret='index')
            if target_selected is None:
                return
            self._connection.DoRequest(None, {'command': 'goto', 'arguments': {'threadId': self._stackTraceView.GetCurrentThreadId(), 'targetId': targets[target_selected]['id']}}, failure_handler)
        if not self._server_capabilities.get('supportsGotoTargetsRequest', False):
            failure_handler("Server doesn't support it")
            return
        self._connection.DoRequest(handle_targets, {'command': 'gotoTargets', 'arguments': {'source': {'path': utils.NormalizePath(file_name)}, 'line': line}}, failure_handler)

    def SetLineBreakpoint(self, file_name, line_num, options, then=None):
        if False:
            while True:
                i = 10
        return self._breakpoints.SetLineBreakpoint(file_name, line_num, options, then)

    def ClearLineBreakpoint(self, file_name, line_num):
        if False:
            i = 10
            return i + 15
        return self._breakpoints.ClearLineBreakpoint(file_name, line_num)

    def ClearBreakpoints(self):
        if False:
            for i in range(10):
                print('nop')
        return self._breakpoints.ClearBreakpoints()

    def ResetExceptionBreakpoints(self):
        if False:
            print('Hello World!')
        return self._breakpoints.ResetExceptionBreakpoints()

    def AddFunctionBreakpoint(self, function, options):
        if False:
            return 10
        return self._breakpoints.AddFunctionBreakpoint(function, options)

def PathsToAllGadgetConfigs(vimspector_base, current_file):
    if False:
        return 10
    yield install.GetGadgetConfigFile(vimspector_base)
    for p in sorted(glob.glob(os.path.join(install.GetGadgetConfigDir(vimspector_base), '*.json'))):
        yield p
    yield utils.PathToConfigFile('.gadgets.json', os.path.dirname(current_file))

def PathsToAllConfigFiles(vimspector_base, current_file, filetypes):
    if False:
        print('Hello World!')
    for ft in filetypes + ['_all']:
        for p in sorted(glob.glob(os.path.join(install.GetConfigDirForFiletype(vimspector_base, ft), '*.json'))):
            yield p
    for ft in filetypes:
        yield utils.PathToConfigFile(f'.vimspector.{ft}.json', os.path.dirname(current_file))
    yield utils.PathToConfigFile('.vimspector.json', os.path.dirname(current_file))

def _SelectProcess(*args):
    if False:
        return 10
    value = 0
    custom_picker = settings.Get('custom_process_picker_func')
    if custom_picker:
        try:
            value = utils.Call(custom_picker, *args)
        except vim.error:
            pass
    else:
        vimspector_process_list: str = None
        try:
            try:
                vimspector_process_list = installer.FindExecutable('vimspector_process_list')
            except installer.MissingExecutable:
                vimspector_process_list = installer.FindExecutable('vimspector_process_list', [os.path.join(install.GetSupportDir(), 'vimspector_process_list')])
        except installer.MissingExecutable:
            pass
        default_pid = None
        if vimspector_process_list:
            output = subprocess.check_output((vimspector_process_list,) + args).decode('utf-8')
            lines = output.splitlines()
            if len(lines) == 2:
                default_pid = lines[-1].split()[0]
            utils.UserMessage(lines)
        value = utils.AskForInput('Enter Process ID: ', default_value=default_pid)
    if value:
        try:
            return int(value)
        except ValueError:
            return 0
    return 0