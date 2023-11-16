import logging
import vim
from vimspector import signs, utils
from vimspector.debug_adapter_connection import DebugAdapterConnection
SIGN_ID = 1

class DisassemblyView(object):

    def __init__(self, window, api_prefix, render_event_emitter):
        if False:
            for i in range(10):
                print('nop')
        self._logger = logging.getLogger(__name__)
        utils.SetUpLogging(self._logger)
        self._render_emitter = render_event_emitter
        self._render_emitter.subscribe(self._DisplayPC)
        self._window = window
        self._buf = None
        self._requesting = False
        self._api_prefix = api_prefix
        self.current_connection: DebugAdapterConnection = None
        self.current_frame = None
        self.current_instructions = None
        self._scratch_buffers = []
        self._signs = {'vimspectorPC': None}
        self._RenderWinBar()
        with utils.LetCurrentWindow(self._window):
            vim.command('augroup VimspectorDisassembly')
            vim.command('autocmd!')
            vim.command(f'autocmd WinScrolled {utils.WindowID(self._window)} call vimspector#internal#disassembly#OnWindowScrolled()')
            vim.command('augroup END')
        signs.DefineProgramCounterSigns()

    def _RenderWinBar(self):
        if False:
            while True:
                i = 10
        with utils.LetCurrentWindow(self._window):
            if utils.UseWinBar():
                utils.SetWinBar(('■ Stop', 'vimspector#Stop()'), ('▶ Cont', 'vimspector#Continue()'), ('▷ Pause', 'vimspector#Pause()'), ('↷ Next', 'vimspector#StepIOver()'), ('→ Step', 'vimspector#StepIInto()'), ('← Out', 'vimspector#StepIOut()'), ('↺', 'vimspector#Restart()'), ('✕', 'vimspector#Reset()'))

    def ConnectionClosed(self, connection: DebugAdapterConnection):
        if False:
            return 10
        if connection != self.current_connection:
            return
        self._UndisplayPC()
        self.current_connection = None
        self.current_frame = None
        self.current_instructions = None

    def WindowIsValid(self):
        if False:
            while True:
                i = 10
        return self._window is not None and self._window.valid

    def IsCurrent(self):
        if False:
            print('Hello World!')
        return vim.current.buffer == self._buf

    def SetCurrentFrame(self, connection: DebugAdapterConnection, frame, should_jump_to_location):
        if False:
            while True:
                i = 10
        if not self._window.valid:
            return
        if not frame or not connection:
            self._UndisplayPC()
            return
        if 'instructionPointerReference' not in frame:
            self._UndisplayPC()
            return
        if self._requesting:
            self._UndisplayPC()
            return
        self._instructionPointerReference = frame['instructionPointerReference']
        self._instructionPointerAddressOffset = 0
        self.current_frame = frame
        self.current_connection = connection
        self.instruction_offset = -self._window.height
        self.instruction_count = self._window.height * 2
        self._RequestInstructions(should_jump_to_location, should_make_visible=True)

    def _RequestInstructions(self, should_jump_to_location, should_make_visible, offset_cursor_by=0):
        if False:
            return 10
        assert not self._requesting
        assert self.instruction_offset <= 0
        assert self._instructionPointerAddressOffset == 0

        def handler(msg):
            if False:
                print('Hello World!')
            self.current_instructions = msg.get('body', {}).get('instructions')
            self._DrawInstructions(should_jump_to_location, should_make_visible, offset_cursor_by)
            self._requesting = False

        def error_handler():
            if False:
                i = 10
                return i + 15
            self._requesting = False
        self._requesting = True
        self.current_connection.DoRequest(handler, {'command': 'disassemble', 'arguments': {'memoryReference': self._instructionPointerReference, 'offset': int(self._instructionPointerAddressOffset), 'instructionOffset': int(self.instruction_offset), 'instructionCount': int(self.instruction_count), 'resolveSymbols': True}}, failure_handler=lambda *args: error_handler())

    def Clear(self):
        if False:
            print('Hello World!')
        self._UndisplayPC()
        with utils.ModifiableScratchBuffer(self._buf):
            utils.ClearBuffer(self._buf)
        self.current_connection = None
        self.current_frame = None
        self.current_instructions = None

    def Reset(self):
        if False:
            i = 10
            return i + 15
        self.Clear()
        vim.command('autocmd! VimspectorDisassembly')
        self._buf = None
        for b in self._scratch_buffers:
            utils.CleanUpHiddenBuffer(b)
        self._scratch_buffers = []

    def IsDisassemblyBuffer(self, file_name):
        if False:
            return 10
        if not self._buf:
            return False
        return utils.NormalizePath(file_name) == utils.NormalizePath(self._buf.name)

    def GetMemoryReference(self):
        if False:
            while True:
                i = 10
        return self._instructionPointerReference

    def GetOffsetForLine(self, line_num):
        if False:
            return 10
        if line_num <= 0 or line_num > self.instruction_count:
            return None
        pc = utils.ParseAddress(self.current_instructions[self._GetPCEntryOffset()]['address'])
        req = utils.ParseAddress(self.current_instructions[line_num - 1]['address'])
        return req - pc

    def ResolveAddressAtLine(self, line_num):
        if False:
            while True:
                i = 10
        if line_num <= 0 or line_num > self.instruction_count:
            return None
        return (self.current_connection, utils.ParseAddress(self.current_instructions[line_num - 1]['address']))

    def FindLineForAddress(self, conn, address):
        if False:
            print('Hello World!')
        if not self.current_instructions:
            return 0
        if self.current_connection != conn:
            return 0
        for (index, instruction) in enumerate(self.current_instructions):
            the_addr = utils.ParseAddress(instruction['address'])
            if the_addr == address:
                return index + 1
        return 0

    def _GetPCEntryOffset(self):
        if False:
            i = 10
            return i + 15
        return -self.instruction_offset

    def GetBufferName(self):
        if False:
            print('Hello World!')
        if not self._buf:
            return None
        return self._buf.name

    def OnWindowScrolled(self, win_id):
        if False:
            return 10
        if not self.current_connection:
            return
        if not self.WindowIsValid() or utils.WindowID(self._window) != win_id:
            return
        if self._requesting:
            return
        current_topline = int(utils.GetWindowInfo(self._window)['topline'])
        window_height = self._window.height
        if current_topline == 1:
            self.instruction_offset -= window_height
            self.instruction_count += window_height
            self._RequestInstructions(should_jump_to_location=False, should_make_visible=False, offset_cursor_by=window_height)
        elif current_topline >= len(self._buf) - window_height:
            self.instruction_offset = min(0, self.instruction_offset + window_height)
            self.instruction_count += window_height
            self._RequestInstructions(should_jump_to_location=False, should_make_visible=False)

    def _DrawInstructions(self, should_jump_to_location, should_make_visible, offset_cursor_by):
        if False:
            i = 10
            return i + 15
        if not self._window.valid:
            return
        if not self.current_instructions:
            return
        buf_name = '_vimspector_disassembly'
        file_name = (self.current_frame.get('source') or {}).get('path') or ''
        self._buf = utils.BufferForFile(buf_name)
        utils.Call('setbufvar', self._buf.number, 'vimspector_disassembly_path', file_name)
        utils.Call('setbufvar', self._buf.number, '&filetype', 'vimspector-disassembly')
        self._scratch_buffers.append(self._buf)
        utils.SetUpHiddenBuffer(self._buf, buf_name)
        instruction_bytes_len = max((len(i.get('instructionBytes', '')) for i in self.current_instructions))
        if not instruction_bytes_len:
            instruction_bytes_len = 1
        with utils.ModifiableScratchBuffer(self._buf):
            utils.SetBufferContents(self._buf, [f"{utils.Hex(utils.ParseAddress(i['address']))}:\t{i.get('instructionBytes', ''):{instruction_bytes_len}}\t{i['instruction']}" for i in self.current_instructions])
        with utils.LetCurrentWindow(self._window):
            utils.OpenFileInCurrentWindow(buf_name)
            if utils.VimIsNeovim():
                self._RenderWinBar()
            utils.SetUpUIWindow(self._window)
            self._window.options['signcolumn'] = 'yes'
        self._render_emitter.emit()
        assert not should_jump_to_location or offset_cursor_by == 0
        try:
            if should_jump_to_location:
                utils.JumpToWindow(self._window)
                utils.SetCursorPosInWindow(self._window, self._GetPCEntryOffset() + 1, 1, make_visible=utils.VisiblePosition.MIDDLE)
            elif should_make_visible:
                with utils.RestoreCursorPosition():
                    utils.SetCursorPosInWindow(self._window, self._GetPCEntryOffset() + 1, 1, make_visible=utils.VisiblePosition.MIDDLE)
        except vim.error as e:
            utils.UserMessage(f'Failed to set cursor position for disassembly: {e}', error=True)
        if offset_cursor_by != 0:
            self._window.cursor = (self._window.cursor[0] + offset_cursor_by, self._window.cursor[1])

    def _DisplayPC(self):
        if False:
            return 10
        self._UndisplayPC()
        if not self.current_connection or not self._buf or (not self.current_instructions):
            return
        if len(self.current_instructions) < self.instruction_count:
            self._logger.warn('Invalid number of instructions returned by adapter: Requested: %s, but got %s', self.instruction_count, len(self.current_instructions))
            return
        self._signs['vimspectorPC'] = SIGN_ID * 92
        pc_line = self._GetPCEntryOffset() + 1
        signs.PlaceSign(self._signs['vimspectorPC'], 'VimspectorDisassembly', 'vimspectorPC', self._buf.name, pc_line)

    def _UndisplayPC(self):
        if False:
            while True:
                i = 10
        if self._signs['vimspectorPC']:
            signs.UnplaceSign(self._signs['vimspectorPC'], 'VimspectorDisassembly')
            self._signs['vimspectorPC'] = None