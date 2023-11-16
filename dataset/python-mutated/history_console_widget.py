from qtpy import QtGui
from traitlets import Bool
from .console_widget import ConsoleWidget

class HistoryConsoleWidget(ConsoleWidget):
    """ A ConsoleWidget that keeps a history of the commands that have been
        executed and provides a readline-esque interface to this history.
    """
    history_lock = Bool(False, config=True)

    def __init__(self, *args, **kw):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kw)
        self._history = []
        self._history_edits = {}
        self._history_index = 0
        self._history_prefix = ''

    def do_execute(self, source, complete, indent):
        if False:
            i = 10
            return i + 15
        ' Reimplemented to the store history. '
        history = self.input_buffer if source is None else source
        super().do_execute(source, complete, indent)
        if complete:
            history = history.rstrip()
            if history and (not self._history or self._history[-1] != history):
                self._history.append(history)
            self._history_edits = {}
            self._history_index = len(self._history)

    def _up_pressed(self, shift_modifier):
        if False:
            i = 10
            return i + 15
        ' Called when the up key is pressed. Returns whether to continue\n            processing the event.\n        '
        prompt_cursor = self._get_prompt_cursor()
        if self._get_cursor().blockNumber() == prompt_cursor.blockNumber():
            if self._history_locked() and (not shift_modifier):
                return False
            pos = self._get_input_buffer_cursor_pos()
            input_buffer = self.input_buffer
            n = min(pos, len(self._history_prefix))
            if self._history_prefix[:n] != input_buffer[:n]:
                self._history_index = len(self._history)
            c = self._get_cursor()
            current_pos = c.position()
            c.movePosition(QtGui.QTextCursor.EndOfBlock)
            at_eol = c.position() == current_pos
            if self._history_index == len(self._history) or not (self._history_prefix == '' and at_eol) or (not self._get_edited_history(self._history_index)[:pos] == input_buffer[:pos]):
                self._history_prefix = input_buffer[:pos]
            self.history_previous(self._history_prefix, as_prefix=not shift_modifier)
            cursor = self._get_prompt_cursor()
            if self._history_prefix:
                cursor.movePosition(QtGui.QTextCursor.Right, n=len(self._history_prefix))
            else:
                cursor.movePosition(QtGui.QTextCursor.EndOfBlock)
            self._set_cursor(cursor)
            return False
        return True

    def _down_pressed(self, shift_modifier):
        if False:
            return 10
        ' Called when the down key is pressed. Returns whether to continue\n            processing the event.\n        '
        end_cursor = self._get_end_cursor()
        if self._get_cursor().blockNumber() == end_cursor.blockNumber():
            if self._history_locked() and (not shift_modifier):
                return False
            replaced = self.history_next(self._history_prefix, as_prefix=not shift_modifier)
            if self._history_prefix and replaced:
                cursor = self._get_prompt_cursor()
                cursor.movePosition(QtGui.QTextCursor.Right, n=len(self._history_prefix))
                self._set_cursor(cursor)
            return False
        return True

    def history_previous(self, substring='', as_prefix=True):
        if False:
            while True:
                i = 10
        ' If possible, set the input buffer to a previous history item.\n\n        Parameters\n        ----------\n        substring : str, optional\n            If specified, search for an item with this substring.\n        as_prefix : bool, optional\n            If True, the substring must match at the beginning (default).\n\n        Returns\n        -------\n        Whether the input buffer was changed.\n        '
        index = self._history_index
        replace = False
        while index > 0:
            index -= 1
            history = self._get_edited_history(index)
            if history == self.input_buffer:
                continue
            if as_prefix and history.startswith(substring) or (not as_prefix and substring in history):
                replace = True
                break
        if replace:
            self._store_edits()
            self._history_index = index
            self.input_buffer = history
        return replace

    def history_next(self, substring='', as_prefix=True):
        if False:
            while True:
                i = 10
        ' If possible, set the input buffer to a subsequent history item.\n\n        Parameters\n        ----------\n        substring : str, optional\n            If specified, search for an item with this substring.\n        as_prefix : bool, optional\n            If True, the substring must match at the beginning (default).\n\n        Returns\n        -------\n        Whether the input buffer was changed.\n        '
        index = self._history_index
        replace = False
        while index < len(self._history):
            index += 1
            history = self._get_edited_history(index)
            if history == self.input_buffer:
                continue
            if as_prefix and history.startswith(substring) or (not as_prefix and substring in history):
                replace = True
                break
        if replace:
            self._store_edits()
            self._history_index = index
            self.input_buffer = history
        return replace

    def history_tail(self, n=10):
        if False:
            while True:
                i = 10
        ' Get the local history list.\n\n        Parameters\n        ----------\n        n : int\n            The (maximum) number of history items to get.\n        '
        return self._history[-n:]

    def _history_locked(self):
        if False:
            i = 10
            return i + 15
        ' Returns whether history movement is locked.\n        '
        return self.history_lock and self._get_edited_history(self._history_index) != self.input_buffer and (self._get_prompt_cursor().blockNumber() != self._get_end_cursor().blockNumber())

    def _get_edited_history(self, index):
        if False:
            print('Hello World!')
        ' Retrieves a history item, possibly with temporary edits.\n        '
        if index in self._history_edits:
            return self._history_edits[index]
        elif index == len(self._history):
            return str()
        return self._history[index]

    def _set_history(self, history):
        if False:
            print('Hello World!')
        ' Replace the current history with a sequence of history items.\n        '
        self._history = list(history)
        self._history_edits = {}
        self._history_index = len(self._history)

    def _store_edits(self):
        if False:
            i = 10
            return i + 15
        ' If there are edits to the current input buffer, store them.\n        '
        current = self.input_buffer
        if self._history_index == len(self._history) or self._history[self._history_index] != current:
            self._history_edits[self._history_index] = current