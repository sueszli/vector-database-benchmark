import sys
from threading import current_thread
from tkinter import BOTH, Button, END, Entry, Frame, Label, LEFT, Listbox, Tk, Toplevel, W
from typing import Any, Union

class TkDialog(Toplevel):
    left_button = 'OK'
    right_button = 'Cancel'

    def __init__(self, message, value=None, **config):
        if False:
            while True:
                i = 10
        self._prevent_execution_with_timeouts()
        self.root = self._get_root()
        self._button_bindings = {}
        super().__init__(self.root)
        self._initialize_dialog()
        self.widget = self._create_body(message, value, **config)
        self._create_buttons()
        self._finalize_dialog()
        self._result = None

    def _prevent_execution_with_timeouts(self):
        if False:
            print('Hello World!')
        if 'linux' not in sys.platform and current_thread().name != 'MainThread':
            raise RuntimeError('Dialogs library is not supported with timeouts on Python on this platform.')

    def _get_root(self) -> Tk:
        if False:
            return 10
        root = Tk()
        root.withdraw()
        return root

    def _initialize_dialog(self):
        if False:
            while True:
                i = 10
        self.withdraw()
        self.title('Robot Framework')
        self.protocol('WM_DELETE_WINDOW', self._close)
        self.bind('<Escape>', self._close)
        if self.left_button == TkDialog.left_button:
            self.bind('<Return>', self._left_button_clicked)

    def _finalize_dialog(self):
        if False:
            for i in range(10):
                print('nop')
        self.update()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        min_width = screen_width // 6
        min_height = screen_height // 10
        width = max(self.winfo_reqwidth(), min_width)
        height = max(self.winfo_reqheight(), min_height)
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.geometry(f'{width}x{height}+{x}+{y}')
        self.lift()
        self.deiconify()
        if self.widget:
            self.widget.focus_set()

    def _create_body(self, message, value, **config) -> Union[Entry, Listbox, None]:
        if False:
            return 10
        frame = Frame(self)
        max_width = self.winfo_screenwidth() // 2
        label = Label(frame, text=message, anchor=W, justify=LEFT, wraplength=max_width)
        label.pack(fill=BOTH)
        widget = self._create_widget(frame, value, **config)
        if widget:
            widget.pack(fill=BOTH)
        frame.pack(padx=5, pady=5, expand=1, fill=BOTH)
        return widget

    def _create_widget(self, frame, value) -> Union[Entry, Listbox, None]:
        if False:
            return 10
        return None

    def _create_buttons(self):
        if False:
            while True:
                i = 10
        frame = Frame(self)
        self._create_button(frame, self.left_button, self._left_button_clicked)
        self._create_button(frame, self.right_button, self._right_button_clicked)
        frame.pack()

    def _create_button(self, parent, label, callback):
        if False:
            print('Hello World!')
        if label:
            button = Button(parent, text=label, width=10, command=callback, underline=0)
            button.pack(side=LEFT, padx=5, pady=5)
            for char in (label[0].upper(), label[0].lower()):
                self.bind(char, callback)
                self._button_bindings[char] = callback

    def _left_button_clicked(self, event=None):
        if False:
            while True:
                i = 10
        if self._validate_value():
            self._result = self._get_value()
            self._close()

    def _validate_value(self) -> bool:
        if False:
            while True:
                i = 10
        return True

    def _get_value(self) -> Any:
        if False:
            i = 10
            return i + 15
        return None

    def _close(self, event=None):
        if False:
            i = 10
            return i + 15
        self.root.destroy()

    def _right_button_clicked(self, event=None):
        if False:
            print('Hello World!')
        self._result = self._get_right_button_value()
        self._close()

    def _get_right_button_value(self) -> Any:
        if False:
            return 10
        return None

    def show(self) -> Any:
        if False:
            return 10
        self.wait_window(self)
        return self._result

class MessageDialog(TkDialog):
    right_button = None

class InputDialog(TkDialog):

    def __init__(self, message, default='', hidden=False):
        if False:
            while True:
                i = 10
        super().__init__(message, default, hidden=hidden)

    def _create_widget(self, parent, default, hidden=False) -> Entry:
        if False:
            print('Hello World!')
        widget = Entry(parent, show='*' if hidden else '')
        widget.insert(0, default)
        widget.select_range(0, END)
        widget.bind('<FocusIn>', self._unbind_buttons)
        widget.bind('<FocusOut>', self._rebind_buttons)
        return widget

    def _unbind_buttons(self, event):
        if False:
            while True:
                i = 10
        for char in self._button_bindings:
            self.unbind(char)

    def _rebind_buttons(self, event):
        if False:
            return 10
        for (char, callback) in self._button_bindings.items():
            self.bind(char, callback)

    def _get_value(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.widget.get()

class SelectionDialog(TkDialog):

    def _create_widget(self, parent, values) -> Listbox:
        if False:
            print('Hello World!')
        widget = Listbox(parent)
        for item in values:
            widget.insert(END, item)
        widget.config(width=0)
        return widget

    def _validate_value(self) -> bool:
        if False:
            return 10
        return bool(self.widget.curselection())

    def _get_value(self) -> str:
        if False:
            while True:
                i = 10
        return self.widget.get(self.widget.curselection())

class MultipleSelectionDialog(TkDialog):

    def _create_widget(self, parent, values) -> Listbox:
        if False:
            print('Hello World!')
        widget = Listbox(parent, selectmode='multiple')
        for item in values:
            widget.insert(END, item)
        widget.config(width=0)
        return widget

    def _get_value(self) -> list:
        if False:
            i = 10
            return i + 15
        selected_values = [self.widget.get(i) for i in self.widget.curselection()]
        return selected_values

class PassFailDialog(TkDialog):
    left_button = 'PASS'
    right_button = 'FAIL'

    def _get_value(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def _get_right_button_value(self) -> bool:
        if False:
            print('Hello World!')
        return False