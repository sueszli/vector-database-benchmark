"""

The ``LineEdit`` and ``MultiLineEdit`` widgets provide a way for the user
to input text.


.. UIExample:: 100

    from flexx import app, event, ui

    class Example(ui.Widget):

        def init(self):
            with ui.VBox():
                self.line = ui.LineEdit(placeholder_text='type here')
                self.l1 = ui.Label(html='<i>when user changes text</i>')
                self.l2 = ui.Label(html='<i>when unfocusing or hitting enter </i>')
                self.l3 = ui.Label(html='<i>when submitting (hitting enter)</i>')
                ui.Widget(flex=1)

        @event.reaction('line.user_text')
        def when_user_changes_text(self, *events):
            self.l1.set_text('user_text: ' + self.line.text)

        @event.reaction('line.user_done')
        def when_user_is_done_changing_text(self, *events):
            self.l2.set_text('user_done: ' + self.line.text)

        @event.reaction('line.submit')
        def when_user_submits_text(self, *events):
            self.l3.set_text('submit: ' + self.line.text)

"""
from ... import event
from . import Widget

class LineEdit(Widget):
    """ An input widget to edit a line of text.

    The ``node`` of this widget is a text
    `<input> <https://developer.mozilla.org/docs/Web/HTML/Element/input>`_.
    """
    DEFAULT_MIN_SIZE = (100, 28)
    CSS = '\n    .flx-LineEdit {\n        color: #333;\n        padding: 0.2em 0.4em;\n        border-radius: 3px;\n        border: 1px solid #aaa;\n        margin: 2px;\n    }\n    .flx-LineEdit:focus  {\n        outline: none;\n        box-shadow: 0px 0px 3px 1px rgba(0, 100, 200, 0.7);\n    }\n    '
    text = event.StringProp(settable=True, doc='\n        The current text of the line edit. Settable. If this is an empty\n        string, the placeholder_text is displayed instead.\n        ')
    password_mode = event.BoolProp(False, settable=True, doc='\n        Whether the insered text should be hidden.\n        ')
    placeholder_text = event.StringProp(settable=True, doc='\n        The placeholder text (shown when the text is an empty string).\n        ')
    autocomp = event.TupleProp(settable=True, doc='\n        A tuple/list of strings for autocompletion. Might not work in all browsers.\n        ')
    disabled = event.BoolProp(False, settable=True, doc='\n        Whether the line edit is disabled.\n        ')

    def _create_dom(self):
        if False:
            i = 10
            return i + 15
        global window
        node = window.document.createElement('input')
        node.setAttribute('type', 'input')
        node.setAttribute('list', self.id)
        self._autocomp = window.document.createElement('datalist')
        self._autocomp.id = self.id
        node.appendChild(self._autocomp)
        f1 = lambda : self.user_text(self.node.value)
        self._addEventListener(node, 'input', f1, False)
        self._addEventListener(node, 'blur', self.user_done, False)
        return node

    @event.emitter
    def user_text(self, text):
        if False:
            i = 10
            return i + 15
        ' Event emitted when the user edits the text. Has ``old_value``\n        and ``new_value`` attributes.\n        '
        d = {'old_value': self.text, 'new_value': text}
        self.set_text(text)
        return d

    @event.emitter
    def user_done(self):
        if False:
            return 10
        ' Event emitted when the user is done editing the text, either by\n        moving the focus elsewhere, or by hitting enter.\n        Has ``old_value`` and ``new_value`` attributes (which are the same).\n        '
        d = {'old_value': self.text, 'new_value': self.text}
        return d

    @event.emitter
    def submit(self):
        if False:
            return 10
        ' Event emitted when the user strikes the enter or return key\n        (but not when losing focus). Has ``old_value`` and ``new_value``\n        attributes (which are the same).\n        '
        self.user_done()
        d = {'old_value': self.text, 'new_value': self.text}
        return d

    @event.emitter
    def key_down(self, e):
        if False:
            for i in range(10):
                print('nop')
        ev = super().key_down(e)
        pkeys = ('Escape',)
        if ev.modifiers and ev.modifiers != ('Shift',) or ev.key in pkeys:
            pass
        else:
            e.stopPropagation()
        if ev.key in ('Enter', 'Return'):
            self.submit()
        elif ev.key == 'Escape':
            self.node.blur()
        return ev

    @event.reaction
    def __text_changed(self):
        if False:
            while True:
                i = 10
        self.node.value = self.text

    @event.reaction
    def __password_mode_changed(self):
        if False:
            return 10
        self.node.type = ['text', 'password'][int(bool(self.password_mode))]

    @event.reaction
    def __placeholder_text_changed(self):
        if False:
            print('Hello World!')
        self.node.placeholder = self.placeholder_text

    @event.reaction
    def __autocomp_changed(self):
        if False:
            print('Hello World!')
        global window
        autocomp = self.autocomp
        for op in self._autocomp:
            self._autocomp.removeChild(op)
        for option in autocomp:
            op = window.document.createElement('option')
            op.value = option
            self._autocomp.appendChild(op)

    @event.reaction
    def __disabled_changed(self):
        if False:
            return 10
        if self.disabled:
            self.node.setAttribute('disabled', 'disabled')
        else:
            self.node.removeAttribute('disabled')

class MultiLineEdit(Widget):
    """ An input widget to edit multiple lines of text.

    The ``node`` of this widget is a
    `<textarea> <https://developer.mozilla.org/docs/Web/HTML/Element/textarea>`_.
    """
    DEFAULT_MIN_SIZE = (100, 50)
    CSS = '\n        .flx-MultiLineEdit {\n            resize: none;\n            overflow-y: scroll;\n            color: #333;\n            padding: 0.2em 0.4em;\n            border-radius: 3px;\n            border: 1px solid #aaa;\n            margin: 2px;\n        }\n        .flx-MultiLineEdit:focus  {\n            outline: none;\n            box-shadow: 0px 0px 3px 1px rgba(0, 100, 200, 0.7);\n        }\n    '
    text = event.StringProp(settable=True, doc='\n        The current text of the multi-line edit. Settable. If this is an empty\n        string, the placeholder_text is displayed instead.\n        ')

    def _create_dom(self):
        if False:
            return 10
        node = window.document.createElement('textarea')
        f1 = lambda : self.user_text(self.node.value)
        self._addEventListener(node, 'input', f1, False)
        self._addEventListener(node, 'blur', self.user_done, False)
        return node

    @event.reaction
    def __text_changed(self):
        if False:
            for i in range(10):
                print('nop')
        self.node.value = self.text

    @event.emitter
    def user_text(self, text):
        if False:
            print('Hello World!')
        ' Event emitted when the user edits the text. Has ``old_value``\n        and ``new_value`` attributes.\n        '
        d = {'old_value': self.text, 'new_value': text}
        self.set_text(text)
        return d

    @event.emitter
    def user_done(self):
        if False:
            print('Hello World!')
        ' Event emitted when the user is done editing the text by\n        moving the focus elsewhere. Has ``old_value`` and ``new_value``\n        attributes (which are the same).\n        '
        d = {'old_value': self.text, 'new_value': self.text}
        return d