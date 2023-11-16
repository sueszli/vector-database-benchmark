""" Button classes

Simple example:

.. UIExample:: 50

    b = ui.Button(text="Push me")


Also see examples: :ref:`buttons.py`.

"""
from ... import event
from .._widget import Widget

class BaseButton(Widget):
    """ Abstract button class.
    """
    DEFAULT_MIN_SIZE = (10, 24)
    CSS = "\n\n    .flx-BaseButton {\n        white-space: nowrap;\n        padding: 0.2em 0.4em;\n        border-radius: 3px;\n        color: #333;\n    }\n    .flx-BaseButton, .flx-BaseButton > input {\n        margin: 2px; /* room for outline */\n    }\n    .flx-BaseButton:focus, .flx-BaseButton > input:focus  {\n        outline: none;\n        box-shadow: 0px 0px 3px 1px rgba(0, 100, 200, 0.7);\n    }\n\n    .flx-Button, .flx-ToggleButton{\n        background: #e8e8e8;\n        border: 1px solid #ccc;\n        transition: background 0.3s;\n    }\n    .flx-Button:hover, .flx-ToggleButton:hover {\n        background: #e8eaff;\n    }\n\n    .flx-ToggleButton {\n        text-align: left;\n    }\n    .flx-ToggleButton.flx-checked {\n        background: #e8eaff;\n    }\n    .flx-ToggleButton::before {\n        content: '\\2610\\00a0 ';\n    }\n    .flx-ToggleButton.flx-checked::before {\n        content: '\\2611\\00a0 ';\n    }\n\n    .flx-RadioButton > input, .flx-CheckBox > input{\n        margin-left: 0.3em;\n        margin-right: 0.3em;\n    }\n\n    .flx-RadioButton > input, .flx-CheckBox > input {\n        color: #333;\n    }\n    .flx-RadioButton:hover > input, .flx-CheckBox:hover > input {\n        color: #036;\n    }\n    "
    text = event.StringProp('', settable=True, doc='\n        The text on the button.\n        ')
    checked = event.BoolProp(False, settable=True, doc='\n        Whether the button is checked.\n        ')
    disabled = event.BoolProp(False, settable=True, doc='\n        Whether the button is disabled.\n        ')

    @event.reaction('pointer_click')
    def __on_pointer_click(self, e):
        if False:
            while True:
                i = 10
        self.node.blur()

    @event.emitter
    def user_checked(self, checked):
        if False:
            return 10
        ' Event emitted when the user (un)checks this button. Has\n        ``old_value`` and ``new_value`` attributes.\n        '
        d = {'old_value': self.checked, 'new_value': checked}
        self.set_checked(checked)
        return d

class Button(BaseButton):
    """ A push button.
    
    The ``node`` of this widget is a
    `<button> <https://developer.mozilla.org/docs/Web/HTML/Element/button>`_.
    """
    DEFAULT_MIN_SIZE = (10, 28)

    def _create_dom(self):
        if False:
            while True:
                i = 10
        global window
        node = window.document.createElement('button')
        return node

    def _render_dom(self):
        if False:
            while True:
                i = 10
        return [self.text]

    @event.reaction('disabled')
    def __disabled_changed(self, *events):
        if False:
            return 10
        if events[-1].new_value:
            self.node.setAttribute('disabled', 'disabled')
        else:
            self.node.removeAttribute('disabled')

class ToggleButton(BaseButton):
    """ A button that can be toggled. It behaves like a checkbox, while
    looking more like a regular button.
    
    The ``node`` of this widget is a
    `<button> <https://developer.mozilla.org/docs/Web/HTML/Element/button>`_.
    """
    DEFAULT_MIN_SIZE = (10, 28)

    def _create_dom(self):
        if False:
            while True:
                i = 10
        global window
        node = window.document.createElement('button')
        return node

    def _render_dom(self):
        if False:
            print('Hello World!')
        return [self.text]

    @event.reaction('pointer_click')
    def __toggle_checked(self, *events):
        if False:
            while True:
                i = 10
        self.user_checked(not self.checked)

    @event.reaction('checked')
    def __check_changed(self, *events):
        if False:
            return 10
        if self.checked:
            self.node.classList.add('flx-checked')
        else:
            self.node.classList.remove('flx-checked')

class RadioButton(BaseButton):
    """ A radio button. Of any group of radio buttons that share the
    same parent, only one can be active.
    
    The ``outernode`` of this widget is a
    `<label> <https://developer.mozilla.org/docs/Web/HTML/Element/label>`_,
    and the ``node`` a radio
    `<input> <https://developer.mozilla.org/docs/Web/HTML/Element/input>`_.
    """

    def _create_dom(self):
        if False:
            while True:
                i = 10
        global window
        outernode = window.document.createElement('label')
        node = window.document.createElement('input')
        outernode.appendChild(node)
        node.setAttribute('type', 'radio')
        node.setAttribute('id', self.id)
        outernode.setAttribute('for', self.id)
        return (outernode, node)

    def _render_dom(self):
        if False:
            for i in range(10):
                print('nop')
        return [self.node, self.text]

    @event.reaction('parent')
    def __update_group(self, *events):
        if False:
            while True:
                i = 10
        if self.parent:
            self.node.name = self.parent.id

    @event.reaction('checked')
    def __check_changed(self, *events):
        if False:
            i = 10
            return i + 15
        self.node.checked = self.checked

    @event.emitter
    def pointer_click(self, e):
        if False:
            for i in range(10):
                print('nop')
        ' This method is called on JS a click event. We *first* update\n        the checked properties, and then emit the Flexx click event.\n        That way, one can connect to the click event and have an\n        up-to-date checked props (even on Py).\n        '
        if self.parent:
            for child in self.parent.children:
                if isinstance(child, RadioButton) and child is not self:
                    child.set_checked(child.node.checked)
        self.user_checked(self.node.checked)
        super().pointer_click(e)

class CheckBox(BaseButton):
    """ A checkbox button.
    
    The ``outernode`` of this widget is a
    `<label> <https://developer.mozilla.org/docs/Web/HTML/Element/label>`_,
    and the ``node`` a checkbox
    `<input> <https://developer.mozilla.org/docs/Web/HTML/Element/input>`_.
    """

    def _create_dom(self):
        if False:
            print('Hello World!')
        global window
        outernode = window.document.createElement('label')
        node = window.document.createElement('input')
        outernode.appendChild(node)
        node.setAttribute('type', 'checkbox')
        node.setAttribute('id', self.id)
        outernode.setAttribute('for', self.id)
        self._addEventListener(node, 'click', self._check_changed_from_dom, 0)
        return (outernode, node)

    def _render_dom(self):
        if False:
            print('Hello World!')
        return [self.node, self.text]

    @event.reaction('checked')
    def __check_changed(self, *events):
        if False:
            while True:
                i = 10
        self.node.checked = self.checked

    def _check_changed_from_dom(self, ev):
        if False:
            i = 10
            return i + 15
        self.user_checked(self.node.checked)