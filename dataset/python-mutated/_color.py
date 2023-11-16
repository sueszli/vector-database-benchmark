""" ColorSelectWidget

.. UIExample:: 50

    from flexx import event, ui

    class Example(ui.Widget):

        def init(self):
            self.c = ui.ColorSelectWidget()

        @event.reaction
        def _color_changed(self):
            self.node.style.background = self.c.color.hex
"""
from ... import event
from . import Widget

class ColorSelectWidget(Widget):
    """ A widget used to select a color.

    The ``node`` of this widget is an
    `<input> <https://developer.mozilla.org/docs/Web/HTML/Element/input>`_
    element of type ``color``. This is supported at least
    on Firefox and Chrome, but not on IE.
    """
    DEFAULT_MIN_SIZE = (28, 28)
    color = event.ColorProp('#000000', settable=True, doc='\n        The currently selected color.\n        ')
    disabled = event.BoolProp(False, settable=True, doc='\n        Whether the color select is disabled.\n        ')

    def _create_dom(self):
        if False:
            return 10
        global window
        node = window.document.createElement('input')
        try:
            node.type = 'color'
        except Exception:
            node = window.document.createElement('div')
            node.innerHTML = 'Not supported'
        self._addEventListener(node, 'input', self._color_changed_from_dom, 0)
        return node

    @event.emitter
    def user_color(self, color):
        if False:
            return 10
        ' Event emitted when the user changes the color. Has ``old_value``\n        and ``new_value`` attributes.\n        '
        d = {'old_value': self.color, 'new_value': color}
        self.set_color(color)
        return d

    @event.reaction('color')
    def _color_changed(self, *events):
        if False:
            for i in range(10):
                print('nop')
        self.node.value = self.color.hex

    def _color_changed_from_dom(self, e):
        if False:
            for i in range(10):
                print('nop')
        self.user_color(self.node.value)

    @event.reaction('disabled')
    def __disabled_changed(self, *events):
        if False:
            return 10
        if self.disabled:
            self.node.setAttribute('disabled', 'disabled')
        else:
            self.node.removeAttribute('disabled')