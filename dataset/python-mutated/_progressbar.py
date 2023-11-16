""" ProgressBar

Example:

.. UIExample:: 100

    from flexx import app, event, ui

    class Example(ui.Widget):

        def init(self):
            with ui.HBox():
                self.b1 = ui.Button(flex=0, text='Less')
                self.b2 = ui.Button(flex=0, text='More')
                self.prog = ui.ProgressBar(flex=1, value=0.1, text='{percent} done')

        @event.reaction('b1.pointer_down', 'b2.pointer_down')
        def _change_progress(self, *events):
            for ev in events:
                if ev.source is self.b1:
                    self.prog.set_value(self.prog.value - 0.1)
                else:
                    self.prog.set_value(self.prog.value + 0.1)
"""
from ... import event
from .._widget import Widget, create_element

class ProgressBar(Widget):
    """ A widget to show progress.
    
    The ``node`` of this widget is a
    `<div> <https://developer.mozilla.org/docs/Web/HTML/Element/div>`_
    containing a few HTML elements for rendering.
    """
    DEFAULT_MIN_SIZE = (40, 16)
    CSS = '\n\n    .flx-ProgressBar {\n        border: 1px solid #ddd;\n        border-radius: 6px;\n        background: #eee;\n    }\n\n    .flx-ProgressBar > .progress-bar {\n        /* Use flexbox to vertically align label text */\n        display: -webkit-flex;\n        display: -ms-flexbox;\n        display: -ms-flex;\n        display: -moz-flex;\n        display: flex;\n        -webkit-flex-flow: column;\n        -ms-flex-flow: column;\n        -moz-flex-flow: column;\n        flex-flow: column;\n        -webkit-justify-content: center;\n        -ms-justify-content: center;\n        -moz-justify-content: center;\n        justify-content: center;\n        white-space: nowrap;\n        align-self: stretch;\n\n        position: absolute; /* need this on Chrome when in a VBox */\n        background: #8be;\n        text-align: center;\n        /*transition: width 0.2s ease; behaves silly on Chrome */\n        }\n\n    '
    value = event.FloatProp(0, settable=True, doc='\n            The progress value.\n            ')
    min = event.FloatProp(0, settable=True, doc='\n        The minimum progress value.\n        ')
    max = event.FloatProp(1, settable=True, doc='\n        The maximum progress value.\n        ')
    text = event.StringProp('', settable=True, doc='\n        The label to display on the progress bar. Occurances of\n        "{percent}" are replaced with the current percentage, and\n        "{value}" with the current value.\n        ')

    @event.action
    def set_value(self, value):
        if False:
            return 10
        value = max(self.min, value)
        value = min(self.max, value)
        self._mutate_value(value)

    @event.reaction('min', 'max')
    def __keep_value_constrained(self, *events):
        if False:
            return 10
        self.set_value(self.value)

    def _render_dom(self):
        if False:
            while True:
                i = 10
        global Math
        value = self.value
        (mi, ma) = (self.min, self.max)
        perc = 100 * (value - mi) / (ma - mi)
        label = self.text
        label = label.replace('{value}', str(value))
        label = label.replace('{percent}', Math.round(perc) + '%')
        attr = {'style__width': perc + '%', 'style__height': '100%', 'className': 'progress-bar'}
        return [create_element('div', attr, label)]