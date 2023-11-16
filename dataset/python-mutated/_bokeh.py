""" BokehWidget

Show Bokeh plots in Flexx. Example:

.. UIExample:: 300

    import numpy as np
    from bokeh.plotting import figure
    from flexx import app, event, ui

    x = np.linspace(0, 6, 50)

    p1 = figure()
    p1.line(x, np.sin(x))

    p2 = figure()
    p2.line(x, np.cos(x))

    class Example(app.PyComponent):
        def init(self):
            with ui.HSplit():
                ui.BokehWidget.from_plot(p1)
                ui.BokehWidget.from_plot(p2)

Also see examples: :ref:`bokehdemo.py`.

"""
import os
from ... import event, app
from . import Widget

def _load_bokeh(ext):
    if False:
        return 10
    ns = {}
    exec('import bokeh.resources', ns, ns)
    bokeh = ns['bokeh']
    dev = os.environ.get('BOKEH_RESOURCES', '') == 'relative-dev'
    res = bokeh.resources.bokehjsdir()
    if dev:
        res = os.path.abspath(os.path.join(bokeh.__file__, '..', '..', 'bokehjs', 'build'))
    modname = 'bokeh' if dev else 'bokeh.min'
    filename = os.path.join(res, ext, modname + '.' + ext)
    return open(filename, 'rb').read().decode()

def _load_bokeh_js():
    if False:
        return 10
    return _load_bokeh('js')

def _load_bokeh_css():
    if False:
        while True:
            i = 10
    try:
        return _load_bokeh('css')
    except FileNotFoundError:
        return ''
app.assets.associate_asset(__name__, 'bokeh.js', _load_bokeh_js)
app.assets.associate_asset(__name__, 'bokeh.css', _load_bokeh_css)

def make_bokeh_widget(plot, **kwargs):
    if False:
        print('Hello World!')
    ns = {}
    exec('from bokeh.models import Plot', ns, ns)
    exec('from bokeh.embed import components', ns, ns)
    (Plot, components) = (ns['Plot'], ns['components'])
    if not isinstance(plot, Plot):
        raise ValueError('plot must be a Bokeh plot object.')
    if plot.sizing_mode == 'fixed':
        plot.sizing_mode = 'stretch_both'
    (script, div) = components(plot)
    script = '\n'.join(script.strip().split('\n')[1:-1])
    widget = BokehWidget(**kwargs)
    widget.set_plot_components(dict(script=script, div=div, id=plot.ref['id']))
    return widget

class BokehWidget(Widget):
    """ A widget that shows a Bokeh plot object.

    For Bokeh 0.12 and up. The plot's ``sizing_mode`` property is set to
    ``stretch_both`` unless it was set to something other than ``fixed``. Other
    responsive modes are 'scale_width', 'scale_height' and 'scale_both`, which
    all keep aspect ratio while being responsive in a certain direction.

    This widget is, like all widgets, a JsComponent; it lives in the browser,
    while the Bokeh plot is a Python object. Therefore we cannot simply use
    a property to set the plot. Use ``ui.BokehWidget.from_plot(plot)`` to
    instantiate the widget from Python.
    """
    DEFAULT_MIN_SIZE = (100, 100)
    CSS = '\n    .flx-BokehWidget > .plotdiv {\n        overflow: hidden;\n    }\n    '

    @classmethod
    def from_plot(cls, plot, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Create a BokehWidget using a Bokeh plot.\n        '
        return make_bokeh_widget(plot, **kwargs)
    plot = event.Attribute(doc='The JS-side of the Bokeh plot object.')

    def _render_dom(self):
        if False:
            print('Hello World!')
        return None

    @event.action
    def set_plot_components(self, d):
        if False:
            print('Hello World!')
        ' Set the plot using its script/html components.\n        '
        global window
        self.node.innerHTML = d.div
        el = window.document.createElement('script')
        el.innerHTML = d.script
        self.node.appendChild(el)

        def getplot():
            if False:
                while True:
                    i = 10
            self._plot = window.Bokeh.index[d.id]
            self.__resize_plot()
        window.setTimeout(getplot, 10)

    @event.reaction('size')
    def __resize_plot(self, *events):
        if False:
            print('Hello World!')
        if self.plot and self.parent:
            if self.plot.resize_layout:
                self.plot.resize_layout()
            elif self.plot.resize:
                self.plot.resize()
            else:
                self.plot.model.document.resize()