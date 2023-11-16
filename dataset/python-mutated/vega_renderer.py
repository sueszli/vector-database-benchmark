import warnings
import json
import random
from .base import Renderer
from ..exporter import Exporter

class VegaRenderer(Renderer):

    def open_figure(self, fig, props):
        if False:
            return 10
        self.props = props
        self.figwidth = int(props['figwidth'] * props['dpi'])
        self.figheight = int(props['figheight'] * props['dpi'])
        self.data = []
        self.scales = []
        self.axes = []
        self.marks = []

    def open_axes(self, ax, props):
        if False:
            for i in range(10):
                print('nop')
        if len(self.axes) > 0:
            warnings.warn('multiple axes not yet supported')
        self.axes = [dict(type='x', scale='x', ticks=10), dict(type='y', scale='y', ticks=10)]
        self.scales = [dict(name='x', domain=props['xlim'], type='linear', range='width'), dict(name='y', domain=props['ylim'], type='linear', range='height')]

    def draw_line(self, data, coordinates, style, label, mplobj=None):
        if False:
            while True:
                i = 10
        if coordinates != 'data':
            warnings.warn('Only data coordinates supported. Skipping this')
        dataname = 'table{0:03d}'.format(len(self.data) + 1)
        self.data.append({'name': dataname, 'values': [dict(x=d[0], y=d[1]) for d in data]})
        self.marks.append({'type': 'line', 'from': {'data': dataname}, 'properties': {'enter': {'interpolate': {'value': 'monotone'}, 'x': {'scale': 'x', 'field': 'data.x'}, 'y': {'scale': 'y', 'field': 'data.y'}, 'stroke': {'value': style['color']}, 'strokeOpacity': {'value': style['alpha']}, 'strokeWidth': {'value': style['linewidth']}}}})

    def draw_markers(self, data, coordinates, style, label, mplobj=None):
        if False:
            for i in range(10):
                print('nop')
        if coordinates != 'data':
            warnings.warn('Only data coordinates supported. Skipping this')
        dataname = 'table{0:03d}'.format(len(self.data) + 1)
        self.data.append({'name': dataname, 'values': [dict(x=d[0], y=d[1]) for d in data]})
        self.marks.append({'type': 'symbol', 'from': {'data': dataname}, 'properties': {'enter': {'interpolate': {'value': 'monotone'}, 'x': {'scale': 'x', 'field': 'data.x'}, 'y': {'scale': 'y', 'field': 'data.y'}, 'fill': {'value': style['facecolor']}, 'fillOpacity': {'value': style['alpha']}, 'stroke': {'value': style['edgecolor']}, 'strokeOpacity': {'value': style['alpha']}, 'strokeWidth': {'value': style['edgewidth']}}}})

    def draw_text(self, text, position, coordinates, style, text_type=None, mplobj=None):
        if False:
            while True:
                i = 10
        if text_type == 'xlabel':
            self.axes[0]['title'] = text
        elif text_type == 'ylabel':
            self.axes[1]['title'] = text

class VegaHTML(object):

    def __init__(self, renderer):
        if False:
            for i in range(10):
                print('nop')
        self.specification = dict(width=renderer.figwidth, height=renderer.figheight, data=renderer.data, scales=renderer.scales, axes=renderer.axes, marks=renderer.marks)

    def html(self):
        if False:
            return 10
        'Build the HTML representation for IPython.'
        id = random.randint(0, 2 ** 16)
        html = '<div id="vis%d"></div>' % id
        html += '<script>\n'
        html += VEGA_TEMPLATE % (json.dumps(self.specification), id)
        html += '</script>\n'
        return html

    def _repr_html_(self):
        if False:
            for i in range(10):
                print('nop')
        return self.html()

def fig_to_vega(fig, notebook=False):
    if False:
        for i in range(10):
            print('nop')
    'Convert a matplotlib figure to vega dictionary\n\n    if notebook=True, then return an object which will display in a notebook\n    otherwise, return an HTML string.\n    '
    renderer = VegaRenderer()
    Exporter(renderer).run(fig)
    vega_html = VegaHTML(renderer)
    if notebook:
        return vega_html
    else:
        return vega_html.html()
VEGA_TEMPLATE = '\n( function() {\n  var _do_plot = function() {\n    if ( (typeof vg == \'undefined\') && (typeof IPython != \'undefined\')) {\n      $([IPython.events]).on("vega_loaded.vincent", _do_plot);\n      return;\n    }\n    vg.parse.spec(%s, function(chart) {\n      chart({el: "#vis%d"}).update();\n    });\n  };\n  _do_plot();\n})();\n'