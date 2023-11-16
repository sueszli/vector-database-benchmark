import warnings
from .base import Renderer
from ..exporter import Exporter

class VincentRenderer(Renderer):

    def open_figure(self, fig, props):
        if False:
            print('Hello World!')
        self.chart = None
        self.figwidth = int(props['figwidth'] * props['dpi'])
        self.figheight = int(props['figheight'] * props['dpi'])

    def draw_line(self, data, coordinates, style, label, mplobj=None):
        if False:
            i = 10
            return i + 15
        import vincent
        if coordinates != 'data':
            warnings.warn('Only data coordinates supported. Skipping this')
        linedata = {'x': data[:, 0], 'y': data[:, 1]}
        line = vincent.Line(linedata, iter_idx='x', width=self.figwidth, height=self.figheight)
        line.scales['color'].range = [style['color']]
        if self.chart is None:
            self.chart = line
        else:
            warnings.warn('Multiple plot elements not yet supported')

    def draw_markers(self, data, coordinates, style, label, mplobj=None):
        if False:
            i = 10
            return i + 15
        import vincent
        if coordinates != 'data':
            warnings.warn('Only data coordinates supported. Skipping this')
        markerdata = {'x': data[:, 0], 'y': data[:, 1]}
        markers = vincent.Scatter(markerdata, iter_idx='x', width=self.figwidth, height=self.figheight)
        markers.scales['color'].range = [style['facecolor']]
        if self.chart is None:
            self.chart = markers
        else:
            warnings.warn('Multiple plot elements not yet supported')

def fig_to_vincent(fig):
    if False:
        return 10
    'Convert a matplotlib figure to a vincent object'
    renderer = VincentRenderer()
    exporter = Exporter(renderer)
    exporter.run(fig)
    return renderer.chart