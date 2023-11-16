from .base import Renderer

class FakeRenderer(Renderer):
    """
    Fake Renderer

    This is a fake renderer which simply outputs a text tree representing the
    elements found in the plot(s).  This is used in the unit tests for the
    package.

    Below are the methods your renderer must implement. You are free to do
    anything you wish within the renderer (i.e. build an XML or JSON
    representation, call an external API, etc.)  Here the renderer just
    builds a simple string representation for testing purposes.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.output = ''

    def open_figure(self, fig, props):
        if False:
            while True:
                i = 10
        self.output += 'opening figure\n'

    def close_figure(self, fig):
        if False:
            return 10
        self.output += 'closing figure\n'

    def open_axes(self, ax, props):
        if False:
            for i in range(10):
                print('nop')
        self.output += '  opening axes\n'

    def close_axes(self, ax):
        if False:
            i = 10
            return i + 15
        self.output += '  closing axes\n'

    def open_legend(self, legend, props):
        if False:
            for i in range(10):
                print('nop')
        self.output += '    opening legend\n'

    def close_legend(self, legend):
        if False:
            i = 10
            return i + 15
        self.output += '    closing legend\n'

    def draw_text(self, text, position, coordinates, style, text_type=None, mplobj=None):
        if False:
            print('Hello World!')
        self.output += "    draw text '{0}' {1}\n".format(text, text_type)

    def draw_path(self, data, coordinates, pathcodes, style, offset=None, offset_coordinates='data', mplobj=None):
        if False:
            return 10
        self.output += '    draw path with {0} vertices\n'.format(data.shape[0])

    def draw_image(self, imdata, extent, coordinates, style, mplobj=None):
        if False:
            for i in range(10):
                print('nop')
        self.output += '    draw image of size {0}\n'.format(len(imdata))

class FullFakeRenderer(FakeRenderer):
    """
    Renderer with the full complement of methods.

    When the following are left undefined, they will be implemented via
    other methods in the class.  They can be defined explicitly for
    more efficient or specialized use within the renderer implementation.
    """

    def draw_line(self, data, coordinates, style, label, mplobj=None):
        if False:
            print('Hello World!')
        self.output += '    draw line with {0} points\n'.format(data.shape[0])

    def draw_markers(self, data, coordinates, style, label, mplobj=None):
        if False:
            while True:
                i = 10
        self.output += '    draw {0} markers\n'.format(data.shape[0])

    def draw_path_collection(self, paths, path_coordinates, path_transforms, offsets, offset_coordinates, offset_order, styles, mplobj=None):
        if False:
            return 10
        self.output += '    draw path collection with {0} offsets\n'.format(offsets.shape[0])