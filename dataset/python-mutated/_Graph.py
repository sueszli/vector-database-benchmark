"""Graph module.

Provides:
 - GraphData - Contains data from which a graph will be drawn, and
   information about its presentation

For drawing capabilities, this module uses reportlab to draw and write
the diagram: http://www.reportlab.com
"""
from reportlab.lib import colors
from math import sqrt

class GraphData:
    """Graph Data.

    Attributes:
     - id    Unique identifier for the data
     - data  Dictionary of describing the data, keyed by position
     - name  String describing the data
     - style String ('bar', 'heat', 'line') describing how to draw the data
     - poscolor     colors.Color for drawing high (some styles) or all
       values
     - negcolor     colors.Color for drawing low values (some styles)
     - linewidth     Int, thickness to draw the line in 'line' styles

    """

    def __init__(self, id=None, data=None, name=None, style='bar', color=colors.lightgreen, altcolor=colors.darkseagreen, center=None, colour=None, altcolour=None):
        if False:
            for i in range(10):
                print('nop')
        "Initialize.\n\n        Arguments:\n         - id    Unique ID for the graph\n         - data  List of (position, value) tuples\n         - name  String describing the graph\n         - style String describing the presentation style ('bar', 'line',\n           'heat')\n         - color   colors.Color describing the color to draw all or the\n           'high' (some styles) values (overridden by backwards\n           compatible argument with UK spelling, colour).\n         - altcolor colors.Color describing the color to draw the 'low'\n           values (some styles only) (overridden by backwards\n           compatible argument with UK spelling, colour).\n         - center Value at which x-axis crosses y-axis.\n\n        "
        if colour is not None:
            color = colour
        if altcolour is not None:
            altcolor = altcolour
        self.id = id
        self.data = {}
        if data is not None:
            self.set_data(data)
        self.name = name
        self.style = style
        self.poscolor = color
        self.negcolor = altcolor
        self.linewidth = 2
        self.center = center

    def set_data(self, data):
        if False:
            return 10
        'Add data as a list of (position, value) tuples.'
        for (pos, val) in data:
            self.data[pos] = val

    def get_data(self):
        if False:
            while True:
                i = 10
        'Return data as a list of sorted (position, value) tuples.'
        data = []
        for xval in self.data:
            yval = self.data[xval]
            data.append((xval, yval))
        data.sort()
        return data

    def add_point(self, point):
        if False:
            i = 10
            return i + 15
        'Add a single point to the set of data as a (position, value) tuple.'
        (pos, val) = point
        self.data[pos] = val

    def quartiles(self):
        if False:
            i = 10
            return i + 15
        'Return (minimum, lowerQ, medianQ, upperQ, maximum) values as tuple.'
        data = sorted(self.data.values())
        datalen = len(data)
        return (data[0], data[datalen // 4], data[datalen // 2], data[3 * datalen // 4], data[-1])

    def range(self):
        if False:
            for i in range(10):
                print('nop')
        'Return range of data as (start, end) tuple.\n\n        Returns the range of the data, i.e. its start and end points on\n        the genome as a (start, end) tuple.\n        '
        positions = sorted(self.data)
        return (positions[0], positions[-1])

    def mean(self):
        if False:
            while True:
                i = 10
        'Return the mean value for the data points (float).'
        data = list(self.data.values())
        return sum(data) / len(data)

    def stdev(self):
        if False:
            i = 10
            return i + 15
        'Return the sample standard deviation for the data (float).'
        data = list(self.data.values())
        m = self.mean()
        runtotal = 0.0
        for entry in data:
            runtotal += (entry - m) ** 2
        return sqrt(runtotal / (len(data) - 1))

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the number of points in the data set.'
        return len(self.data)

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        'Return data value(s) at the given position.\n\n        Given an integer representing position on the sequence\n        returns a float - the data value at the passed position.\n\n        If a slice, returns graph data from the region as a list or\n        (position, value) tuples. Slices with step are not supported.\n        '
        if isinstance(index, int):
            return self.data[index]
        elif isinstance(index, slice):
            low = index.start
            high = index.stop
            if index.step is not None and index.step != 1:
                raise ValueError
            outlist = []
            for pos in sorted(self.data):
                if pos >= low and pos <= high:
                    outlist.append((pos, self.data[pos]))
            return outlist
        else:
            raise TypeError('Need an integer or a slice')

    def __str__(self):
        if False:
            while True:
                i = 10
        'Return a string describing the graph data.'
        outstr = [f'\nGraphData: {self.name}, ID: {self.id}']
        outstr.append('Number of points: %d' % len(self.data))
        outstr.append(f'Mean data value: {self.mean()}')
        outstr.append(f'Sample SD: {self.stdev():.3f}')
        outstr.append('Minimum: %s\n1Q: %s\n2Q: %s\n3Q: %s\nMaximum: %s' % self.quartiles())
        outstr.append('Sequence Range: %s..%s' % self.range())
        return '\n'.join(outstr)