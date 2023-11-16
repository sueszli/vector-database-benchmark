"""GraphSet module.

Provides:
 - GraphSet - container for GraphData objects

For drawing capabilities, this module uses reportlab to draw and write
the diagram: http://www.reportlab.com
"""
from reportlab.lib import colors
from ._Graph import GraphData

class GraphSet:
    """Graph Set.

    Attributes:
     - id        Unique identifier for the set
     - name      String describing the set

    """

    def __init__(self, name=None):
        if False:
            print('Hello World!')
        'Initialize.\n\n        Arguments:\n         - name      String identifying the graph set sensibly\n\n        '
        self.id = id
        self._next_id = 0
        self._graphs = {}
        self.name = name

    def new_graph(self, data, name=None, style='bar', color=colors.lightgreen, altcolor=colors.darkseagreen, linewidth=1, center=None, colour=None, altcolour=None, centre=None):
        if False:
            print('Hello World!')
        "Add a GraphData object to the diagram.\n\n        Arguments:\n         - data      List of (position, value) int tuples\n         - name      String, description of the graph\n         - style     String ('bar', 'heat', 'line') describing how the graph\n           will be drawn\n         - color    colors.Color describing the color to draw all or 'high'\n           (some styles) data (overridden by backwards compatible\n           argument with UK spelling, colour).\n         - altcolor  colors.Color describing the color to draw 'low' (some\n           styles) data (overridden by backwards compatible argument\n           with UK spelling, colour).\n         - linewidth     Float describing linewidth for graph\n         - center        Float setting the value at which the x-axis\n           crosses the y-axis (overridden by backwards\n           compatible argument with UK spelling, centre)\n\n        Add a GraphData object to the diagram (will be stored internally).\n        "
        if colour is not None:
            color = colour
        if altcolour is not None:
            altcolor = altcolour
        if centre is not None:
            center = centre
        id = self._next_id
        graph = GraphData(id, data, name, style, color, altcolor, center)
        graph.linewidth = linewidth
        self._graphs[id] = graph
        self._next_id += 1
        return graph

    def del_graph(self, graph_id):
        if False:
            return 10
        'Remove a graph from the set, indicated by its id.'
        del self._graphs[graph_id]

    def get_graphs(self):
        if False:
            while True:
                i = 10
        'Return list of all graphs in the graph set, sorted by id.\n\n        Sorting is to ensure reliable stacking.\n        '
        return [self._graphs[id] for id in sorted(self._graphs)]

    def get_ids(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a list of all ids for the graph set.'
        return list(self._graphs.keys())

    def range(self):
        if False:
            print('Hello World!')
        'Return the lowest and highest base (or mark) numbers as a tuple.'
        (lows, highs) = ([], [])
        for graph in self._graphs.values():
            (low, high) = graph.range()
            lows.append(low)
            highs.append(high)
        return (min(lows), max(highs))

    def data_quartiles(self):
        if False:
            print('Hello World!')
        'Return (minimum, lowerQ, medianQ, upperQ, maximum) values as a tuple.'
        data = []
        for graph in self._graphs.values():
            data += list(graph.data.values())
        data.sort()
        datalen = len(data)
        return (data[0], data[datalen / 4], data[datalen / 2], data[3 * datalen / 4], data[-1])

    def to_string(self, verbose=0):
        if False:
            print('Hello World!')
        'Return a formatted string with information about the set.\n\n        Arguments:\n            - verbose - Flag indicating whether a short or complete account\n              of the set is required\n\n        '
        if not verbose:
            return f'{self}'
        else:
            outstr = [f'\n<{self.__class__}: {self.name}>']
            outstr.append('%d graphs' % len(self._graphs))
            for key in self._graphs:
                outstr.append(f'{self._graphs[key]}')
            return '\n'.join(outstr)

    def __len__(self):
        if False:
            i = 10
            return i + 15
        'Return the number of graphs in the set.'
        return len(self._graphs)

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        'Return a graph, keyed by id.'
        return self._graphs[key]

    def __str__(self):
        if False:
            while True:
                i = 10
        'Return a formatted string with information about the feature set.'
        outstr = [f'\n<{self.__class__}: {self.name}>']
        outstr.append('%d graphs' % len(self._graphs))
        outstr = '\n'.join(outstr)
        return outstr