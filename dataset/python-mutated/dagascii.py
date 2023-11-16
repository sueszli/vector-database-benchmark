"""Draws DAG in ASCII."""
import math
import os
from grandalf.graphs import Edge, Graph, Vertex
from grandalf.layouts import SugiyamaLayout
from grandalf.routing import EdgeViewer, route_with_lines
from dvc.log import logger
logger = logger.getChild(__name__)

class VertexViewer:
    """Class to define vertex box boundaries that will be accounted for during
    graph building by grandalf.

    Args:
        name (str): name of the vertex.
    """
    HEIGHT = 3

    def __init__(self, name):
        if False:
            return 10
        self._h = self.HEIGHT
        self._w = len(name) + 2

    @property
    def h(self):
        if False:
            i = 10
            return i + 15
        'Height of the box.'
        return self._h

    @property
    def w(self):
        if False:
            return 10
        'Width of the box.'
        return self._w

class AsciiCanvas:
    """Class for drawing in ASCII.

    Args:
        cols (int): number of columns in the canvas. Should be > 1.
        lines (int): number of lines in the canvas. Should be > 1.
    """
    TIMEOUT = 10

    def __init__(self, cols, lines):
        if False:
            return 10
        assert cols > 1
        assert lines > 1
        self.cols = cols
        self.lines = lines
        self.canvas = [[' '] * cols for line in range(lines)]

    def draw(self):
        if False:
            for i in range(10):
                print('nop')
        'Draws ASCII canvas on the screen.'
        lines = map(''.join, self.canvas)
        return os.linesep.join(lines)

    def point(self, x, y, char):
        if False:
            i = 10
            return i + 15
        'Create a point on ASCII canvas.\n\n        Args:\n            x (int): x coordinate. Should be >= 0 and < number of columns in\n                the canvas.\n            y (int): y coordinate. Should be >= 0 an < number of lines in the\n                canvas.\n            char (str): character to place in the specified point on the\n                canvas.\n        '
        assert len(char) == 1
        assert x >= 0
        assert x < self.cols
        assert y >= 0
        assert y < self.lines
        self.canvas[y][x] = char

    def line(self, x0, y0, x1, y1, char):
        if False:
            print('Hello World!')
        'Create a line on ASCII canvas.\n\n        Args:\n            x0 (int): x coordinate where the line should start.\n            y0 (int): y coordinate where the line should start.\n            x1 (int): x coordinate where the line should end.\n            y1 (int): y coordinate where the line should end.\n            char (str): character to draw the line with.\n        '
        if x0 > x1:
            (x1, x0) = (x0, x1)
            (y1, y0) = (y0, y1)
        dx = x1 - x0
        dy = y1 - y0
        if dx == 0 and dy == 0:
            self.point(x0, y0, char)
        elif abs(dx) >= abs(dy):
            for x in range(x0, x1 + 1):
                if dx == 0:
                    y = y0
                else:
                    y = y0 + int(round((x - x0) * dy / float(dx)))
                self.point(x, y, char)
        elif y0 < y1:
            for y in range(y0, y1 + 1):
                if dy == 0:
                    x = x0
                else:
                    x = x0 + int(round((y - y0) * dx / float(dy)))
                self.point(x, y, char)
        else:
            for y in range(y1, y0 + 1):
                if dy == 0:
                    x = x0
                else:
                    x = x1 + int(round((y - y1) * dx / float(dy)))
                self.point(x, y, char)

    def text(self, x, y, text):
        if False:
            i = 10
            return i + 15
        'Print a text on ASCII canvas.\n\n        Args:\n            x (int): x coordinate where the text should start.\n            y (int): y coordinate where the text should start.\n            text (str): string that should be printed.\n        '
        for (i, char) in enumerate(text):
            self.point(x + i, y, char)

    def box(self, x0, y0, width, height):
        if False:
            i = 10
            return i + 15
        'Create a box on ASCII canvas.\n\n        Args:\n            x0 (int): x coordinate of the box corner.\n            y0 (int): y coordinate of the box corner.\n            width (int): box width.\n            height (int): box height.\n        '
        assert width > 1
        assert height > 1
        width -= 1
        height -= 1
        for x in range(x0, x0 + width):
            self.point(x, y0, '-')
            self.point(x, y0 + height, '-')
        for y in range(y0, y0 + height):
            self.point(x0, y, '|')
            self.point(x0 + width, y, '|')
        self.point(x0, y0, '+')
        self.point(x0 + width, y0, '+')
        self.point(x0, y0 + height, '+')
        self.point(x0 + width, y0 + height, '+')

def _build_sugiyama_layout(vertices, edges):
    if False:
        while True:
            i = 10
    vertices = {v: Vertex(f' {v} ') for v in vertices}
    edges = [Edge(vertices[e], vertices[s]) for (s, e) in edges]
    vertices = vertices.values()
    graph = Graph(vertices, edges)
    for vertex in vertices:
        vertex.view = VertexViewer(vertex.data)
    minw = min((v.view.w for v in vertices))
    for edge in edges:
        edge.view = EdgeViewer()
    sug = SugiyamaLayout(graph.C[0])
    graph = graph.C[0]
    roots = list(filter(lambda x: len(x.e_in()) == 0, graph.sV))
    sug.init_all(roots=roots, optimize=True)
    sug.yspace = VertexViewer.HEIGHT
    sug.xspace = minw
    sug.route_edge = route_with_lines
    sug.draw()
    return sug

def draw(vertices, edges):
    if False:
        while True:
            i = 10
    'Build a DAG and draw it in ASCII.\n\n    Args:\n        vertices (list): list of graph vertices.\n        edges (list): list of graph edges.\n\n    Returns:\n        str: ASCII representation\n\n    Example:\n        >>> from dvc.dagascii import draw\n        >>> vertices = [1, 2, 3, 4]\n        >>> edges = [(1, 2), (2, 3), (2, 4), (1, 4)]\n        >>> print(draw(vertices, edges))\n        +---+     +---+\n        | 3 |     | 4 |\n        +---+    *+---+\n          *    **   *\n          *  **     *\n          * *       *\n        +---+       *\n        | 2 |      *\n        +---+     *\n             *    *\n              *  *\n               **\n             +---+\n             | 1 |\n             +---+\n    '
    Xs = []
    Ys = []
    sug = _build_sugiyama_layout(vertices, edges)
    for vertex in sug.g.sV:
        Xs.append(vertex.view.xy[0] - vertex.view.w / 2.0)
        Xs.append(vertex.view.xy[0] + vertex.view.w / 2.0)
        Ys.append(vertex.view.xy[1])
        Ys.append(vertex.view.xy[1] + vertex.view.h)
    for edge in sug.g.sE:
        for (x, y) in edge.view._pts:
            Xs.append(x)
            Ys.append(y)
    minx = min(Xs)
    miny = min(Ys)
    maxx = max(Xs)
    maxy = max(Ys)
    canvas_cols = int(math.ceil(math.ceil(maxx) - math.floor(minx))) + 1
    canvas_lines = int(round(maxy - miny))
    canvas = AsciiCanvas(canvas_cols, canvas_lines)
    for edge in sug.g.sE:
        assert len(edge.view._pts) > 1
        for index in range(1, len(edge.view._pts)):
            start = edge.view._pts[index - 1]
            end = edge.view._pts[index]
            start_x = int(round(start[0] - minx))
            start_y = int(round(start[1] - miny))
            end_x = int(round(end[0] - minx))
            end_y = int(round(end[1] - miny))
            assert start_x >= 0
            assert start_y >= 0
            assert end_x >= 0
            assert end_y >= 0
            canvas.line(start_x, start_y, end_x, end_y, '*')
    for vertex in sug.g.sV:
        x = vertex.view.xy[0] - vertex.view.w / 2.0
        y = vertex.view.xy[1]
        canvas.box(int(round(x - minx)), int(round(y - miny)), vertex.view.w, vertex.view.h)
        canvas.text(int(round(x - minx)) + 1, int(round(y - miny)) + 1, vertex.data)
    return canvas.draw()