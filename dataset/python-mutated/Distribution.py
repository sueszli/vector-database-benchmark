"""Display information distributed across a Chromosome-like object.

These classes are meant to show the distribution of some kind of information
as it changes across any kind of segment. It was designed with chromosome
distributions in mind, but could also work for chromosome regions, BAC clones
or anything similar.

Reportlab is used for producing the graphical output.
"""
import math
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.barcharts import BarChartProperties
from reportlab.graphics.widgetbase import TypedPropertyCollection
from Bio.Graphics import _write

class DistributionPage:
    """Display a grouping of distributions on a page.

    This organizes Distributions, and will display them nicely
    on a single page.
    """

    def __init__(self, output_format='pdf'):
        if False:
            i = 10
            return i + 15
        'Initialize the class.'
        self.distributions = []
        self.number_of_columns = 1
        self.page_size = letter
        self.title_size = 20
        self.output_format = output_format

    def draw(self, output_file, title):
        if False:
            while True:
                i = 10
        'Draw out the distribution information.\n\n        Arguments:\n         - output_file - The name of the file to output the information to,\n           or a handle to write to.\n         - title - A title to display on the graphic.\n\n        '
        (width, height) = self.page_size
        cur_drawing = Drawing(width, height)
        self._draw_title(cur_drawing, title, width, height)
        cur_x_pos = inch * 0.5
        end_x_pos = width - inch * 0.5
        cur_y_pos = height - 1.5 * inch
        end_y_pos = 0.5 * inch
        x_pos_change = (end_x_pos - cur_x_pos) / self.number_of_columns
        num_y_rows = math.ceil(len(self.distributions) / self.number_of_columns)
        y_pos_change = (cur_y_pos - end_y_pos) / num_y_rows
        self._draw_distributions(cur_drawing, cur_x_pos, x_pos_change, cur_y_pos, y_pos_change, num_y_rows)
        self._draw_legend(cur_drawing, 2.5 * inch, width)
        return _write(cur_drawing, output_file, self.output_format)

    def _draw_title(self, cur_drawing, title, width, height):
        if False:
            while True:
                i = 10
        'Add the title of the figure to the drawing (PRIVATE).'
        title_string = String(width / 2, height - inch, title)
        title_string.fontName = 'Helvetica-Bold'
        title_string.fontSize = self.title_size
        title_string.textAnchor = 'middle'
        cur_drawing.add(title_string)

    def _draw_distributions(self, cur_drawing, start_x_pos, x_pos_change, start_y_pos, y_pos_change, num_y_drawings):
        if False:
            for i in range(10):
                print('nop')
        "Draw all of the distributions on the page (PRIVATE).\n\n        Arguments:\n         - cur_drawing - The drawing we are working with.\n         - start_x_pos - The x position on the page to start drawing at.\n         - x_pos_change - The change in x position between each figure.\n         - start_y_pos - The y position on the page to start drawing at.\n         - y_pos_change - The change in y position between each figure.\n         - num_y_drawings - The number of drawings we'll have in the y\n           (up/down) direction.\n\n        "
        for y_drawing in range(int(num_y_drawings)):
            if (y_drawing + 1) * self.number_of_columns > len(self.distributions):
                num_x_drawings = len(self.distributions) - y_drawing * self.number_of_columns
            else:
                num_x_drawings = self.number_of_columns
            for x_drawing in range(num_x_drawings):
                dist_num = y_drawing * self.number_of_columns + x_drawing
                cur_distribution = self.distributions[dist_num]
                x_pos = start_x_pos + x_drawing * x_pos_change
                end_x_pos = x_pos + x_pos_change
                end_y_pos = start_y_pos - y_drawing * y_pos_change
                y_pos = end_y_pos - y_pos_change
                cur_distribution.draw(cur_drawing, x_pos, y_pos, end_x_pos, end_y_pos)

    def _draw_legend(self, cur_drawing, start_y, width):
        if False:
            i = 10
            return i + 15
        'Add a legend to the figure (PRIVATE).\n\n        Subclasses can implement to provide a specialized legend.\n        '

class BarChartDistribution:
    """Display the distribution of values as a bunch of bars."""

    def __init__(self, display_info=None):
        if False:
            for i in range(10):
                print('nop')
        'Initialize a Bar Chart display of distribution info.\n\n        Attributes:\n         - display_info - the information to be displayed in the distribution.\n           This should be ordered as a list of lists, where each internal list\n           is a data set to display in the bar chart.\n\n        '
        if display_info is None:
            display_info = []
        self.display_info = display_info
        self.x_axis_title = ''
        self.y_axis_title = ''
        self.chart_title = ''
        self.chart_title_size = 10
        self.padding_percent = 0.15

    def draw(self, cur_drawing, start_x, start_y, end_x, end_y):
        if False:
            return 10
        'Draw a bar chart with the info in the specified range.'
        bar_chart = VerticalBarChart()
        if self.chart_title:
            self._draw_title(cur_drawing, self.chart_title, start_x, start_y, end_x, end_y)
        (x_start, x_end, y_start, y_end) = self._determine_position(start_x, start_y, end_x, end_y)
        bar_chart.x = x_start
        bar_chart.y = y_start
        bar_chart.width = abs(x_start - x_end)
        bar_chart.height = abs(y_start - y_end)
        bar_chart.data = self.display_info
        bar_chart.valueAxis.valueMin = min(self.display_info[0])
        bar_chart.valueAxis.valueMax = max(self.display_info[0])
        for data_set in self.display_info[1:]:
            if min(data_set) < bar_chart.valueAxis.valueMin:
                bar_chart.valueAxis.valueMin = min(data_set)
            if max(data_set) > bar_chart.valueAxis.valueMax:
                bar_chart.valueAxis.valueMax = max(data_set)
        if len(self.display_info) == 1:
            bar_chart.groupSpacing = 0
            style = TypedPropertyCollection(BarChartProperties)
            style.strokeWidth = 0
            style.strokeColor = colors.green
            style[0].fillColor = colors.green
            bar_chart.bars = style
        cur_drawing.add(bar_chart)

    def _draw_title(self, cur_drawing, title, start_x, start_y, end_x, end_y):
        if False:
            for i in range(10):
                print('nop')
        'Add the title of the figure to the drawing (PRIVATE).'
        x_center = start_x + (end_x - start_x) / 2
        y_pos = end_y + self.padding_percent * (start_y - end_y) / 2
        title_string = String(x_center, y_pos, title)
        title_string.fontName = 'Helvetica-Bold'
        title_string.fontSize = self.chart_title_size
        title_string.textAnchor = 'middle'
        cur_drawing.add(title_string)

    def _determine_position(self, start_x, start_y, end_x, end_y):
        if False:
            for i in range(10):
                print('nop')
        'Calculate the position of the chart with blank space (PRIVATE).\n\n        This uses some padding around the chart, and takes into account\n        whether the chart has a title. It returns 4 values, which are,\n        in order, the x_start, x_end, y_start and y_end of the chart\n        itself.\n        '
        x_padding = self.padding_percent * (end_x - start_x)
        y_padding = self.padding_percent * (start_y - end_y)
        new_x_start = start_x + x_padding
        new_x_end = end_x - x_padding
        if self.chart_title:
            new_y_start = start_y - y_padding - self.chart_title_size
        else:
            new_y_start = start_y - y_padding
        new_y_end = end_y + y_padding
        return (new_x_start, new_x_end, new_y_start, new_y_end)

class LineDistribution:
    """Display the distribution of values as connected lines.

    This distribution displays the change in values across the object as
    lines. This also allows multiple distributions to be displayed on a
    single graph.
    """

    def __init__(self):
        if False:
            return 10
        'Initialize the class.'

    def draw(self, cur_drawing, start_x, start_y, end_x, end_y):
        if False:
            i = 10
            return i + 15
        'Draw a line distribution into the current drawing.'