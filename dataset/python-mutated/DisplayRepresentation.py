"""Represent information for graphical display.

Classes in this module are designed to hold information in a way that
makes it easy to draw graphical figures.
"""
from reportlab.lib import colors
from Bio.Graphics.BasicChromosome import ChromosomeSegment
from Bio.Graphics.BasicChromosome import TelomereSegment
RAINBOW_COLORS = {(1, 1): colors.violet, (2, 2): colors.indigo, (3, 3): colors.blue, (4, 4): colors.green, (5, 5): colors.yellow, (6, 6): colors.orange, (7, 20): colors.red}

class ChromosomeCounts:
    """Represent a chromosome with count information.

    This is used to display information about counts along a chromosome.
    The segments are expected to have different count information, which
    will be displayed using a color scheme.

    I envision using this class when you think that certain regions of
    the chromosome will be especially abundant in the counts, and you
    want to pick those out.
    """

    def __init__(self, segment_names, color_scheme=RAINBOW_COLORS):
        if False:
            return 10
        'Initialize a representation of chromosome counts.\n\n        Arguments:\n         - segment_names - An ordered list of all segment names along\n           the chromosome. The count and other information will be added\n           to these.\n         - color_scheme - A coloring scheme to use in the counts. This\n           should be a dictionary mapping count ranges to colors (specified\n           in reportlab.lib.colors).\n\n        '
        self._names = segment_names
        self._count_info = {}
        self._label_info = {}
        self._scale_info = {}
        for name in self._names:
            self._count_info[name] = 0
            self._label_info[name] = None
            self._scale_info[name] = 1
        self._color_scheme = color_scheme

    def add_count(self, segment_name, count=1):
        if False:
            return 10
        'Add counts to the given segment name.\n\n        Arguments:\n         - segment_name - The name of the segment we should add counts to.\n           If the name is not present, a KeyError will be raised.\n         - count - The counts to add the current segment. This defaults to\n           a single count.\n\n        '
        try:
            self._count_info[segment_name] += count
        except KeyError:
            raise KeyError(f'Segment name {segment_name} not found.') from None

    def scale_segment_value(self, segment_name, scale_value=None):
        if False:
            while True:
                i = 10
        "Divide the counts for a segment by some kind of scale value.\n\n        This is useful if segments aren't represented by raw counts, but\n        are instead counts divided by some number.\n        "
        try:
            self._count_info[segment_name] /= scale_value
        except KeyError:
            raise KeyError(f'Segment name {segment_name} not found.') from None

    def add_label(self, segment_name, label):
        if False:
            return 10
        'Add a label to a specific segment.\n\n        Raises a KeyError is the specified segment name is not found.\n        '
        if segment_name in self._label_info:
            self._label_info[segment_name] = label
        else:
            raise KeyError(f'Segment name {segment_name} not found.')

    def set_scale(self, segment_name, scale):
        if False:
            while True:
                i = 10
        'Set the scale for a specific chromosome segment.\n\n        By default all segments have the same scale -- this allows scaling\n        by the size of the segment.\n\n        Raises a KeyError is the specified segment name is not found.\n        '
        if segment_name in self._label_info:
            self._scale_info[segment_name] = scale
        else:
            raise KeyError(f'Segment name {segment_name} not found.')

    def get_segment_info(self):
        if False:
            print('Hello World!')
        'Retrieve the color and label info about the segments.\n\n        Returns a list consisting of two tuples specifying the counts and\n        label name for each segment. The list is ordered according to the\n        original listing of names. Labels are set as None if no label\n        was specified.\n        '
        order_info = []
        for seg_name in self._names:
            order_info.append((self._count_info[seg_name], self._label_info[seg_name]))
        return order_info

    def fill_chromosome(self, chromosome):
        if False:
            i = 10
            return i + 15
        'Add the collected segment information to a chromosome for drawing.\n\n        Arguments:\n         - chromosome - A Chromosome graphics object that we can add\n           chromosome segments to.\n\n        This creates ChromosomeSegment (and TelomereSegment) objects to\n        fill in the chromosome. The information is derived from the\n        label and count information, with counts transformed to the\n        specified color map.\n\n        Returns the chromosome with all of the segments added.\n        '
        for seg_num in range(len(self._names)):
            is_end_segment = 0
            if seg_num == 0:
                cur_segment = TelomereSegment()
                is_end_segment = 1
            elif seg_num == len(self._names) - 1:
                cur_segment = TelomereSegment(1)
                is_end_segment = 1
            else:
                cur_segment = ChromosomeSegment()
            seg_name = self._names[seg_num]
            if self._count_info[seg_name] > 0:
                color = self._color_from_count(self._count_info[seg_name])
                cur_segment.fill_color = color
            if self._label_info[seg_name] is not None:
                cur_segment.label = self._label_info[seg_name]
            if is_end_segment:
                cur_segment.scale = 3
            else:
                cur_segment.scale = self._scale_info[seg_name]
            chromosome.add(cur_segment)
        return chromosome

    def _color_from_count(self, count):
        if False:
            i = 10
            return i + 15
        'Translate the given count into a color using the color scheme (PRIVATE).'
        for (count_start, count_end) in self._color_scheme:
            if count >= count_start and count <= count_end:
                return self._color_scheme[count_start, count_end]
        raise ValueError(f'Count value {count} was not found in the color scheme.')