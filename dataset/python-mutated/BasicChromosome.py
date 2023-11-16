"""Draw representations of organism chromosomes with added information.

These classes are meant to model the drawing of pictures of chromosomes.
This can be useful for lots of things, including displaying markers on
a chromosome (ie. for genetic mapping) and showing syteny between two
chromosomes.

The structure of these classes is intended to be a Composite, so that
it will be easy to plug in and switch different parts without
breaking the general drawing capabilities of the system. The
relationship between classes is that everything derives from
_ChromosomeComponent, which specifies the overall interface. The parts
then are related so that an Organism contains Chromosomes, and these
Chromosomes contain ChromosomeSegments. This representation differs
from the canonical composite structure in that we don't really have
'leaf' nodes here -- all components can potentially hold sub-components.

Most of the time the ChromosomeSegment class is what you'll want to
customize for specific drawing tasks.

For providing drawing capabilities, these classes use reportlab:

http://www.reportlab.com

This provides nice output in PDF, SVG and postscript.  If you have
reportlab's renderPM module installed you can also use PNG etc.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.shapes import Drawing, String, Line, Rect, Wedge, ArcPath
from reportlab.graphics.widgetbase import Widget
from Bio.Graphics import _write
from Bio.Graphics.GenomeDiagram import _Colors
_color_trans = _Colors.ColorTranslator()

class _ChromosomeComponent(Widget):
    """Base class specifying the interface for a component of the system.

    This class should not be instantiated directly, but should be used
    from derived classes.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        'Initialize a chromosome component.\n\n        Attributes:\n        - _sub_components -- Any components which are contained under\n        this parent component. This attribute should be accessed through\n        the add() and remove() functions.\n\n        '
        self._sub_components = []

    def add(self, component):
        if False:
            for i in range(10):
                print('nop')
        'Add a sub_component to the list of components under this item.'
        if not isinstance(component, _ChromosomeComponent):
            raise TypeError(f'Expected a _ChromosomeComponent object, got {component}')
        self._sub_components.append(component)

    def remove(self, component):
        if False:
            return 10
        'Remove the specified component from the subcomponents.\n\n        Raises a ValueError if the component is not registered as a\n        sub_component.\n        '
        try:
            self._sub_components.remove(component)
        except ValueError:
            raise ValueError(f'Component {component} not found in sub_components.') from None

    def draw(self):
        if False:
            i = 10
            return i + 15
        'Draw the specified component.'
        raise AssertionError('Subclasses must implement.')

class Organism(_ChromosomeComponent):
    """Top level class for drawing chromosomes.

    This class holds information about an organism and all of its
    chromosomes, and provides the top level object which could be used
    for drawing a chromosome representation of an organism.

    Chromosomes should be added and removed from the Organism via the
    add and remove functions.
    """

    def __init__(self, output_format='pdf'):
        if False:
            print('Hello World!')
        'Initialize the class.'
        _ChromosomeComponent.__init__(self)
        self.page_size = letter
        self.title_size = 20
        self._legend_height = 0
        self.output_format = output_format

    def draw(self, output_file, title):
        if False:
            for i in range(10):
                print('nop')
        'Draw out the information for the Organism.\n\n        Arguments:\n         - output_file -- The name of a file specifying where the\n           document should be saved, or a handle to be written to.\n           The output format is set when creating the Organism object.\n           Alternatively, output_file=None will return the drawing using\n           the low-level ReportLab objects (for further processing, such\n           as adding additional graphics, before writing).\n         - title -- The output title of the produced document.\n\n        '
        (width, height) = self.page_size
        cur_drawing = Drawing(width, height)
        self._draw_title(cur_drawing, title, width, height)
        cur_x_pos = inch * 0.5
        if len(self._sub_components) > 0:
            x_pos_change = (width - inch) / len(self._sub_components)
        else:
            pass
        for sub_component in self._sub_components:
            sub_component.start_x_position = cur_x_pos + 0.05 * x_pos_change
            sub_component.end_x_position = cur_x_pos + 0.95 * x_pos_change
            sub_component.start_y_position = height - 1.5 * inch
            sub_component.end_y_position = self._legend_height + 1 * inch
            sub_component.draw(cur_drawing)
            cur_x_pos += x_pos_change
        self._draw_legend(cur_drawing, self._legend_height + 0.5 * inch, width)
        if output_file is None:
            return cur_drawing
        return _write(cur_drawing, output_file, self.output_format)

    def _draw_title(self, cur_drawing, title, width, height):
        if False:
            while True:
                i = 10
        'Write out the title of the organism figure (PRIVATE).'
        title_string = String(width / 2, height - inch, title)
        title_string.fontName = 'Helvetica-Bold'
        title_string.fontSize = self.title_size
        title_string.textAnchor = 'middle'
        cur_drawing.add(title_string)

    def _draw_legend(self, cur_drawing, start_y, width):
        if False:
            i = 10
            return i + 15
        'Draw a legend for the figure (PRIVATE).\n\n        Subclasses should implement this (see also self._legend_height) to\n        provide specialized legends.\n        '

class Chromosome(_ChromosomeComponent):
    """Class for drawing a chromosome of an organism.

    This organizes the drawing of a single organisms chromosome. This
    class can be instantiated directly, but the draw method makes the
    most sense to be called in the context of an organism.
    """

    def __init__(self, chromosome_name):
        if False:
            print('Hello World!')
        'Initialize a Chromosome for drawing.\n\n        Arguments:\n         - chromosome_name - The label for the chromosome.\n\n        Attributes:\n         - start_x_position, end_x_position - The x positions on the page\n           where the chromosome should be drawn. This allows multiple\n           chromosomes to be drawn on a single page.\n         - start_y_position, end_y_position - The y positions on the page\n           where the chromosome should be contained.\n\n        Configuration Attributes:\n         - title_size - The size of the chromosome title.\n         - scale_num - A number of scale the drawing by. This is useful if\n           you want to draw multiple chromosomes of different sizes at the\n           same scale. If this is not set, then the chromosome drawing will\n           be scaled by the number of segments in the chromosome (so each\n           chromosome will be the exact same final size).\n\n        '
        _ChromosomeComponent.__init__(self)
        self._name = chromosome_name
        self.start_x_position = -1
        self.end_x_position = -1
        self.start_y_position = -1
        self.end_y_position = -1
        self.title_size = 20
        self.scale_num = None
        self.label_size = 6
        self.chr_percent = 0.25
        self.label_sep_percent = self.chr_percent * 0.5
        self._color_labels = False

    def subcomponent_size(self):
        if False:
            while True:
                i = 10
        'Return the scaled size of all subcomponents of this component.'
        total_sub = 0
        for sub_component in self._sub_components:
            total_sub += sub_component.scale
        return total_sub

    def draw(self, cur_drawing):
        if False:
            i = 10
            return i + 15
        "Draw a chromosome on the specified template.\n\n        Ideally, the x_position and y_*_position attributes should be\n        set prior to drawing -- otherwise we're going to have some problems.\n        "
        for position in (self.start_x_position, self.end_x_position, self.start_y_position, self.end_y_position):
            assert position != -1, 'Need to set drawing coordinates.'
        cur_y_pos = self.start_y_position
        if self.scale_num:
            y_pos_change = (self.start_y_position * 0.95 - self.end_y_position) / self.scale_num
        elif len(self._sub_components) > 0:
            y_pos_change = (self.start_y_position * 0.95 - self.end_y_position) / self.subcomponent_size()
        else:
            pass
        left_labels = []
        right_labels = []
        for sub_component in self._sub_components:
            this_y_pos_change = sub_component.scale * y_pos_change
            sub_component.start_x_position = self.start_x_position
            sub_component.end_x_position = self.end_x_position
            sub_component.start_y_position = cur_y_pos
            sub_component.end_y_position = cur_y_pos - this_y_pos_change
            sub_component._left_labels = []
            sub_component._right_labels = []
            sub_component.draw(cur_drawing)
            left_labels += sub_component._left_labels
            right_labels += sub_component._right_labels
            cur_y_pos -= this_y_pos_change
        self._draw_labels(cur_drawing, left_labels, right_labels)
        self._draw_label(cur_drawing, self._name)

    def _draw_label(self, cur_drawing, label_name):
        if False:
            i = 10
            return i + 15
        'Draw a label for the chromosome (PRIVATE).'
        x_position = 0.5 * (self.start_x_position + self.end_x_position)
        y_position = self.end_y_position
        label_string = String(x_position, y_position, label_name)
        label_string.fontName = 'Times-BoldItalic'
        label_string.fontSize = self.title_size
        label_string.textAnchor = 'middle'
        cur_drawing.add(label_string)

    def _draw_labels(self, cur_drawing, left_labels, right_labels):
        if False:
            print('Hello World!')
        'Layout and draw sub-feature labels for the chromosome (PRIVATE).\n\n        Tries to place each label at the same vertical position as the\n        feature it applies to, but will adjust the positions to avoid or\n        at least reduce label overlap.\n\n        Draws the label text and a coloured line linking it to the\n        location (i.e. feature) it applies to.\n        '
        if not self._sub_components:
            return
        color_label = self._color_labels
        segment_width = (self.end_x_position - self.start_x_position) * self.chr_percent
        label_sep = (self.end_x_position - self.start_x_position) * self.label_sep_percent
        segment_x = self.start_x_position + 0.5 * (self.end_x_position - self.start_x_position - segment_width)
        y_limits = []
        for sub_component in self._sub_components:
            y_limits.extend((sub_component.start_y_position, sub_component.end_y_position))
        y_min = min(y_limits)
        y_max = max(y_limits)
        del y_limits
        h = self.label_size
        for (x1, x2, labels, anchor) in [(segment_x, segment_x - label_sep, _place_labels(left_labels, y_min, y_max, h), 'end'), (segment_x + segment_width, segment_x + segment_width + label_sep, _place_labels(right_labels, y_min, y_max, h), 'start')]:
            for (y1, y2, color, back_color, name) in labels:
                cur_drawing.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=0.25))
                label_string = String(x2, y2, name, textAnchor=anchor)
                label_string.fontName = 'Helvetica'
                label_string.fontSize = h
                if color_label:
                    label_string.fillColor = color
                if back_color:
                    w = stringWidth(name, label_string.fontName, label_string.fontSize)
                    if x1 > x2:
                        w = w * -1.0
                    cur_drawing.add(Rect(x2, y2 - 0.1 * h, w, h, strokeColor=back_color, fillColor=back_color))
                cur_drawing.add(label_string)

class ChromosomeSegment(_ChromosomeComponent):
    """Draw a segment of a chromosome.

    This class provides the important configurable functionality of drawing
    a Chromosome. Each segment has some customization available here, or can
    be subclassed to define additional functionality. Most of the interesting
    drawing stuff is likely to happen at the ChromosomeSegment level.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        'Initialize a ChromosomeSegment.\n\n        Attributes:\n         - start_x_position, end_x_position - Defines the x range we have\n           to draw things in.\n         - start_y_position, end_y_position - Defines the y range we have\n           to draw things in.\n\n        Configuration Attributes:\n         - scale - A scaling value for the component. By default this is\n           set at 1 (ie -- has the same scale as everything else). Higher\n           values give more size to the component, smaller values give less.\n         - fill_color - A color to fill in the segment with. Colors are\n           available in reportlab.lib.colors\n         - label - A label to place on the chromosome segment. This should\n           be a text string specifying what is to be included in the label.\n         - label_size - The size of the label.\n         - chr_percent - The percentage of area that the chromosome\n           segment takes up.\n\n        '
        _ChromosomeComponent.__init__(self)
        self.start_x_position = -1
        self.end_x_position = -1
        self.start_y_position = -1
        self.end_y_position = -1
        self.scale = 1
        self.fill_color = None
        self.label = None
        self.label_size = 6
        self.chr_percent = 0.25

    def draw(self, cur_drawing):
        if False:
            i = 10
            return i + 15
        'Draw a chromosome segment.\n\n        Before drawing, the range we are drawing in needs to be set.\n        '
        for position in (self.start_x_position, self.end_x_position, self.start_y_position, self.end_y_position):
            assert position != -1, 'Need to set drawing coordinates.'
        self._draw_subcomponents(cur_drawing)
        self._draw_segment(cur_drawing)
        self._overdraw_subcomponents(cur_drawing)
        self._draw_label(cur_drawing)

    def _draw_subcomponents(self, cur_drawing):
        if False:
            for i in range(10):
                print('nop')
        'Draw any subcomponents of the chromosome segment (PRIVATE).\n\n        This should be overridden in derived classes if there are\n        subcomponents to be drawn.\n        '

    def _draw_segment(self, cur_drawing):
        if False:
            return 10
        'Draw the current chromosome segment (PRIVATE).'
        segment_y = self.end_y_position
        segment_width = (self.end_x_position - self.start_x_position) * self.chr_percent
        segment_height = self.start_y_position - self.end_y_position
        segment_x = self.start_x_position + 0.5 * (self.end_x_position - self.start_x_position - segment_width)
        right_line = Line(segment_x, segment_y, segment_x, segment_y + segment_height)
        left_line = Line(segment_x + segment_width, segment_y, segment_x + segment_width, segment_y + segment_height)
        cur_drawing.add(right_line)
        cur_drawing.add(left_line)
        if self.fill_color is not None:
            fill_rectangle = Rect(segment_x, segment_y, segment_width, segment_height)
            fill_rectangle.fillColor = self.fill_color
            fill_rectangle.strokeColor = None
            cur_drawing.add(fill_rectangle)

    def _overdraw_subcomponents(self, cur_drawing):
        if False:
            print('Hello World!')
        'Draw any subcomponents of the chromosome segment over the main part (PRIVATE).\n\n        This should be overridden in derived classes if there are\n        subcomponents to be drawn.\n        '

    def _draw_label(self, cur_drawing):
        if False:
            while True:
                i = 10
        'Add a label to the chromosome segment (PRIVATE).\n\n        The label will be applied to the right of the segment.\n\n        This may be overlapped by any sub-feature labels on other segments!\n        '
        if self.label is not None:
            label_x = 0.5 * (self.start_x_position + self.end_x_position) + (self.chr_percent + 0.05) * (self.end_x_position - self.start_x_position)
            label_y = (self.start_y_position - self.end_y_position) / 2 + self.end_y_position
            label_string = String(label_x, label_y, self.label)
            label_string.fontName = 'Helvetica'
            label_string.fontSize = self.label_size
            cur_drawing.add(label_string)

def _spring_layout(desired, minimum, maximum, gap=0):
    if False:
        print('Hello World!')
    'Try to layout label coordinates or other floats (PRIVATE).\n\n    Originally written for the y-axis vertical positioning of labels on a\n    chromosome diagram (where the minimum gap between y-axis coordinates is\n    the label height), it could also potentially be used for x-axis placement,\n    or indeed radial placement for circular chromosomes within GenomeDiagram.\n\n    In essence this is an optimisation problem, balancing the desire to have\n    each label as close as possible to its data point, but also to spread out\n    the labels to avoid overlaps. This could be described with a cost function\n    (modelling the label distance from the desired placement, and the inter-\n    label separations as springs) and solved as a multi-variable minimization\n    problem - perhaps with NumPy or SciPy.\n\n    For now however, the implementation is a somewhat crude ad hoc algorithm.\n\n    NOTE - This expects the input data to have been sorted!\n    '
    count = len(desired)
    if count <= 1:
        return desired
    if minimum >= maximum:
        raise ValueError(f'Bad min/max {minimum:f} and {maximum:f}')
    if min(desired) < minimum or max(desired) > maximum:
        raise ValueError('Data %f to %f out of bounds (%f to %f)' % (min(desired), max(desired), minimum, maximum))
    equal_step = (maximum - minimum) / (count - 1)
    if equal_step < gap:
        import warnings
        from Bio import BiopythonWarning
        warnings.warn('Too many labels to avoid overlap', BiopythonWarning)
        return [minimum + i * equal_step for i in range(count)]
    good = True
    if gap:
        prev = desired[0]
        for next in desired[1:]:
            if prev - next < gap:
                good = False
                break
    if good:
        return desired
    span = maximum - minimum
    for split in [0.5 * span, span / 3.0, 2 * span / 3.0, 0.25 * span, 0.75 * span]:
        midpoint = minimum + split
        low = [x for x in desired if x <= midpoint - 0.5 * gap]
        high = [x for x in desired if x > midpoint + 0.5 * gap]
        if len(low) + len(high) < count:
            continue
        elif not low and len(high) * gap <= span - split + 0.5 * gap:
            return _spring_layout(high, midpoint + 0.5 * gap, maximum, gap)
        elif not high and len(low) * gap <= split + 0.5 * gap:
            return _spring_layout(low, minimum, midpoint - 0.5 * gap, gap)
        elif len(low) * gap <= split - 0.5 * gap and len(high) * gap <= span - split - 0.5 * gap:
            return _spring_layout(low, minimum, midpoint - 0.5 * gap, gap) + _spring_layout(high, midpoint + 0.5 * gap, maximum, gap)
    low = min(desired)
    high = max(desired)
    if (high - low) / (count - 1) >= gap:
        equal_step = (high - low) / (count - 1)
        return [low + i * equal_step for i in range(count)]
    low = 0.5 * (minimum + min(desired))
    high = 0.5 * (max(desired) + maximum)
    if (high - low) / (count - 1) >= gap:
        equal_step = (high - low) / (count - 1)
        return [low + i * equal_step for i in range(count)]
    return [minimum + i * equal_step for i in range(count)]

def _place_labels(desired_etc, minimum, maximum, gap=0):
    if False:
        while True:
            i = 10
    desired_etc.sort()
    placed = _spring_layout([row[0] for row in desired_etc], minimum, maximum, gap)
    for (old, y2) in zip(desired_etc, placed):
        yield ((old[0], y2) + tuple(old[1:]))

class AnnotatedChromosomeSegment(ChromosomeSegment):
    """Annotated chromosome segment.

    This is like the ChromosomeSegment, but accepts a list of features.
    """

    def __init__(self, bp_length, features, default_feature_color=colors.blue, name_qualifiers=('gene', 'label', 'name', 'locus_tag', 'product')):
        if False:
            print('Hello World!')
        "Initialize.\n\n        The features can either be SeqFeature objects, or tuples of values:\n        start (int), end (int), strand (+1, -1, O or None), label (string),\n        ReportLab color (string or object), and optional ReportLab fill color.\n\n        Note we require 0 <= start <= end <= bp_length, and within the vertical\n        space allocated to this segment lines will be places according to the\n        start/end coordinates (starting from the top).\n\n        Positive stand features are drawn on the right, negative on the left,\n        otherwise all the way across.\n\n        We recommend using consistent units for all the segment's scale values\n        (e.g. their length in base pairs).\n\n        When providing features as SeqFeature objects, the default color\n        is used, unless the feature's qualifiers include an Artemis colour\n        string (functionality also in GenomeDiagram). The caption also follows\n        the GenomeDiagram approach and takes the first qualifier from the list\n        or tuple specified in name_qualifiers.\n\n        Note additional attribute label_sep_percent controls the percentage of\n        area that the chromosome segment takes up, by default half of the\n        chr_percent attribute (half of 25%, thus 12.5%)\n\n        "
        ChromosomeSegment.__init__(self)
        self.bp_length = bp_length
        self.features = features
        self.default_feature_color = default_feature_color
        self.name_qualifiers = name_qualifiers
        self.label_sep_percent = self.chr_percent * 0.5

    def _overdraw_subcomponents(self, cur_drawing):
        if False:
            print('Hello World!')
        'Draw any annotated features on the chromosome segment (PRIVATE).\n\n        Assumes _draw_segment already called to fill out the basic shape,\n        and assmes that uses the same boundaries.\n        '
        segment_y = self.end_y_position
        segment_width = (self.end_x_position - self.start_x_position) * self.chr_percent
        label_sep = (self.end_x_position - self.start_x_position) * self.label_sep_percent
        segment_height = self.start_y_position - self.end_y_position
        segment_x = self.start_x_position + 0.5 * (self.end_x_position - self.start_x_position - segment_width)
        left_labels = []
        right_labels = []
        for f in self.features:
            try:
                start = f.location.start
                end = f.location.end
                strand = f.strand
                try:
                    color = _color_trans.translate(f.qualifiers['color'][0])
                except Exception:
                    color = self.default_feature_color
                fill_color = color
                name = ''
                for qualifier in self.name_qualifiers:
                    if qualifier in f.qualifiers:
                        name = f.qualifiers[qualifier][0]
                        break
            except AttributeError:
                (start, end, strand, name, color) = f[:5]
                color = _color_trans.translate(color)
                if len(f) > 5:
                    fill_color = _color_trans.translate(f[5])
                else:
                    fill_color = color
            assert 0 <= start <= end <= self.bp_length
            if strand == +1:
                x = segment_x + segment_width * 0.6
                w = segment_width * 0.4
            elif strand == -1:
                x = segment_x
                w = segment_width * 0.4
            else:
                x = segment_x
                w = segment_width
            local_scale = segment_height / self.bp_length
            fill_rectangle = Rect(x, segment_y + segment_height - local_scale * start, w, local_scale * (start - end))
            fill_rectangle.fillColor = fill_color
            fill_rectangle.strokeColor = color
            cur_drawing.add(fill_rectangle)
            if name:
                if fill_color == color:
                    back_color = None
                else:
                    back_color = fill_color
                value = (segment_y + segment_height - local_scale * start, color, back_color, name)
                if strand == -1:
                    self._left_labels.append(value)
                else:
                    self._right_labels.append(value)

class TelomereSegment(ChromosomeSegment):
    """A segment that is located at the end of a linear chromosome.

    This is just like a regular segment, but it draws the end of a chromosome
    which is represented by a half circle. This just overrides the
    _draw_segment class of ChromosomeSegment to provide that specialized
    drawing.
    """

    def __init__(self, inverted=0):
        if False:
            return 10
        'Initialize a segment at the end of a chromosome.\n\n        See ChromosomeSegment for all of the attributes that can be\n        customized in a TelomereSegments.\n\n        Arguments:\n         - inverted -- Whether or not the telomere should be inverted\n           (ie. drawn on the bottom of a chromosome)\n\n        '
        ChromosomeSegment.__init__(self)
        self._inverted = inverted

    def _draw_segment(self, cur_drawing):
        if False:
            return 10
        'Draw a half circle representing the end of a linear chromosome (PRIVATE).'
        width = (self.end_x_position - self.start_x_position) * self.chr_percent
        height = self.start_y_position - self.end_y_position
        center_x = 0.5 * (self.end_x_position + self.start_x_position)
        start_x = center_x - 0.5 * width
        if self._inverted:
            center_y = self.start_y_position
            start_angle = 180
            end_angle = 360
        else:
            center_y = self.end_y_position
            start_angle = 0
            end_angle = 180
        cap_wedge = Wedge(center_x, center_y, width / 2, start_angle, end_angle, height)
        cap_wedge.strokeColor = None
        cap_wedge.fillColor = self.fill_color
        cur_drawing.add(cap_wedge)
        cap_arc = ArcPath()
        cap_arc.addArc(center_x, center_y, width / 2, start_angle, end_angle, height)
        cur_drawing.add(cap_arc)

class SpacerSegment(ChromosomeSegment):
    """A segment that is located at the end of a linear chromosome.

    Doesn't draw anything, just empty space which can be helpful
    for layout purposes (e.g. making room for feature labels).
    """

    def draw(self, cur_diagram):
        if False:
            while True:
                i = 10
        'Draw nothing to the current diagram (dummy method).\n\n        The segment spacer has no actual image in the diagram,\n        so this method therefore does nothing, but is defined\n        to match the expected API of the other segment objects.\n        '