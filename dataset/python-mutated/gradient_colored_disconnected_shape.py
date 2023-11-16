"""
This class represents a generic disjoint shape (specified by a List of lines),
that will be colored according to a gradient.
It has convenience methods to calculate width and height, perform scaling, etc
"""
import enum
import math
from decimal import Decimal
from borb.pdf.canvas.color.color import Color
from borb.pdf.canvas.color.color import HSVColor
from borb.pdf.canvas.color.color import RGBColor
from borb.pdf.canvas.geometry.rectangle import Rectangle
from borb.pdf.canvas.layout.shape.disconnected_shape import DisconnectedShape
from borb.pdf.page.page import Page

class GradientColoredDisconnectedShape(DisconnectedShape):
    """
    This class represents a generic disconnected shape (specified by a List of lines),
    that will be colored according to a gradient.
    It has convenience methods to calculate width and height, perform scaling, etc
    """

    class GradientType(enum.Enum):
        """
        This enumeration represents the type
        of gradient that will be applied to this DisjointShape.
        """
        DIAGONAL = 1
        HORIZONTAL = 2
        RADIAL = 3
        VERTICAL = 4

    def __init__(self, shape: DisconnectedShape, from_color: Color, to_color: Color, gradient_type: GradientType=GradientType.RADIAL):
        if False:
            print('Hello World!')
        super(GradientColoredDisconnectedShape, self).__init__(lines=shape._lines, stroke_color=shape._stroke_color, line_width=shape._line_width, background_color=shape._background_color, border_bottom=shape._border_bottom, border_color=shape._border_color, border_left=shape._border_left, border_radius_bottom_left=shape._border_radius_bottom_left, border_radius_bottom_right=shape._border_radius_bottom_right, border_radius_top_left=shape._border_radius_top_left, border_radius_top_right=shape._border_radius_top_right, border_right=shape._border_right, border_top=shape._border_top, border_width=shape._border_width, horizontal_alignment=shape._horizontal_alignment, margin_bottom=shape._margin_bottom, margin_left=shape._margin_left, margin_right=shape._margin_right, margin_top=shape._margin_top, padding_bottom=shape._padding_bottom, padding_left=shape._padding_left, padding_right=shape._padding_right, padding_top=shape._padding_top, vertical_alignment=shape._vertical_alignment)
        self._from_color: Color = from_color
        self._to_color: Color = to_color
        self._gradient_type: GradientColoredDisconnectedShape.GradientType = gradient_type

    def _get_content_box(self, available_space: Rectangle) -> Rectangle:
        if False:
            for i in range(10):
                print('nop')
        return super(GradientColoredDisconnectedShape, self)._get_content_box(available_space)

    def _paint_content_box(self, page: Page, bounding_box: Rectangle) -> None:
        if False:
            i = 10
            return i + 15
        self.move_to(bounding_box.x, bounding_box.y + bounding_box.height - self.get_height())
        content = 'q %d w ' % (float(self._line_width),)
        min_x: Decimal = min([min(l[0][0], l[1][0]) for l in self._lines])
        min_y: Decimal = min([min(l[0][1], l[1][1]) for l in self._lines])
        max_x: Decimal = max([min(l[0][0], l[1][0]) for l in self._lines])
        max_y: Decimal = max([max(l[0][1], l[1][1]) for l in self._lines])
        n: Decimal = Decimal(1)
        if self._gradient_type == GradientColoredDisconnectedShape.GradientType.DIAGONAL:
            n = Decimal(math.sqrt((min_x - max_x) ** 2 + (min_y - max_y) ** 2))
        if self._gradient_type == GradientColoredDisconnectedShape.GradientType.HORIZONTAL:
            n = max_x - min_x
        if self._gradient_type == GradientColoredDisconnectedShape.GradientType.RADIAL:
            mid_x = (max_x - min_x) / 2 + min_x
            mid_y = (max_y - min_y) / 2 + min_y
            n = Decimal(max([math.sqrt((l[0][0] - mid_x) ** 2 + (l[0][1] - mid_y) ** 2) for l in self._lines]))
        if self._gradient_type == GradientColoredDisconnectedShape.GradientType.VERTICAL:
            n = max_y - min_y
        n = Decimal(n)
        start_color: HSVColor = HSVColor.from_rgb(self._from_color.to_rgb())
        end_color: HSVColor = HSVColor.from_rgb(self._to_color.to_rgb())
        for l in self._lines:
            d: Decimal = Decimal(0)
            if self._gradient_type == GradientColoredDisconnectedShape.GradientType.DIAGONAL:
                d = Decimal(math.sqrt((l[0][0] - min_x) ** 2 + (l[0][1] - min_y) ** 2))
            if self._gradient_type == GradientColoredDisconnectedShape.GradientType.HORIZONTAL:
                d = Decimal(l[0][0] - min_x)
            if self._gradient_type == GradientColoredDisconnectedShape.GradientType.RADIAL:
                d = Decimal(math.sqrt((l[0][0] - mid_x) ** 2 + (l[0][1] - mid_y) ** 2))
            if self._gradient_type == GradientColoredDisconnectedShape.GradientType.VERTICAL:
                d = Decimal(l[0][1] - min_y)
            h = start_color.hue + (end_color.hue - start_color.hue) * (d / n)
            s = start_color.saturation + (end_color.saturation - start_color.saturation) * (d / n)
            v = start_color.value + (end_color.value - start_color.value) * (d / n)
            stroke_color: RGBColor = HSVColor(h, s, v).to_rgb()
            r: float = float(stroke_color.red)
            g: float = float(stroke_color.green)
            b: float = float(stroke_color.blue)
            content += '%f %f %f RG %f %f m %f %f l S ' % (r, g, b, float(l[0][0]), float(l[0][1]), float(l[1][0]), float(l[1][1]))
        content += ' Q'
        page.append_to_content_stream(content)