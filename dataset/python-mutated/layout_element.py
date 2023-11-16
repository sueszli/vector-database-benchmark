"""
This file contains all the classes needed to perform layout.
This includes an Alignment Enum type, and the base implementation of LayoutElement
"""
import typing
from decimal import Decimal
from enum import Enum
from borb.pdf.canvas.color.color import Color
from borb.pdf.canvas.color.color import HexColor
from borb.pdf.canvas.geometry.rectangle import Rectangle
from borb.pdf.canvas.line_art.blob_factory import BlobFactory

class Alignment(Enum):
    """
    In typesetting and page layout, alignment or range is the setting of text flow or image placement relative to a page,
    column (measure), table cell, or tab.
    The type alignment setting is sometimes referred to as text alignment,
    text justification, or type justification.
    The edge of a page or column is known as a margin, and a gap between columns is known as a gutter.
    """
    LEFT = 2
    CENTERED = 3
    RIGHT = 5
    JUSTIFIED = 7
    TOP = 11
    MIDDLE = 13
    BOTTOM = 17

class LayoutElement:
    """
    This class contains the common base methods for any object that can be laid out on a Page.
    e.g. the placement of borders, margins, padding, background color, etc
    """

    def __init__(self, background_color: typing.Optional[Color]=None, border_bottom: bool=False, border_color: Color=HexColor('000000'), border_left: bool=False, border_radius_bottom_left: Decimal=Decimal(0), border_radius_bottom_right: Decimal=Decimal(0), border_radius_top_left: Decimal=Decimal(0), border_radius_top_right: Decimal=Decimal(0), border_right: bool=False, border_top: bool=False, border_width: Decimal=Decimal(1), font_size: typing.Optional[Decimal]=None, horizontal_alignment: Alignment=Alignment.LEFT, margin_bottom: typing.Optional[Decimal]=Decimal(0), margin_left: typing.Optional[Decimal]=Decimal(0), margin_right: typing.Optional[Decimal]=Decimal(0), margin_top: typing.Optional[Decimal]=Decimal(0), padding_bottom: Decimal=Decimal(0), padding_left: Decimal=Decimal(0), padding_right: Decimal=Decimal(0), padding_top: Decimal=Decimal(0), parent: typing.Optional['LayoutElement']=None, vertical_alignment: Alignment=Alignment.TOP):
        if False:
            i = 10
            return i + 15
        self._background_color = background_color
        self._border_top = border_top
        self._border_right = border_right
        self._border_bottom = border_bottom
        self._border_left = border_left
        assert border_radius_top_right >= 0, 'border_radius_top_right must be a non-negative integer'
        assert border_radius_top_left >= 0, 'border_radius_top_left must be a non-negative integer'
        assert border_radius_bottom_left >= 0, 'border_radius_bottom_left must be a non-negative integer'
        assert border_radius_bottom_right >= 0, 'border_radius_bottom_right must be a non-negative integer'
        self._border_radius_top_left: Decimal = border_radius_top_left
        self._border_radius_top_right: Decimal = border_radius_top_right
        self._border_radius_bottom_right: Decimal = border_radius_bottom_right
        self._border_radius_bottom_left: Decimal = border_radius_bottom_left
        assert border_width >= 0
        self._border_width = border_width
        self._border_color = border_color
        self._font_size = font_size
        assert margin_top is None or margin_top >= 0
        assert margin_right is None or margin_right >= 0
        assert margin_bottom is None or margin_bottom >= 0
        assert margin_left is None or margin_left >= 0
        self._margin_top = margin_top
        self._margin_right = margin_right
        self._margin_bottom = margin_bottom
        self._margin_left = margin_left
        assert padding_top >= 0
        assert padding_right >= 0
        assert padding_bottom >= 0
        assert padding_left >= 0
        self._padding_top = padding_top
        self._padding_right = padding_right
        self._padding_bottom = padding_bottom
        self._padding_left = padding_left
        assert horizontal_alignment in [Alignment.LEFT, Alignment.CENTERED, Alignment.RIGHT, Alignment.JUSTIFIED]
        assert vertical_alignment in [Alignment.TOP, Alignment.MIDDLE, Alignment.BOTTOM]
        self._horizontal_alignment = horizontal_alignment
        self._vertical_alignment = vertical_alignment
        self._previous_layout_box: typing.Optional[Rectangle] = None
        self._previous_paint_box: typing.Optional[Rectangle] = None
        self._parent = parent

    def _get_border_outline(self, border_box: Rectangle) -> typing.List[typing.Optional[typing.Tuple[Decimal, Decimal]]]:
        if False:
            while True:
                i = 10
        n: int = 0
        xll: Decimal = round(border_box.get_x(), n)
        yll: Decimal = round(border_box.get_y(), n)
        xur: Decimal = round(border_box.get_x() + border_box.get_width(), n)
        yur: Decimal = round(border_box.get_y() + border_box.get_height(), n)
        points: typing.List[typing.Optional[typing.Tuple[Decimal, Decimal]]] = []
        if self._border_top and self._border_left and (self._border_radius_top_left != 0):
            points += [(xll, yur - self._border_radius_top_left)] + BlobFactory.smooth_closed_polygon([(xll, yur - self._border_radius_top_left), (xll, yur), (xll + self._border_radius_top_left, yur)], 2)[:-6]
        if self._border_left and self._border_radius_top_left == 0:
            points += [(xll, yur - self._border_radius_top_left)]
            points += [(xll, yur)]
        if self._border_top and self._border_radius_top_left == 0:
            points += [(xll + self._border_radius_top_left, yur)]
        if self._border_top:
            points += [(xur - self._border_radius_top_right, yur)]
        else:
            points += [None]
        if self._border_top and self._border_right and (self._border_radius_top_right != 0):
            points += BlobFactory.smooth_closed_polygon([(xur - self._border_radius_top_right, yur), (xur, yur), (xur, yur - self._border_radius_top_right)], 2)[:-6]
        if self._border_top and self._border_radius_top_right == 0:
            points += [(xur, yur)]
        if self._border_right and self._border_radius_top_right == 0:
            points += [(xur, yur - self._border_radius_top_right)]
        if self._border_right:
            points += [(xur, yll + self._border_radius_bottom_right)]
        else:
            points += [None]
        if self._border_bottom and self._border_right and (self._border_radius_bottom_right != 0):
            points += BlobFactory.smooth_closed_polygon([(xur, yll + self._border_radius_bottom_right), (xur, yll), (xur - self._border_radius_bottom_right, yll)], 2)[:-6]
        if self._border_right and self._border_radius_bottom_right == 0:
            points += [(xur, yll)]
        if self._border_bottom and self._border_radius_bottom_right == 0:
            points += [(xur - self._border_radius_bottom_right, yll)]
        if self._border_bottom:
            points += [(xll + self._border_radius_bottom_left, yll)]
        else:
            points += [None]
        if self._border_bottom and self._border_left and (self._border_radius_bottom_left != 0):
            points += BlobFactory.smooth_closed_polygon([(xll + self._border_radius_bottom_left, yll), (xll, yll), (xll, yll + self._border_radius_bottom_left)], 2)[:-6]
        if self._border_bottom and self._border_radius_bottom_left == 0:
            points += [(xll, yll)]
        if self._border_left and self._border_radius_bottom_left == 0:
            points += [(xll, yll + self._border_radius_bottom_right)]
        if self._border_left:
            points += [(xll, yur - self._border_radius_top_left)]
        else:
            points += [None]
        return points

    def _get_content_box(self, available_space: Rectangle) -> Rectangle:
        if False:
            while True:
                i = 10
        return Rectangle(available_space.get_x(), available_space.get_y() + available_space.get_height(), Decimal(0), Decimal(0))

    def _needs_to_be_tagged(self, p: 'Page') -> bool:
        if False:
            return 10
        '\n        This function returns whether this LayoutElement needs to be tagged\n        :param p:   the Page on which this LayoutElement is to be painted\n        :return:    true if this LayoutElement needs to be tagged, False otherwise\n        '
        document: typing.Optional['Document'] = p.get_document()
        if document is None:
            return False
        conformance_level: typing.Optional['ConformanceLevel'] = document.get_document_info().get_conformance_level_upon_create()
        if conformance_level is None:
            return False
        return conformance_level.get_conformance_level() in ['A', 'U']

    def _paint_background(self, page: 'Page', background_box: Rectangle):
        if False:
            i = 10
            return i + 15
        if not self._background_color:
            return
        assert self._background_color
        rgb_color = self._background_color.to_rgb()
        if self._border_radius_top_right == 0 and self._border_radius_top_left == 0 and (self._border_radius_bottom_left == 0) and (self._border_radius_bottom_right == 0):
            content = '\n                q %f %f %f rg %f %f m\n                %f %f l\n                %f %f l\n                %f %f l\n                %f %f l\n                f\n                Q\n                ' % (float(rgb_color.red), float(rgb_color.green), float(rgb_color.blue), background_box.get_x(), background_box.get_y() + background_box.get_height(), background_box.get_x() + background_box.get_width(), background_box.get_y() + background_box.get_height(), background_box.get_x() + background_box.get_width(), background_box.get_y(), background_box.get_x(), background_box.get_y(), background_box.get_x(), background_box.get_y() + background_box.get_height())
            page.append_to_content_stream(content)
            return
        before = [self._border_top, self._border_right, self._border_bottom, self._border_left]
        self._border_top = True
        self._border_right = True
        self._border_bottom = True
        self._border_left = True
        outline_points = self._get_border_outline(background_box)
        assert outline_points[0] is not None
        self._border_top = before[0]
        self._border_right = before[1]
        self._border_bottom = before[2]
        self._border_left = before[3]
        content = '\n            q %f %f %f rg %f %f m\n            ' % (float(rgb_color.red), float(rgb_color.green), float(rgb_color.blue), float(outline_points[0][0]), float(outline_points[0][1]))
        for p in outline_points:
            assert p is not None
            content += ' %f %f l' % (float(p[0]), float(p[1]))
        content += ' f Q'
        page.append_to_content_stream(content)

    def _paint_borders(self, page: 'Page', border_box: Rectangle):
        if False:
            print('Hello World!')
        if self._border_top == self._border_right == self._border_bottom == self._border_left == False:
            return
        if self._border_width == 0:
            return
        rgb_color = self._border_color.to_rgb()
        content = 'q %f %f %f RG %f w ' % (float(rgb_color.red), float(rgb_color.green), float(rgb_color.blue), float(self._border_width))
        points = self._get_border_outline(border_box)
        for (i, p) in enumerate(points[:-1]):
            p0: typing.Optional[typing.Tuple[Decimal, Decimal]] = p
            p1: typing.Optional[typing.Tuple[Decimal, Decimal]] = points[i + 1]
            if p0 is None or p1 is None:
                continue
            content += ' %d %d m %d %d l s' % (float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1]))
        content += ' Q'
        page.append_to_content_stream(content)

    def _paint_content_box(self, page: 'Page', content_box: Rectangle) -> None:
        if False:
            while True:
                i = 10
        pass

    def get_font_size(self) -> Decimal:
        if False:
            return 10
        '\n        This function returns the font size of this LayoutElement\n        '
        return self._font_size or Decimal(0)

    def get_golden_ratio_landscape_box(self) -> typing.Optional[Rectangle]:
        if False:
            print('Hello World!')
        '\n        This function returns the layout box that fits this LayoutElement\n        and whose ratio of dimensions (width / height) are closest to the golden ratio.\n        :return:    the layout box (in landscape mode) with ratio closest to the golden ratio\n        '
        GOLDEN_RATIO: Decimal = Decimal(1.618)
        INVERSE_GOLDEN_RATIO = Decimal(1) / GOLDEN_RATIO
        best_landscape_box: typing.Optional[Rectangle] = None
        for w in range(0, 2048, 10):
            try:
                landscape_box: Rectangle = self.get_layout_box(Rectangle(Decimal(0), Decimal(0), Decimal(w), Decimal(w) * INVERSE_GOLDEN_RATIO))
                if landscape_box.get_width() > w:
                    continue
                if best_landscape_box is None:
                    best_landscape_box = landscape_box
                    continue
                ratio: Decimal = landscape_box.get_width() / landscape_box.get_height()
                best_ratio: Decimal = best_landscape_box.get_width() / best_landscape_box.get_height()
                if abs(ratio - GOLDEN_RATIO) < abs(best_ratio - GOLDEN_RATIO):
                    best_landscape_box = landscape_box
                    continue
                if ratio > GOLDEN_RATIO:
                    break
            except:
                pass
        return best_landscape_box

    def get_golden_ratio_portrait_box(self) -> typing.Optional[Rectangle]:
        if False:
            print('Hello World!')
        '\n        This function returns the layout box that fits this LayoutElement\n        and whose ratio of dimensions (height / width) are closest to the golden ratio.\n        :return:    the layout box (in portrait mode) with ratio closest to the golden ratio\n        '
        GOLDEN_RATIO: Decimal = Decimal(1.618)
        INVERSE_GOLDEN_RATIO = Decimal(1) / GOLDEN_RATIO
        best_portrait_box: typing.Optional[Rectangle] = None
        for h in range(0, 2048, 10):
            try:
                portrait_box: Rectangle = self.get_layout_box(Rectangle(Decimal(0), Decimal(0), Decimal(h * INVERSE_GOLDEN_RATIO), Decimal(h)))
                if portrait_box.get_height() > h:
                    continue
                if best_portrait_box is None:
                    best_portrait_box = portrait_box
                    continue
                ratio: Decimal = portrait_box.get_height() / portrait_box.get_width()
                best_ratio: Decimal = best_portrait_box.get_height() / best_portrait_box.get_width()
                if abs(ratio - GOLDEN_RATIO) < abs(best_ratio - GOLDEN_RATIO):
                    best_portrait_box = portrait_box
                    continue
                if ratio < GOLDEN_RATIO:
                    break
            except:
                pass
        return best_portrait_box

    def get_largest_landscape_box(self) -> typing.Optional[Rectangle]:
        if False:
            print('Hello World!')
        '\n        This function returns the largest (in landscape mode) box that will fit this LayoutElement.\n        For most (all) LayoutElements, this also ought to be the layout box with the smallest height, and largest width.\n        :return:    the largest layout box (in landscape mode)\n        '
        try:
            return self.get_layout_box(Rectangle(Decimal(0), Decimal(0), Decimal(2048), Decimal(2048)))
        except:
            return None

    def get_layout_box(self, available_space: Rectangle):
        if False:
            print('Hello World!')
        '\n        This function returns the previous result of layout\n        :return:    the Rectangle that was the result of the previous layout operation\n        '
        horizontal_border_width: Decimal = Decimal(0)
        if self._border_left:
            horizontal_border_width += self._border_width
        if self._border_right:
            horizontal_border_width += self._border_width
        vertical_border_width: Decimal = Decimal(0)
        if self._border_top:
            vertical_border_width += self._border_width
        if self._border_bottom:
            vertical_border_width += self._border_width
        cbox_available_space: Rectangle = Rectangle(available_space.get_x() + self._padding_left + (self._border_width if self._border_left else Decimal(0)), available_space.get_y() + self._padding_bottom + (self._border_width if self._border_bottom else Decimal(0)), max(Decimal(0), available_space.get_width() - self._padding_left - self._padding_right - horizontal_border_width), max(Decimal(0), available_space.get_height() - self._padding_top - self._padding_bottom - vertical_border_width))
        cbox: Rectangle = self._get_content_box(cbox_available_space)
        delta_x: Decimal = Decimal(0)
        delta_y: Decimal = Decimal(0)
        if self._vertical_alignment == Alignment.MIDDLE:
            delta_y = (cbox_available_space.get_height() - cbox.get_height()) / Decimal(2)
            cbox.y -= delta_y
        if self._vertical_alignment == Alignment.BOTTOM:
            delta_y = cbox_available_space.get_height() - cbox.get_height()
            cbox.y -= delta_y
        if self._horizontal_alignment == Alignment.CENTERED:
            delta_x = (cbox_available_space.get_width() - cbox.get_width()) / Decimal(2)
            cbox.x += delta_x
        if self._horizontal_alignment == Alignment.RIGHT:
            delta_x = cbox_available_space.get_width() - cbox.get_width()
            cbox.x += delta_x
        self._previous_layout_box = Rectangle(cbox.get_x() - self._padding_left - (self._border_width if self._border_left else Decimal(0)), cbox.get_y() - self._padding_bottom - (self._border_width if self._border_bottom else Decimal(0)), cbox.get_width() + self._padding_left + self._padding_right + horizontal_border_width, cbox.get_height() + self._padding_top + self._padding_bottom + vertical_border_width)
        return self._previous_layout_box

    def get_margin_bottom(self) -> Decimal:
        if False:
            while True:
                i = 10
        '\n        This function returns the bottom margin of this LayoutElement\n        '
        return self._margin_bottom or Decimal(0)

    def get_margin_left(self) -> Decimal:
        if False:
            print('Hello World!')
        '\n        This function returns the left margin of this LayoutElement\n        '
        return self._margin_left or Decimal(0)

    def get_margin_right(self) -> Decimal:
        if False:
            for i in range(10):
                print('nop')
        '\n        This function returns the right margin of this LayoutElement\n        '
        return self._margin_right or Decimal(0)

    def get_margin_top(self) -> Decimal:
        if False:
            return 10
        '\n        This function returns the top margin of this LayoutElement\n        '
        return self._margin_top or Decimal(0)

    def get_previous_layout_box(self) -> typing.Optional[Rectangle]:
        if False:
            i = 10
            return i + 15
        '\n        This function returns the previous result of layout of this LayoutElement\n        :return:    the Rectangle that was the result of the previous layout operation\n        '
        return self._previous_layout_box

    def get_previous_paint_box(self) -> typing.Optional[Rectangle]:
        if False:
            for i in range(10):
                print('nop')
        '\n        This function returns the previous result of painting this LayoutElement\n        :return:    the Rectangle that was the result of the previous paint operation\n        '
        return self._previous_paint_box

    def get_smallest_landscape_box(self) -> typing.Optional[Rectangle]:
        if False:
            print('Hello World!')
        '\n        This function returns the smallest (in landscape mode) box that will fit this LayoutElement.\n        For most (all) LayoutElements, this also ought to be the layout box with the smallest width, and largest height.\n        :return:    the smallest layout box (in landscape mode)\n        '
        max_width: Decimal = Decimal(2048)
        min_width: Decimal = Decimal(0)
        midpoint_width: Decimal = (max_width + min_width) / Decimal(2)
        landscape_box: typing.Optional[Rectangle] = None
        while abs(max_width - min_width) > Decimal(1):
            try:
                landscape_box: Rectangle = self.get_layout_box(Rectangle(Decimal(0), Decimal(0), midpoint_width, Decimal(2048)))
                if landscape_box.get_width() > midpoint_width:
                    min_width = midpoint_width
                else:
                    max_width = midpoint_width
                midpoint_width = (max_width + min_width) / Decimal(2)
            except:
                break
        return landscape_box

    def paint(self, page: 'Page', available_space: Rectangle) -> None:
        if False:
            i = 10
            return i + 15
        '\n        This method paints this LayoutElement on the given Page, in the available space\n        :param page:                the Page on which to paint this LayoutElement\n        :param available_space:     the available space (as a Rectangle) on which to paint this LayoutElement\n        :return:                    None\n        '
        horizontal_border_width: Decimal = Decimal(0)
        if self._border_left:
            horizontal_border_width += self._border_width
        if self._border_right:
            horizontal_border_width += self._border_width
        vertical_border_width: Decimal = Decimal(0)
        if self._border_top:
            vertical_border_width += self._border_width
        if self._border_bottom:
            vertical_border_width += self._border_width
        cbox_available_space: Rectangle = Rectangle(available_space.get_x() + self._padding_left + (self._border_width if self._border_left else Decimal(0)), available_space.get_y() + self._padding_bottom + (self._border_width if self._border_bottom else Decimal(0)), max(Decimal(0), available_space.get_width() - self._padding_left - self._padding_right - horizontal_border_width), max(Decimal(0), available_space.get_height() - self._padding_top - self._padding_bottom - vertical_border_width))
        cbox: Rectangle = self._get_content_box(cbox_available_space)
        if round(cbox.get_height(), 2) > round(cbox_available_space.get_height(), 2):
            assert False, f'{self.__class__.__name__} is too tall to fit inside column / page. Needed {round(cbox.get_height(), 2)} pts, only {round(cbox_available_space.get_height(), 2)} pts available.'
        if round(cbox.get_width(), 2) > round(cbox_available_space.get_width(), 2):
            self._get_content_box(cbox_available_space)
            assert False, f'{self.__class__.__name__} is too wide to fit inside column / page. Needed {round(cbox.get_width(), 2)} pts, only {round(cbox_available_space.get_width(), 2)} pts available.'
        delta_x: Decimal = Decimal(0)
        delta_y: Decimal = Decimal(0)
        if self._vertical_alignment == Alignment.MIDDLE:
            delta_y = (cbox_available_space.get_height() - cbox.get_height()) / Decimal(2)
            cbox.y -= delta_y
        if self._vertical_alignment == Alignment.BOTTOM:
            delta_y = cbox_available_space.get_height() - cbox.get_height()
            cbox.y -= delta_y
        if self._horizontal_alignment == Alignment.CENTERED:
            delta_x = (cbox_available_space.get_width() - cbox.get_width()) / Decimal(2)
            cbox.x += delta_x
        if self._horizontal_alignment == Alignment.RIGHT:
            delta_x = cbox_available_space.get_width() - cbox.get_width()
            cbox.x += delta_x
        bgbox: Rectangle = Rectangle(cbox.get_x() - self._padding_left - (self._border_width if self._border_left else Decimal(0)), cbox.get_y() - self._padding_bottom - (self._border_width if self._border_bottom else Decimal(0)), cbox.get_width() + self._padding_left + self._padding_right + horizontal_border_width, cbox.get_height() + self._padding_top + self._padding_bottom + vertical_border_width)
        self._paint_background(page, bgbox)
        self._paint_borders(page, bgbox)
        self._paint_content_box(page, cbox)
        self._previous_paint_box = bgbox