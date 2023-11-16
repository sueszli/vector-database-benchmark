"""
This implementation of LayoutElement aggregates other LayoutElements
and lays them out underneath each other.
"""
import typing
from decimal import Decimal
from borb.pdf.canvas.geometry.rectangle import Rectangle
from borb.pdf.canvas.layout.layout_element import LayoutElement

class BlockFlow(LayoutElement):
    """
    This implementation of LayoutElement aggregates other LayoutElements
    and lays them out underneath each other.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(BlockFlow, self).__init__()
        self._content: typing.List[LayoutElement] = []

    def _get_content_box(self, available_space: Rectangle) -> Rectangle:
        if False:
            i = 10
            return i + 15
        tallest_y_coordinate: Decimal = available_space.get_y() + available_space.get_height()
        if len(self._content) > 0:
            tallest_y_coordinate -= self._content[0].get_margin_top()
            tallest_y_coordinate = max(tallest_y_coordinate, Decimal(0))
        for (i, e) in enumerate(self._content):
            lbox: Rectangle = e.get_layout_box(Rectangle(available_space.get_x(), available_space.get_y(), available_space.get_width(), max(tallest_y_coordinate - available_space.get_y(), Decimal(0))))
            tallest_y_coordinate = lbox.get_y()
            if i + 1 < len(self._content):
                margin: Decimal = max(e.get_margin_bottom(), self._content[i + 1].get_margin_top())
                tallest_y_coordinate -= margin
            else:
                tallest_y_coordinate -= e.get_margin_bottom()
        return Rectangle(available_space.get_x(), tallest_y_coordinate, available_space.get_width(), max(available_space.get_y() + available_space.get_height() - tallest_y_coordinate, Decimal(0)))

    def _paint_content_box(self, page: 'Page', content_box: Rectangle) -> None:
        if False:
            for i in range(10):
                print('nop')
        tallest_y_coordinate: Decimal = content_box.get_y() + content_box.get_height()
        if len(self._content) > 0:
            tallest_y_coordinate -= self._content[0].get_margin_top()
            tallest_y_coordinate = max(tallest_y_coordinate, Decimal(0))
        for (i, e) in enumerate(self._content):
            e.paint(page, Rectangle(content_box.get_x(), content_box.get_y(), content_box.get_width(), max(tallest_y_coordinate - content_box.get_y(), Decimal(0))))
            tallest_y_coordinate = e.get_previous_paint_box().get_y()
            if i + 1 < len(self._content):
                margin: Decimal = max(e.get_margin_bottom(), self._content[i + 1].get_margin_top())
                tallest_y_coordinate -= margin
            else:
                tallest_y_coordinate -= e.get_margin_bottom()

    def add(self, e: LayoutElement) -> 'BlockFlow':
        if False:
            for i in range(10):
                print('nop')
        '\n        This function adds a LayoutElement to this BlockFlow\n        :param e:   the LayoutElement to be added\n        :return:    self\n        '
        if len(self._content) > 0 and self._content[-1].__class__.__name__ == 'InlineFlow' and (e.__class__.__name__ == 'InlineFlow'):
            self._content[-1].add(e)
            return self
        self._content.append(e)
        return self

    def extend(self, es: typing.List[LayoutElement]) -> 'BlockFlow':
        if False:
            while True:
                i = 10
        '\n        This function adds a typing.List of LayoutElement(s) to this BlockFlow\n        :param es:   the LayoutElements to be added\n        :return:    self\n        '
        for e in es:
            self.add(e)
        return self