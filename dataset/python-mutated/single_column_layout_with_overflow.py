"""
This implementation of PageLayout adds left/right/top/bottom margins to a Page
and lays out the content on the Page as if there were was a single column to flow text, images, etc into.
Once this column is full, the next page is automatically created.
"""
import copy
import typing
from decimal import Decimal
from borb.pdf.canvas.geometry.rectangle import Rectangle
from borb.pdf.canvas.layout.layout_element import LayoutElement
from borb.pdf.canvas.layout.page_layout.multi_column_layout import SingleColumnLayout

class SingleColumnLayoutWithOverflow(SingleColumnLayout):
    """
    This implementation of PageLayout adds left/right/top/bottom margins to a Page
    and lays out the content on the Page as if there were was a single column to flow text, images, etc into.
    Once this column is full, the next page is automatically created.
    """

    @staticmethod
    def _prepare_table_for_relayout(layout_element: LayoutElement):
        if False:
            return 10
        from borb.pdf.canvas.layout.table.table import Table
        assert isinstance(layout_element, Table)
        layout_element._previous_layout_box = None
        layout_element._previous_paint_box = None
        for tc in layout_element._content:
            tc._previous_layout_box = None
            tc._previous_paint_box = None
            tc._forced_layout_box = None
            tc.get_layout_element()._previous_layout_box = None
            tc.get_layout_element()._previous_paint_box = None

    def _split_blockflow(self, layout_element: LayoutElement, available_height: Decimal) -> typing.List[LayoutElement]:
        if False:
            while True:
                i = 10
        from borb.pdf.canvas.layout.page_layout.block_flow import BlockFlow
        assert isinstance(layout_element, BlockFlow)
        return layout_element._content

    def _split_table(self, layout_element: LayoutElement, available_height: Decimal) -> typing.List[LayoutElement]:
        if False:
            for i in range(10):
                print('nop')
        from borb.pdf.canvas.layout.table.table import Table
        assert isinstance(layout_element, Table)
        top_y: typing.Optional[Decimal] = None
        best_row_for_split: typing.Optional[int] = None
        for i in range(0, layout_element.get_number_of_rows()):
            prev_layout_box: typing.Optional[Rectangle] = layout_element.get_cells_at_row(i)[0].get_previous_layout_box()
            if top_y is None or top_y < prev_layout_box.get_y() + prev_layout_box.get_height():
                top_y = prev_layout_box.get_y() + prev_layout_box.get_height()
            assert top_y is not None
            if any([x.get_row_span() != 1 for x in layout_element.get_cells_at_row(i)]):
                continue
            assert prev_layout_box is not None
            y: Decimal = prev_layout_box.get_y()
            h: Decimal = round(top_y - y, 2)
            if h < available_height:
                best_row_for_split = i
        if best_row_for_split is None:
            assert False, '%s is too tall to fit inside column / page.' % layout_element.__class__.__name__
        t0 = copy.deepcopy(layout_element)
        t0._number_of_rows = best_row_for_split + 1
        t0._content = [x for x in t0._content if all([y[0] <= best_row_for_split for y in x.get_table_coordinates()])]
        SingleColumnLayoutWithOverflow._prepare_table_for_relayout(t0)
        t1 = copy.deepcopy(layout_element)
        t1._number_of_rows = layout_element.get_number_of_rows() - best_row_for_split - 1
        t1._content = [x for x in t1._content if all([y[0] > best_row_for_split for y in x.get_table_coordinates()])]
        for tc in t1._content:
            tc._table_coordinates = [(y - best_row_for_split - 1, x) for (y, x) in tc.get_table_coordinates()]
        SingleColumnLayoutWithOverflow._prepare_table_for_relayout(t1)
        return [t0, t1]

    def add(self, layout_element: LayoutElement) -> 'PageLayout':
        if False:
            while True:
                i = 10
        '\n        This method adds a `LayoutElement` to the current `Page`.\n        '
        if layout_element.__class__.__name__ not in ['BlockFlow', 'FlexibleColumnWidthTable', 'FixedColumnWidthTable']:
            return super(SingleColumnLayout, self).add(layout_element)
        page_width: typing.Optional[Decimal] = self._page.get_page_info().get_width()
        page_height: typing.Optional[Decimal] = self._page.get_page_info().get_height()
        assert page_width is not None
        assert page_height is not None
        max_y: Decimal = page_height - self._margin_top
        min_y: Decimal = self._margin_bottom
        if self._previous_layout_element is not None:
            max_y = self._previous_layout_element.get_previous_layout_box().get_y()
            max_y -= super()._calculate_leading_between(self._previous_layout_element, layout_element)
            max_y -= max(self._previous_layout_element.get_margin_bottom(), layout_element.get_margin_top())
        available_height: Decimal = max_y - min_y
        if available_height < 0:
            self.switch_to_next_column()
            return self.add(layout_element)
        available_box: Rectangle = Rectangle(self._margin_left + sum(self._column_widths[0:self._active_column]) + sum(self._inter_column_margins[0:self._active_column]) + layout_element.get_margin_left(), min_y, self._column_widths[self._active_column] - layout_element.get_margin_right() - layout_element.get_margin_left(), available_height)
        layout_box = layout_element.get_layout_box(available_box)
        if round(layout_box.get_width(), 2) > round(self._column_widths[self._active_column], 2):
            assert False, f'{layout_element.__class__.__name__} is too wide to fit inside column / page. Needed {round(layout_box.get_width())} pts, only {round(available_box.get_width())} pts available.'
        if round(layout_box.get_height(), 2) <= round(available_box.get_height(), 2):
            return super(SingleColumnLayout, self).add(layout_element)
        else:
            if self._previous_layout_element is not None:
                self.switch_to_next_column()
                return self.add(layout_element)
            if layout_element.__class__.__name__ in ['FlexibleColumnWidthTable', 'FixedColumnWidthTable']:
                for t in self._split_table(layout_element, available_height):
                    super(SingleColumnLayoutWithOverflow, self).add(t)
                return self
            if layout_element.__class__.__name__ in ['BlockFlow']:
                for t in self._split_blockflow(layout_element, available_height):
                    super(SingleColumnLayoutWithOverflow, self).add(t)
                return self
            assert False, f'{layout_element.__class__.__name__} is too tall to fit inside column / page. Needed {round(layout_box.get_height())} pts, only {round(available_box.get_height())} pts available.'