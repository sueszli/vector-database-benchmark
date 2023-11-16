"""
This class represents a Table with columns that will assume
a width based on their contents. It tries to emulate the behaviour
of <table> elements in HTML
"""
import typing
from decimal import Decimal
from borb.pdf.canvas.color.color import Color
from borb.pdf.canvas.color.color import HexColor
from borb.pdf.canvas.geometry.rectangle import Rectangle
from borb.pdf.canvas.layout.layout_element import Alignment
from borb.pdf.canvas.layout.table.table import Table
from borb.pdf.canvas.layout.text.paragraph import Paragraph
from borb.pdf.page.page import Page

class FlexibleColumnWidthTable(Table):
    """
    This class represents a Table with columns that will assume
    a width based on their contents. It tries to emulate the behaviour
    of <table> elements in HTML
    """

    def __init__(self, number_of_rows: int, number_of_columns: int, border_top: bool=False, border_right: bool=False, border_bottom: bool=False, border_left: bool=False, border_radius_bottom_left: Decimal=Decimal(0), border_radius_bottom_right: Decimal=Decimal(0), border_radius_top_left: Decimal=Decimal(0), border_radius_top_right: Decimal=Decimal(0), border_color: Color=HexColor('000000'), border_width: Decimal=Decimal(1), padding_top: Decimal=Decimal(0), padding_right: Decimal=Decimal(0), padding_bottom: Decimal=Decimal(0), padding_left: Decimal=Decimal(0), margin_top: Decimal=Decimal(0), margin_right: Decimal=Decimal(0), margin_bottom: Decimal=Decimal(0), margin_left: Decimal=Decimal(0), horizontal_alignment: Alignment=Alignment.LEFT, vertical_alignment: Alignment=Alignment.TOP, background_color: typing.Optional[Color]=None):
        if False:
            while True:
                i = 10
        super(FlexibleColumnWidthTable, self).__init__(number_of_rows=number_of_rows, number_of_columns=number_of_columns, background_color=background_color, border_bottom=border_bottom, border_color=border_color, border_left=border_left, border_radius_bottom_left=border_radius_bottom_left, border_radius_bottom_right=border_radius_bottom_right, border_radius_top_left=border_radius_top_left, border_radius_top_right=border_radius_top_right, border_right=border_right, border_top=border_top, border_width=border_width, horizontal_alignment=horizontal_alignment, margin_bottom=margin_bottom if margin_bottom is not None else Decimal(5), margin_left=margin_left if margin_left is not None else Decimal(5), margin_right=margin_right if margin_right is not None else Decimal(5), margin_top=margin_top if margin_top is not None else Decimal(5), padding_bottom=padding_bottom, padding_left=padding_left, padding_right=padding_right, padding_top=padding_top, vertical_alignment=vertical_alignment)

    def _get_content_box(self, available_space: Rectangle) -> Rectangle:
        if False:
            i = 10
            return i + 15
        number_of_cells: int = self._number_of_rows * self._number_of_columns
        empty_cells: int = number_of_cells - sum([x.get_row_span() * x.get_column_span() for x in self._content])
        for _ in range(0, empty_cells):
            self.add(Paragraph(' ', respect_spaces_in_text=True))
        m = self._get_grid_coordinates(available_space)
        min_x: Decimal = m[0][0][0]
        max_x: Decimal = m[-1][-1][0]
        min_y: Decimal = m[-1][-1][1]
        max_y: Decimal = m[0][0][1]
        return Rectangle(available_space.get_x(), min_y, Decimal(max_x - min_x), max_y - min_y)

    def _get_grid_coordinates(self, available_space: Rectangle) -> typing.List[typing.List[typing.Tuple[Decimal, Decimal]]]:
        if False:
            while True:
                i = 10
        for t in self._content:
            r0: typing.Optional[Rectangle] = t.get_largest_landscape_box()
            assert r0 is not None
            t._max_width = r0.get_width()
            t._min_height = r0.get_height()
            r1: typing.Optional[Rectangle] = t.get_smallest_landscape_box()
            assert r1 is not None
            t._min_width = r1.get_width()
            t._max_height = r1.get_height()
        min_column_widths: typing.List[Decimal] = [self._get_min_column_width(i) for i in range(0, self._number_of_columns)]
        max_column_widths: typing.List[Decimal] = [self._get_max_column_width(i) for i in range(0, self._number_of_columns)]
        for table_cell in self._content:
            if table_cell.get_column_span() == 1:
                continue
            column_indices: typing.Set[int] = set([y for (x, y) in table_cell.get_table_coordinates()])
            sum_of_min_col_spans: Decimal = Decimal(sum([min_column_widths[x] for x in column_indices]))
            assert table_cell.get_min_width() is not None
            if sum_of_min_col_spans < table_cell.get_min_width():
                delta: Decimal = table_cell.get_min_width() - sum_of_min_col_spans
                min_column_widths = [w + delta / table_cell.get_column_span() if i in column_indices else w for (i, w) in enumerate(min_column_widths)]
            sum_of_max_col_spans: Decimal = Decimal(sum([max_column_widths[x] for x in column_indices]))
            assert table_cell.get_max_width() is not None
            if sum_of_max_col_spans < table_cell.get_max_width():
                delta = table_cell.get_max_width() - sum_of_max_col_spans
                max_column_widths = [w + delta / table_cell.get_column_span() if i in column_indices else w for (i, w) in enumerate(max_column_widths)]
        column_widths: typing.List[Decimal] = [x for x in min_column_widths]
        number_of_expandable_columns: int = sum([1 for i in range(0, len(column_widths)) if column_widths[i] < max_column_widths[i]])
        delta: Decimal = Decimal(1)
        while round(sum(column_widths) + number_of_expandable_columns * delta, 2) <= round(available_space.get_width(), 2) and number_of_expandable_columns > 0:
            for i in range(0, len(column_widths)):
                if column_widths[i] < max_column_widths[i]:
                    column_widths[i] += delta
            number_of_expandable_columns = sum([1 for i in range(0, len(column_widths)) if column_widths[i] < max_column_widths[i]])
        grid_x_to_page_x: typing.List[Decimal] = [available_space.get_x()]
        for i in range(1, self._number_of_columns + 1):
            prev_x: Decimal = grid_x_to_page_x[-1]
            new_x: Decimal = prev_x + column_widths[i - 1]
            grid_x_to_page_x.append(new_x)
        grid_y_to_page_y: typing.List[Decimal] = [available_space.get_y() + available_space.get_height()]
        for r in range(0, self._number_of_rows):
            prev_row_lboxes: typing.List[Rectangle] = []
            for e in [x for x in self.get_cells_at_row(r) if x.get_row_span() == 1]:
                grid_x: int = min([p[1] for p in e.get_table_coordinates()])
                prev_vertical_alignment = e.get_layout_element()._vertical_alignment
                e.get_layout_element()._vertical_alignment = Alignment.TOP
                prev_row_lboxes.append(e.get_layout_box(Rectangle(grid_x_to_page_x[grid_x], available_space.get_y(), grid_x_to_page_x[grid_x + e.get_column_span()] - grid_x_to_page_x[grid_x], max(grid_y_to_page_y[r] - available_space.get_y(), Decimal(0)))))
                e.get_layout_element()._vertical_alignment = prev_vertical_alignment
            new_y: Decimal = min([lbox.get_y() for lbox in prev_row_lboxes])
            row_height: Decimal = grid_y_to_page_y[-1] - new_y
            grid_y_to_page_y.append(new_y)
            for e in [x for x in self.get_cells_at_row(r) if x.get_row_span() == 1]:
                grid_x: int = min([p[1] for p in e.get_table_coordinates()])
                if e.get_layout_element()._vertical_alignment == Alignment.TOP:
                    continue
                e.get_layout_box(Rectangle(grid_x_to_page_x[grid_x], new_y, grid_x_to_page_x[grid_x + e.get_column_span()] - grid_x_to_page_x[grid_x], row_height))
        return [[(x, y) for y in grid_y_to_page_y] for x in grid_x_to_page_x]

    def _get_max_column_width(self, col: int) -> Decimal:
        if False:
            print('Hello World!')
        widths: typing.List[Decimal] = []
        for table_cell in [x for x in self.get_cells_at_column(col) if x.get_column_span() == 1]:
            if table_cell.get_max_width() is None:
                widths.append(Decimal(2048))
                continue
            if table_cell.get_preferred_width() is None:
                widths.append(table_cell.get_max_width())
                continue
            if table_cell.get_preferred_width() < table_cell.get_max_width():
                widths.append(table_cell.get_preferred_width())
                continue
            widths.append(table_cell.get_max_width())
        if len(widths) == 0:
            return Decimal(2048)
        return max(widths)

    def _get_min_column_width(self, col: int) -> Decimal:
        if False:
            print('Hello World!')
        widths: typing.List[Decimal] = []
        for table_cell in [x for x in self.get_cells_at_column(col) if x.get_column_span() == 1]:
            if table_cell.get_min_width() is None:
                widths.append(Decimal(0))
                continue
            if table_cell.get_preferred_width() is None:
                assert table_cell.get_min_width() is not None
                widths.append(table_cell.get_min_width())
                continue
            if table_cell.get_preferred_width() > table_cell.get_min_width():
                assert table_cell.get_preferred_width() is not None
                widths.append(table_cell.get_preferred_width())
                continue
            assert table_cell.get_min_width() is not None
            widths.append(table_cell.get_min_width())
        if len(widths) == 0:
            return Decimal(0)
        return max(widths)

    def _paint_content_box(self, page: Page, available_space: Rectangle) -> None:
        if False:
            print('Hello World!')
        number_of_cells: int = self._number_of_rows * self._number_of_columns
        empty_cells: int = number_of_cells - sum([x.get_row_span() * x.get_column_span() for x in self._content])
        for _ in range(0, empty_cells):
            self.add(Paragraph(' ', respect_spaces_in_text=True))
        m: typing.List[typing.List[typing.Tuple[Decimal, Decimal]]] = self._get_grid_coordinates(available_space)
        for e in self._content:
            grid_x: int = min([p[1] for p in e.get_table_coordinates()])
            grid_y: int = min([p[0] for p in e.get_table_coordinates()])
            x: Decimal = m[grid_x][grid_y][0]
            y: Decimal = m[grid_x][grid_y + e.get_row_span()][1]
            w: Decimal = m[grid_x + e.get_column_span()][grid_y][0] - x
            h: Decimal = m[grid_x][grid_y][1] - y
            cbox: Rectangle = Rectangle(x, y, w, h)
            e._set_layout_box(cbox)
            e.paint(page, cbox)