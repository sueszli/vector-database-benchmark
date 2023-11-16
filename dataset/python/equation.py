#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
In mathematics, an equation is a formula that expresses the equality of two expressions,
by connecting them with the equals sign =. The word equation and its cognates in other languages may have subtly
different meanings; for example, in French an équation is defined as containing one or more variables,
while in English, any well-formed formula consisting of two expressions related with an equals sign is an equation.
"""
import typing
from decimal import Decimal

from borb.pdf.canvas.color.color import Color
from borb.pdf.canvas.color.color import HexColor
from borb.pdf.canvas.font.font import Font
from borb.pdf.canvas.layout.equation.parser import Parser
from borb.pdf.canvas.layout.equation.token import Token
from borb.pdf.canvas.layout.equation.token import TokenType
from borb.pdf.canvas.layout.layout_element import Alignment
from borb.pdf.canvas.layout.layout_element import LayoutElement
from borb.pdf.canvas.layout.page_layout.inline_flow import InlineFlow
from borb.pdf.canvas.layout.table.flexible_column_width_table import (
    FlexibleColumnWidthTable,
)
from borb.pdf.canvas.layout.table.table import Table
from borb.pdf.canvas.layout.table.table import TableCell
from borb.pdf.canvas.layout.text.chunk_of_text import ChunkOfText
from borb.pdf.canvas.layout.text.paragraph import Paragraph


class Equation(InlineFlow):
    """
    In mathematics, an equation is a formula that expresses the equality of two expressions,
    by connecting them with the equals sign =. The word equation and its cognates in other languages may have subtly
    different meanings; for example, in French an équation is defined as containing one or more variables,
    while in English, any well-formed formula consisting of two expressions related with an equals sign is an equation.
    """

    #
    # CONSTRUCTOR
    #

    def __init__(
        self,
        text: str,
        font: typing.Union[Font, str] = "Helvetica",
        font_size: Decimal = Decimal(12),
        vertical_alignment: Alignment = Alignment.TOP,
        horizontal_alignment: Alignment = Alignment.LEFT,
        font_color: Color = HexColor("000000"),
        border_top: bool = False,
        border_right: bool = False,
        border_bottom: bool = False,
        border_left: bool = False,
        border_radius_top_left: Decimal = Decimal(0),
        border_radius_top_right: Decimal = Decimal(0),
        border_radius_bottom_right: Decimal = Decimal(0),
        border_radius_bottom_left: Decimal = Decimal(0),
        border_color: Color = HexColor("000000"),
        border_width: Decimal = Decimal(1),
        padding_top: Decimal = Decimal(0),
        padding_right: Decimal = Decimal(0),
        padding_bottom: Decimal = Decimal(0),
        padding_left: Decimal = Decimal(0),
        margin_top: typing.Optional[Decimal] = None,
        margin_right: typing.Optional[Decimal] = None,
        margin_bottom: typing.Optional[Decimal] = None,
        margin_left: typing.Optional[Decimal] = None,
        background_color: typing.Optional[Color] = None,
    ):
        super(Equation, self).__init__()
        self._font = font
        self._font_size = font_size
        self._vertical_alignment = vertical_alignment
        self._horizontal_alignment = horizontal_alignment
        self._font_color = font_color
        self._border_top = border_top
        self._border_right = border_right
        self._border_bottom = border_bottom
        self._border_left = border_left
        self._border_radius_top_left = border_radius_top_left
        self._border_radius_top_right = border_radius_top_right
        self._border_radius_bottom_right = border_radius_bottom_right
        self._border_radius_bottom_left = border_radius_bottom_left
        self._border_color = border_color
        self._border_width = border_width
        self._padding_top = padding_top
        self._padding_right = padding_right
        self._padding_bottom = padding_bottom
        self._padding_left = padding_left
        self._margin_top = margin_top
        self._margin_right = margin_right
        self._margin_bottom = margin_bottom
        self._margin_left = margin_left
        self._background_color = background_color
        self.add(
            self._build(
                Parser.to_abstract_syntax_tree(text),
                font_color=font_color,
                font_size=font_size,
            )
        )

    #
    # PRIVATE
    #

    def _build(
        self,
        t: Token,
        font_color: Color = HexColor("#000000"),
        font_size: Decimal = Decimal(12),
    ) -> LayoutElement:
        if t.get_type() == TokenType.COMMA:
            return ChunkOfText(",", font_size=font_size, font_color=font_color)
        if t.get_type() == TokenType.FUNCTION:
            #
            # abs
            #
            if t.get_text() == "abs":
                c0 = self._build(
                    t.get_children()[0], font_size=font_size, font_color=font_color
                )
                c0._border_width = Decimal(0.75)
                c0._border_left = True
                c0._border_right = True
                return c0
            #
            # any other FUNCTION
            #
            c0: LayoutElement = self._build(
                t.get_children()[0], font_size=font_size, font_color=font_color
            )
            c0._vertical_alignment = Alignment.MIDDLE
            out: Table = FlexibleColumnWidthTable(number_of_columns=4, number_of_rows=1)
            out.add(
                ChunkOfText(
                    t.get_text(),
                    font_size=font_size,
                    font_color=font_color,
                    vertical_alignment=Alignment.MIDDLE,
                )
            )
            out.add(
                ChunkOfText(
                    "(",
                    font_size=font_size,
                    font_color=font_color,
                    vertical_alignment=Alignment.MIDDLE,
                )
            )
            out.add(c0)
            out.add(
                ChunkOfText(
                    ")",
                    font_size=font_size,
                    font_color=font_color,
                    vertical_alignment=Alignment.MIDDLE,
                )
            )
            out.no_borders()
            return out
        if t.get_type() == TokenType.LEFT_PARENTHESIS:
            return ChunkOfText("(", font_size=font_size, font_color=font_color)
        if t.get_type() == TokenType.NUMBER:
            return ChunkOfText(t.get_text(), font_size=font_size, font_color=font_color)
        if t.get_type() == TokenType.OPERATOR:
            #
            # unary minus
            #
            if t.get_text() == "-" and t.get_number_of_arguments() == 1:
                out: Table = FlexibleColumnWidthTable(
                    number_of_columns=2, number_of_rows=1
                )
                out.add(
                    ChunkOfText(
                        "-",
                        font_size=font_size,
                        font_color=font_color,
                        vertical_alignment=Alignment.MIDDLE,
                    )
                )
                out.add(
                    self._build(
                        t.get_children()[0], font_size=font_size, font_color=font_color
                    )
                )
                out.no_borders()
                return out
            #
            # division
            #
            if t.get_text() == "/":
                e0: LayoutElement = self._build(
                    t.get_children()[0], font_size=font_size, font_color=font_color
                )
                e0._horizontal_alignment = Alignment.CENTERED
                e1: LayoutElement = self._build(
                    t.get_children()[1], font_size=font_size, font_color=font_color
                )
                e1._horizontal_alignment = Alignment.CENTERED
                out: Table = FlexibleColumnWidthTable(
                    number_of_columns=1, number_of_rows=2
                )
                out.add(
                    TableCell(
                        e1,
                        border_top=False,
                        border_right=False,
                        border_bottom=True,
                        border_left=False,
                        border_color=font_color,
                        border_width=Decimal(0.75),
                    )
                )
                out.add(
                    TableCell(
                        e0,
                        border_top=False,
                        border_right=False,
                        border_bottom=False,
                        border_left=False,
                        padding_top=Decimal(3),
                    )
                )
                return out
            #
            # power
            #
            if t.get_text() == "^":
                e0: LayoutElement = self._build(
                    t.get_children()[0],
                    font_size=font_size * Decimal(0.5),
                    font_color=font_color,
                )
                e0._horizontal_alignment = Alignment.BOTTOM
                e1: LayoutElement = self._build(
                    t.get_children()[1], font_size=font_size, font_color=font_color
                )
                e1._horizontal_alignment = Alignment.CENTERED
                out: Table = FlexibleColumnWidthTable(
                    number_of_columns=2, number_of_rows=2
                )
                out.add(Paragraph(" ", font_size=font_size * Decimal(0.5)))
                out.add(e0)
                out.add(e1)
                out.add(Paragraph(" "))
                out.set_border_width_on_all_cells(Decimal(0))
                out.no_borders()
                return out
            #
            # any other OPERATOR
            #
            op_text: str = t.get_text()
            if op_text == "*":
                op_text = "·"
            c0: LayoutElement = self._build(
                t.get_children()[0], font_size=font_size, font_color=font_color
            )
            c0._vertical_alignment = Alignment.MIDDLE
            c1: LayoutElement = self._build(
                t.get_children()[1], font_size=font_size, font_color=font_color
            )
            c1._vertical_alignment = Alignment.MIDDLE
            out: Table = FlexibleColumnWidthTable(number_of_columns=3, number_of_rows=1)
            out.add(c1)
            out.add(
                ChunkOfText(
                    op_text,
                    font_size=font_size,
                    font_color=font_color,
                    vertical_alignment=Alignment.MIDDLE,
                    padding_right=Decimal(2),
                    padding_left=Decimal(2),
                )
            )
            out.add(c0)
            out.no_borders()
            return out

        if t.get_type() == TokenType.RIGHT_PARENTHESIS:
            return ChunkOfText(")", font_size=font_size, font_color=font_color)
        if t.get_type() == TokenType.VARIABLE:
            return ChunkOfText(t.get_text(), font_size=font_size, font_color=font_color)

    #
    # PUBLIC
    #
