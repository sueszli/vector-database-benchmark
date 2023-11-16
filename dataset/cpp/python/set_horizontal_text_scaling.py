#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Set the horizontal scaling, Th , to (scale ÷ 100). scale shall be a number
specifying the percentage of the normal width. Initial value: 100 (normal
width).
"""
import typing
from decimal import Decimal

from borb.io.read.types import AnyPDFType
from borb.pdf.canvas.operator.canvas_operator import CanvasOperator


class SetHorizontalScaling(CanvasOperator):
    """
    Set the horizontal scaling, Th , to (scale ÷ 100). scale shall be a number
    specifying the percentage of the normal width. Initial value: 100 (normal
    width).
    """

    def __init__(self):
        super().__init__("Tz", 1)

    def invoke(
        self,
        canvas_stream_processor: "CanvasStreamProcessor",  # type: ignore [name-defined]
        operands: typing.List[AnyPDFType] = [],
        event_listeners: typing.List["EventListener"] = [],  # type: ignore [name-defined]
    ) -> None:
        """
        Invoke the Tz operator
        """
        assert isinstance(operands[0], Decimal), "Operand 0 of Tz must be a Decimal"
        canvas = canvas_stream_processor.get_canvas()
        canvas.graphics_state.horizontal_scaling = operands[0]
