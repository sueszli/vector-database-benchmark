#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Set the line width in the graphics state (see 8.4.3.2, "Line Width").
"""
import typing
from decimal import Decimal

from borb.io.read.types import AnyPDFType
from borb.pdf.canvas.operator.canvas_operator import CanvasOperator


class SetLineWidth(CanvasOperator):
    """
    Set the line width in the graphics state (see 8.4.3.2, "Line Width").
    """

    def __init__(self):
        super().__init__("w", 1)

    def invoke(
        self,
        canvas_stream_processor: "CanvasStreamProcessor",  # type: ignore [name-defined]
        operands: typing.List[AnyPDFType] = [],
        event_listeners: typing.List["EventListener"] = [],  # type: ignore [name-defined]
    ) -> None:
        """
        Invoke the w operator
        """
        assert isinstance(operands[0], Decimal), "Operand 0 of w must be a Decimal"
        canvas = canvas_stream_processor.get_canvas()
        canvas.graphics_state.line_width = operands[0]
