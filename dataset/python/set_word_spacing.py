#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Set the word spacing, T w , to wordSpace, which shall be a number
expressed in unscaled text space units. Word spacing shall be used by
the Tj, TJ, and ' operators. Initial value: 0.
"""
import typing
from decimal import Decimal

from borb.io.read.types import AnyPDFType
from borb.pdf.canvas.operator.canvas_operator import CanvasOperator


class SetWordSpacing(CanvasOperator):
    """
    Set the word spacing, T w , to wordSpace, which shall be a number
    expressed in unscaled text space units. Word spacing shall be used by
    the Tj, TJ, and ' operators. Initial value: 0.
    """

    def __init__(self):
        super().__init__("Tw", 1)

    def invoke(
        self,
        canvas_stream_processor: "CanvasStreamProcessor",  # type: ignore [name-defined]
        operands: typing.List[AnyPDFType] = [],
        event_listeners: typing.List["EventListener"] = [],  # type: ignore [name-defined]
    ) -> None:
        """
        Invoke the Tw operator
        """
        assert isinstance(operands[0], Decimal)
        canvas = canvas_stream_processor.get_canvas()
        canvas.graphics_state.word_spacing = operands[0]
