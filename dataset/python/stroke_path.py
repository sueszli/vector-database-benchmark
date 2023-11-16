#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stroke the path.
"""

import typing

from borb.io.read.types import AnyPDFType
from borb.pdf.canvas.event.line_render_event import LineRenderEvent
from borb.pdf.canvas.operator.canvas_operator import CanvasOperator


class StrokePath(CanvasOperator):
    """
    Stroke the path.
    """

    def __init__(self):
        super().__init__("S", 0)

    def invoke(
        self,
        canvas_stream_processor: "CanvasStreamProcessor",  # type: ignore [name-defined]
        operands: typing.List[AnyPDFType] = [],
        event_listeners: typing.List["EventListener"] = [],  # type: ignore [name-defined]
    ) -> None:
        """
        Invoke the S operator
        """

        # get graphic state
        canvas = canvas_stream_processor.get_canvas()
        gs = canvas.graphics_state

        # notify listeners
        for el in event_listeners:
            for l in gs.path:
                # noinspection PyProtectedMember
                el._event_occurred(LineRenderEvent(gs, l))

        # clear path
        gs.path = []
