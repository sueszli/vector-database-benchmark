"""
This module contains all classes needed to apply redaction on a Page in a PDF Document
"""
import typing
from decimal import Decimal
from borb.io.read.types import AnyPDFType
from borb.io.read.types import HexadecimalString
from borb.io.read.types import List
from borb.io.read.types import Name
from borb.io.read.types import String
from borb.pdf.canvas.canvas_stream_processor import CanvasStreamProcessor
from borb.pdf.canvas.event.chunk_of_text_render_event import ChunkOfTextRenderEvent
from borb.pdf.canvas.geometry.rectangle import Rectangle
from borb.pdf.canvas.operator.canvas_operator import CanvasOperator

class CopyCommandOperator(CanvasOperator):
    """
    This CanvasOperator copies an existing operator and writes its bytes to the content stream of the canvas
    """

    def __init__(self, operator_to_copy: CanvasOperator):
        if False:
            return 10
        super().__init__('', 0)
        self._operator_to_copy = operator_to_copy

    def get_number_of_operands(self) -> int:
        if False:
            print('Hello World!')
        '\n        Return the number of operands for this CanvasOperator\n        '
        return self._operator_to_copy.get_number_of_operands()

    def get_text(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Return the str that invokes this CanvasOperator\n        '
        return self._operator_to_copy.get_text()

    def invoke(self, canvas_stream_processor: 'CanvasStreamProcessor', operands: typing.List[AnyPDFType]=[], event_listeners: typing.List['EventListener']=[]) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Invokes this CanvasOperator\n        '
        self._operator_to_copy.invoke(canvas_stream_processor, operands)
        canvas = canvas_stream_processor.get_canvas()
        op_str: typing.List[str] = []
        for op in operands:
            if isinstance(op, Decimal):
                op_str.append(str(op))
                continue
            if isinstance(op, HexadecimalString):
                op_str.append('<' + str(op) + '>')
                continue
            if isinstance(op, String):
                op_str.append('(' + str(op) + ')')
                continue
            if isinstance(op, Name):
                op_str.append('/' + str(op))
                continue
        assert isinstance(canvas_stream_processor, RedactedCanvasStreamProcessor)
        canvas_stream_processor.append_to_redacted_content(('\n' + ''.join([s + ' ' for s in op_str]) + self.get_text()).encode('latin1'))

class ShowTextMod(CanvasOperator):
    """
    Show a text string.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__('Tj', 1)

    def _show_text_unmodified(self, canvas_stream_processor: 'CanvasStreamProcessor', s: String) -> None:
        if False:
            return 10
        assert isinstance(canvas_stream_processor, RedactedCanvasStreamProcessor)
        if isinstance(s, HexadecimalString):
            canvas_stream_processor.append_to_redacted_content(('\n<' + str(s) + '> Tj').encode('latin1'))
            return
        if isinstance(s, String):
            canvas_stream_processor.append_to_redacted_content(('\n(' + str(s) + ') Tj').encode('latin1'))

    def _write_chunk_of_text(self, canvas_stream_processor: 'CanvasStreamProcessor', s: str, f: 'Font'):
        if False:
            i = 10
            return i + 15
        from borb.pdf.canvas.layout.text.chunk_of_text import ChunkOfText
        assert isinstance(canvas_stream_processor, RedactedCanvasStreamProcessor)
        canvas_stream_processor.append_to_redacted_content(b'\n')
        canvas_stream_processor.append_to_redacted_content(ChunkOfText(s, f)._write_text_bytes().encode('latin1'))

    def invoke(self, canvas_stream_processor: 'CanvasStreamProcessor', operands: typing.List[AnyPDFType]=[], event_listeners: typing.List['EventListener']=[]) -> None:
        if False:
            return 10
        '\n        Invokes this CanvasOperator\n        '
        assert isinstance(operands[0], String)
        assert isinstance(canvas_stream_processor, RedactedCanvasStreamProcessor)
        canvas = canvas_stream_processor.get_canvas()
        assert canvas.graphics_state.font is not None
        font_name: typing.Optional[Name] = None
        if isinstance(canvas.graphics_state.font, Name):
            font_name = canvas.graphics_state.font
            canvas.graphics_state.font = canvas_stream_processor.get_resource('Font', str(canvas.graphics_state.font))
        bounding_box: typing.Optional[Rectangle] = ChunkOfTextRenderEvent(canvas.graphics_state, operands[0]).get_previous_layout_box()
        assert bounding_box is not None
        jump_from_redacted: bool = False
        for evt in ChunkOfTextRenderEvent(canvas.graphics_state, operands[0]).split_on_glyphs():
            letter_should_be_redacted: bool = any([x.intersects(evt.get_previous_layout_box()) for x in canvas_stream_processor._redacted_rectangles])
            graphics_state = canvas_stream_processor.get_canvas().graphics_state
            event_bounding_box: typing.Optional[Rectangle] = evt.get_previous_layout_box()
            assert event_bounding_box is not None
            w: Decimal = event_bounding_box.get_width()
            if letter_should_be_redacted:
                graphics_state.text_matrix[2][0] += w
                jump_from_redacted = True
            else:
                if jump_from_redacted:
                    canvas_stream_processor._redacted_content += '\n%f %f %f %f %f %f Tm' % (graphics_state.text_matrix[0][0], graphics_state.text_matrix[0][1], graphics_state.text_matrix[1][0], graphics_state.text_matrix[1][1], graphics_state.text_matrix[2][0], graphics_state.text_matrix[2][1])
                    jump_from_redacted = False
                self._write_chunk_of_text(canvas_stream_processor, evt.get_text(), evt.get_font())
                graphics_state.text_matrix[2][0] += w
        if font_name is not None:
            canvas.graphics_state.font = font_name

class ShowTextWithGlyphPositioningMod(CanvasOperator):
    """
    This operator represents a modified version of the TJ operator
    Instead of always rendering the text, it takes into account the location
    at which the text is to be rendered, if the text falls in one of the redacted areas
    it will not render the text.
    """

    def __init__(self):
        if False:
            return 10
        super().__init__('TJ', 1)

    def _write_chunk_of_text(self, canvas_stream_processor: 'CanvasStreamProcessor', s: str, f: 'Font'):
        if False:
            return 10
        from borb.pdf.canvas.layout.text.chunk_of_text import ChunkOfText
        assert isinstance(canvas_stream_processor, RedactedCanvasStreamProcessor)
        canvas_stream_processor.append_to_redacted_content(b'\n')
        canvas_stream_processor.append_to_redacted_content(ChunkOfText(s, f)._write_text_bytes().encode('latin1'))

    def invoke(self, canvas_stream_processor: 'CanvasStreamProcessor', operands: typing.List[AnyPDFType]=[], event_listeners: typing.List['EventListener']=[]) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Invoke the TJ operator\n        '
        canvas = canvas_stream_processor.get_canvas()
        assert canvas.graphics_state.font is not None
        font_name: typing.Optional[Name] = None
        if isinstance(canvas.graphics_state.font, Name):
            font_name = canvas.graphics_state.font
            canvas.graphics_state.font = canvas_stream_processor.get_resource('Font', str(canvas.graphics_state.font))
        assert isinstance(operands[0], List)
        for i in range(0, len(operands[0])):
            obj = operands[0][i]
            if isinstance(obj, String):
                assert isinstance(obj, String)
                jump_from_redacted: bool = False
                for evt in ChunkOfTextRenderEvent(canvas.graphics_state, obj).split_on_glyphs():
                    letter_should_be_redacted: bool = any([x.intersects(evt.get_previous_layout_box()) for x in canvas_stream_processor._redacted_rectangles])
                    graphics_state = canvas_stream_processor.get_canvas().graphics_state
                    event_bounding_box: typing.Optional[Rectangle] = evt.get_previous_layout_box()
                    assert event_bounding_box is not None
                    w: Decimal = event_bounding_box.get_width()
                    if letter_should_be_redacted:
                        graphics_state.text_matrix[2][0] += w
                        jump_from_redacted = True
                    else:
                        if jump_from_redacted:
                            canvas_stream_processor._redacted_content += '\n%f %f %f %f %f %f Tm' % (graphics_state.text_matrix[0][0], graphics_state.text_matrix[0][1], graphics_state.text_matrix[1][0], graphics_state.text_matrix[1][1], graphics_state.text_matrix[2][0], graphics_state.text_matrix[2][1])
                            jump_from_redacted = False
                        self._write_chunk_of_text(canvas_stream_processor, evt.get_text(), evt.get_font())
                        graphics_state.text_matrix[2][0] += w
            if isinstance(obj, Decimal):
                assert isinstance(obj, Decimal)
                gs = canvas.graphics_state
                adjust_unscaled = obj
                adjust_scaled = -adjust_unscaled * Decimal(0.001) * gs.font_size * (gs.horizontal_scaling / 100)
                gs.text_matrix[2][0] -= adjust_scaled
                assert isinstance(canvas_stream_processor, RedactedCanvasStreamProcessor)
                canvas_stream_processor.append_to_redacted_content(b'\n%f %f %f %f %f %f Tm' % (gs.text_matrix[0][0], gs.text_matrix[0][1], gs.text_matrix[1][0], gs.text_matrix[1][1], gs.text_matrix[2][0], gs.text_matrix[2][1]))
        if font_name is not None:
            canvas.graphics_state.font = font_name

class RedactedCanvasStreamProcessor(CanvasStreamProcessor):
    """
    In computer science and visualization, a canvas is a container that holds various drawing elements
    (lines, shapes, text, frames containing other elements, etc.).
    It takes its name from the canvas used in visual arts.
    This implementation of Canvas automatically handles redaction (removal of content).
    """

    def __init__(self, page: 'Page', canvas: 'Canvas', redacted_rectangles: typing.List[Rectangle]):
        if False:
            while True:
                i = 10
        super(RedactedCanvasStreamProcessor, self).__init__(page, canvas, [])
        self._redacted_content: str = ''
        self._redacted_rectangles = redacted_rectangles
        for (name, operator) in self._canvas_operators.items():
            self._canvas_operators[name] = CopyCommandOperator(self._canvas_operators[name])
        self._canvas_operators['Tj'] = ShowTextMod()
        self._canvas_operators['TJ'] = ShowTextWithGlyphPositioningMod()

    def get_redacted_content(self) -> bytes:
        if False:
            return 10
        '\n        This function returns the redacted content of this implementation of CanvasStreamProcessor\n        '
        return self._redacted_content.encode('latin1')

    def set_redacted_content(self, bts: bytes) -> 'RedactedCanvasStreamProcessor':
        if False:
            for i in range(10):
                print('nop')
        '\n        This function sets the (redacted) content of this RedactedCanvasStreamProcessor\n        :param bts:     the content to be set\n        :return:        self\n        '
        self._redacted_content = bts.decode('latin1')
        return self

    def append_to_redacted_content(self, bts: bytes) -> 'RedactedCanvasStreamProcessor':
        if False:
            print('Hello World!')
        '\n        This function appends the given bytes to the (redacted) content of this RedactedCanvasStreamProcessor\n        :param bts:     the bytes to append\n        :return:        self\n        '
        self._redacted_content += bts.decode('latin1')
        return self