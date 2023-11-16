"""
Show one or more text strings, allowing individual glyph positioning. Each
element of array shall be either a string or a number.
"""
import typing
from borb.io.read.types import AnyPDFType
from borb.io.read.types import Decimal as bDecimal
from borb.io.read.types import HexadecimalString
from borb.io.read.types import Name
from borb.io.read.types import String
from borb.io.write.font.copy_command_operator import CopyCommandOperator
from borb.pdf.canvas.font.font import Font
from borb.pdf.canvas.operator.text.show_text_with_glyph_positioning import ShowTextWithGlyphPositioning

class SubSetShowTextWithGlyphPositioning(CopyCommandOperator):
    """
    Show one or more text strings, allowing individual glyph positioning. Each
    element of array shall be either a string or a number. If the element is a
    string, this operator shall show the string. If it is a number, the operator
    shall adjust the text position by that amount; that is, it shall translate the
    text matrix, T . The number shall be expressed in thousandths of a unit
    mof text space (see 9.4.4, "Text Space Details"). This amount shall be
    subtracted from the current horizontal or vertical coordinate, depending
    on the writing mode. In the default coordinate system, a positive
    adjustment has the effect of moving the next glyph painted either to the
    left or down by the given amount. Figure 46 shows an example of the
    effect of passing offsets to TJ.
    """

    def __init__(self, old_fonts: typing.List[Font], new_fonts: typing.List[Font], s: bytearray):
        if False:
            print('Hello World!')
        super(SubSetShowTextWithGlyphPositioning, self).__init__(ShowTextWithGlyphPositioning(), s)
        self._old_fonts: typing.List[Font] = old_fonts
        self._new_fonts: typing.List[Font] = new_fonts
        self._s: bytearray = s

    def _to_hex(self, i: int) -> str:
        if False:
            return 10
        s: str = hex(int(i))[2:]
        while len(s) < 2:
            s = '0' + s
        return s

    def invoke(self, canvas_stream_processor: 'CanvasStreamProcessor', operands: typing.List[AnyPDFType]=[], event_listeners: typing.List['EventListener']=[]) -> None:
        if False:
            while True:
                i = 10
        '\n        Invoke the TJ operator\n        '
        assert isinstance(operands[0], typing.List), 'Operand 0 of TJ must be a List'
        canvas = canvas_stream_processor.get_canvas()
        assert canvas.graphics_state.font is not None
        font_name: typing.Optional[Name] = None
        if isinstance(canvas.graphics_state.font, Name):
            font_name = canvas.graphics_state.font
            canvas.graphics_state.font = canvas_stream_processor.get_resource('Font', canvas.graphics_state.font)
        if canvas.graphics_state.font not in self._old_fonts:
            return super(SubSetShowTextWithGlyphPositioning, self).invoke(canvas_stream_processor, operands, event_listeners)
        old_font: Font = canvas.graphics_state.font
        new_font: Font = self._new_fonts[self._old_fonts.index(old_font)]
        operands_out: typing.List[AnyPDFType] = []
        for i in range(0, len(operands[0])):
            obj = operands[0][i]
            if isinstance(obj, bDecimal):
                operands_out.append(obj)
                continue
            if isinstance(obj, String):
                str_in_prev_font: typing.Optional[str] = old_font.character_identifier_to_unicode(int(str(obj), 16))
                assert str_in_prev_font is not None
                char_id_in_new_font: typing.Optional[int] = new_font.unicode_to_character_identifier(str_in_prev_font)
                assert char_id_in_new_font is not None
                operands_out.append(HexadecimalString(self._to_hex(char_id_in_new_font)))
        return super(SubSetShowTextWithGlyphPositioning, self).invoke(canvas_stream_processor, [operands_out], event_listeners)