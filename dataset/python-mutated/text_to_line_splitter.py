"""
This class is responsible for splitting text (to be fit into a Paragraph)
"""
import re
import typing
from decimal import Decimal
from borb.pdf.canvas.font.font import Font
from borb.pdf.canvas.font.glyph_line import GlyphLine
from borb.pdf.canvas.geometry.rectangle import Rectangle
from borb.pdf.canvas.layout.hyphenation.hyphenation import Hyphenation

class TextToLineSplitter:
    """
    This class is responsible for splitting text (to be fit into a Paragraph)
    """
    HYPHENATION_CHARACTER: str = '-'

    @staticmethod
    def text_to_lines(bounding_box: Rectangle, font: Font, font_size: Decimal, text: str, hyphenation: typing.Optional[Hyphenation]=None, respect_newlines: bool=False, respect_spaces: bool=False) -> typing.List[str]:
        if False:
            print('Hello World!')
        '\n        This function splits a large str into smaller parts for layout.\n        :param bounding_box:        the bounding box in which the str(s) must fit\n        :param font:                the Font in which to render the str(s)\n        :param font_size:           the font_size in which to render the str(s)\n        :param text:                the text to split\n        :param hyphenation:         a Hyphenation object, or None (default None)\n        :param respect_newlines:    whether to respect newline characters in the input (default False)\n        :param respect_spaces:      whether to respect spaces in the input (default False)\n        :return:\n\n        '
        if text == '':
            return ['']
        if text == ' ':
            return [' '] if respect_spaces else ['']
        if '\n' in text:
            if respect_newlines:
                out: typing.List[str] = []
                for partial_text in text.split('\n'):
                    out.extend(TextToLineSplitter.text_to_lines(bounding_box=bounding_box, font=font, font_size=font_size, text=partial_text, respect_spaces=respect_spaces, respect_newlines=False))
                return out
            else:
                text = re.sub('\n+', ' ', text)
        if not respect_spaces:
            text = re.sub('[ \t]+', ' ', text)
            text = text.strip()
        out: typing.List[typing.List[str]] = []
        chars_per_line_estimate: int = max(int(bounding_box.get_width() / (Decimal(0.5) * font_size)), 1)
        tokens: typing.List[str] = []
        for c in text:
            if c.isspace():
                tokens.append(c)
                tokens.append('')
            elif len(tokens) == 0:
                tokens.append(c)
            else:
                tokens[-1] += c
        while len(tokens) > 0:
            tokens_in_line: typing.List[str] = []
            while len(tokens) and sum([len(x) for x in tokens_in_line]) < chars_per_line_estimate:
                if not respect_spaces and tokens[0].isspace() and (len(tokens_in_line) == 0):
                    tokens.pop(0)
                    continue
                tokens_in_line.append(tokens[0])
                tokens.pop(0)
            line_width: Decimal = GlyphLine.from_str(''.join(tokens_in_line), font, font_size).get_width_in_text_space()
            free_line_width: Decimal = Decimal(round(bounding_box.width - line_width, 2))
            if free_line_width == 0:
                out.append(tokens_in_line)
                continue
            if free_line_width > 0:
                while free_line_width > 0 and len(tokens) > 0:
                    future_tokens_in_line: typing.List[str] = tokens_in_line + [tokens[0]]
                    future_line_width = GlyphLine.from_str(''.join(future_tokens_in_line), font, font_size).get_width_in_text_space()
                    future_free_line_width = Decimal(round(bounding_box.width - future_line_width, 2))
                    if future_free_line_width >= 0:
                        free_line_width = future_line_width
                        tokens_in_line = future_tokens_in_line
                        tokens.pop(0)
                    else:
                        break
                if hyphenation is not None and len(tokens) > 0:
                    token_parts: typing.List[str] = hyphenation.hyphenate(tokens[0]).split(chr(173))
                    max_hyphenation_index: typing.Optional[int] = None
                    for i in range(1, len(token_parts) + 1):
                        future_tokens_in_line: typing.List[str] = tokens_in_line + token_parts[:i] + [TextToLineSplitter.HYPHENATION_CHARACTER]
                        future_line_width = GlyphLine.from_str(''.join(future_tokens_in_line), font, font_size).get_width_in_text_space()
                        future_free_line_width = Decimal(round(bounding_box.width - future_line_width, 2))
                        if future_free_line_width >= 0:
                            max_hyphenation_index = i
                    if max_hyphenation_index is not None:
                        tokens_in_line += hyphenation.hyphenate(tokens[0]).split(chr(173))[:max_hyphenation_index] + [TextToLineSplitter.HYPHENATION_CHARACTER]
                        tokens = hyphenation.hyphenate(tokens[0]).split(chr(173))[max_hyphenation_index:] + tokens[1:]
                out.append(tokens_in_line)
                chars_per_line_estimate = max(sum([sum([len(t) for t in l]) for l in out]) // len(out), 1)
                continue
            if free_line_width < 0:
                while free_line_width < 0:
                    tokens.insert(0, tokens_in_line[-1])
                    tokens_in_line.pop(-1)
                    line_width = GlyphLine.from_str(''.join(tokens_in_line), font, font_size).get_width_in_text_space()
                    free_line_width = Decimal(round(bounding_box.width - line_width, 2))
                if hyphenation is not None and len(tokens) > 0:
                    token_parts: typing.List[str] = hyphenation.hyphenate(tokens[0]).split(chr(173))
                    max_hyphenation_index: typing.Optional[int] = None
                    for i in range(1, len(token_parts) + 1):
                        future_tokens_in_line: typing.List[str] = tokens_in_line + token_parts[:i] + [TextToLineSplitter.HYPHENATION_CHARACTER]
                        future_line_width = GlyphLine.from_str(''.join(future_tokens_in_line), font, font_size).get_width_in_text_space()
                        future_free_line_width = Decimal(round(bounding_box.width - future_line_width, 2))
                        if future_free_line_width >= 0:
                            max_hyphenation_index = i
                    if max_hyphenation_index is not None:
                        tokens_in_line += hyphenation.hyphenate(tokens[0]).split(chr(173))[:max_hyphenation_index] + [TextToLineSplitter.HYPHENATION_CHARACTER]
                        tokens = hyphenation.hyphenate(tokens[0]).split(chr(173))[max_hyphenation_index:] + tokens[1:]
                if len(tokens_in_line) == 0:
                    assert False, f"Text '{text}' can not be split to inside the given bounds ({bounding_box.width}, {bounding_box.height})"
                out.append(tokens_in_line)
                chars_per_line_estimate = max(sum([sum([len(t) for t in l]) for l in out]) // len(out), 1)
                continue
        if not respect_spaces:
            for i in range(0, len(out)):
                while len(out) > 0 and out[i][-1] == ' ':
                    out[i] = out[i][:-1]
        if not respect_newlines:
            while len(out) > 0 and len(out[-1]) == 1 and (out[-1][-1] == ''):
                out.pop(-1)
        return [''.join(l) for l in out]