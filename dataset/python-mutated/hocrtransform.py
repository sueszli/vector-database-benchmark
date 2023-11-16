"""Transform .hocr and page image to text PDF."""
from __future__ import annotations
import argparse
import os
import re
import warnings
from math import atan, cos, sin
from pathlib import Path
from typing import Any, NamedTuple
from xml.etree import ElementTree
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*load_module.*')
    from reportlab.lib.colors import black, cyan, magenta, red
    from reportlab.lib.units import inch
    from reportlab.pdfgen.canvas import Canvas
HOCR_OK_LANGS = frozenset(['afr', 'alb', 'ast', 'baq', 'bre', 'cos', 'eng', 'eus', 'fao', 'gla', 'glg', 'glv', 'ice', 'ind', 'isl', 'ita', 'ltz', 'mal', 'mga', 'nor', 'oci', 'por', 'roh', 'sco', 'sma', 'spa', 'sqi', 'swa', 'swe', 'tgl', 'wln', 'cat', 'cym', 'dan', 'deu', 'dut', 'est', 'fin', 'fra', 'hun', 'kur', 'nld', 'wel'])
Element = ElementTree.Element

class Rect(NamedTuple):
    """A rectangle for managing PDF coordinates."""
    x1: Any
    y1: Any
    x2: Any
    y2: Any

class HocrTransformError(Exception):
    """Error while applying hOCR transform."""

class HocrTransform:
    """A class for converting documents from the hOCR format.

    For details of the hOCR format, see:
    http://kba.cloud/hocr-spec/.
    """
    box_pattern = re.compile('bbox((\\s+\\d+){4})')
    baseline_pattern = re.compile('\n        baseline \\s+\n        ([\\-\\+]?\\d*\\.?\\d*) \\s+  # +/- decimal float\n        ([\\-\\+]?\\d+)            # +/- int', re.VERBOSE)
    ligatures = str.maketrans({'ﬀ': 'ff', 'ﬃ': 'f\u200cf\u200ci', 'ﬄ': 'f\u200cf\u200cl', 'ﬁ': 'fi', 'ﬂ': 'fl'})

    def __init__(self, *, hocr_filename: str | Path, dpi: float):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the HocrTransform object.'
        self.dpi = dpi
        self.hocr = ElementTree.parse(os.fspath(hocr_filename))
        matches = re.match('({.*})html', self.hocr.getroot().tag)
        self.xmlns = ''
        if matches:
            self.xmlns = matches.group(1)
        (self.width, self.height) = (None, None)
        for div in self.hocr.findall(self._child_xpath('div', 'ocr_page')):
            coords = self.element_coordinates(div)
            pt_coords = self.pt_from_pixel(coords)
            self.width = pt_coords.x2 - pt_coords.x1
            self.height = pt_coords.y2 - pt_coords.y1
            break
        if self.width is None or self.height is None:
            raise HocrTransformError('hocr file is missing page dimensions')

    def __str__(self):
        if False:
            while True:
                i = 10
        'Return the textual content of the HTML body.'
        if self.hocr is None:
            return ''
        body = self.hocr.find(self._child_xpath('body'))
        if body:
            return self._get_element_text(body)
        else:
            return ''

    def _get_element_text(self, element: Element):
        if False:
            for i in range(10):
                print('nop')
        'Return the textual content of the element and its children.'
        text = ''
        if element.text is not None:
            text += element.text
        for child in element:
            text += self._get_element_text(child)
        if element.tail is not None:
            text += element.tail
        return text

    @classmethod
    def element_coordinates(cls, element: Element) -> Rect:
        if False:
            print('Hello World!')
        'Get coordinates of the bounding box around an element.'
        out = Rect._make((0 for _ in range(4)))
        if 'title' in element.attrib:
            matches = cls.box_pattern.search(element.attrib['title'])
            if matches:
                coords = matches.group(1).split()
                out = Rect._make((int(coords[n]) for n in range(4)))
        return out

    @classmethod
    def baseline(cls, element: Element) -> tuple[float, float]:
        if False:
            for i in range(10):
                print('nop')
        "Get baseline's slope and intercept."
        if 'title' in element.attrib:
            matches = cls.baseline_pattern.search(element.attrib['title'])
            if matches:
                return (float(matches.group(1)), int(matches.group(2)))
        return (0.0, 0.0)

    def pt_from_pixel(self, pxl) -> Rect:
        if False:
            for i in range(10):
                print('nop')
        'Returns the quantity in PDF units (pt) given quantity in pixels.'
        return Rect._make((c / self.dpi * inch for c in pxl))

    def _child_xpath(self, html_tag: str, html_class: str | None=None) -> str:
        if False:
            print('Hello World!')
        xpath = f'.//{self.xmlns}{html_tag}'
        if html_class:
            xpath += f"[@class='{html_class}']"
        return xpath

    @classmethod
    def replace_unsupported_chars(cls, s: str) -> str:
        if False:
            i = 10
            return i + 15
        'Replaces characters with those available in the Helvetica typeface.'
        return s.translate(cls.ligatures)

    def to_pdf(self, *, out_filename: Path, image_filename: Path | None=None, show_bounding_boxes: bool=False, fontname: str='Helvetica', invisible_text: bool=False, interword_spaces: bool=False) -> None:
        if False:
            while True:
                i = 10
        'Creates a PDF file with an image superimposed on top of the text.\n\n        Text is positioned according to the bounding box of the lines in\n        the hOCR file.\n        The image need not be identical to the image used to create the hOCR\n        file.\n        It can have a lower resolution, different color mode, etc.\n\n        Arguments:\n            out_filename: Path of PDF to write.\n            image_filename: Image to use for this file. If omitted, the OCR text\n                is shown.\n            show_bounding_boxes: Show bounding boxes around various text regions,\n                for debugging.\n            fontname: Name of font to use.\n            invisible_text: If True, text is rendered invisible so that is\n                selectable but never drawn. If False, text is visible and may\n                be seen if the image is skipped or deleted in Acrobat.\n            interword_spaces: If True, insert spaces between words rather than\n                drawing each word without spaces. Generally this improves text\n                extraction.\n        '
        pdf = Canvas(os.fspath(out_filename), pagesize=(self.width, self.height), pageCompression=1)
        pdf.setStrokeColor(cyan)
        pdf.setFillColor(cyan)
        pdf.setLineWidth(0)
        for elem in self.hocr.iterfind(self._child_xpath('p', 'ocr_par')):
            elemtxt = self._get_element_text(elem).rstrip()
            if len(elemtxt) == 0:
                continue
            pxl_coords = self.element_coordinates(elem)
            pt = self.pt_from_pixel(pxl_coords)
            if show_bounding_boxes:
                pdf.rect(pt.x1, self.height - pt.y2, pt.x2 - pt.x1, pt.y2 - pt.y1, fill=1)
        found_lines = False
        for line in (element for element in self.hocr.iterfind(self._child_xpath('span')) if 'class' in element.attrib and element.attrib['class'] in {'ocr_header', 'ocr_line', 'ocr_textfloat'}):
            found_lines = True
            self._do_line(pdf, line, 'ocrx_word', fontname, invisible_text, interword_spaces, show_bounding_boxes)
        if not found_lines:
            root = self.hocr.find(self._child_xpath('div', 'ocr_page'))
            self._do_line(pdf, root, 'ocrx_word', fontname, invisible_text, interword_spaces, show_bounding_boxes)
        if image_filename is not None:
            pdf.drawImage(os.fspath(image_filename), 0, 0, width=self.width, height=self.height)
        pdf.showPage()
        pdf.save()

    @classmethod
    def polyval(cls, poly, x):
        if False:
            print('Hello World!')
        'Calculate the value of a polynomial at a point.'
        return x * poly[0] + poly[1]

    def _do_line(self, pdf: Canvas, line: Element | None, elemclass: str, fontname: str, invisible_text: bool, interword_spaces: bool, show_bounding_boxes: bool):
        if False:
            while True:
                i = 10
        if line is None:
            return
        pxl_line_coords = self.element_coordinates(line)
        line_box = self.pt_from_pixel(pxl_line_coords)
        line_height = line_box.y2 - line_box.y1
        (slope, pxl_intercept) = self.baseline(line)
        if abs(slope) < 0.005:
            slope = 0.0
        angle = atan(slope)
        (cos_a, sin_a) = (cos(angle), sin(angle))
        text = pdf.beginText()
        intercept = pxl_intercept / self.dpi * inch
        fontsize = (line_height - abs(intercept)) / cos_a
        text.setFont(fontname, fontsize)
        if invisible_text:
            text.setTextRenderMode(3)
        baseline_y2 = self.height - (line_box.y2 + intercept)
        if show_bounding_boxes:
            pdf.setDash()
            pdf.setStrokeColor(magenta)
            pdf.setLineWidth(0.5)
            pdf.line(line_box.x1, baseline_y2, line_box.x2, self.polyval((-slope, baseline_y2), line_box.x2 - line_box.x1))
            pdf.setDash(6, 3)
            pdf.setStrokeColor(red)
        text.setTextTransform(cos_a, -sin_a, sin_a, cos_a, line_box.x1, baseline_y2)
        pdf.setFillColor(black)
        elements = line.findall(self._child_xpath('span', elemclass))
        for elem in elements:
            elemtxt = self._get_element_text(elem).strip()
            elemtxt = self.replace_unsupported_chars(elemtxt)
            if elemtxt == '':
                continue
            pxl_coords = self.element_coordinates(elem)
            box = self.pt_from_pixel(pxl_coords)
            if interword_spaces:
                elemtxt += ' '
                box = Rect._make((box.x1, line_box.y1, box.x2 + pdf.stringWidth(' ', fontname, line_height), line_box.y2))
            box_width = box.x2 - box.x1
            font_width = pdf.stringWidth(elemtxt, fontname, fontsize)
            if show_bounding_boxes:
                pdf.rect(box.x1, self.height - line_box.y2, box_width, line_height, fill=0)
            cursor = text.getStartOfLine()
            dx = box.x1 - cursor[0]
            dy = baseline_y2 - cursor[1]
            text.moveCursor(dx, dy)
            if font_width > 0:
                text.setHorizScale(100 * box_width / font_width)
                text.textOut(elemtxt)
        pdf.drawText(text)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert hocr file to PDF')
    parser.add_argument('-b', '--boundingboxes', action='store_true', default=False, help='Show bounding boxes borders')
    parser.add_argument('-r', '--resolution', type=int, default=300, help='Resolution of the image that was OCRed')
    parser.add_argument('-i', '--image', default=None, help='Path to the image to be placed above the text')
    parser.add_argument('--interword-spaces', action='store_true', default=False, help='Add spaces between words')
    parser.add_argument('hocrfile', help='Path to the hocr file to be parsed')
    parser.add_argument('outputfile', help='Path to the PDF file to be generated')
    args = parser.parse_args()
    hocr = HocrTransform(hocr_filename=args.hocrfile, dpi=args.resolution)
    hocr.to_pdf(out_filename=args.outputfile, image_filename=args.image, show_bounding_boxes=args.boundingboxes, interword_spaces=args.interword_spaces)