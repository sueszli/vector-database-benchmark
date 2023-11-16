import unittest
from borb.io.read.types import Boolean
from borb.io.read.types import CanvasOperatorName
from borb.io.read.types import Decimal
from borb.io.read.types import Dictionary
from borb.io.read.types import Function
from borb.io.read.types import HexadecimalString
from borb.io.read.types import List
from borb.io.read.types import Name
from borb.io.read.types import Reference
from borb.io.read.types import Stream
from borb.io.read.types import String
from borb.pdf import Document
from borb.pdf import HexColor
from borb.pdf import Page
from borb.pdf.canvas.canvas import Canvas
from borb.pdf.canvas.font.simple_font.font_type_1 import StandardType1Font
from borb.pdf.canvas.geometry.rectangle import Rectangle
from borb.pdf.canvas.layout.annotation.annotation import Annotation
from borb.pdf.canvas.layout.annotation.link_annotation import DestinationType
from borb.pdf.canvas.layout.annotation.link_annotation import LinkAnnotation
from borb.pdf.canvas.layout.annotation.square_annotation import SquareAnnotation

class TestHashBorbTypes(unittest.TestCase):
    """
    This test checks whether borb objects can be hashed.
    """

    def test_hash_bool(self):
        if False:
            return 10
        hash(Boolean(True))
        hash(Boolean(False))

    def test_hash_canvas(self):
        if False:
            print('Hello World!')
        hash(Canvas())

    def test_hash_canvas_operator_name(self):
        if False:
            print('Hello World!')
        hash(CanvasOperatorName('Tj'))

    def test_hash_decimal(self):
        if False:
            return 10
        hash(Decimal(0.5))

    def test_hash_dict(self):
        if False:
            i = 10
            return i + 15
        obj0 = Dictionary()
        obj0[Name('Root')] = Reference(object_number=10)
        obj0[Name('Marked')] = Boolean(True)
        hash(obj0)

    def test_hash_document(self):
        if False:
            i = 10
            return i + 15
        hash(Document())

    def test_hash_document_information(self):
        if False:
            return 10
        d0: Document = Document()
        hash(d0.get_document_info())

    def test_hash_font(self):
        if False:
            for i in range(10):
                print('nop')
        hash(StandardType1Font('Helvetica'))

    def test_hash_function(self):
        if False:
            print('Hello World!')
        hash(Function())

    def test_hash_hexadecimal_string(self):
        if False:
            print('Hello World!')
        hash(HexadecimalString('00FF00'))

    def test_hash_link_annotation(self):
        if False:
            i = 10
            return i + 15
        a0: Annotation = LinkAnnotation(bounding_box=Rectangle(Decimal(0), Decimal(0), Decimal(100), Decimal(100)), page=Decimal(0), destination_type=DestinationType.FIT)
        hash(a0)

    def test_hash_list(self):
        if False:
            print('Hello World!')
        obj0 = List()
        obj0.append(Name('Red'))
        obj0.append(Decimal(0.5))
        hash(obj0)

    def test_hash_name(self):
        if False:
            print('Hello World!')
        hash(Name('Red'))

    def test_hash_page(self):
        if False:
            print('Hello World!')
        hash(Page())

    def test_hash_page_information(self):
        if False:
            i = 10
            return i + 15
        d0: Document = Document()
        p0: Page = Page()
        d0.add_page(Page())
        hash(p0.get_page_info())

    def test_hash_reference(self):
        if False:
            return 10
        hash(Reference())

    def test_hash_square_annotation(self):
        if False:
            i = 10
            return i + 15
        a0: Annotation = SquareAnnotation(bounding_box=Rectangle(Decimal(0), Decimal(0), Decimal(100), Decimal(100)), fill_color=HexColor('ff0000'), stroke_color=HexColor('ff0000'))
        hash(a0)

    def test_hash_stream(self):
        if False:
            return 10
        hash(Stream())

    def test_hash_string(self):
        if False:
            return 10
        hash(String('Lorem Ipsum'))