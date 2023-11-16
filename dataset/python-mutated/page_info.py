"""
    This class represents the meta-information belonging to a single page in a PDF document
"""
import typing
from decimal import Decimal
from borb.io.read.types import Dictionary
from borb.pdf.page.page_size import PageSize

class PageInfo(Dictionary):
    """
    This class represents the meta-information belonging to a single page in a PDF document
    """

    def __init__(self, page: 'Page'):
        if False:
            print('Hello World!')
        super(PageInfo, self).__init__()
        self._page = page

    def get_height(self) -> typing.Optional[Decimal]:
        if False:
            while True:
                i = 10
        '\n        Return the height of the MediaBox. This is a rectangle (see 7.9.5, "Rectangles"),\n        expressed in default user space units, that shall define the\n        boundaries of the physical medium on which the page shall be\n        displayed or printed (see 14.11.2, "Page Boundaries").\n        '
        return self._page['MediaBox'][3]

    def get_page_number(self) -> typing.Optional[Decimal]:
        if False:
            for i in range(10):
                print('nop')
        '\n        This function returns the page number\n        '
        kids = self._page.get_parent().get_parent().get('Kids')
        l = int(self._page.get_parent().get_parent().get('Count'))
        for i in range(0, l):
            if kids[i] == self._page:
                return Decimal(i)
        return None

    def get_size(self) -> typing.Tuple[Decimal, Decimal]:
        if False:
            while True:
                i = 10
        '\n        Return the (width, height) of the MediaBox. This is a rectangle (see 7.9.5, "Rectangles"),\n        expressed in default user space units, that shall define the\n        boundaries of the physical medium on which the page shall be\n        displayed or printed (see 14.11.2, "Page Boundaries").\n        '
        return (self.get_width() or Decimal(0), self.get_height() or Decimal(0))

    def get_size_as_enum(self) -> typing.Optional[PageSize]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the size of the MediaBox as a convenient, well-known,\n        well-defined property (e.g. A4_PORTRAIT).\n        This is a rectangle (see 7.9.5, "Rectangles"),\n        expressed in default user space units, that shall define the\n        boundaries of the physical medium on which the page shall be\n        displayed or printed (see 14.11.2, "Page Boundaries").\n        '
        w: typing.Optional[Decimal] = self.get_width()
        h: typing.Optional[Decimal] = self.get_height()
        if w is None or h is None:
            return None
        assert w is not None
        assert h is not None
        for p in PageSize:
            if abs(w - p.value[1]) <= 1 and abs(h - p.value[1]) <= 1:
                return p
        return None

    def get_width(self) -> typing.Optional[Decimal]:
        if False:
            while True:
                i = 10
        '\n        Return the width of the MediaBox. This is a rectangle (see 7.9.5, "Rectangles"),\n        expressed in default user space units, that shall define the\n        boundaries of the physical medium on which the page shall be\n        displayed or printed (see 14.11.2, "Page Boundaries").\n        '
        return self._page['MediaBox'][2]

    def uses_color_images(self) -> typing.Optional[bool]:
        if False:
            return 10
        '\n        The PDF operators used in content streams are grouped into categories of related operators called procedure\n        sets (see Table 314). Each procedure set corresponds to a named resource containing the implementations of\n        the operators in that procedure set. The ProcSet entry in a content stream’s resource dictionary (see 7.8.3,\n        “Resource Dictionaries”) shall hold an array consisting of the names of the procedure sets used in that content\n        stream.\n        This method returns whether this PDF uses operators from the "ImageC" procedure set.\n        '
        return 'ImageC' in self._page['Resources']['ProcSet']

    def uses_grayscale_images(self) -> typing.Optional[bool]:
        if False:
            for i in range(10):
                print('nop')
        '\n        The PDF operators used in content streams are grouped into categories of related operators called procedure\n        sets (see Table 314). Each procedure set corresponds to a named resource containing the implementations of\n        the operators in that procedure set. The ProcSet entry in a content stream’s resource dictionary (see 7.8.3,\n        “Resource Dictionaries”) shall hold an array consisting of the names of the procedure sets used in that content\n        stream.\n        This method returns whether this PDF uses operators from the "ImageB" procedure set.\n        '
        return 'ImageB' in self._page['Resources']['ProcSet']

    def uses_indexed_images(self) -> typing.Optional[bool]:
        if False:
            for i in range(10):
                print('nop')
        '\n        The PDF operators used in content streams are grouped into categories of related operators called procedure\n        sets (see Table 314). Each procedure set corresponds to a named resource containing the implementations of\n        the operators in that procedure set. The ProcSet entry in a content stream’s resource dictionary (see 7.8.3,\n        “Resource Dictionaries”) shall hold an array consisting of the names of the procedure sets used in that content\n        stream.\n        This method returns whether this PDF uses operators from the "ImageI" procedure set.\n        '
        return 'ImageI' in self._page['Resources']['ProcSet']

    def uses_painting_and_graphics_state(self) -> typing.Optional[bool]:
        if False:
            i = 10
            return i + 15
        '\n        The PDF operators used in content streams are grouped into categories of related operators called procedure\n        sets (see Table 314). Each procedure set corresponds to a named resource containing the implementations of\n        the operators in that procedure set. The ProcSet entry in a content stream’s resource dictionary (see 7.8.3,\n        “Resource Dictionaries”) shall hold an array consisting of the names of the procedure sets used in that content\n        stream.\n        This method returns whether this PDF uses operators from the "PDF" procedure set.\n        '
        return 'PDF' in self._page['Resources']['ProcSet']

    def uses_text(self) -> typing.Optional[bool]:
        if False:
            while True:
                i = 10
        '\n        The PDF operators used in content streams are grouped into categories of related operators called procedure\n        sets (see Table 314). Each procedure set corresponds to a named resource containing the implementations of\n        the operators in that procedure set. The ProcSet entry in a content stream’s resource dictionary (see 7.8.3,\n        “Resource Dictionaries”) shall hold an array consisting of the names of the procedure sets used in that content\n        stream.\n        This method returns whether this PDF uses operators from the "Text" procedure set.\n        '
        return 'Text' in self._page['Resources']['ProcSet']