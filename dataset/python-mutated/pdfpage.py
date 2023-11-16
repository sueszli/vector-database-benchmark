import itertools
import logging
from typing import BinaryIO, Container, Dict, Iterator, List, Optional, Tuple
from pdfminer.utils import Rect
from . import settings
from .pdfdocument import PDFDocument, PDFTextExtractionNotAllowed, PDFNoPageLabels
from .pdfparser import PDFParser
from .pdftypes import PDFObjectNotFound
from .pdftypes import dict_value
from .pdftypes import int_value
from .pdftypes import list_value
from .pdftypes import resolve1
from .psparser import LIT
log = logging.getLogger(__name__)
LITERAL_PAGE = LIT('Page')
LITERAL_PAGES = LIT('Pages')

class PDFPage:
    """An object that holds the information about a page.

    A PDFPage object is merely a convenience class that has a set
    of keys and values, which describe the properties of a page
    and point to its contents.

    Attributes:
      doc: a PDFDocument object.
      pageid: any Python object that can uniquely identify the page.
      attrs: a dictionary of page attributes.
      contents: a list of PDFStream objects that represents the page content.
      lastmod: the last modified time of the page.
      resources: a dictionary of resources used by the page.
      mediabox: the physical size of the page.
      cropbox: the crop rectangle of the page.
      rotate: the page rotation (in degree).
      annots: the page annotations.
      beads: a chain that represents natural reading order.
      label: the page's label (typically, the logical page number).
    """

    def __init__(self, doc: PDFDocument, pageid: object, attrs: object, label: Optional[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialize a page object.\n\n        doc: a PDFDocument object.\n        pageid: any Python object that can uniquely identify the page.\n        attrs: a dictionary of page attributes.\n        label: page label string.\n        '
        self.doc = doc
        self.pageid = pageid
        self.attrs = dict_value(attrs)
        self.label = label
        self.lastmod = resolve1(self.attrs.get('LastModified'))
        self.resources: Dict[object, object] = resolve1(self.attrs.get('Resources', dict()))
        self.mediabox: Rect = resolve1(self.attrs['MediaBox'])
        if 'CropBox' in self.attrs:
            self.cropbox: Rect = resolve1(self.attrs['CropBox'])
        else:
            self.cropbox = self.mediabox
        self.rotate = (int_value(self.attrs.get('Rotate', 0)) + 360) % 360
        self.annots = self.attrs.get('Annots')
        self.beads = self.attrs.get('B')
        if 'Contents' in self.attrs:
            contents = resolve1(self.attrs['Contents'])
        else:
            contents = []
        if not isinstance(contents, list):
            contents = [contents]
        self.contents: List[object] = contents

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return '<PDFPage: Resources={!r}, MediaBox={!r}>'.format(self.resources, self.mediabox)
    INHERITABLE_ATTRS = {'Resources', 'MediaBox', 'CropBox', 'Rotate'}

    @classmethod
    def create_pages(cls, document: PDFDocument) -> Iterator['PDFPage']:
        if False:
            while True:
                i = 10

        def search(obj: object, parent: Dict[str, object]) -> Iterator[Tuple[int, Dict[object, Dict[object, object]]]]:
            if False:
                print('Hello World!')
            if isinstance(obj, int):
                objid = obj
                tree = dict_value(document.getobj(objid)).copy()
            else:
                objid = obj.objid
                tree = dict_value(obj).copy()
            for (k, v) in parent.items():
                if k in cls.INHERITABLE_ATTRS and k not in tree:
                    tree[k] = v
            tree_type = tree.get('Type')
            if tree_type is None and (not settings.STRICT):
                tree_type = tree.get('type')
            if tree_type is LITERAL_PAGES and 'Kids' in tree:
                log.debug('Pages: Kids=%r', tree['Kids'])
                for c in list_value(tree['Kids']):
                    yield from search(c, tree)
            elif tree_type is LITERAL_PAGE:
                log.debug('Page: %r', tree)
                yield (objid, tree)
        try:
            page_labels: Iterator[Optional[str]] = document.get_page_labels()
        except PDFNoPageLabels:
            page_labels = itertools.repeat(None)
        pages = False
        if 'Pages' in document.catalog:
            objects = search(document.catalog['Pages'], document.catalog)
            for (objid, tree) in objects:
                yield cls(document, objid, tree, next(page_labels))
                pages = True
        if not pages:
            for xref in document.xrefs:
                for objid in xref.get_objids():
                    try:
                        obj = document.getobj(objid)
                        if isinstance(obj, dict) and obj.get('Type') is LITERAL_PAGE:
                            yield cls(document, objid, obj, next(page_labels))
                    except PDFObjectNotFound:
                        pass
        return

    @classmethod
    def get_pages(cls, fp: BinaryIO, pagenos: Optional[Container[int]]=None, maxpages: int=0, password: str='', caching: bool=True, check_extractable: bool=False) -> Iterator['PDFPage']:
        if False:
            print('Hello World!')
        parser = PDFParser(fp)
        doc = PDFDocument(parser, password=password, caching=caching)
        if not doc.is_extractable:
            if check_extractable:
                error_msg = 'Text extraction is not allowed: %r' % fp
                raise PDFTextExtractionNotAllowed(error_msg)
            else:
                warning_msg = 'The PDF %r contains a metadata field indicating that it should not allow text extraction. Ignoring this field and proceeding. Use the check_extractable if you want to raise an error in this case' % fp
                log.warning(warning_msg)
        for (pageno, page) in enumerate(cls.create_pages(doc)):
            if pagenos and pageno not in pagenos:
                continue
            yield page
            if maxpages and maxpages <= pageno + 1:
                break
        return