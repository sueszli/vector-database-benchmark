import codecs
import collections
import decimal
import enum
import hashlib
import re
import uuid
import warnings
from io import BytesIO, FileIO, IOBase
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Callable, Deque, Dict, Iterable, List, Optional, Pattern, Tuple, Type, Union, cast
from ._cmap import build_char_map_from_dict
from ._encryption import EncryptAlgorithm, Encryption
from ._page import PageObject, _VirtualList
from ._page_labels import nums_clear_range, nums_insert, nums_next
from ._reader import PdfReader
from ._utils import StrByteType, StreamType, _get_max_pdf_version_header, b_, deprecate_with_replacement, deprecation_bookmark, deprecation_with_replacement, logger_warning
from .annotations import Link
from .constants import AnnotationDictionaryAttributes as AA
from .constants import CatalogAttributes as CA
from .constants import CatalogDictionary, FieldFlag, FileSpecificationDictionaryEntries, GoToActionArguments, ImageType, InteractiveFormDictEntries, PageLabelStyle, TypFitArguments, UserAccessPermissions
from .constants import CatalogDictionary as CD
from .constants import Core as CO
from .constants import FieldDictionaryAttributes as FA
from .constants import PageAttributes as PG
from .constants import PagesAttributes as PA
from .constants import TrailerKeys as TK
from .errors import PyPdfError
from .generic import PAGE_FIT, ArrayObject, BooleanObject, ByteStringObject, ContentStream, DecodedStreamObject, Destination, DictionaryObject, Fit, FloatObject, IndirectObject, NameObject, NullObject, NumberObject, PdfObject, RectangleObject, StreamObject, TextStringObject, TreeObject, ViewerPreferences, create_string_object, hex_to_rgb
from .pagerange import PageRange, PageRangeSpec
from .types import AnnotationSubtype, BorderArrayType, FitType, LayoutType, OutlineItemType, OutlineType, PagemodeType, ZoomArgType
OPTIONAL_READ_WRITE_FIELD = FieldFlag(0)
ALL_DOCUMENT_PERMISSIONS = UserAccessPermissions(2 ** 31 - 1 - 3)

class ObjectDeletionFlag(enum.IntFlag):
    NONE = 0
    TEXT = enum.auto()
    LINKS = enum.auto()
    ATTACHMENTS = enum.auto()
    OBJECTS_3D = enum.auto()
    ALL_ANNOTATIONS = enum.auto()
    XOBJECT_IMAGES = enum.auto()
    INLINE_IMAGES = enum.auto()
    DRAWING_IMAGES = enum.auto()
    IMAGES = XOBJECT_IMAGES | INLINE_IMAGES | DRAWING_IMAGES

def _rolling_checksum(stream: BytesIO, blocksize: int=65536) -> str:
    if False:
        while True:
            i = 10
    hash = hashlib.md5()
    for block in iter(lambda : stream.read(blocksize), b''):
        hash.update(block)
    return hash.hexdigest()

class PdfWriter:
    """
    Write a PDF file out, given pages produced by another class.

    Typically data is added from a :class:`PdfReader<pypdf.PdfReader>`.
    """

    def __init__(self, fileobj: StrByteType='', clone_from: Union[None, PdfReader, StrByteType, Path]=None) -> None:
        if False:
            i = 10
            return i + 15
        self._header = b'%PDF-1.3'
        self._objects: List[PdfObject] = []
        'The indirect objects in the PDF.'
        self._idnum_hash: Dict[bytes, IndirectObject] = {}
        'Maps hash values of indirect objects to their IndirectObject instances.'
        self._id_translated: Dict[int, Dict[int, int]] = {}
        pages = DictionaryObject()
        pages.update({NameObject(PA.TYPE): NameObject('/Pages'), NameObject(PA.COUNT): NumberObject(0), NameObject(PA.KIDS): ArrayObject()})
        self._pages = self._add_object(pages)
        info = DictionaryObject()
        info.update({NameObject('/Producer'): create_string_object(codecs.BOM_UTF16_BE + 'pypdf'.encode('utf-16be'))})
        self._info = self._add_object(info)
        self._root_object = DictionaryObject()
        self._root_object.update({NameObject(PA.TYPE): NameObject(CO.CATALOG), NameObject(CO.PAGES): self._pages})
        self._root = self._add_object(self._root_object)
        if clone_from is not None:
            if not isinstance(clone_from, PdfReader):
                clone_from = PdfReader(clone_from)
            self.clone_document_from_reader(clone_from)
        self.fileobj = fileobj
        self.with_as_usage = False
        self._encryption: Optional[Encryption] = None
        self._encrypt_entry: Optional[DictionaryObject] = None

    def __enter__(self) -> 'PdfWriter':
        if False:
            while True:
                i = 10
        "Store that writer is initialized by 'with'."
        self.with_as_usage = True
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Write data to the fileobj.'
        if self.fileobj:
            self.write(self.fileobj)

    @property
    def pdf_header(self) -> bytes:
        if False:
            while True:
                i = 10
        "\n        Header of the PDF document that is written.\n\n        This should be something like ``b'%PDF-1.5'``. It is recommended to set\n        the lowest version that supports all features which are used within the\n        PDF file.\n        "
        return self._header

    @pdf_header.setter
    def pdf_header(self, new_header: bytes) -> None:
        if False:
            return 10
        self._header = new_header

    def _add_object(self, obj: PdfObject) -> IndirectObject:
        if False:
            i = 10
            return i + 15
        if hasattr(obj, 'indirect_reference') and obj.indirect_reference.pdf == self:
            return obj.indirect_reference
        if isinstance(obj, DictionaryObject) and isinstance(obj.get(PG.CONTENTS, None), (ArrayObject, DictionaryObject)):
            obj[NameObject(PG.CONTENTS)] = self._add_object(obj[PG.CONTENTS])
        self._objects.append(obj)
        obj.indirect_reference = IndirectObject(len(self._objects), 0, self)
        return obj.indirect_reference

    def get_object(self, indirect_reference: Union[None, int, IndirectObject]=None, ido: Optional[IndirectObject]=None) -> PdfObject:
        if False:
            print('Hello World!')
        if ido is not None:
            if indirect_reference is not None:
                raise ValueError("Please only set 'indirect_reference'. The 'ido' argument is deprecated.")
            else:
                indirect_reference = ido
                warnings.warn("The parameter 'ido' is depreciated and will be removed in pypdf 4.0.0.", DeprecationWarning)
        assert indirect_reference is not None
        if isinstance(indirect_reference, int):
            return self._objects[indirect_reference - 1]
        if indirect_reference.pdf != self:
            raise ValueError('pdf must be self')
        return self._objects[indirect_reference.idnum - 1]

    def getObject(self, ido: Union[int, IndirectObject]) -> PdfObject:
        if False:
            print('Hello World!')
        '\n        Use :meth:`get_object` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('getObject', 'get_object', '3.0.0')
        return self.get_object(ido)

    def _replace_object(self, indirect_reference: Union[int, IndirectObject], obj: PdfObject) -> PdfObject:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(indirect_reference, IndirectObject):
            if indirect_reference.pdf != self:
                raise ValueError('pdf must be self')
            indirect_reference = indirect_reference.idnum
        gen = self._objects[indirect_reference - 1].indirect_reference.generation
        self._objects[indirect_reference - 1] = obj
        obj.indirect_reference = IndirectObject(indirect_reference, gen, self)
        return self._objects[indirect_reference - 1]

    def _add_page(self, page: PageObject, action: Callable[[Any, IndirectObject], None], excluded_keys: Iterable[str]=()) -> PageObject:
        if False:
            for i in range(10):
                print('nop')
        assert cast(str, page[PA.TYPE]) == CO.PAGE
        page_org = page
        excluded_keys = list(excluded_keys)
        excluded_keys += [PA.PARENT, '/StructParents']
        try:
            del self._id_translated[id(page_org.indirect_reference.pdf)][page_org.indirect_reference.idnum]
        except Exception:
            pass
        page = cast('PageObject', page_org.clone(self, False, excluded_keys))
        if page_org.pdf is not None:
            other = page_org.pdf.pdf_header
            if isinstance(other, str):
                other = other.encode()
            self.pdf_header = _get_max_pdf_version_header(self.pdf_header, other)
        page[NameObject(PA.PARENT)] = self._pages
        pages = cast(DictionaryObject, self.get_object(self._pages))
        assert page.indirect_reference is not None
        action(pages[PA.KIDS], page.indirect_reference)
        page_count = cast(int, pages[PA.COUNT])
        pages[NameObject(PA.COUNT)] = NumberObject(page_count + 1)
        return page

    def set_need_appearances_writer(self, state: bool=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the "NeedAppearances" flag in the PDF writer.\n\n        The "NeedAppearances" flag indicates whether the appearance dictionary\n        for form fields should be automatically generated by the PDF viewer or\n        if the embedded appearence should be used.\n\n        Args:\n            state: The actual value of the NeedAppearances flag.\n\n        Returns:\n            None\n        '
        try:
            if CatalogDictionary.ACRO_FORM not in self._root_object:
                self._root_object[NameObject(CatalogDictionary.ACRO_FORM)] = self._add_object(DictionaryObject())
            need_appearances = NameObject(InteractiveFormDictEntries.NeedAppearances)
            cast(DictionaryObject, self._root_object[CatalogDictionary.ACRO_FORM])[need_appearances] = BooleanObject(state)
        except Exception as exc:
            logger_warning(f'set_need_appearances_writer({state}) catch : {exc}', __name__)

    @property
    def viewer_preferences(self) -> Optional[ViewerPreferences]:
        if False:
            i = 10
            return i + 15
        'Returns the existing ViewerPreferences as an overloaded dictionary.'
        o = self._root_object.get(CD.VIEWER_PREFERENCES, None)
        if o is None:
            return None
        o = o.get_object()
        if not isinstance(o, ViewerPreferences):
            o = ViewerPreferences(o)
            if hasattr(o, 'indirect_reference'):
                self._replace_object(o.indirect_reference, o)
            else:
                self._root_object[NameObject(CD.VIEWER_PREFERENCES)] = o
        return o

    def create_viewer_preferences(self) -> ViewerPreferences:
        if False:
            print('Hello World!')
        o = ViewerPreferences()
        self._root_object[NameObject(CD.VIEWER_PREFERENCES)] = self._add_object(o)
        return o

    def add_page(self, page: PageObject, excluded_keys: Iterable[str]=()) -> PageObject:
        if False:
            for i in range(10):
                print('nop')
        '\n        Add a page to this PDF file.\n\n        Recommended for advanced usage including the adequate excluded_keys.\n\n        The page is usually acquired from a :class:`PdfReader<pypdf.PdfReader>`\n        instance.\n\n        Args:\n            page: The page to add to the document. Should be\n                an instance of :class:`PageObject<pypdf._page.PageObject>`\n            excluded_keys:\n\n        Returns:\n            The added PageObject.\n        '
        return self._add_page(page, list.append, excluded_keys)

    def addPage(self, page: PageObject, excluded_keys: Iterable[str]=()) -> PageObject:
        if False:
            print('Hello World!')
        '\n        Use :meth:`add_page` instead.\n\n        .. deprecated:: 1.28.0.\n        '
        deprecation_with_replacement('addPage', 'add_page', '3.0.0')
        return self.add_page(page, excluded_keys)

    def insert_page(self, page: PageObject, index: int=0, excluded_keys: Iterable[str]=()) -> PageObject:
        if False:
            print('Hello World!')
        '\n        Insert a page in this PDF file. The page is usually acquired from a\n        :class:`PdfReader<pypdf.PdfReader>` instance.\n\n        Args:\n            page: The page to add to the document.\n            index: Position at which the page will be inserted.\n            excluded_keys:\n\n        Returns:\n            The added PageObject.\n        '
        return self._add_page(page, lambda kids, p: kids.insert(index, p))

    def insertPage(self, page: PageObject, index: int=0, excluded_keys: Iterable[str]=()) -> PageObject:
        if False:
            return 10
        '\n        Use :meth:`insert_page` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('insertPage', 'insert_page', '3.0.0')
        return self.insert_page(page, index, excluded_keys)

    def get_page(self, page_number: Optional[int]=None, pageNumber: Optional[int]=None) -> PageObject:
        if False:
            return 10
        '\n        Retrieve a page by number from this PDF file.\n\n        Args:\n            page_number: The page number to retrieve\n                (pages begin at zero)\n\n        Returns:\n            The page at the index given by *page_number*\n        '
        if pageNumber is not None:
            if page_number is not None:
                raise ValueError('Please only use the page_number parameter')
            deprecate_with_replacement('get_page(pageNumber)', 'get_page(page_number)', '4.0.0')
            page_number = pageNumber
        if page_number is None and pageNumber is None:
            raise ValueError('Please specify the page_number')
        pages = cast(Dict[str, Any], self.get_object(self._pages))
        return cast(PageObject, pages[PA.KIDS][page_number].get_object())

    def getPage(self, pageNumber: int) -> PageObject:
        if False:
            print('Hello World!')
        '\n        Use :code:`writer.pages[page_number]` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('getPage', 'writer.pages[page_number]', '3.0.0')
        return self.get_page(pageNumber)

    def _get_num_pages(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        pages = cast(Dict[str, Any], self.get_object(self._pages))
        return int(pages[NameObject('/Count')])

    def getNumPages(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Use :code:`len(writer.pages)` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('getNumPages', 'len(writer.pages)', '3.0.0')
        return self._get_num_pages()

    @property
    def pages(self) -> List[PageObject]:
        if False:
            while True:
                i = 10
        '\n        Property that emulates a list of :class:`PageObject<pypdf._page.PageObject>`.\n        this property allows to get a page or  a range of pages.\n\n        It provides also capability to remove a page/range of page from the list\n        (through del operator)\n        Note: only the page entry is removed. As the objects beneath can be used\n        somewhere else.\n        a solution to completely remove them - if they are not used somewhere -\n        is to write to a buffer/temporary and to then load it into a new PdfWriter\n        object.\n        '
        return _VirtualList(self._get_num_pages, self.get_page)

    def add_blank_page(self, width: Optional[float]=None, height: Optional[float]=None) -> PageObject:
        if False:
            print('Hello World!')
        '\n        Append a blank page to this PDF file and returns it.\n\n        If no page size is specified, use the size of the last page.\n\n        Args:\n            width: The width of the new page expressed in default user\n                space units.\n            height: The height of the new page expressed in default\n                user space units.\n\n        Returns:\n            The newly appended page\n\n        Raises:\n            PageSizeNotDefinedError: if width and height are not defined\n                and previous page does not exist.\n        '
        page = PageObject.create_blank_page(self, width, height)
        return self.add_page(page)

    def addBlankPage(self, width: Optional[float]=None, height: Optional[float]=None) -> PageObject:
        if False:
            print('Hello World!')
        '\n        Use :meth:`add_blank_page` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('addBlankPage', 'add_blank_page', '3.0.0')
        return self.add_blank_page(width, height)

    def insert_blank_page(self, width: Optional[Union[float, decimal.Decimal]]=None, height: Optional[Union[float, decimal.Decimal]]=None, index: int=0) -> PageObject:
        if False:
            print('Hello World!')
        '\n        Insert a blank page to this PDF file and returns it.\n\n        If no page size is specified, use the size of the last page.\n\n        Args:\n            width: The width of the new page expressed in default user\n                space units.\n            height: The height of the new page expressed in default\n                user space units.\n            index: Position to add the page.\n\n        Returns:\n            The newly appended page\n\n        Raises:\n            PageSizeNotDefinedError: if width and height are not defined\n                and previous page does not exist.\n        '
        if width is None or (height is None and self._get_num_pages() - 1 >= index):
            oldpage = self.pages[index]
            width = oldpage.mediabox.width
            height = oldpage.mediabox.height
        page = PageObject.create_blank_page(self, width, height)
        self.insert_page(page, index)
        return page

    def insertBlankPage(self, width: Optional[Union[float, decimal.Decimal]]=None, height: Optional[Union[float, decimal.Decimal]]=None, index: int=0) -> PageObject:
        if False:
            i = 10
            return i + 15
        '\n        Use :meth:`insertBlankPage` instead.\n\n        .. deprecated:: 1.28.0.\n        '
        deprecation_with_replacement('insertBlankPage', 'insert_blank_page', '3.0.0')
        return self.insert_blank_page(width, height, index)

    @property
    def open_destination(self) -> Union[None, Destination, TextStringObject, ByteStringObject]:
        if False:
            print('Hello World!')
        '\n        Property to access the opening destination (``/OpenAction`` entry in\n        the PDF catalog). It returns ``None`` if the entry does not exist is not\n        set.\n\n        Raises:\n            Exception: If a destination is invalid.\n        '
        if '/OpenAction' not in self._root_object:
            return None
        oa = self._root_object['/OpenAction']
        if isinstance(oa, (str, bytes)):
            return create_string_object(str(oa))
        elif isinstance(oa, ArrayObject):
            try:
                (page, typ) = oa[0:2]
                array = oa[2:]
                fit = Fit(typ, tuple(array))
                return Destination('OpenAction', page, fit)
            except Exception as exc:
                raise Exception(f'Invalid Destination {oa}: {exc}')
        else:
            return None

    @open_destination.setter
    def open_destination(self, dest: Union[None, str, Destination, PageObject]) -> None:
        if False:
            for i in range(10):
                print('nop')
        if dest is None:
            try:
                del self._root_object['/OpenAction']
            except KeyError:
                pass
        elif isinstance(dest, str):
            self._root_object[NameObject('/OpenAction')] = TextStringObject(dest)
        elif isinstance(dest, Destination):
            self._root_object[NameObject('/OpenAction')] = dest.dest_array
        elif isinstance(dest, PageObject):
            self._root_object[NameObject('/OpenAction')] = Destination('Opening', dest.indirect_reference if dest.indirect_reference is not None else NullObject(), PAGE_FIT).dest_array

    def add_js(self, javascript: str) -> None:
        if False:
            return 10
        '\n        Add Javascript which will launch upon opening this PDF.\n\n        Args:\n            javascript: Your Javascript.\n\n        >>> output.add_js("this.print({bUI:true,bSilent:false,bShrinkToFit:true});")\n        # Example: This will launch the print window when the PDF is opened.\n        '
        if '/Names' not in self._root_object:
            self._root_object[NameObject(CA.NAMES)] = DictionaryObject()
        names = cast(DictionaryObject, self._root_object[CA.NAMES])
        if '/JavaScript' not in names:
            names[NameObject('/JavaScript')] = DictionaryObject({NameObject('/Names'): ArrayObject()})
        js_list = cast(ArrayObject, cast(DictionaryObject, names['/JavaScript'])['/Names'])
        js = DictionaryObject()
        js.update({NameObject(PA.TYPE): NameObject('/Action'), NameObject('/S'): NameObject('/JavaScript'), NameObject('/JS'): TextStringObject(f'{javascript}')})
        js_list.append(create_string_object(str(uuid.uuid4())))
        js_list.append(self._add_object(js))

    def addJS(self, javascript: str) -> None:
        if False:
            print('Hello World!')
        '\n        Use :meth:`add_js` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('addJS', 'add_js', '3.0.0')
        return self.add_js(javascript)

    def add_attachment(self, filename: str, data: Union[str, bytes]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Embed a file inside the PDF.\n\n        Reference:\n        https://www.adobe.com/content/dam/Adobe/en/devnet/acrobat/pdfs/PDF32000_2008.pdf\n        Section 7.11.3\n\n        Args:\n            filename: The filename to display.\n            data: The data in the file.\n        '
        file_entry = DecodedStreamObject()
        file_entry.set_data(b_(data))
        file_entry.update({NameObject(PA.TYPE): NameObject('/EmbeddedFile')})
        ef_entry = DictionaryObject()
        ef_entry.update({NameObject('/F'): self._add_object(file_entry)})
        filespec = DictionaryObject()
        filespec.update({NameObject(PA.TYPE): NameObject('/Filespec'), NameObject(FileSpecificationDictionaryEntries.F): create_string_object(filename), NameObject(FileSpecificationDictionaryEntries.EF): ef_entry})
        if CA.NAMES not in self._root_object:
            self._root_object[NameObject(CA.NAMES)] = self._add_object(DictionaryObject())
        if '/EmbeddedFiles' not in cast(DictionaryObject, self._root_object[CA.NAMES]):
            embedded_files_names_dictionary = DictionaryObject({NameObject(CA.NAMES): ArrayObject()})
            cast(DictionaryObject, self._root_object[CA.NAMES])[NameObject('/EmbeddedFiles')] = self._add_object(embedded_files_names_dictionary)
        else:
            embedded_files_names_dictionary = cast(DictionaryObject, cast(DictionaryObject, self._root_object[CA.NAMES])['/EmbeddedFiles'])
        cast(ArrayObject, embedded_files_names_dictionary[CA.NAMES]).extend([create_string_object(filename), filespec])

    def addAttachment(self, fname: str, fdata: Union[str, bytes]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Use :meth:`add_attachment` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('addAttachment', 'add_attachment', '3.0.0')
        return self.add_attachment(fname, fdata)

    def append_pages_from_reader(self, reader: PdfReader, after_page_append: Optional[Callable[[PageObject], None]]=None) -> None:
        if False:
            print('Hello World!')
        "\n        Copy pages from reader to writer. Includes an optional callback\n        parameter which is invoked after pages are appended to the writer.\n\n        ``append`` should be prefered.\n\n        Args:\n            reader: a PdfReader object from which to copy page\n                annotations to this writer object.  The writer's annots\n                will then be updated\n            after_page_append:\n                Callback function that is invoked after each page is appended to\n                the writer. Signature includes a reference to the appended page\n                (delegates to append_pages_from_reader). The single parameter of\n                the callback is a reference to the page just appended to the\n                document.\n        "
        reader_num_pages = len(reader.pages)
        for reader_page_number in range(reader_num_pages):
            reader_page = reader.pages[reader_page_number]
            writer_page = self.add_page(reader_page)
            if callable(after_page_append):
                after_page_append(writer_page)

    def appendPagesFromReader(self, reader: PdfReader, after_page_append: Optional[Callable[[PageObject], None]]=None) -> None:
        if False:
            return 10
        '\n        Use :meth:`append_pages_from_reader` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('appendPagesFromReader', 'append_pages_from_reader', '3.0.0')
        self.append_pages_from_reader(reader, after_page_append)

    def _get_qualified_field_name(self, parent: DictionaryObject) -> Optional[str]:
        if False:
            return 10
        if '/TM' in parent:
            return cast(str, parent['/TM'])
        elif '/T' not in parent:
            return None
        elif '/Parent' in parent:
            qualified_parent = self._get_qualified_field_name(cast(DictionaryObject, parent['/Parent']))
            if qualified_parent is not None:
                return qualified_parent + '.' + cast(str, parent['/T'])
        return cast(str, parent['/T'])

    def _update_text_field(self, field: DictionaryObject) -> None:
        if False:
            for i in range(10):
                print('nop')
        _rct = cast(RectangleObject, field[AA.Rect])
        rct = RectangleObject((0, 0, _rct[2] - _rct[0], _rct[3] - _rct[1]))
        da = cast(str, field[AA.DA])
        font_properties = da.replace('\n', ' ').replace('\r', ' ').split(' ')
        font_properties = [x for x in font_properties if x != '']
        font_name = font_properties[font_properties.index('Tf') - 2]
        font_height = float(font_properties[font_properties.index('Tf') - 1])
        if font_height == 0:
            font_height = rct.height - 2
            font_properties[font_properties.index('Tf') - 1] = str(font_height)
            da = ' '.join(font_properties)
        y_offset = rct.height - 1 - font_height
        dr: Any = cast(DictionaryObject, cast(DictionaryObject, field.get('/DR', DictionaryObject())).get_object())
        dr = dr.get('/Font', DictionaryObject()).get_object()
        if font_name not in dr:
            dr = cast(Dict[Any, Any], cast(DictionaryObject, self._root_object['/AcroForm']).get('/DR', {}))
            if isinstance(dr, IndirectObject):
                dr = dr.get_object()
            dr = dr.get('/Font', DictionaryObject()).get_object()
        font_res = dr.get(font_name)
        if font_res is not None:
            font_res = cast(DictionaryObject, font_res.get_object())
            (font_subtype, _, font_encoding, font_map) = build_char_map_from_dict(200, font_res)
            try:
                del font_map[-1]
            except KeyError:
                pass
            font_full_rev: Dict[str, bytes]
            if isinstance(font_encoding, str):
                font_full_rev = {v: k.encode(font_encoding) for (k, v) in font_map.items()}
            else:
                font_full_rev = {v: bytes((k,)) for (k, v) in font_encoding.items()}
                font_encoding_rev = {v: bytes((k,)) for (k, v) in font_encoding.items()}
                for (kk, v) in font_map.items():
                    font_full_rev[v] = font_encoding_rev.get(kk, kk)
        else:
            logger_warning(f'Font dictionary for {font_name} not found.', __name__)
            font_full_rev = {}
        field_flags = field.get(FA.Ff, 0)
        if field.get(FA.FT, '/Tx') == '/Ch' and field_flags & FA.FfBits.Combo == 0:
            txt = '\n'.join(field.get(FA.Opt, {}))
            sel = field.get('/V', [])
            if not isinstance(sel, list):
                sel = [sel]
        else:
            txt = field.get('/V', '')
            sel = []
        txt = txt.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
        ap_stream = f'q\n/Tx BMC \nq\n1 1 {rct.width - 1} {rct.height - 1} re\nW\nBT\n{da}\n'.encode()
        for (line_number, line) in enumerate(txt.replace('\n', '\r').split('\r')):
            if line in sel:
                ap_stream += f'1 {y_offset - line_number * font_height * 1.4 - 1} {rct.width - 2} {font_height + 2} re\n0.5 0.5 0.5 rg s\n{field[AA.DA]}\n'.encode()
            if line_number == 0:
                ap_stream += f'2 {y_offset} Td\n'.encode()
            else:
                ap_stream += f'0 {-font_height * 1.4} Td\n'.encode()
            enc_line: List[bytes] = [font_full_rev.get(c, c.encode('utf-16-be')) for c in line]
            if any((len(c) >= 2 for c in enc_line)):
                ap_stream += b'<' + b''.join(enc_line).hex().encode() + b'> Tj\n'
            else:
                ap_stream += b'(' + b''.join(enc_line) + b') Tj\n'
        ap_stream += b'ET\nQ\nEMC\nQ\n'
        dct = DecodedStreamObject.initialize_from_dictionary({NameObject('/Type'): NameObject('/XObject'), NameObject('/Subtype'): NameObject('/Form'), NameObject('/BBox'): rct, '__streamdata__': ByteStringObject(ap_stream), '/Length': 0})
        if font_res is not None:
            dct[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject(font_name): getattr(font_res, 'indirect_reference', font_res)})})
        if AA.AP not in field:
            field[NameObject(AA.AP)] = DictionaryObject({NameObject('/N'): self._add_object(dct)})
        elif '/N' not in cast(DictionaryObject, field[AA.AP]):
            cast(DictionaryObject, field[NameObject(AA.AP)])[NameObject('/N')] = self._add_object(dct)
        else:
            n = field[AA.AP]['/N'].indirect_reference.idnum
            self._objects[n - 1] = dct
            dct.indirect_reference = IndirectObject(n, 0, self)

    def update_page_form_field_values(self, page: PageObject, fields: Dict[str, Any], flags: FieldFlag=OPTIONAL_READ_WRITE_FIELD, auto_regenerate: Optional[bool]=True) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Update the form field values for a given page from a fields dictionary.\n\n        Copy field texts and values from fields to page.\n        If the field links to a parent object, add the information to the parent.\n\n        Args:\n            page: Page reference from PDF writer where the\n                annotations and field data will be updated.\n            fields: a Python dictionary of field names (/T) and text\n                values (/V)\n            flags: An integer (0 to 7). The first bit sets ReadOnly, the\n                second bit sets Required, the third bit sets NoExport. See\n                PDF Reference Table 8.70 for details.\n            auto_regenerate: set/unset the need_appearances flag ;\n                the flag is unchanged if auto_regenerate is None\n        '
        if CatalogDictionary.ACRO_FORM not in self._root_object:
            raise PyPdfError('No /AcroForm dictionary in PdfWriter Object')
        af = cast(DictionaryObject, self._root_object[CatalogDictionary.ACRO_FORM])
        if InteractiveFormDictEntries.Fields not in af:
            raise PyPdfError('No /Fields dictionary in Pdf in PdfWriter Object')
        if isinstance(auto_regenerate, bool):
            self.set_need_appearances_writer(auto_regenerate)
        if PG.ANNOTS not in page:
            logger_warning('No fields to update on this page', __name__)
            return
        default_da = af.get(InteractiveFormDictEntries.DA, TextStringObject('/Helvetica 0 Tf 0 g'))
        for writer_annot in page[PG.ANNOTS]:
            writer_annot = cast(DictionaryObject, writer_annot.get_object())
            writer_parent_annot = writer_annot.get(PG.PARENT, DictionaryObject()).get_object()
            for (field, value) in fields.items():
                if writer_annot.get(FA.T) == field or self._get_qualified_field_name(writer_annot) == field:
                    if isinstance(value, list):
                        lst = ArrayObject((TextStringObject(v) for v in value))
                        writer_annot[NameObject(FA.V)] = lst
                    else:
                        writer_annot[NameObject(FA.V)] = TextStringObject(value)
                    if writer_annot.get(FA.FT) in '/Btn':
                        writer_annot[NameObject(AA.AS)] = NameObject(value)
                    elif writer_annot.get(FA.FT) == '/Tx' or writer_annot.get(FA.FT) == '/Ch':
                        if AA.DA not in writer_annot:
                            f = writer_annot
                            da = default_da
                            while AA.DA not in f:
                                f = f.get('/Parent')
                                if f is None:
                                    break
                                f = f.get_object()
                                if AA.DA in f:
                                    da = f[AA.DA]
                            writer_annot[NameObject(AA.DA)] = da
                        self._update_text_field(writer_annot)
                    elif writer_annot.get(FA.FT) == '/Sig':
                        logger_warning('Signature forms not implemented yet', __name__)
                    if flags:
                        writer_annot[NameObject(FA.Ff)] = NumberObject(flags)
                elif writer_parent_annot.get(FA.T) == field or self._get_qualified_field_name(writer_parent_annot) == field:
                    writer_parent_annot[NameObject(FA.V)] = TextStringObject(value)
                    for k in writer_parent_annot[NameObject(FA.Kids)]:
                        k = k.get_object()
                        k[NameObject(AA.AS)] = NameObject(value if value in k[AA.AP]['/N'] else '/Off')

    def updatePageFormFieldValues(self, page: PageObject, fields: Dict[str, Any], flags: FieldFlag=OPTIONAL_READ_WRITE_FIELD) -> None:
        if False:
            print('Hello World!')
        '\n        Use :meth:`update_page_form_field_values` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('updatePageFormFieldValues', 'update_page_form_field_values', '3.0.0')
        return self.update_page_form_field_values(page, fields, flags)

    def clone_reader_document_root(self, reader: PdfReader) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Copy the reader document root to the writer and all sub elements,\n        including pages, threads, outlines,... For partial insertion, ``append``\n        should be considered.\n\n        Args:\n            reader: PdfReader from the document root should be copied.\n        '
        self._objects.clear()
        self._root_object = cast(DictionaryObject, reader.trailer[TK.ROOT].clone(self))
        self._root = self._root_object.indirect_reference
        self._pages = self._root_object.raw_get('/Pages')
        self._flatten()
        for p in self.flattened_pages:
            o = p.get_object()
            self._objects[p.idnum - 1] = PageObject(self, p)
            self._objects[p.idnum - 1].update(o.items())
        self._root_object[NameObject('/Pages')][NameObject('/Kids')] = self.flattened_pages
        del self.flattened_pages

    def cloneReaderDocumentRoot(self, reader: PdfReader) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Use :meth:`clone_reader_document_root` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('cloneReaderDocumentRoot', 'clone_reader_document_root', '3.0.0')
        self.clone_reader_document_root(reader)

    def _flatten(self, pages: Union[None, DictionaryObject, PageObject]=None, inherit: Optional[Dict[str, Any]]=None, indirect_reference: Optional[IndirectObject]=None) -> None:
        if False:
            i = 10
            return i + 15
        inheritable_page_attributes = (NameObject(PG.RESOURCES), NameObject(PG.MEDIABOX), NameObject(PG.CROPBOX), NameObject(PG.ROTATE))
        if inherit is None:
            inherit = {}
        if pages is None:
            pages = cast(DictionaryObject, self._root_object['/Pages'])
            self.flattened_pages = ArrayObject()
        assert pages is not None
        if PA.TYPE in pages:
            t = str(pages[PA.TYPE])
        elif PA.KIDS not in pages:
            t = '/Page'
        else:
            t = '/Pages'
        if t == '/Pages':
            for attr in inheritable_page_attributes:
                if attr in pages:
                    inherit[attr] = pages[attr]
            for page in cast(ArrayObject, pages[PA.KIDS]):
                addt = {}
                if isinstance(page, IndirectObject):
                    addt['indirect_reference'] = page
                self._flatten(page.get_object(), inherit, **addt)
        elif t == '/Page':
            for (attr_in, value) in list(inherit.items()):
                if attr_in not in pages:
                    pages[attr_in] = value
            pages[NameObject('/Parent')] = cast(IndirectObject, self._root_object.raw_get('/Pages'))
            self.flattened_pages.append(indirect_reference)

    def clone_document_from_reader(self, reader: PdfReader, after_page_append: Optional[Callable[[PageObject], None]]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Create a copy (clone) of a document from a PDF file reader cloning\n        section '/Root' and '/Info' and '/ID' of the pdf.\n\n        Args:\n            reader: PDF file reader instance from which the clone\n                should be created.\n            after_page_append:\n                Callback function that is invoked after each page is appended to\n                the writer. Signature includes a reference to the appended page\n                (delegates to append_pages_from_reader). The single parameter of\n                the callback is a reference to the page just appended to the\n                document.\n        "
        self.clone_reader_document_root(reader)
        if TK.INFO in reader.trailer:
            self._info = reader.trailer[TK.INFO].clone(self).indirect_reference
        try:
            self._ID = cast(ArrayObject, reader.trailer[TK.ID].clone(self))
        except KeyError:
            pass
        if callable(after_page_append):
            for page in cast(ArrayObject, cast(DictionaryObject, self._pages.get_object())['/Kids']):
                after_page_append(page.get_object())

    def cloneDocumentFromReader(self, reader: PdfReader, after_page_append: Optional[Callable[[PageObject], None]]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Use :meth:`clone_document_from_reader` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('cloneDocumentFromReader', 'clone_document_from_reader', '3.0.0')
        self.clone_document_from_reader(reader, after_page_append)

    def _compute_document_identifier_from_content(self) -> ByteStringObject:
        if False:
            return 10
        stream = BytesIO()
        self._write_pdf_structure(stream)
        stream.seek(0)
        return ByteStringObject(_rolling_checksum(stream).encode('utf8'))

    def generate_file_identifiers(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Generate an identifier for the PDF that will be written.\n\n        The only point of this is ensuring uniqueness. Reproducibility is not\n        required; see 14.4 "File Identifiers".\n        '
        if hasattr(self, '_ID') and self._ID and (len(self._ID) == 2):
            ID_1 = self._ID[0]
        else:
            ID_1 = self._compute_document_identifier_from_content()
        ID_2 = self._compute_document_identifier_from_content()
        self._ID = ArrayObject((ID_1, ID_2))

    def encrypt(self, user_password: Optional[str]=None, owner_password: Optional[str]=None, use_128bit: bool=True, permissions_flag: UserAccessPermissions=ALL_DOCUMENT_PERMISSIONS, user_pwd: Optional[str]=None, owner_pwd: Optional[str]=None, *, algorithm: Optional[str]=None) -> None:
        if False:
            return 10
        '\n        Encrypt this PDF file with the PDF Standard encryption handler.\n\n        Args:\n            user_password: The password which allows for opening\n                and reading the PDF file with the restrictions provided.\n            owner_password: The password which allows for\n                opening the PDF files without any restrictions.  By default,\n                the owner password is the same as the user password.\n            use_128bit: flag as to whether to use 128bit\n                encryption.  When false, 40bit encryption will be used.\n                By default, this flag is on.\n            permissions_flag: permissions as described in\n                TABLE 3.20 of the PDF 1.7 specification. A bit value of 1 means\n                the permission is grantend.\n                Hence an integer value of -1 will set all flags.\n                Bit position 3 is for printing, 4 is for modifying content,\n                5 and 6 control annotations, 9 for form fields,\n                10 for extraction of text and graphics.\n            algorithm: encrypt algorithm. Values maybe one of "RC4-40", "RC4-128",\n                "AES-128", "AES-256-R5", "AES-256". If it\'s valid,\n                `use_128bit` will be ignored.\n        '
        if user_pwd is not None:
            if user_password is not None:
                raise ValueError("Please only set 'user_password'. The 'user_pwd' argument is deprecated.")
            else:
                warnings.warn("Please use 'user_password' instead of 'user_pwd'. The 'user_pwd' argument is deprecated and will be removed in pypdf 4.0.0.")
                user_password = user_pwd
        if user_password is None:
            raise ValueError('user_password may not be None')
        if owner_pwd is not None:
            if owner_password is not None:
                raise ValueError('The argument owner_pwd of encrypt is deprecated. Use owner_password only.')
            else:
                old_term = 'owner_pwd'
                new_term = 'owner_password'
                warnings.warn(message=f'{old_term} is deprecated as an argument and will be removed in pypdf 4.0.0. Use {new_term} instead', category=DeprecationWarning)
                owner_password = owner_pwd
        if owner_password is None:
            owner_password = user_password
        if algorithm is not None:
            try:
                alg = getattr(EncryptAlgorithm, algorithm.replace('-', '_'))
            except AttributeError:
                raise ValueError(f"algorithm '{algorithm}' NOT supported")
        else:
            alg = EncryptAlgorithm.RC4_128
            if not use_128bit:
                alg = EncryptAlgorithm.RC4_40
        self.generate_file_identifiers()
        self._encryption = Encryption.make(alg, permissions_flag, self._ID[0])
        entry = self._encryption.write_entry(user_password, owner_password)
        if self._encrypt_entry:
            assert self._encrypt_entry.indirect_reference is not None
            entry.indirect_reference = self._encrypt_entry.indirect_reference
            self._objects[entry.indirect_reference.idnum - 1] = entry
        else:
            self._add_object(entry)
        self._encrypt_entry = entry

    def write_stream(self, stream: StreamType) -> None:
        if False:
            while True:
                i = 10
        if hasattr(stream, 'mode') and 'b' not in stream.mode:
            logger_warning(f'File <{stream.name}> to write to is not in binary mode. It may not be written to correctly.', __name__)
        if not self._root:
            self._root = self._add_object(self._root_object)
        self._sweep_indirect_references(self._root)
        object_positions = self._write_pdf_structure(stream)
        xref_location = self._write_xref_table(stream, object_positions)
        self._write_trailer(stream, xref_location)

    def write(self, stream: Union[Path, StrByteType]) -> Tuple[bool, IO[Any]]:
        if False:
            i = 10
            return i + 15
        '\n        Write the collection of pages added to this object out as a PDF file.\n\n        Args:\n            stream: An object to write the file to.  The object can support\n                the write method and the tell method, similar to a file object, or\n                be a file path, just like the fileobj, just named it stream to keep\n                existing workflow.\n\n        Returns:\n            A tuple (bool, IO)\n        '
        my_file = False
        if stream == '':
            raise ValueError(f'Output(stream={stream}) is empty.')
        if isinstance(stream, (str, Path)):
            stream = FileIO(stream, 'wb')
            self.with_as_usage = True
            my_file = True
        self.write_stream(stream)
        if self.with_as_usage:
            stream.close()
        return (my_file, stream)

    def _write_pdf_structure(self, stream: StreamType) -> List[int]:
        if False:
            print('Hello World!')
        object_positions = []
        stream.write(self.pdf_header + b'\n')
        stream.write(b'%\xe2\xe3\xcf\xd3\n')
        for (i, obj) in enumerate(self._objects):
            if obj is not None:
                idnum = i + 1
                object_positions.append(stream.tell())
                stream.write(f'{idnum} 0 obj\n'.encode())
                if self._encryption and obj != self._encrypt_entry:
                    obj = self._encryption.encrypt_object(obj, idnum, 0)
                obj.write_to_stream(stream)
                stream.write(b'\nendobj\n')
        return object_positions

    def _write_xref_table(self, stream: StreamType, object_positions: List[int]) -> int:
        if False:
            print('Hello World!')
        xref_location = stream.tell()
        stream.write(b'xref\n')
        stream.write(f'0 {len(self._objects) + 1}\n'.encode())
        stream.write(f'{0:0>10} {65535:0>5} f \n'.encode())
        for offset in object_positions:
            stream.write(f'{offset:0>10} {0:0>5} n \n'.encode())
        return xref_location

    def _write_trailer(self, stream: StreamType, xref_location: int) -> None:
        if False:
            return 10
        '\n        Write the PDF trailer to the stream.\n\n        To quote the PDF specification:\n            [The] trailer [gives] the location of the cross-reference table and\n            of certain special objects within the body of the file.\n        '
        stream.write(b'trailer\n')
        trailer = DictionaryObject()
        trailer.update({NameObject(TK.SIZE): NumberObject(len(self._objects) + 1), NameObject(TK.ROOT): self._root, NameObject(TK.INFO): self._info})
        if hasattr(self, '_ID'):
            trailer[NameObject(TK.ID)] = self._ID
        if self._encrypt_entry:
            trailer[NameObject(TK.ENCRYPT)] = self._encrypt_entry.indirect_reference
        trailer.write_to_stream(stream)
        stream.write(f'\nstartxref\n{xref_location}\n%%EOF\n'.encode())

    def add_metadata(self, infos: Dict[str, Any]) -> None:
        if False:
            while True:
                i = 10
        '\n        Add custom metadata to the output.\n\n        Args:\n            infos: a Python dictionary where each key is a field\n                and each value is your new metadata.\n        '
        args = {}
        if isinstance(infos, PdfObject):
            infos = cast(DictionaryObject, infos.get_object())
        for (key, value) in list(infos.items()):
            if isinstance(value, PdfObject):
                value = value.get_object()
            args[NameObject(key)] = create_string_object(str(value))
        cast(DictionaryObject, self._info.get_object()).update(args)

    def addMetadata(self, infos: Dict[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Use :meth:`add_metadata` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('addMetadata', 'add_metadata', '3.0.0')
        self.add_metadata(infos)

    def _sweep_indirect_references(self, root: Union[ArrayObject, BooleanObject, DictionaryObject, FloatObject, IndirectObject, NameObject, PdfObject, NumberObject, TextStringObject, NullObject]) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Resolving any circular references to Page objects.\n\n        Circular references to Page objects can arise when objects such as\n        annotations refer to their associated page. If these references are not\n        properly handled, the PDF file will contain multiple copies of the same\n        Page object. To address this problem, Page objects store their original\n        object reference number. This method adds the reference number of any\n        circularly referenced Page objects to an external reference map. This\n        ensures that self-referencing trees reference the correct new object\n        location, rather than copying in a new copy of the Page object.\n\n        Args:\n            root: The root of the PDF object tree to sweep.\n        '
        stack: Deque[Tuple[Any, Optional[Any], Any, List[PdfObject]]] = collections.deque()
        discovered = []
        parent = None
        grant_parents: List[PdfObject] = []
        key_or_id = None
        stack.append((root, parent, key_or_id, grant_parents))
        while len(stack):
            (data, parent, key_or_id, grant_parents) = stack.pop()
            if isinstance(data, (ArrayObject, DictionaryObject)):
                for (key, value) in data.items():
                    stack.append((value, data, key, grant_parents + [parent] if parent is not None else []))
            elif isinstance(data, IndirectObject) and data.pdf != self:
                data = self._resolve_indirect_object(data)
                if str(data) not in discovered:
                    discovered.append(str(data))
                    stack.append((data.get_object(), None, None, []))
            if isinstance(parent, (DictionaryObject, ArrayObject)):
                if isinstance(data, StreamObject):
                    data = self._resolve_indirect_object(self._add_object(data))
                update_hashes = []
                if parent[key_or_id] != data:
                    update_hashes = [parent.hash_value()] + [grant_parent.hash_value() for grant_parent in grant_parents]
                    parent[key_or_id] = data
                for old_hash in update_hashes:
                    indirect_reference = self._idnum_hash.pop(old_hash, None)
                    if indirect_reference is not None:
                        indirect_reference_obj = indirect_reference.get_object()
                        if indirect_reference_obj is not None:
                            self._idnum_hash[indirect_reference_obj.hash_value()] = indirect_reference

    def _resolve_indirect_object(self, data: IndirectObject) -> IndirectObject:
        if False:
            return 10
        "\n        Resolves an indirect object to an indirect object in this PDF file.\n\n        If the input indirect object already belongs to this PDF file, it is\n        returned directly. Otherwise, the object is retrieved from the input\n        object's PDF file using the object's ID number and generation number. If\n        the object cannot be found, a warning is logged and a `NullObject` is\n        returned.\n\n        If the object is not already in this PDF file, it is added to the file's\n        list of objects and assigned a new ID number and generation number of 0.\n        The hash value of the object is then added to the `_idnum_hash`\n        dictionary, with the corresponding `IndirectObject` reference as the\n        value.\n\n        Args:\n            data: The `IndirectObject` to resolve.\n\n        Returns:\n            The resolved `IndirectObject` in this PDF file.\n\n        Raises:\n            ValueError: If the input stream is closed.\n        "
        if hasattr(data.pdf, 'stream') and data.pdf.stream.closed:
            raise ValueError(f'I/O operation on closed file: {data.pdf.stream.name}')
        if data.pdf == self:
            return data
        real_obj = data.pdf.get_object(data)
        if real_obj is None:
            logger_warning(f'Unable to resolve [{data.__class__.__name__}: {data}], returning NullObject instead', __name__)
            real_obj = NullObject()
        hash_value = real_obj.hash_value()
        if hash_value in self._idnum_hash:
            return self._idnum_hash[hash_value]
        if data.pdf == self:
            self._idnum_hash[hash_value] = IndirectObject(data.idnum, 0, self)
        else:
            self._idnum_hash[hash_value] = self._add_object(real_obj)
        return self._idnum_hash[hash_value]

    def get_reference(self, obj: PdfObject) -> IndirectObject:
        if False:
            i = 10
            return i + 15
        idnum = self._objects.index(obj) + 1
        ref = IndirectObject(idnum, 0, self)
        assert ref.get_object() == obj
        return ref

    def getReference(self, obj: PdfObject) -> IndirectObject:
        if False:
            print('Hello World!')
        '\n        Use :meth:`get_reference` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('getReference', 'get_reference', '3.0.0')
        return self.get_reference(obj)

    def get_outline_root(self) -> TreeObject:
        if False:
            for i in range(10):
                print('nop')
        if CO.OUTLINES in self._root_object:
            outline = cast(TreeObject, self._root_object[CO.OUTLINES])
            if not isinstance(outline, TreeObject):
                t = TreeObject(outline)
                self._replace_object(outline.indirect_reference.idnum, t)
                outline = t
            idnum = self._objects.index(outline) + 1
            outline_ref = IndirectObject(idnum, 0, self)
            assert outline_ref.get_object() == outline
        else:
            outline = TreeObject()
            outline.update({})
            outline_ref = self._add_object(outline)
            self._root_object[NameObject(CO.OUTLINES)] = outline_ref
        return outline

    def get_threads_root(self) -> ArrayObject:
        if False:
            i = 10
            return i + 15
        '\n        The list of threads.\n\n        See §8.3.2 from PDF 1.7 spec.\n\n        Returns:\n            An array (possibly empty) of Dictionaries with ``/F`` and\n            ``/I`` properties.\n        '
        if CO.THREADS in self._root_object:
            threads = cast(ArrayObject, self._root_object[CO.THREADS])
        else:
            threads = ArrayObject()
            self._root_object[NameObject(CO.THREADS)] = threads
        return threads

    @property
    def threads(self) -> ArrayObject:
        if False:
            return 10
        '\n        Read-only property for the list of threads.\n\n        See §8.3.2 from PDF 1.7 spec.\n\n        Each element is a dictionaries with ``/F`` and ``/I`` keys.\n        '
        return self.get_threads_root()

    def getOutlineRoot(self) -> TreeObject:
        if False:
            print('Hello World!')
        '\n        Use :meth:`get_outline_root` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('getOutlineRoot', 'get_outline_root', '3.0.0')
        return self.get_outline_root()

    def get_named_dest_root(self) -> ArrayObject:
        if False:
            print('Hello World!')
        if CA.NAMES in self._root_object and isinstance(self._root_object[CA.NAMES], DictionaryObject):
            names = cast(DictionaryObject, self._root_object[CA.NAMES])
            names_ref = names.indirect_reference
            if CA.DESTS in names and isinstance(names[CA.DESTS], DictionaryObject):
                dests = cast(DictionaryObject, names[CA.DESTS])
                dests_ref = dests.indirect_reference
                if CA.NAMES in dests:
                    nd = cast(ArrayObject, dests[CA.NAMES])
                else:
                    nd = ArrayObject()
                    dests[NameObject(CA.NAMES)] = nd
            else:
                dests = DictionaryObject()
                dests_ref = self._add_object(dests)
                names[NameObject(CA.DESTS)] = dests_ref
                nd = ArrayObject()
                dests[NameObject(CA.NAMES)] = nd
        else:
            names = DictionaryObject()
            names_ref = self._add_object(names)
            self._root_object[NameObject(CA.NAMES)] = names_ref
            dests = DictionaryObject()
            dests_ref = self._add_object(dests)
            names[NameObject(CA.DESTS)] = dests_ref
            nd = ArrayObject()
            dests[NameObject(CA.NAMES)] = nd
        return nd

    def getNamedDestRoot(self) -> ArrayObject:
        if False:
            print('Hello World!')
        '\n        Use :meth:`get_named_dest_root` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('getNamedDestRoot', 'get_named_dest_root', '3.0.0')
        return self.get_named_dest_root()

    def add_outline_item_destination(self, page_destination: Union[None, IndirectObject, PageObject, TreeObject]=None, parent: Union[None, TreeObject, IndirectObject]=None, before: Union[None, TreeObject, IndirectObject]=None, is_open: bool=True, dest: Union[None, PageObject, TreeObject]=None) -> IndirectObject:
        if False:
            while True:
                i = 10
        if page_destination is not None and dest is not None:
            raise ValueError('The argument dest of add_outline_item_destination is deprecated. Use page_destination only.')
        if dest is not None:
            old_term = 'dest'
            new_term = 'page_destination'
            warnings.warn(message=f'{old_term} is deprecated as an argument and will be removed in pypdf 4.0.0. Use {new_term} instead', category=DeprecationWarning)
            page_destination = dest
        if page_destination is None:
            raise ValueError('page_destination may not be None')
        page_destination = cast(PageObject, page_destination.get_object())
        if isinstance(page_destination, PageObject):
            return self.add_outline_item_destination(Destination(f'page #{page_destination.page_number}', cast(IndirectObject, page_destination.indirect_reference), Fit.fit()))
        if parent is None:
            parent = self.get_outline_root()
        page_destination[NameObject('/%is_open%')] = BooleanObject(is_open)
        parent = cast(TreeObject, parent.get_object())
        page_destination_ref = self._add_object(page_destination)
        if before is not None:
            before = before.indirect_reference
        parent.insert_child(page_destination_ref, before, self, page_destination.inc_parent_counter_outline if is_open else lambda x, y: 0)
        if '/Count' not in page_destination:
            page_destination[NameObject('/Count')] = NumberObject(0)
        return page_destination_ref

    def add_bookmark_destination(self, dest: Union[PageObject, TreeObject], parent: Union[None, TreeObject, IndirectObject]=None) -> IndirectObject:
        if False:
            while True:
                i = 10
        '\n        Use :meth:`add_outline_item_destination` instead.\n\n        .. deprecated:: 2.9.0\n        '
        deprecation_with_replacement('add_bookmark_destination', 'add_outline_item_destination', '3.0.0')
        return self.add_outline_item_destination(dest, parent)

    def addBookmarkDestination(self, dest: PageObject, parent: Optional[TreeObject]=None) -> IndirectObject:
        if False:
            for i in range(10):
                print('nop')
        '\n        Use :meth:`add_outline_item_destination` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('addBookmarkDestination', 'add_outline_item_destination', '3.0.0')
        return self.add_outline_item_destination(dest, parent)

    @deprecation_bookmark(bookmark='outline_item')
    def add_outline_item_dict(self, outline_item: OutlineItemType, parent: Union[None, TreeObject, IndirectObject]=None, before: Union[None, TreeObject, IndirectObject]=None, is_open: bool=True) -> IndirectObject:
        if False:
            i = 10
            return i + 15
        outline_item_object = TreeObject()
        outline_item_object.update(outline_item)
        if '/A' in outline_item:
            action = DictionaryObject()
            a_dict = cast(DictionaryObject, outline_item['/A'])
            for (k, v) in list(a_dict.items()):
                action[NameObject(str(k))] = v
            action_ref = self._add_object(action)
            outline_item_object[NameObject('/A')] = action_ref
        return self.add_outline_item_destination(outline_item_object, parent, before, is_open)

    @deprecation_bookmark(bookmark='outline_item')
    def add_bookmark_dict(self, outline_item: OutlineItemType, parent: Optional[TreeObject]=None) -> IndirectObject:
        if False:
            print('Hello World!')
        '\n        Use :meth:`add_outline_item_dict` instead.\n\n        .. deprecated:: 2.9.0\n        '
        deprecation_with_replacement('add_bookmark_dict', 'add_outline_item_dict', '3.0.0')
        return self.add_outline_item_dict(outline_item, parent)

    @deprecation_bookmark(bookmark='outline_item')
    def addBookmarkDict(self, outline_item: OutlineItemType, parent: Optional[TreeObject]=None) -> IndirectObject:
        if False:
            while True:
                i = 10
        '\n        Use :meth:`add_outline_item_dict` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('addBookmarkDict', 'add_outline_item_dict', '3.0.0')
        return self.add_outline_item_dict(outline_item, parent)

    def add_outline_item(self, title: str, page_number: Union[None, PageObject, IndirectObject, int], parent: Union[None, TreeObject, IndirectObject]=None, before: Union[None, TreeObject, IndirectObject]=None, color: Optional[Union[Tuple[float, float, float], str]]=None, bold: bool=False, italic: bool=False, fit: Fit=PAGE_FIT, is_open: bool=True, pagenum: Optional[int]=None) -> IndirectObject:
        if False:
            print('Hello World!')
        '\n        Add an outline item (commonly referred to as a "Bookmark") to the PDF file.\n\n        Args:\n            title: Title to use for this outline item.\n            page_number: Page number this outline item will point to.\n            parent: A reference to a parent outline item to create nested\n                outline items.\n            before:\n            color: Color of the outline item\'s font as a red, green, blue tuple\n                from 0.0 to 1.0 or as a Hex String (#RRGGBB)\n            bold: Outline item font is bold\n            italic: Outline item font is italic\n            fit: The fit of the destination page.\n\n        Returns:\n            The added outline item as an indirect object.\n        '
        page_ref: Union[None, NullObject, IndirectObject, NumberObject]
        if isinstance(italic, Fit):
            if fit is not None and page_number is None:
                page_number = fit
            return self.add_outline_item(title, page_number, parent, None, before, color, bold, italic, is_open=is_open)
        if page_number is not None and pagenum is not None:
            raise ValueError('The argument pagenum of add_outline_item is deprecated. Use page_number only.')
        if page_number is None:
            action_ref = None
        else:
            if isinstance(page_number, IndirectObject):
                page_ref = page_number
            elif isinstance(page_number, PageObject):
                page_ref = page_number.indirect_reference
            elif isinstance(page_number, int):
                try:
                    page_ref = self.pages[page_number].indirect_reference
                except IndexError:
                    page_ref = NumberObject(page_number)
            if page_ref is None:
                logger_warning(f'can not find reference of page {page_number}', __name__)
                page_ref = NullObject()
            dest = Destination(NameObject('/' + title + ' outline item'), page_ref, fit)
            action_ref = self._add_object(DictionaryObject({NameObject(GoToActionArguments.D): dest.dest_array, NameObject(GoToActionArguments.S): NameObject('/GoTo')}))
        outline_item = self._add_object(_create_outline_item(action_ref, title, color, italic, bold))
        if parent is None:
            parent = self.get_outline_root()
        return self.add_outline_item_destination(outline_item, parent, before, is_open)

    def add_bookmark(self, title: str, pagenum: int, parent: Union[None, TreeObject, IndirectObject]=None, color: Optional[Tuple[float, float, float]]=None, bold: bool=False, italic: bool=False, fit: FitType='/Fit', *args: ZoomArgType) -> IndirectObject:
        if False:
            while True:
                i = 10
        '\n        Use :meth:`add_outline_item` instead.\n\n        .. deprecated:: 2.9.0\n        '
        deprecation_with_replacement('add_bookmark', 'add_outline_item', '3.0.0')
        return self.add_outline_item(title, pagenum, parent, color, bold, italic, Fit(fit_type=fit, fit_args=args))

    def addBookmark(self, title: str, pagenum: int, parent: Union[None, TreeObject, IndirectObject]=None, color: Optional[Tuple[float, float, float]]=None, bold: bool=False, italic: bool=False, fit: FitType='/Fit', *args: ZoomArgType) -> IndirectObject:
        if False:
            while True:
                i = 10
        '\n        Use :meth:`add_outline_item` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('addBookmark', 'add_outline_item', '3.0.0')
        return self.add_outline_item(title, pagenum, parent, None, color, bold, italic, Fit(fit_type=fit, fit_args=args))

    def add_outline(self) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError('This method is not yet implemented. Use :meth:`add_outline_item` instead.')

    def add_named_destination_array(self, title: TextStringObject, destination: Union[IndirectObject, ArrayObject]) -> None:
        if False:
            return 10
        nd = self.get_named_dest_root()
        i = 0
        while i < len(nd):
            if title < nd[i]:
                nd.insert(i, destination)
                nd.insert(i, TextStringObject(title))
                return
            else:
                i += 2
        nd.extend([TextStringObject(title), destination])
        return

    def add_named_destination_object(self, page_destination: Optional[PdfObject]=None, dest: Optional[PdfObject]=None) -> IndirectObject:
        if False:
            while True:
                i = 10
        if page_destination is not None and dest is not None:
            raise ValueError('The argument dest of add_named_destination_object is deprecated. Use page_destination only.')
        if dest is not None:
            old_term = 'dest'
            new_term = 'page_destination'
            warnings.warn(message=f'{old_term} is deprecated as an argument and will be removed in pypdf 4.0.0. Use {new_term} instead', category=DeprecationWarning)
            page_destination = dest
        if page_destination is None:
            raise ValueError('page_destination may not be None')
        page_destination_ref = self._add_object(page_destination.dest_array)
        self.add_named_destination_array(cast('TextStringObject', page_destination['/Title']), page_destination_ref)
        return page_destination_ref

    def addNamedDestinationObject(self, dest: Destination) -> IndirectObject:
        if False:
            while True:
                i = 10
        '\n        Use :meth:`add_named_destination_object` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('addNamedDestinationObject', 'add_named_destination_object', '3.0.0')
        return self.add_named_destination_object(dest)

    def add_named_destination(self, title: str, page_number: Optional[int]=None, pagenum: Optional[int]=None) -> IndirectObject:
        if False:
            while True:
                i = 10
        if page_number is not None and pagenum is not None:
            raise ValueError('The argument pagenum of add_outline_item is deprecated. Use page_number only.')
        if pagenum is not None:
            old_term = 'pagenum'
            new_term = 'page_number'
            warnings.warn(message=f'{old_term} is deprecated as an argument and will be removed in pypdf 4.0.0. Use {new_term} instead', category=DeprecationWarning)
            page_number = pagenum
        if page_number is None:
            raise ValueError('page_number may not be None')
        page_ref = self.get_object(self._pages)[PA.KIDS][page_number]
        dest = DictionaryObject()
        dest.update({NameObject(GoToActionArguments.D): ArrayObject([page_ref, NameObject(TypFitArguments.FIT_H), NumberObject(826)]), NameObject(GoToActionArguments.S): NameObject('/GoTo')})
        dest_ref = self._add_object(dest)
        if not isinstance(title, TextStringObject):
            title = TextStringObject(str(title))
        self.add_named_destination_array(title, dest_ref)
        return dest_ref

    def addNamedDestination(self, title: str, pagenum: int) -> IndirectObject:
        if False:
            return 10
        '\n        Use :meth:`add_named_destination` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('addNamedDestination', 'add_named_destination', '3.0.0')
        return self.add_named_destination(title, pagenum)

    def remove_links(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Remove links and annotations from this output.'
        for page in self.pages:
            self.remove_objects_from_page(page, ObjectDeletionFlag.ALL_ANNOTATIONS)

    def removeLinks(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Use :meth:`remove_links` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('removeLinks', 'remove_links', '3.0.0')
        return self.remove_links()

    def remove_annotations(self, subtypes: Optional[Union[AnnotationSubtype, Iterable[AnnotationSubtype]]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Remove annotations by annotation subtype.\n\n        Args:\n            subtypes: SubType or list of SubTypes to be removed.\n                Examples are: "/Link", "/FileAttachment", "/Sound",\n                "/Movie", "/Screen", ...\n                If you want to remove all annotations, use subtypes=None.\n        '
        for page in self.pages:
            self._remove_annots_from_page(page, subtypes)

    def _remove_annots_from_page(self, page: Union[IndirectObject, PageObject, DictionaryObject], subtypes: Optional[Iterable[str]]) -> None:
        if False:
            print('Hello World!')
        page = cast(DictionaryObject, page.get_object())
        if PG.ANNOTS in page:
            i = 0
            while i < len(cast(ArrayObject, page[PG.ANNOTS])):
                an = cast(ArrayObject, page[PG.ANNOTS])[i]
                obj = cast(DictionaryObject, an.get_object())
                if subtypes is None or cast(str, obj['/Subtype']) in subtypes:
                    if isinstance(an, IndirectObject):
                        self._objects[an.idnum - 1] = NullObject()
                    del page[PG.ANNOTS][i]
                else:
                    i += 1

    def remove_objects_from_page(self, page: Union[PageObject, DictionaryObject], to_delete: Union[ObjectDeletionFlag, Iterable[ObjectDeletionFlag]]) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Remove objects specified by ``to_delete`` from the given page.\n\n        Args:\n            page: Page object to clean up.\n            to_delete: Objects to be deleted; can be a ``ObjectDeletionFlag``\n                or a list of ObjectDeletionFlag\n        '
        if isinstance(to_delete, (list, tuple)):
            for to_d in to_delete:
                self.remove_objects_from_page(page, to_d)
            return
        assert isinstance(to_delete, ObjectDeletionFlag)
        if to_delete & ObjectDeletionFlag.LINKS:
            return self._remove_annots_from_page(page, ('/Link',))
        if to_delete & ObjectDeletionFlag.ATTACHMENTS:
            return self._remove_annots_from_page(page, ('/FileAttachment', '/Sound', '/Movie', '/Screen'))
        if to_delete & ObjectDeletionFlag.OBJECTS_3D:
            return self._remove_annots_from_page(page, ('/3D',))
        if to_delete & ObjectDeletionFlag.ALL_ANNOTATIONS:
            return self._remove_annots_from_page(page, None)
        jump_operators = []
        if to_delete & ObjectDeletionFlag.DRAWING_IMAGES:
            jump_operators = [b'w', b'J', b'j', b'M', b'd', b'i'] + [b'W', b'W*'] + [b'b', b'b*', b'B', b'B*', b'S', b's', b'f', b'f*', b'F', b'n'] + [b'm', b'l', b'c', b'v', b'y', b'h', b're'] + [b'sh']
        if to_delete & ObjectDeletionFlag.TEXT:
            jump_operators = [b'Tj', b'TJ', b"'", b'"']

        def clean(content: ContentStream, images: List[str], forms: List[str]) -> None:
            if False:
                i = 10
                return i + 15
            nonlocal jump_operators, to_delete
            i = 0
            while i < len(content.operations):
                (operands, operator) = content.operations[i]
                if operator == b'INLINE IMAGE' and to_delete & ObjectDeletionFlag.INLINE_IMAGES or operator in jump_operators or (operator == b'Do' and to_delete & ObjectDeletionFlag.XOBJECT_IMAGES and (operands[0] in images)):
                    del content.operations[i]
                else:
                    i += 1
            content.get_data()

        def clean_forms(elt: DictionaryObject, stack: List[DictionaryObject]) -> Tuple[List[str], List[str]]:
            if False:
                while True:
                    i = 10
            nonlocal to_delete
            if elt in stack:
                return ([], [])
            try:
                d = cast(Dict[Any, Any], cast(DictionaryObject, elt['/Resources'])['/XObject'])
            except KeyError:
                d = {}
            images = []
            forms = []
            for (k, v) in d.items():
                o = v.get_object()
                try:
                    content: Any = None
                    if to_delete & ObjectDeletionFlag.XOBJECT_IMAGES and o['/Subtype'] == '/Image':
                        content = NullObject()
                        images.append(k)
                    if o['/Subtype'] == '/Form':
                        forms.append(k)
                        if isinstance(o, ContentStream):
                            content = o
                        else:
                            content = ContentStream(o, self)
                            content.update({k1: v1 for (k1, v1) in o.items() if k1 not in ['/Length', '/Filter', '/DecodeParms']})
                        clean_forms(content, stack + [elt])
                    if content is not None:
                        if isinstance(v, IndirectObject):
                            self._objects[v.idnum - 1] = content
                        else:
                            d[k] = self._add_object(content)
                except (TypeError, KeyError):
                    pass
            for im in images:
                del d[im]
            if isinstance(elt, StreamObject):
                if not isinstance(elt, ContentStream):
                    e = ContentStream(elt, self)
                    e.update(elt.items())
                    elt = e
                clean(elt, images, forms)
            return (images, forms)
        if not isinstance(page, PageObject):
            page = PageObject(self, page.indirect_reference)
        if '/Contents' in page:
            content = cast(ContentStream, page.get_contents())
            (images, forms) = clean_forms(page, [])
            clean(content, images, forms)
            page.replace_contents(content)

    def remove_images(self, to_delete: ImageType=ImageType.ALL, ignore_byte_string_object: Optional[bool]=None) -> None:
        if False:
            return 10
        '\n        Remove images from this output.\n\n        Args:\n            to_delete : The type of images to be deleted\n                (default = all images types)\n            ignore_byte_string_object: deprecated\n        '
        if isinstance(to_delete, bool):
            ignore_byte_string_object = to_delete
            to_delete = ImageType.ALL
        if ignore_byte_string_object is not None:
            warnings.warn("The 'ignore_byte_string_object' argument of remove_images is deprecated and will be removed in pypdf 4.0.0.", category=DeprecationWarning)
        i = (ObjectDeletionFlag.XOBJECT_IMAGES if to_delete & ImageType.XOBJECT_IMAGES else ObjectDeletionFlag.NONE) | (ObjectDeletionFlag.INLINE_IMAGES if to_delete & ImageType.INLINE_IMAGES else ObjectDeletionFlag.NONE) | (ObjectDeletionFlag.DRAWING_IMAGES if to_delete & ImageType.DRAWING_IMAGES else ObjectDeletionFlag.NONE)
        for page in self.pages:
            self.remove_objects_from_page(page, i)

    def removeImages(self, ignoreByteStringObject: bool=False) -> None:
        if False:
            return 10
        '\n        Use :meth:`remove_images` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('removeImages', 'remove_images', '3.0.0')
        return self.remove_images()

    def remove_text(self, ignore_byte_string_object: Optional[bool]=None) -> None:
        if False:
            while True:
                i = 10
        '\n        Remove text from this output.\n\n        Args:\n            ignore_byte_string_object: deprecated\n        '
        if ignore_byte_string_object is not None:
            warnings.warn("The 'ignore_byte_string_object' argument of remove_images is deprecated and will be removed in pypdf 4.0.0.", category=DeprecationWarning)
        for page in self.pages:
            self.remove_objects_from_page(page, ObjectDeletionFlag.TEXT)

    def removeText(self, ignoreByteStringObject: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Use :meth:`remove_text` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('removeText', 'remove_text', '3.0.0')
        return self.remove_text(ignoreByteStringObject)

    def add_uri(self, page_number: int, uri: str, rect: RectangleObject, border: Optional[ArrayObject]=None, pagenum: Optional[int]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Add an URI from a rectangular area to the specified page.\n\n        This uses the basic structure of :meth:`add_link`\n\n        Args:\n            page_number: index of the page on which to place the URI action.\n            uri: URI of resource to link to.\n            rect: :class:`RectangleObject<pypdf.generic.RectangleObject>` or\n                array of four integers specifying the clickable rectangular area\n                ``[xLL, yLL, xUR, yUR]``, or string in the form\n                ``"[ xLL yLL xUR yUR ]"``.\n            border: if provided, an array describing border-drawing\n                properties. See the PDF spec for details. No border will be\n                drawn if this argument is omitted.\n        '
        if pagenum is not None:
            warnings.warn("The 'pagenum' argument of add_uri is deprecated and will be removed in pypdf 4.0.0. Use 'page_number' instead.", category=DeprecationWarning)
            page_number = pagenum
        page_link = self.get_object(self._pages)[PA.KIDS][page_number]
        page_ref = cast(Dict[str, Any], self.get_object(page_link))
        border_arr: BorderArrayType
        if border is not None:
            border_arr = [NameObject(n) for n in border[:3]]
            if len(border) == 4:
                dash_pattern = ArrayObject([NameObject(n) for n in border[3]])
                border_arr.append(dash_pattern)
        else:
            border_arr = [NumberObject(2), NumberObject(2), NumberObject(2)]
        if isinstance(rect, str):
            rect = NameObject(rect)
        elif isinstance(rect, RectangleObject):
            pass
        else:
            rect = RectangleObject(rect)
        lnk2 = DictionaryObject()
        lnk2.update({NameObject('/S'): NameObject('/URI'), NameObject('/URI'): TextStringObject(uri)})
        lnk = DictionaryObject()
        lnk.update({NameObject(AA.Type): NameObject(PG.ANNOTS), NameObject(AA.Subtype): NameObject('/Link'), NameObject(AA.P): page_link, NameObject(AA.Rect): rect, NameObject('/H'): NameObject('/I'), NameObject(AA.Border): ArrayObject(border_arr), NameObject('/A'): lnk2})
        lnk_ref = self._add_object(lnk)
        if PG.ANNOTS in page_ref:
            page_ref[PG.ANNOTS].append(lnk_ref)
        else:
            page_ref[NameObject(PG.ANNOTS)] = ArrayObject([lnk_ref])

    def addURI(self, pagenum: int, uri: str, rect: RectangleObject, border: Optional[ArrayObject]=None) -> None:
        if False:
            return 10
        '\n        Use :meth:`add_uri` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('addURI', 'add_uri', '3.0.0')
        return self.add_uri(pagenum, uri, rect, border)

    def add_link(self, pagenum: int, page_destination: int, rect: RectangleObject, border: Optional[ArrayObject]=None, fit: FitType='/Fit', *args: ZoomArgType) -> DictionaryObject:
        if False:
            for i in range(10):
                print('nop')
        deprecation_with_replacement('add_link', 'add_annotation(pypdf.annotations.Link(...))')
        if isinstance(rect, str):
            rect = rect.strip()[1:-1]
            rect = RectangleObject([float(num) for num in rect.split(' ') if len(num) > 0])
        elif isinstance(rect, RectangleObject):
            pass
        else:
            rect = RectangleObject(rect)
        annotation = Link(rect=rect, border=border, target_page_index=page_destination, fit=Fit(fit_type=fit, fit_args=args))
        return self.add_annotation(page_number=pagenum, annotation=annotation)

    def addLink(self, pagenum: int, page_destination: int, rect: RectangleObject, border: Optional[ArrayObject]=None, fit: FitType='/Fit', *args: ZoomArgType) -> None:
        if False:
            while True:
                i = 10
        '\n        Use :meth:`add_link` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecate_with_replacement('addLink', 'add_annotation(pypdf.annotations.Link(...))', '4.0.0')
        self.add_link(pagenum, page_destination, rect, border, fit, *args)
    _valid_layouts = ('/NoLayout', '/SinglePage', '/OneColumn', '/TwoColumnLeft', '/TwoColumnRight', '/TwoPageLeft', '/TwoPageRight')

    def _get_page_layout(self) -> Optional[LayoutType]:
        if False:
            i = 10
            return i + 15
        try:
            return cast(LayoutType, self._root_object['/PageLayout'])
        except KeyError:
            return None

    def getPageLayout(self) -> Optional[LayoutType]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Use :py:attr:`page_layout` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('getPageLayout', 'page_layout', '3.0.0')
        return self._get_page_layout()

    def _set_page_layout(self, layout: Union[NameObject, LayoutType]) -> None:
        if False:
            while True:
                i = 10
        '\n        Set the page layout.\n\n        Args:\n            layout: The page layout to be used.\n\n        .. list-table:: Valid ``layout`` arguments\n           :widths: 50 200\n\n           * - /NoLayout\n             - Layout explicitly not specified\n           * - /SinglePage\n             - Show one page at a time\n           * - /OneColumn\n             - Show one column at a time\n           * - /TwoColumnLeft\n             - Show pages in two columns, odd-numbered pages on the left\n           * - /TwoColumnRight\n             - Show pages in two columns, odd-numbered pages on the right\n           * - /TwoPageLeft\n             - Show two pages at a time, odd-numbered pages on the left\n           * - /TwoPageRight\n             - Show two pages at a time, odd-numbered pages on the right\n        '
        if not isinstance(layout, NameObject):
            if layout not in self._valid_layouts:
                logger_warning(f"Layout should be one of: {('', ''.join(self._valid_layouts))}", __name__)
            layout = NameObject(layout)
        self._root_object.update({NameObject('/PageLayout'): layout})

    def set_page_layout(self, layout: LayoutType) -> None:
        if False:
            print('Hello World!')
        '\n        Set the page layout.\n\n        Args:\n            layout: The page layout to be used\n\n        .. list-table:: Valid ``layout`` arguments\n           :widths: 50 200\n\n           * - /NoLayout\n             - Layout explicitly not specified\n           * - /SinglePage\n             - Show one page at a time\n           * - /OneColumn\n             - Show one column at a time\n           * - /TwoColumnLeft\n             - Show pages in two columns, odd-numbered pages on the left\n           * - /TwoColumnRight\n             - Show pages in two columns, odd-numbered pages on the right\n           * - /TwoPageLeft\n             - Show two pages at a time, odd-numbered pages on the left\n           * - /TwoPageRight\n             - Show two pages at a time, odd-numbered pages on the right\n        '
        self._set_page_layout(layout)

    def setPageLayout(self, layout: LayoutType) -> None:
        if False:
            while True:
                i = 10
        '\n        Use :py:attr:`page_layout` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('writer.setPageLayout(val)', 'writer.page_layout = val', '3.0.0')
        return self._set_page_layout(layout)

    @property
    def page_layout(self) -> Optional[LayoutType]:
        if False:
            print('Hello World!')
        '\n        Page layout property.\n\n        .. list-table:: Valid ``layout`` values\n           :widths: 50 200\n\n           * - /NoLayout\n             - Layout explicitly not specified\n           * - /SinglePage\n             - Show one page at a time\n           * - /OneColumn\n             - Show one column at a time\n           * - /TwoColumnLeft\n             - Show pages in two columns, odd-numbered pages on the left\n           * - /TwoColumnRight\n             - Show pages in two columns, odd-numbered pages on the right\n           * - /TwoPageLeft\n             - Show two pages at a time, odd-numbered pages on the left\n           * - /TwoPageRight\n             - Show two pages at a time, odd-numbered pages on the right\n        '
        return self._get_page_layout()

    @page_layout.setter
    def page_layout(self, layout: LayoutType) -> None:
        if False:
            while True:
                i = 10
        self._set_page_layout(layout)

    @property
    def pageLayout(self) -> Optional[LayoutType]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Use :py:attr:`page_layout` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('pageLayout', 'page_layout', '3.0.0')
        return self.page_layout

    @pageLayout.setter
    def pageLayout(self, layout: LayoutType) -> None:
        if False:
            while True:
                i = 10
        '\n        Use :py:attr:`page_layout` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('pageLayout', 'page_layout', '3.0.0')
        self.page_layout = layout
    _valid_modes = ('/UseNone', '/UseOutlines', '/UseThumbs', '/FullScreen', '/UseOC', '/UseAttachments')

    def _get_page_mode(self) -> Optional[PagemodeType]:
        if False:
            i = 10
            return i + 15
        try:
            return cast(PagemodeType, self._root_object['/PageMode'])
        except KeyError:
            return None

    def getPageMode(self) -> Optional[PagemodeType]:
        if False:
            i = 10
            return i + 15
        '\n        Use :py:attr:`page_mode` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('getPageMode', 'page_mode', '3.0.0')
        return self._get_page_mode()

    def set_page_mode(self, mode: PagemodeType) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Use :py:attr:`page_mode` instead.\n\n        .. deprecated:: 1.28.0\n        '
        if isinstance(mode, NameObject):
            mode_name: NameObject = mode
        else:
            if mode not in self._valid_modes:
                logger_warning(f"Mode should be one of: {', '.join(self._valid_modes)}", __name__)
            mode_name = NameObject(mode)
        self._root_object.update({NameObject('/PageMode'): mode_name})

    def setPageMode(self, mode: PagemodeType) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Use :py:attr:`page_mode` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('writer.setPageMode(val)', 'writer.page_mode = val', '3.0.0')
        self.set_page_mode(mode)

    @property
    def page_mode(self) -> Optional[PagemodeType]:
        if False:
            print('Hello World!')
        '\n        Page mode property.\n\n        .. list-table:: Valid ``mode`` values\n           :widths: 50 200\n\n           * - /UseNone\n             - Do not show outline or thumbnails panels\n           * - /UseOutlines\n             - Show outline (aka bookmarks) panel\n           * - /UseThumbs\n             - Show page thumbnails panel\n           * - /FullScreen\n             - Fullscreen view\n           * - /UseOC\n             - Show Optional Content Group (OCG) panel\n           * - /UseAttachments\n             - Show attachments panel\n        '
        return self._get_page_mode()

    @page_mode.setter
    def page_mode(self, mode: PagemodeType) -> None:
        if False:
            i = 10
            return i + 15
        self.set_page_mode(mode)

    @property
    def pageMode(self) -> Optional[PagemodeType]:
        if False:
            return 10
        '\n        Use :py:attr:`page_mode` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('pageMode', 'page_mode', '3.0.0')
        return self.page_mode

    @pageMode.setter
    def pageMode(self, mode: PagemodeType) -> None:
        if False:
            print('Hello World!')
        '\n        Use :py:attr:`page_mode` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('pageMode', 'page_mode', '3.0.0')
        self.page_mode = mode

    def add_annotation(self, page_number: Union[int, PageObject], annotation: Dict[str, Any]) -> DictionaryObject:
        if False:
            return 10
        '\n        Add a single annotation to the page.\n        The added annotation must be a new annotation.\n        It can not be recycled.\n\n        Args:\n            page_number: PageObject or page index.\n            annotation: Annotation to be added (created with annotation).\n\n        Returns:\n            The inserted object\n            This can be used for pop-up creation, for example\n        '
        page = page_number
        if isinstance(page, int):
            page = self.pages[page]
        elif not isinstance(page, PageObject):
            raise TypeError('page: invalid type')
        to_add = cast(DictionaryObject, _pdf_objectify(annotation))
        to_add[NameObject('/P')] = page.indirect_reference
        if page.annotations is None:
            page[NameObject('/Annots')] = ArrayObject()
        assert page.annotations is not None
        if to_add.get('/Subtype') == '/Link' and '/Dest' in to_add:
            tmp = cast(Dict[Any, Any], to_add[NameObject('/Dest')])
            dest = Destination(NameObject('/LinkName'), tmp['target_page_index'], Fit(fit_type=tmp['fit'], fit_args=dict(tmp)['fit_args']))
            to_add[NameObject('/Dest')] = dest.dest_array
        page.annotations.append(self._add_object(to_add))
        if to_add.get('/Subtype') == '/Popup' and NameObject('/Parent') in to_add:
            cast(DictionaryObject, to_add['/Parent'].get_object())[NameObject('/Popup')] = to_add.indirect_reference
        return to_add

    def clean_page(self, page: Union[PageObject, IndirectObject]) -> PageObject:
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform some clean up in the page.\n        Currently: convert NameObject nameddestination to TextStringObject\n        (required for names/dests list)\n\n        Args:\n            page:\n\n        Returns:\n            The cleaned PageObject\n        '
        page = cast('PageObject', page.get_object())
        for a in page.get('/Annots', []):
            a_obj = a.get_object()
            d = a_obj.get('/Dest', None)
            act = a_obj.get('/A', None)
            if isinstance(d, NameObject):
                a_obj[NameObject('/Dest')] = TextStringObject(d)
            elif act is not None:
                act = act.get_object()
                d = act.get('/D', None)
                if isinstance(d, NameObject):
                    act[NameObject('/D')] = TextStringObject(d)
        return page

    def _create_stream(self, fileobj: Union[Path, StrByteType, PdfReader]) -> Tuple[IOBase, Optional[Encryption]]:
        if False:
            return 10
        encryption_obj = None
        stream: IOBase
        if isinstance(fileobj, (str, Path)):
            with FileIO(fileobj, 'rb') as f:
                stream = BytesIO(f.read())
        elif isinstance(fileobj, PdfReader):
            if fileobj._encryption:
                encryption_obj = fileobj._encryption
            orig_tell = fileobj.stream.tell()
            fileobj.stream.seek(0)
            stream = BytesIO(fileobj.stream.read())
            fileobj.stream.seek(orig_tell)
        elif hasattr(fileobj, 'seek') and hasattr(fileobj, 'read'):
            fileobj.seek(0)
            filecontent = fileobj.read()
            stream = BytesIO(filecontent)
        else:
            raise NotImplementedError('PdfMerger.merge requires an object that PdfReader can parse. Typically, that is a Path or a string representing a Path, a file object, or an object implementing .seek and .read. Passing a PdfReader directly works as well.')
        return (stream, encryption_obj)

    def append(self, fileobj: Union[StrByteType, PdfReader, Path], outline_item: Union[str, None, PageRange, Tuple[int, int], Tuple[int, int, int], List[int]]=None, pages: Union[None, PageRange, Tuple[int, int], Tuple[int, int, int], List[int], List[PageObject]]=None, import_outline: bool=True, excluded_fields: Optional[Union[List[str], Tuple[str, ...]]]=None) -> None:
        if False:
            print('Hello World!')
        "\n        Identical to the :meth:`merge()<merge>` method, but assumes you want to\n        concatenate all pages onto the end of the file instead of specifying a\n        position.\n\n        Args:\n            fileobj: A File Object or an object that supports the standard\n                read and seek methods similar to a File Object. Could also be a\n                string representing a path to a PDF file.\n            outline_item: Optionally, you may specify a string to build an\n                outline (aka 'bookmark') to identify the beginning of the\n                included file.\n            pages: Can be a :class:`PageRange<pypdf.pagerange.PageRange>`\n                or a ``(start, stop[, step])`` tuple\n                or a list of pages to be processed\n                to merge only the specified range of pages from the source\n                document into the output document.\n            import_outline: You may prevent the source document's\n                outline (collection of outline items, previously referred to as\n                'bookmarks') from being imported by specifying this as ``False``.\n            excluded_fields: Provide the list of fields/keys to be ignored\n                if ``/Annots`` is part of the list, the annotation will be ignored\n                if ``/B`` is part of the list, the articles will be ignored\n        "
        if excluded_fields is None:
            excluded_fields = ()
        if isinstance(outline_item, (tuple, list, PageRange)):
            if isinstance(pages, bool):
                if not isinstance(import_outline, bool):
                    excluded_fields = import_outline
                import_outline = pages
            pages = outline_item
            self.merge(None, fileobj, None, pages, import_outline, excluded_fields)
        else:
            self.merge(None, fileobj, outline_item, pages, import_outline, excluded_fields)

    @deprecation_bookmark(bookmark='outline_item', import_bookmarks='import_outline')
    def merge(self, position: Optional[int], fileobj: Union[Path, StrByteType, PdfReader], outline_item: Optional[str]=None, pages: Optional[Union[PageRangeSpec, List[PageObject]]]=None, import_outline: bool=True, excluded_fields: Optional[Union[List[str], Tuple[str, ...]]]=()) -> None:
        if False:
            print('Hello World!')
        "\n        Merge the pages from the given file into the output file at the\n        specified page number.\n\n        Args:\n            position: The *page number* to insert this file. File will\n                be inserted after the given number.\n            fileobj: A File Object or an object that supports the standard\n                read and seek methods similar to a File Object. Could also be a\n                string representing a path to a PDF file.\n            outline_item: Optionally, you may specify a string to build an outline\n                (aka 'bookmark') to identify the\n                beginning of the included file.\n            pages: can be a :class:`PageRange<pypdf.pagerange.PageRange>`\n                or a ``(start, stop[, step])`` tuple\n                or a list of pages to be processed\n                to merge only the specified range of pages from the source\n                document into the output document.\n            import_outline: You may prevent the source document's\n                outline (collection of outline items, previously referred to as\n                'bookmarks') from being imported by specifying this as ``False``.\n            excluded_fields: provide the list of fields/keys to be ignored\n                if ``/Annots`` is part of the list, the annotation will be ignored\n                if ``/B`` is part of the list, the articles will be ignored\n\n        Raises:\n            TypeError: The pages attribute is not configured properly\n        "
        if isinstance(fileobj, PdfReader):
            reader = fileobj
        else:
            (stream, encryption_obj) = self._create_stream(fileobj)
            reader = PdfReader(stream, strict=False)
        if excluded_fields is None:
            excluded_fields = ()
        if pages is None:
            pages = list(range(len(reader.pages)))
        elif isinstance(pages, PageRange):
            pages = list(range(*pages.indices(len(reader.pages))))
        elif isinstance(pages, list):
            pass
        elif isinstance(pages, tuple) and len(pages) <= 3:
            pages = list(range(*pages))
        elif not isinstance(pages, tuple):
            raise TypeError('"pages" must be a tuple of (start, stop[, step]) or a list')
        srcpages = {}
        for page in pages:
            if isinstance(page, PageObject):
                pg = page
            else:
                pg = reader.pages[page]
            assert pg.indirect_reference is not None
            if position is None:
                srcpages[pg.indirect_reference.idnum] = self.add_page(pg, list(excluded_fields) + [1, '/B', 1, '/Annots'])
            else:
                srcpages[pg.indirect_reference.idnum] = self.insert_page(pg, position, list(excluded_fields) + [1, '/B', 1, '/Annots'])
                position += 1
            srcpages[pg.indirect_reference.idnum].original_page = pg
        reader._namedDests = reader.named_destinations
        for dest in reader._namedDests.values():
            arr = dest.dest_array
            if '/Names' in self._root_object and dest['/Title'] in cast(List[Any], cast(DictionaryObject, cast(DictionaryObject, self._root_object['/Names'])['/Dests'])['/Names']):
                pass
            elif isinstance(dest['/Page'], NullObject):
                pass
            elif isinstance(dest['/Page'], int):
                p = reader.pages[dest['/Page']]
                assert p.indirect_reference is not None
                try:
                    arr[NumberObject(0)] = NumberObject(srcpages[p.indirect_reference.idnum].page_number)
                    self.add_named_destination_array(dest['/Title'], arr)
                except KeyError:
                    pass
            elif dest['/Page'].indirect_reference.idnum in srcpages:
                arr[NumberObject(0)] = srcpages[dest['/Page'].indirect_reference.idnum].indirect_reference
                self.add_named_destination_array(dest['/Title'], arr)
        outline_item_typ: TreeObject
        if outline_item is not None:
            outline_item_typ = cast('TreeObject', self.add_outline_item(TextStringObject(outline_item), next(iter(srcpages.values())).indirect_reference, fit=PAGE_FIT).get_object())
        else:
            outline_item_typ = self.get_outline_root()
        _ro = cast('DictionaryObject', reader.trailer[TK.ROOT])
        if import_outline and CO.OUTLINES in _ro:
            outline = self._get_filtered_outline(_ro.get(CO.OUTLINES, None), srcpages, reader)
            self._insert_filtered_outline(outline, outline_item_typ, None)
        if '/Annots' not in excluded_fields:
            for pag in srcpages.values():
                lst = self._insert_filtered_annotations(pag.original_page.get('/Annots', ()), pag, srcpages, reader)
                if len(lst) > 0:
                    pag[NameObject('/Annots')] = lst
                self.clean_page(pag)
        if '/AcroForm' in _ro and _ro['/AcroForm'] is not None:
            if '/AcroForm' not in self._root_object:
                self._root_object[NameObject('/AcroForm')] = self._add_object(cast(DictionaryObject, cast(DictionaryObject, reader.trailer['/Root'])['/AcroForm']).clone(self, False, ('/Fields',)))
                arr = ArrayObject()
            else:
                arr = cast(ArrayObject, cast(DictionaryObject, self._root_object['/AcroForm'])['/Fields'])
            trslat = self._id_translated[id(reader)]
            try:
                for f in reader.trailer['/Root']['/AcroForm']['/Fields']:
                    try:
                        ind = IndirectObject(trslat[f.idnum], 0, self)
                        if ind not in arr:
                            arr.append(ind)
                    except KeyError:
                        pass
            except KeyError:
                arr = self._add_object(ArrayObject())
            cast(DictionaryObject, self._root_object['/AcroForm'])[NameObject('/Fields')] = arr
        if '/B' not in excluded_fields:
            self.add_filtered_articles('', srcpages, reader)

    def _add_articles_thread(self, thread: DictionaryObject, pages: Dict[int, PageObject], reader: PdfReader) -> IndirectObject:
        if False:
            print('Hello World!')
        '\n        Clone the thread with only the applicable articles.\n\n        Args:\n            thread:\n            pages:\n            reader:\n\n        Returns:\n            The added thread as an indirect reference\n        '
        nthread = thread.clone(self, force_duplicate=True, ignore_fields=('/F',))
        self.threads.append(nthread.indirect_reference)
        first_article = cast('DictionaryObject', thread['/F'])
        current_article: Optional[DictionaryObject] = first_article
        new_article: Optional[DictionaryObject] = None
        while current_article is not None:
            pag = self._get_cloned_page(cast('PageObject', current_article['/P']), pages, reader)
            if pag is not None:
                if new_article is None:
                    new_article = cast('DictionaryObject', self._add_object(DictionaryObject()).get_object())
                    new_first = new_article
                    nthread[NameObject('/F')] = new_article.indirect_reference
                else:
                    new_article2 = cast('DictionaryObject', self._add_object(DictionaryObject({NameObject('/V'): new_article.indirect_reference})).get_object())
                    new_article[NameObject('/N')] = new_article2.indirect_reference
                    new_article = new_article2
                new_article[NameObject('/P')] = pag
                new_article[NameObject('/T')] = nthread.indirect_reference
                new_article[NameObject('/R')] = current_article['/R']
                pag_obj = cast('PageObject', pag.get_object())
                if '/B' not in pag_obj:
                    pag_obj[NameObject('/B')] = ArrayObject()
                cast('ArrayObject', pag_obj['/B']).append(new_article.indirect_reference)
            current_article = cast('DictionaryObject', current_article['/N'])
            if current_article == first_article:
                new_article[NameObject('/N')] = new_first.indirect_reference
                new_first[NameObject('/V')] = new_article.indirect_reference
                current_article = None
        assert nthread.indirect_reference is not None
        return nthread.indirect_reference

    def add_filtered_articles(self, fltr: Union[Pattern[Any], str], pages: Dict[int, PageObject], reader: PdfReader) -> None:
        if False:
            while True:
                i = 10
        '\n        Add articles matching the defined criteria.\n\n        Args:\n            fltr:\n            pages:\n            reader:\n        '
        if isinstance(fltr, str):
            fltr = re.compile(fltr)
        elif not isinstance(fltr, Pattern):
            fltr = re.compile('')
        for p in pages.values():
            pp = p.original_page
            for a in pp.get('/B', ()):
                thr = a.get_object().get('/T')
                if thr is None:
                    continue
                else:
                    thr = thr.get_object()
                if thr.indirect_reference.idnum not in self._id_translated[id(reader)] and fltr.search((thr['/I'] if '/I' in thr else {}).get('/Title', '')):
                    self._add_articles_thread(thr, pages, reader)

    def _get_cloned_page(self, page: Union[None, int, IndirectObject, PageObject, NullObject], pages: Dict[int, PageObject], reader: PdfReader) -> Optional[IndirectObject]:
        if False:
            while True:
                i = 10
        if isinstance(page, NullObject):
            return None
        if isinstance(page, int):
            _i = reader.pages[page].indirect_reference
        elif isinstance(page, DictionaryObject) and page.get('/Type', '') == '/Page':
            _i = page.indirect_reference
        elif isinstance(page, IndirectObject):
            _i = page
        try:
            return pages[_i.idnum].indirect_reference
        except Exception:
            return None

    def _insert_filtered_annotations(self, annots: Union[IndirectObject, List[DictionaryObject]], page: PageObject, pages: Dict[int, PageObject], reader: PdfReader) -> List[Destination]:
        if False:
            i = 10
            return i + 15
        outlist = ArrayObject()
        if isinstance(annots, IndirectObject):
            annots = cast('List[Any]', annots.get_object())
        for an in annots:
            ano = cast('DictionaryObject', an.get_object())
            if ano['/Subtype'] != '/Link' or '/A' not in ano or cast('DictionaryObject', ano['/A'])['/S'] != '/GoTo' or ('/Dest' in ano):
                if '/Dest' not in ano:
                    outlist.append(self._add_object(ano.clone(self)))
                else:
                    d = ano['/Dest']
                    if isinstance(d, str):
                        if str(d) in self.get_named_dest_root():
                            outlist.append(ano.clone(self).indirect_reference)
                    else:
                        d = cast('ArrayObject', d)
                        p = self._get_cloned_page(d[0], pages, reader)
                        if p is not None:
                            anc = ano.clone(self, ignore_fields=('/Dest',))
                            anc[NameObject('/Dest')] = ArrayObject([p] + d[1:])
                            outlist.append(self._add_object(anc))
            else:
                d = cast('DictionaryObject', ano['/A'])['/D']
                if isinstance(d, str):
                    if str(d) in self.get_named_dest_root():
                        outlist.append(ano.clone(self).indirect_reference)
                else:
                    d = cast('ArrayObject', d)
                    p = self._get_cloned_page(d[0], pages, reader)
                    if p is not None:
                        anc = ano.clone(self, ignore_fields=('/D',))
                        cast('DictionaryObject', anc['/A'])[NameObject('/D')] = ArrayObject([p] + d[1:])
                        outlist.append(self._add_object(anc))
        return outlist

    def _get_filtered_outline(self, node: Any, pages: Dict[int, PageObject], reader: PdfReader) -> List[Destination]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Extract outline item entries that are part of the specified page set.\n\n        Args:\n            node:\n            pages:\n            reader:\n\n        Returns:\n            A list of destination objects.\n        '
        new_outline = []
        if node is None:
            node = NullObject()
        node = node.get_object()
        if isinstance(node, NullObject):
            node = DictionaryObject()
        if node.get('/Type', '') == '/Outlines' or '/Title' not in node:
            node = node.get('/First', None)
            if node is not None:
                node = node.get_object()
                new_outline += self._get_filtered_outline(node, pages, reader)
        else:
            v: Union[None, IndirectObject, NullObject]
            while node is not None:
                node = node.get_object()
                o = cast('Destination', reader._build_outline_item(node))
                v = self._get_cloned_page(cast('PageObject', o['/Page']), pages, reader)
                if v is None:
                    v = NullObject()
                o[NameObject('/Page')] = v
                if '/First' in node:
                    o.childs = self._get_filtered_outline(node['/First'], pages, reader)
                else:
                    o.childs = []
                if not isinstance(o['/Page'], NullObject) or len(o.childs) > 0:
                    new_outline.append(o)
                node = node.get('/Next', None)
        return new_outline

    def _clone_outline(self, dest: Destination) -> TreeObject:
        if False:
            for i in range(10):
                print('nop')
        n_ol = TreeObject()
        self._add_object(n_ol)
        n_ol[NameObject('/Title')] = TextStringObject(dest['/Title'])
        if not isinstance(dest['/Page'], NullObject):
            if dest.node is not None and '/A' in dest.node:
                n_ol[NameObject('/A')] = dest.node['/A'].clone(self)
            else:
                n_ol[NameObject('/Dest')] = dest.dest_array
        if dest.node is not None:
            n_ol[NameObject('/F')] = NumberObject(dest.node.get('/F', 0))
            n_ol[NameObject('/C')] = ArrayObject(dest.node.get('/C', [FloatObject(0.0), FloatObject(0.0), FloatObject(0.0)]))
        return n_ol

    def _insert_filtered_outline(self, outlines: List[Destination], parent: Union[TreeObject, IndirectObject], before: Union[None, TreeObject, IndirectObject]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        for dest in outlines:
            if dest.get('/Type', '') == '/Outlines' or '/Title' not in dest:
                np = parent
            else:
                np = self._clone_outline(dest)
                cast(TreeObject, parent.get_object()).insert_child(np, before, self)
            self._insert_filtered_outline(dest.childs, np, None)

    def close(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'To match the functions from Merger.'
        return

    def find_outline_item(self, outline_item: Dict[str, Any], root: Optional[OutlineType]=None) -> Optional[List[int]]:
        if False:
            return 10
        if root is None:
            o = self.get_outline_root()
        else:
            o = cast('TreeObject', root)
        i = 0
        while o is not None:
            if o.indirect_reference == outline_item or o.get('/Title', None) == outline_item:
                return [i]
            elif '/First' in o:
                res = self.find_outline_item(outline_item, cast(OutlineType, o['/First']))
                if res:
                    return ([i] if '/Title' in o else []) + res
            if '/Next' in o:
                i += 1
                o = cast(TreeObject, o['/Next'])
            else:
                return None

    @deprecation_bookmark(bookmark='outline_item')
    def find_bookmark(self, outline_item: Dict[str, Any], root: Optional[OutlineType]=None) -> Optional[List[int]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        .. deprecated:: 2.9.0\n            Use :meth:`find_outline_item` instead.\n        '
        return self.find_outline_item(outline_item, root)

    def reset_translation(self, reader: Union[None, PdfReader, IndirectObject]=None) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Reset the translation table between reader and the writer object.\n\n        Late cloning will create new independent objects.\n\n        Args:\n            reader: PdfReader or IndirectObject refering a PdfReader object.\n                if set to None or omitted, all tables will be reset.\n        '
        if reader is None:
            self._id_translated = {}
        elif isinstance(reader, PdfReader):
            try:
                del self._id_translated[id(reader)]
            except Exception:
                pass
        elif isinstance(reader, IndirectObject):
            try:
                del self._id_translated[id(reader.pdf)]
            except Exception:
                pass
        else:
            raise Exception('invalid parameter {reader}')

    def set_page_label(self, page_index_from: int, page_index_to: int, style: Optional[PageLabelStyle]=None, prefix: Optional[str]=None, start: Optional[int]=0) -> None:
        if False:
            print('Hello World!')
        "\n        Set a page label to a range of pages.\n\n        Page indexes must be given starting from 0.\n        Labels must have a style, a prefix or both.\n        If to a range is not assigned any page label a decimal label starting from 1 is applied.\n\n        Args:\n            page_index_from: page index of the beginning of the range starting from 0\n            page_index_to: page index of the beginning of the range starting from 0\n            style:  The numbering style to be used for the numeric portion of each page label:\n                        '/D' Decimal arabic numerals\n                        '/R' Uppercase roman numerals\n                        '/r' Lowercase roman numerals\n                        '/A' Uppercase letters (A to Z for the first 26 pages,\n                             AA to ZZ for the next 26, and so on)\n                        '/a' Lowercase letters (a to z for the first 26 pages,\n                             aa to zz for the next 26, and so on)\n            prefix: The label prefix for page labels in this range.\n            start:  The value of the numeric portion for the first page label\n                    in the range.\n                    Subsequent pages are numbered sequentially from this value,\n                    which must be greater than or equal to 1.\n                    Default value: 1.\n        "
        if style is None and prefix is None:
            raise ValueError('at least one between style and prefix must be given')
        if page_index_from < 0:
            raise ValueError('page_index_from must be equal or greater then 0')
        if page_index_to < page_index_from:
            raise ValueError('page_index_to must be equal or greater then page_index_from')
        if page_index_to >= len(self.pages):
            raise ValueError('page_index_to exceeds number of pages')
        if start is not None and start != 0 and (start < 1):
            raise ValueError('if given, start must be equal or greater than one')
        self._set_page_label(page_index_from, page_index_to, style, prefix, start)

    def _set_page_label(self, page_index_from: int, page_index_to: int, style: Optional[PageLabelStyle]=None, prefix: Optional[str]=None, start: Optional[int]=0) -> None:
        if False:
            return 10
        '\n        Set a page label to a range of pages.\n\n        Page indexes must be given\n        starting from 0. Labels must have a style, a prefix or both. If to a\n        range is not assigned any page label a decimal label starting from 1 is\n        applied.\n\n        Args:\n            page_index_from: page index of the beginning of the range starting from 0\n            page_index_to: page index of the beginning of the range starting from 0\n            style:  The numbering style to be used for the numeric portion of each page label:\n                        /D Decimal arabic numerals\n                        /R Uppercase roman numerals\n                        /r Lowercase roman numerals\n                        /A Uppercase letters (A to Z for the first 26 pages,\n                           AA to ZZ for the next 26, and so on)\n                        /a Lowercase letters (a to z for the first 26 pages,\n                           aa to zz for the next 26, and so on)\n            prefix: The label prefix for page labels in this range.\n            start:  The value of the numeric portion for the first page label\n                    in the range.\n                    Subsequent pages are numbered sequentially from this value,\n                    which must be greater than or equal to 1. Default value: 1.\n        '
        default_page_label = DictionaryObject()
        default_page_label[NameObject('/S')] = NameObject('/D')
        new_page_label = DictionaryObject()
        if style is not None:
            new_page_label[NameObject('/S')] = NameObject(style)
        if prefix is not None:
            new_page_label[NameObject('/P')] = TextStringObject(prefix)
        if start != 0:
            new_page_label[NameObject('/St')] = NumberObject(start)
        if NameObject(CatalogDictionary.PAGE_LABELS) not in self._root_object:
            nums = ArrayObject()
            nums_insert(NumberObject(0), default_page_label, nums)
            page_labels = TreeObject()
            page_labels[NameObject('/Nums')] = nums
            self._root_object[NameObject(CatalogDictionary.PAGE_LABELS)] = page_labels
        page_labels = cast(TreeObject, self._root_object[NameObject(CatalogDictionary.PAGE_LABELS)])
        nums = cast(ArrayObject, page_labels[NameObject('/Nums')])
        nums_insert(NumberObject(page_index_from), new_page_label, nums)
        nums_clear_range(NumberObject(page_index_from), page_index_to, nums)
        (next_label_pos, *_) = nums_next(NumberObject(page_index_from), nums)
        if next_label_pos != page_index_to + 1 and page_index_to + 1 < len(self.pages):
            nums_insert(NumberObject(page_index_to + 1), default_page_label, nums)
        page_labels[NameObject('/Nums')] = nums
        self._root_object[NameObject(CatalogDictionary.PAGE_LABELS)] = page_labels

def _pdf_objectify(obj: Union[Dict[str, Any], str, int, List[Any]]) -> PdfObject:
    if False:
        while True:
            i = 10
    if isinstance(obj, PdfObject):
        return obj
    if isinstance(obj, dict):
        to_add = DictionaryObject()
        for (key, value) in obj.items():
            name_key = NameObject(key)
            casted_value = _pdf_objectify(value)
            to_add[name_key] = casted_value
        return to_add
    elif isinstance(obj, list):
        return ArrayObject((_pdf_objectify(el) for el in obj))
    elif isinstance(obj, str):
        if obj.startswith('/'):
            return NameObject(obj)
        else:
            return TextStringObject(obj)
    elif isinstance(obj, (int, float)):
        return FloatObject(obj)
    else:
        raise NotImplementedError(f'type(obj)={type(obj)} could not be casted to PdfObject')

def _create_outline_item(action_ref: Union[None, IndirectObject], title: str, color: Union[Tuple[float, float, float], str, None], italic: bool, bold: bool) -> TreeObject:
    if False:
        i = 10
        return i + 15
    outline_item = TreeObject()
    if action_ref is not None:
        outline_item[NameObject('/A')] = action_ref
    outline_item.update({NameObject('/Title'): create_string_object(title)})
    if color:
        if isinstance(color, str):
            color = hex_to_rgb(color)
        outline_item.update({NameObject('/C'): ArrayObject([FloatObject(c) for c in color])})
    if italic or bold:
        format_flag = 0
        if italic:
            format_flag += 1
        if bold:
            format_flag += 2
        outline_item.update({NameObject('/F'): NumberObject(format_flag)})
    return outline_item

class PdfFileWriter(PdfWriter):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        deprecation_with_replacement('PdfFileWriter', 'PdfWriter', '3.0.0')
        super().__init__(*args, **kwargs)