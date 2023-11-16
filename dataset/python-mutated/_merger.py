import warnings
from io import BytesIO, FileIO, IOBase
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union, cast
from ._encryption import Encryption
from ._page import PageObject
from ._reader import PdfReader
from ._utils import StrByteType, deprecate_with_replacement, deprecation_bookmark, deprecation_with_replacement, str_
from ._writer import PdfWriter
from .constants import GoToActionArguments, TypArguments, TypFitArguments
from .constants import PagesAttributes as PA
from .generic import PAGE_FIT, ArrayObject, Destination, DictionaryObject, Fit, FloatObject, IndirectObject, NameObject, NullObject, NumberObject, OutlineItem, TextStringObject, TreeObject
from .pagerange import PageRange, PageRangeSpec
from .types import FitType, LayoutType, OutlineType, PagemodeType, ZoomArgType
ERR_CLOSED_WRITER = 'close() was called and thus the writer cannot be used anymore'

class _MergedPage:
    """Collect necessary information on each page that is being merged."""

    def __init__(self, pagedata: PageObject, src: PdfReader, id: int) -> None:
        if False:
            i = 10
            return i + 15
        self.src = src
        self.pagedata = pagedata
        self.out_pagedata = None
        self.id = id

class PdfMerger:
    """
    Use :class:`PdfWriter` instead.

    .. deprecated:: 5.0.0
    """

    @deprecation_bookmark(bookmarks='outline')
    def __init__(self, strict: bool=False, fileobj: Union[Path, StrByteType]='') -> None:
        if False:
            print('Hello World!')
        deprecate_with_replacement('PdfMerger', 'PdfWriter', '5.0.0')
        self.inputs: List[Tuple[Any, PdfReader]] = []
        self.pages: List[Any] = []
        self.output: Optional[PdfWriter] = PdfWriter()
        self.outline: OutlineType = []
        self.named_dests: List[Any] = []
        self.id_count = 0
        self.fileobj = fileobj
        self.strict = strict

    def __enter__(self) -> 'PdfMerger':
        if False:
            i = 10
            return i + 15
        deprecate_with_replacement('PdfMerger', 'PdfWriter', '5.0.0')
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Write to the fileobj and close the merger.'
        if self.fileobj:
            self.write(self.fileobj)
        self.close()

    @deprecation_bookmark(bookmark='outline_item', import_bookmarks='import_outline')
    def merge(self, page_number: Optional[int]=None, fileobj: Union[None, Path, StrByteType, PdfReader]=None, outline_item: Optional[str]=None, pages: Optional[PageRangeSpec]=None, import_outline: bool=True, position: Optional[int]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Merge the pages from the given file into the output file at the\n        specified page number.\n\n        Args:\n            page_number: The *page number* to insert this file. File will\n                be inserted after the given number.\n            fileobj: A File Object or an object that supports the standard\n                read and seek methods similar to a File Object. Could also be a\n                string representing a path to a PDF file.\n                None as an argument is deprecated.\n            outline_item: Optionally, you may specify an outline item\n                (previously referred to as a 'bookmark') to be applied at the\n                beginning of the included file by supplying the text of the outline item.\n            pages: can be a :class:`PageRange<pypdf.pagerange.PageRange>`\n                or a ``(start, stop[, step])`` tuple\n                to merge only the specified range of pages from the source\n                document into the output document.\n                Can also be a list of pages to merge.\n           import_outline: You may prevent the source document's\n                outline (collection of outline items, previously referred to as\n                'bookmarks') from being imported by specifying this as ``False``.\n        "
        if position is not None:
            if page_number is None:
                page_number = position
                old_term = 'position'
                new_term = 'page_number'
                warnings.warn(f'{old_term} is deprecated as an argument and will be removed in pypdf=4.0.0. Use {new_term} instead', DeprecationWarning)
            else:
                raise ValueError('The argument position of merge is deprecated. Use page_number only.')
        if page_number is None:
            raise ValueError('page_number may not be None')
        if fileobj is None:
            raise ValueError('fileobj may not be None')
        (stream, encryption_obj) = self._create_stream(fileobj)
        reader = PdfReader(stream, strict=self.strict)
        self.inputs.append((stream, reader))
        if encryption_obj is not None:
            reader._encryption = encryption_obj
        if pages is None:
            pages = (0, len(reader.pages))
        elif isinstance(pages, PageRange):
            pages = pages.indices(len(reader.pages))
        elif isinstance(pages, list):
            pass
        elif not isinstance(pages, tuple):
            raise TypeError('"pages" must be a tuple of (start, stop[, step])')
        srcpages = []
        outline = []
        if import_outline:
            outline = reader.outline
            outline = self._trim_outline(reader, outline, pages)
        if outline_item:
            outline_item_typ = OutlineItem(TextStringObject(outline_item), NumberObject(self.id_count), Fit.fit())
            self.outline += [outline_item_typ, outline]
        else:
            self.outline += outline
        dests = reader.named_destinations
        trimmed_dests = self._trim_dests(reader, dests, pages)
        self.named_dests += trimmed_dests
        for i in range(*pages):
            page = reader.pages[i]
            id = self.id_count
            self.id_count += 1
            mp = _MergedPage(page, reader, id)
            srcpages.append(mp)
        self._associate_dests_to_pages(srcpages)
        self._associate_outline_items_to_pages(srcpages)
        self.pages[page_number:page_number] = srcpages

    def _create_stream(self, fileobj: Union[Path, StrByteType, PdfReader]) -> Tuple[IOBase, Optional[Encryption]]:
        if False:
            return 10
        encryption_obj = None
        stream: IOBase
        if isinstance(fileobj, (str, Path)):
            stream = FileIO(fileobj, 'rb')
        elif isinstance(fileobj, PdfReader):
            if fileobj._encryption:
                encryption_obj = fileobj._encryption
            orig_tell = fileobj.stream.tell()
            fileobj.stream.seek(0)
            stream = BytesIO(fileobj.stream.read())
            fileobj.stream.seek(orig_tell)
        elif hasattr(fileobj, 'seek') and hasattr(fileobj, 'read'):
            fileobj.seek(0)
            file_content = fileobj.read()
            stream = BytesIO(file_content)
        else:
            raise NotImplementedError('PdfMerger.merge requires an object that PdfReader can parse. Typically, that is a Path or a string representing a Path, a file object, or an object implementing .seek and .read. Passing a PdfReader directly works as well.')
        return (stream, encryption_obj)

    @deprecation_bookmark(bookmark='outline_item', import_bookmarks='import_outline')
    def append(self, fileobj: Union[StrByteType, PdfReader, Path], outline_item: Optional[str]=None, pages: Union[None, PageRange, Tuple[int, int], Tuple[int, int, int], List[int]]=None, import_outline: bool=True) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Identical to the :meth:`merge()<merge>` method, but assumes you want to\n        concatenate all pages onto the end of the file instead of specifying a\n        position.\n\n        Args:\n            fileobj: A File Object or an object that supports the standard\n                read and seek methods similar to a File Object. Could also be a\n                string representing a path to a PDF file.\n            outline_item: Optionally, you may specify an outline item\n                (previously referred to as a 'bookmark') to be applied at the\n                beginning of the included file by supplying the text of the outline item.\n            pages: can be a :class:`PageRange<pypdf.pagerange.PageRange>`\n                or a ``(start, stop[, step])`` tuple\n                to merge only the specified range of pages from the source\n                document into the output document.\n                Can also be a list of pages to append.\n            import_outline: You may prevent the source document's\n                outline (collection of outline items, previously referred to as\n                'bookmarks') from being imported by specifying this as ``False``.\n        "
        self.merge(len(self.pages), fileobj, outline_item, pages, import_outline)

    def write(self, fileobj: Union[Path, StrByteType]) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Write all data that has been merged to the given output file.\n\n        Args:\n            fileobj: Output file. Can be a filename or any kind of\n                file-like object.\n        '
        if self.output is None:
            raise RuntimeError(ERR_CLOSED_WRITER)
        for page in self.pages:
            self.output.add_page(page.pagedata)
            pages_obj = cast(Dict[str, Any], self.output._pages.get_object())
            page.out_pagedata = self.output.get_reference(pages_obj[PA.KIDS][-1].get_object())
        self._write_dests()
        self._write_outline()
        (my_file, ret_fileobj) = self.output.write(fileobj)
        if my_file:
            ret_fileobj.close()

    def close(self) -> None:
        if False:
            print('Hello World!')
        'Shut all file descriptors (input and output) and clear all memory usage.'
        self.pages = []
        for (fo, _reader) in self.inputs:
            fo.close()
        self.inputs = []
        self.output = None

    def add_metadata(self, infos: Dict[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Add custom metadata to the output.\n\n        Args:\n            infos: a Python dictionary where each key is a field\n                and each value is your new metadata.\n                An example is ``{'/Title': 'My title'}``\n        "
        if self.output is None:
            raise RuntimeError(ERR_CLOSED_WRITER)
        self.output.add_metadata(infos)

    def addMetadata(self, infos: Dict[str, Any]) -> None:
        if False:
            while True:
                i = 10
        '\n        Use :meth:`add_metadata` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('addMetadata', 'add_metadata')
        self.add_metadata(infos)

    def setPageLayout(self, layout: LayoutType) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Use :meth:`set_page_layout` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('setPageLayout', 'set_page_layout')
        self.set_page_layout(layout)

    def set_page_layout(self, layout: LayoutType) -> None:
        if False:
            while True:
                i = 10
        '\n        Set the page layout.\n\n        Args:\n            layout: The page layout to be used\n\n        .. list-table:: Valid ``layout`` arguments\n           :widths: 50 200\n\n           * - /NoLayout\n             - Layout explicitly not specified\n           * - /SinglePage\n             - Show one page at a time\n           * - /OneColumn\n             - Show one column at a time\n           * - /TwoColumnLeft\n             - Show pages in two columns, odd-numbered pages on the left\n           * - /TwoColumnRight\n             - Show pages in two columns, odd-numbered pages on the right\n           * - /TwoPageLeft\n             - Show two pages at a time, odd-numbered pages on the left\n           * - /TwoPageRight\n             - Show two pages at a time, odd-numbered pages on the right\n        '
        if self.output is None:
            raise RuntimeError(ERR_CLOSED_WRITER)
        self.output._set_page_layout(layout)

    def setPageMode(self, mode: PagemodeType) -> None:
        if False:
            print('Hello World!')
        '\n        Use :meth:`set_page_mode` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('setPageMode', 'set_page_mode', '3.0.0')
        self.set_page_mode(mode)

    def set_page_mode(self, mode: PagemodeType) -> None:
        if False:
            print('Hello World!')
        '\n        Set the page mode.\n\n        Args:\n            mode: The page mode to use.\n\n        .. list-table:: Valid ``mode`` arguments\n           :widths: 50 200\n\n           * - /UseNone\n             - Do not show outline or thumbnails panels\n           * - /UseOutlines\n             - Show outline (aka bookmarks) panel\n           * - /UseThumbs\n             - Show page thumbnails panel\n           * - /FullScreen\n             - Fullscreen view\n           * - /UseOC\n             - Show Optional Content Group (OCG) panel\n           * - /UseAttachments\n             - Show attachments panel\n        '
        if self.output is None:
            raise RuntimeError(ERR_CLOSED_WRITER)
        self.output.set_page_mode(mode)

    def _trim_dests(self, pdf: PdfReader, dests: Dict[str, Dict[str, Any]], pages: Union[Tuple[int, int], Tuple[int, int, int], List[int]]) -> List[Dict[str, Any]]:
        if False:
            print('Hello World!')
        '\n        Remove named destinations that are not a part of the specified page set.\n\n        Args:\n            pdf:\n            dests:\n            pages:\n        '
        new_dests = []
        lst = pages if isinstance(pages, list) else list(range(*pages))
        for (key, obj) in dests.items():
            for j in lst:
                if pdf.pages[j].get_object() == obj['/Page'].get_object():
                    obj[NameObject('/Page')] = obj['/Page'].get_object()
                    assert str_(key) == str_(obj['/Title'])
                    new_dests.append(obj)
                    break
        return new_dests

    def _trim_outline(self, pdf: PdfReader, outline: OutlineType, pages: Union[Tuple[int, int], Tuple[int, int, int], List[int]]) -> OutlineType:
        if False:
            return 10
        '\n        Remove outline item entries that are not a part of the specified page set.\n\n        Args:\n            pdf:\n            outline:\n            pages:\n\n        Returns:\n            An outline type\n        '
        new_outline = []
        prev_header_added = True
        lst = pages if isinstance(pages, list) else list(range(*pages))
        for (i, outline_item) in enumerate(outline):
            if isinstance(outline_item, list):
                sub = self._trim_outline(pdf, outline_item, lst)
                if sub:
                    if not prev_header_added:
                        new_outline.append(outline[i - 1])
                    new_outline.append(sub)
            else:
                prev_header_added = False
                for j in lst:
                    if outline_item['/Page'] is None:
                        continue
                    if pdf.pages[j].get_object() == outline_item['/Page'].get_object():
                        outline_item[NameObject('/Page')] = outline_item['/Page'].get_object()
                        new_outline.append(outline_item)
                        prev_header_added = True
                        break
        return new_outline

    def _write_dests(self) -> None:
        if False:
            return 10
        if self.output is None:
            raise RuntimeError(ERR_CLOSED_WRITER)
        for named_dest in self.named_dests:
            page_index = None
            if '/Page' in named_dest:
                for (page_index, page) in enumerate(self.pages):
                    if page.id == named_dest['/Page']:
                        named_dest[NameObject('/Page')] = page.out_pagedata
                        break
            if page_index is not None:
                self.output.add_named_destination_object(named_dest)

    @deprecation_bookmark(bookmarks='outline')
    def _write_outline(self, outline: Optional[Iterable[OutlineItem]]=None, parent: Optional[TreeObject]=None) -> None:
        if False:
            i = 10
            return i + 15
        if self.output is None:
            raise RuntimeError(ERR_CLOSED_WRITER)
        if outline is None:
            outline = self.outline
        assert outline is not None, 'hint for mypy'
        last_added = None
        for outline_item in outline:
            if isinstance(outline_item, list):
                self._write_outline(outline_item, last_added)
                continue
            page_no = None
            if '/Page' in outline_item:
                for (page_no, page) in enumerate(self.pages):
                    if page.id == outline_item['/Page']:
                        self._write_outline_item_on_page(outline_item, page)
                        break
            if page_no is not None:
                del outline_item['/Page'], outline_item['/Type']
                last_added = self.output.add_outline_item_dict(outline_item, parent)

    @deprecation_bookmark(bookmark='outline_item')
    def _write_outline_item_on_page(self, outline_item: Union[OutlineItem, Destination], page: _MergedPage) -> None:
        if False:
            print('Hello World!')
        oi_type = cast(str, outline_item['/Type'])
        args = [NumberObject(page.id), NameObject(oi_type)]
        fit2arg_keys: Dict[str, Tuple[str, ...]] = {TypFitArguments.FIT_H: (TypArguments.TOP,), TypFitArguments.FIT_BH: (TypArguments.TOP,), TypFitArguments.FIT_V: (TypArguments.LEFT,), TypFitArguments.FIT_BV: (TypArguments.LEFT,), TypFitArguments.XYZ: (TypArguments.LEFT, TypArguments.TOP, '/Zoom'), TypFitArguments.FIT_R: (TypArguments.LEFT, TypArguments.BOTTOM, TypArguments.RIGHT, TypArguments.TOP)}
        for arg_key in fit2arg_keys.get(oi_type, ()):
            if arg_key in outline_item and (not isinstance(outline_item[arg_key], NullObject)):
                args.append(FloatObject(outline_item[arg_key]))
            else:
                args.append(FloatObject(0))
            del outline_item[arg_key]
        outline_item[NameObject('/A')] = DictionaryObject({NameObject(GoToActionArguments.S): NameObject('/GoTo'), NameObject(GoToActionArguments.D): ArrayObject(args)})

    def _associate_dests_to_pages(self, pages: List[_MergedPage]) -> None:
        if False:
            while True:
                i = 10
        for named_dest in self.named_dests:
            page_index = None
            np = named_dest['/Page']
            if isinstance(np, NumberObject):
                continue
            for page in pages:
                if np.get_object() == page.pagedata.get_object():
                    page_index = page.id
            if page_index is None:
                raise ValueError(f"Unresolved named destination '{named_dest['/Title']}'")
            named_dest[NameObject('/Page')] = NumberObject(page_index)

    @deprecation_bookmark(bookmarks='outline')
    def _associate_outline_items_to_pages(self, pages: List[_MergedPage], outline: Optional[Iterable[OutlineItem]]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        if outline is None:
            outline = self.outline
        assert outline is not None, 'hint for mypy'
        for outline_item in outline:
            if isinstance(outline_item, list):
                self._associate_outline_items_to_pages(pages, outline_item)
                continue
            page_index = None
            outline_item_page = outline_item['/Page']
            if isinstance(outline_item_page, NumberObject):
                continue
            for p in pages:
                if outline_item_page.get_object() == p.pagedata.get_object():
                    page_index = p.id
            if page_index is not None:
                outline_item[NameObject('/Page')] = NumberObject(page_index)

    @deprecation_bookmark(bookmark='outline_item')
    def find_outline_item(self, outline_item: Dict[str, Any], root: Optional[OutlineType]=None) -> Optional[List[int]]:
        if False:
            return 10
        if root is None:
            root = self.outline
        for (i, oi_enum) in enumerate(root):
            if isinstance(oi_enum, list):
                res = self.find_outline_item(outline_item, oi_enum)
                if res:
                    return [i] + res
            elif oi_enum == outline_item or cast(Dict[Any, Any], oi_enum['/Title']) == outline_item:
                return [i]
        return None

    @deprecation_bookmark(bookmark='outline_item')
    def find_bookmark(self, outline_item: Dict[str, Any], root: Optional[OutlineType]=None) -> Optional[List[int]]:
        if False:
            while True:
                i = 10
        '\n        .. deprecated:: 2.9.0\n            Use :meth:`find_outline_item` instead.\n        '
        return self.find_outline_item(outline_item, root)

    def add_outline_item(self, title: str, page_number: Optional[int]=None, parent: Union[None, TreeObject, IndirectObject]=None, color: Optional[Tuple[float, float, float]]=None, bold: bool=False, italic: bool=False, fit: Fit=PAGE_FIT, pagenum: Optional[int]=None) -> IndirectObject:
        if False:
            i = 10
            return i + 15
        '\n        Add an outline item (commonly referred to as a "Bookmark") to this PDF file.\n\n        Args:\n            title: Title to use for this outline item.\n            page_number: Page number this outline item will point to.\n            parent: A reference to a parent outline item to create nested\n                outline items.\n            color: Color of the outline item\'s font as a red, green, blue tuple\n                from 0.0 to 1.0\n            bold: Outline item font is bold\n            italic: Outline item font is italic\n            fit: The fit of the destination page.\n        '
        if page_number is not None and pagenum is not None:
            raise ValueError('The argument pagenum of add_outline_item is deprecated. Use page_number only.')
        if pagenum is not None:
            old_term = 'pagenum'
            new_term = 'page_number'
            warnings.warn(f'{old_term} is deprecated as an argument and will be removed in pypdf==4.0.0. Use {new_term} instead', DeprecationWarning)
            page_number = pagenum
        if page_number is None:
            raise ValueError('page_number may not be None')
        writer = self.output
        if writer is None:
            raise RuntimeError(ERR_CLOSED_WRITER)
        return writer.add_outline_item(title, page_number, parent, None, color, bold, italic, fit)

    def addBookmark(self, title: str, pagenum: int, parent: Union[None, TreeObject, IndirectObject]=None, color: Optional[Tuple[float, float, float]]=None, bold: bool=False, italic: bool=False, fit: FitType='/Fit', *args: ZoomArgType) -> IndirectObject:
        if False:
            print('Hello World!')
        '\n        .. deprecated:: 1.28.0\n            Use :meth:`add_outline_item` instead.\n        '
        deprecation_with_replacement('addBookmark', 'add_outline_item', '3.0.0')
        return self.add_outline_item(title, pagenum, parent, color, bold, italic, Fit(fit_type=fit, fit_args=args))

    def add_bookmark(self, title: str, pagenum: int, parent: Union[None, TreeObject, IndirectObject]=None, color: Optional[Tuple[float, float, float]]=None, bold: bool=False, italic: bool=False, fit: FitType='/Fit', *args: ZoomArgType) -> IndirectObject:
        if False:
            print('Hello World!')
        '\n        .. deprecated:: 2.9.0\n            Use :meth:`add_outline_item` instead.\n        '
        deprecation_with_replacement('addBookmark', 'add_outline_item', '3.0.0')
        return self.add_outline_item(title, pagenum, parent, color, bold, italic, Fit(fit_type=fit, fit_args=args))

    def addNamedDestination(self, title: str, pagenum: int) -> None:
        if False:
            return 10
        '\n        .. deprecated:: 1.28.0\n            Use :meth:`add_named_destination` instead.\n        '
        deprecation_with_replacement('addNamedDestination', 'add_named_destination', '3.0.0')
        return self.add_named_destination(title, pagenum)

    def add_named_destination(self, title: str, page_number: Optional[int]=None, pagenum: Optional[int]=None) -> None:
        if False:
            print('Hello World!')
        '\n        Add a destination to the output.\n\n        Args:\n            title: Title to use\n            page_number: Page number this destination points at.\n        '
        if page_number is not None and pagenum is not None:
            raise ValueError('The argument pagenum of add_named_destination is deprecated. Use page_number only.')
        if pagenum is not None:
            old_term = 'pagenum'
            new_term = 'page_number'
            warnings.warn(f'{old_term} is deprecated as an argument and will be removed in pypdf==4.0.0. Use {new_term} instead', DeprecationWarning)
            page_number = pagenum
        if page_number is None:
            raise ValueError('page_number may not be None')
        dest = Destination(TextStringObject(title), NumberObject(page_number), Fit.fit_horizontally(top=826))
        self.named_dests.append(dest)

class PdfFileMerger(PdfMerger):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        deprecation_with_replacement('PdfFileMerger', 'PdfMerger', '3.0.0')
        if 'strict' not in kwargs and len(args) < 1:
            kwargs['strict'] = True
        super().__init__(*args, **kwargs)