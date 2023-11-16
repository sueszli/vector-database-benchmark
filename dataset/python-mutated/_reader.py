import os
import re
import struct
import zlib
from datetime import datetime
from io import BytesIO, UnsupportedOperation
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Union, cast
from ._encryption import Encryption, PasswordType
from ._page import PageObject, _VirtualList
from ._page_labels import index2label as page_index2page_label
from ._utils import StrByteType, StreamType, b_, deprecate_no_replacement, deprecation_no_replacement, deprecation_with_replacement, logger_warning, parse_iso8824_date, read_non_whitespace, read_previous_line, read_until_whitespace, skip_over_comment, skip_over_whitespace
from .constants import CatalogAttributes as CA
from .constants import CatalogDictionary as CD
from .constants import CheckboxRadioButtonAttributes, GoToActionArguments
from .constants import Core as CO
from .constants import DocumentInformationAttributes as DI
from .constants import FieldDictionaryAttributes as FA
from .constants import PageAttributes as PG
from .constants import PagesAttributes as PA
from .constants import TrailerKeys as TK
from .errors import EmptyFileError, FileNotDecryptedError, PdfReadError, PdfStreamError, WrongPasswordError
from .generic import ArrayObject, BooleanObject, ContentStream, DecodedStreamObject, Destination, DictionaryObject, EncodedStreamObject, Field, Fit, FloatObject, IndirectObject, NameObject, NullObject, NumberObject, PdfObject, TextStringObject, TreeObject, ViewerPreferences, read_object
from .types import OutlineType, PagemodeType
from .xmp import XmpInformation

def convert_to_int(d: bytes, size: int) -> Union[int, Tuple[Any, ...]]:
    if False:
        return 10
    if size > 8:
        raise PdfReadError('invalid size in convert_to_int')
    d = b'\x00\x00\x00\x00\x00\x00\x00\x00' + d
    d = d[-8:]
    return struct.unpack('>q', d)[0]

def convertToInt(d: bytes, size: int) -> Union[int, Tuple[Any, ...]]:
    if False:
        while True:
            i = 10
    deprecation_with_replacement('convertToInt', 'convert_to_int')
    return convert_to_int(d, size)

class DocumentInformation(DictionaryObject):
    """
    A class representing the basic document metadata provided in a PDF File.
    This class is accessible through
    :py:class:`PdfReader.metadata<pypdf.PdfReader.metadata>`.

    All text properties of the document metadata have
    *two* properties, eg. author and author_raw. The non-raw property will
    always return a ``TextStringObject``, making it ideal for a case where
    the metadata is being displayed. The raw property can sometimes return
    a ``ByteStringObject``, if pypdf was unable to decode the string's
    text encoding; this requires additional safety in the caller and
    therefore is not as commonly accessed.
    """

    def __init__(self) -> None:
        if False:
            return 10
        DictionaryObject.__init__(self)

    def _get_text(self, key: str) -> Optional[str]:
        if False:
            while True:
                i = 10
        retval = self.get(key, None)
        if isinstance(retval, TextStringObject):
            return retval
        return None

    def getText(self, key: str) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Use the attributes (e.g. :py:attr:`title` / :py:attr:`author`).\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_no_replacement('getText', '3.0.0')
        return self._get_text(key)

    @property
    def title(self) -> Optional[str]:
        if False:
            return 10
        "\n        Read-only property accessing the document's title.\n\n        Returns a ``TextStringObject`` or ``None`` if the title is not\n        specified.\n        "
        return self._get_text(DI.TITLE) or self.get(DI.TITLE).get_object() if self.get(DI.TITLE) else None

    @property
    def title_raw(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        'The "raw" version of title; can return a ``ByteStringObject``.'
        return self.get(DI.TITLE)

    @property
    def author(self) -> Optional[str]:
        if False:
            return 10
        "\n        Read-only property accessing the document's author.\n\n        Returns a ``TextStringObject`` or ``None`` if the author is not\n        specified.\n        "
        return self._get_text(DI.AUTHOR)

    @property
    def author_raw(self) -> Optional[str]:
        if False:
            return 10
        'The "raw" version of author; can return a ``ByteStringObject``.'
        return self.get(DI.AUTHOR)

    @property
    def subject(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        "\n        Read-only property accessing the document's subject.\n\n        Returns a ``TextStringObject`` or ``None`` if the subject is not\n        specified.\n        "
        return self._get_text(DI.SUBJECT)

    @property
    def subject_raw(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        'The "raw" version of subject; can return a ``ByteStringObject``.'
        return self.get(DI.SUBJECT)

    @property
    def creator(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        "\n        Read-only property accessing the document's creator.\n\n        If the document was converted to PDF from another format, this is the\n        name of the application (e.g. OpenOffice) that created the original\n        document from which it was converted. Returns a ``TextStringObject`` or\n        ``None`` if the creator is not specified.\n        "
        return self._get_text(DI.CREATOR)

    @property
    def creator_raw(self) -> Optional[str]:
        if False:
            return 10
        'The "raw" version of creator; can return a ``ByteStringObject``.'
        return self.get(DI.CREATOR)

    @property
    def producer(self) -> Optional[str]:
        if False:
            return 10
        "\n        Read-only property accessing the document's producer.\n\n        If the document was converted to PDF from another format, this is the\n        name of the application (for example, OSX Quartz) that converted it to\n        PDF. Returns a ``TextStringObject`` or ``None`` if the producer is not\n        specified.\n        "
        return self._get_text(DI.PRODUCER)

    @property
    def producer_raw(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        'The "raw" version of producer; can return a ``ByteStringObject``.'
        return self.get(DI.PRODUCER)

    @property
    def creation_date(self) -> Optional[datetime]:
        if False:
            for i in range(10):
                print('nop')
        "Read-only property accessing the document's creation date."
        return parse_iso8824_date(self._get_text(DI.CREATION_DATE))

    @property
    def creation_date_raw(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        The "raw" version of creation date; can return a ``ByteStringObject``.\n\n        Typically in the format ``D:YYYYMMDDhhmmss[+Z-]hh\'mm`` where the suffix\n        is the offset from UTC.\n        '
        return self.get(DI.CREATION_DATE)

    @property
    def modification_date(self) -> Optional[datetime]:
        if False:
            while True:
                i = 10
        "\n        Read-only property accessing the document's modification date.\n\n        The date and time the document was most recently modified.\n        "
        return parse_iso8824_date(self._get_text(DI.MOD_DATE))

    @property
    def modification_date_raw(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        The "raw" version of modification date; can return a\n        ``ByteStringObject``.\n\n        Typically in the format ``D:YYYYMMDDhhmmss[+Z-]hh\'mm`` where the suffix\n        is the offset from UTC.\n        '
        return self.get(DI.MOD_DATE)

class PdfReader:
    """
    Initialize a PdfReader object.

    This operation can take some time, as the PDF stream's cross-reference
    tables are read into memory.

    Args:
        stream: A File object or an object that supports the standard read
            and seek methods similar to a File object. Could also be a
            string representing a path to a PDF file.
        strict: Determines whether user should be warned of all
            problems and also causes some correctable problems to be fatal.
            Defaults to ``False``.
        password: Decrypt PDF file at initialization. If the
            password is None, the file will not be decrypted.
            Defaults to ``None``
    """

    @property
    def viewer_preferences(self) -> Optional[ViewerPreferences]:
        if False:
            return 10
        'Returns the existing ViewerPreferences as an overloaded dictionary.'
        o = cast(DictionaryObject, self.trailer['/Root']).get(CD.VIEWER_PREFERENCES, None)
        if o is None:
            return None
        o = o.get_object()
        if not isinstance(o, ViewerPreferences):
            o = ViewerPreferences(o)
        return o

    def __init__(self, stream: Union[StrByteType, Path], strict: bool=False, password: Union[None, str, bytes]=None) -> None:
        if False:
            while True:
                i = 10
        self.strict = strict
        self.flattened_pages: Optional[List[PageObject]] = None
        self.resolved_objects: Dict[Tuple[Any, Any], Optional[PdfObject]] = {}
        self.xref_index = 0
        self._page_id2num: Optional[Dict[Any, Any]] = None
        if hasattr(stream, 'mode') and 'b' not in stream.mode:
            logger_warning('PdfReader stream/file object is not in binary mode. It may not be read correctly.', __name__)
        if isinstance(stream, (str, Path)):
            with open(stream, 'rb') as fh:
                stream = BytesIO(fh.read())
        self.read(stream)
        self.stream = stream
        self._override_encryption = False
        self._encryption: Optional[Encryption] = None
        if self.is_encrypted:
            self._override_encryption = True
            id_entry = self.trailer.get(TK.ID)
            id1_entry = id_entry[0].get_object().original_bytes if id_entry else b''
            encrypt_entry = cast(DictionaryObject, self.trailer[TK.ENCRYPT].get_object())
            self._encryption = Encryption.read(encrypt_entry, id1_entry)
            pwd = password if password is not None else b''
            if self._encryption.verify(pwd) == PasswordType.NOT_DECRYPTED and password is not None:
                raise WrongPasswordError('Wrong password')
            self._override_encryption = False
        elif password is not None:
            raise PdfReadError('Not encrypted file')

    @property
    def pdf_header(self) -> str:
        if False:
            print('Hello World!')
        "\n        The first 8 bytes of the file.\n\n        This is typically something like ``'%PDF-1.6'`` and can be used to\n        detect if the file is actually a PDF file and which version it is.\n        "
        loc = self.stream.tell()
        self.stream.seek(0, 0)
        pdf_file_version = self.stream.read(8).decode('utf-8', 'backslashreplace')
        self.stream.seek(loc, 0)
        return pdf_file_version

    @property
    def metadata(self) -> Optional[DocumentInformation]:
        if False:
            i = 10
            return i + 15
        "\n        Retrieve the PDF file's document information dictionary, if it exists.\n\n        Note that some PDF files use metadata streams instead of docinfo\n        dictionaries, and these metadata streams will not be accessed by this\n        function.\n        "
        if TK.INFO not in self.trailer:
            return None
        obj = self.trailer[TK.INFO]
        retval = DocumentInformation()
        if isinstance(obj, type(None)):
            raise PdfReadError('trailer not found or does not point to document information directory')
        retval.update(obj)
        return retval

    def getDocumentInfo(self) -> Optional[DocumentInformation]:
        if False:
            return 10
        '\n        Use the attribute :py:attr:`metadata` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('getDocumentInfo', 'metadata', '3.0.0')
        return self.metadata

    @property
    def documentInfo(self) -> Optional[DocumentInformation]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Use the attribute :py:attr:`metadata` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('documentInfo', 'metadata', '3.0.0')
        return self.metadata

    @property
    def xmp_metadata(self) -> Optional[XmpInformation]:
        if False:
            i = 10
            return i + 15
        'XMP (Extensible Metadata Platform) data.'
        try:
            self._override_encryption = True
            return self.trailer[TK.ROOT].xmp_metadata
        finally:
            self._override_encryption = False

    def getXmpMetadata(self) -> Optional[XmpInformation]:
        if False:
            while True:
                i = 10
        '\n        Use the attribute :py:attr:`metadata` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('getXmpMetadata', 'xmp_metadata', '3.0.0')
        return self.xmp_metadata

    @property
    def xmpMetadata(self) -> Optional[XmpInformation]:
        if False:
            while True:
                i = 10
        '\n        Use the attribute :py:attr:`xmp_metadata` instead.\n\n        .. deprecated:: 1.28.0.\n        '
        deprecation_with_replacement('xmpMetadata', 'xmp_metadata', '3.0.0')
        return self.xmp_metadata

    def _get_num_pages(self) -> int:
        if False:
            return 10
        '\n        Calculate the number of pages in this PDF file.\n\n        Returns:\n            The number of pages of the parsed PDF file\n\n        Raises:\n            PdfReadError: if file is encrypted and restrictions prevent\n                this action.\n        '
        if self.is_encrypted:
            return self.trailer[TK.ROOT]['/Pages']['/Count']
        else:
            if self.flattened_pages is None:
                self._flatten()
            return len(self.flattened_pages)

    def getNumPages(self) -> int:
        if False:
            return 10
        '\n        Use :code:`len(reader.pages)` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('reader.getNumPages', 'len(reader.pages)', '3.0.0')
        return self._get_num_pages()

    @property
    def numPages(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Use :code:`len(reader.pages)` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('reader.numPages', 'len(reader.pages)', '3.0.0')
        return self._get_num_pages()

    def getPage(self, pageNumber: int) -> PageObject:
        if False:
            i = 10
            return i + 15
        '\n        Use :code:`reader.pages[page_number]` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('reader.getPage(pageNumber)', 'reader.pages[page_number]', '3.0.0')
        return self._get_page(pageNumber)

    def _get_page(self, page_number: int) -> PageObject:
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieve a page by number from this PDF file.\n\n        Args:\n            page_number: The page number to retrieve\n                (pages begin at zero)\n\n        Returns:\n            A :class:`PageObject<pypdf._page.PageObject>` instance.\n        '
        if self.flattened_pages is None:
            self._flatten()
        assert self.flattened_pages is not None, 'hint for mypy'
        return self.flattened_pages[page_number]

    @property
    def namedDestinations(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        '\n        Use :py:attr:`named_destinations` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('namedDestinations', 'named_destinations', '3.0.0')
        return self.named_destinations

    @property
    def named_destinations(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        '\n        A read-only dictionary which maps names to\n        :class:`Destinations<pypdf.generic.Destination>`\n        '
        return self._get_named_destinations()

    def get_fields(self, tree: Optional[TreeObject]=None, retval: Optional[Dict[Any, Any]]=None, fileobj: Optional[Any]=None) -> Optional[Dict[str, Any]]:
        if False:
            return 10
        '\n        Extract field data if this PDF contains interactive form fields.\n\n        The *tree* and *retval* parameters are for recursive use.\n\n        Args:\n            tree:\n            retval:\n            fileobj: A file object (usually a text file) to write\n                a report to on all interactive form fields found.\n\n        Returns:\n            A dictionary where each key is a field name, and each\n            value is a :class:`Field<pypdf.generic.Field>` object. By\n            default, the mapping name is used for keys.\n            ``None`` if form data could not be located.\n        '
        field_attributes = FA.attributes_dict()
        field_attributes.update(CheckboxRadioButtonAttributes.attributes_dict())
        if retval is None:
            retval = {}
            catalog = cast(DictionaryObject, self.trailer[TK.ROOT])
            if CD.ACRO_FORM in catalog:
                tree = cast(Optional[TreeObject], catalog[CD.ACRO_FORM])
            else:
                return None
        if tree is None:
            return retval
        self._check_kids(tree, retval, fileobj)
        for attr in field_attributes:
            if attr in tree:
                self._build_field(tree, retval, fileobj, field_attributes)
                break
        if '/Fields' in tree:
            fields = cast(ArrayObject, tree['/Fields'])
            for f in fields:
                field = f.get_object()
                self._build_field(field, retval, fileobj, field_attributes)
        return retval

    def getFields(self, tree: Optional[TreeObject]=None, retval: Optional[Dict[Any, Any]]=None, fileobj: Optional[Any]=None) -> Optional[Dict[str, Any]]:
        if False:
            return 10
        '\n        Use :meth:`get_fields` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('getFields', 'get_fields', '3.0.0')
        return self.get_fields(tree, retval, fileobj)

    def _get_qualified_field_name(self, parent: DictionaryObject) -> str:
        if False:
            for i in range(10):
                print('nop')
        if '/TM' in parent:
            return cast(str, parent['/TM'])
        elif '/Parent' in parent:
            return self._get_qualified_field_name(cast(DictionaryObject, parent['/Parent'])) + '.' + cast(str, parent['/T'])
        else:
            return cast(str, parent['/T'])

    def _build_field(self, field: Union[TreeObject, DictionaryObject], retval: Dict[Any, Any], fileobj: Any, field_attributes: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._check_kids(field, retval, fileobj)
        try:
            key = cast(str, field['/TM'])
        except KeyError:
            try:
                if '/Parent' in field:
                    key = self._get_qualified_field_name(cast(DictionaryObject, field['/Parent'])) + '.'
                else:
                    key = ''
                key += cast(str, field['/T'])
            except KeyError:
                return
        if fileobj:
            self._write_field(fileobj, field, field_attributes)
            fileobj.write('\n')
        retval[key] = Field(field)
        obj = retval[key].indirect_reference.get_object()
        if obj.get(FA.FT, '') == '/Ch':
            retval[key][NameObject('/_States_')] = obj[NameObject(FA.Opt)]
        if obj.get(FA.FT, '') == '/Btn' and '/AP' in obj:
            retval[key][NameObject('/_States_')] = ArrayObject(list(obj['/AP']['/N'].keys()))
            if '/Off' not in retval[key]['/_States_']:
                retval[key][NameObject('/_States_')].append(NameObject('/Off'))
        elif obj.get(FA.FT, '') == '/Btn' and obj.get(FA.Ff, 0) & FA.FfBits.Radio != 0:
            states: List[str] = []
            retval[key][NameObject('/_States_')] = ArrayObject(states)
            for k in obj.get(FA.Kids, {}):
                k = k.get_object()
                for s in list(k['/AP']['/N'].keys()):
                    if s not in states:
                        states.append(s)
                retval[key][NameObject('/_States_')] = ArrayObject(states)
            if obj.get(FA.Ff, 0) & FA.FfBits.NoToggleToOff != 0 and '/Off' in retval[key]['/_States_']:
                del retval[key]['/_States_'][retval[key]['/_States_'].index('/Off')]

    def _check_kids(self, tree: Union[TreeObject, DictionaryObject], retval: Any, fileobj: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        if PA.KIDS in tree:
            for kid in tree[PA.KIDS]:
                self.get_fields(kid.get_object(), retval, fileobj)

    def _write_field(self, fileobj: Any, field: Any, field_attributes: Any) -> None:
        if False:
            while True:
                i = 10
        field_attributes_tuple = FA.attributes()
        field_attributes_tuple = field_attributes_tuple + CheckboxRadioButtonAttributes.attributes()
        for attr in field_attributes_tuple:
            if attr in (FA.Kids, FA.AA):
                continue
            attr_name = field_attributes[attr]
            try:
                if attr == FA.FT:
                    types = {'/Btn': 'Button', '/Tx': 'Text', '/Ch': 'Choice', '/Sig': 'Signature'}
                    if field[attr] in types:
                        fileobj.write(f'{attr_name}: {types[field[attr]]}\n')
                elif attr == FA.Parent:
                    try:
                        name = field[attr][FA.TM]
                    except KeyError:
                        name = field[attr][FA.T]
                    fileobj.write(f'{attr_name}: {name}\n')
                else:
                    fileobj.write(f'{attr_name}: {field[attr]}\n')
            except KeyError:
                pass

    def get_form_text_fields(self, full_qualified_name: bool=False) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        '\n        Retrieve form fields from the document with textual data.\n\n        Args:\n            full_qualified_name: to get full name\n\n        Returns:\n            A dictionary. The key is the name of the form field,\n            the value is the content of the field.\n\n            If the document contains multiple form fields with the same name, the\n            second and following will get the suffix .2, .3, ...\n        '

        def indexed_key(k: str, fields: Dict[Any, Any]) -> str:
            if False:
                for i in range(10):
                    print('nop')
            if k not in fields:
                return k
            else:
                return k + '.' + str(sum([1 for kk in fields if kk.startswith(k + '.')]) + 2)
        formfields = self.get_fields()
        if formfields is None:
            return {}
        ff = {}
        for (field, value) in formfields.items():
            if value.get('/FT') == '/Tx':
                if full_qualified_name:
                    ff[field] = value.get('/V')
                else:
                    ff[indexed_key(cast(str, value['/T']), ff)] = value.get('/V')
        return ff

    def getFormTextFields(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        '\n        Use :meth:`get_form_text_fields` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('getFormTextFields', 'get_form_text_fields', '3.0.0')
        return self.get_form_text_fields()

    def _get_named_destinations(self, tree: Union[TreeObject, None]=None, retval: Optional[Any]=None) -> Dict[str, Any]:
        if False:
            return 10
        '\n        Retrieve the named destinations present in the document.\n\n        Args:\n            tree:\n            retval:\n\n        Returns:\n            A dictionary which maps names to\n            :class:`Destinations<pypdf.generic.Destination>`.\n        '
        if retval is None:
            retval = {}
            catalog = cast(DictionaryObject, self.trailer[TK.ROOT])
            if CA.DESTS in catalog:
                tree = cast(TreeObject, catalog[CA.DESTS])
            elif CA.NAMES in catalog:
                names = cast(DictionaryObject, catalog[CA.NAMES])
                if CA.DESTS in names:
                    tree = cast(TreeObject, names[CA.DESTS])
        if tree is None:
            return retval
        if PA.KIDS in tree:
            for kid in cast(ArrayObject, tree[PA.KIDS]):
                self._get_named_destinations(kid.get_object(), retval)
        elif CA.NAMES in tree:
            names = cast(DictionaryObject, tree[CA.NAMES])
            i = 0
            while i < len(names):
                key = cast(str, names[i].get_object())
                i += 1
                if not isinstance(key, str):
                    continue
                try:
                    value = names[i].get_object()
                except IndexError:
                    break
                i += 1
                if isinstance(value, DictionaryObject) and '/D' in value:
                    value = value['/D']
                dest = self._build_destination(key, value)
                if dest is not None:
                    retval[key] = dest
        else:
            for (k__, v__) in tree.items():
                val = v__.get_object()
                if isinstance(val, DictionaryObject):
                    val = val['/D'].get_object()
                dest = self._build_destination(k__, val)
                if dest is not None:
                    retval[k__] = dest
        return retval

    def getNamedDestinations(self, tree: Union[TreeObject, None]=None, retval: Optional[Any]=None) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        '\n        Use :py:attr:`named_destinations` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('getNamedDestinations', 'named_destinations', '3.0.0')
        return self._get_named_destinations(tree, retval)

    @property
    def outline(self) -> OutlineType:
        if False:
            for i in range(10):
                print('nop')
        "\n        Read-only property for the outline present in the document.\n\n        (i.e., a collection of 'outline items' which are also known as\n        'bookmarks')\n        "
        return self._get_outline()

    @property
    def outlines(self) -> OutlineType:
        if False:
            for i in range(10):
                print('nop')
        '\n        Use :py:attr:`outline` instead.\n\n        .. deprecated:: 2.9.0\n        '
        deprecation_with_replacement('outlines', 'outline', '3.0.0')
        return self.outline

    def _get_outline(self, node: Optional[DictionaryObject]=None, outline: Optional[Any]=None) -> OutlineType:
        if False:
            i = 10
            return i + 15
        if outline is None:
            outline = []
            catalog = cast(DictionaryObject, self.trailer[TK.ROOT])
            if CO.OUTLINES in catalog:
                lines = cast(DictionaryObject, catalog[CO.OUTLINES])
                if isinstance(lines, NullObject):
                    return outline
                if lines is not None and '/First' in lines:
                    node = cast(DictionaryObject, lines['/First'])
            self._namedDests = self._get_named_destinations()
        if node is None:
            return outline
        while True:
            outline_obj = self._build_outline_item(node)
            if outline_obj:
                outline.append(outline_obj)
            if '/First' in node:
                sub_outline: List[Any] = []
                self._get_outline(cast(DictionaryObject, node['/First']), sub_outline)
                if sub_outline:
                    outline.append(sub_outline)
            if '/Next' not in node:
                break
            node = cast(DictionaryObject, node['/Next'])
        return outline

    def getOutlines(self, node: Optional[DictionaryObject]=None, outline: Optional[Any]=None) -> OutlineType:
        if False:
            for i in range(10):
                print('nop')
        '\n        Use :py:attr:`outline` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('getOutlines', 'outline', '3.0.0')
        return self._get_outline(node, outline)

    @property
    def threads(self) -> Optional[ArrayObject]:
        if False:
            i = 10
            return i + 15
        '\n        Read-only property for the list of threads.\n\n        See §8.3.2 from PDF 1.7 spec.\n\n        It\'s an array of dictionaries with "/F" and "/I" properties or\n        None if there are no articles.\n        '
        catalog = cast(DictionaryObject, self.trailer[TK.ROOT])
        if CO.THREADS in catalog:
            return cast('ArrayObject', catalog[CO.THREADS])
        else:
            return None

    def _get_page_number_by_indirect(self, indirect_reference: Union[None, int, NullObject, IndirectObject]) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Generate _page_id2num.\n\n        Args:\n            indirect_reference:\n\n        Returns:\n            The page number.\n        '
        if self._page_id2num is None:
            self._page_id2num = {x.indirect_reference.idnum: i for (i, x) in enumerate(self.pages)}
        if indirect_reference is None or isinstance(indirect_reference, NullObject):
            return -1
        if isinstance(indirect_reference, int):
            idnum = indirect_reference
        else:
            idnum = indirect_reference.idnum
        assert self._page_id2num is not None, 'hint for mypy'
        ret = self._page_id2num.get(idnum, -1)
        return ret

    def get_page_number(self, page: PageObject) -> int:
        if False:
            return 10
        '\n        Retrieve page number of a given PageObject.\n\n        Args:\n            page: The page to get page number. Should be\n                an instance of :class:`PageObject<pypdf._page.PageObject>`\n\n        Returns:\n            The page number or -1 if page is not found\n        '
        return self._get_page_number_by_indirect(page.indirect_reference)

    def getPageNumber(self, page: PageObject) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Use :meth:`get_page_number` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('getPageNumber', 'get_page_number', '3.0.0')
        return self.get_page_number(page)

    def get_destination_page_number(self, destination: Destination) -> int:
        if False:
            return 10
        '\n        Retrieve page number of a given Destination object.\n\n        Args:\n            destination: The destination to get page number.\n\n        Returns:\n            The page number or -1 if page is not found\n        '
        return self._get_page_number_by_indirect(destination.page)

    def getDestinationPageNumber(self, destination: Destination) -> int:
        if False:
            while True:
                i = 10
        '\n        Use :meth:`get_destination_page_number` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('getDestinationPageNumber', 'get_destination_page_number', '3.0.0')
        return self.get_destination_page_number(destination)

    def _build_destination(self, title: str, array: Optional[List[Union[NumberObject, IndirectObject, None, NullObject, DictionaryObject]]]) -> Destination:
        if False:
            while True:
                i = 10
        (page, typ) = (None, None)
        if isinstance(array, (NullObject, str)) or (isinstance(array, ArrayObject) and len(array) == 0) or array is None:
            page = NullObject()
            return Destination(title, page, Fit.fit())
        else:
            (page, typ) = array[0:2]
            array = array[2:]
            try:
                return Destination(title, page, Fit(fit_type=typ, fit_args=array))
            except PdfReadError:
                logger_warning(f'Unknown destination: {title} {array}', __name__)
                if self.strict:
                    raise
                tmp = self.pages[0].indirect_reference
                indirect_reference = NullObject() if tmp is None else tmp
                return Destination(title, indirect_reference, Fit.fit())

    def _build_outline_item(self, node: DictionaryObject) -> Optional[Destination]:
        if False:
            while True:
                i = 10
        (dest, title, outline_item) = (None, None, None)
        try:
            title = cast('str', node['/Title'])
        except KeyError:
            if self.strict:
                raise PdfReadError(f'Outline Entry Missing /Title attribute: {node!r}')
            title = ''
        if '/A' in node:
            action = cast(DictionaryObject, node['/A'])
            action_type = cast(NameObject, action[GoToActionArguments.S])
            if action_type == '/GoTo':
                dest = action[GoToActionArguments.D]
        elif '/Dest' in node:
            dest = node['/Dest']
            if isinstance(dest, DictionaryObject) and '/D' in dest:
                dest = dest['/D']
        if isinstance(dest, ArrayObject):
            outline_item = self._build_destination(title, dest)
        elif isinstance(dest, str):
            try:
                outline_item = self._build_destination(title, self._namedDests[dest].dest_array)
            except KeyError:
                outline_item = self._build_destination(title, None)
        elif dest is None:
            outline_item = self._build_destination(title, dest)
        else:
            if self.strict:
                raise PdfReadError(f'Unexpected destination {dest!r}')
            else:
                logger_warning(f'Removed unexpected destination {dest!r} from destination', __name__)
            outline_item = self._build_destination(title, None)
        if outline_item:
            if '/C' in node:
                outline_item[NameObject('/C')] = ArrayObject((FloatObject(c) for c in node['/C']))
            if '/F' in node:
                outline_item[NameObject('/F')] = node['/F']
            if '/Count' in node:
                outline_item[NameObject('/Count')] = node['/Count']
            outline_item[NameObject('/%is_open%')] = BooleanObject(node.get('/Count', 0) >= 0)
        outline_item.node = node
        try:
            outline_item.indirect_reference = node.indirect_reference
        except AttributeError:
            pass
        return outline_item

    @property
    def pages(self) -> List[PageObject]:
        if False:
            while True:
                i = 10
        'Read-only property that emulates a list of :py:class:`Page<pypdf._page.Page>` objects.'
        return _VirtualList(self._get_num_pages, self._get_page)

    @property
    def page_labels(self) -> List[str]:
        if False:
            print('Hello World!')
        '\n        A list of labels for the pages in this document.\n\n        This property is read-only. The labels are in the order that the pages\n        appear in the document.\n        '
        return [page_index2page_label(self, i) for i in range(len(self.pages))]

    @property
    def page_layout(self) -> Optional[str]:
        if False:
            return 10
        '\n        Get the page layout currently being used.\n\n        .. list-table:: Valid ``layout`` values\n           :widths: 50 200\n\n           * - /NoLayout\n             - Layout explicitly not specified\n           * - /SinglePage\n             - Show one page at a time\n           * - /OneColumn\n             - Show one column at a time\n           * - /TwoColumnLeft\n             - Show pages in two columns, odd-numbered pages on the left\n           * - /TwoColumnRight\n             - Show pages in two columns, odd-numbered pages on the right\n           * - /TwoPageLeft\n             - Show two pages at a time, odd-numbered pages on the left\n           * - /TwoPageRight\n             - Show two pages at a time, odd-numbered pages on the right\n        '
        trailer = cast(DictionaryObject, self.trailer[TK.ROOT])
        if CD.PAGE_LAYOUT in trailer:
            return cast(NameObject, trailer[CD.PAGE_LAYOUT])
        return None

    def getPageLayout(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        '\n        Use :py:attr:`page_layout` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('getPageLayout', 'page_layout', '3.0.0')
        return self.page_layout

    @property
    def pageLayout(self) -> Optional[str]:
        if False:
            print('Hello World!')
        '\n        Use :py:attr:`page_layout` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('pageLayout', 'page_layout', '3.0.0')
        return self.page_layout

    @property
    def page_mode(self) -> Optional[PagemodeType]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the page mode currently being used.\n\n        .. list-table:: Valid ``mode`` values\n           :widths: 50 200\n\n           * - /UseNone\n             - Do not show outline or thumbnails panels\n           * - /UseOutlines\n             - Show outline (aka bookmarks) panel\n           * - /UseThumbs\n             - Show page thumbnails panel\n           * - /FullScreen\n             - Fullscreen view\n           * - /UseOC\n             - Show Optional Content Group (OCG) panel\n           * - /UseAttachments\n             - Show attachments panel\n        '
        try:
            return self.trailer[TK.ROOT]['/PageMode']
        except KeyError:
            return None

    def getPageMode(self) -> Optional[PagemodeType]:
        if False:
            i = 10
            return i + 15
        '\n        Use :py:attr:`page_mode` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('getPageMode', 'page_mode', '3.0.0')
        return self.page_mode

    @property
    def pageMode(self) -> Optional[PagemodeType]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Use :py:attr:`page_mode` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('pageMode', 'page_mode', '3.0.0')
        return self.page_mode

    def _flatten(self, pages: Union[None, DictionaryObject, PageObject]=None, inherit: Optional[Dict[str, Any]]=None, indirect_reference: Optional[IndirectObject]=None) -> None:
        if False:
            while True:
                i = 10
        inheritable_page_attributes = (NameObject(PG.RESOURCES), NameObject(PG.MEDIABOX), NameObject(PG.CROPBOX), NameObject(PG.ROTATE))
        if inherit is None:
            inherit = {}
        if pages is None:
            catalog = self.trailer[TK.ROOT].get_object()
            pages = catalog['/Pages'].get_object()
            self.flattened_pages = []
        if PA.TYPE in pages:
            t = pages[PA.TYPE]
        elif PA.KIDS not in pages:
            t = '/Page'
        else:
            t = '/Pages'
        if t == '/Pages':
            for attr in inheritable_page_attributes:
                if attr in pages:
                    inherit[attr] = pages[attr]
            for page in pages[PA.KIDS]:
                addt = {}
                if isinstance(page, IndirectObject):
                    addt['indirect_reference'] = page
                obj = page.get_object()
                if obj:
                    self._flatten(obj, inherit, **addt)
        elif t == '/Page':
            for (attr_in, value) in list(inherit.items()):
                if attr_in not in pages:
                    pages[attr_in] = value
            page_obj = PageObject(self, indirect_reference)
            page_obj.update(pages)
            self.flattened_pages.append(page_obj)

    def _get_object_from_stream(self, indirect_reference: IndirectObject) -> Union[int, PdfObject, str]:
        if False:
            return 10
        (stmnum, idx) = self.xref_objStm[indirect_reference.idnum]
        obj_stm: EncodedStreamObject = IndirectObject(stmnum, 0, self).get_object()
        assert cast(str, obj_stm['/Type']) == '/ObjStm'
        assert idx < obj_stm['/N']
        stream_data = BytesIO(b_(obj_stm.get_data()))
        for i in range(obj_stm['/N']):
            read_non_whitespace(stream_data)
            stream_data.seek(-1, 1)
            objnum = NumberObject.read_from_stream(stream_data)
            read_non_whitespace(stream_data)
            stream_data.seek(-1, 1)
            offset = NumberObject.read_from_stream(stream_data)
            read_non_whitespace(stream_data)
            stream_data.seek(-1, 1)
            if objnum != indirect_reference.idnum:
                continue
            if self.strict and idx != i:
                raise PdfReadError('Object is in wrong index.')
            stream_data.seek(int(obj_stm['/First'] + offset), 0)
            read_non_whitespace(stream_data)
            stream_data.seek(-1, 1)
            try:
                obj = read_object(stream_data, self)
            except PdfStreamError as exc:
                logger_warning(f'Invalid stream (index {i}) within object {indirect_reference.idnum} {indirect_reference.generation}: {exc}', __name__)
                if self.strict:
                    raise PdfReadError(f"Can't read object stream: {exc}")
                obj = NullObject()
            return obj
        if self.strict:
            raise PdfReadError('This is a fatal error in strict mode.')
        return NullObject()

    def _get_indirect_object(self, num: int, gen: int) -> Optional[PdfObject]:
        if False:
            i = 10
            return i + 15
        '\n        Used to ease development.\n\n        This is equivalent to generic.IndirectObject(num,gen,self).get_object()\n\n        Args:\n            num: The object number of the indirect object.\n            gen: The generation number of the indirect object.\n\n        Returns:\n            A PdfObject\n        '
        return IndirectObject(num, gen, self).get_object()

    def get_object(self, indirect_reference: Union[int, IndirectObject]) -> Optional[PdfObject]:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(indirect_reference, int):
            indirect_reference = IndirectObject(indirect_reference, 0, self)
        retval = self.cache_get_indirect_object(indirect_reference.generation, indirect_reference.idnum)
        if retval is not None:
            return retval
        if indirect_reference.generation == 0 and indirect_reference.idnum in self.xref_objStm:
            retval = self._get_object_from_stream(indirect_reference)
        elif indirect_reference.generation in self.xref and indirect_reference.idnum in self.xref[indirect_reference.generation]:
            if self.xref_free_entry.get(indirect_reference.generation, {}).get(indirect_reference.idnum, False):
                return NullObject()
            start = self.xref[indirect_reference.generation][indirect_reference.idnum]
            self.stream.seek(start, 0)
            try:
                (idnum, generation) = self.read_object_header(self.stream)
            except Exception:
                if hasattr(self.stream, 'getbuffer'):
                    buf = bytes(self.stream.getbuffer())
                else:
                    p = self.stream.tell()
                    self.stream.seek(0, 0)
                    buf = self.stream.read(-1)
                    self.stream.seek(p, 0)
                m = re.search(f'\\s{indirect_reference.idnum}\\s+{indirect_reference.generation}\\s+obj'.encode(), buf)
                if m is not None:
                    logger_warning(f'Object ID {indirect_reference.idnum},{indirect_reference.generation} ref repaired', __name__)
                    self.xref[indirect_reference.generation][indirect_reference.idnum] = m.start(0) + 1
                    self.stream.seek(m.start(0) + 1)
                    (idnum, generation) = self.read_object_header(self.stream)
                else:
                    idnum = -1
            if idnum != indirect_reference.idnum and self.xref_index:
                if self.strict:
                    raise PdfReadError(f'Expected object ID ({indirect_reference.idnum} {indirect_reference.generation}) does not match actual ({idnum} {generation}); xref table not zero-indexed.')
            elif idnum != indirect_reference.idnum and self.strict:
                raise PdfReadError(f'Expected object ID ({indirect_reference.idnum} {indirect_reference.generation}) does not match actual ({idnum} {generation}).')
            if self.strict:
                assert generation == indirect_reference.generation
            retval = read_object(self.stream, self)
            if not self._override_encryption and self._encryption is not None:
                if not self._encryption.is_decrypted():
                    raise FileNotDecryptedError('File has not been decrypted')
                retval = cast(PdfObject, retval)
                retval = self._encryption.decrypt_object(retval, indirect_reference.idnum, indirect_reference.generation)
        else:
            if hasattr(self.stream, 'getbuffer'):
                buf = bytes(self.stream.getbuffer())
            else:
                p = self.stream.tell()
                self.stream.seek(0, 0)
                buf = self.stream.read(-1)
                self.stream.seek(p, 0)
            m = re.search(f'\\s{indirect_reference.idnum}\\s+{indirect_reference.generation}\\s+obj'.encode(), buf)
            if m is not None:
                logger_warning(f'Object {indirect_reference.idnum} {indirect_reference.generation} found', __name__)
                if indirect_reference.generation not in self.xref:
                    self.xref[indirect_reference.generation] = {}
                self.xref[indirect_reference.generation][indirect_reference.idnum] = m.start(0) + 1
                self.stream.seek(m.end(0) + 1)
                skip_over_whitespace(self.stream)
                self.stream.seek(-1, 1)
                retval = read_object(self.stream, self)
                if not self._override_encryption and self._encryption is not None:
                    if not self._encryption.is_decrypted():
                        raise FileNotDecryptedError('File has not been decrypted')
                    retval = cast(PdfObject, retval)
                    retval = self._encryption.decrypt_object(retval, indirect_reference.idnum, indirect_reference.generation)
            else:
                logger_warning(f'Object {indirect_reference.idnum} {indirect_reference.generation} not defined.', __name__)
                if self.strict:
                    raise PdfReadError('Could not find object.')
        self.cache_indirect_object(indirect_reference.generation, indirect_reference.idnum, retval)
        return retval

    def getObject(self, indirectReference: IndirectObject) -> Optional[PdfObject]:
        if False:
            return 10
        '\n        Use :meth:`get_object` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('getObject', 'get_object', '3.0.0')
        return self.get_object(indirectReference)

    def read_object_header(self, stream: StreamType) -> Tuple[int, int]:
        if False:
            for i in range(10):
                print('nop')
        extra = False
        skip_over_comment(stream)
        extra |= skip_over_whitespace(stream)
        stream.seek(-1, 1)
        idnum = read_until_whitespace(stream)
        extra |= skip_over_whitespace(stream)
        stream.seek(-1, 1)
        generation = read_until_whitespace(stream)
        extra |= skip_over_whitespace(stream)
        stream.seek(-1, 1)
        _obj = stream.read(3)
        read_non_whitespace(stream)
        stream.seek(-1, 1)
        if extra and self.strict:
            logger_warning(f'Superfluous whitespace found in object header {idnum} {generation}', __name__)
        return (int(idnum), int(generation))

    def readObjectHeader(self, stream: StreamType) -> Tuple[int, int]:
        if False:
            i = 10
            return i + 15
        '\n        Use :meth:`read_object_header` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('readObjectHeader', 'read_object_header', '3.0.0')
        return self.read_object_header(stream)

    def cache_get_indirect_object(self, generation: int, idnum: int) -> Optional[PdfObject]:
        if False:
            for i in range(10):
                print('nop')
        return self.resolved_objects.get((generation, idnum))

    def cacheGetIndirectObject(self, generation: int, idnum: int) -> Optional[PdfObject]:
        if False:
            print('Hello World!')
        '\n        Use :meth:`cache_get_indirect_object` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('cacheGetIndirectObject', 'cache_get_indirect_object', '3.0.0')
        return self.cache_get_indirect_object(generation, idnum)

    def cache_indirect_object(self, generation: int, idnum: int, obj: Optional[PdfObject]) -> Optional[PdfObject]:
        if False:
            while True:
                i = 10
        if (generation, idnum) in self.resolved_objects:
            msg = f'Overwriting cache for {generation} {idnum}'
            if self.strict:
                raise PdfReadError(msg)
            logger_warning(msg, __name__)
        self.resolved_objects[generation, idnum] = obj
        if obj is not None:
            obj.indirect_reference = IndirectObject(idnum, generation, self)
        return obj

    def cacheIndirectObject(self, generation: int, idnum: int, obj: Optional[PdfObject]) -> Optional[PdfObject]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Use :meth:`cache_indirect_object` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('cacheIndirectObject', 'cache_indirect_object')
        return self.cache_indirect_object(generation, idnum, obj)

    def read(self, stream: StreamType) -> None:
        if False:
            i = 10
            return i + 15
        self._basic_validation(stream)
        self._find_eof_marker(stream)
        startxref = self._find_startxref_pos(stream)
        xref_issue_nr = self._get_xref_issues(stream, startxref)
        if xref_issue_nr != 0:
            if self.strict and xref_issue_nr:
                raise PdfReadError('Broken xref table')
            logger_warning(f'incorrect startxref pointer({xref_issue_nr})', __name__)
        self._read_xref_tables_and_trailers(stream, startxref, xref_issue_nr)
        if self.xref_index and (not self.strict):
            loc = stream.tell()
            for (gen, xref_entry) in self.xref.items():
                if gen == 65535:
                    continue
                xref_k = sorted(xref_entry.keys())
                for id in xref_k:
                    stream.seek(xref_entry[id], 0)
                    try:
                        (pid, _pgen) = self.read_object_header(stream)
                    except ValueError:
                        break
                    if pid == id - self.xref_index:
                        self.xref[gen][pid] = self.xref[gen][id]
                        del self.xref[gen][id]
            stream.seek(loc, 0)

    def _basic_validation(self, stream: StreamType) -> None:
        if False:
            while True:
                i = 10
        'Ensure file is not empty. Read at most 5 bytes.'
        stream.seek(0, os.SEEK_SET)
        try:
            header_byte = stream.read(5)
        except UnicodeDecodeError:
            raise UnsupportedOperation('cannot read header')
        if header_byte == b'':
            raise EmptyFileError('Cannot read an empty file')
        elif header_byte != b'%PDF-':
            if self.strict:
                raise PdfReadError(f"PDF starts with '{header_byte.decode('utf8')}', but '%PDF-' expected")
            else:
                logger_warning(f'invalid pdf header: {header_byte}', __name__)
        stream.seek(0, os.SEEK_END)

    def _find_eof_marker(self, stream: StreamType) -> None:
        if False:
            return 10
        '\n        Jump to the %%EOF marker.\n\n        According to the specs, the %%EOF marker should be at the very end of\n        the file. Hence for standard-compliant PDF documents this function will\n        read only the last part (DEFAULT_BUFFER_SIZE).\n        '
        HEADER_SIZE = 8
        line = b''
        while line[:5] != b'%%EOF':
            if stream.tell() < HEADER_SIZE:
                if self.strict:
                    raise PdfReadError('EOF marker not found')
                else:
                    logger_warning('EOF marker not found', __name__)
            line = read_previous_line(stream)

    def _find_startxref_pos(self, stream: StreamType) -> int:
        if False:
            return 10
        '\n        Find startxref entry - the location of the xref table.\n\n        Args:\n            stream:\n\n        Returns:\n            The bytes offset\n        '
        line = read_previous_line(stream)
        try:
            startxref = int(line)
        except ValueError:
            if not line.startswith(b'startxref'):
                raise PdfReadError('startxref not found')
            startxref = int(line[9:].strip())
            logger_warning('startxref on same line as offset', __name__)
        else:
            line = read_previous_line(stream)
            if line[:9] != b'startxref':
                raise PdfReadError('startxref not found')
        return startxref

    def _read_standard_xref_table(self, stream: StreamType) -> None:
        if False:
            return 10
        ref = stream.read(3)
        if ref != b'ref':
            raise PdfReadError('xref table read error')
        read_non_whitespace(stream)
        stream.seek(-1, 1)
        first_time = True
        while True:
            num = cast(int, read_object(stream, self))
            if first_time and num != 0:
                self.xref_index = num
                if self.strict:
                    logger_warning('Xref table not zero-indexed. ID numbers for objects will be corrected.', __name__)
            first_time = False
            read_non_whitespace(stream)
            stream.seek(-1, 1)
            size = cast(int, read_object(stream, self))
            read_non_whitespace(stream)
            stream.seek(-1, 1)
            cnt = 0
            while cnt < size:
                line = stream.read(20)
                while line[0] in b'\r\n':
                    stream.seek(-20 + 1, 1)
                    line = stream.read(20)
                if line[-1] in b'0123456789t':
                    stream.seek(-1, 1)
                try:
                    (offset_b, generation_b) = line[:16].split(b' ')
                    entry_type_b = line[17:18]
                    (offset, generation) = (int(offset_b), int(generation_b))
                except Exception:
                    if hasattr(stream, 'getbuffer'):
                        buf = bytes(stream.getbuffer())
                    else:
                        p = stream.tell()
                        stream.seek(0, 0)
                        buf = stream.read(-1)
                        stream.seek(p)
                    f = re.search(f'{num}\\s+(\\d+)\\s+obj'.encode(), buf)
                    if f is None:
                        logger_warning(f'entry {num} in Xref table invalid; object not found', __name__)
                        generation = 65535
                        offset = -1
                    else:
                        logger_warning(f'entry {num} in Xref table invalid but object found', __name__)
                        generation = int(f.group(1))
                        offset = f.start()
                if generation not in self.xref:
                    self.xref[generation] = {}
                    self.xref_free_entry[generation] = {}
                if num in self.xref[generation]:
                    pass
                else:
                    self.xref[generation][num] = offset
                    try:
                        self.xref_free_entry[generation][num] = entry_type_b == b'f'
                    except Exception:
                        pass
                    try:
                        self.xref_free_entry[65535][num] = entry_type_b == b'f'
                    except Exception:
                        pass
                cnt += 1
                num += 1
            read_non_whitespace(stream)
            stream.seek(-1, 1)
            trailer_tag = stream.read(7)
            if trailer_tag != b'trailer':
                stream.seek(-7, 1)
            else:
                break

    def _read_xref_tables_and_trailers(self, stream: StreamType, startxref: Optional[int], xref_issue_nr: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.xref: Dict[int, Dict[Any, Any]] = {}
        self.xref_free_entry: Dict[int, Dict[Any, Any]] = {}
        self.xref_objStm: Dict[int, Tuple[Any, Any]] = {}
        self.trailer = DictionaryObject()
        while startxref is not None:
            stream.seek(startxref, 0)
            x = stream.read(1)
            if x in b'\r\n':
                x = stream.read(1)
            if x == b'x':
                startxref = self._read_xref(stream)
            elif xref_issue_nr:
                try:
                    self._rebuild_xref_table(stream)
                    break
                except Exception:
                    xref_issue_nr = 0
            elif x.isdigit():
                try:
                    xrefstream = self._read_pdf15_xref_stream(stream)
                except Exception as e:
                    if TK.ROOT in self.trailer:
                        logger_warning(f'Previous trailer can not be read {e.args}', __name__)
                        break
                    else:
                        raise PdfReadError(f'trailer can not be read {e.args}')
                trailer_keys = (TK.ROOT, TK.ENCRYPT, TK.INFO, TK.ID, TK.SIZE)
                for key in trailer_keys:
                    if key in xrefstream and key not in self.trailer:
                        self.trailer[NameObject(key)] = xrefstream.raw_get(key)
                if '/XRefStm' in xrefstream:
                    p = stream.tell()
                    stream.seek(cast(int, xrefstream['/XRefStm']) + 1, 0)
                    self._read_pdf15_xref_stream(stream)
                    stream.seek(p, 0)
                if '/Prev' in xrefstream:
                    startxref = cast(int, xrefstream['/Prev'])
                else:
                    break
            else:
                startxref = self._read_xref_other_error(stream, startxref)

    def _read_xref(self, stream: StreamType) -> Optional[int]:
        if False:
            while True:
                i = 10
        self._read_standard_xref_table(stream)
        read_non_whitespace(stream)
        stream.seek(-1, 1)
        new_trailer = cast(Dict[str, Any], read_object(stream, self))
        for (key, value) in new_trailer.items():
            if key not in self.trailer:
                self.trailer[key] = value
        if '/XRefStm' in new_trailer:
            p = stream.tell()
            stream.seek(cast(int, new_trailer['/XRefStm']) + 1, 0)
            try:
                self._read_pdf15_xref_stream(stream)
            except Exception:
                logger_warning(f"XRef object at {new_trailer['/XRefStm']} can not be read, some object may be missing", __name__)
            stream.seek(p, 0)
        if '/Prev' in new_trailer:
            startxref = new_trailer['/Prev']
            return startxref
        else:
            return None

    def _read_xref_other_error(self, stream: StreamType, startxref: int) -> Optional[int]:
        if False:
            for i in range(10):
                print('nop')
        if startxref == 0:
            if self.strict:
                raise PdfReadError('/Prev=0 in the trailer (try opening with strict=False)')
            logger_warning('/Prev=0 in the trailer - assuming there is no previous xref table', __name__)
            return None
        stream.seek(-11, 1)
        tmp = stream.read(20)
        xref_loc = tmp.find(b'xref')
        if xref_loc != -1:
            startxref -= 10 - xref_loc
            return startxref
        stream.seek(startxref, 0)
        for look in range(25):
            if stream.read(1).isdigit():
                startxref += look
                return startxref
        if '/Root' in self.trailer and (not self.strict):
            logger_warning('Invalid parent xref., rebuild xref', __name__)
            try:
                self._rebuild_xref_table(stream)
                return None
            except Exception:
                raise PdfReadError('can not rebuild xref')
        raise PdfReadError('Could not find xref table at specified location')

    def _read_pdf15_xref_stream(self, stream: StreamType) -> Union[ContentStream, EncodedStreamObject, DecodedStreamObject]:
        if False:
            while True:
                i = 10
        stream.seek(-1, 1)
        (idnum, generation) = self.read_object_header(stream)
        xrefstream = cast(ContentStream, read_object(stream, self))
        assert cast(str, xrefstream['/Type']) == '/XRef'
        self.cache_indirect_object(generation, idnum, xrefstream)
        stream_data = BytesIO(b_(xrefstream.get_data()))
        idx_pairs = xrefstream.get('/Index', [0, xrefstream.get('/Size')])
        entry_sizes = cast(Dict[Any, Any], xrefstream.get('/W'))
        assert len(entry_sizes) >= 3
        if self.strict and len(entry_sizes) > 3:
            raise PdfReadError(f'Too many entry sizes: {entry_sizes}')

        def get_entry(i: int) -> Union[int, Tuple[int, ...]]:
            if False:
                print('Hello World!')
            if entry_sizes[i] > 0:
                d = stream_data.read(entry_sizes[i])
                return convert_to_int(d, entry_sizes[i])
            if i == 0:
                return 1
            else:
                return 0

        def used_before(num: int, generation: Union[int, Tuple[int, ...]]) -> bool:
            if False:
                print('Hello World!')
            return num in self.xref.get(generation, []) or num in self.xref_objStm
        self._read_xref_subsections(idx_pairs, get_entry, used_before)
        return xrefstream

    @staticmethod
    def _get_xref_issues(stream: StreamType, startxref: int) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Return an int which indicates an issue. 0 means there is no issue.\n\n        Args:\n            stream:\n            startxref:\n\n        Returns:\n            0 means no issue, other values represent specific issues.\n        '
        stream.seek(startxref - 1, 0)
        line = stream.read(1)
        if line == b'j':
            line = stream.read(1)
        if line not in b'\r\n \t':
            return 1
        line = stream.read(4)
        if line != b'xref':
            line = b''
            while line in b'0123456789 \t':
                line = stream.read(1)
                if line == b'':
                    return 2
            line += stream.read(2)
            if line.lower() != b'obj':
                return 3
        return 0

    def _rebuild_xref_table(self, stream: StreamType) -> None:
        if False:
            print('Hello World!')
        self.xref = {}
        stream.seek(0, 0)
        f_ = stream.read(-1)
        for m in re.finditer(b'[\\r\\n \\t][ \\t]*(\\d+)[ \\t]+(\\d+)[ \\t]+obj', f_):
            idnum = int(m.group(1))
            generation = int(m.group(2))
            if generation not in self.xref:
                self.xref[generation] = {}
            self.xref[generation][idnum] = m.start(1)
        stream.seek(0, 0)
        for m in re.finditer(b'[\\r\\n \\t][ \\t]*trailer[\\r\\n \\t]*(<<)', f_):
            stream.seek(m.start(1), 0)
            new_trailer = cast(Dict[Any, Any], read_object(stream, self))
            for (key, value) in list(new_trailer.items()):
                self.trailer[key] = value

    def _read_xref_subsections(self, idx_pairs: List[int], get_entry: Callable[[int], Union[int, Tuple[int, ...]]], used_before: Callable[[int, Union[int, Tuple[int, ...]]], bool]) -> None:
        if False:
            i = 10
            return i + 15
        for (start, size) in self._pairs(idx_pairs):
            for num in range(start, start + size):
                xref_type = get_entry(0)
                if xref_type == 0:
                    next_free_object = get_entry(1)
                    next_generation = get_entry(2)
                elif xref_type == 1:
                    byte_offset = get_entry(1)
                    generation = get_entry(2)
                    if generation not in self.xref:
                        self.xref[generation] = {}
                    if not used_before(num, generation):
                        self.xref[generation][num] = byte_offset
                elif xref_type == 2:
                    objstr_num = get_entry(1)
                    obstr_idx = get_entry(2)
                    generation = 0
                    if not used_before(num, generation):
                        self.xref_objStm[num] = (objstr_num, obstr_idx)
                elif self.strict:
                    raise PdfReadError(f'Unknown xref type: {xref_type}')

    def _pairs(self, array: List[int]) -> Iterable[Tuple[int, int]]:
        if False:
            for i in range(10):
                print('nop')
        i = 0
        while True:
            yield (array[i], array[i + 1])
            i += 2
            if i + 1 >= len(array):
                break

    def read_next_end_line(self, stream: StreamType, limit_offset: int=0) -> bytes:
        if False:
            print('Hello World!')
        '.. deprecated:: 2.1.0'
        deprecate_no_replacement('read_next_end_line', removed_in='4.0.0')
        line_parts = []
        while True:
            if stream.tell() == 0 or stream.tell() == limit_offset:
                raise PdfReadError('Could not read malformed PDF file')
            x = stream.read(1)
            if stream.tell() < 2:
                raise PdfReadError('EOL marker not found')
            stream.seek(-2, 1)
            if x in (b'\n', b'\r'):
                crlf = False
                while x in (b'\n', b'\r'):
                    x = stream.read(1)
                    if x in (b'\n', b'\r'):
                        stream.seek(-1, 1)
                        crlf = True
                    if stream.tell() < 2:
                        raise PdfReadError('EOL marker not found')
                    stream.seek(-2, 1)
                stream.seek(2 if crlf else 1, 1)
                break
            else:
                line_parts.append(x)
        line_parts.reverse()
        return b''.join(line_parts)

    def readNextEndLine(self, stream: StreamType, limit_offset: int=0) -> bytes:
        if False:
            print('Hello World!')
        '.. deprecated:: 1.28.0'
        deprecation_no_replacement('readNextEndLine', '3.0.0')
        return self.read_next_end_line(stream, limit_offset)

    def decrypt(self, password: Union[str, bytes]) -> PasswordType:
        if False:
            for i in range(10):
                print('nop')
        "\n        When using an encrypted / secured PDF file with the PDF Standard\n        encryption handler, this function will allow the file to be decrypted.\n        It checks the given password against the document's user password and\n        owner password, and then stores the resulting decryption key if either\n        password is correct.\n\n        It does not matter which password was matched.  Both passwords provide\n        the correct decryption key that will allow the document to be used with\n        this library.\n\n        Args:\n            password: The password to match.\n\n        Returns:\n            An indicator if the document was decrypted and weather it was the\n            owner password or the user password.\n        "
        if not self._encryption:
            raise PdfReadError('Not encrypted file')
        return self._encryption.verify(password)

    def decode_permissions(self, permissions_code: int) -> Dict[str, bool]:
        if False:
            print('Hello World!')
        permissions = {}
        permissions['print'] = permissions_code & 1 << 3 - 1 != 0
        permissions['modify'] = permissions_code & 1 << 4 - 1 != 0
        permissions['copy'] = permissions_code & 1 << 5 - 1 != 0
        permissions['annotations'] = permissions_code & 1 << 6 - 1 != 0
        permissions['forms'] = permissions_code & 1 << 9 - 1 != 0
        permissions['accessability'] = permissions_code & 1 << 10 - 1 != 0
        permissions['assemble'] = permissions_code & 1 << 11 - 1 != 0
        permissions['print_high_quality'] = permissions_code & 1 << 12 - 1 != 0
        return permissions

    @property
    def is_encrypted(self) -> bool:
        if False:
            print('Hello World!')
        '\n        Read-only boolean property showing whether this PDF file is encrypted.\n\n        Note that this property, if true, will remain true even after the\n        :meth:`decrypt()<pypdf.PdfReader.decrypt>` method is called.\n        '
        return TK.ENCRYPT in self.trailer

    def getIsEncrypted(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Use :py:attr:`is_encrypted` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('getIsEncrypted', 'is_encrypted', '3.0.0')
        return self.is_encrypted

    @property
    def isEncrypted(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Use :py:attr:`is_encrypted` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('isEncrypted', 'is_encrypted', '3.0.0')
        return self.is_encrypted

    @property
    def xfa(self) -> Optional[Dict[str, Any]]:
        if False:
            print('Hello World!')
        tree: Optional[TreeObject] = None
        retval: Dict[str, Any] = {}
        catalog = cast(DictionaryObject, self.trailer[TK.ROOT])
        if '/AcroForm' not in catalog or not catalog['/AcroForm']:
            return None
        tree = cast(TreeObject, catalog['/AcroForm'])
        if '/XFA' in tree:
            fields = cast(ArrayObject, tree['/XFA'])
            i = iter(fields)
            for f in i:
                tag = f
                f = next(i)
                if isinstance(f, IndirectObject):
                    field = cast(Optional[EncodedStreamObject], f.get_object())
                    if field:
                        es = zlib.decompress(b_(field._data))
                        retval[tag] = es
        return retval

    def add_form_topname(self, name: str) -> Optional[DictionaryObject]:
        if False:
            print('Hello World!')
        '\n        Add a top level form that groups all form fields below it.\n\n        Args:\n            name: text string of the "/T" Attribute of the created object\n\n        Returns:\n            The created object. ``None`` means no object was created.\n        '
        catalog = cast(DictionaryObject, self.trailer[TK.ROOT])
        if '/AcroForm' not in catalog or not isinstance(catalog['/AcroForm'], DictionaryObject):
            return None
        acroform = cast(DictionaryObject, catalog[NameObject('/AcroForm')])
        if '/Fields' not in acroform:
            return None
        interim = DictionaryObject()
        interim[NameObject('/T')] = TextStringObject(name)
        interim[NameObject('/Kids')] = acroform[NameObject('/Fields')]
        self.cache_indirect_object(0, max([i for (g, i) in self.resolved_objects if g == 0]) + 1, interim)
        arr = ArrayObject()
        arr.append(interim.indirect_reference)
        acroform[NameObject('/Fields')] = arr
        for o in cast(ArrayObject, interim['/Kids']):
            obj = o.get_object()
            if '/Parent' in obj:
                logger_warning(f'Top Level Form Field {obj.indirect_reference} have a non-expected parent', __name__)
            obj[NameObject('/Parent')] = interim.indirect_reference
        return interim

    def rename_form_topname(self, name: str) -> Optional[DictionaryObject]:
        if False:
            print('Hello World!')
        '\n        Rename top level form field that all form fields below it.\n\n        Args:\n            name: text string of the "/T" field of the created object\n\n        Returns:\n            The modified object. ``None`` means no object was modified.\n        '
        catalog = cast(DictionaryObject, self.trailer[TK.ROOT])
        if '/AcroForm' not in catalog or not isinstance(catalog['/AcroForm'], DictionaryObject):
            return None
        acroform = cast(DictionaryObject, catalog[NameObject('/AcroForm')])
        if '/Fields' not in acroform:
            return None
        interim = cast(DictionaryObject, cast(ArrayObject, acroform[NameObject('/Fields')])[0].get_object())
        interim[NameObject('/T')] = TextStringObject(name)
        return interim

    @property
    def attachments(self) -> Mapping[str, List[bytes]]:
        if False:
            return 10
        return LazyDict({name: (self._get_attachment_list, name) for name in self._list_attachments()})

    def _list_attachments(self) -> List[str]:
        if False:
            i = 10
            return i + 15
        '\n        Retrieves the list of filenames of file attachments.\n\n        Returns:\n            list of filenames\n        '
        catalog = cast(DictionaryObject, self.trailer['/Root'])
        try:
            filenames = cast(ArrayObject, cast(DictionaryObject, cast(DictionaryObject, catalog['/Names'])['/EmbeddedFiles'])['/Names'])
        except KeyError:
            return []
        attachments_names = [f for f in filenames if isinstance(f, str)]
        return attachments_names

    def _get_attachment_list(self, name: str) -> List[bytes]:
        if False:
            return 10
        out = self._get_attachments(name)[name]
        if isinstance(out, list):
            return out
        return [out]

    def _get_attachments(self, filename: Optional[str]=None) -> Dict[str, Union[bytes, List[bytes]]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieves all or selected file attachments of the PDF as a dictionary of file names\n        and the file data as a bytestring.\n\n        Args:\n            filename: If filename is None, then a dictionary of all attachments\n                will be returned, where the key is the filename and the value\n                is the content. Otherwise, a dictionary with just a single key\n                - the filename - and its content will be returned.\n\n        Returns:\n            dictionary of filename -> Union[bytestring or List[ByteString]]\n            if the filename exists multiple times a List of the different version will be provided\n        '
        catalog = cast(DictionaryObject, self.trailer['/Root'])
        try:
            filenames = cast(ArrayObject, cast(DictionaryObject, cast(DictionaryObject, catalog['/Names'])['/EmbeddedFiles'])['/Names'])
        except KeyError:
            return {}
        attachments: Dict[str, Union[bytes, List[bytes]]] = {}
        for i in range(len(filenames)):
            f = filenames[i]
            if isinstance(f, str):
                if filename is not None and f != filename:
                    continue
                name = f
                f_dict = filenames[i + 1].get_object()
                f_data = f_dict['/EF']['/F'].get_data()
                if name in attachments:
                    if not isinstance(attachments[name], list):
                        attachments[name] = [attachments[name]]
                    attachments[name].append(f_data)
                else:
                    attachments[name] = f_data
        return attachments

class LazyDict(Mapping[Any, Any]):

    def __init__(self, *args: Any, **kw: Any) -> None:
        if False:
            while True:
                i = 10
        self._raw_dict = dict(*args, **kw)

    def __getitem__(self, key: str) -> Any:
        if False:
            while True:
                i = 10
        (func, arg) = self._raw_dict.__getitem__(key)
        return func(arg)

    def __iter__(self) -> Iterator[Any]:
        if False:
            return 10
        return iter(self._raw_dict)

    def __len__(self) -> int:
        if False:
            i = 10
            return i + 15
        return len(self._raw_dict)

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'LazyDict(keys={list(self.keys())})'

class PdfFileReader(PdfReader):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        deprecation_with_replacement('PdfFileReader', 'PdfReader', '3.0.0')
        if 'strict' not in kwargs and len(args) < 2:
            kwargs['strict'] = True
        super().__init__(*args, **kwargs)