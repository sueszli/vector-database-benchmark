__author__ = 'Mathieu Fenniak'
__author_email__ = 'biziqe@mathieu.fenniak.net'
import logging
import re
from io import BytesIO
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, cast
from .._protocols import PdfReaderProtocol, PdfWriterProtocol
from .._utils import WHITESPACES, StreamType, b_, deprecate_no_replacement, deprecate_with_replacement, deprecation_with_replacement, logger_warning, read_non_whitespace, read_until_regex, skip_over_comment
from ..constants import CheckboxRadioButtonAttributes, FieldDictionaryAttributes, OutlineFontFlag
from ..constants import FilterTypes as FT
from ..constants import StreamAttributes as SA
from ..constants import TypArguments as TA
from ..constants import TypFitArguments as TF
from ..errors import STREAM_TRUNCATED_PREMATURELY, PdfReadError, PdfStreamError
from ._base import BooleanObject, FloatObject, IndirectObject, NameObject, NullObject, NumberObject, PdfObject, TextStringObject
from ._fit import Fit
from ._utils import read_hex_string_from_stream, read_string_from_stream
logger = logging.getLogger(__name__)
NumberSigns = b'+-'
IndirectPattern = re.compile(b'[+-]?(\\d+)\\s+(\\d+)\\s+R[^a-zA-Z]')

class ArrayObject(List[Any], PdfObject):

    def clone(self, pdf_dest: PdfWriterProtocol, force_duplicate: bool=False, ignore_fields: Optional[Sequence[Union[str, int]]]=()) -> 'ArrayObject':
        if False:
            return 10
        'Clone object into pdf_dest.'
        try:
            if self.indirect_reference.pdf == pdf_dest and (not force_duplicate):
                return self
        except Exception:
            pass
        arr = cast('ArrayObject', self._reference_clone(ArrayObject(), pdf_dest, force_duplicate))
        for data in self:
            if isinstance(data, StreamObject):
                dup = data._reference_clone(data.clone(pdf_dest, force_duplicate, ignore_fields), pdf_dest, force_duplicate)
                arr.append(dup.indirect_reference)
            elif hasattr(data, 'clone'):
                arr.append(data.clone(pdf_dest, force_duplicate, ignore_fields))
            else:
                arr.append(data)
        return arr

    def items(self) -> Iterable[Any]:
        if False:
            while True:
                i = 10
        'Emulate DictionaryObject.items for a list (index, object).'
        return enumerate(self)

    def write_to_stream(self, stream: StreamType, encryption_key: Union[None, str, bytes]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        if encryption_key is not None:
            deprecate_no_replacement('the encryption_key parameter of write_to_stream', '5.0.0')
        stream.write(b'[')
        for data in self:
            stream.write(b' ')
            data.write_to_stream(stream)
        stream.write(b' ]')

    def writeToStream(self, stream: StreamType, encryption_key: Union[None, str, bytes]) -> None:
        if False:
            return 10
        deprecation_with_replacement('writeToStream', 'write_to_stream', '3.0.0')
        self.write_to_stream(stream)

    @staticmethod
    def read_from_stream(stream: StreamType, pdf: Optional[PdfReaderProtocol], forced_encoding: Union[None, str, List[str], Dict[int, str]]=None) -> 'ArrayObject':
        if False:
            print('Hello World!')
        arr = ArrayObject()
        tmp = stream.read(1)
        if tmp != b'[':
            raise PdfReadError('Could not read array')
        while True:
            tok = stream.read(1)
            while tok.isspace():
                tok = stream.read(1)
            stream.seek(-1, 1)
            peek_ahead = stream.read(1)
            if peek_ahead == b']':
                break
            stream.seek(-1, 1)
            arr.append(read_object(stream, pdf, forced_encoding))
        return arr

    @staticmethod
    def readFromStream(stream: StreamType, pdf: PdfReaderProtocol) -> 'ArrayObject':
        if False:
            for i in range(10):
                print('nop')
        deprecation_with_replacement('readFromStream', 'read_from_stream', '3.0.0')
        return ArrayObject.read_from_stream(stream, pdf)

class DictionaryObject(Dict[Any, Any], PdfObject):

    def clone(self, pdf_dest: PdfWriterProtocol, force_duplicate: bool=False, ignore_fields: Optional[Sequence[Union[str, int]]]=()) -> 'DictionaryObject':
        if False:
            print('Hello World!')
        'Clone object into pdf_dest.'
        try:
            if self.indirect_reference.pdf == pdf_dest and (not force_duplicate):
                return self
        except Exception:
            pass
        visited: Set[Tuple[int, int]] = set()
        d__ = cast('DictionaryObject', self._reference_clone(self.__class__(), pdf_dest, force_duplicate))
        if ignore_fields is None:
            ignore_fields = []
        if len(d__.keys()) == 0:
            d__._clone(self, pdf_dest, force_duplicate, ignore_fields, visited)
        return d__

    def _clone(self, src: 'DictionaryObject', pdf_dest: PdfWriterProtocol, force_duplicate: bool, ignore_fields: Optional[Sequence[Union[str, int]]], visited: Set[Tuple[int, int]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Update the object from src.\n\n        Args:\n            src: "DictionaryObject":\n            pdf_dest:\n            force_duplicate:\n            ignore_fields:\n        '
        x = 0
        assert ignore_fields is not None
        ignore_fields = list(ignore_fields)
        while x < len(ignore_fields):
            if isinstance(ignore_fields[x], int):
                if cast(int, ignore_fields[x]) <= 0:
                    del ignore_fields[x]
                    del ignore_fields[x]
                    continue
                else:
                    ignore_fields[x] -= 1
            x += 1
        if any((field not in ignore_fields and field in src and isinstance(src.raw_get(field), IndirectObject) and isinstance(src[field], DictionaryObject) and (src.get('/Type', None) is None or cast(DictionaryObject, src[field]).get('/Type', None) is None or src.get('/Type', None) == cast(DictionaryObject, src[field]).get('/Type', None)) for field in ['/Next', '/Prev', '/N', '/V'])):
            ignore_fields = list(ignore_fields)
            for lst in (('/Next', '/Prev'), ('/N', '/V')):
                for k in lst:
                    objs = []
                    if k in src and k not in self and isinstance(src.raw_get(k), IndirectObject) and isinstance(src[k], DictionaryObject) and (src.get('/Type', None) is None or cast(DictionaryObject, src[k]).get('/Type', None) is None or src.get('/Type', None) == cast(DictionaryObject, src[k]).get('/Type', None)):
                        cur_obj: Optional[DictionaryObject] = cast('DictionaryObject', src[k])
                        prev_obj: Optional[DictionaryObject] = self
                        while cur_obj is not None:
                            clon = cast('DictionaryObject', cur_obj._reference_clone(cur_obj.__class__(), pdf_dest, force_duplicate))
                            if clon.indirect_reference is not None:
                                idnum = clon.indirect_reference.idnum
                                generation = clon.indirect_reference.generation
                                if (idnum, generation) in visited:
                                    cur_obj = None
                                    break
                                visited.add((idnum, generation))
                            objs.append((cur_obj, clon))
                            assert prev_obj is not None
                            prev_obj[NameObject(k)] = clon.indirect_reference
                            prev_obj = clon
                            try:
                                if cur_obj == src:
                                    cur_obj = None
                                else:
                                    cur_obj = cast('DictionaryObject', cur_obj[k])
                            except Exception:
                                cur_obj = None
                        for (s, c) in objs:
                            c._clone(s, pdf_dest, force_duplicate, ignore_fields, visited)
        for (k, v) in src.items():
            if k not in ignore_fields:
                if isinstance(v, StreamObject):
                    if not hasattr(v, 'indirect_reference'):
                        v.indirect_reference = None
                    vv = v.clone(pdf_dest, force_duplicate, ignore_fields)
                    assert vv.indirect_reference is not None
                    self[k.clone(pdf_dest)] = vv.indirect_reference
                elif k not in self:
                    self[NameObject(k)] = v.clone(pdf_dest, force_duplicate, ignore_fields) if hasattr(v, 'clone') else v

    def raw_get(self, key: Any) -> Any:
        if False:
            print('Hello World!')
        return dict.__getitem__(self, key)

    def __setitem__(self, key: Any, value: Any) -> Any:
        if False:
            while True:
                i = 10
        if not isinstance(key, PdfObject):
            raise ValueError('key must be PdfObject')
        if not isinstance(value, PdfObject):
            raise ValueError('value must be PdfObject')
        return dict.__setitem__(self, key, value)

    def setdefault(self, key: Any, value: Optional[Any]=None) -> Any:
        if False:
            while True:
                i = 10
        if not isinstance(key, PdfObject):
            raise ValueError('key must be PdfObject')
        if not isinstance(value, PdfObject):
            raise ValueError('value must be PdfObject')
        return dict.setdefault(self, key, value)

    def __getitem__(self, key: Any) -> PdfObject:
        if False:
            print('Hello World!')
        return dict.__getitem__(self, key).get_object()

    @property
    def xmp_metadata(self) -> Optional[PdfObject]:
        if False:
            i = 10
            return i + 15
        '\n        Retrieve XMP (Extensible Metadata Platform) data relevant to the this\n        object, if available.\n\n        Stability: Added in v1.12, will exist for all future v1.x releases.\n        See Table 315 – Additional entries in a metadata stream dictionary\n\n        Returns:\n          Returns a {@link #xmp.XmpInformation XmlInformation} instance\n          that can be used to access XMP metadata from the document.  Can also\n          return None if no metadata was found on the document root.\n        '
        from ..xmp import XmpInformation
        metadata = self.get('/Metadata', None)
        if metadata is None:
            return None
        metadata = metadata.get_object()
        if not isinstance(metadata, XmpInformation):
            metadata = XmpInformation(metadata)
            self[NameObject('/Metadata')] = metadata
        return metadata

    def getXmpMetadata(self) -> Optional[PdfObject]:
        if False:
            while True:
                i = 10
        '\n        Use :meth:`xmp_metadata` instead.\n\n        .. deprecated:: 1.28.3\n        '
        deprecation_with_replacement('getXmpMetadata', 'xmp_metadata', '3.0.0')
        return self.xmp_metadata

    @property
    def xmpMetadata(self) -> Optional[PdfObject]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Use :meth:`xmp_metadata` instead.\n\n        .. deprecated:: 1.28.3\n        '
        deprecation_with_replacement('xmpMetadata', 'xmp_metadata', '3.0.0')
        return self.xmp_metadata

    def write_to_stream(self, stream: StreamType, encryption_key: Union[None, str, bytes]=None) -> None:
        if False:
            while True:
                i = 10
        if encryption_key is not None:
            deprecate_no_replacement('the encryption_key parameter of write_to_stream', '5.0.0')
        stream.write(b'<<\n')
        for (key, value) in list(self.items()):
            if len(key) > 2 and key[1] == '%' and (key[-1] == '%'):
                continue
            key.write_to_stream(stream, encryption_key)
            stream.write(b' ')
            value.write_to_stream(stream)
            stream.write(b'\n')
        stream.write(b'>>')

    def writeToStream(self, stream: StreamType, encryption_key: Union[None, str, bytes]) -> None:
        if False:
            i = 10
            return i + 15
        deprecation_with_replacement('writeToStream', 'write_to_stream', '3.0.0')
        self.write_to_stream(stream)

    @staticmethod
    def read_from_stream(stream: StreamType, pdf: Optional[PdfReaderProtocol], forced_encoding: Union[None, str, List[str], Dict[int, str]]=None) -> 'DictionaryObject':
        if False:
            i = 10
            return i + 15

        def get_next_obj_pos(p: int, p1: int, rem_gens: List[int], pdf: PdfReaderProtocol) -> int:
            if False:
                return 10
            loc = pdf.xref[rem_gens[0]]
            for o in loc:
                if p1 > loc[o] and p < loc[o]:
                    p1 = loc[o]
            if len(rem_gens) == 1:
                return p1
            else:
                return get_next_obj_pos(p, p1, rem_gens[1:], pdf)

        def read_unsized_from_steam(stream: StreamType, pdf: PdfReaderProtocol) -> bytes:
            if False:
                return 10
            eon = get_next_obj_pos(stream.tell(), 2 ** 32, list(pdf.xref), pdf) - 1
            curr = stream.tell()
            rw = stream.read(eon - stream.tell())
            p = rw.find(b'endstream')
            if p < 0:
                raise PdfReadError(f"Unable to find 'endstream' marker for obj starting at {curr}.")
            stream.seek(curr + p + 9)
            return rw[:p - 1]
        tmp = stream.read(2)
        if tmp != b'<<':
            raise PdfReadError(f"Dictionary read error at byte {hex(stream.tell())}: stream must begin with '<<'")
        data: Dict[Any, Any] = {}
        while True:
            tok = read_non_whitespace(stream)
            if tok == b'\x00':
                continue
            elif tok == b'%':
                stream.seek(-1, 1)
                skip_over_comment(stream)
                continue
            if not tok:
                raise PdfStreamError(STREAM_TRUNCATED_PREMATURELY)
            if tok == b'>':
                stream.read(1)
                break
            stream.seek(-1, 1)
            try:
                key = read_object(stream, pdf)
                tok = read_non_whitespace(stream)
                stream.seek(-1, 1)
                value = read_object(stream, pdf, forced_encoding)
            except Exception as exc:
                if pdf is not None and pdf.strict:
                    raise PdfReadError(exc.__repr__())
                logger_warning(exc.__repr__(), __name__)
                retval = DictionaryObject()
                retval.update(data)
                return retval
            if not data.get(key):
                data[key] = value
            else:
                msg = f'Multiple definitions in dictionary at byte {hex(stream.tell())} for key {key}'
                if pdf is not None and pdf.strict:
                    raise PdfReadError(msg)
                logger_warning(msg, __name__)
        pos = stream.tell()
        s = read_non_whitespace(stream)
        if s == b's' and stream.read(5) == b'tream':
            eol = stream.read(1)
            while eol == b' ':
                eol = stream.read(1)
            if eol not in (b'\n', b'\r'):
                raise PdfStreamError('Stream data must be followed by a newline')
            if eol == b'\r' and stream.read(1) != b'\n':
                stream.seek(-1, 1)
            if SA.LENGTH not in data:
                if pdf is not None and pdf.strict:
                    raise PdfStreamError('Stream length not defined')
                else:
                    logger_warning(f'Stream length not defined @pos={stream.tell()}', __name__)
                data[NameObject(SA.LENGTH)] = NumberObject(-1)
            length = data[SA.LENGTH]
            if isinstance(length, IndirectObject):
                t = stream.tell()
                assert pdf is not None
                length = pdf.get_object(length)
                stream.seek(t, 0)
            if length is None:
                length = -1
            pstart = stream.tell()
            if length > 0:
                data['__streamdata__'] = stream.read(length)
            else:
                data['__streamdata__'] = read_until_regex(stream, re.compile(b'endstream'))
            e = read_non_whitespace(stream)
            ndstream = stream.read(8)
            if e + ndstream != b'endstream':
                pos = stream.tell()
                stream.seek(-10, 1)
                end = stream.read(9)
                if end == b'endstream':
                    data['__streamdata__'] = data['__streamdata__'][:-1]
                elif pdf is not None and (not pdf.strict):
                    stream.seek(pstart, 0)
                    data['__streamdata__'] = read_unsized_from_steam(stream, pdf)
                    pos = stream.tell()
                else:
                    stream.seek(pos, 0)
                    raise PdfReadError(f"Unable to find 'endstream' marker after stream at byte {hex(stream.tell())} (nd='{ndstream!r}', end='{end!r}').")
        else:
            stream.seek(pos, 0)
        if '__streamdata__' in data:
            return StreamObject.initialize_from_dictionary(data)
        else:
            retval = DictionaryObject()
            retval.update(data)
            return retval

    @staticmethod
    def readFromStream(stream: StreamType, pdf: PdfReaderProtocol) -> 'DictionaryObject':
        if False:
            for i in range(10):
                print('nop')
        deprecation_with_replacement('readFromStream', 'read_from_stream', '3.0.0')
        return DictionaryObject.read_from_stream(stream, pdf)

class TreeObject(DictionaryObject):

    def __init__(self, dct: Optional[DictionaryObject]=None) -> None:
        if False:
            print('Hello World!')
        DictionaryObject.__init__(self)
        if dct:
            self.update(dct)

    def hasChildren(self) -> bool:
        if False:
            while True:
                i = 10
        deprecate_with_replacement('hasChildren', 'has_children', '4.0.0')
        return self.has_children()

    def has_children(self) -> bool:
        if False:
            while True:
                i = 10
        return '/First' in self

    def __iter__(self) -> Any:
        if False:
            print('Hello World!')
        return self.children()

    def children(self) -> Iterable[Any]:
        if False:
            i = 10
            return i + 15
        if not self.has_children():
            return
        child_ref = self[NameObject('/First')]
        child = child_ref.get_object()
        while True:
            yield child
            if child == self[NameObject('/Last')]:
                return
            child_ref = child.get(NameObject('/Next'))
            if child_ref is None:
                return
            child = child_ref.get_object()

    def addChild(self, child: Any, pdf: Any) -> None:
        if False:
            i = 10
            return i + 15
        deprecation_with_replacement('addChild', 'add_child', '3.0.0')
        self.add_child(child, pdf)

    def add_child(self, child: Any, pdf: PdfWriterProtocol) -> None:
        if False:
            return 10
        self.insert_child(child, None, pdf)

    def inc_parent_counter_default(self, parent: Union[None, IndirectObject, 'TreeObject'], n: int) -> None:
        if False:
            print('Hello World!')
        if parent is None:
            return
        parent = cast('TreeObject', parent.get_object())
        if '/Count' in parent:
            parent[NameObject('/Count')] = NumberObject(max(0, cast(int, parent[NameObject('/Count')]) + n))
            self.inc_parent_counter_default(parent.get('/Parent', None), n)

    def inc_parent_counter_outline(self, parent: Union[None, IndirectObject, 'TreeObject'], n: int) -> None:
        if False:
            while True:
                i = 10
        if parent is None:
            return
        parent = cast('TreeObject', parent.get_object())
        opn = parent.get('/%is_open%', True) == True
        c = cast(int, parent.get('/Count', 0))
        if c < 0:
            c = abs(c)
        parent[NameObject('/Count')] = NumberObject((c + n) * (1 if opn else -1))
        if not opn:
            return
        self.inc_parent_counter_outline(parent.get('/Parent', None), n)

    def insert_child(self, child: Any, before: Any, pdf: PdfWriterProtocol, inc_parent_counter: Optional[Callable[..., Any]]=None) -> IndirectObject:
        if False:
            while True:
                i = 10
        if inc_parent_counter is None:
            inc_parent_counter = self.inc_parent_counter_default
        child_obj = child.get_object()
        child = child.indirect_reference
        prev: Optional[DictionaryObject]
        if '/First' not in self:
            self[NameObject('/First')] = child
            self[NameObject('/Count')] = NumberObject(0)
            self[NameObject('/Last')] = child
            child_obj[NameObject('/Parent')] = self.indirect_reference
            inc_parent_counter(self, child_obj.get('/Count', 1))
            if '/Next' in child_obj:
                del child_obj['/Next']
            if '/Prev' in child_obj:
                del child_obj['/Prev']
            return child
        else:
            prev = cast('DictionaryObject', self['/Last'])
        while prev.indirect_reference != before:
            if '/Next' in prev:
                prev = cast('TreeObject', prev['/Next'])
            else:
                prev[NameObject('/Next')] = cast('TreeObject', child)
                child_obj[NameObject('/Prev')] = prev.indirect_reference
                child_obj[NameObject('/Parent')] = self.indirect_reference
                if '/Next' in child_obj:
                    del child_obj['/Next']
                self[NameObject('/Last')] = child
                inc_parent_counter(self, child_obj.get('/Count', 1))
                return child
        try:
            assert isinstance(prev['/Prev'], DictionaryObject)
            prev['/Prev'][NameObject('/Next')] = child
            child_obj[NameObject('/Prev')] = prev['/Prev']
        except Exception:
            del child_obj['/Next']
        child_obj[NameObject('/Next')] = prev
        prev[NameObject('/Prev')] = child
        child_obj[NameObject('/Parent')] = self.indirect_reference
        inc_parent_counter(self, child_obj.get('/Count', 1))
        return child

    def removeChild(self, child: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        deprecation_with_replacement('removeChild', 'remove_child', '3.0.0')
        self.remove_child(child)

    def _remove_node_from_tree(self, prev: Any, prev_ref: Any, cur: Any, last: Any) -> None:
        if False:
            while True:
                i = 10
        '\n        Adjust the pointers of the linked list and tree node count.\n\n        Args:\n            prev:\n            prev_ref:\n            cur:\n            last:\n        '
        next_ref = cur.get(NameObject('/Next'), None)
        if prev is None:
            if next_ref:
                next_obj = next_ref.get_object()
                del next_obj[NameObject('/Prev')]
                self[NameObject('/First')] = next_ref
                self[NameObject('/Count')] = NumberObject(self[NameObject('/Count')] - 1)
            else:
                self[NameObject('/Count')] = NumberObject(0)
                del self[NameObject('/First')]
                if NameObject('/Last') in self:
                    del self[NameObject('/Last')]
        else:
            if next_ref:
                next_obj = next_ref.get_object()
                next_obj[NameObject('/Prev')] = prev_ref
                prev[NameObject('/Next')] = next_ref
            else:
                assert cur == last
                del prev[NameObject('/Next')]
                self[NameObject('/Last')] = prev_ref
            self[NameObject('/Count')] = NumberObject(self[NameObject('/Count')] - 1)

    def remove_child(self, child: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        child_obj = child.get_object()
        child = child_obj.indirect_reference
        if NameObject('/Parent') not in child_obj:
            raise ValueError('Removed child does not appear to be a tree item')
        elif child_obj[NameObject('/Parent')] != self:
            raise ValueError('Removed child is not a member of this tree')
        found = False
        prev_ref = None
        prev = None
        cur_ref: Optional[Any] = self[NameObject('/First')]
        cur: Optional[Dict[str, Any]] = cur_ref.get_object()
        last_ref = self[NameObject('/Last')]
        last = last_ref.get_object()
        while cur is not None:
            if cur == child_obj:
                self._remove_node_from_tree(prev, prev_ref, cur, last)
                found = True
                break
            prev_ref = cur_ref
            prev = cur
            if NameObject('/Next') in cur:
                cur_ref = cur[NameObject('/Next')]
                cur = cur_ref.get_object()
            else:
                cur_ref = None
                cur = None
        if not found:
            raise ValueError("Removal couldn't find item in tree")
        _reset_node_tree_relationship(child_obj)

    def remove_from_tree(self) -> None:
        if False:
            return 10
        'Remove the object from the tree it is in.'
        if NameObject('/Parent') not in self:
            raise ValueError('Removed child does not appear to be a tree item')
        else:
            cast('TreeObject', self['/Parent']).remove_child(self)

    def emptyTree(self) -> None:
        if False:
            while True:
                i = 10
        deprecate_with_replacement('emptyTree', 'empty_tree', '4.0.0')
        self.empty_tree()

    def empty_tree(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        for child in self:
            child_obj = child.get_object()
            _reset_node_tree_relationship(child_obj)
        if NameObject('/Count') in self:
            del self[NameObject('/Count')]
        if NameObject('/First') in self:
            del self[NameObject('/First')]
        if NameObject('/Last') in self:
            del self[NameObject('/Last')]

def _reset_node_tree_relationship(child_obj: Any) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Call this after a node has been removed from a tree.\n\n    This resets the nodes attributes in respect to that tree.\n\n    Args:\n        child_obj:\n    '
    del child_obj[NameObject('/Parent')]
    if NameObject('/Next') in child_obj:
        del child_obj[NameObject('/Next')]
    if NameObject('/Prev') in child_obj:
        del child_obj[NameObject('/Prev')]

class StreamObject(DictionaryObject):

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self._data: Union[bytes, str] = b''
        self.decoded_self: Optional[DecodedStreamObject] = None

    def _clone(self, src: DictionaryObject, pdf_dest: PdfWriterProtocol, force_duplicate: bool, ignore_fields: Optional[Sequence[Union[str, int]]], visited: Set[Tuple[int, int]]) -> None:
        if False:
            while True:
                i = 10
        '\n        Update the object from src.\n\n        Args:\n            src:\n            pdf_dest:\n            force_duplicate:\n            ignore_fields:\n        '
        self._data = cast('StreamObject', src)._data
        try:
            decoded_self = cast('StreamObject', src).decoded_self
            if decoded_self is None:
                self.decoded_self = None
            else:
                self.decoded_self = cast('DecodedStreamObject', decoded_self.clone(pdf_dest, force_duplicate, ignore_fields))
        except Exception:
            pass
        super()._clone(src, pdf_dest, force_duplicate, ignore_fields, visited)

    def get_data(self) -> Union[bytes, str]:
        if False:
            print('Hello World!')
        return self._data

    def set_data(self, data: bytes) -> None:
        if False:
            return 10
        self._data = data

    def getData(self) -> Any:
        if False:
            for i in range(10):
                print('nop')
        deprecation_with_replacement('getData', 'get_data', '3.0.0')
        return self._data

    def setData(self, data: Any) -> None:
        if False:
            while True:
                i = 10
        deprecation_with_replacement('setData', 'set_data', '3.0.0')
        self.set_data(data)

    def hash_value_data(self) -> bytes:
        if False:
            print('Hello World!')
        data = super().hash_value_data()
        data += b_(self._data)
        return data

    @property
    def decodedSelf(self) -> Optional['DecodedStreamObject']:
        if False:
            print('Hello World!')
        deprecation_with_replacement('decodedSelf', 'decoded_self', '3.0.0')
        return self.decoded_self

    @decodedSelf.setter
    def decodedSelf(self, value: 'DecodedStreamObject') -> None:
        if False:
            while True:
                i = 10
        deprecation_with_replacement('decodedSelf', 'decoded_self', '3.0.0')
        self.decoded_self = value

    def write_to_stream(self, stream: StreamType, encryption_key: Union[None, str, bytes]=None) -> None:
        if False:
            print('Hello World!')
        if encryption_key is not None:
            deprecate_no_replacement('the encryption_key parameter of write_to_stream', '5.0.0')
        self[NameObject(SA.LENGTH)] = NumberObject(len(self._data))
        DictionaryObject.write_to_stream(self, stream)
        del self[SA.LENGTH]
        stream.write(b'\nstream\n')
        stream.write(self._data)
        stream.write(b'\nendstream')

    @staticmethod
    def initializeFromDictionary(data: Dict[str, Any]) -> Union['EncodedStreamObject', 'DecodedStreamObject']:
        if False:
            i = 10
            return i + 15
        return StreamObject.initialize_from_dictionary(data)

    @staticmethod
    def initialize_from_dictionary(data: Dict[str, Any]) -> Union['EncodedStreamObject', 'DecodedStreamObject']:
        if False:
            return 10
        retval: Union[EncodedStreamObject, DecodedStreamObject]
        if SA.FILTER in data:
            retval = EncodedStreamObject()
        else:
            retval = DecodedStreamObject()
        retval._data = data['__streamdata__']
        del data['__streamdata__']
        del data[SA.LENGTH]
        retval.update(data)
        return retval

    def flateEncode(self) -> 'EncodedStreamObject':
        if False:
            print('Hello World!')
        deprecation_with_replacement('flateEncode', 'flate_encode', '3.0.0')
        return self.flate_encode()

    def flate_encode(self, level: int=-1) -> 'EncodedStreamObject':
        if False:
            while True:
                i = 10
        from ..filters import FlateDecode
        if SA.FILTER in self:
            f = self[SA.FILTER]
            if isinstance(f, ArrayObject):
                f = ArrayObject([NameObject(FT.FLATE_DECODE), *f])
                try:
                    parms = ArrayObject([NullObject(), *self.get(SA.DECODE_PARMS, ArrayObject())])
                except TypeError:
                    parms = ArrayObject([NullObject(), self.get(SA.DECODE_PARMS, ArrayObject())])
            else:
                f = ArrayObject([NameObject(FT.FLATE_DECODE), f])
                parms = ArrayObject([NullObject(), self.get(SA.DECODE_PARMS, NullObject())])
        else:
            f = NameObject(FT.FLATE_DECODE)
            parms = None
        retval = EncodedStreamObject()
        retval.update(self)
        retval[NameObject(SA.FILTER)] = f
        if parms is not None:
            retval[NameObject(SA.DECODE_PARMS)] = parms
        retval._data = FlateDecode.encode(b_(self._data), level)
        return retval

class DecodedStreamObject(StreamObject):
    pass

class EncodedStreamObject(StreamObject):

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.decoded_self: Optional[DecodedStreamObject] = None

    @property
    def decodedSelf(self) -> Optional['DecodedStreamObject']:
        if False:
            for i in range(10):
                print('nop')
        deprecation_with_replacement('decodedSelf', 'decoded_self', '3.0.0')
        return self.decoded_self

    @decodedSelf.setter
    def decodedSelf(self, value: DecodedStreamObject) -> None:
        if False:
            return 10
        deprecation_with_replacement('decodedSelf', 'decoded_self', '3.0.0')
        self.decoded_self = value

    def get_data(self) -> Union[bytes, str]:
        if False:
            i = 10
            return i + 15
        from ..filters import decode_stream_data
        if self.decoded_self is not None:
            return self.decoded_self.get_data()
        else:
            decoded = DecodedStreamObject()
            decoded.set_data(b_(decode_stream_data(self)))
            for (key, value) in list(self.items()):
                if key not in (SA.LENGTH, SA.FILTER, SA.DECODE_PARMS):
                    decoded[key] = value
            self.decoded_self = decoded
            return decoded.get_data()

    def set_data(self, data: bytes) -> None:
        if False:
            for i in range(10):
                print('nop')
        from ..filters import FlateDecode
        if self.get(SA.FILTER, '') == FT.FLATE_DECODE:
            if not isinstance(data, bytes):
                raise TypeError('data must be bytes')
            assert self.decoded_self is not None
            self.decoded_self.set_data(data)
            super().set_data(FlateDecode.encode(data))
        else:
            raise PdfReadError('Streams encoded with different filter from only FlateDecode is not supported')

class ContentStream(DecodedStreamObject):
    """
    In order to be fast, this datastructure can contain either:
    * raw data in ._data
    * parsed stream operations in ._operations

    At any time, ContentStream object can either have one or both of those fields defined,
    and zero or one of those fields set to None.

    Those fields are "rebuilt" lazily, when accessed:
    * when .get_data() is called, if ._data is None, it is rebuilt from ._operations
    * when .operations is called, if ._operations is None, it is rebuilt from ._data

    On the other side, those fields can be invalidated:
    * when .set_data() is called, ._operations is set to None
    * when .operations is set, ._data is set to None
    """

    def __init__(self, stream: Any, pdf: Any, forced_encoding: Union[None, str, List[str], Dict[int, str]]=None) -> None:
        if False:
            while True:
                i = 10
        self.pdf = pdf
        self._operations: List[Tuple[Any, Any]] = []
        if stream is None:
            super().set_data(b'')
        else:
            stream = stream.get_object()
            if isinstance(stream, ArrayObject):
                data = b''
                for s in stream:
                    data += b_(s.get_object().get_data())
                    if len(data) == 0 or data[-1] != b'\n':
                        data += b'\n'
                super().set_data(bytes(data))
            else:
                stream_data = stream.get_data()
                assert stream_data is not None
                super().set_data(b_(stream_data))
            self.forced_encoding = forced_encoding

    def clone(self, pdf_dest: Any, force_duplicate: bool=False, ignore_fields: Optional[Sequence[Union[str, int]]]=()) -> 'ContentStream':
        if False:
            for i in range(10):
                print('nop')
        '\n        Clone object into pdf_dest.\n\n        Args:\n            pdf_dest:\n            force_duplicate:\n            ignore_fields:\n\n        Returns:\n            The cloned ContentStream\n        '
        try:
            if self.indirect_reference.pdf == pdf_dest and (not force_duplicate):
                return self
        except Exception:
            pass
        visited: Set[Tuple[int, int]] = set()
        d__ = cast('ContentStream', self._reference_clone(self.__class__(None, None), pdf_dest, force_duplicate))
        if ignore_fields is None:
            ignore_fields = []
        d__._clone(self, pdf_dest, force_duplicate, ignore_fields, visited)
        return d__

    def _clone(self, src: DictionaryObject, pdf_dest: PdfWriterProtocol, force_duplicate: bool, ignore_fields: Optional[Sequence[Union[str, int]]], visited: Set[Tuple[int, int]]) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Update the object from src.\n\n        Args:\n            src:\n            pdf_dest:\n            force_duplicate:\n            ignore_fields:\n        '
        src_cs = cast('ContentStream', src)
        super().set_data(b_(src_cs._data))
        self.pdf = pdf_dest
        self._operations = list(src_cs._operations)
        self.forced_encoding = src_cs.forced_encoding

    def _parse_content_stream(self, stream: StreamType) -> None:
        if False:
            print('Hello World!')
        stream.seek(0, 0)
        operands: List[Union[int, str, PdfObject]] = []
        while True:
            peek = read_non_whitespace(stream)
            if peek == b'' or peek == 0:
                break
            stream.seek(-1, 1)
            if peek.isalpha() or peek in (b"'", b'"'):
                operator = read_until_regex(stream, NameObject.delimiter_pattern)
                if operator == b'BI':
                    assert operands == []
                    ii = self._read_inline_image(stream)
                    self._operations.append((ii, b'INLINE IMAGE'))
                else:
                    self._operations.append((operands, operator))
                    operands = []
            elif peek == b'%':
                while peek not in (b'\r', b'\n', b''):
                    peek = stream.read(1)
            else:
                operands.append(read_object(stream, None, self.forced_encoding))

    def _read_inline_image(self, stream: StreamType) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        settings = DictionaryObject()
        while True:
            tok = read_non_whitespace(stream)
            stream.seek(-1, 1)
            if tok == b'I':
                break
            key = read_object(stream, self.pdf)
            tok = read_non_whitespace(stream)
            stream.seek(-1, 1)
            value = read_object(stream, self.pdf)
            settings[key] = value
        tmp = stream.read(3)
        assert tmp[:2] == b'ID'
        data = BytesIO()
        while True:
            buf = stream.read(8192)
            if not buf:
                raise PdfReadError('Unexpected end of stream')
            loc = buf.find(b'E')
            if loc == -1:
                data.write(buf)
            else:
                data.write(buf[0:loc])
                stream.seek(loc - len(buf), 1)
                tok = stream.read(1)
                tok2 = stream.read(1)
                if tok2 != b'I':
                    stream.seek(-1, 1)
                    data.write(tok)
                    continue
                info = tok + tok2
                tok3 = stream.read(1)
                if tok3 not in WHITESPACES:
                    stream.seek(-2, 1)
                    data.write(tok)
                elif buf[loc - 1:loc] in WHITESPACES:
                    while tok3 in WHITESPACES:
                        tok3 = stream.read(1)
                    stream.seek(-1, 1)
                    break
                else:
                    while tok3 in WHITESPACES:
                        info += tok3
                        tok3 = stream.read(1)
                    stream.seek(-1, 1)
                    if tok3 == b'Q':
                        break
                    elif tok3 == b'E':
                        ope = stream.read(3)
                        stream.seek(-3, 1)
                        if ope == b'EMC':
                            break
                    else:
                        data.write(info)
        return {'settings': settings, 'data': data.getvalue()}

    def get_data(self) -> bytes:
        if False:
            return 10
        if not self._data:
            new_data = BytesIO()
            for (operands, operator) in self._operations:
                if operator == b'INLINE IMAGE':
                    new_data.write(b'BI')
                    dict_text = BytesIO()
                    operands['settings'].write_to_stream(dict_text)
                    new_data.write(dict_text.getvalue()[2:-2])
                    new_data.write(b'ID ')
                    new_data.write(operands['data'])
                    new_data.write(b'EI')
                else:
                    for op in operands:
                        op.write_to_stream(new_data)
                        new_data.write(b' ')
                    new_data.write(b_(operator))
                new_data.write(b'\n')
            self._data = new_data.getvalue()
        return b_(self._data)

    def set_data(self, data: bytes) -> None:
        if False:
            i = 10
            return i + 15
        super().set_data(data)
        self._operations = []

    @property
    def operations(self) -> List[Tuple[Any, Any]]:
        if False:
            while True:
                i = 10
        if not self._operations and self._data:
            self._parse_content_stream(BytesIO(b_(self._data)))
            self._data = b''
        return self._operations

    @operations.setter
    def operations(self, operations: List[Tuple[Any, Any]]) -> None:
        if False:
            while True:
                i = 10
        self._operations = operations
        self._data = b''

    def isolate_graphics_state(self) -> None:
        if False:
            print('Hello World!')
        if self._operations:
            self._operations.insert(0, ([], 'q'))
            self._operations.append(([], 'Q'))
        elif self._data:
            self._data = b'q\n' + b_(self._data) + b'Q\n'

    def write_to_stream(self, stream: StreamType, encryption_key: Union[None, str, bytes]=None) -> None:
        if False:
            while True:
                i = 10
        if not self._data and self._operations:
            self.get_data()
        super().write_to_stream(stream, encryption_key)

def read_object(stream: StreamType, pdf: Optional[PdfReaderProtocol], forced_encoding: Union[None, str, List[str], Dict[int, str]]=None) -> Union[PdfObject, int, str, ContentStream]:
    if False:
        print('Hello World!')
    tok = stream.read(1)
    stream.seek(-1, 1)
    if tok == b'/':
        return NameObject.read_from_stream(stream, pdf)
    elif tok == b'<':
        peek = stream.read(2)
        stream.seek(-2, 1)
        if peek == b'<<':
            return DictionaryObject.read_from_stream(stream, pdf, forced_encoding)
        else:
            return read_hex_string_from_stream(stream, forced_encoding)
    elif tok == b'[':
        return ArrayObject.read_from_stream(stream, pdf, forced_encoding)
    elif tok == b't' or tok == b'f':
        return BooleanObject.read_from_stream(stream)
    elif tok == b'(':
        return read_string_from_stream(stream, forced_encoding)
    elif tok == b'e' and stream.read(6) == b'endobj':
        stream.seek(-6, 1)
        return NullObject()
    elif tok == b'n':
        return NullObject.read_from_stream(stream)
    elif tok == b'%':
        while tok not in (b'\r', b'\n'):
            tok = stream.read(1)
            if len(tok) <= 0:
                raise PdfStreamError('File ended unexpectedly.')
        tok = read_non_whitespace(stream)
        stream.seek(-1, 1)
        return read_object(stream, pdf, forced_encoding)
    elif tok in b'0123456789+-.':
        peek = stream.read(20)
        stream.seek(-len(peek), 1)
        if IndirectPattern.match(peek) is not None:
            assert pdf is not None
            return IndirectObject.read_from_stream(stream, pdf)
        else:
            return NumberObject.read_from_stream(stream)
    else:
        stream.seek(-20, 1)
        raise PdfReadError(f'Invalid Elementary Object starting with {tok!r} @{stream.tell()}: {stream.read(80).__repr__()}')

class Field(TreeObject):
    """
    A class representing a field dictionary.

    This class is accessed through
    :meth:`get_fields()<pypdf.PdfReader.get_fields>`
    """

    def __init__(self, data: DictionaryObject) -> None:
        if False:
            for i in range(10):
                print('nop')
        DictionaryObject.__init__(self)
        field_attributes = FieldDictionaryAttributes.attributes() + CheckboxRadioButtonAttributes.attributes()
        self.indirect_reference = data.indirect_reference
        for attr in field_attributes:
            try:
                self[NameObject(attr)] = data[attr]
            except KeyError:
                pass
        if isinstance(self.get('/V'), EncodedStreamObject):
            d = cast(EncodedStreamObject, self[NameObject('/V')]).get_data()
            if isinstance(d, bytes):
                d_str = d.decode()
            elif d is None:
                d_str = ''
            else:
                raise Exception('Should never happen')
            self[NameObject('/V')] = TextStringObject(d_str)

    @property
    def field_type(self) -> Optional[NameObject]:
        if False:
            return 10
        'Read-only property accessing the type of this field.'
        return self.get(FieldDictionaryAttributes.FT)

    @property
    def fieldType(self) -> Optional[NameObject]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Use :py:attr:`field_type` instead.\n\n        .. deprecated:: 1.28.3\n        '
        deprecation_with_replacement('fieldType', 'field_type', '3.0.0')
        return self.field_type

    @property
    def parent(self) -> Optional[DictionaryObject]:
        if False:
            print('Hello World!')
        'Read-only property accessing the parent of this field.'
        return self.get(FieldDictionaryAttributes.Parent)

    @property
    def kids(self) -> Optional['ArrayObject']:
        if False:
            print('Hello World!')
        'Read-only property accessing the kids of this field.'
        return self.get(FieldDictionaryAttributes.Kids)

    @property
    def name(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        'Read-only property accessing the name of this field.'
        return self.get(FieldDictionaryAttributes.T)

    @property
    def alternate_name(self) -> Optional[str]:
        if False:
            print('Hello World!')
        'Read-only property accessing the alternate name of this field.'
        return self.get(FieldDictionaryAttributes.TU)

    @property
    def altName(self) -> Optional[str]:
        if False:
            print('Hello World!')
        '\n        Use :py:attr:`alternate_name` instead.\n\n        .. deprecated:: 1.28.3\n        '
        deprecation_with_replacement('altName', 'alternate_name', '3.0.0')
        return self.alternate_name

    @property
    def mapping_name(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Read-only property accessing the mapping name of this field.\n\n        This name is used by pypdf as a key in the dictionary returned by\n        :meth:`get_fields()<pypdf.PdfReader.get_fields>`\n        '
        return self.get(FieldDictionaryAttributes.TM)

    @property
    def mappingName(self) -> Optional[str]:
        if False:
            print('Hello World!')
        '\n        Use :py:attr:`mapping_name` instead.\n\n        .. deprecated:: 1.28.3\n        '
        deprecation_with_replacement('mappingName', 'mapping_name', '3.0.0')
        return self.mapping_name

    @property
    def flags(self) -> Optional[int]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Read-only property accessing the field flags, specifying various\n        characteristics of the field (see Table 8.70 of the PDF 1.7 reference).\n        '
        return self.get(FieldDictionaryAttributes.Ff)

    @property
    def value(self) -> Optional[Any]:
        if False:
            while True:
                i = 10
        '\n        Read-only property accessing the value of this field.\n\n        Format varies based on field type.\n        '
        return self.get(FieldDictionaryAttributes.V)

    @property
    def default_value(self) -> Optional[Any]:
        if False:
            i = 10
            return i + 15
        'Read-only property accessing the default value of this field.'
        return self.get(FieldDictionaryAttributes.DV)

    @property
    def defaultValue(self) -> Optional[Any]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Use :py:attr:`default_value` instead.\n\n        .. deprecated:: 1.28.3\n        '
        deprecation_with_replacement('defaultValue', 'default_value', '3.0.0')
        return self.default_value

    @property
    def additional_actions(self) -> Optional[DictionaryObject]:
        if False:
            while True:
                i = 10
        "\n        Read-only property accessing the additional actions dictionary.\n\n        This dictionary defines the field's behavior in response to trigger\n        events. See Section 8.5.2 of the PDF 1.7 reference.\n        "
        return self.get(FieldDictionaryAttributes.AA)

    @property
    def additionalActions(self) -> Optional[DictionaryObject]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Use :py:attr:`additional_actions` instead.\n\n        .. deprecated:: 1.28.3\n        '
        deprecation_with_replacement('additionalActions', 'additional_actions', '3.0.0')
        return self.additional_actions

class Destination(TreeObject):
    """
    A class representing a destination within a PDF file.

    See section 8.2.1 of the PDF 1.6 reference.

    Args:
        title: Title of this destination.
        page: Reference to the page of this destination. Should
            be an instance of :class:`IndirectObject<pypdf.generic.IndirectObject>`.
        fit: How the destination is displayed.

    Raises:
        PdfReadError: If destination type is invalid.
    """
    node: Optional[DictionaryObject] = None
    childs: List[Any] = []

    def __init__(self, title: str, page: Union[NumberObject, IndirectObject, NullObject, DictionaryObject], fit: Fit) -> None:
        if False:
            for i in range(10):
                print('nop')
        typ = fit.fit_type
        args = fit.fit_args
        DictionaryObject.__init__(self)
        self[NameObject('/Title')] = TextStringObject(title)
        self[NameObject('/Page')] = page
        self[NameObject('/Type')] = typ
        if typ == '/XYZ':
            if len(args) < 1:
                args.append(NumberObject(0.0))
            if len(args) < 2:
                args.append(NumberObject(0.0))
            if len(args) < 3:
                args.append(NumberObject(0.0))
            (self[NameObject(TA.LEFT)], self[NameObject(TA.TOP)], self[NameObject('/Zoom')]) = args
        elif len(args) == 0:
            pass
        elif typ == TF.FIT_R:
            (self[NameObject(TA.LEFT)], self[NameObject(TA.BOTTOM)], self[NameObject(TA.RIGHT)], self[NameObject(TA.TOP)]) = args
        elif typ in [TF.FIT_H, TF.FIT_BH]:
            try:
                (self[NameObject(TA.TOP)],) = args
            except Exception:
                (self[NameObject(TA.TOP)],) = (NullObject(),)
        elif typ in [TF.FIT_V, TF.FIT_BV]:
            try:
                (self[NameObject(TA.LEFT)],) = args
            except Exception:
                (self[NameObject(TA.LEFT)],) = (NullObject(),)
        elif typ in [TF.FIT, TF.FIT_B]:
            pass
        else:
            raise PdfReadError(f'Unknown Destination Type: {typ!r}')

    @property
    def dest_array(self) -> 'ArrayObject':
        if False:
            i = 10
            return i + 15
        return ArrayObject([self.raw_get('/Page'), self['/Type']] + [self[x] for x in ['/Left', '/Bottom', '/Right', '/Top', '/Zoom'] if x in self])

    def getDestArray(self) -> 'ArrayObject':
        if False:
            for i in range(10):
                print('nop')
        '\n        Use :py:attr:`dest_array` instead.\n\n        .. deprecated:: 1.28.3\n        '
        deprecation_with_replacement('getDestArray', 'dest_array', '3.0.0')
        return self.dest_array

    def write_to_stream(self, stream: StreamType, encryption_key: Union[None, str, bytes]=None) -> None:
        if False:
            i = 10
            return i + 15
        if encryption_key is not None:
            deprecate_no_replacement('the encryption_key parameter of write_to_stream', '5.0.0')
        stream.write(b'<<\n')
        key = NameObject('/D')
        key.write_to_stream(stream)
        stream.write(b' ')
        value = self.dest_array
        value.write_to_stream(stream)
        key = NameObject('/S')
        key.write_to_stream(stream)
        stream.write(b' ')
        value_s = NameObject('/GoTo')
        value_s.write_to_stream(stream)
        stream.write(b'\n')
        stream.write(b'>>')

    @property
    def title(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        'Read-only property accessing the destination title.'
        return self.get('/Title')

    @property
    def page(self) -> Optional[int]:
        if False:
            return 10
        'Read-only property accessing the destination page number.'
        return self.get('/Page')

    @property
    def typ(self) -> Optional[str]:
        if False:
            print('Hello World!')
        'Read-only property accessing the destination type.'
        return self.get('/Type')

    @property
    def zoom(self) -> Optional[int]:
        if False:
            i = 10
            return i + 15
        'Read-only property accessing the zoom factor.'
        return self.get('/Zoom', None)

    @property
    def left(self) -> Optional[FloatObject]:
        if False:
            return 10
        'Read-only property accessing the left horizontal coordinate.'
        return self.get('/Left', None)

    @property
    def right(self) -> Optional[FloatObject]:
        if False:
            for i in range(10):
                print('nop')
        'Read-only property accessing the right horizontal coordinate.'
        return self.get('/Right', None)

    @property
    def top(self) -> Optional[FloatObject]:
        if False:
            for i in range(10):
                print('nop')
        'Read-only property accessing the top vertical coordinate.'
        return self.get('/Top', None)

    @property
    def bottom(self) -> Optional[FloatObject]:
        if False:
            while True:
                i = 10
        'Read-only property accessing the bottom vertical coordinate.'
        return self.get('/Bottom', None)

    @property
    def color(self) -> Optional['ArrayObject']:
        if False:
            print('Hello World!')
        'Read-only property accessing the color in (R, G, B) with values 0.0-1.0.'
        return self.get('/C', ArrayObject([FloatObject(0), FloatObject(0), FloatObject(0)]))

    @property
    def font_format(self) -> Optional[OutlineFontFlag]:
        if False:
            i = 10
            return i + 15
        '\n        Read-only property accessing the font type.\n\n        1=italic, 2=bold, 3=both\n        '
        return self.get('/F', 0)

    @property
    def outline_count(self) -> Optional[int]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Read-only property accessing the outline count.\n\n        positive = expanded\n        negative = collapsed\n        absolute value = number of visible descendants at all levels\n        '
        return self.get('/Count', None)