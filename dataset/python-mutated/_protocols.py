"""Helpers for working with PDF types."""
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Tuple, Union
try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol
from ._utils import StrByteType, StreamType

class PdfObjectProtocol(Protocol):
    indirect_reference: Any

    def clone(self, pdf_dest: Any, force_duplicate: bool=False, ignore_fields: Union[Tuple[str, ...], List[str], None]=()) -> Any:
        if False:
            i = 10
            return i + 15
        ...

    def _reference_clone(self, clone: Any, pdf_dest: Any) -> Any:
        if False:
            print('Hello World!')
        ...

    def get_object(self) -> Optional['PdfObjectProtocol']:
        if False:
            i = 10
            return i + 15
        ...

    def hash_value(self) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        ...

    def write_to_stream(self, stream: StreamType, encryption_key: Union[None, str, bytes]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        ...

class PdfReaderProtocol(Protocol):

    @property
    def pdf_header(self) -> str:
        if False:
            while True:
                i = 10
        ...

    @property
    def strict(self) -> bool:
        if False:
            while True:
                i = 10
        ...

    @property
    def xref(self) -> Dict[int, Dict[int, Any]]:
        if False:
            while True:
                i = 10
        ...

    @property
    def pages(self) -> List[Any]:
        if False:
            return 10
        ...

    @property
    def trailer(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        ...

    def get_object(self, indirect_reference: Any) -> Optional[PdfObjectProtocol]:
        if False:
            for i in range(10):
                print('nop')
        ...

class PdfWriterProtocol(Protocol):
    _objects: List[Any]
    _id_translated: Dict[int, Dict[int, int]]

    def get_object(self, indirect_reference: Any) -> Optional[PdfObjectProtocol]:
        if False:
            while True:
                i = 10
        ...

    def write(self, stream: Union[Path, StrByteType]) -> Tuple[bool, IO[Any]]:
        if False:
            for i in range(10):
                print('nop')
        ...

    def _add_object(self, obj: Any) -> Any:
        if False:
            return 10
        ...

    @property
    def pages(self) -> List[Any]:
        if False:
            i = 10
            return i + 15
        ...

    @property
    def pdf_header(self) -> bytes:
        if False:
            print('Hello World!')
        ...