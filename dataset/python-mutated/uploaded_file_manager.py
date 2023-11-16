import io
from abc import abstractmethod
from typing import List, NamedTuple, Sequence
from typing_extensions import Protocol
from streamlit import util
from streamlit.proto.Common_pb2 import FileURLs as FileURLsProto
from streamlit.runtime.stats import CacheStatsProvider

class UploadedFileRec(NamedTuple):
    """Metadata and raw bytes for an uploaded file. Immutable."""
    file_id: str
    name: str
    type: str
    data: bytes

class UploadFileUrlInfo(NamedTuple):
    """Information we provide for single file in get_upload_urls"""
    file_id: str
    upload_url: str
    delete_url: str

class DeletedFile(NamedTuple):
    """Represents a deleted file in deserialized values for st.file_uploader and
    st.camera_input

    Return this from st.file_uploader and st.camera_input deserialize (so they can
    be used in session_state), when widget value contains file record that is missing
    from the storage.
    DeleteFile instances filtered out before return final value to the user in script,
    or before sending to frontend."""
    file_id: str

class UploadedFile(io.BytesIO):
    """A mutable uploaded file.

    This class extends BytesIO, which has copy-on-write semantics when
    initialized with `bytes`.
    """

    def __init__(self, record: UploadedFileRec, file_urls: FileURLsProto):
        if False:
            return 10
        super().__init__(record.data)
        self.file_id = record.file_id
        self.name = record.name
        self.type = record.type
        self.size = len(record.data)
        self._file_urls = file_urls

    def __eq__(self, other: object) -> bool:
        if False:
            return 10
        if not isinstance(other, UploadedFile):
            return NotImplemented
        return self.file_id == other.file_id

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return util.repr_(self)

class UploadedFileManager(CacheStatsProvider, Protocol):
    """UploadedFileManager protocol, that should be implemented by the concrete
    uploaded file managers.

    It is responsible for:
        - retrieving files by session_id and file_id for st.file_uploader and
            st.camera_input
        - cleaning up uploaded files associated with session on session end

    It should be created during Runtime initialization.

    Optionally UploadedFileManager could be responsible for issuing URLs which will be
    used by frontend to upload files to.
    """

    @abstractmethod
    def get_files(self, session_id: str, file_ids: Sequence[str]) -> List[UploadedFileRec]:
        if False:
            i = 10
            return i + 15
        'Return a  list of UploadedFileRec for a given sequence of file_ids.\n\n        Parameters\n        ----------\n        session_id\n            The ID of the session that owns the files.\n        file_ids\n            The sequence of ids associated with files to retrieve.\n\n        Returns\n        -------\n        List[UploadedFileRec]\n            A list of URL UploadedFileRec instances, each instance contains information\n            about uploaded file.\n        '
        raise NotImplementedError

    @abstractmethod
    def remove_session_files(self, session_id: str) -> None:
        if False:
            while True:
                i = 10
        'Remove all files associated with a given session.'
        raise NotImplementedError

    def get_upload_urls(self, session_id: str, file_names: Sequence[str]) -> List[UploadFileUrlInfo]:
        if False:
            i = 10
            return i + 15
        'Return a list of UploadFileUrlInfo for a given sequence of file_names.\n        Optional to implement, issuing of URLs could be done by other service.\n\n        Parameters\n        ----------\n        session_id\n            The ID of the session that request URLs.\n        file_names\n            The sequence of file names for which URLs are requested\n\n        Returns\n        -------\n        List[UploadFileUrlInfo]\n            A list of UploadFileUrlInfo instances, each instance contains information\n            about uploaded file URLs.\n        '
        raise NotImplementedError