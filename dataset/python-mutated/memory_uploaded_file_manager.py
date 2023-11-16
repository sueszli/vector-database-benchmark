import uuid
from collections import defaultdict
from typing import Dict, List, Sequence
from streamlit import util
from streamlit.logger import get_logger
from streamlit.runtime.stats import CacheStat
from streamlit.runtime.uploaded_file_manager import UploadedFileManager, UploadedFileRec, UploadFileUrlInfo
LOGGER = get_logger(__name__)

class MemoryUploadedFileManager(UploadedFileManager):
    """Holds files uploaded by users of the running Streamlit app.
    This class can be used safely from multiple threads simultaneously.
    """

    def __init__(self, upload_endpoint: str):
        if False:
            i = 10
            return i + 15
        self.file_storage: Dict[str, Dict[str, UploadedFileRec]] = defaultdict(dict)
        self.endpoint = upload_endpoint

    def get_files(self, session_id: str, file_ids: Sequence[str]) -> List[UploadedFileRec]:
        if False:
            for i in range(10):
                print('nop')
        'Return a  list of UploadedFileRec for a given sequence of file_ids.\n\n        Parameters\n        ----------\n        session_id\n            The ID of the session that owns the files.\n        file_ids\n            The sequence of ids associated with files to retrieve.\n\n        Returns\n        -------\n        List[UploadedFileRec]\n            A list of URL UploadedFileRec instances, each instance contains information\n            about uploaded file.\n        '
        session_storage = self.file_storage[session_id]
        file_recs = []
        for file_id in file_ids:
            file_rec = session_storage.get(file_id, None)
            if file_rec is not None:
                file_recs.append(file_rec)
        return file_recs

    def remove_session_files(self, session_id: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Remove all files associated with a given session.'
        self.file_storage.pop(session_id, None)

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return util.repr_(self)

    def add_file(self, session_id: str, file: UploadedFileRec) -> None:
        if False:
            while True:
                i = 10
        '\n        Safe to call from any thread.\n\n        Parameters\n        ----------\n        session_id\n            The ID of the session that owns the file.\n        file\n            The file to add.\n        '
        self.file_storage[session_id][file.file_id] = file

    def remove_file(self, session_id, file_id):
        if False:
            i = 10
            return i + 15
        'Remove file with given file_id associated with a given session.'
        session_storage = self.file_storage[session_id]
        session_storage.pop(file_id, None)

    def get_upload_urls(self, session_id: str, file_names: Sequence[str]) -> List[UploadFileUrlInfo]:
        if False:
            for i in range(10):
                print('nop')
        'Return a list of UploadFileUrlInfo for a given sequence of file_names.'
        result = []
        for _ in file_names:
            file_id = str(uuid.uuid4())
            result.append(UploadFileUrlInfo(file_id=file_id, upload_url=f'{self.endpoint}/{session_id}/{file_id}', delete_url=f'{self.endpoint}/{session_id}/{file_id}'))
        return result

    def get_stats(self) -> List[CacheStat]:
        if False:
            i = 10
            return i + 15
        "Return the manager's CacheStats.\n\n        Safe to call from any thread.\n        "
        all_files: List[UploadedFileRec] = []
        file_storage_copy = self.file_storage.copy()
        for session_storage in file_storage_copy.values():
            all_files.extend(session_storage.values())
        return [CacheStat(category_name='UploadedFileManager', cache_name='', byte_length=len(file.data)) for file in all_files]