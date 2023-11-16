"""Provides global MediaFileManager object as `media_file_manager`."""
import collections
import threading
from typing import Dict, Optional, Set, Union
from streamlit.logger import get_logger
from streamlit.runtime.media_file_storage import MediaFileKind, MediaFileStorage
LOGGER = get_logger(__name__)

def _get_session_id() -> str:
    if False:
        while True:
            i = 10
    "Get the active AppSession's session_id."
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    ctx = get_script_run_ctx()
    if ctx is None:
        return 'dontcare'
    else:
        return ctx.session_id

class MediaFileMetadata:
    """Metadata that the MediaFileManager needs for each file it manages."""

    def __init__(self, kind: MediaFileKind=MediaFileKind.MEDIA):
        if False:
            i = 10
            return i + 15
        self._kind = kind
        self._is_marked_for_delete = False

    @property
    def kind(self) -> MediaFileKind:
        if False:
            print('Hello World!')
        return self._kind

    @property
    def is_marked_for_delete(self) -> bool:
        if False:
            print('Hello World!')
        return self._is_marked_for_delete

    def mark_for_delete(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._is_marked_for_delete = True

class MediaFileManager:
    """In-memory file manager for MediaFile objects.

    This keeps track of:
    - Which files exist, and what their IDs are. This is important so we can
      serve files by ID -- that's the whole point of this class!
    - Which files are being used by which AppSession (by ID). This is
      important so we can remove files from memory when no more sessions need
      them.
    - The exact location in the app where each file is being used (i.e. the
      file's "coordinates"). This is is important so we can mark a file as "not
      being used by a certain session" if it gets replaced by another file at
      the same coordinates. For example, when doing an animation where the same
      image is constantly replace with new frames. (This doesn't solve the case
      where the file's coordinates keep changing for some reason, though! e.g.
      if new elements keep being prepended to the app. Unlikely to happen, but
      we should address it at some point.)
    """

    def __init__(self, storage: MediaFileStorage):
        if False:
            while True:
                i = 10
        self._storage = storage
        self._file_metadata: Dict[str, MediaFileMetadata] = dict()
        self._files_by_session_and_coord: Dict[str, Dict[str, str]] = collections.defaultdict(dict)
        self._lock = threading.Lock()

    def _get_inactive_file_ids(self) -> Set[str]:
        if False:
            for i in range(10):
                print('nop')
        'Compute the set of files that are stored in the manager, but are\n        not referenced by any active session. These are files that can be\n        safely deleted.\n\n        Thread safety: callers must hold `self._lock`.\n        '
        file_ids = set(self._file_metadata.keys())
        for session_file_ids_by_coord in self._files_by_session_and_coord.values():
            file_ids.difference_update(session_file_ids_by_coord.values())
        return file_ids

    def remove_orphaned_files(self) -> None:
        if False:
            print('Hello World!')
        'Remove all files that are no longer referenced by any active session.\n\n        Safe to call from any thread.\n        '
        LOGGER.debug('Removing orphaned files...')
        with self._lock:
            for file_id in self._get_inactive_file_ids():
                file = self._file_metadata[file_id]
                if file.kind == MediaFileKind.MEDIA:
                    self._delete_file(file_id)
                elif file.kind == MediaFileKind.DOWNLOADABLE:
                    if file.is_marked_for_delete:
                        self._delete_file(file_id)
                    else:
                        file.mark_for_delete()

    def _delete_file(self, file_id: str) -> None:
        if False:
            return 10
        'Delete the given file from storage, and remove its metadata from\n        self._files_by_id.\n\n        Thread safety: callers must hold `self._lock`.\n        '
        LOGGER.debug('Deleting File: %s', file_id)
        self._storage.delete_file(file_id)
        del self._file_metadata[file_id]

    def clear_session_refs(self, session_id: Optional[str]=None) -> None:
        if False:
            i = 10
            return i + 15
        "Remove the given session's file references.\n\n        (This does not remove any files from the manager - you must call\n        `remove_orphaned_files` for that.)\n\n        Should be called whenever ScriptRunner completes and when a session ends.\n\n        Safe to call from any thread.\n        "
        if session_id is None:
            session_id = _get_session_id()
        LOGGER.debug('Disconnecting files for session with ID %s', session_id)
        with self._lock:
            if session_id in self._files_by_session_and_coord:
                del self._files_by_session_and_coord[session_id]
        LOGGER.debug('Sessions still active: %r', self._files_by_session_and_coord.keys())
        LOGGER.debug('Files: %s; Sessions with files: %s', len(self._file_metadata), len(self._files_by_session_and_coord))

    def add(self, path_or_data: Union[bytes, str], mimetype: str, coordinates: str, file_name: Optional[str]=None, is_for_static_download: bool=False) -> str:
        if False:
            print('Hello World!')
        'Add a new MediaFile with the given parameters and return its URL.\n\n        If an identical file already exists, return the existing URL\n        and registers the current session as a user.\n\n        Safe to call from any thread.\n\n        Parameters\n        ----------\n        path_or_data : bytes or str\n            If bytes: the media file\'s raw data. If str: the name of a file\n            to load from disk.\n        mimetype : str\n            The mime type for the file. E.g. "audio/mpeg".\n            This string will be used in the "Content-Type" header when the file\n            is served over HTTP.\n        coordinates : str\n            Unique string identifying an element\'s location.\n            Prevents memory leak of "forgotten" file IDs when element media\n            is being replaced-in-place (e.g. an st.image stream).\n            coordinates should be of the form: "1.(3.-14).5"\n        file_name : str or None\n            Optional file_name. Used to set the filename in the response header.\n        is_for_static_download: bool\n            Indicate that data stored for downloading as a file,\n            not as a media for rendering at page. [default: False]\n\n        Returns\n        -------\n        str\n            The url that the frontend can use to fetch the media.\n\n        Raises\n        ------\n        If a filename is passed, any Exception raised when trying to read the\n        file will be re-raised.\n        '
        session_id = _get_session_id()
        with self._lock:
            kind = MediaFileKind.DOWNLOADABLE if is_for_static_download else MediaFileKind.MEDIA
            file_id = self._storage.load_and_get_id(path_or_data, mimetype, kind, file_name)
            metadata = MediaFileMetadata(kind=kind)
            self._file_metadata[file_id] = metadata
            self._files_by_session_and_coord[session_id][coordinates] = file_id
            return self._storage.get_url(file_id)