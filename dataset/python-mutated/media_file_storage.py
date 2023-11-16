from abc import abstractmethod
from enum import Enum
from typing import Optional, Union
from typing_extensions import Protocol

class MediaFileKind(Enum):
    MEDIA = 'media'
    DOWNLOADABLE = 'downloadable'

class MediaFileStorageError(Exception):
    """Exception class for errors raised by MediaFileStorage.

    When running in "development mode", the full text of these errors
    is displayed in the frontend, so errors should be human-readable
    (and actionable).

    When running in "release mode", errors are redacted on the
    frontend; we instead show a generic "Something went wrong!" message.
    """

class MediaFileStorage(Protocol):

    @abstractmethod
    def load_and_get_id(self, path_or_data: Union[str, bytes], mimetype: str, kind: MediaFileKind, filename: Optional[str]=None) -> str:
        if False:
            while True:
                i = 10
        "Load the given file path or bytes into the manager and return\n        an ID that uniquely identifies it.\n\n        It’s an error to pass a URL to this function. (Media stored at\n        external URLs can be served directly to the Streamlit frontend;\n        there’s no need to store this data in MediaFileStorage.)\n\n        Parameters\n        ----------\n        path_or_data\n            A path to a file, or the file's raw data as bytes.\n\n        mimetype\n            The media’s mimetype. Used to set the Content-Type header when\n            serving the media over HTTP.\n\n        kind\n            The kind of file this is: either MEDIA, or DOWNLOADABLE.\n\n        filename : str or None\n            Optional filename. Used to set the filename in the response header.\n\n        Returns\n        -------\n        str\n            The unique ID of the media file.\n\n        Raises\n        ------\n        MediaFileStorageError\n            Raised if the media can't be loaded (for example, if a file\n            path is invalid).\n\n        "
        raise NotImplementedError

    @abstractmethod
    def get_url(self, file_id: str) -> str:
        if False:
            while True:
                i = 10
        "Return a URL for a file in the manager.\n\n        Parameters\n        ----------\n        file_id\n            The file's ID, returned from load_media_and_get_id().\n\n        Returns\n        -------\n        str\n            A URL that the frontend can load the file from. Because this\n            URL may expire, it should not be cached!\n\n        Raises\n        ------\n        MediaFileStorageError\n            Raised if the manager doesn't contain an object with the given ID.\n\n        "
        raise NotImplementedError

    @abstractmethod
    def delete_file(self, file_id: str) -> None:
        if False:
            return 10
        "Delete a file from the manager.\n\n        This should be called when a given file is no longer referenced\n        by any connected client, so that the MediaFileStorage can free its\n        resources.\n\n        Calling `delete_file` on a file_id that doesn't exist is allowed,\n        and is a no-op. (This means that multiple `delete_file` calls with\n        the same file_id is not an error.)\n\n        Note: implementations can choose to ignore `delete_file` calls -\n        this function is a *suggestion*, not a *command*. Callers should\n        not rely on file deletion happening immediately (or at all).\n\n        Parameters\n        ----------\n        file_id\n            The file's ID, returned from load_media_and_get_id().\n\n        Returns\n        -------\n        None\n\n        Raises\n        ------\n        MediaFileStorageError\n            Raised if file deletion fails for any reason. Note that these\n            failures will generally not be shown on the frontend (file\n            deletion usually occurs on session disconnect).\n\n        "
        raise NotImplementedError