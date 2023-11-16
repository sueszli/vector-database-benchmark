import functools
import os
import re
import string
from typing import Any, Callable, List, TypeVar, Union, cast
NEW_FORMAT_ID_RE = re.compile('^\\d\\d\\d\\d-\\d\\d-\\d\\d')
F = TypeVar('F', bound=Callable[..., str])

def _wrap_in_base_path(func: F) -> F:
    if False:
        return 10
    'Takes a function that returns a relative path and turns it into an\n    absolute path based on the location of the primary media store\n    '

    @functools.wraps(func)
    def _wrapped(self: 'MediaFilePaths', *args: Any, **kwargs: Any) -> str:
        if False:
            while True:
                i = 10
        path = func(self, *args, **kwargs)
        return os.path.join(self.base_path, path)
    return cast(F, _wrapped)
GetPathMethod = TypeVar('GetPathMethod', bound=Union[Callable[..., str], Callable[..., List[str]]])

def _wrap_with_jail_check(relative: bool) -> Callable[[GetPathMethod], GetPathMethod]:
    if False:
        for i in range(10):
            print('nop')
    'Wraps a path-returning method to check that the returned path(s) do not escape\n    the media store directory.\n\n    The path-returning method may return either a single path, or a list of paths.\n\n    The check is not expected to ever fail, unless `func` is missing a call to\n    `_validate_path_component`, or `_validate_path_component` is buggy.\n\n    Args:\n        relative: A boolean indicating whether the wrapped method returns paths relative\n            to the media store directory.\n\n    Returns:\n        A method which will wrap a path-returning method, adding a check to ensure that\n        the returned path(s) lie within the media store directory. The check will raise\n        a `ValueError` if it fails.\n    '

    def _wrap_with_jail_check_inner(func: GetPathMethod) -> GetPathMethod:
        if False:
            print('Hello World!')

        @functools.wraps(func)
        def _wrapped(self: 'MediaFilePaths', *args: Any, **kwargs: Any) -> Union[str, List[str]]:
            if False:
                i = 10
                return i + 15
            path_or_paths = func(self, *args, **kwargs)
            if isinstance(path_or_paths, list):
                paths_to_check = path_or_paths
            else:
                paths_to_check = [path_or_paths]
            for path in paths_to_check:
                if relative:
                    path = os.path.join(self.base_path, path)
                normalized_path = os.path.normpath(path)
                if os.path.commonpath([normalized_path, self.normalized_base_path]) != self.normalized_base_path:
                    raise ValueError(f'Invalid media store path: {path!r}')
            return path_or_paths
        return cast(GetPathMethod, _wrapped)
    return _wrap_with_jail_check_inner
ALLOWED_CHARACTERS = set(string.ascii_letters + string.digits + '_-' + '.[]:')
FORBIDDEN_NAMES = {'', os.path.curdir, os.path.pardir}

def _validate_path_component(name: str) -> str:
    if False:
        while True:
            i = 10
    'Checks that the given string can be safely used as a path component\n\n    Args:\n        name: The path component to check.\n\n    Returns:\n        The path component if valid.\n\n    Raises:\n        ValueError: If `name` cannot be safely used as a path component.\n    '
    if not ALLOWED_CHARACTERS.issuperset(name) or name in FORBIDDEN_NAMES:
        raise ValueError(f'Invalid path component: {name!r}')
    return name

class MediaFilePaths:
    """Describes where files are stored on disk.

    Most of the functions have a `*_rel` variant which returns a file path that
    is relative to the base media store path. This is mainly used when we want
    to write to the backup media store (when one is configured)
    """

    def __init__(self, primary_base_path: str):
        if False:
            return 10
        self.base_path = primary_base_path
        self.normalized_base_path = os.path.normpath(self.base_path)
        assert os.path.sep not in ALLOWED_CHARACTERS
        assert os.path.altsep not in ALLOWED_CHARACTERS
        assert os.name == 'posix'

    @_wrap_with_jail_check(relative=True)
    def local_media_filepath_rel(self, media_id: str) -> str:
        if False:
            print('Hello World!')
        return os.path.join('local_content', _validate_path_component(media_id[0:2]), _validate_path_component(media_id[2:4]), _validate_path_component(media_id[4:]))
    local_media_filepath = _wrap_in_base_path(local_media_filepath_rel)

    @_wrap_with_jail_check(relative=True)
    def local_media_thumbnail_rel(self, media_id: str, width: int, height: int, content_type: str, method: str) -> str:
        if False:
            while True:
                i = 10
        (top_level_type, sub_type) = content_type.split('/')
        file_name = '%i-%i-%s-%s-%s' % (width, height, top_level_type, sub_type, method)
        return os.path.join('local_thumbnails', _validate_path_component(media_id[0:2]), _validate_path_component(media_id[2:4]), _validate_path_component(media_id[4:]), _validate_path_component(file_name))
    local_media_thumbnail = _wrap_in_base_path(local_media_thumbnail_rel)

    @_wrap_with_jail_check(relative=False)
    def local_media_thumbnail_dir(self, media_id: str) -> str:
        if False:
            return 10
        '\n        Retrieve the local store path of thumbnails of a given media_id\n\n        Args:\n            media_id: The media ID to query.\n        Returns:\n            Path of local_thumbnails from media_id\n        '
        return os.path.join(self.base_path, 'local_thumbnails', _validate_path_component(media_id[0:2]), _validate_path_component(media_id[2:4]), _validate_path_component(media_id[4:]))

    @_wrap_with_jail_check(relative=True)
    def remote_media_filepath_rel(self, server_name: str, file_id: str) -> str:
        if False:
            print('Hello World!')
        return os.path.join('remote_content', _validate_path_component(server_name), _validate_path_component(file_id[0:2]), _validate_path_component(file_id[2:4]), _validate_path_component(file_id[4:]))
    remote_media_filepath = _wrap_in_base_path(remote_media_filepath_rel)

    @_wrap_with_jail_check(relative=True)
    def remote_media_thumbnail_rel(self, server_name: str, file_id: str, width: int, height: int, content_type: str, method: str) -> str:
        if False:
            while True:
                i = 10
        (top_level_type, sub_type) = content_type.split('/')
        file_name = '%i-%i-%s-%s-%s' % (width, height, top_level_type, sub_type, method)
        return os.path.join('remote_thumbnail', _validate_path_component(server_name), _validate_path_component(file_id[0:2]), _validate_path_component(file_id[2:4]), _validate_path_component(file_id[4:]), _validate_path_component(file_name))
    remote_media_thumbnail = _wrap_in_base_path(remote_media_thumbnail_rel)

    @_wrap_with_jail_check(relative=True)
    def remote_media_thumbnail_rel_legacy(self, server_name: str, file_id: str, width: int, height: int, content_type: str) -> str:
        if False:
            i = 10
            return i + 15
        (top_level_type, sub_type) = content_type.split('/')
        file_name = '%i-%i-%s-%s' % (width, height, top_level_type, sub_type)
        return os.path.join('remote_thumbnail', _validate_path_component(server_name), _validate_path_component(file_id[0:2]), _validate_path_component(file_id[2:4]), _validate_path_component(file_id[4:]), _validate_path_component(file_name))

    @_wrap_with_jail_check(relative=False)
    def remote_media_thumbnail_dir(self, server_name: str, file_id: str) -> str:
        if False:
            print('Hello World!')
        return os.path.join(self.base_path, 'remote_thumbnail', _validate_path_component(server_name), _validate_path_component(file_id[0:2]), _validate_path_component(file_id[2:4]), _validate_path_component(file_id[4:]))

    @_wrap_with_jail_check(relative=True)
    def url_cache_filepath_rel(self, media_id: str) -> str:
        if False:
            while True:
                i = 10
        if NEW_FORMAT_ID_RE.match(media_id):
            return os.path.join('url_cache', _validate_path_component(media_id[:10]), _validate_path_component(media_id[11:]))
        else:
            return os.path.join('url_cache', _validate_path_component(media_id[0:2]), _validate_path_component(media_id[2:4]), _validate_path_component(media_id[4:]))
    url_cache_filepath = _wrap_in_base_path(url_cache_filepath_rel)

    @_wrap_with_jail_check(relative=False)
    def url_cache_filepath_dirs_to_delete(self, media_id: str) -> List[str]:
        if False:
            return 10
        'The dirs to try and remove if we delete the media_id file'
        if NEW_FORMAT_ID_RE.match(media_id):
            return [os.path.join(self.base_path, 'url_cache', _validate_path_component(media_id[:10]))]
        else:
            return [os.path.join(self.base_path, 'url_cache', _validate_path_component(media_id[0:2]), _validate_path_component(media_id[2:4])), os.path.join(self.base_path, 'url_cache', _validate_path_component(media_id[0:2]))]

    @_wrap_with_jail_check(relative=True)
    def url_cache_thumbnail_rel(self, media_id: str, width: int, height: int, content_type: str, method: str) -> str:
        if False:
            print('Hello World!')
        (top_level_type, sub_type) = content_type.split('/')
        file_name = '%i-%i-%s-%s-%s' % (width, height, top_level_type, sub_type, method)
        if NEW_FORMAT_ID_RE.match(media_id):
            return os.path.join('url_cache_thumbnails', _validate_path_component(media_id[:10]), _validate_path_component(media_id[11:]), _validate_path_component(file_name))
        else:
            return os.path.join('url_cache_thumbnails', _validate_path_component(media_id[0:2]), _validate_path_component(media_id[2:4]), _validate_path_component(media_id[4:]), _validate_path_component(file_name))
    url_cache_thumbnail = _wrap_in_base_path(url_cache_thumbnail_rel)

    @_wrap_with_jail_check(relative=True)
    def url_cache_thumbnail_directory_rel(self, media_id: str) -> str:
        if False:
            print('Hello World!')
        if NEW_FORMAT_ID_RE.match(media_id):
            return os.path.join('url_cache_thumbnails', _validate_path_component(media_id[:10]), _validate_path_component(media_id[11:]))
        else:
            return os.path.join('url_cache_thumbnails', _validate_path_component(media_id[0:2]), _validate_path_component(media_id[2:4]), _validate_path_component(media_id[4:]))
    url_cache_thumbnail_directory = _wrap_in_base_path(url_cache_thumbnail_directory_rel)

    @_wrap_with_jail_check(relative=False)
    def url_cache_thumbnail_dirs_to_delete(self, media_id: str) -> List[str]:
        if False:
            print('Hello World!')
        'The dirs to try and remove if we delete the media_id thumbnails'
        if NEW_FORMAT_ID_RE.match(media_id):
            return [os.path.join(self.base_path, 'url_cache_thumbnails', _validate_path_component(media_id[:10]), _validate_path_component(media_id[11:])), os.path.join(self.base_path, 'url_cache_thumbnails', _validate_path_component(media_id[:10]))]
        else:
            return [os.path.join(self.base_path, 'url_cache_thumbnails', _validate_path_component(media_id[0:2]), _validate_path_component(media_id[2:4]), _validate_path_component(media_id[4:])), os.path.join(self.base_path, 'url_cache_thumbnails', _validate_path_component(media_id[0:2]), _validate_path_component(media_id[2:4])), os.path.join(self.base_path, 'url_cache_thumbnails', _validate_path_component(media_id[0:2]))]