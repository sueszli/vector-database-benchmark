import inspect
import os
from typing import Iterable
from synapse.media.filepath import MediaFilePaths, _wrap_with_jail_check
from tests import unittest

class MediaFilePathsTestCase(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.filepaths = MediaFilePaths('/media_store')

    def test_local_media_filepath(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test local media paths'
        self.assertEqual(self.filepaths.local_media_filepath_rel('GerZNDnDZVjsOtardLuwfIBg'), 'local_content/Ge/rZ/NDnDZVjsOtardLuwfIBg')
        self.assertEqual(self.filepaths.local_media_filepath('GerZNDnDZVjsOtardLuwfIBg'), '/media_store/local_content/Ge/rZ/NDnDZVjsOtardLuwfIBg')

    def test_local_media_thumbnail(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test local media thumbnail paths'
        self.assertEqual(self.filepaths.local_media_thumbnail_rel('GerZNDnDZVjsOtardLuwfIBg', 800, 600, 'image/jpeg', 'scale'), 'local_thumbnails/Ge/rZ/NDnDZVjsOtardLuwfIBg/800-600-image-jpeg-scale')
        self.assertEqual(self.filepaths.local_media_thumbnail('GerZNDnDZVjsOtardLuwfIBg', 800, 600, 'image/jpeg', 'scale'), '/media_store/local_thumbnails/Ge/rZ/NDnDZVjsOtardLuwfIBg/800-600-image-jpeg-scale')

    def test_local_media_thumbnail_dir(self) -> None:
        if False:
            while True:
                i = 10
        'Test local media thumbnail directory paths'
        self.assertEqual(self.filepaths.local_media_thumbnail_dir('GerZNDnDZVjsOtardLuwfIBg'), '/media_store/local_thumbnails/Ge/rZ/NDnDZVjsOtardLuwfIBg')

    def test_remote_media_filepath(self) -> None:
        if False:
            print('Hello World!')
        'Test remote media paths'
        self.assertEqual(self.filepaths.remote_media_filepath_rel('example.com', 'GerZNDnDZVjsOtardLuwfIBg'), 'remote_content/example.com/Ge/rZ/NDnDZVjsOtardLuwfIBg')
        self.assertEqual(self.filepaths.remote_media_filepath('example.com', 'GerZNDnDZVjsOtardLuwfIBg'), '/media_store/remote_content/example.com/Ge/rZ/NDnDZVjsOtardLuwfIBg')

    def test_remote_media_thumbnail(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test remote media thumbnail paths'
        self.assertEqual(self.filepaths.remote_media_thumbnail_rel('example.com', 'GerZNDnDZVjsOtardLuwfIBg', 800, 600, 'image/jpeg', 'scale'), 'remote_thumbnail/example.com/Ge/rZ/NDnDZVjsOtardLuwfIBg/800-600-image-jpeg-scale')
        self.assertEqual(self.filepaths.remote_media_thumbnail('example.com', 'GerZNDnDZVjsOtardLuwfIBg', 800, 600, 'image/jpeg', 'scale'), '/media_store/remote_thumbnail/example.com/Ge/rZ/NDnDZVjsOtardLuwfIBg/800-600-image-jpeg-scale')

    def test_remote_media_thumbnail_legacy(self) -> None:
        if False:
            while True:
                i = 10
        'Test old-style remote media thumbnail paths'
        self.assertEqual(self.filepaths.remote_media_thumbnail_rel_legacy('example.com', 'GerZNDnDZVjsOtardLuwfIBg', 800, 600, 'image/jpeg'), 'remote_thumbnail/example.com/Ge/rZ/NDnDZVjsOtardLuwfIBg/800-600-image-jpeg')

    def test_remote_media_thumbnail_dir(self) -> None:
        if False:
            while True:
                i = 10
        'Test remote media thumbnail directory paths'
        self.assertEqual(self.filepaths.remote_media_thumbnail_dir('example.com', 'GerZNDnDZVjsOtardLuwfIBg'), '/media_store/remote_thumbnail/example.com/Ge/rZ/NDnDZVjsOtardLuwfIBg')

    def test_url_cache_filepath(self) -> None:
        if False:
            print('Hello World!')
        'Test URL cache paths'
        self.assertEqual(self.filepaths.url_cache_filepath_rel('2020-01-02_GerZNDnDZVjsOtar'), 'url_cache/2020-01-02/GerZNDnDZVjsOtar')
        self.assertEqual(self.filepaths.url_cache_filepath('2020-01-02_GerZNDnDZVjsOtar'), '/media_store/url_cache/2020-01-02/GerZNDnDZVjsOtar')

    def test_url_cache_filepath_legacy(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test old-style URL cache paths'
        self.assertEqual(self.filepaths.url_cache_filepath_rel('GerZNDnDZVjsOtardLuwfIBg'), 'url_cache/Ge/rZ/NDnDZVjsOtardLuwfIBg')
        self.assertEqual(self.filepaths.url_cache_filepath('GerZNDnDZVjsOtardLuwfIBg'), '/media_store/url_cache/Ge/rZ/NDnDZVjsOtardLuwfIBg')

    def test_url_cache_filepath_dirs_to_delete(self) -> None:
        if False:
            return 10
        'Test URL cache cleanup paths'
        self.assertEqual(self.filepaths.url_cache_filepath_dirs_to_delete('2020-01-02_GerZNDnDZVjsOtar'), ['/media_store/url_cache/2020-01-02'])

    def test_url_cache_filepath_dirs_to_delete_legacy(self) -> None:
        if False:
            print('Hello World!')
        'Test old-style URL cache cleanup paths'
        self.assertEqual(self.filepaths.url_cache_filepath_dirs_to_delete('GerZNDnDZVjsOtardLuwfIBg'), ['/media_store/url_cache/Ge/rZ', '/media_store/url_cache/Ge'])

    def test_url_cache_thumbnail(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test URL cache thumbnail paths'
        self.assertEqual(self.filepaths.url_cache_thumbnail_rel('2020-01-02_GerZNDnDZVjsOtar', 800, 600, 'image/jpeg', 'scale'), 'url_cache_thumbnails/2020-01-02/GerZNDnDZVjsOtar/800-600-image-jpeg-scale')
        self.assertEqual(self.filepaths.url_cache_thumbnail('2020-01-02_GerZNDnDZVjsOtar', 800, 600, 'image/jpeg', 'scale'), '/media_store/url_cache_thumbnails/2020-01-02/GerZNDnDZVjsOtar/800-600-image-jpeg-scale')

    def test_url_cache_thumbnail_legacy(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test old-style URL cache thumbnail paths'
        self.assertEqual(self.filepaths.url_cache_thumbnail_rel('GerZNDnDZVjsOtardLuwfIBg', 800, 600, 'image/jpeg', 'scale'), 'url_cache_thumbnails/Ge/rZ/NDnDZVjsOtardLuwfIBg/800-600-image-jpeg-scale')
        self.assertEqual(self.filepaths.url_cache_thumbnail('GerZNDnDZVjsOtardLuwfIBg', 800, 600, 'image/jpeg', 'scale'), '/media_store/url_cache_thumbnails/Ge/rZ/NDnDZVjsOtardLuwfIBg/800-600-image-jpeg-scale')

    def test_url_cache_thumbnail_directory(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test URL cache thumbnail directory paths'
        self.assertEqual(self.filepaths.url_cache_thumbnail_directory_rel('2020-01-02_GerZNDnDZVjsOtar'), 'url_cache_thumbnails/2020-01-02/GerZNDnDZVjsOtar')
        self.assertEqual(self.filepaths.url_cache_thumbnail_directory('2020-01-02_GerZNDnDZVjsOtar'), '/media_store/url_cache_thumbnails/2020-01-02/GerZNDnDZVjsOtar')

    def test_url_cache_thumbnail_directory_legacy(self) -> None:
        if False:
            return 10
        'Test old-style URL cache thumbnail directory paths'
        self.assertEqual(self.filepaths.url_cache_thumbnail_directory_rel('GerZNDnDZVjsOtardLuwfIBg'), 'url_cache_thumbnails/Ge/rZ/NDnDZVjsOtardLuwfIBg')
        self.assertEqual(self.filepaths.url_cache_thumbnail_directory('GerZNDnDZVjsOtardLuwfIBg'), '/media_store/url_cache_thumbnails/Ge/rZ/NDnDZVjsOtardLuwfIBg')

    def test_url_cache_thumbnail_dirs_to_delete(self) -> None:
        if False:
            while True:
                i = 10
        'Test URL cache thumbnail cleanup paths'
        self.assertEqual(self.filepaths.url_cache_thumbnail_dirs_to_delete('2020-01-02_GerZNDnDZVjsOtar'), ['/media_store/url_cache_thumbnails/2020-01-02/GerZNDnDZVjsOtar', '/media_store/url_cache_thumbnails/2020-01-02'])

    def test_url_cache_thumbnail_dirs_to_delete_legacy(self) -> None:
        if False:
            return 10
        'Test old-style URL cache thumbnail cleanup paths'
        self.assertEqual(self.filepaths.url_cache_thumbnail_dirs_to_delete('GerZNDnDZVjsOtardLuwfIBg'), ['/media_store/url_cache_thumbnails/Ge/rZ/NDnDZVjsOtardLuwfIBg', '/media_store/url_cache_thumbnails/Ge/rZ', '/media_store/url_cache_thumbnails/Ge'])

    def test_server_name_validation(self) -> None:
        if False:
            print('Hello World!')
        'Test validation of server names'
        self._test_path_validation(['remote_media_filepath_rel', 'remote_media_filepath', 'remote_media_thumbnail_rel', 'remote_media_thumbnail', 'remote_media_thumbnail_rel_legacy', 'remote_media_thumbnail_dir'], parameter='server_name', valid_values=['matrix.org', 'matrix.org:8448', 'matrix-federation.matrix.org', 'matrix-federation.matrix.org:8448', '10.1.12.123', '10.1.12.123:8448', '[fd00:abcd::ffff]', '[fd00:abcd::ffff]:8448'], invalid_values=['/matrix.org', 'matrix.org/..', 'matrix.org\x00', '', '.', '..', '/'])

    def test_file_id_validation(self) -> None:
        if False:
            return 10
        'Test validation of local, remote and legacy URL cache file / media IDs'
        valid_file_ids = ['GerZNDnDZVjsOtardLuwfIBg', 'GerZN']
        invalid_file_ids = ['/erZNDnDZVjsOtardLuwfIBg', 'Ge/ZNDnDZVjsOtardLuwfIBg', 'GerZ/DnDZVjsOtardLuwfIBg', 'GerZ/..', 'G\x00rZNDnDZVjsOtardLuwfIBg', 'Ger\x00NDnDZVjsOtardLuwfIBg', 'GerZNDnDZVjsOtardLuwfIBg\x00', '', 'Ge', 'GerZ', 'GerZ.', '..rZNDnDZVjsOtardLuwfIBg', 'Ge..NDnDZVjsOtardLuwfIBg', 'GerZ..', 'GerZ/']
        self._test_path_validation(['local_media_filepath_rel', 'local_media_filepath', 'local_media_thumbnail_rel', 'local_media_thumbnail', 'local_media_thumbnail_dir', 'url_cache_filepath_rel', 'url_cache_filepath', 'url_cache_thumbnail_rel', 'url_cache_thumbnail', 'url_cache_thumbnail_directory_rel', 'url_cache_thumbnail_directory', 'url_cache_thumbnail_dirs_to_delete'], parameter='media_id', valid_values=valid_file_ids, invalid_values=invalid_file_ids)
        self._test_path_validation(['url_cache_filepath_dirs_to_delete'], parameter='media_id', valid_values=valid_file_ids, invalid_values=['/erZNDnDZVjsOtardLuwfIBg', 'Ge/ZNDnDZVjsOtardLuwfIBg', 'G\x00rZNDnDZVjsOtardLuwfIBg', 'Ger\x00NDnDZVjsOtardLuwfIBg', '', 'Ge', '..rZNDnDZVjsOtardLuwfIBg', 'Ge..NDnDZVjsOtardLuwfIBg'])
        self._test_path_validation(['remote_media_filepath_rel', 'remote_media_filepath', 'remote_media_thumbnail_rel', 'remote_media_thumbnail', 'remote_media_thumbnail_rel_legacy', 'remote_media_thumbnail_dir'], parameter='file_id', valid_values=valid_file_ids, invalid_values=invalid_file_ids)

    def test_url_cache_media_id_validation(self) -> None:
        if False:
            while True:
                i = 10
        'Test validation of URL cache media IDs'
        self._test_path_validation(['url_cache_filepath_rel', 'url_cache_filepath', 'url_cache_thumbnail_rel', 'url_cache_thumbnail', 'url_cache_thumbnail_directory_rel', 'url_cache_thumbnail_directory', 'url_cache_thumbnail_dirs_to_delete'], parameter='media_id', valid_values=['2020-01-02_GerZNDnDZVjsOtar', '2020-01-02_G'], invalid_values=['2020-01-02', '2020-01-02-', '2020-01-02-.', '2020-01-02-..', '2020-01-02-/', '2020-01-02-/GerZNDnDZVjsOtar', '2020-01-02-GerZNDnDZVjsOtar/..', '2020-01-02-GerZNDnDZVjsOtar\x00'])

    def test_content_type_validation(self) -> None:
        if False:
            return 10
        'Test validation of thumbnail content types'
        self._test_path_validation(['local_media_thumbnail_rel', 'local_media_thumbnail', 'remote_media_thumbnail_rel', 'remote_media_thumbnail', 'remote_media_thumbnail_rel_legacy', 'url_cache_thumbnail_rel', 'url_cache_thumbnail'], parameter='content_type', valid_values=['image/jpeg'], invalid_values=['', 'image/jpeg/abc', 'image/jpeg\x00'])

    def test_thumbnail_method_validation(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test validation of thumbnail methods'
        self._test_path_validation(['local_media_thumbnail_rel', 'local_media_thumbnail', 'remote_media_thumbnail_rel', 'remote_media_thumbnail', 'url_cache_thumbnail_rel', 'url_cache_thumbnail'], parameter='method', valid_values=['crop', 'scale'], invalid_values=['/scale', 'scale/..', 'scale\x00', '/'])

    def _test_path_validation(self, methods: Iterable[str], parameter: str, valid_values: Iterable[str], invalid_values: Iterable[str]) -> None:
        if False:
            return 10
        'Test that the specified methods validate the named parameter as expected\n\n        Args:\n            methods: The names of `MediaFilePaths` methods to test\n            parameter: The name of the parameter to test\n            valid_values: A list of parameter values that are expected to be accepted\n            invalid_values: A list of parameter values that are expected to be rejected\n\n        Raises:\n            AssertionError: If a value was accepted when it should have failed\n                validation.\n            ValueError: If a value failed validation when it should have been accepted.\n        '
        for method in methods:
            get_path = getattr(self.filepaths, method)
            parameters = inspect.signature(get_path).parameters
            kwargs = {'server_name': 'matrix.org', 'media_id': 'GerZNDnDZVjsOtardLuwfIBg', 'file_id': 'GerZNDnDZVjsOtardLuwfIBg', 'width': 800, 'height': 600, 'content_type': 'image/jpeg', 'method': 'scale'}
            if get_path.__name__.startswith('url_'):
                kwargs['media_id'] = '2020-01-02_GerZNDnDZVjsOtar'
            kwargs = {k: v for (k, v) in kwargs.items() if k in parameters}
            kwargs.pop(parameter)
            for value in valid_values:
                kwargs[parameter] = value
                get_path(**kwargs)
            for value in invalid_values:
                with self.assertRaises(ValueError):
                    kwargs[parameter] = value
                    path_or_list = get_path(**kwargs)
                    self.fail(f'{value!r} unexpectedly passed validation: {method} returned {path_or_list!r}')

class MediaFilePathsJailTestCase(unittest.TestCase):

    def _check_relative_path(self, filepaths: MediaFilePaths, path: str) -> None:
        if False:
            while True:
                i = 10
        'Passes a relative path through the jail check.\n\n        Args:\n            filepaths: The `MediaFilePaths` instance.\n            path: A path relative to the media store directory.\n\n        Raises:\n            ValueError: If the jail check fails.\n        '

        @_wrap_with_jail_check(relative=True)
        def _make_relative_path(self: MediaFilePaths, path: str) -> str:
            if False:
                for i in range(10):
                    print('nop')
            return path
        _make_relative_path(filepaths, path)

    def _check_absolute_path(self, filepaths: MediaFilePaths, path: str) -> None:
        if False:
            while True:
                i = 10
        'Passes an absolute path through the jail check.\n\n        Args:\n            filepaths: The `MediaFilePaths` instance.\n            path: A path relative to the media store directory.\n\n        Raises:\n            ValueError: If the jail check fails.\n        '

        @_wrap_with_jail_check(relative=False)
        def _make_absolute_path(self: MediaFilePaths, path: str) -> str:
            if False:
                i = 10
                return i + 15
            return os.path.join(self.base_path, path)
        _make_absolute_path(filepaths, path)

    def test_traversal_inside(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test the jail check for paths that stay within the media directory.'
        filepaths = MediaFilePaths('/media_store')
        path = 'url_cache/2020-01-02/../../GerZNDnDZVjsOtar'
        self._check_relative_path(filepaths, path)
        self._check_absolute_path(filepaths, path)

    def test_traversal_outside(self) -> None:
        if False:
            print('Hello World!')
        'Test that the jail check fails for paths that escape the media directory.'
        filepaths = MediaFilePaths('/media_store')
        path = 'url_cache/2020-01-02/../../../GerZNDnDZVjsOtar'
        with self.assertRaises(ValueError):
            self._check_relative_path(filepaths, path)
        with self.assertRaises(ValueError):
            self._check_absolute_path(filepaths, path)

    def test_traversal_reentry(self) -> None:
        if False:
            while True:
                i = 10
        'Test the jail check for paths that exit and re-enter the media directory.'
        filepaths = MediaFilePaths('/media_store')
        path = 'url_cache/2020-01-02/../../../media_store/GerZNDnDZVjsOtar'
        self._check_relative_path(filepaths, path)
        self._check_absolute_path(filepaths, path)

    def test_symlink(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that a symlink does not cause the jail check to fail.'
        media_store_path = self.mktemp()
        os.symlink('/mnt/synapse/media_store', media_store_path)
        filepaths = MediaFilePaths(media_store_path)
        self._check_relative_path(filepaths, 'url_cache/2020-01-02/GerZNDnDZVjsOtar')
        self._check_absolute_path(filepaths, 'url_cache/2020-01-02/GerZNDnDZVjsOtar')
        filepaths = MediaFilePaths(os.path.abspath(media_store_path))
        self._check_relative_path(filepaths, 'url_cache/2020-01-02/GerZNDnDZVjsOtar')
        self._check_absolute_path(filepaths, 'url_cache/2020-01-02/GerZNDnDZVjsOtar')

    def test_symlink_subdirectory(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test that a symlinked subdirectory does not cause the jail check to fail.'
        media_store_path = self.mktemp()
        os.mkdir(media_store_path)
        os.symlink('/mnt/synapse/media_store_url_cache', os.path.join(media_store_path, 'url_cache'))
        filepaths = MediaFilePaths(media_store_path)
        self._check_relative_path(filepaths, 'url_cache/2020-01-02/GerZNDnDZVjsOtar')
        self._check_absolute_path(filepaths, 'url_cache/2020-01-02/GerZNDnDZVjsOtar')
        filepaths = MediaFilePaths(os.path.abspath(media_store_path))
        self._check_relative_path(filepaths, 'url_cache/2020-01-02/GerZNDnDZVjsOtar')
        self._check_absolute_path(filepaths, 'url_cache/2020-01-02/GerZNDnDZVjsOtar')