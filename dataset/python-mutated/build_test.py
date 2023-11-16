"""Unit tests for scripts/build.py."""
from __future__ import annotations
import ast
import collections
import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
import threading
from core import feconf
from core import utils
from core.tests import test_utils
from typing import ContextManager, Deque, Dict, Iterator, List, Tuple, Union
from . import build
from . import common
from . import install_python_dev_dependencies
from . import scripts_test_utils
from . import servers
TEST_DIR = os.path.join('core', 'tests', 'build', '')
TEST_SOURCE_DIR = os.path.join('core', 'tests', 'build_sources')
MOCK_ASSETS_DEV_DIR = os.path.join(TEST_SOURCE_DIR, 'assets', '')
MOCK_ASSETS_OUT_DIR = os.path.join(TEST_DIR, 'static', 'assets', '')
MOCK_EXTENSIONS_DEV_DIR = os.path.join(TEST_SOURCE_DIR, 'extensions', '')
MOCK_TEMPLATES_DEV_DIR = os.path.join(TEST_SOURCE_DIR, 'templates', '')
MOCK_TSC_OUTPUT_LOG_FILEPATH = os.path.join(TEST_SOURCE_DIR, 'mock_tsc_output_log.txt')
INVALID_FILENAME = 'invalid_filename.css'
INVALID_INPUT_FILEPATH = os.path.join(TEST_DIR, INVALID_FILENAME)
INVALID_OUTPUT_FILEPATH = os.path.join(TEST_DIR, INVALID_FILENAME)
EMPTY_DIR = os.path.join(TEST_DIR, 'empty', '')

def mock_managed_process(*unused_args: str, **unused_kwargs: str) -> ContextManager[scripts_test_utils.PopenStub]:
    if False:
        print('Hello World!')
    'Mock method for replacing the managed_process() functions.\n\n    Returns:\n        Context manager. A context manager that always yields a mock\n        process.\n    '
    return contextlib.nullcontext(enter_result=scripts_test_utils.PopenStub(alive=False))

class BuildTests(test_utils.GenericTestBase):
    """Test the build methods."""

    def tearDown(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().tearDown()
        build.safe_delete_directory_tree(TEST_DIR)
        build.safe_delete_directory_tree(EMPTY_DIR)

    def test_minify_func_with_invalid_filepath(self) -> None:
        if False:
            return 10
        'Tests minify_func with an invalid filepath.'
        with self.assertRaisesRegex(OSError, '\\[Errno 2\\] No such file or directory:'):
            build.minify_func(INVALID_INPUT_FILEPATH, INVALID_OUTPUT_FILEPATH, INVALID_FILENAME)

    def test_minify_and_create_sourcemap(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests _minify_and_create_sourcemap with an invalid filepath.'
        with self.assertRaisesRegex(subprocess.CalledProcessError, 'returned non-zero exit status 1') as called_process:
            build._minify_and_create_sourcemap(INVALID_INPUT_FILEPATH, INVALID_OUTPUT_FILEPATH)
        self.assertEqual(called_process.exception.returncode, 1)

    def test_minify_and_create_sourcemap_under_docker_environment(self) -> None:
        if False:
            while True:
                i = 10
        'Tests _minify_and_create_sourcemap with an invalid filepath.'

        def mock_subprocess_check_call(command: str, **kwargs: bool) -> None:
            if False:
                for i in range(10):
                    print('nop')
            'Mock method for replacing subprocess.check_call().'
            excepted_cmd = "node /app/oppia/node_modules/uglify-js/bin/uglifyjs /app/oppia/third_party/generated/js/third_party.js -c -m --source-map includeSources,url='third_party.min.js.map' -o /app/oppia/third_party/generated/js/third_party.min.js"
            self.assertEqual(command, excepted_cmd)
        with self.swap(feconf, 'OPPIA_IS_DOCKERIZED', True):
            with self.swap(subprocess, 'check_call', mock_subprocess_check_call):
                build._minify_and_create_sourcemap(INVALID_INPUT_FILEPATH, INVALID_OUTPUT_FILEPATH)

    def test_join_files(self) -> None:
        if False:
            while True:
                i = 10
        'Determine third_party.js contains the content of the first 10 JS\n        files in /third_party/static.\n        '
        third_party_js_stream = io.StringIO()
        dependency_filepaths = build.get_dependencies_filepaths()
        build._join_files(dependency_filepaths['js'], third_party_js_stream)
        counter = 0
        js_file_count = 10
        for js_filepath in dependency_filepaths['js']:
            if counter == js_file_count:
                break
            with utils.open_file(js_filepath, 'r') as js_file:
                for line in js_file:
                    self.assertIn(line, third_party_js_stream.getvalue())
            counter += 1

    def test_generate_copy_tasks_for_fonts(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test _generate_copy_tasks_for_fonts ensures that the number of copy\n        tasks matches the number of font files.\n        '
        copy_tasks: Deque[threading.Thread] = collections.deque()
        dependency_filepaths = build.get_dependencies_filepaths()
        test_target = os.path.join('target', 'fonts', '')
        self.assertEqual(len(copy_tasks), 0)
        copy_tasks += build._generate_copy_tasks_for_fonts(dependency_filepaths['fonts'], test_target)
        self.assertEqual(len(copy_tasks), len(dependency_filepaths['fonts']))

    def test_insert_hash(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test _insert_hash returns correct filenames with provided hashes.'
        self.assertEqual(build._insert_hash('file.js', '123456'), 'file.123456.js')
        self.assertEqual(build._insert_hash('path/to/file.js', '654321'), 'path/to/file.654321.js')
        self.assertEqual(build._insert_hash('file.min.js', 'abcdef'), 'file.min.abcdef.js')
        self.assertEqual(build._insert_hash('path/to/file.min.js', 'fedcba'), 'path/to/file.min.fedcba.js')

    def test_get_file_count(self) -> None:
        if False:
            print('Hello World!')
        'Test get_file_count returns the correct number of files, excluding\n        file with extensions in FILE_EXTENSIONS_TO_IGNORE and files that should\n        not be built.\n        '
        all_inclusive_file_count = 0
        for (_, _, files) in os.walk(MOCK_EXTENSIONS_DEV_DIR):
            all_inclusive_file_count += len(files)
        ignored_file_count = 0
        for (_, _, files) in os.walk(MOCK_EXTENSIONS_DEV_DIR):
            for filename in files:
                if not build.should_file_be_built(filename) or any((filename.endswith(p) for p in build.FILE_EXTENSIONS_TO_IGNORE)):
                    ignored_file_count += 1
        self.assertEqual(all_inclusive_file_count - ignored_file_count, build.get_file_count(MOCK_EXTENSIONS_DEV_DIR))

    def test_compare_file_count(self) -> None:
        if False:
            while True:
                i = 10
        'Test _compare_file_count raises exception when there is a\n        mismatched file count between 2 dirs list.\n        '
        build.ensure_directory_exists(EMPTY_DIR)
        source_dir_file_count = build.get_file_count(EMPTY_DIR)
        assert source_dir_file_count == 0
        target_dir_file_count = build.get_file_count(MOCK_ASSETS_DEV_DIR)
        assert target_dir_file_count > 0
        with self.assertRaisesRegex(ValueError, '%s files in first dir list != %s files in second dir list' % (source_dir_file_count, target_dir_file_count)):
            build._compare_file_count([EMPTY_DIR], [MOCK_ASSETS_DEV_DIR])
        mock_extensions_dir_list = [MOCK_EXTENSIONS_DEV_DIR]
        target_dir_file_count = build.get_file_count(MOCK_EXTENSIONS_DEV_DIR)
        assert target_dir_file_count > 0
        with self.assertRaisesRegex(ValueError, '%s files in first dir list != %s files in second dir list' % (source_dir_file_count, target_dir_file_count)):
            build._compare_file_count([EMPTY_DIR], mock_extensions_dir_list)
        build.safe_delete_directory_tree(EMPTY_DIR)

    def test_verify_filepath_hash(self) -> None:
        if False:
            print('Hello World!')
        'Test _verify_filepath_hash raises exception:\n            1) When there is an empty hash dict.\n            2) When a filename is expected to contain hash but does not.\n            3) When there is a hash in filename that cannot be found in\n                hash dict.\n        '
        file_hashes: Dict[str, str] = {}
        base_filename = 'base.html'
        with self.assertRaisesRegex(ValueError, 'Hash dict is empty'):
            build._verify_filepath_hash(base_filename, file_hashes)
        file_hashes = {base_filename: test_utils.generate_random_hexa_str()}
        with self.assertRaisesRegex(ValueError, '%s is expected to contain MD5 hash' % base_filename):
            build._verify_filepath_hash(base_filename, file_hashes)
        base_without_hash_filename = 'base_without_hash.html'
        build._verify_filepath_hash(base_without_hash_filename, file_hashes)
        bad_filepath = 'README'
        with self.assertRaisesRegex(ValueError, 'Filepath has less than 2 partitions after splitting'):
            build._verify_filepath_hash(bad_filepath, file_hashes)
        hashed_base_filename = build._insert_hash(base_filename, test_utils.generate_random_hexa_str())
        with self.assertRaisesRegex(KeyError, 'Hash from file named %s does not match hash dict values' % hashed_base_filename):
            build._verify_filepath_hash(hashed_base_filename, file_hashes)

    def test_process_html(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test process_html removes whitespaces.'
        base_html_source_path = os.path.join(MOCK_TEMPLATES_DEV_DIR, 'base.html')
        build._ensure_files_exist([base_html_source_path])
        minified_html_file_stream = io.StringIO()
        with utils.open_file(base_html_source_path, 'r') as source_base_file:
            source_base_file_content = source_base_file.read()
            self.assertRegex(source_base_file_content, '\\s{2,}', msg='No white spaces detected in %s unexpectedly' % base_html_source_path)
        with utils.open_file(base_html_source_path, 'r') as source_base_file:
            build.process_html(source_base_file, minified_html_file_stream)
        minified_html_file_content = minified_html_file_stream.getvalue()
        self.assertNotRegex(minified_html_file_content, '\\s{2,}', msg='All white spaces must be removed from %s' % base_html_source_path)

    def test_should_file_be_built(self) -> None:
        if False:
            print('Hello World!')
        'Test should_file_be_built returns the correct boolean value for\n        filepath that should be built.\n        '
        service_ts_filepath = os.path.join('core', 'pages', 'AudioService.ts')
        spec_js_filepath = os.path.join('core', 'pages', 'AudioServiceSpec.js')
        webdriverio_filepath = os.path.join('extensions', 'webdriverio.js')
        python_controller_filepath = os.path.join('base.py')
        pyc_test_filepath = os.path.join('core', 'controllers', 'base.pyc')
        python_test_filepath = os.path.join('core', 'tests', 'base_test.py')
        self.assertFalse(build.should_file_be_built(spec_js_filepath))
        self.assertFalse(build.should_file_be_built(webdriverio_filepath))
        self.assertFalse(build.should_file_be_built(service_ts_filepath))
        self.assertFalse(build.should_file_be_built(python_test_filepath))
        self.assertFalse(build.should_file_be_built(pyc_test_filepath))
        self.assertTrue(build.should_file_be_built(python_controller_filepath))
        with self.swap(build, 'JS_FILENAME_SUFFIXES_TO_IGNORE', ('Service.js',)):
            self.assertTrue(build.should_file_be_built(spec_js_filepath))

    def test_hash_should_be_inserted(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test hash_should_be_inserted returns the correct boolean value\n        for filepath that should be hashed.\n        '
        with self.swap(build, 'FILEPATHS_NOT_TO_RENAME', ('*.py', 'path/to/fonts/*', 'path/to/third_party.min.js.map', 'path/to/third_party.min.css.map')):
            self.assertFalse(build.hash_should_be_inserted('path/to/fonts/fontawesome-webfont.svg'))
            self.assertFalse(build.hash_should_be_inserted('path/to/third_party.min.css.map'))
            self.assertFalse(build.hash_should_be_inserted('path/to/third_party.min.js.map'))
            self.assertTrue(build.hash_should_be_inserted('path/to/wrongFonts/fonta.eot'))
            self.assertTrue(build.hash_should_be_inserted('rich_text_components/Video/protractor.js'))
            self.assertFalse(build.hash_should_be_inserted('main.py'))
            self.assertFalse(build.hash_should_be_inserted('extensions/domain.py'))

    def test_generate_copy_tasks_to_copy_from_source_to_target(self) -> None:
        if False:
            return 10
        'Test generate_copy_tasks_to_copy_from_source_to_target queues up\n        the same number of copy tasks as the number of files in the directory.\n        '
        assets_hashes = build.get_file_hashes(MOCK_ASSETS_DEV_DIR)
        total_file_count = build.get_file_count(MOCK_ASSETS_DEV_DIR)
        copy_tasks: Deque[threading.Thread] = collections.deque()
        self.assertEqual(len(copy_tasks), 0)
        copy_tasks += build.generate_copy_tasks_to_copy_from_source_to_target(MOCK_ASSETS_DEV_DIR, MOCK_ASSETS_OUT_DIR, assets_hashes)
        self.assertEqual(len(copy_tasks), total_file_count)

    def test_is_file_hash_provided_to_frontend(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test is_file_hash_provided_to_frontend returns the correct boolean\n        value for filepath that should be provided to frontend.\n        '
        with self.swap(build, 'FILEPATHS_PROVIDED_TO_FRONTEND', ('path/to/file.js', 'path/to/file.html', 'file.js')):
            self.assertTrue(build.is_file_hash_provided_to_frontend('path/to/file.js'))
            self.assertTrue(build.is_file_hash_provided_to_frontend('path/to/file.html'))
            self.assertTrue(build.is_file_hash_provided_to_frontend('file.js'))
        with self.swap(build, 'FILEPATHS_PROVIDED_TO_FRONTEND', ('path/to/*', '*.js', '*_end.html')):
            self.assertTrue(build.is_file_hash_provided_to_frontend('path/to/file.js'))
            self.assertTrue(build.is_file_hash_provided_to_frontend('path/to/file.html'))
            self.assertTrue(build.is_file_hash_provided_to_frontend('file.js'))
            self.assertFalse(build.is_file_hash_provided_to_frontend('path/file.css'))
            self.assertTrue(build.is_file_hash_provided_to_frontend('good_end.html'))
            self.assertFalse(build.is_file_hash_provided_to_frontend('bad_end.css'))

    def test_get_filepaths_by_extensions(self) -> None:
        if False:
            while True:
                i = 10
        'Test get_filepaths_by_extensions only returns filepaths in\n        directory with given extensions.\n        '
        filepaths: List[str] = []
        build.ensure_directory_exists(MOCK_ASSETS_DEV_DIR)
        extensions: Tuple[str, ...] = ('.json', '.svg')
        self.assertEqual(len(filepaths), 0)
        filepaths = build.get_filepaths_by_extensions(MOCK_ASSETS_DEV_DIR, extensions)
        for filepath in filepaths:
            self.assertTrue(any((filepath.endswith(p) for p in extensions)))
        file_count = 0
        for (_, _, filenames) in os.walk(MOCK_ASSETS_DEV_DIR):
            for filename in filenames:
                if any((filename.endswith(p) for p in extensions)):
                    file_count += 1
        self.assertEqual(len(filepaths), file_count)
        filepaths = []
        extensions = ('.pdf', '.viminfo', '.idea')
        self.assertEqual(len(filepaths), 0)
        filepaths = build.get_filepaths_by_extensions(MOCK_ASSETS_DEV_DIR, extensions)
        self.assertEqual(len(filepaths), 0)

    def test_get_file_hashes(self) -> None:
        if False:
            while True:
                i = 10
        'Test get_file_hashes gets hashes of all files in directory,\n        excluding file with extensions in FILE_EXTENSIONS_TO_IGNORE.\n        '
        with self.swap(build, 'FILE_EXTENSIONS_TO_IGNORE', ('.html',)):
            file_hashes: Dict[str, str] = {}
            self.assertEqual(len(file_hashes), 0)
            file_hashes = build.get_file_hashes(MOCK_EXTENSIONS_DEV_DIR)
            self.assertGreater(len(file_hashes), 0)
            for filepath in file_hashes:
                abs_filepath = os.path.join(MOCK_EXTENSIONS_DEV_DIR, filepath)
                self.assertTrue(os.path.isfile(abs_filepath))
                self.assertFalse(filepath.endswith('.html'))

    def test_filter_hashes(self) -> None:
        if False:
            print('Hello World!')
        'Test filter_hashes filters the provided hash correctly.'
        with self.swap(build, 'FILEPATHS_PROVIDED_TO_FRONTEND', ('*',)):
            hashes = {'path/to/file.js': '123456', 'path/file.min.js': '123456'}
            filtered_hashes = build.filter_hashes(hashes)
            self.assertEqual(filtered_hashes['/path/to/file.js'], hashes['path/to/file.js'])
            self.assertEqual(filtered_hashes['/path/file.min.js'], hashes['path/file.min.js'])
        with self.swap(build, 'FILEPATHS_PROVIDED_TO_FRONTEND', ('test_path/*', 'path/to/file.js')):
            hashes = {'path/to/file.js': '123456', 'test_path/to/file.html': '123456', 'test_path/to/file.js': 'abcdef', 'path/path/file.js': 'zyx123', 'file.html': '321xyz'}
            filtered_hashes = build.filter_hashes(hashes)
            self.assertIn('/path/to/file.js', filtered_hashes)
            self.assertIn('/test_path/to/file.html', filtered_hashes)
            self.assertIn('/test_path/to/file.js', filtered_hashes)
            self.assertNotIn('/path/path/file.js', filtered_hashes)
            self.assertNotIn('/file.html', filtered_hashes)

    def test_save_hashes_to_file(self) -> None:
        if False:
            return 10
        'Test save_hashes_to_file saves provided hash dict correctly to\n        JSON file.\n        '
        hashes_path = os.path.join(MOCK_ASSETS_OUT_DIR, 'hashes.json')
        with self.swap(build, 'FILEPATHS_PROVIDED_TO_FRONTEND', ('*',)):
            with self.swap(build, 'HASHES_JSON_FILEPATH', hashes_path):
                hashes = {'path/file.js': '123456'}
                build.save_hashes_to_file(hashes)
                with utils.open_file(hashes_path, 'r') as hashes_file:
                    self.assertEqual(hashes_file.read(), '{"/path/file.js": "123456"}\n')
                hashes = {'file.js': '123456', 'file.min.js': '654321'}
                build.save_hashes_to_file(hashes)
                with utils.open_file(hashes_path, 'r') as hashes_file:
                    self.assertEqual(ast.literal_eval(hashes_file.read()), {'/file.min.js': '654321', '/file.js': '123456'})
                os.remove(hashes_path)

    def test_execute_tasks(self) -> None:
        if False:
            return 10
        'Test _execute_tasks joins all threads after executing all tasks.'
        build_tasks: Deque[threading.Thread] = collections.deque()
        build_thread_names: List[Union[threading.Thread, str]] = []
        task_count = 2
        count = task_count
        while count:
            thread_name = 'Build-test-thread-%s' % count
            build_thread_names.append(thread_name)
            task = threading.Thread(name=thread_name, target=build.minify_func, args=(INVALID_INPUT_FILEPATH, INVALID_OUTPUT_FILEPATH, INVALID_FILENAME))
            build_tasks.append(task)
            count -= 1
        extra_build_threads = [thread.name for thread in threading.enumerate() if thread in build_thread_names]
        self.assertEqual(len(extra_build_threads), 0)
        build._execute_tasks(build_tasks)
        with self.assertRaisesRegex(OSError, 'threads can only be started once'):
            build._execute_tasks(build_tasks)
        extra_build_threads = [thread.name for thread in threading.enumerate() if thread in build_thread_names]
        self.assertEqual(len(extra_build_threads), 0)

    def test_generate_build_tasks_to_build_all_files_in_directory(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test generate_build_tasks_to_build_all_files_in_directory queues up\n        the same number of build tasks as the number of files in the source\n        directory.\n        '
        tasks: Deque[threading.Thread] = collections.deque()
        self.assertEqual(len(tasks), 0)
        tasks = build.generate_build_tasks_to_build_all_files_in_directory(MOCK_ASSETS_DEV_DIR, MOCK_ASSETS_OUT_DIR)
        total_file_count = build.get_file_count(MOCK_ASSETS_DEV_DIR)
        self.assertEqual(len(tasks), total_file_count)

    def test_generate_build_tasks_to_build_files_from_filepaths(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test generate_build_tasks_to_build_files_from_filepaths queues up a\n        corresponding number of build tasks to the number of file changes.\n        '
        new_filename = 'dependencies.json'
        recently_changed_filenames = [os.path.join(MOCK_ASSETS_DEV_DIR, new_filename)]
        build_tasks: Deque[threading.Thread] = collections.deque()
        self.assertEqual(len(build_tasks), 0)
        build_tasks += build.generate_build_tasks_to_build_files_from_filepaths(MOCK_ASSETS_DEV_DIR, MOCK_ASSETS_OUT_DIR, recently_changed_filenames)
        self.assertEqual(len(build_tasks), len(recently_changed_filenames))
        build_tasks.clear()
        svg_filepaths = build.get_filepaths_by_extensions(MOCK_ASSETS_DEV_DIR, ('.svg',))
        self.assertGreater(len(svg_filepaths), 0)
        self.assertEqual(len(build_tasks), 0)
        build_tasks += build.generate_build_tasks_to_build_files_from_filepaths(MOCK_ASSETS_DEV_DIR, MOCK_ASSETS_OUT_DIR, svg_filepaths)
        self.assertEqual(len(build_tasks), len(svg_filepaths))

    def test_generate_build_tasks_to_build_directory(self) -> None:
        if False:
            print('Hello World!')
        'Test generate_build_tasks_to_build_directory queues up a\n        corresponding number of build tasks according to the given scenario.\n        '
        extensions_dirnames_to_dirpaths = {'dev_dir': MOCK_EXTENSIONS_DEV_DIR, 'staging_dir': os.path.join(TEST_DIR, 'backend_prod_files', 'extensions', ''), 'out_dir': os.path.join(TEST_DIR, 'build', 'extensions', '')}
        file_hashes = build.get_file_hashes(MOCK_EXTENSIONS_DEV_DIR)
        build_dir_tasks: Deque[threading.Thread] = collections.deque()
        build_all_files_tasks = build.generate_build_tasks_to_build_all_files_in_directory(MOCK_EXTENSIONS_DEV_DIR, extensions_dirnames_to_dirpaths['out_dir'])
        self.assertGreater(len(build_all_files_tasks), 0)
        self.assertEqual(len(build_dir_tasks), 0)
        build_dir_tasks += build.generate_build_tasks_to_build_directory(extensions_dirnames_to_dirpaths)
        self.assertEqual(len(build_dir_tasks), len(build_all_files_tasks))
        build.safe_delete_directory_tree(TEST_DIR)
        build_dir_tasks.clear()
        build.ensure_directory_exists(extensions_dirnames_to_dirpaths['staging_dir'])
        self.assertEqual(len(build_dir_tasks), 0)
        build_dir_tasks += build.generate_build_tasks_to_build_directory(extensions_dirnames_to_dirpaths)
        self.assertEqual(len(build_dir_tasks), len(build_all_files_tasks))
        build.safe_delete_directory_tree(TEST_DIR)
        build.ensure_directory_exists(extensions_dirnames_to_dirpaths['staging_dir'])
        build._execute_tasks(build_dir_tasks)
        self.assertEqual(threading.active_count(), 1)
        build._execute_tasks(build.generate_copy_tasks_to_copy_from_source_to_target(extensions_dirnames_to_dirpaths['staging_dir'], extensions_dirnames_to_dirpaths['out_dir'], file_hashes))
        build_dir_tasks.clear()
        self.assertEqual(len(build_dir_tasks), 0)
        build_dir_tasks += build.generate_build_tasks_to_build_directory(extensions_dirnames_to_dirpaths)
        file_extensions_to_always_rebuild = ('.html', '.py')
        always_rebuilt_filepaths = build.get_filepaths_by_extensions(MOCK_EXTENSIONS_DEV_DIR, file_extensions_to_always_rebuild)
        self.assertGreater(len(always_rebuilt_filepaths), 0)
        self.assertEqual(len(build_dir_tasks), len(always_rebuilt_filepaths))
        build.safe_delete_directory_tree(TEST_DIR)

    def test_re_build_recently_changed_files_at_dev_dir(self) -> None:
        if False:
            while True:
                i = 10
        temp_file = tempfile.NamedTemporaryFile()
        temp_file_name = '%ssome_file.js' % MOCK_EXTENSIONS_DEV_DIR
        setattr(temp_file, 'name', temp_file_name)
        with utils.open_file('%ssome_file.js' % MOCK_EXTENSIONS_DEV_DIR, 'w') as tmp:
            tmp.write(u'Some content.')
        extensions_dirnames_to_dirpaths = {'dev_dir': MOCK_EXTENSIONS_DEV_DIR, 'staging_dir': os.path.join(TEST_DIR, 'backend_prod_files', 'extensions', ''), 'out_dir': os.path.join(TEST_DIR, 'build', 'extensions', '')}
        build_dir_tasks: Deque[threading.Thread] = collections.deque()
        build_all_files_tasks = build.generate_build_tasks_to_build_all_files_in_directory(MOCK_EXTENSIONS_DEV_DIR, extensions_dirnames_to_dirpaths['out_dir'])
        self.assertGreater(len(build_all_files_tasks), 0)
        self.assertEqual(len(build_dir_tasks), 0)
        build_dir_tasks += build.generate_build_tasks_to_build_directory(extensions_dirnames_to_dirpaths)
        self.assertEqual(len(build_dir_tasks), len(build_all_files_tasks))
        build.safe_delete_directory_tree(TEST_DIR)
        build_dir_tasks.clear()
        build.ensure_directory_exists(extensions_dirnames_to_dirpaths['staging_dir'])
        self.assertEqual(len(build_dir_tasks), 0)
        build_dir_tasks = build.generate_build_tasks_to_build_directory(extensions_dirnames_to_dirpaths)
        file_extensions_to_always_rebuild = ('.py', '.js', '.html')
        always_rebuilt_filepaths = build.get_filepaths_by_extensions(MOCK_EXTENSIONS_DEV_DIR, file_extensions_to_always_rebuild)
        self.assertEqual(sorted(always_rebuilt_filepaths), sorted(['base.py', 'CodeRepl.py', '__init__.py', 'some_file.js', 'DragAndDropSortInput.py', 'code_repl_prediction.html']))
        self.assertGreater(len(always_rebuilt_filepaths), 0)
        self.assertEqual(len(build_dir_tasks), len(always_rebuilt_filepaths))
        self.assertIn('some_file.js', always_rebuilt_filepaths)
        self.assertNotIn('some_file.js', build_dir_tasks)
        build.safe_delete_directory_tree(TEST_DIR)
        temp_file.close()
        if os.path.isfile(temp_file_name):
            os.remove(temp_file_name)

    def test_get_recently_changed_filenames(self) -> None:
        if False:
            return 10
        'Test get_recently_changed_filenames detects file recently added.'
        build.ensure_directory_exists(EMPTY_DIR)
        assets_hashes = build.get_file_hashes(MOCK_ASSETS_DEV_DIR)
        recently_changed_filenames: List[str] = []
        self.assertEqual(len(recently_changed_filenames), 0)
        recently_changed_filenames = build.get_recently_changed_filenames(assets_hashes, EMPTY_DIR)
        with self.swap(build, 'FILE_EXTENSIONS_TO_IGNORE', ('.html', '.py')):
            self.assertEqual(len(recently_changed_filenames), build.get_file_count(MOCK_ASSETS_DEV_DIR))
        build.safe_delete_directory_tree(EMPTY_DIR)

    def test_generate_delete_tasks_to_remove_deleted_files(self) -> None:
        if False:
            return 10
        'Test generate_delete_tasks_to_remove_deleted_files queues up the\n        same number of deletion task as the number of deleted files.\n        '
        delete_tasks: Deque[threading.Thread] = collections.deque()
        file_hashes: Dict[str, str] = {}
        self.assertEqual(len(delete_tasks), 0)
        delete_tasks += build.generate_delete_tasks_to_remove_deleted_files(file_hashes, MOCK_TEMPLATES_DEV_DIR)
        self.assertEqual(len(delete_tasks), build.get_file_count(MOCK_TEMPLATES_DEV_DIR))

    def test_generate_app_yaml_with_deploy_mode(self) -> None:
        if False:
            i = 10
            return i + 15
        mock_dev_yaml_filepath = 'mock_app_dev.yaml'
        mock_yaml_filepath = 'mock_app.yaml'
        app_dev_yaml_filepath_swap = self.swap(build, 'APP_DEV_YAML_FILEPATH', mock_dev_yaml_filepath)
        app_yaml_filepath_swap = self.swap(build, 'APP_YAML_FILEPATH', mock_yaml_filepath)
        env_vars_to_remove_from_deployed_app_yaml_swap = self.swap(build, 'ENV_VARS_TO_REMOVE_FROM_DEPLOYED_APP_YAML', ['FIREBASE_AUTH_EMULATOR_HOST'])
        app_dev_yaml_temp_file = tempfile.NamedTemporaryFile()
        setattr(app_dev_yaml_temp_file, 'name', mock_dev_yaml_filepath)
        with utils.open_file(mock_dev_yaml_filepath, 'w') as tmp:
            with self.swap(feconf, 'OPPIA_IS_DOCKERIZED', True):
                tmp.write('Some content in mock_app_dev.yaml\n')
                tmp.write('  FIREBASE_AUTH_EMULATOR_HOST: "firebase:9099"\n')
                tmp.write('version: default')
        app_yaml_temp_file = tempfile.NamedTemporaryFile()
        setattr(app_yaml_temp_file, 'name', mock_yaml_filepath)
        with utils.open_file(mock_yaml_filepath, 'w') as tmp:
            tmp.write(u'Initial content in mock_app.yaml')
        with app_dev_yaml_filepath_swap, app_yaml_filepath_swap:
            with env_vars_to_remove_from_deployed_app_yaml_swap:
                build.generate_app_yaml(deploy_mode=True)
        with utils.open_file(mock_yaml_filepath, 'r') as yaml_file:
            content = yaml_file.read()
        self.assertEqual(content, '# THIS FILE IS AUTOGENERATED, DO NOT MODIFY\nSome content in mock_app_dev.yaml\n')
        app_yaml_temp_file.close()
        app_dev_yaml_temp_file.close()

    def test_generate_app_yaml_with_deploy_mode_with_nonexistent_var_raises(self) -> None:
        if False:
            while True:
                i = 10
        mock_dev_yaml_filepath = 'mock_app_dev.yaml'
        mock_yaml_filepath = 'mock_app.yaml'
        app_dev_yaml_filepath_swap = self.swap(build, 'APP_DEV_YAML_FILEPATH', mock_dev_yaml_filepath)
        app_yaml_filepath_swap = self.swap(build, 'APP_YAML_FILEPATH', mock_yaml_filepath)
        env_vars_to_remove_from_deployed_app_yaml_swap = self.swap(build, 'ENV_VARS_TO_REMOVE_FROM_DEPLOYED_APP_YAML', ['DATASTORE_HOST'])
        app_dev_yaml_temp_file = tempfile.NamedTemporaryFile()
        setattr(app_dev_yaml_temp_file, 'name', mock_dev_yaml_filepath)
        firebase_host = 'firebase' if feconf.OPPIA_IS_DOCKERIZED else 'localhost'
        with utils.open_file(mock_dev_yaml_filepath, 'w') as tmp:
            tmp.write('Some content in mock_app_dev.yaml\n')
            tmp.write('  FIREBASE_AUTH_EMULATOR_HOST: "%s:9099"\n' % firebase_host)
            tmp.write('version: default')
        app_yaml_temp_file = tempfile.NamedTemporaryFile()
        setattr(app_yaml_temp_file, 'name', mock_yaml_filepath)
        with utils.open_file(mock_yaml_filepath, 'w') as tmp:
            tmp.write('Initial content in mock_app.yaml')
        with app_dev_yaml_filepath_swap, app_yaml_filepath_swap:
            with env_vars_to_remove_from_deployed_app_yaml_swap:
                with self.assertRaisesRegex(Exception, "Environment variable 'DATASTORE_HOST' to be removed does not exist."):
                    build.generate_app_yaml(deploy_mode=True)
        with utils.open_file(mock_yaml_filepath, 'r') as yaml_file:
            content = yaml_file.read()
        self.assertEqual(content, 'Initial content in mock_app.yaml')
        app_yaml_temp_file.close()
        app_dev_yaml_temp_file.close()

    def test_safe_delete_file(self) -> None:
        if False:
            print('Hello World!')
        'Test safe_delete_file with both existent and non-existent\n        filepath.\n        '
        temp_file = tempfile.NamedTemporaryFile()
        setattr(temp_file, 'name', 'some_file.txt')
        with utils.open_file('some_file.txt', 'w') as tmp:
            tmp.write(u'Some content.')
        self.assertTrue(os.path.isfile('some_file.txt'))
        build.safe_delete_file('some_file.txt')
        self.assertFalse(os.path.isfile('some_file.txt'))
        non_existent_filepaths = [INVALID_INPUT_FILEPATH]
        error_message = 'File %s does not exist.' % re.escape(non_existent_filepaths[0])
        with self.assertRaisesRegex(OSError, error_message):
            build.safe_delete_file(non_existent_filepaths[0])

    def test_minify_third_party_libs(self) -> None:
        if False:
            for i in range(10):
                print('nop')

        def _mock_safe_delete_file(unused_filepath: str) -> None:
            if False:
                return 10
            'Mocks build.safe_delete_file().'
            pass
        self.assertFalse(os.path.isfile('core/tests/data/third_party/css/third_party.min.css'))
        self.assertFalse(os.path.isfile('core/tests/data/third_party/js/third_party.min.js'))
        self.assertFalse(os.path.isfile('core/tests/data/third_party/js/third_party.min.js.map'))
        with self.swap(build, 'safe_delete_file', _mock_safe_delete_file):
            build.minify_third_party_libs('core/tests/data/third_party')
        self.assertTrue(os.path.isfile('core/tests/data/third_party/css/third_party.min.css'))
        self.assertTrue(os.path.isfile('core/tests/data/third_party/js/third_party.min.js'))
        self.assertTrue(os.path.isfile('core/tests/data/third_party/js/third_party.min.js.map'))
        self.assertLess(os.path.getsize('core/tests/data/third_party/css/third_party.min.css'), os.path.getsize('core/tests/data/third_party/css/third_party.css'))
        self.assertLess(os.path.getsize('core/tests/data/third_party/js/third_party.min.js'), os.path.getsize('core/tests/data/third_party/js/third_party.js'))
        build.safe_delete_file('core/tests/data/third_party/css/third_party.min.css')
        build.safe_delete_file('core/tests/data/third_party/js/third_party.min.js')
        build.safe_delete_file('core/tests/data/third_party/js/third_party.min.js.map')

    def test_clean(self) -> None:
        if False:
            print('Hello World!')
        check_function_calls = {'safe_delete_directory_tree_gets_called': 0}
        expected_check_function_calls = {'safe_delete_directory_tree_gets_called': 3}

        def mock_safe_delete_directory_tree(unused_path: str) -> None:
            if False:
                print('Hello World!')
            check_function_calls['safe_delete_directory_tree_gets_called'] += 1
        with self.swap(build, 'safe_delete_directory_tree', mock_safe_delete_directory_tree):
            build.clean()
        self.assertEqual(check_function_calls, expected_check_function_calls)

    def test_build_with_prod_env(self) -> None:
        if False:
            return 10
        check_function_calls = {'build_using_webpack_gets_called': False, 'ensure_files_exist_gets_called': False, 'modify_constants_gets_called': False, 'compare_file_count_gets_called': False, 'generate_python_package_called': False, 'clean_gets_called': False}
        expected_check_function_calls = {'build_using_webpack_gets_called': True, 'ensure_files_exist_gets_called': True, 'modify_constants_gets_called': True, 'compare_file_count_gets_called': True, 'generate_python_package_called': True, 'clean_gets_called': True}
        expected_config_path = build.WEBPACK_PROD_CONFIG

        def mock_build_using_webpack(config_path: str) -> None:
            if False:
                return 10
            self.assertEqual(config_path, expected_config_path)
            check_function_calls['build_using_webpack_gets_called'] = True

        def mock_ensure_files_exist(unused_filepaths: List[str]) -> None:
            if False:
                i = 10
                return i + 15
            check_function_calls['ensure_files_exist_gets_called'] = True

        def mock_modify_constants(prod_env: bool, emulator_mode: bool, maintenance_mode: bool) -> None:
            if False:
                return 10
            check_function_calls['modify_constants_gets_called'] = True

        def mock_compare_file_count(unused_first_dir: str, unused_second_dir: str) -> None:
            if False:
                for i in range(10):
                    print('nop')
            check_function_calls['compare_file_count_gets_called'] = True

        def mock_generate_python_package() -> None:
            if False:
                print('Hello World!')
            check_function_calls['generate_python_package_called'] = True

        def mock_clean() -> None:
            if False:
                i = 10
                return i + 15
            check_function_calls['clean_gets_called'] = True
        ensure_files_exist_swap = self.swap(build, '_ensure_files_exist', mock_ensure_files_exist)
        build_using_webpack_swap = self.swap(build, 'build_using_webpack', mock_build_using_webpack)
        modify_constants_swap = self.swap(common, 'modify_constants', mock_modify_constants)
        compare_file_count_swap = self.swap(build, '_compare_file_count', mock_compare_file_count)
        generate_python_package_swap = self.swap(build, 'generate_python_package', mock_generate_python_package)
        clean_swap = self.swap(build, 'clean', mock_clean)
        with ensure_files_exist_swap, build_using_webpack_swap, clean_swap:
            with modify_constants_swap, compare_file_count_swap:
                with generate_python_package_swap:
                    build.main(args=['--prod_env'])
        self.assertEqual(check_function_calls, expected_check_function_calls)

    def test_build_with_prod_source_maps(self) -> None:
        if False:
            return 10
        check_function_calls = {'build_using_webpack_gets_called': False, 'ensure_files_exist_gets_called': False, 'modify_constants_gets_called': False, 'compare_file_count_gets_called': False, 'clean_gets_called': False}
        expected_check_function_calls = {'build_using_webpack_gets_called': True, 'ensure_files_exist_gets_called': True, 'modify_constants_gets_called': True, 'compare_file_count_gets_called': True, 'clean_gets_called': True}
        expected_config_path = build.WEBPACK_PROD_SOURCE_MAPS_CONFIG

        def mock_build_using_webpack(config_path: str) -> None:
            if False:
                while True:
                    i = 10
            self.assertEqual(config_path, expected_config_path)
            check_function_calls['build_using_webpack_gets_called'] = True

        def mock_ensure_files_exist(unused_filepaths: List[str]) -> None:
            if False:
                while True:
                    i = 10
            check_function_calls['ensure_files_exist_gets_called'] = True

        def mock_modify_constants(prod_env: bool, emulator_mode: bool, maintenance_mode: bool) -> None:
            if False:
                i = 10
                return i + 15
            check_function_calls['modify_constants_gets_called'] = True

        def mock_compare_file_count(unused_first_dir: str, unused_second_dir: str) -> None:
            if False:
                for i in range(10):
                    print('nop')
            check_function_calls['compare_file_count_gets_called'] = True

        def mock_clean() -> None:
            if False:
                print('Hello World!')
            check_function_calls['clean_gets_called'] = True
        ensure_files_exist_swap = self.swap(build, '_ensure_files_exist', mock_ensure_files_exist)
        build_using_webpack_swap = self.swap(build, 'build_using_webpack', mock_build_using_webpack)
        modify_constants_swap = self.swap(common, 'modify_constants', mock_modify_constants)
        compare_file_count_swap = self.swap(build, '_compare_file_count', mock_compare_file_count)
        clean_swap = self.swap(build, 'clean', mock_clean)
        install_python_dev_dependencies_swap = self.swap_with_checks(install_python_dev_dependencies, 'main', lambda _: None, expected_args=[(['--uninstall'],)])
        with ensure_files_exist_swap, build_using_webpack_swap:
            with modify_constants_swap, compare_file_count_swap:
                with clean_swap, install_python_dev_dependencies_swap:
                    build.main(args=['--prod_env', '--source_maps'])
        self.assertEqual(check_function_calls, expected_check_function_calls)

    def test_build_with_watcher(self) -> None:
        if False:
            print('Hello World!')
        check_function_calls = {'ensure_files_exist_gets_called': False, 'modify_constants_gets_called': False, 'clean_gets_called': False}
        expected_check_function_calls = {'ensure_files_exist_gets_called': True, 'modify_constants_gets_called': True, 'clean_gets_called': True}

        def mock_ensure_files_exist(unused_filepaths: List[str]) -> None:
            if False:
                while True:
                    i = 10
            check_function_calls['ensure_files_exist_gets_called'] = True

        def mock_modify_constants(prod_env: bool, emulator_mode: bool, maintenance_mode: bool) -> None:
            if False:
                return 10
            check_function_calls['modify_constants_gets_called'] = True

        def mock_clean() -> None:
            if False:
                return 10
            check_function_calls['clean_gets_called'] = True
        ensure_files_exist_swap = self.swap(build, '_ensure_files_exist', mock_ensure_files_exist)
        modify_constants_swap = self.swap(common, 'modify_constants', mock_modify_constants)
        clean_swap = self.swap(build, 'clean', mock_clean)
        with ensure_files_exist_swap, modify_constants_swap, clean_swap:
            build.main(args=[])
        self.assertEqual(check_function_calls, expected_check_function_calls)

    def test_cannot_maintenance_mode_in_dev_mode(self) -> None:
        if False:
            return 10
        assert_raises_regexp_context_manager = self.assertRaisesRegex(Exception, 'maintenance_mode should only be enabled in prod build.')
        with assert_raises_regexp_context_manager:
            build.main(args=['--maintenance_mode'])

    def test_cannot_minify_third_party_libs_in_dev_mode(self) -> None:
        if False:
            while True:
                i = 10
        check_function_calls = {'ensure_files_exist_gets_called': False, 'clean_gets_called': False}
        expected_check_function_calls = {'ensure_files_exist_gets_called': True, 'clean_gets_called': True}

        def mock_ensure_files_exist(unused_filepaths: List[str]) -> None:
            if False:
                print('Hello World!')
            check_function_calls['ensure_files_exist_gets_called'] = True

        def mock_clean() -> None:
            if False:
                for i in range(10):
                    print('nop')
            check_function_calls['clean_gets_called'] = True
        ensure_files_exist_swap = self.swap(build, '_ensure_files_exist', mock_ensure_files_exist)
        clean_swap = self.swap(build, 'clean', mock_clean)
        assert_raises_regexp_context_manager = self.assertRaisesRegex(Exception, 'minify_third_party_libs_only should not be set in non-prod env.')
        with ensure_files_exist_swap, assert_raises_regexp_context_manager:
            with clean_swap:
                build.main(args=['--minify_third_party_libs_only'])
        self.assertEqual(check_function_calls, expected_check_function_calls)

    def test_only_minify_third_party_libs_in_dev_mode(self) -> None:
        if False:
            return 10
        check_function_calls = {'ensure_files_exist_gets_called': False, 'ensure_modify_constants_gets_called': False, 'clean_gets_called': False}
        expected_check_function_calls = {'ensure_files_exist_gets_called': True, 'ensure_modify_constants_gets_called': False, 'clean_gets_called': True}

        def mock_ensure_files_exist(unused_filepaths: List[str]) -> None:
            if False:
                i = 10
                return i + 15
            check_function_calls['ensure_files_exist_gets_called'] = True

        def mock_modify_constants(unused_prod_env: bool, maintenance_mode: bool) -> None:
            if False:
                return 10
            check_function_calls['ensure_modify_constants_gets_called'] = True

        def mock_clean() -> None:
            if False:
                return 10
            check_function_calls['clean_gets_called'] = True
        ensure_files_exist_swap = self.swap(build, '_ensure_files_exist', mock_ensure_files_exist)
        modify_constants_swap = self.swap(common, 'modify_constants', mock_modify_constants)
        clean_swap = self.swap(build, 'clean', mock_clean)
        with ensure_files_exist_swap, modify_constants_swap, clean_swap:
            build.main(args=['--prod_env', '--minify_third_party_libs_only'])
        self.assertEqual(check_function_calls, expected_check_function_calls)

    def test_build_using_webpack_command(self) -> None:
        if False:
            i = 10
            return i + 15

        @contextlib.contextmanager
        def mock_managed_webpack_compiler(config_path: str, max_old_space_size: int) -> Iterator[scripts_test_utils.PopenStub]:
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(config_path, build.WEBPACK_PROD_CONFIG)
            self.assertEqual(max_old_space_size, 8192)
            yield scripts_test_utils.PopenStub()

        def mock_get_file_count(unused_path: str) -> int:
            if False:
                print('Hello World!')
            return 1
        webpack_compiler_swap = self.swap(servers, 'managed_webpack_compiler', mock_managed_webpack_compiler)
        get_file_count_swap = self.swap(build, 'get_file_count', mock_get_file_count)
        with webpack_compiler_swap, get_file_count_swap:
            build.build_using_webpack(build.WEBPACK_PROD_CONFIG)

    def test_build_using_webpack_command_with_incorrect_filecount_fails(self) -> None:
        if False:
            while True:
                i = 10

        @contextlib.contextmanager
        def mock_managed_webpack_compiler(config_path: str, max_old_space_size: int) -> Iterator[scripts_test_utils.PopenStub]:
            if False:
                while True:
                    i = 10
            self.assertEqual(config_path, build.WEBPACK_PROD_CONFIG)
            self.assertEqual(max_old_space_size, 8192)
            yield scripts_test_utils.PopenStub()

        def mock_get_file_count(unused_path: str) -> int:
            if False:
                i = 10
                return i + 15
            return 0
        webpack_compiler_swap = self.swap(servers, 'managed_webpack_compiler', mock_managed_webpack_compiler)
        get_file_count_swap = self.swap(build, 'get_file_count', mock_get_file_count)
        with webpack_compiler_swap, get_file_count_swap:
            with self.assertRaisesRegex(AssertionError, 'webpack_bundles should be non-empty.'):
                build.build_using_webpack(build.WEBPACK_PROD_CONFIG)

class E2EAndAcceptanceBuildTests(test_utils.GenericTestBase):
    """Test the end to end build methods."""

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        super().setUp()
        self.exit_stack = contextlib.ExitStack()

    def tearDown(self) -> None:
        if False:
            return 10
        try:
            self.exit_stack.close()
        finally:
            super().tearDown()

    def test_run_webpack_compilation_success(self) -> None:
        if False:
            print('Hello World!')
        old_os_path_isdir = os.path.isdir

        def mock_os_path_isdir(path: str) -> bool:
            if False:
                return 10
            if path == 'webpack_bundles':
                return True
            return old_os_path_isdir(path)
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_webpack_compiler', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(sys, 'exit', lambda _: None, called=False))
        self.exit_stack.enter_context(self.swap_with_checks(os.path, 'isdir', mock_os_path_isdir))
        build.run_webpack_compilation()

    def test_run_webpack_compilation_failed(self) -> None:
        if False:
            while True:
                i = 10
        old_os_path_isdir = os.path.isdir

        def mock_os_path_isdir(path: str) -> bool:
            if False:
                return 10
            if path == 'webpack_bundles':
                return False
            return old_os_path_isdir(path)
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_webpack_compiler', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(os.path, 'isdir', mock_os_path_isdir))
        self.exit_stack.enter_context(self.swap_with_checks(sys, 'exit', lambda _: None, expected_args=[(1,)]))
        build.run_webpack_compilation()

    def test_build_js_files_in_dev_mode_with_hash_file_exists(self) -> None:
        if False:
            print('Hello World!')
        old_os_path_isdir = os.path.isdir

        def mock_os_path_isdir(path: str) -> bool:
            if False:
                print('Hello World!')
            if path == 'webpack_bundles':
                return True
            return old_os_path_isdir(path)
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_webpack_compiler', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(build, 'main', lambda *_, **__: None, expected_kwargs=[{'args': []}]))
        self.exit_stack.enter_context(self.swap_with_checks(os.path, 'isdir', mock_os_path_isdir))
        self.exit_stack.enter_context(self.swap_with_checks(sys, 'exit', lambda _: None, called=False))
        build.build_js_files(True)

    def test_build_js_files_in_dev_mode_with_exception_raised(self) -> None:
        if False:
            return 10
        return_code = 2
        self.exit_stack.enter_context(self.swap_to_always_raise(servers, 'managed_webpack_compiler', error=subprocess.CalledProcessError(return_code, [])))
        self.exit_stack.enter_context(self.swap_with_checks(build, 'main', lambda *_, **__: None, expected_kwargs=[{'args': []}]))
        self.exit_stack.enter_context(self.swap_with_checks(sys, 'exit', lambda _: None, expected_args=[(return_code,), (return_code,), (return_code,), (return_code,), (return_code,), (1,)]))
        build.build_js_files(True)

    def test_build_js_files_in_prod_mode(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.exit_stack.enter_context(self.swap_with_checks(common, 'run_cmd', lambda *_: None, called=False))
        self.exit_stack.enter_context(self.swap_with_checks(build, 'main', lambda *_, **__: None, expected_kwargs=[{'args': ['--prod_env']}]))
        build.build_js_files(False)

    def test_build_js_files_in_prod_mode_with_source_maps(self) -> None:
        if False:
            print('Hello World!')
        self.exit_stack.enter_context(self.swap_with_checks(common, 'run_cmd', lambda *_: None, called=False))
        self.exit_stack.enter_context(self.swap_with_checks(build, 'main', lambda *_, **__: None, expected_kwargs=[{'args': ['--prod_env', '--source_maps']}]))
        build.build_js_files(False, source_maps=True)

    def test_webpack_compilation_in_dev_mode_with_source_maps(self) -> None:
        if False:
            print('Hello World!')
        self.exit_stack.enter_context(self.swap_with_checks(common, 'run_cmd', lambda *_: None, called=False))
        self.exit_stack.enter_context(self.swap_with_checks(build, 'main', lambda *_, **__: None, expected_kwargs=[{'args': []}]))
        self.exit_stack.enter_context(self.swap_with_checks(build, 'run_webpack_compilation', lambda **_: None, expected_kwargs=[{'source_maps': True}]))
        build.build_js_files(True, source_maps=True)