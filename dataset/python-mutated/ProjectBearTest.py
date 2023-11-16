from concurrent.futures import ThreadPoolExecutor
from unittest.mock import ANY, patch
from coalib.core.ProjectBear import ProjectBear
from coalib.settings.Section import Section
from tests.core.CoreTestBase import CoreTestBase

class TestProjectBear(ProjectBear):

    def analyze(self, files):
        if False:
            while True:
                i = 10
        yield '\n'.join((filename + ':' + str(files[filename]) for filename in sorted(files)))

class TestProjectBearWithParameters(ProjectBear):

    def analyze(self, files, prefix: str='---'):
        if False:
            while True:
                i = 10
        yield '\n'.join((prefix + filename + ':' + str(files[filename]) for filename in sorted(files)))

class ProjectBearTest(CoreTestBase):

    def assertResultsEqual(self, bear_type, expected, section=None, file_dict=None, cache=None):
        if False:
            i = 10
            return i + 15
        '\n        Asserts whether the expected results do match the output of the bear.\n\n        Asserts for the results out-of-order.\n\n        :param bear_type:\n            The bear class to check.\n        :param expected:\n            A sequence of expected results.\n        :param section:\n            A section for the bear to use. By default uses a new section with\n            name ``test-section``.\n        :param file_dict:\n            A file-dictionary for the bear to use. By default uses an empty\n            dictionary.\n        :param cache:\n            A cache the bear can use to speed up runs. If ``None``, no cache\n            will be used.\n        '
        if section is None:
            section = Section('test-section')
        if file_dict is None:
            file_dict = {}
        uut = bear_type(section, file_dict)
        results = self.execute_run({uut}, cache)
        self.assertEqual(sorted(expected), sorted(results))

    def test_bear_without_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertResultsEqual(TestProjectBear, file_dict={}, expected=[''])
        self.assertResultsEqual(TestProjectBear, file_dict={'fileX': []}, expected=['fileX:[]'])
        self.assertResultsEqual(TestProjectBear, file_dict={'fileX': [], 'fileY': ['hello']}, expected=["fileX:[]\nfileY:['hello']"])
        self.assertResultsEqual(TestProjectBear, file_dict={'fileX': [], 'fileY': ['hello'], 'fileZ': ['x\n', 'y']}, expected=["fileX:[]\nfileY:['hello']\nfileZ:['x\\n', 'y']"])

    def test_bear_with_parameters_but_keep_defaults(self):
        if False:
            while True:
                i = 10
        self.assertResultsEqual(TestProjectBearWithParameters, file_dict={}, expected=[''])
        self.assertResultsEqual(TestProjectBearWithParameters, file_dict={'fileX': []}, expected=['---fileX:[]'])
        self.assertResultsEqual(TestProjectBearWithParameters, file_dict={'fileX': [], 'fileY': ['hello']}, expected=["---fileX:[]\n---fileY:['hello']"])
        self.assertResultsEqual(TestProjectBearWithParameters, file_dict={'fileX': [], 'fileY': ['hello'], 'fileZ': ['x\n', 'y']}, expected=["---fileX:[]\n---fileY:['hello']\n---fileZ:['x\\n', 'y']"])

    def test_bear_with_parameters(self):
        if False:
            while True:
                i = 10
        section = Section('test-section')
        section['prefix'] = '___'
        self.assertResultsEqual(TestProjectBearWithParameters, section=section, file_dict={}, expected=[''])
        self.assertResultsEqual(TestProjectBearWithParameters, section=section, file_dict={'fileX': []}, expected=['___fileX:[]'])
        self.assertResultsEqual(TestProjectBearWithParameters, section=section, file_dict={'fileX': [], 'fileY': ['hello']}, expected=["___fileX:[]\n___fileY:['hello']"])
        self.assertResultsEqual(TestProjectBearWithParameters, section=section, file_dict={'fileX': [], 'fileY': ['hello'], 'fileZ': ['x\ny']}, expected=["___fileX:[]\n___fileY:['hello']\n___fileZ:['x\\ny']"])

class ProjectBearOnThreadPoolExecutorTest(ProjectBearTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.executor = (ThreadPoolExecutor, tuple(), dict(max_workers=8))

    def test_cache(self):
        if False:
            print('Hello World!')
        section = Section('test-section')
        filedict1 = {'file.txt': []}
        filedict2 = {'file.txt': ['first-line\n']}
        expected_results1 = ['file.txt:[]']
        expected_results2 = ["file.txt:['first-line\\n']"]
        cache = {}
        with patch.object(TestProjectBear, 'analyze', autospec=True, side_effect=TestProjectBear.analyze) as mock:
            self.assertResultsEqual(TestProjectBear, section=section, file_dict=filedict1, cache=cache, expected=expected_results1)
            mock.assert_called_once_with(ANY, filedict1)
            self.assertEqual(len(cache), 1)
            self.assertEqual(len(next(iter(cache.values()))), 1)
            mock.reset_mock()
            self.assertResultsEqual(TestProjectBear, section=section, file_dict=filedict1, cache=cache, expected=expected_results1)
            self.assertFalse(mock.called)
            self.assertEqual(len(cache), 1)
            self.assertEqual(len(next(iter(cache.values()))), 1)
            self.assertResultsEqual(TestProjectBear, section=section, file_dict=filedict2, cache=cache, expected=expected_results2)
            mock.assert_called_once_with(ANY, filedict2)
            self.assertEqual(len(cache), 1)
            self.assertEqual(len(next(iter(cache.values()))), 2)