from typing import Dict
import testslide
from ..exceptions import InvalidConfiguration
from ..unwatched import UnwatchedDependency, UnwatchedFiles

class UnwatchedDependencyTest(testslide.TestCase):

    def test_unwatched_files_from_json(self) -> None:
        if False:
            for i in range(10):
                print('nop')

        def assert_parsed(input: Dict[str, object], expected: UnwatchedFiles) -> None:
            if False:
                i = 10
                return i + 15
            self.assertEqual(UnwatchedFiles.from_json(input), expected)

        def assert_not_parsed(input: Dict[str, object]) -> None:
            if False:
                while True:
                    i = 10
            with self.assertRaises(InvalidConfiguration):
                UnwatchedFiles.from_json(input)
        assert_not_parsed({})
        assert_not_parsed({'derp': 42})
        assert_not_parsed({'root': 42})
        assert_not_parsed({'root': 'foo'})
        assert_not_parsed({'checksum_path': []})
        assert_not_parsed({'checksum_path': 'bar'})
        assert_not_parsed({'root': 'foo', 'checksum_path': True})
        assert_not_parsed({'root': {}, 'checksum_path': 'bar'})
        assert_parsed({'root': 'foo', 'checksum_path': 'bar'}, UnwatchedFiles(root='foo', checksum_path='bar'))

    def test_unwatched_dependency_from_json(self) -> None:
        if False:
            while True:
                i = 10

        def assert_parsed(input: Dict[str, object], expected: UnwatchedDependency) -> None:
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(UnwatchedDependency.from_json(input), expected)

        def assert_not_parsed(input: Dict[str, object]) -> None:
            if False:
                print('Hello World!')
            with self.assertRaises(InvalidConfiguration):
                UnwatchedDependency.from_json(input)
        assert_not_parsed({})
        assert_not_parsed({'derp': 42})
        assert_not_parsed({'change_indicator': 42})
        assert_not_parsed({'change_indicator': 'foo'})
        assert_not_parsed({'change_indicator': 'foo', 'files': 42})
        assert_not_parsed({'change_indicator': 'foo', 'files': {}})
        assert_not_parsed({'change_indicator': 'foo', 'files': {'root': 'foo'}})
        assert_parsed({'change_indicator': 'foo', 'files': {'root': 'bar', 'checksum_path': 'baz'}}, UnwatchedDependency(change_indicator='foo', files=UnwatchedFiles(root='bar', checksum_path='baz')))