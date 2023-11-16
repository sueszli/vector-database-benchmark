import textwrap
from pathlib import Path
from typing import Callable, List, Mapping, TypeVar
import testslide
from ..patch_specs import Action, action_from_json, AddAction, AddPosition, DeleteAction, DeleteImportAction, FilePatch, Patch, QualifiedName, ReadPatchException, ReplaceAction

class PatchTest(testslide.TestCase):

    def test_qualified_name(self) -> None:
        if False:
            for i in range(10):
                print('nop')

        def assert_name_preserved(name: str) -> None:
            if False:
                i = 10
                return i + 15
            self.assertEqual(QualifiedName.from_string(name).to_string(), name)
        assert_name_preserved('')
        assert_name_preserved('foo')
        assert_name_preserved('foo.bar')
        assert_name_preserved('foo.bar.baz')
        self.assertTrue(QualifiedName.from_string('').is_empty())
        self.assertFalse(QualifiedName.from_string('foo').is_empty())
T = TypeVar('T')
U = TypeVar('U')

class PatchReaderTest(testslide.TestCase):

    def _assert_parsed(self, input: T, parser: Callable[[T], U], expected: U) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(parser(input), expected)

    def _assert_not_parsed(self, input: T, parser: Callable[[T], U]) -> None:
        if False:
            while True:
                i = 10
        with self.assertRaises(ReadPatchException):
            parser(input)

    def assert_parsed_parent(self, input: object, expected: QualifiedName) -> None:
        if False:
            print('Hello World!')
        self._assert_parsed(input, QualifiedName.from_json, expected)

    def assert_not_parsed_parent(self, input: object) -> None:
        if False:
            while True:
                i = 10
        self._assert_not_parsed(input, QualifiedName.from_json)

    def test_read_parent(self) -> None:
        if False:
            return 10
        self.assert_parsed_parent('foo', expected=QualifiedName(['foo']))
        self.assert_parsed_parent('foo.bar', expected=QualifiedName(['foo', 'bar']))
        self.assert_not_parsed_parent(42)
        self.assert_not_parsed_parent([])
        self.assert_not_parsed_parent({})
        self.assert_not_parsed_parent(False)

    def assert_parsed_add_position(self, input: object, expected: AddPosition) -> None:
        if False:
            i = 10
            return i + 15
        self._assert_parsed(input, AddPosition.from_json, expected)

    def assert_not_parsed_add_position(self, input: object) -> None:
        if False:
            print('Hello World!')
        self._assert_not_parsed(input, AddPosition.from_json)

    def test_read_add_position(self) -> None:
        if False:
            while True:
                i = 10
        self.assert_parsed_add_position('top', expected=AddPosition.TOP_OF_SCOPE)
        self.assert_parsed_add_position('bottom', expected=AddPosition.BOTTOM_OF_SCOPE)
        self.assert_not_parsed_add_position(42)
        self.assert_not_parsed_add_position([])
        self.assert_not_parsed_add_position({})
        self.assert_not_parsed_add_position(False)

    def assert_parsed_action(self, input: Mapping[str, object], expected: Action) -> None:
        if False:
            i = 10
            return i + 15
        self._assert_parsed(input, action_from_json, expected)

    def assert_not_parsed_action(self, input: Mapping[str, object]) -> None:
        if False:
            i = 10
            return i + 15
        self._assert_not_parsed(input, action_from_json)

    def test_read_add_action(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assert_parsed_action({'action': 'add', 'content': 'derp'}, AddAction(content='derp'))
        self.assert_parsed_action({'action': 'add', 'content': 'derp', 'position': 'top'}, AddAction(content='derp', position=AddPosition.TOP_OF_SCOPE))
        self.assert_not_parsed_action({'doom': 'eternal'})
        self.assert_not_parsed_action({'action': 'doom'})
        self.assert_not_parsed_action({'action': 'add', 'doom': 'eternal'})

    def test_read_delete_action(self) -> None:
        if False:
            while True:
                i = 10
        self.assert_parsed_action({'action': 'delete', 'name': 'derp'}, DeleteAction(name='derp'))
        self.assert_not_parsed_action({'doom': 'eternal'})
        self.assert_not_parsed_action({'action': 'doom'})
        self.assert_not_parsed_action({'action': 'delete', 'doom': 'eternal'})

    def test_read_delete_import_action(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assert_parsed_action({'action': 'delete_import', 'name': 'derp'}, DeleteImportAction(name='derp'))
        self.assert_not_parsed_action({'doom': 'eternal'})
        self.assert_not_parsed_action({'action': 'doom'})
        self.assert_not_parsed_action({'action': 'delete_import', 'doom': 'eternal'})

    def test_read_replace_action(self) -> None:
        if False:
            print('Hello World!')
        self.assert_parsed_action({'action': 'replace', 'name': 'doom', 'content': 'BFG'}, ReplaceAction(name='doom', content='BFG'))
        self.assert_not_parsed_action({'doom': 'eternal'})
        self.assert_not_parsed_action({'action': 'doom'})
        self.assert_not_parsed_action({'action': 'replace', 'doom': 'eternal'})
        self.assert_not_parsed_action({'action': 'replace', 'name': 'doom'})
        self.assert_not_parsed_action({'action': 'replace', 'content': 'BFG'})

    def assert_parsed_patch(self, input: object, expected: Patch) -> None:
        if False:
            print('Hello World!')
        self._assert_parsed(input, Patch.from_json, expected)

    def assert_not_parsed_patch(self, input: object) -> None:
        if False:
            print('Hello World!')
        self._assert_not_parsed(input, Patch.from_json)

    def test_read_patch(self) -> None:
        if False:
            return 10
        self.assert_parsed_patch({'parent': 'foo.bar', 'action': 'delete', 'name': 'doom'}, Patch(parent=QualifiedName(['foo', 'bar']), action=DeleteAction(name='doom')))
        self.assert_parsed_patch({'action': 'delete', 'name': 'doom'}, Patch(action=DeleteAction(name='doom'), parent=QualifiedName.from_string('')))
        self.assert_not_parsed_patch(42)
        self.assert_not_parsed_patch([])
        self.assert_not_parsed_patch(False)
        self.assert_not_parsed_patch({})
        self.assert_not_parsed_patch({'parent': 'foo.bar'})

    def assert_parsed_file_patches(self, input: Mapping[str, object], expected: List[FilePatch]) -> None:
        if False:
            while True:
                i = 10
        self._assert_parsed(input, FilePatch.from_json, expected)

    def assert_not_parsed_file_patches(self, input: Mapping[str, object]) -> None:
        if False:
            return 10
        self._assert_not_parsed(input, FilePatch.from_json)

    def test_read_file_patches(self) -> None:
        if False:
            while True:
                i = 10
        self.assert_parsed_file_patches({}, [])
        self.assert_parsed_file_patches({'foo.pyi': []}, [FilePatch(path=Path('foo.pyi'), patches=[])])
        self.assert_parsed_file_patches({'foo.pyi': [{'action': 'add', 'content': 'doom slayer'}, {'action': 'delete', 'name': 'cyberdemon', 'parent': 'Hell'}]}, [FilePatch(path=Path('foo.pyi'), patches=[Patch(action=AddAction(content='doom slayer'), parent=QualifiedName.from_string('')), Patch(parent=QualifiedName(['Hell']), action=DeleteAction(name='cyberdemon'))])])
        self.assert_parsed_file_patches({'foo.pyi': [{'action': 'add', 'content': 'doom slayer'}], 'bar.pyi': [{'action': 'delete', 'name': 'cyberdemon', 'parent': 'Hell'}]}, [FilePatch(path=Path('foo.pyi'), patches=[Patch(action=AddAction(content='doom slayer'), parent=QualifiedName.from_string(''))]), FilePatch(path=Path('bar.pyi'), patches=[Patch(parent=QualifiedName(['Hell']), action=DeleteAction(name='cyberdemon'))])])
        self.assert_not_parsed_file_patches({'foo.pyi': 42})
        self.assert_not_parsed_file_patches({'foo.pyi': [{'doom': 'eternal'}]})

    def assert_parsed_toml(self, input: str, expected: List[FilePatch]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._assert_parsed(textwrap.dedent(input), FilePatch.from_toml_string, expected)

    def assert_not_parsed_toml(self, input: str) -> None:
        if False:
            i = 10
            return i + 15
        self._assert_not_parsed(textwrap.dedent(input), FilePatch.from_toml_string)

    def test_read_toml_basic(self) -> None:
        if False:
            return 10
        self.assert_parsed_toml('', [])
        self.assert_parsed_toml('\n            [["foo.pyi"]]\n            action = "add"\n            content = "doom slayer"\n\n            [["foo.pyi"]]\n            action = "delete"\n            name = "cyberdemon"\n            parent = "Hell"\n            ', [FilePatch(path=Path('foo.pyi'), patches=[Patch(action=AddAction(content='doom slayer'), parent=QualifiedName.from_string('')), Patch(parent=QualifiedName(['Hell']), action=DeleteAction(name='cyberdemon'))])])
        self.assert_parsed_toml('\n            [["foo.pyi"]]\n            action = "add"\n            content = "doom slayer"\n\n            [["bar.pyi"]]\n            action = "delete"\n            name = "cyberdemon"\n            parent = "Hell"\n            ', [FilePatch(path=Path('foo.pyi'), patches=[Patch(action=AddAction(content='doom slayer'), parent=QualifiedName.from_string(''))]), FilePatch(path=Path('bar.pyi'), patches=[Patch(parent=QualifiedName(['Hell']), action=DeleteAction(name='cyberdemon'))])])
        self.assert_not_parsed_toml('derp')
        self.assert_not_parsed_toml('42')
        self.assert_not_parsed_toml('[["foo.pyi"]]')
        self.assert_not_parsed_toml('\n            [["foo.pyi"]]\n            action = "add"\n            ')
        self.assert_not_parsed_toml('\n            [["foo.pyi"]]\n            action = "add"\n            content = "test"\n            parent = 42\n            ')