import unittest
from pathlib import Path
from typing import List
from ..download_typeshed import PatchedTypeshed, FileEntry, _find_entry

class EntryPathToPatchPathTest(unittest.TestCase):

    def assert_path_is(self, path: str, expected: Path) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(PatchedTypeshed._entry_path_to_patch_path(path), expected)

    def test_path_is(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assert_path_is('typeshed-master/stubs/foo.pyi', Path('stubs/foo.patch'))
        self.assert_path_is('typeshed-master', Path(''))

class FindEntryTest(unittest.TestCase):

    def assert_found_as(self, path: str, entries: List[FileEntry], expected: FileEntry) -> None:
        if False:
            return 10
        self.assertEqual(_find_entry(Path(path), entries), expected)

    def assert_not_found(self, path: str, entries: List[FileEntry]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(_find_entry(Path(path), entries), None)
    decimal_entry = FileEntry('typeshed-master/stdlib/decimal.pyi', bytes([1, 5, 0]))
    chunk_entry = FileEntry('typeshed-master/stdlib/chunk.pyi', bytes([1, 0, 0]))
    core_entry = FileEntry('typeshed-master/stubs/click/click/core.pyi', bytes([5, 0]))
    no_data_entry = FileEntry('typeshed-master/stdlib/fake/fake.pyi', None)
    example_entries: List[FileEntry] = [decimal_entry, chunk_entry, core_entry, no_data_entry]

    def test_found_as(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assert_found_as(path='stdlib/decimal.patch', entries=self.example_entries, expected=self.decimal_entry)
        self.assert_found_as(path='stdlib/decimal.pyi', entries=self.example_entries, expected=self.decimal_entry)
        self.assert_found_as(path='stdlib/decimal', entries=self.example_entries, expected=self.decimal_entry)
        self.assert_found_as(path='stubs/click/click/core.patch', entries=self.example_entries, expected=self.core_entry)

    def test_not_found(self) -> None:
        if False:
            while True:
                i = 10
        self.assert_not_found(path='stdlib/fake.patch', entries=self.example_entries)