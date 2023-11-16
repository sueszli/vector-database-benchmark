import unittest
from unittest.mock import MagicMock, patch
from ....api import query
from .. import subclass_generator

class SubclassGeneratorTest(unittest.TestCase):

    @patch.object(query, 'get_cached_class_hierarchy')
    def test_get_all_subclasses_from_pyre(self, get_cached_class_hierarchy_mock: MagicMock) -> None:
        if False:
            i = 10
            return i + 15
        class_hierarchy = query.ClassHierarchy({'GrandChild': ['WantedChild1'], 'WantedChild1': ['WantedParent'], 'WantedChild2': ['WantedParent'], 'UnwantedChild1': ['UnwantedParent'], 'UnwantedParent': ['object'], 'WantedParent': ['object'], 'object': []})
        get_cached_class_hierarchy_mock.return_value = class_hierarchy
        self.assertEqual(subclass_generator.get_all_subclasses_from_pyre(['WantedParent'], None), {'WantedParent': ['WantedChild1', 'WantedChild2']})
        self.assertEqual(subclass_generator.get_all_subclasses_from_pyre(['WantedParent'], None, True), {'WantedParent': ['GrandChild', 'WantedChild1', 'WantedChild2']})

    @patch.object(query, 'defines')
    @patch.object(subclass_generator, 'get_all_subclasses_from_pyre')
    def test_get_all_subclass_defines_from_pyre(self, get_all_subclasses_from_pyre_mock: MagicMock, defines_mock: MagicMock) -> None:
        if False:
            while True:
                i = 10
        subclasses_dict = {'WantedParent': ['WantedChild1', 'WantedChild2']}
        get_all_subclasses_from_pyre_mock.return_value = subclasses_dict
        subclass_defines = [query.Define(name='WantedChild1', parameters=[], return_annotation='None'), query.Define(name='WantedChild2', parameters=[], return_annotation='None')]
        defines_mock.return_value = subclass_defines
        self.assertEqual(subclass_generator.get_all_subclass_defines_from_pyre(['WantedParent'], None), {'WantedParent': subclass_defines})