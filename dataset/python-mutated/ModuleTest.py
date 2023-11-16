from types import ModuleType
import coalib.bearlib.aspects
from coalib.bearlib.aspects.exceptions import AspectNotFoundError, MultipleAspectFoundError
import pytest
import unittest

class aspectsModuleTest(unittest.TestCase):

    def test_module(self):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(coalib.bearlib.aspects, ModuleType)
        assert type(coalib.bearlib.aspects) is not ModuleType
        assert type(coalib.bearlib.aspects) is coalib.bearlib.aspects.aspectsModule

    def test__getitem__(self):
        if False:
            print('Hello World!')
        dict_spelling = coalib.bearlib.aspects.Root.Spelling.DictionarySpelling
        for aspectname in ['DictionarySpelling', 'spelling.DictionarySpelling', 'root.SPELLING.DictionarySpelling']:
            assert coalib.bearlib.aspects[aspectname] is dict_spelling
        for aspectname in ['Spelling', 'SPELLING', 'ROOT.spelling']:
            assert coalib.bearlib.aspects[aspectname] is coalib.bearlib.aspects.Root.Spelling
        for aspectname in ['Root', 'root', 'ROOT']:
            assert coalib.bearlib.aspects[aspectname] is coalib.bearlib.aspects.Root

    def test__getitem__no_match(self):
        if False:
            print('Hello World!')
        for aspectname in ['noaspect', 'NOASPECT', 'Root.DictionarySpelling']:
            with pytest.raises(AspectNotFoundError) as exc:
                coalib.bearlib.aspects[aspectname]
            exc.match("^No aspect named '%s'$" % aspectname)

    def test__getitem__multi_match(self):
        if False:
            for i in range(10):
                print('nop')
        for aspectname in ['Length', 'length', 'LENGTH']:
            with pytest.raises(MultipleAspectFoundError) as exc:
                coalib.bearlib.aspects[aspectname]
            exc.match("^Multiple aspects named '%s'. " % aspectname + "Choose from \\[<aspectclass 'Root.Formatting.Length'>, <aspectclass 'Root.Metadata.CommitMessage.Body.Length'>, <aspectclass 'Root.Metadata.CommitMessage.Shortlog.Length'>\\]$")

    def test_get(self):
        if False:
            for i in range(10):
                print('nop')
        for aspectname in ['clone', 'redundancy.clone', 'root.redundancy.clone']:
            self.assertIs(coalib.bearlib.aspects.get(aspectname), coalib.bearlib.aspects.Root.Redundancy.Clone)
        for aspectname in ['Spelling', 'SPELLING', 'ROOT.spelling']:
            self.assertIs(coalib.bearlib.aspects.get(aspectname), coalib.bearlib.aspects.Root.Spelling)
        for aspectname in ['Root', 'root', 'ROOT']:
            self.assertIs(coalib.bearlib.aspects.get(aspectname), coalib.bearlib.aspects.Root)

    def test_get_no_match(self):
        if False:
            for i in range(10):
                print('nop')
        for aspectname in ['noaspect', 'NOASPECT', 'Root.aspectsYEAH']:
            self.assertIsNone(coalib.bearlib.aspects.get(aspectname))

    def test_get_multi_match(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(MultipleAspectFoundError, "^Multiple aspects named 'length'. Choose from \\[<aspectclass 'Root.Formatting.Length'>, <aspectclass 'Root.Metadata.CommitMessage.Body.Length'>, <aspectclass 'Root.Metadata.CommitMessage.Shortlog.Length'>\\]$"):
            coalib.bearlib.aspects.get('length')