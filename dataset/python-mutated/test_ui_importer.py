"""Tests the TerminalImportSession. The tests are the same as in the

test_importer module. But here the test importer inherits from
``TerminalImportSession``. So we test this class, too.
"""
import unittest
from test import test_importer
from test._common import DummyIO
from beets import config, importer
from beets.ui.commands import TerminalImportSession

class TerminalImportSessionFixture(TerminalImportSession):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self.io = kwargs.pop('io')
        super().__init__(*args, **kwargs)
        self._choices = []
    default_choice = importer.action.APPLY

    def add_choice(self, choice):
        if False:
            for i in range(10):
                print('nop')
        self._choices.append(choice)

    def clear_choices(self):
        if False:
            print('Hello World!')
        self._choices = []

    def choose_match(self, task):
        if False:
            print('Hello World!')
        self._add_choice_input()
        return super().choose_match(task)

    def choose_item(self, task):
        if False:
            while True:
                i = 10
        self._add_choice_input()
        return super().choose_item(task)

    def _add_choice_input(self):
        if False:
            return 10
        try:
            choice = self._choices.pop(0)
        except IndexError:
            choice = self.default_choice
        if choice == importer.action.APPLY:
            self.io.addinput('A')
        elif choice == importer.action.ASIS:
            self.io.addinput('U')
        elif choice == importer.action.ALBUMS:
            self.io.addinput('G')
        elif choice == importer.action.TRACKS:
            self.io.addinput('T')
        elif choice == importer.action.SKIP:
            self.io.addinput('S')
        elif isinstance(choice, int):
            self.io.addinput('M')
            self.io.addinput(str(choice))
            self._add_choice_input()
        else:
            raise Exception('Unknown choice %s' % choice)

class TerminalImportSessionSetup:
    """Overwrites test_importer.ImportHelper to provide a terminal importer"""

    def _setup_import_session(self, import_dir=None, delete=False, threaded=False, copy=True, singletons=False, move=False, autotag=True):
        if False:
            while True:
                i = 10
        config['import']['copy'] = copy
        config['import']['delete'] = delete
        config['import']['timid'] = True
        config['threaded'] = False
        config['import']['singletons'] = singletons
        config['import']['move'] = move
        config['import']['autotag'] = autotag
        config['import']['resume'] = False
        if not hasattr(self, 'io'):
            self.io = DummyIO()
        self.io.install()
        self.importer = TerminalImportSessionFixture(self.lib, loghandler=None, query=None, io=self.io, paths=[import_dir or self.import_dir])

class NonAutotaggedImportTest(TerminalImportSessionSetup, test_importer.NonAutotaggedImportTest):
    pass

class ImportTest(TerminalImportSessionSetup, test_importer.ImportTest):
    pass

class ImportSingletonTest(TerminalImportSessionSetup, test_importer.ImportSingletonTest):
    pass

class ImportTracksTest(TerminalImportSessionSetup, test_importer.ImportTracksTest):
    pass

class ImportCompilationTest(TerminalImportSessionSetup, test_importer.ImportCompilationTest):
    pass

class ImportExistingTest(TerminalImportSessionSetup, test_importer.ImportExistingTest):
    pass

class ChooseCandidateTest(TerminalImportSessionSetup, test_importer.ChooseCandidateTest):
    pass

class GroupAlbumsImportTest(TerminalImportSessionSetup, test_importer.GroupAlbumsImportTest):
    pass

class GlobalGroupAlbumsImportTest(TerminalImportSessionSetup, test_importer.GlobalGroupAlbumsImportTest):
    pass

def suite():
    if False:
        for i in range(10):
            print('nop')
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main(defaultTest='suite')