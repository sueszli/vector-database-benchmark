import typing
import string
import random
from unittest.mock import MagicMock, patch
import pytest
from hamcrest import *
import autokey.model.folder
from autokey.configmanager.configmanager import ConfigManager
from autokey.service import PhraseRunner
import autokey.service
from autokey.scripting import Engine
folder_param = 'create_new_folder'

@pytest.fixture
def create_engine() -> typing.Tuple[Engine, autokey.model.folder.Folder]:
    if False:
        for i in range(10):
            print('nop')
    test_folder = autokey.model.folder.Folder('Test folder')
    test_folder.persist = MagicMock()
    with patch('autokey.model.phrase.Phrase.persist'), patch('autokey.model.folder.Folder.persist'), patch('autokey.configmanager.configmanager.ConfigManager.load_global_config', new=lambda self: self.folders.append(test_folder)):
        engine = Engine(ConfigManager(MagicMock()), MagicMock(spec=PhraseRunner))
        engine.configManager.config_altered(False)
    return (engine, test_folder)

def create_random_string(length=10):
    if False:
        print('Hello World!')
    return ''.join([random.choice(string.ascii_letters + string.digits) for n in range(length)])

def replace_folder_param_in_args(folder, args):
    if False:
        return 10
    if isinstance(args, str):
        return args
    args = [folder if x == folder_param else x for x in args]
    return args

def get_item_with_hotkey(engine, hotkey):
    if False:
        i = 10
        return i + 15
    modifiers = sorted(hotkey[0])
    item = engine.configManager.get_item_with_hotkey(modifiers, hotkey[1])
    return item

def assert_both_phrases_with_hotkey_exist(engine, p1, p2, hotkey):
    if False:
        return 10
    phrases = [p1, p2]
    for _ in phrases:
        phrase = get_item_with_hotkey(engine, hotkey)
        assert phrase in phrases
        phrase.unset_hotkey()
        phrases.remove(phrase)

def create_test_hotkey(engine, folder, hotkey, replaceExisting=False, windowFilter=None):
    if False:
        i = 10
        return i + 15
    with patch('autokey.model.phrase.Phrase.persist'):
        return engine.create_phrase(folder, create_random_string(), 'ABC', hotkey=hotkey, replace_existing_hotkey=replaceExisting, window_filter=windowFilter)