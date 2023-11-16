import typing
import sys
import os
from unittest.mock import MagicMock, patch
import pytest
from hamcrest import *
from tests.engine_helpers import *
import autokey.model.folder as akfolder
from autokey.configmanager.configmanager import ConfigManager
from autokey.configmanager.configmanager_constants import CONFIG_DEFAULT_FOLDER
import autokey.configmanager.predefined_user_files
from autokey.service import PhraseRunner
import autokey.service
from autokey.scripting import Engine

def get_autokey_dir():
    if False:
        for i in range(10):
            print('nop')
    return os.path.dirname(os.path.realpath(sys.argv[0]))

def test_get_item_with_hotkey(create_engine):
    if False:
        print('Hello World!')
    (engine, folder) = create_engine
    modifiers = ['<ctrl>', '<alt>', '<super>', '<shift>']
    key = 'a'
    hotkey = (modifiers, key)
    testHK = create_test_hotkey(engine, folder, hotkey)
    resultHK = engine.configManager.get_item_with_hotkey(modifiers, key, None)
    assert_that(resultHK, is_(equal_to(testHK)))

def test_item_has_same_hotkey(create_engine):
    if False:
        while True:
            i = 10
    (engine, folder) = create_engine
    modifiers = ['<ctrl>', '<alt>', '<super>', '<shift>']
    key = 'a'
    hotkey = (modifiers, key)
    testHK = create_test_hotkey(engine, folder, hotkey)
    assert ConfigManager.item_has_same_hotkey(testHK, modifiers, key, None)

def test_get_all_folders(create_engine):
    if False:
        for i in range(10):
            print('nop')
    (engine, folder) = create_engine
    cm = engine.configManager
    first_child = akfolder.Folder('first child')
    first_grandchild = akfolder.Folder('first grandchild')
    second_grandchild = akfolder.Folder('second grandchild')
    first_child.add_folder(first_grandchild)
    first_child.add_folder(second_grandchild)
    cm.folders.append(first_child)
    expected = [folder, first_child, first_grandchild, second_grandchild]
    result = cm.get_all_folders()
    assert_that(result, equal_to(expected))

def test_create_predefined_user_files_my_phrases_folder(create_engine):
    if False:
        i = 10
        return i + 15
    (engine, folder) = create_engine
    os.makedirs(CONFIG_DEFAULT_FOLDER, exist_ok=True)
    phrases_folder = autokey.configmanager.predefined_user_files.create_my_phrases_folder()
    scripts_folder = autokey.configmanager.predefined_user_files.create_sample_scripts_folder()