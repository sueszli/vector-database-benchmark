from __future__ import annotations
import unittest
from ansible.playbook.taggable import Taggable
from units.mock.loader import DictDataLoader

class TaggableTestObj(Taggable):

    def __init__(self):
        if False:
            while True:
                i = 10
        self._loader = DictDataLoader({})
        self.tags = []
        self._parent = None

class TestTaggable(unittest.TestCase):

    def assert_evaluate_equal(self, test_value, tags, only_tags, skip_tags):
        if False:
            print('Hello World!')
        taggable_obj = TaggableTestObj()
        taggable_obj.tags = tags
        evaluate = taggable_obj.evaluate_tags(only_tags, skip_tags, {})
        self.assertEqual(test_value, evaluate)

    def test_evaluate_tags_tag_in_only_tags(self):
        if False:
            return 10
        self.assert_evaluate_equal(True, ['tag1', 'tag2'], ['tag1'], [])

    def test_evaluate_tags_tag_in_skip_tags(self):
        if False:
            while True:
                i = 10
        self.assert_evaluate_equal(False, ['tag1', 'tag2'], [], ['tag1'])

    def test_evaluate_tags_special_always_in_object_tags(self):
        if False:
            i = 10
            return i + 15
        self.assert_evaluate_equal(True, ['tag', 'always'], ['random'], [])

    def test_evaluate_tags_tag_in_skip_tags_special_always_in_object_tags(self):
        if False:
            i = 10
            return i + 15
        self.assert_evaluate_equal(False, ['tag', 'always'], ['random'], ['tag'])

    def test_evaluate_tags_special_always_in_skip_tags_and_always_in_tags(self):
        if False:
            return 10
        self.assert_evaluate_equal(False, ['tag', 'always'], [], ['always'])

    def test_evaluate_tags_special_tagged_in_only_tags_and_object_tagged(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_evaluate_equal(True, ['tag'], ['tagged'], [])

    def test_evaluate_tags_special_tagged_in_only_tags_and_object_untagged(self):
        if False:
            while True:
                i = 10
        self.assert_evaluate_equal(False, [], ['tagged'], [])

    def test_evaluate_tags_special_tagged_in_skip_tags_and_object_tagged(self):
        if False:
            i = 10
            return i + 15
        self.assert_evaluate_equal(False, ['tag'], [], ['tagged'])

    def test_evaluate_tags_special_tagged_in_skip_tags_and_object_untagged(self):
        if False:
            print('Hello World!')
        self.assert_evaluate_equal(True, [], [], ['tagged'])

    def test_evaluate_tags_special_untagged_in_only_tags_and_object_tagged(self):
        if False:
            i = 10
            return i + 15
        self.assert_evaluate_equal(False, ['tag'], ['untagged'], [])

    def test_evaluate_tags_special_untagged_in_only_tags_and_object_untagged(self):
        if False:
            return 10
        self.assert_evaluate_equal(True, [], ['untagged'], [])

    def test_evaluate_tags_special_untagged_in_skip_tags_and_object_tagged(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_evaluate_equal(True, ['tag'], [], ['untagged'])

    def test_evaluate_tags_special_untagged_in_skip_tags_and_object_untagged(self):
        if False:
            i = 10
            return i + 15
        self.assert_evaluate_equal(False, [], [], ['untagged'])

    def test_evaluate_tags_special_all_in_only_tags(self):
        if False:
            while True:
                i = 10
        self.assert_evaluate_equal(True, ['tag'], ['all'], ['untagged'])

    def test_evaluate_tags_special_all_in_only_tags_and_object_untagged(self):
        if False:
            return 10
        self.assert_evaluate_equal(True, [], ['all'], [])

    def test_evaluate_tags_special_all_in_skip_tags(self):
        if False:
            print('Hello World!')
        self.assert_evaluate_equal(False, ['tag'], ['tag'], ['all'])

    def test_evaluate_tags_special_all_in_only_tags_and_special_all_in_skip_tags(self):
        if False:
            print('Hello World!')
        self.assert_evaluate_equal(False, ['tag'], ['all'], ['all'])

    def test_evaluate_tags_special_all_in_skip_tags_and_always_in_object_tags(self):
        if False:
            print('Hello World!')
        self.assert_evaluate_equal(True, ['tag', 'always'], [], ['all'])

    def test_evaluate_tags_special_all_in_skip_tags_and_special_always_in_skip_tags_and_always_in_object_tags(self):
        if False:
            i = 10
            return i + 15
        self.assert_evaluate_equal(False, ['tag', 'always'], [], ['all', 'always'])

    def test_evaluate_tags_accepts_lists(self):
        if False:
            i = 10
            return i + 15
        self.assert_evaluate_equal(True, ['tag1', 'tag2'], ['tag2'], [])

    def test_evaluate_tags_with_repeated_tags(self):
        if False:
            return 10
        self.assert_evaluate_equal(False, ['tag', 'tag'], [], ['tag'])