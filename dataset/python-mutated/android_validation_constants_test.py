"""Tests that the corresponding constants in android_validation_constants and
constants are the same.
"""
from __future__ import annotations
from core import android_validation_constants
from core.constants import constants
from core.tests import test_utils

class AndroidValidationConstantsTest(test_utils.GenericTestBase):
    """Tests verifying the character limits."""

    def test_that_character_limits_in_both_files_are_equal(self) -> None:
        if False:
            return 10
        self.assertEqual(android_validation_constants.MAX_CHARS_IN_ABBREV_TOPIC_NAME, constants.MAX_CHARS_IN_ABBREV_TOPIC_NAME)
        self.assertEqual(android_validation_constants.MAX_CHARS_IN_TOPIC_NAME, constants.MAX_CHARS_IN_TOPIC_NAME)
        self.assertEqual(android_validation_constants.MAX_CHARS_IN_TOPIC_DESCRIPTION, constants.MAX_CHARS_IN_TOPIC_DESCRIPTION)
        self.assertEqual(android_validation_constants.MAX_CHARS_IN_SUBTOPIC_TITLE, constants.MAX_CHARS_IN_SUBTOPIC_TITLE)
        self.assertEqual(android_validation_constants.MAX_CHARS_IN_SKILL_DESCRIPTION, constants.MAX_CHARS_IN_SKILL_DESCRIPTION)
        self.assertEqual(android_validation_constants.MAX_CHARS_IN_STORY_TITLE, constants.MAX_CHARS_IN_STORY_TITLE)
        self.assertEqual(android_validation_constants.MAX_CHARS_IN_EXPLORATION_TITLE, constants.MAX_CHARS_IN_EXPLORATION_TITLE)
        self.assertEqual(android_validation_constants.MAX_CHARS_IN_CHAPTER_DESCRIPTION, constants.MAX_CHARS_IN_CHAPTER_DESCRIPTION)
        self.assertEqual(android_validation_constants.MAX_CHARS_IN_MISCONCEPTION_NAME, constants.MAX_CHARS_IN_MISCONCEPTION_NAME)
        self.assertEqual(android_validation_constants.MAX_CHARS_IN_STORY_DESCRIPTION, constants.MAX_CHARS_IN_STORY_DESCRIPTION)

    def test_exploration_constants_in_both_files_are_equal(self) -> None:
        if False:
            print('Hello World!')
        interaction_ids_in_constants = []
        language_ids_in_constants = []
        constants_interactions_list = constants.ALLOWED_EXPLORATION_IN_STORY_INTERACTION_CATEGORIES
        constants_languages_list = constants.SUPPORTED_CONTENT_LANGUAGES_FOR_ANDROID
        for obj in constants_interactions_list:
            interaction_ids_in_constants.extend(obj['interaction_ids'])
        for obj in constants_languages_list:
            language_ids_in_constants.append(obj['code'])
        self.assertItemsEqual(interaction_ids_in_constants, android_validation_constants.VALID_INTERACTION_IDS)
        self.assertItemsEqual(constants.VALID_RTE_COMPONENTS_FOR_ANDROID, android_validation_constants.VALID_RTE_COMPONENTS)
        self.assertItemsEqual(language_ids_in_constants, android_validation_constants.SUPPORTED_LANGUAGES)