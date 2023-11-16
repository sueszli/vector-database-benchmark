"""Tests for config services."""
from __future__ import annotations
from core.domain import config_domain
from core.domain import config_services
from core.tests import test_utils

class ConfigServicesTests(test_utils.GenericTestBase):
    """Tests for config services."""

    def test_can_set_config_property(self) -> None:
        if False:
            return 10
        self.assertEqual(config_domain.CLASSROOM_PAGES_DATA.value, [{'name': 'math', 'url_fragment': 'math', 'topic_ids': [], 'course_details': '', 'topic_list_intro': ''}])
        config_services.set_property('admin', 'classroom_pages_data', [{'name': 'math', 'url_fragment': 'math', 'topic_ids': [], 'course_details': 'Detailed math classroom.', 'topic_list_intro': ''}])
        self.assertEqual(config_domain.CLASSROOM_PAGES_DATA.value, [{'name': 'math', 'url_fragment': 'math', 'topic_ids': [], 'course_details': 'Detailed math classroom.', 'topic_list_intro': ''}])

    def test_can_not_set_config_property_with_invalid_config_property_name(self) -> None:
        if False:
            return 10
        with self.assertRaisesRegex(Exception, 'No config property with name new_config_property_name found.'):
            config_services.set_property('admin', 'new_config_property_name', True)

    def test_can_revert_config_property(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assertEqual(config_domain.CLASSROOM_PAGES_DATA.value, [{'name': 'math', 'url_fragment': 'math', 'topic_ids': [], 'course_details': '', 'topic_list_intro': ''}])
        config_services.set_property('admin', 'classroom_pages_data', [{'name': 'math', 'url_fragment': 'math', 'topic_ids': [], 'course_details': 'Detailed math classroom.', 'topic_list_intro': ''}])
        self.assertEqual(config_domain.CLASSROOM_PAGES_DATA.value, [{'name': 'math', 'url_fragment': 'math', 'topic_ids': [], 'course_details': 'Detailed math classroom.', 'topic_list_intro': ''}])
        config_services.revert_property('admin', 'classroom_pages_data')
        self.assertEqual(config_domain.CLASSROOM_PAGES_DATA.value, [{'name': 'math', 'url_fragment': 'math', 'topic_ids': [], 'course_details': '', 'topic_list_intro': ''}])

    def test_can_not_revert_config_property_with_invalid_config_property_name(self) -> None:
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(Exception, 'No config property with name new_config_property_name found.'):
            config_services.revert_property('admin', 'new_config_property_name')