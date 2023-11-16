"""Unit tests for classroom_domain.py"""
from __future__ import annotations
from core.domain import classroom_domain
from core.tests import test_utils

class ClassroomDomainTests(test_utils.GenericTestBase):

    def test_that_domain_object_is_created_correctly(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        classroom_data = classroom_domain.Classroom('exp', 'exp/', [], 'general details', 'general intro')
        self.assertEqual(classroom_data.name, 'exp')
        self.assertEqual(classroom_data.url_fragment, 'exp/')
        self.assertEqual(classroom_data.topic_ids, [])
        self.assertEqual(classroom_data.course_details, 'general details')
        self.assertEqual(classroom_data.topic_list_intro, 'general intro')

    def test_to_dict_returns_correct_dict(self) -> None:
        if False:
            return 10
        classroom_data = classroom_domain.Classroom('exp', 'exp/', [], 'general details', 'general intro')
        self.assertEqual(classroom_data.to_dict(), {'name': classroom_data.name, 'url_fragment': classroom_data.url_fragment, 'topic_ids': classroom_data.topic_ids, 'course_details': classroom_data.course_details, 'topic_list_intro': classroom_data.topic_list_intro})