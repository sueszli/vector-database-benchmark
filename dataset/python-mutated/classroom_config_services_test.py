"""Tests for classroom services."""
from __future__ import annotations
from core.constants import constants
from core.domain import classroom_config_domain
from core.domain import classroom_config_services
from core.platform import models
from core.tests import test_utils
MYPY = False
if MYPY:
    from mypy_imports import classroom_models
(classroom_models,) = models.Registry.import_models([models.Names.CLASSROOM])

class ClassroomServicesTests(test_utils.GenericTestBase):
    """Tests for classroom services."""

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        super().setUp()
        self.math_classroom_dict: classroom_config_domain.ClassroomDict = {'classroom_id': 'math_classroom_id', 'name': 'math', 'url_fragment': 'math', 'course_details': 'Curated math foundations course.', 'topic_list_intro': 'Start from the basics with our first topic.', 'topic_id_to_prerequisite_topic_ids': {'topic_id_1': ['topic_id_2', 'topic_id_3'], 'topic_id_2': [], 'topic_id_3': []}}
        self.math_classroom = classroom_config_domain.Classroom.from_dict(self.math_classroom_dict)
        classroom_models.ClassroomModel.create(self.math_classroom.classroom_id, self.math_classroom.name, self.math_classroom.url_fragment, self.math_classroom.course_details, self.math_classroom.topic_list_intro, self.math_classroom.topic_id_to_prerequisite_topic_ids)
        self.physics_classroom_dict: classroom_config_domain.ClassroomDict = {'classroom_id': 'physics_classroom_id', 'name': 'physics', 'url_fragment': 'physics', 'course_details': 'Curated physics foundations course.', 'topic_list_intro': 'Start from the basics with our first topic.', 'topic_id_to_prerequisite_topic_ids': {'topic_id_1': ['topic_id_2', 'topic_id_3'], 'topic_id_2': [], 'topic_id_3': []}}
        self.physics_classroom = classroom_config_domain.Classroom.from_dict(self.physics_classroom_dict)
        classroom_models.ClassroomModel.create(self.physics_classroom.classroom_id, self.physics_classroom.name, self.physics_classroom.url_fragment, self.physics_classroom.course_details, self.physics_classroom.topic_list_intro, self.physics_classroom.topic_id_to_prerequisite_topic_ids)

    def test_get_classroom_by_id(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        classroom = classroom_config_services.get_classroom_by_id('math_classroom_id')
        self.assertEqual(classroom.to_dict(), self.math_classroom_dict)
        self.assertIsNone(classroom_config_services.get_classroom_by_id('incorrect_id', strict=False))

    def test_get_classroom_by_url_fragment(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        classroom = classroom_config_services.get_classroom_by_url_fragment('math')
        assert classroom is not None
        self.assertEqual(classroom.to_dict(), self.math_classroom_dict)
        self.assertIsNone(classroom_config_services.get_classroom_by_url_fragment('incorrect_url_fragment'))

    def test_get_classroom_url_fragment_for_existing_topic(self) -> None:
        if False:
            i = 10
            return i + 15
        chemistry_classroom_dict: classroom_config_domain.ClassroomDict = {'classroom_id': 'chem_classroom_id', 'name': 'chem', 'url_fragment': 'chem', 'course_details': 'Curated Chemistry foundations course.', 'topic_list_intro': 'Start from the basics with our first topic.', 'topic_id_to_prerequisite_topic_ids': {'topic_id_chem': []}}
        chemistry_classroom = classroom_config_domain.Classroom.from_dict(chemistry_classroom_dict)
        classroom_models.ClassroomModel.create(chemistry_classroom.classroom_id, chemistry_classroom.name, chemistry_classroom.url_fragment, chemistry_classroom.course_details, chemistry_classroom.topic_list_intro, chemistry_classroom.topic_id_to_prerequisite_topic_ids)
        classroom_url_fragment = classroom_config_services.get_classroom_url_fragment_for_topic_id('topic_id_chem')
        self.assertEqual(classroom_url_fragment, 'chem')

    def test_get_classroom_url_fragment_for_non_existing_topic(self) -> None:
        if False:
            print('Hello World!')
        classroom_url_fragment = classroom_config_services.get_classroom_url_fragment_for_topic_id('non_existing_topic_id')
        self.assertEqual(classroom_url_fragment, constants.CLASSROOM_URL_FRAGMENT_FOR_UNATTACHED_TOPICS)

    def test_get_all_classrooms(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        classrooms = classroom_config_services.get_all_classrooms()
        classroom_dicts = [classroom.to_dict() for classroom in classrooms]
        self.assertEqual(classroom_dicts, [self.math_classroom_dict, self.physics_classroom_dict])

    def test_get_classroom_id_to_classroom_name_dict(self) -> None:
        if False:
            i = 10
            return i + 15
        classroom_id_to_classroom_name_dict = {'math_classroom_id': 'math', 'physics_classroom_id': 'physics'}
        self.assertEqual(classroom_config_services.get_classroom_id_to_classroom_name_dict(), classroom_id_to_classroom_name_dict)

    def test_get_new_classroom_id(self) -> None:
        if False:
            print('Hello World!')
        classroom_id = classroom_config_services.get_new_classroom_id()
        self.assertFalse(classroom_id == self.math_classroom.classroom_id)
        self.assertFalse(classroom_id == self.physics_classroom.classroom_id)

    def test_create_new_classroom_model(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        new_classroom_id = classroom_config_services.get_new_classroom_id()
        chemistry_classroom = classroom_config_domain.Classroom(new_classroom_id, 'chemistry', 'chemistry', 'Curated chemistry foundations course.', 'Start from the basics with our first topic.', {'topic_id_1': ['topic_id_2', 'topic_id_3'], 'topic_id_2': [], 'topic_id_3': []})
        self.assertIsNone(classroom_config_services.get_classroom_by_id(new_classroom_id, strict=False))
        classroom_config_services.update_or_create_classroom_model(chemistry_classroom)
        self.assertEqual(classroom_config_services.get_classroom_by_id(new_classroom_id).to_dict(), chemistry_classroom.to_dict())

    def test_update_existing_classroom_model(self) -> None:
        if False:
            return 10
        self.assertEqual(classroom_config_services.get_classroom_by_id('physics_classroom_id').name, 'physics')
        self.physics_classroom.name = 'Quantum physics'
        classroom_config_services.update_or_create_classroom_model(self.physics_classroom)
        self.assertEqual(classroom_config_services.get_classroom_by_id('physics_classroom_id').name, 'Quantum physics')

    def test_delete_classroom_model(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assertIsNotNone(classroom_config_services.get_classroom_by_id('math_classroom_id'))
        classroom_config_services.delete_classroom('math_classroom_id')
        self.assertIsNone(classroom_config_services.get_classroom_by_id('math_classroom_id', strict=False))