from django.test import TestCase
from posthog.models.entity import TREND_FILTER_TYPE_ACTIONS, TREND_FILTER_TYPE_EVENTS, Entity

class TestEntity(TestCase):

    def test_can_init_without_id(self):
        if False:
            return 10
        entity = Entity({'type': TREND_FILTER_TYPE_EVENTS})
        self.assertEqual(entity.id, None)

    def test_inclusion(self):
        if False:
            print('Hello World!')
        entity1 = Entity({'id': 'e1', 'type': TREND_FILTER_TYPE_EVENTS, 'properties': [{'key': 'email', 'value': 'test@posthog.com', 'type': 'person'}, {'key': 'current_url', 'value': 'test@posthog.com', 'type': 'element'}]})
        entity2 = Entity({'id': 'e1', 'type': TREND_FILTER_TYPE_EVENTS, 'properties': [{'key': 'current_url', 'value': 'test@posthog.com', 'type': 'element'}]})
        self.assertTrue(entity2.is_superset(entity1))
        self.assertFalse(entity1.is_superset(entity2))

    def test_inclusion_unordered(self):
        if False:
            print('Hello World!')
        entity1 = Entity({'id': 'e1', 'type': TREND_FILTER_TYPE_EVENTS, 'properties': [{'key': 'browser', 'value': 'chrome', 'type': 'person'}, {'key': 'current_url', 'value': 'test@posthog.com', 'type': 'element'}, {'key': 'email', 'value': 'test@posthog.com', 'type': 'person'}]})
        entity2 = Entity({'id': 'e1', 'type': TREND_FILTER_TYPE_EVENTS, 'properties': [{'key': 'current_url', 'value': 'test@posthog.com', 'type': 'element'}]})
        self.assertTrue(entity2.is_superset(entity1))
        self.assertFalse(entity1.is_superset(entity2))

    def test_equality_with_ids(self):
        if False:
            for i in range(10):
                print('nop')
        entity1 = Entity({'id': 'e1', 'type': TREND_FILTER_TYPE_ACTIONS})
        entity2 = Entity({'id': 'e1', 'type': TREND_FILTER_TYPE_ACTIONS})
        self.assertTrue(entity1.equals(entity2))
        entity2 = Entity({'id': 'e2', 'type': TREND_FILTER_TYPE_ACTIONS})
        self.assertFalse(entity1.equals(entity2))

    def test_equality_with_type(self):
        if False:
            while True:
                i = 10
        entity1 = Entity({'id': 'e1', 'type': TREND_FILTER_TYPE_EVENTS})
        entity2 = Entity({'id': 'e1', 'type': TREND_FILTER_TYPE_EVENTS})
        self.assertTrue(entity1.equals(entity2))
        entity1 = Entity({'id': 'e1', 'type': TREND_FILTER_TYPE_EVENTS})
        entity2 = Entity({'id': 'e1', 'type': TREND_FILTER_TYPE_ACTIONS})
        self.assertFalse(entity1.equals(entity2))

    def test_equality_with_simple_properties(self):
        if False:
            i = 10
            return i + 15
        entity1 = Entity({'id': 'e1', 'type': TREND_FILTER_TYPE_EVENTS, 'properties': [{'key': 'email', 'value': 'test@posthog.com', 'type': 'person'}, {'key': 'current_url', 'value': 'test@posthog.com', 'type': 'element'}]})
        entity2 = Entity({'id': 'e1', 'type': TREND_FILTER_TYPE_EVENTS, 'properties': [{'key': 'current_url', 'value': 'test@posthog.com', 'type': 'element'}, {'key': 'email', 'value': 'test@posthog.com', 'type': 'person'}]})
        self.assertTrue(entity1.equals(entity2))
        entity2 = Entity({'id': 'e1', 'type': TREND_FILTER_TYPE_EVENTS, 'properties': [{'key': 'current$url', 'value': 'test@posthog.com', 'type': 'element'}, {'key': 'email', 'value': 'test@posthog.com', 'type': 'person'}]})
        self.assertFalse(entity1.equals(entity2))

    def test_equality_with_complex_operator_properties(self):
        if False:
            while True:
                i = 10
        entity1 = Entity({'id': 'e1', 'type': TREND_FILTER_TYPE_EVENTS, 'properties': [{'key': 'count', 'operator': 'lt', 'value': 12, 'type': 'element'}, {'key': 'email', 'operator': 'in', 'value': ['a, b'], 'type': 'person'}, {'key': 'selector', 'value': ['.btn'], 'operator': 'exact', 'type': 'element'}, {'key': 'test_prop', 'value': 1.2, 'operator': 'gt'}]})
        entity2 = Entity({'id': 'e1', 'type': TREND_FILTER_TYPE_EVENTS, 'properties': [{'key': 'test_prop', 'value': 1.2, 'operator': 'gt'}, {'key': 'count', 'operator': 'lt', 'value': 12, 'type': 'element'}, {'key': 'selector', 'value': ['.btn'], 'operator': 'exact', 'type': 'element'}, {'key': 'email', 'operator': 'in', 'value': ['a, b'], 'type': 'person'}]})
        self.assertTrue(entity1.equals(entity2))
        entity2 = Entity({'id': 'e1', 'type': TREND_FILTER_TYPE_EVENTS, 'properties': [{'key': 'test_prop', 'value': 1.2, 'operator': 'gt'}, {'key': 'count', 'operator': 'lt', 'value': 12, 'type': 'element'}, {'key': 'selector', 'value': ['.btn'], 'operator': 'exact', 'type': 'element'}, {'key': 'email', 'operator': 'in', 'value': ['a, b'], 'type': 'person'}]})
        self.assertTrue(entity1.equals(entity2))
        entity2 = Entity({'id': 'e1', 'type': TREND_FILTER_TYPE_EVENTS, 'properties': [{'key': 'test_prop', 'value': 1.2001, 'operator': 'gt'}, {'key': 'count', 'operator': 'lt', 'value': 12, 'type': 'element'}, {'key': 'selector', 'value': ['.btn'], 'operator': 'exact', 'type': 'element'}, {'key': 'email', 'operator': 'in', 'value': ['a, b'], 'type': 'person'}]})
        self.assertFalse(entity1.equals(entity2))

    def test_equality_with_old_style_and_new_style_properties(self):
        if False:
            while True:
                i = 10
        entity1 = Entity({'id': 'e1', 'type': TREND_FILTER_TYPE_EVENTS, 'properties': {'key': 'value'}})
        entity2 = Entity({'id': 'e1', 'type': TREND_FILTER_TYPE_EVENTS, 'properties': [{'key': 'key', 'value': 'value'}]})
        self.assertTrue(entity1.equals(entity2))