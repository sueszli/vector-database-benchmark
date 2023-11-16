import random
import unittest
from string import Template
from embedchain import App
from embedchain.config import AppConfig, BaseLlmConfig
from embedchain.helper.json_serializable import JSONSerializable, register_deserializable

class TestJsonSerializable(unittest.TestCase):
    """Test that the datatype detection is working, based on the input."""

    def test_base_function(self):
        if False:
            while True:
                i = 10
        'Test that the base premise of serialization and deserealization is working'

        @register_deserializable
        class TestClass(JSONSerializable):

            def __init__(self):
                if False:
                    print('Hello World!')
                self.rng = random.random()
        original_class = TestClass()
        serial = original_class.serialize()
        negative_test_class = TestClass()
        self.assertNotEqual(original_class.rng, negative_test_class.rng)
        positive_test_class: TestClass = TestClass().deserialize(serial)
        self.assertEqual(original_class.rng, positive_test_class.rng)
        self.assertTrue(isinstance(positive_test_class, TestClass))
        positive_test_class: TestClass = TestClass.deserialize(serial)
        self.assertEqual(original_class.rng, positive_test_class.rng)

    def test_registration_required(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that registration is required, and that without registration the default class is returned.'

        class SecondTestClass(JSONSerializable):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.default = True
        app = SecondTestClass()
        app.default = False
        serial = app.serialize()
        app: SecondTestClass = SecondTestClass().deserialize(serial)
        self.assertTrue(app.default)
        SecondTestClass._register_class_as_deserializable(SecondTestClass)
        app: SecondTestClass = SecondTestClass().deserialize(serial)
        self.assertFalse(app.default)

    def test_recursive(self):
        if False:
            i = 10
            return i + 15
        'Test recursiveness with the real app'
        random_id = str(random.random())
        config = AppConfig(id=random_id, collect_metrics=False)
        app = App(config=config)
        s = app.serialize()
        new_app: App = App.deserialize(s)
        self.assertEqual(random_id, new_app.config.id)

    def test_special_subclasses(self):
        if False:
            i = 10
            return i + 15
        'Test special subclasses that are not serializable by default.'
        config = BaseLlmConfig(template=Template('My custom template with $query, $context and $history.'))
        s = config.serialize()
        new_config: BaseLlmConfig = BaseLlmConfig.deserialize(s)
        self.assertEqual(config.template.template, new_config.template.template)