import unittest
from pika import amqp_object

class AMQPObjectTests(unittest.TestCase):

    def test_base_name(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(amqp_object.AMQPObject().NAME, 'AMQPObject')

    def test_repr_no_items(self):
        if False:
            print('Hello World!')
        obj = amqp_object.AMQPObject()
        self.assertEqual(repr(obj), '<AMQPObject>')

    def test_repr_items(self):
        if False:
            i = 10
            return i + 15
        obj = amqp_object.AMQPObject()
        setattr(obj, 'foo', 'bar')
        setattr(obj, 'baz', 'qux')
        self.assertEqual(repr(obj), "<AMQPObject(['baz=qux', 'foo=bar'])>")

    def test_equality(self):
        if False:
            return 10
        a = amqp_object.AMQPObject()
        b = amqp_object.AMQPObject()
        self.assertEqual(a, b)
        setattr(a, 'a_property', 'test')
        self.assertNotEqual(a, b)
        setattr(b, 'a_property', 'test')
        self.assertEqual(a, b)

class ClassTests(unittest.TestCase):

    def test_base_name(self):
        if False:
            print('Hello World!')
        self.assertEqual(amqp_object.Class().NAME, 'Unextended Class')

    def test_equality(self):
        if False:
            return 10
        a = amqp_object.Class()
        b = amqp_object.Class()
        self.assertEqual(a, b)

class MethodTests(unittest.TestCase):

    def test_base_name(self):
        if False:
            while True:
                i = 10
        self.assertEqual(amqp_object.Method().NAME, 'Unextended Method')

    def test_set_content_body(self):
        if False:
            while True:
                i = 10
        properties = amqp_object.Properties()
        body = 'This is a test'
        obj = amqp_object.Method()
        obj._set_content(properties, body)
        self.assertEqual(obj._body, body)

    def test_set_content_properties(self):
        if False:
            for i in range(10):
                print('nop')
        properties = amqp_object.Properties()
        body = 'This is a test'
        obj = amqp_object.Method()
        obj._set_content(properties, body)
        self.assertEqual(obj._properties, properties)

    def test_get_body(self):
        if False:
            return 10
        properties = amqp_object.Properties()
        body = 'This is a test'
        obj = amqp_object.Method()
        obj._set_content(properties, body)
        self.assertEqual(obj.get_body(), body)

    def test_get_properties(self):
        if False:
            for i in range(10):
                print('nop')
        properties = amqp_object.Properties()
        body = 'This is a test'
        obj = amqp_object.Method()
        obj._set_content(properties, body)
        self.assertEqual(obj.get_properties(), properties)

class PropertiesTests(unittest.TestCase):

    def test_base_name(self):
        if False:
            print('Hello World!')
        self.assertEqual(amqp_object.Properties().NAME, 'Unextended Properties')