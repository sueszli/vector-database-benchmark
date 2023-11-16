import pytest
from mongoengine import *
from tests.utils import MongoDBTestCase

class TestFloatField(MongoDBTestCase):

    def test_float_ne_operator(self):
        if False:
            while True:
                i = 10

        class TestDocument(Document):
            float_fld = FloatField()
        TestDocument.drop_collection()
        TestDocument(float_fld=None).save()
        TestDocument(float_fld=1).save()
        assert 1 == TestDocument.objects(float_fld__ne=None).count()
        assert 1 == TestDocument.objects(float_fld__ne=1).count()

    def test_validation(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure that invalid values cannot be assigned to float fields.'

        class Person(Document):
            height = FloatField(min_value=0.1, max_value=3.5)

        class BigPerson(Document):
            height = FloatField()
        person = Person()
        person.height = 1.89
        person.validate()
        person.height = '2.0'
        with pytest.raises(ValidationError):
            person.validate()
        person.height = 0.01
        with pytest.raises(ValidationError):
            person.validate()
        person.height = 4.0
        with pytest.raises(ValidationError):
            person.validate()
        person_2 = Person(height='something invalid')
        with pytest.raises(ValidationError):
            person_2.validate()
        big_person = BigPerson()
        big_person.height = int(0)
        big_person.validate()
        big_person.height = 2 ** 500
        big_person.validate()
        big_person.height = 2 ** 100000
        with pytest.raises(ValidationError):
            big_person.validate()

    def test_query_none_value_dont_raise(self):
        if False:
            i = 10
            return i + 15

        class BigPerson(Document):
            height = FloatField()
        _ = list(BigPerson.objects(height=None))