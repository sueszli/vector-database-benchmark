"""Tests about developer experience: help messages, errors, etc."""
import collections
import unittest
import factory
import factory.errors
Country = collections.namedtuple('Country', ['name', 'continent', 'capital_city'])
City = collections.namedtuple('City', ['name', 'population'])

class DeclarationTests(unittest.TestCase):

    def test_subfactory_to_model(self):
        if False:
            i = 10
            return i + 15
        'A helpful error message occurs when pointing a subfactory to a model.'

        class CountryFactory(factory.Factory):

            class Meta:
                model = Country
            name = factory.Faker('country')
            continent = 'Antarctica'
            capital_city = factory.SubFactory(City)
        with self.assertRaises(factory.errors.AssociatedClassError) as raised:
            CountryFactory()
        self.assertIn('City', str(raised.exception))
        self.assertIn('Country', str(raised.exception))

    def test_subfactory_to_factorylike_model(self):
        if False:
            print('Hello World!')
        'A helpful error message occurs when pointing a subfactory to a model.\n\n        This time with a model that looks more like a factory (ie has a `._meta`).'

        class CityModel:
            _meta = None
            name = 'Coruscant'
            population = 0

        class CountryFactory(factory.Factory):

            class Meta:
                model = Country
            name = factory.Faker('country')
            continent = 'Antarctica'
            capital_city = factory.SubFactory(CityModel)
        with self.assertRaises(factory.errors.AssociatedClassError) as raised:
            CountryFactory()
        self.assertIn('CityModel', str(raised.exception))
        self.assertIn('Country', str(raised.exception))