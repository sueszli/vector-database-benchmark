import pytest
from faker import Faker
from faker.exceptions import UniquenessException

class TestUniquenessClass:

    def test_uniqueness(self):
        if False:
            print('Hello World!')
        fake = Faker('en_US')
        names = set()
        for i in range(250):
            first_name = fake.unique.first_name()
            assert first_name not in names
            names.add(first_name)

    def test_sanity_escape(self):
        if False:
            print('Hello World!')
        fake = Faker()
        with pytest.raises(UniquenessException, match='Got duplicated values after [\\d,]+ iterations.'):
            for i in range(3):
                _ = fake.unique.boolean()

    def test_uniqueness_clear(self):
        if False:
            while True:
                i = 10
        fake = Faker()
        for i in range(2):
            fake.unique.boolean()
        fake.unique.clear()
        fake.unique.boolean()

    def test_exclusive_arguments(self):
        if False:
            return 10
        'Calls through the "unique" portal will only affect\n        calls with that specific function signature.\n        '
        fake = Faker()
        for i in range(10):
            fake.unique.random_int(min=1, max=10)
        fake.unique.random_int(min=2, max=10)

    def test_functions_only(self):
        if False:
            print('Hello World!')
        'Accessing non-functions through the `.unique` attribute\n        will throw a TypeError.'
        fake = Faker()
        with pytest.raises(TypeError, match='Accessing non-functions through .unique is not supported.'):
            fake.unique.locales