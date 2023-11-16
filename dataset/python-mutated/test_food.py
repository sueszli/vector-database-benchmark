import re
import pytest
from mimesis import Food
from . import patterns

class TestFood:

    def test_str(self, food):
        if False:
            while True:
                i = 10
        assert re.match(patterns.DATA_PROVIDER_STR_REGEX, str(food))

    def test_vegetable(self, food):
        if False:
            for i in range(10):
                print('nop')
        result = food.vegetable()
        assert result in food._data['vegetables']

    def test_fruit(self, food):
        if False:
            print('Hello World!')
        result = food.fruit()
        assert result in food._data['fruits']

    def test_dish(self, food):
        if False:
            for i in range(10):
                print('nop')
        result = food.dish()
        assert result in food._data['dishes']

    def test_drink(self, food):
        if False:
            return 10
        result = food.drink()
        assert result in food._data['drinks']

    def test_spices(self, food):
        if False:
            i = 10
            return i + 15
        result = food.spices()
        assert result in food._data['spices']

class TestSeededFood:

    @pytest.fixture
    def fd1(self, seed):
        if False:
            for i in range(10):
                print('nop')
        return Food(seed=seed)

    @pytest.fixture
    def fd2(self, seed):
        if False:
            print('Hello World!')
        return Food(seed=seed)

    def test_vegetable(self, fd1, fd2):
        if False:
            while True:
                i = 10
        assert fd1.vegetable() == fd2.vegetable()

    def test_fruit(self, fd1, fd2):
        if False:
            for i in range(10):
                print('nop')
        assert fd1.fruit() == fd2.fruit()

    def test_dish(self, fd1, fd2):
        if False:
            print('Hello World!')
        assert fd1.dish() == fd2.dish()

    def test_drink(self, fd1, fd2):
        if False:
            return 10
        assert fd1.drink() == fd2.drink()

    def test_spices(self, fd1, fd2):
        if False:
            while True:
                i = 10
        assert fd1.spices() == fd2.spices()