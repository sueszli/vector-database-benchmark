import unittest
from patterns.structural.adapter import Adapter, Car, Cat, Dog, Human

class ClassTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.dog = Dog()
        self.cat = Cat()
        self.human = Human()
        self.car = Car()

    def test_dog_shall_bark(self):
        if False:
            while True:
                i = 10
        noise = self.dog.bark()
        expected_noise = 'woof!'
        self.assertEqual(noise, expected_noise)

    def test_cat_shall_meow(self):
        if False:
            for i in range(10):
                print('nop')
        noise = self.cat.meow()
        expected_noise = 'meow!'
        self.assertEqual(noise, expected_noise)

    def test_human_shall_speak(self):
        if False:
            return 10
        noise = self.human.speak()
        expected_noise = "'hello'"
        self.assertEqual(noise, expected_noise)

    def test_car_shall_make_loud_noise(self):
        if False:
            print('Hello World!')
        noise = self.car.make_noise(1)
        expected_noise = 'vroom!'
        self.assertEqual(noise, expected_noise)

    def test_car_shall_make_very_loud_noise(self):
        if False:
            i = 10
            return i + 15
        noise = self.car.make_noise(10)
        expected_noise = 'vroom!!!!!!!!!!'
        self.assertEqual(noise, expected_noise)

class AdapterTest(unittest.TestCase):

    def test_dog_adapter_shall_make_noise(self):
        if False:
            while True:
                i = 10
        dog = Dog()
        dog_adapter = Adapter(dog, make_noise=dog.bark)
        noise = dog_adapter.make_noise()
        expected_noise = 'woof!'
        self.assertEqual(noise, expected_noise)

    def test_cat_adapter_shall_make_noise(self):
        if False:
            for i in range(10):
                print('nop')
        cat = Cat()
        cat_adapter = Adapter(cat, make_noise=cat.meow)
        noise = cat_adapter.make_noise()
        expected_noise = 'meow!'
        self.assertEqual(noise, expected_noise)

    def test_human_adapter_shall_make_noise(self):
        if False:
            for i in range(10):
                print('nop')
        human = Human()
        human_adapter = Adapter(human, make_noise=human.speak)
        noise = human_adapter.make_noise()
        expected_noise = "'hello'"
        self.assertEqual(noise, expected_noise)

    def test_car_adapter_shall_make_loud_noise(self):
        if False:
            return 10
        car = Car()
        car_adapter = Adapter(car, make_noise=car.make_noise)
        noise = car_adapter.make_noise(1)
        expected_noise = 'vroom!'
        self.assertEqual(noise, expected_noise)

    def test_car_adapter_shall_make_very_loud_noise(self):
        if False:
            i = 10
            return i + 15
        car = Car()
        car_adapter = Adapter(car, make_noise=car.make_noise)
        noise = car_adapter.make_noise(10)
        expected_noise = 'vroom!!!!!!!!!!'
        self.assertEqual(noise, expected_noise)