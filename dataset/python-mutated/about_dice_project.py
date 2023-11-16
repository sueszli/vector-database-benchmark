from runner.koan import *
import random

class DiceSet:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._values = None

    @property
    def values(self):
        if False:
            return 10
        return self._values

    def roll(self, n):
        if False:
            for i in range(10):
                print('nop')
        pass

class AboutDiceProject(Koan):

    def test_can_create_a_dice_set(self):
        if False:
            while True:
                i = 10
        dice = DiceSet()
        self.assertTrue(dice)

    def test_rolling_the_dice_returns_a_set_of_integers_between_1_and_6(self):
        if False:
            while True:
                i = 10
        dice = DiceSet()
        dice.roll(5)
        self.assertTrue(isinstance(dice.values, list), 'should be a list')
        self.assertEqual(5, len(dice.values))
        for value in dice.values:
            self.assertTrue(value >= 1 and value <= 6, 'value ' + str(value) + ' must be between 1 and 6')

    def test_dice_values_do_not_change_unless_explicitly_rolled(self):
        if False:
            print('Hello World!')
        dice = DiceSet()
        dice.roll(5)
        first_time = dice.values
        second_time = dice.values
        self.assertEqual(first_time, second_time)

    def test_dice_values_should_change_between_rolls(self):
        if False:
            while True:
                i = 10
        dice = DiceSet()
        dice.roll(5)
        first_time = dice.values
        dice.roll(5)
        second_time = dice.values
        self.assertNotEqual(first_time, second_time, 'Two rolls should not be equal')

    def test_you_can_roll_different_numbers_of_dice(self):
        if False:
            print('Hello World!')
        dice = DiceSet()
        dice.roll(3)
        self.assertEqual(3, len(dice.values))
        dice.roll(1)
        self.assertEqual(1, len(dice.values))