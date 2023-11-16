import unittest
from pygame.tests.test_utils import question, prompt
import pygame
import pygame._sdl2.controller

class JoystickTypeTest(unittest.TestCase):

    def todo_test_Joystick(self):
        if False:
            i = 10
            return i + 15
        self.fail()

class JoystickModuleTest(unittest.TestCase):

    def test_get_init(self):
        if False:
            return 10

        def error_check_get_init():
            if False:
                print('Hello World!')
            try:
                pygame.joystick.get_count()
            except pygame.error:
                return False
            return True
        self.assertEqual(pygame.joystick.get_init(), False)
        pygame.joystick.init()
        self.assertEqual(pygame.joystick.get_init(), error_check_get_init())
        pygame.joystick.quit()
        self.assertEqual(pygame.joystick.get_init(), error_check_get_init())
        pygame.joystick.init()
        pygame.joystick.init()
        self.assertEqual(pygame.joystick.get_init(), error_check_get_init())
        pygame.joystick.quit()
        self.assertEqual(pygame.joystick.get_init(), error_check_get_init())
        pygame.joystick.quit()
        self.assertEqual(pygame.joystick.get_init(), error_check_get_init())
        for i in range(100):
            pygame.joystick.init()
        self.assertEqual(pygame.joystick.get_init(), error_check_get_init())
        pygame.joystick.quit()
        self.assertEqual(pygame.joystick.get_init(), error_check_get_init())
        for i in range(100):
            pygame.joystick.quit()
        self.assertEqual(pygame.joystick.get_init(), error_check_get_init())

    def test_init(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This unit test is for joystick.init()\n        It was written to help reduce maintenance costs\n        and to help test against changes to the code or\n        different platforms.\n        '
        pygame.quit()
        pygame.init()
        self.assertEqual(pygame.joystick.get_init(), True)
        pygame._sdl2.controller.quit()
        pygame.joystick.quit()
        with self.assertRaises(pygame.error):
            pygame.joystick.get_count()
        iterations = 20
        for i in range(iterations):
            pygame.joystick.init()
        self.assertEqual(pygame.joystick.get_init(), True)
        self.assertIsNotNone(pygame.joystick.get_count())

    def test_quit(self):
        if False:
            for i in range(10):
                print('nop')
        'Test if joystick.quit works.'
        pygame.joystick.init()
        self.assertIsNotNone(pygame.joystick.get_count())
        pygame.joystick.quit()
        with self.assertRaises(pygame.error):
            pygame.joystick.get_count()

    def test_get_count(self):
        if False:
            print('Hello World!')
        pygame.joystick.init()
        try:
            count = pygame.joystick.get_count()
            self.assertGreaterEqual(count, 0, 'joystick.get_count() must return a value >= 0')
        finally:
            pygame.joystick.quit()

class JoystickInteractiveTest(unittest.TestCase):
    __tags__ = ['interactive']

    def test_get_count_interactive(self):
        if False:
            while True:
                i = 10
        prompt('Please connect any joysticks/controllers now before starting the joystick.get_count() test.')
        pygame.joystick.init()
        count = pygame.joystick.get_count()
        response = question(f'NOTE: Having Steam open may add an extra virtual controller for each joystick/controller physically plugged in.\njoystick.get_count() thinks there is [{count}] joystick(s)/controller(s)connected to this system. Is this correct?')
        self.assertTrue(response)
        if count != 0:
            for x in range(count):
                pygame.joystick.Joystick(x)
            with self.assertRaises(pygame.error):
                pygame.joystick.Joystick(count)
        pygame.joystick.quit()
if __name__ == '__main__':
    unittest.main()