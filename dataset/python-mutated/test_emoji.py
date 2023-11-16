import unittest
from faker import Faker

class TestGlobal(unittest.TestCase):
    """Test emoji provider methods"""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.fake = Faker()
        Faker.seed(0)

    def test_emoji(self):
        if False:
            for i in range(10):
                print('nop')
        emoji = self.fake.emoji()
        assert isinstance(emoji, str)