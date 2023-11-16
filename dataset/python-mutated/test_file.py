import re
import unittest
from faker import Faker

class TestFile(unittest.TestCase):
    """Tests file"""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.fake = Faker()
        Faker.seed(0)

    def test_file_path(self):
        if False:
            while True:
                i = 10
        for _ in range(100):
            file_path = self.fake.file_path()
            assert re.search('\\/\\w+\\/\\w+\\.\\w+', file_path)
            file_path = self.fake.file_path(absolute=False)
            assert re.search('\\w+\\/\\w+\\.\\w+', file_path)
            file_path = self.fake.file_path(depth=3)
            assert re.search('\\/\\w+\\/\\w+\\/\\w+\\.\\w+', file_path)
            file_path = self.fake.file_path(extension='pdf')
            assert re.search('\\/\\w+\\/\\w+\\.pdf', file_path)
            file_path = self.fake.file_path(category='image')
            assert re.search('\\/\\w+\\/\\w+\\.(bmp|gif|jpeg|jpg|png|tiff)', file_path)

    def test_unix_device(self):
        if False:
            return 10
        reg_device = re.compile('^/dev/(vd|sd|xvd)[a-z]$')
        for _ in range(100):
            path = self.fake.unix_device()
            assert reg_device.match(path)
        for _ in range(100):
            path = self.fake.unix_device('sd')
            assert reg_device.match(path)
            assert path.startswith('/dev/sd')

    def test_unix_partition(self):
        if False:
            print('Hello World!')
        reg_part = re.compile('^/dev/(vd|sd|xvd)[a-z]\\d$')
        for _ in range(100):
            path = self.fake.unix_partition()
            assert reg_part.match(path)
        for _ in range(100):
            path = self.fake.unix_partition('sd')
            assert reg_part.match(path)
            assert path.startswith('/dev/sd')