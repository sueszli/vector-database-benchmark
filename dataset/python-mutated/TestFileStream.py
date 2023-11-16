import unittest
from antlr4.FileStream import FileStream

class TestFileStream(unittest.TestCase):

    def testStream(self):
        if False:
            i = 10
            return i + 15
        stream = FileStream(__file__)
        self.assertTrue(stream.size > 0)