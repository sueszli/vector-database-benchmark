import re
import unittest
from ...core.inputscanner import InputScanner

class TestInputScanner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        pass

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.value = 'howdy'
        self.inputscanner = InputScanner(self.value)

    def test_new(self):
        if False:
            while True:
                i = 10
        inputscanner = InputScanner(None)
        self.assertEqual(inputscanner.hasNext(), False)

    def test_next(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.inputscanner.next(), self.value[0])
        self.assertEqual(self.inputscanner.next(), self.value[1])
        pattern = re.compile('howdy')
        self.inputscanner.readUntilAfter(pattern)
        self.assertEqual(self.inputscanner.next(), None)

    def test_peek(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.inputscanner.peek(3), self.value[3])
        self.inputscanner.next()
        self.assertEqual(self.inputscanner.peek(3), self.value[4])
        self.assertEqual(self.inputscanner.peek(-2), None)
        self.assertEqual(self.inputscanner.peek(5), None)

    def test_no_param(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.inputscanner.peek(), self.value[0])
        self.inputscanner.next()
        self.assertEqual(self.inputscanner.peek(), self.value[1])

    def test_pattern(self):
        if False:
            for i in range(10):
                print('nop')
        pattern = re.compile('how')
        index = 0
        self.assertEqual(self.inputscanner.test(pattern, index), True)
        self.inputscanner.next()
        self.assertEqual(self.inputscanner.test(pattern, index), False)

    def test_Char(self):
        if False:
            i = 10
            return i + 15
        pattern = re.compile('o')
        index = 1
        self.assertEqual(self.inputscanner.testChar(pattern, index), True)

    def test_restart(self):
        if False:
            for i in range(10):
                print('nop')
        self.inputscanner.next()
        self.assertEqual(self.inputscanner.peek(), self.value[1])
        self.inputscanner.restart()
        self.assertEqual(self.inputscanner.peek(), self.value[0])

    def test_back(self):
        if False:
            i = 10
            return i + 15
        self.inputscanner.next()
        self.assertEqual(self.inputscanner.peek(), self.value[1])
        self.inputscanner.back()
        self.assertEqual(self.inputscanner.peek(), self.value[0])
        self.inputscanner.back()
        self.assertEqual(self.inputscanner.peek(), self.value[0])

    def test_hasNext(self):
        if False:
            print('Hello World!')
        pattern = re.compile('howd')
        self.inputscanner.readUntilAfter(pattern)
        self.assertEqual(self.inputscanner.hasNext(), True)
        self.inputscanner.next()
        self.assertEqual(self.inputscanner.hasNext(), False)

    def test_match(self):
        if False:
            return 10
        pattern = re.compile('how')
        patternmatch = self.inputscanner.match(pattern)
        self.assertEqual(self.inputscanner.peek(), self.value[3])
        self.assertNotEqual(patternmatch, None)
        self.assertEqual(patternmatch.group(0), 'how')
        self.inputscanner.restart()
        pattern = re.compile('test')
        patternmatch = self.inputscanner.match(pattern)
        self.assertEqual(self.inputscanner.peek(), self.value[0])
        self.assertEqual(patternmatch, None)

    def test_read(self):
        if False:
            while True:
                i = 10
        pattern = re.compile('how')
        patternmatch = self.inputscanner.read(pattern)
        self.assertEqual(patternmatch, 'how')
        self.inputscanner.restart()
        pattern = re.compile('ow')
        patternmatch = self.inputscanner.read(pattern)
        self.assertEqual(patternmatch, '')
        self.inputscanner.restart()
        startPattern = re.compile('how')
        untilPattern = re.compile('dy')
        untilAfter = True
        patternmatch = self.inputscanner.read(startPattern, untilPattern, untilAfter)
        self.assertEqual(patternmatch, 'howdy')
        self.inputscanner.restart()
        startPattern = re.compile('how')
        untilPattern = re.compile('dy')
        untilAfter = False
        patternmatch = self.inputscanner.read(startPattern, untilPattern, untilAfter)
        self.assertEqual(patternmatch, 'how')
        self.inputscanner.restart()
        startPattern = None
        untilPattern = re.compile('how')
        untilAfter = True
        patternmatch = self.inputscanner.read(startPattern, untilPattern, untilAfter)
        self.inputscanner.restart()
        startPattern = None
        untilPattern = re.compile('how')
        untilAfter = False
        patternmatch = self.inputscanner.read(startPattern, untilPattern, untilAfter)
        self.assertEqual(patternmatch, '')

    def test_readUntil(self):
        if False:
            print('Hello World!')
        pattern = re.compile('how')
        untilAfter = True
        patternmatch = self.inputscanner.readUntil(pattern, untilAfter)
        self.assertEqual(patternmatch, 'how')
        self.inputscanner.restart()
        pattern = re.compile('wd')
        untilAfter = False
        patternmatch = self.inputscanner.readUntil(pattern, untilAfter)
        self.assertEqual(patternmatch, 'ho')
        self.inputscanner.restart()
        pattern = re.compile('how')
        untilAfter = False
        patternmatch = self.inputscanner.readUntil(pattern, untilAfter)
        self.assertEqual(patternmatch, '')

    def test_readUntilAfter(self):
        if False:
            print('Hello World!')
        pattern = re.compile('how')
        patternmatch = self.inputscanner.readUntilAfter(pattern)
        self.assertEqual(patternmatch, 'how')

    def test_get_regexp(self):
        if False:
            return 10
        pattern = re.compile('ow')
        self.assertEqual(self.inputscanner.get_regexp('ow'), pattern)

    def test_peekUntilAfter(self):
        if False:
            while True:
                i = 10
        pattern = re.compile('how')
        self.assertEqual(self.inputscanner.peek(), self.value[0])
        self.assertEqual(self.inputscanner.peekUntilAfter(pattern), 'how')
        self.assertEqual(self.inputscanner.peek(), self.value[0])

    def test_lookBack(self):
        if False:
            while True:
                i = 10
        testVal = 'how'
        pattern = re.compile('howd')
        self.inputscanner.readUntilAfter(pattern)
        self.assertEqual(self.inputscanner.lookBack(testVal), True)
        testVal = 'ho'
        self.assertEqual(self.inputscanner.lookBack(testVal), False)
if __name__ == '__main__':
    unittest.main()