import unittest

class TestReverseWords(unittest.TestCase):

    def testReverseWords(self, func):
        if False:
            while True:
                i = 10
        self.assertEqual(func('the sun is hot'), 'eht nus si toh')
        self.assertEqual(func(''), None)
        self.assertEqual(func('123 456 789'), '321 654 987')
        self.assertEqual(func('magic'), 'cigam')
        print('Success: reverse_words')

def main():
    if False:
        return 10
    test = TestReverseWords()
    test.testReverseWords(reverse_words)
if __name__ == '__main__':
    main()