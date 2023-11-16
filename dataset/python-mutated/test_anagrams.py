import unittest

class TestAnagrams(unittest.TestCase):

    def test_group_anagrams(self):
        if False:
            while True:
                i = 10
        anagram = Anagram()
        self.assertRaises(TypeError, anagram.group_anagrams, None)
        data = ['ram', 'act', 'arm', 'bat', 'cat', 'tab']
        expected = ['ram', 'arm', 'act', 'cat', 'bat', 'tab']
        self.assertEqual(anagram.group_anagrams(data), expected)
        print('Success: test_group_anagrams')

def main():
    if False:
        i = 10
        return i + 15
    test = TestAnagrams()
    test.test_group_anagrams()
if __name__ == '__main__':
    main()