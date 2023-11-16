import unittest

class TestPowerSet(unittest.TestCase):

    def test_power_set(self):
        if False:
            print('Hello World!')
        input_set = ''
        expected = ['']
        self.run_test(input_set, expected)
        input_set = 'a'
        expected = ['a', '']
        self.run_test(input_set, expected)
        input_set = 'ab'
        expected = ['a', 'ab', 'b', '']
        self.run_test(input_set, expected)
        input_set = 'abc'
        expected = ['a', 'ab', 'abc', 'ac', 'b', 'bc', 'c', '']
        self.run_test(input_set, expected)
        input_set = 'aabc'
        expected = ['a', 'aa', 'aab', 'aabc', 'aac', 'ab', 'abc', 'ac', 'b', 'bc', 'c', '']
        self.run_test(input_set, expected)
        print('Success: test_power_set')

    def run_test(self, input_set, expected):
        if False:
            return 10
        combinatoric = Combinatoric()
        result = combinatoric.find_power_set(input_set)
        self.assertEqual(result, expected)

def main():
    if False:
        i = 10
        return i + 15
    test = TestPowerSet()
    test.test_power_set()
if __name__ == '__main__':
    main()