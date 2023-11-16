"""Tests for syncing metrics related data from GitHub."""
import unittest
from ddt import ddt, data
import ghutilities

@ddt
class GhutilitiesTestCase(unittest.TestCase):

    @data(('sample text with mention @mention', ['mention']), ('Data without mention', []), ('sample text with several mentions @first, @second @third', ['first', 'second', 'third']))
    def test_findMentions_finds_mentions_by_pattern(self, params):
        if False:
            return 10
        (input, expectedResult) = params
        result = ghutilities.findMentions(input)
        self.assertEqual(expectedResult, result)

    def test_findCommentReviewers(self):
        if False:
            for i in range(10):
                print('nop')
        result = 'some tesxt \n body'
if __name__ == '__main__':
    unittest.main()