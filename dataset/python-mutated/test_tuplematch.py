from twisted.trial import unittest
from buildbot.test.util import tuplematching
from buildbot.util import tuplematch

class MatchTuple(tuplematching.TupleMatchingMixin, unittest.TestCase):

    def do_test_match(self, routingKey, shouldMatch, filter):
        if False:
            return 10
        result = tuplematch.matchTuple(routingKey, filter)
        should_match_string = 'should match' if shouldMatch else "shouldn't match"
        msg = f'{repr(routingKey)} {should_match_string} {repr(filter)}'
        self.assertEqual(shouldMatch, result, msg)