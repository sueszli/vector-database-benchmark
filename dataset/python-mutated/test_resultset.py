from config import TweepyTestCase
from tweepy.models import ResultSet

class NoIdItem:
    pass

class IdItem:

    def __init__(self, id):
        if False:
            return 10
        self.id = id
ids_fixture = [1, 10, 8, 50, 2, 100, 5]

class TweepyResultSetTests(TweepyTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.results = ResultSet()
        for i in ids_fixture:
            self.results.append(IdItem(i))
            self.results.append(NoIdItem())

    def testids(self):
        if False:
            return 10
        ids = self.results.ids()
        self.assertListEqual(ids, ids_fixture)

    def testmaxid(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.results.max_id, 0)

    def testsinceid(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.results.since_id, 100)