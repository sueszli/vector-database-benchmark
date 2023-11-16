from . import Framework

class Issue945(Framework.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.repo = self.g.get_user('openframeworks').get_repo('openFrameworks')
        self.list = self.repo.get_issues()
        self.list_with_headers = self.repo.get_stargazers_with_dates()

    def testReservedPaginatedListAttributePreservation(self):
        if False:
            while True:
                i = 10
        r1 = self.list.reversed
        self.assertEqual(self.list._PaginatedList__contentClass, r1._PaginatedList__contentClass)
        self.assertEqual(self.list._PaginatedList__requester, r1._PaginatedList__requester)
        self.assertEqual(self.list._PaginatedList__firstUrl, r1._PaginatedList__firstUrl)
        self.assertEqual(self.list._PaginatedList__firstParams, r1._PaginatedList__firstParams)
        self.assertEqual(self.list._PaginatedList__headers, r1._PaginatedList__headers)
        self.assertEqual(self.list._PaginatedList__list_item, r1._PaginatedList__list_item)
        self.assertTrue(self.list_with_headers._PaginatedList__headers is not None)
        r2 = self.list_with_headers.reversed
        self.assertEqual(self.list_with_headers._PaginatedList__contentClass, r2._PaginatedList__contentClass)
        self.assertEqual(self.list_with_headers._PaginatedList__requester, r2._PaginatedList__requester)
        self.assertEqual(self.list_with_headers._PaginatedList__firstUrl, r2._PaginatedList__firstUrl)
        self.assertEqual(self.list_with_headers._PaginatedList__firstParams, r2._PaginatedList__firstParams)
        self.assertEqual(self.list_with_headers._PaginatedList__headers, r2._PaginatedList__headers)
        self.assertEqual(self.list_with_headers._PaginatedList__list_item, r2._PaginatedList__list_item)