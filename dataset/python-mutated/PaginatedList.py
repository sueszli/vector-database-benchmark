from github.PaginatedList import PaginatedList as PaginatedListImpl
from . import Framework

class PaginatedList(Framework.TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.repo = self.g.get_user('openframeworks').get_repo('openFrameworks')
        self.list = self.repo.get_issues()
        self.licenses = self.g.get_enterprise('beaver-group').get_consumed_licenses()

    def testIteration(self):
        if False:
            while True:
                i = 10
        self.assertEqual(len(list(self.list)), 333)

    def testIterationWithPrefetchedFirstPage(self):
        if False:
            print('Hello World!')
        users = self.licenses.get_users()
        self.assertEqual(len(list(users)), 102)
        self.assertEqual(len({user.github_com_login for user in users}), 102)

    def testSeveralIterations(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(len(list(self.list)), 333)
        self.assertEqual(len(list(self.list)), 333)
        self.assertEqual(len(list(self.list)), 333)
        self.assertEqual(len(list(self.list)), 333)

    def testIntIndexingInFirstPage(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.list[0].id, 4772349)
        self.assertEqual(self.list[24].id, 4286936)

    def testReversedIterationWithSinglePage(self):
        if False:
            print('Hello World!')
        r = self.list.reversed
        self.assertEqual(r[0].id, 4286936)
        self.assertEqual(r[1].id, 4317009)

    def testReversedIterationWithMultiplePages(self):
        if False:
            while True:
                i = 10
        r = self.list.reversed
        self.assertEqual(r[0].id, 94898)
        self.assertEqual(r[1].id, 104702)
        self.assertEqual(r[13].id, 166211)
        self.assertEqual(r[14].id, 166212)
        self.assertEqual(r[15].id, 166214)

    def testReversedIterationSupportsIterator(self):
        if False:
            i = 10
            return i + 15
        r = self.list.reversed
        for i in r:
            self.assertEqual(i.id, 4286936)
            return
        self.fail('empty iterator')

    def testGettingTheReversedListDoesNotModifyTheOriginalList(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.list[0].id, 18345408)
        self.assertEqual(self.list[30].id, 17916118)
        r = self.list.reversed
        self.assertEqual(self.list[0].id, 18345408)
        self.assertEqual(self.list[30].id, 17916118)
        self.assertEqual(r[0].id, 132373)
        self.assertEqual(r[30].id, 543694)

    def testIntIndexingInThirdPage(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.list[50].id, 3911629)
        self.assertEqual(self.list[74].id, 3605277)

    def testGetFirstPage(self):
        if False:
            print('Hello World!')
        self.assertListKeyEqual(self.list.get_page(0), lambda i: i.id, [4772349, 4767675, 4758608, 4700182, 4662873, 4608132, 4604661, 4588997, 4557803, 4554058, 4539985, 4507572, 4507492, 4507416, 4447561, 4406584, 4384548, 4383465, 4373361, 4373201, 4370619, 4356530, 4352401, 4317009, 4286936])

    def testGetThirdPage(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertListKeyEqual(self.list.get_page(2), lambda i: i.id, [3911629, 3911537, 3910580, 3910555, 3910549, 3897090, 3883598, 3856005, 3850655, 3825582, 3813852, 3812318, 3812275, 3807459, 3799872, 3799653, 3795495, 3754055, 3710293, 3662214, 3647640, 3631618, 3627067, 3614231, 3605277])

    def testIntIndexingAfterIteration(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(len(list(self.list)), 333)
        self.assertEqual(self.list[11].id, 4507572)
        self.assertEqual(self.list[73].id, 3614231)
        self.assertEqual(self.list[332].id, 94898)

    def testSliceIndexingInFirstPage(self):
        if False:
            return 10
        self.assertListKeyEqual(self.list[:13], lambda i: i.id, [4772349, 4767675, 4758608, 4700182, 4662873, 4608132, 4604661, 4588997, 4557803, 4554058, 4539985, 4507572, 4507492])
        self.assertListKeyEqual(self.list[:13:3], lambda i: i.id, [4772349, 4700182, 4604661, 4554058, 4507492])
        self.assertListKeyEqual(self.list[10:13], lambda i: i.id, [4539985, 4507572, 4507492])
        self.assertListKeyEqual(self.list[5:13:3], lambda i: i.id, [4608132, 4557803, 4507572])

    def testSliceIndexingUntilFourthPage(self):
        if False:
            return 10
        self.assertListKeyEqual(self.list[:99:10], lambda i: i.id, [4772349, 4539985, 4370619, 4207350, 4063366, 3911629, 3813852, 3647640, 3528378, 3438233])
        self.assertListKeyEqual(self.list[73:78], lambda i: i.id, [3614231, 3605277, 3596240, 3594731, 3593619])
        self.assertListKeyEqual(self.list[70:80:2], lambda i: i.id, [3647640, 3627067, 3605277, 3594731, 3593430])

    def testSliceIndexingUntilEnd(self):
        if False:
            while True:
                i = 10
        self.assertListKeyEqual(self.list[310::3], lambda i: i.id, [268332, 204247, 169176, 166211, 165898, 163959, 132373, 104702])
        self.assertListKeyEqual(self.list[310:], lambda i: i.id, [268332, 211418, 205935, 204247, 172424, 171615, 169176, 166214, 166212, 166211, 166209, 166208, 165898, 165537, 165409, 163959, 132671, 132377, 132373, 130269, 111018, 104702, 94898])

    def testInterruptedIteration(self):
        if False:
            i = 10
            return i + 15
        count = 0
        for element in self.list:
            count += 1
            if count == 75:
                break

    def testInterruptedIterationInSlice(self):
        if False:
            i = 10
            return i + 15
        count = 0
        for element in self.list[:100]:
            count += 1
            if count == 75:
                break

    def testTotalCountWithNoLastPage(self):
        if False:
            for i in range(10):
                print('nop')
        repos = self.g.get_repos()
        self.assertEqual(0, repos.totalCount)

    def testTotalCountWithDictionary(self):
        if False:
            return 10
        pr = self.g.get_repo('PyGithub/PyGithub').get_pull(2078)
        review_requests = pr.get_review_requests()
        self.assertEqual(review_requests[0].totalCount, 0)
        self.assertEqual(review_requests[1].totalCount, 0)

    def testCustomPerPage(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.g.per_page, 30)
        self.g.per_page = 100
        self.assertEqual(self.g.per_page, 100)
        self.assertEqual(len(list(self.repo.get_issues())), 456)

    def testCustomPerPageWithNoUrlParams(self):
        if False:
            i = 10
            return i + 15
        from . import CommitComment
        self.g.per_page = 100
        PaginatedListImpl(CommitComment.CommitComment, self.repo._requester, f'{self.repo.url}/comments', None)

    def testCustomPerPageWithNoUrlParams2(self):
        if False:
            while True:
                i = 10
        self.g.per_page = 100
        self.assertEqual(len(list(self.repo.get_comments())), 325)

    def testCustomPerPageWithGetPage(self):
        if False:
            for i in range(10):
                print('nop')
        self.g.per_page = 100
        self.assertEqual(len(self.repo.get_issues().get_page(2)), 100)

    def testNoFirstPage(self):
        if False:
            print('Hello World!')
        self.assertFalse(next(iter(self.list), None))