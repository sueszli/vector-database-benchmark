import unittest

class TestPDistReportCommand(unittest.TestCase):

    def _callFUT(self, **kw):
        if False:
            i = 10
            return i + 15
        argv = []
        from pyramid.scripts.pdistreport import main
        return main(argv, **kw)

    def test_no_dists(self):
        if False:
            while True:
                i = 10

        def platform():
            if False:
                while True:
                    i = 10
            return 'myplatform'
        pkg_resources = DummyPkgResources()
        L = []

        def out(*args):
            if False:
                return 10
            L.extend(args)
        result = self._callFUT(pkg_resources=pkg_resources, platform=platform, out=out)
        self.assertEqual(result, None)
        self.assertEqual(L, ['Pyramid version:', '1', 'Platform:', 'myplatform', 'Packages:'])

    def test_with_dists(self):
        if False:
            i = 10
            return i + 15

        def platform():
            if False:
                i = 10
                return i + 15
            return 'myplatform'
        working_set = (DummyDistribution('abc'), DummyDistribution('def'))
        pkg_resources = DummyPkgResources(working_set)
        L = []

        def out(*args):
            if False:
                i = 10
                return i + 15
            L.extend(args)
        result = self._callFUT(pkg_resources=pkg_resources, platform=platform, out=out)
        self.assertEqual(result, None)
        self.assertEqual(L, ['Pyramid version:', '1', 'Platform:', 'myplatform', 'Packages:', ' ', 'abc', '1', '   ', '/projects/abc', ' ', 'def', '1', '   ', '/projects/def'])

class DummyPkgResources:

    def __init__(self, working_set=()):
        if False:
            return 10
        self.working_set = working_set

    def get_distribution(self, name):
        if False:
            return 10
        return Version('1')

class Version:

    def __init__(self, version):
        if False:
            for i in range(10):
                print('nop')
        self.version = version

class DummyDistribution:

    def __init__(self, name):
        if False:
            print('Hello World!')
        self.project_name = name
        self.version = '1'
        self.location = '/projects/%s' % name