import pytest

@pytest.mark.usefixtures('resetSettings')
class TestSiteStorage:

    def testWalk(self, site):
        if False:
            while True:
                i = 10
        walk_root = list(site.storage.walk(''))
        assert 'content.json' in walk_root
        assert 'css/all.css' in walk_root
        assert list(site.storage.walk('data-default')) == ['data.json', 'users/content-default.json']

    def testList(self, site):
        if False:
            for i in range(10):
                print('nop')
        list_root = list(site.storage.list(''))
        assert 'content.json' in list_root
        assert 'css/all.css' not in list_root
        assert set(site.storage.list('data-default')) == set(['data.json', 'users'])

    def testDbRebuild(self, site):
        if False:
            print('Hello World!')
        assert site.storage.rebuildDb()