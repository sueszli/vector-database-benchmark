import shutil
import os
import pytest
from Site import SiteManager
TEST_DATA_PATH = 'src/Test/testdata'

@pytest.mark.usefixtures('resetSettings')
class TestSite:

    def testClone(self, site):
        if False:
            for i in range(10):
                print('nop')
        assert site.storage.directory == TEST_DATA_PATH + '/1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT'
        if os.path.isdir(TEST_DATA_PATH + '/159EGD5srUsMP97UpcLy8AtKQbQLK2AbbL'):
            shutil.rmtree(TEST_DATA_PATH + '/159EGD5srUsMP97UpcLy8AtKQbQLK2AbbL')
        assert not os.path.isfile(TEST_DATA_PATH + '/159EGD5srUsMP97UpcLy8AtKQbQLK2AbbL/content.json')
        new_site = site.clone('159EGD5srUsMP97UpcLy8AtKQbQLK2AbbL', '5JU2p5h3R7B1WrbaEdEDNZR7YHqRLGcjNcqwqVQzX2H4SuNe2ee', address_index=1)
        assert new_site.address == '159EGD5srUsMP97UpcLy8AtKQbQLK2AbbL'
        assert new_site.storage.isFile('content.json')
        assert new_site.storage.isFile('index.html')
        assert new_site.storage.isFile('data/users/content.json')
        assert new_site.storage.isFile('data/zeroblog.db')
        assert new_site.storage.verifyFiles()['bad_files'] == []
        assert new_site.storage.query("SELECT * FROM keyvalue WHERE key = 'title'").fetchone()['value'] == 'MyZeroBlog'
        assert len(new_site.storage.loadJson('content.json').get('files_optional', {})) == 0
        new_site.storage.write('index.html', b'this will be overwritten')
        assert new_site.storage.read('index.html') == b'this will be overwritten'
        changed_contentjson = new_site.storage.loadJson('content.json')
        changed_contentjson['description'] = 'Update Description Test'
        new_site.storage.writeJson('content.json', changed_contentjson)
        changed_data = new_site.storage.loadJson('data/data.json')
        changed_data['title'] = 'UpdateTest'
        new_site.storage.writeJson('data/data.json', changed_data)
        assert new_site.storage.query("SELECT * FROM keyvalue WHERE key = 'title'").fetchone()['value'] == 'UpdateTest'
        site.log.debug('Re-cloning')
        site.clone('159EGD5srUsMP97UpcLy8AtKQbQLK2AbbL')
        assert new_site.storage.loadJson('data/data.json')['title'] == 'UpdateTest'
        assert new_site.storage.loadJson('content.json')['description'] == 'Update Description Test'
        assert new_site.storage.read('index.html') != 'this will be overwritten'
        new_site.storage.deleteFiles()
        assert not os.path.isdir(TEST_DATA_PATH + '/159EGD5srUsMP97UpcLy8AtKQbQLK2AbbL')
        assert new_site.address in SiteManager.site_manager.sites
        SiteManager.site_manager.delete(new_site.address)
        assert new_site.address not in SiteManager.site_manager.sites