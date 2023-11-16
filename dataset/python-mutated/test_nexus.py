import pytest
import salt.modules.nexus as nexus
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        return 10
    return {nexus: {}}

def test_artifact_get_metadata():
    if False:
        while True:
            i = 10
    with patch('salt.modules.nexus._get_artifact_metadata_xml', MagicMock(return_value='<?xml version="1.0" encoding="UTF-8"?>\n<metadata>\n  <groupId>com.company.sampleapp.web-module</groupId>\n  <artifactId>web</artifactId>\n  <versioning>\n    <release>0.1.0</release>\n    <versions>\n      <version>0.0.1</version>\n      <version>0.0.2</version>\n      <version>0.0.3</version>\n      <version>0.1.0</version>\n    </versions>\n    <lastUpdated>20171010143552</lastUpdated>\n  </versioning>\n</metadata>')):
        metadata = nexus._get_artifact_metadata(nexus_url='http://nexus.example.com/repository', repository='libs-releases', group_id='com.company.sampleapp.web-module', artifact_id='web', headers={})
        assert metadata['latest_version'] == '0.1.0'

def test_snapshot_version_get_metadata():
    if False:
        while True:
            i = 10
    with patch('salt.modules.nexus._get_snapshot_version_metadata_xml', MagicMock(return_value='<?xml version="1.0" encoding="UTF-8"?>\n<metadata modelVersion="1.1.0">\n  <groupId>com.company.sampleapp.web-module</groupId>\n  <artifactId>web</artifactId>\n  <version>0.0.2-SNAPSHOT</version>\n  <versioning>\n    <snapshot>\n      <timestamp>20170920.212353</timestamp>\n      <buildNumber>3</buildNumber>\n    </snapshot>\n    <lastUpdated>20171112171500</lastUpdated>\n    <snapshotVersions>\n      <snapshotVersion>\n        <classifier>sans-externalized</classifier>\n        <extension>jar</extension>\n        <value>0.0.2-20170920.212353-3</value>\n        <updated>20170920212353</updated>\n      </snapshotVersion>\n      <snapshotVersion>\n        <classifier>dist</classifier>\n        <extension>zip</extension>\n        <value>0.0.2-20170920.212353-3</value>\n        <updated>20170920212353</updated>\n      </snapshotVersion>\n    </snapshotVersions>\n  </versioning>\n</metadata>')):
        metadata = nexus._get_snapshot_version_metadata(nexus_url='http://nexus.example.com/repository', repository='libs-releases', group_id='com.company.sampleapp.web-module', artifact_id='web', version='0.0.2-SNAPSHOT', headers={})
        assert metadata['snapshot_versions']['dist'] == '0.0.2-20170920.212353-3'

def test_artifact_metadata_url():
    if False:
        for i in range(10):
            print('nop')
    metadata_url = nexus._get_artifact_metadata_url(nexus_url='http://nexus.example.com/repository', repository='libs-releases', group_id='com.company.sampleapp.web-module', artifact_id='web')
    assert metadata_url == 'http://nexus.example.com/repository/libs-releases/com/company/sampleapp/web-module/web/maven-metadata.xml'

def test_snapshot_version_metadata_url():
    if False:
        for i in range(10):
            print('nop')
    metadata_url = nexus._get_snapshot_version_metadata_url(nexus_url='http://nexus.example.com/repository', repository='libs-snapshots', group_id='com.company.sampleapp.web-module', artifact_id='web', version='0.0.2-SNAPSHOT')
    assert metadata_url == 'http://nexus.example.com/repository/libs-snapshots/com/company/sampleapp/web-module/web/0.0.2-SNAPSHOT/maven-metadata.xml'

def test_construct_url_for_released_version():
    if False:
        for i in range(10):
            print('nop')
    (artifact_url, file_name) = nexus._get_release_url(repository='libs-releases', group_id='com.company.sampleapp.web-module', artifact_id='web', packaging='zip', nexus_url='http://nexus.example.com/repository', version='0.1.0')
    assert artifact_url == 'http://nexus.example.com/repository/libs-releases/com/company/sampleapp/web-module/web/0.1.0/web-0.1.0.zip'
    assert file_name == 'web-0.1.0.zip'

def test_construct_url_for_snapshot_version():
    if False:
        print('Hello World!')
    with patch('salt.modules.nexus._get_snapshot_version_metadata', MagicMock(return_value={'snapshot_versions': {'zip': '0.0.2-20170920.212353-3'}})):
        (artifact_url, file_name) = nexus._get_snapshot_url(nexus_url='http://nexus.example.com/repository', repository='libs-snapshots', group_id='com.company.sampleapp.web-module', artifact_id='web', version='0.2.0-SNAPSHOT', packaging='zip', headers={})
        assert artifact_url == 'http://nexus.example.com/repository/libs-snapshots/com/company/sampleapp/web-module/web/0.2.0-SNAPSHOT/web-0.0.2-20170920.212353-3.zip'
        assert file_name == 'web-0.0.2-20170920.212353-3.zip'