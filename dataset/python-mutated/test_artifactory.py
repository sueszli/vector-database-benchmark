import pytest
import salt.modules.artifactory as artifactory
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        while True:
            i = 10
    return {artifactory: {}}

def test_artifact_get_metadata():
    if False:
        print('Hello World!')
    with patch('salt.modules.artifactory._get_artifact_metadata_xml', MagicMock(return_value='<?xml version="1.0" encoding="UTF-8"?>\n            <metadata>\n              <groupId>com.company.sampleapp.web-module</groupId>\n              <artifactId>web</artifactId>\n              <versioning>\n                <latest>1.1_RC11</latest>\n                <release>1.0.1</release>\n                <versions>\n                  <version>1.0_RC20</version>\n                  <version>1.0_RC22</version>\n                </versions>\n                <lastUpdated>20140623120632</lastUpdated>\n              </versioning>\n            </metadata>\n        ')):
        metadata = artifactory._get_artifact_metadata(artifactory_url='http://artifactory.example.com/artifactory', repository='libs-releases', group_id='com.company.sampleapp.web-module', artifact_id='web', headers={})
        assert metadata['latest_version'] == '1.1_RC11'

def test_snapshot_version_get_metadata():
    if False:
        while True:
            i = 10
    with patch('salt.modules.artifactory._get_snapshot_version_metadata_xml', MagicMock(return_value='<?xml version="1.0" encoding="UTF-8"?>\n                <metadata>\n                  <groupId>com.company.sampleapp.web-module</groupId>\n                  <artifactId>web</artifactId>\n                  <version>1.1_RC8-SNAPSHOT</version>\n                  <versioning>\n                    <snapshot>\n                      <timestamp>20140418.150212</timestamp>\n                      <buildNumber>1</buildNumber>\n                    </snapshot>\n                    <lastUpdated>20140623104055</lastUpdated>\n                    <snapshotVersions>\n                      <snapshotVersion>\n                        <extension>pom</extension>\n                        <value>1.1_RC8-20140418.150212-1</value>\n                        <updated>20140418150212</updated>\n                      </snapshotVersion>\n                      <snapshotVersion>\n                        <extension>war</extension>\n                        <value>1.1_RC8-20140418.150212-1</value>\n                        <updated>20140418150212</updated>\n                      </snapshotVersion>\n                    </snapshotVersions>\n                  </versioning>\n                </metadata>\n            ')):
        metadata = artifactory._get_snapshot_version_metadata(artifactory_url='http://artifactory.example.com/artifactory', repository='libs-releases', group_id='com.company.sampleapp.web-module', artifact_id='web', version='1.1_RC8-SNAPSHOT', headers={})
        assert metadata['snapshot_versions']['war'] == '1.1_RC8-20140418.150212-1'

def test_artifact_metadata_url():
    if False:
        for i in range(10):
            print('nop')
    metadata_url = artifactory._get_artifact_metadata_url(artifactory_url='http://artifactory.example.com/artifactory', repository='libs-releases', group_id='com.company.sampleapp.web-module', artifact_id='web')
    assert metadata_url == 'http://artifactory.example.com/artifactory/libs-releases/com/company/sampleapp/web-module/web/maven-metadata.xml'

def test_snapshot_version_metadata_url():
    if False:
        for i in range(10):
            print('nop')
    metadata_url = artifactory._get_snapshot_version_metadata_url(artifactory_url='http://artifactory.example.com/artifactory', repository='libs-snapshots', group_id='com.company.sampleapp.web-module', artifact_id='web', version='1.0_RC10-SNAPSHOT')
    assert metadata_url == 'http://artifactory.example.com/artifactory/libs-snapshots/com/company/sampleapp/web-module/web/1.0_RC10-SNAPSHOT/maven-metadata.xml'

def test_construct_url_for_released_version():
    if False:
        while True:
            i = 10
    (artifact_url, file_name) = artifactory._get_release_url(repository='libs-releases', group_id='com.company.sampleapp.web-module', artifact_id='web', packaging='war', artifactory_url='http://artifactory.example.com/artifactory', version='1.0_RC20')
    assert artifact_url == 'http://artifactory.example.com/artifactory/libs-releases/com/company/sampleapp/web-module/web/1.0_RC20/web-1.0_RC20.war'
    assert file_name == 'web-1.0_RC20.war'

def test_construct_url_for_snapshot_version():
    if False:
        i = 10
        return i + 15
    with patch('salt.modules.artifactory._get_snapshot_version_metadata', MagicMock(return_value={'snapshot_versions': {'war': '1.0_RC10-20131127.105838-2'}})):
        (artifact_url, file_name) = artifactory._get_snapshot_url(artifactory_url='http://artifactory.example.com/artifactory', repository='libs-snapshots', group_id='com.company.sampleapp.web-module', artifact_id='web', version='1.0_RC10-SNAPSHOT', packaging='war', headers={})
        assert artifact_url == 'http://artifactory.example.com/artifactory/libs-snapshots/com/company/sampleapp/web-module/web/1.0_RC10-SNAPSHOT/web-1.0_RC10-20131127.105838-2.war'
        assert file_name == 'web-1.0_RC10-20131127.105838-2.war'

def test_get_snapshot_url_with_classifier():
    if False:
        i = 10
        return i + 15
    with patch('salt.modules.artifactory._get_snapshot_version_metadata_xml', MagicMock(return_value='<?xml version="1.0" encoding="UTF-8"?>\n                <metadata>\n                  <groupId>com.company.sampleapp.web-module</groupId>\n                  <artifactId>web</artifactId>\n                  <version>1.1_RC8-SNAPSHOT</version>\n                  <versioning>\n                    <snapshot>\n                      <timestamp>20140418.150212</timestamp>\n                      <buildNumber>1</buildNumber>\n                    </snapshot>\n                    <lastUpdated>20140623104055</lastUpdated>\n                    <snapshotVersions>\n                      <snapshotVersion>\n                        <extension>pom</extension>\n                        <value>1.1_RC8-20140418.150212-1</value>\n                        <updated>20140418150212</updated>\n                      </snapshotVersion>\n                      <snapshotVersion>\n                        <classifier>test</classifier>\n                        <extension>war</extension>\n                        <value>1.1_RC8-20140418.150212-1</value>\n                        <updated>20140418150212</updated>\n                      </snapshotVersion>\n                    </snapshotVersions>\n                  </versioning>\n                </metadata>\n            ')):
        (artifact_url, file_name) = artifactory._get_snapshot_url(artifactory_url='http://artifactory.example.com/artifactory', repository='libs-snapshots', group_id='com.company.sampleapp.web-module', artifact_id='web', version='1.1_RC8-SNAPSHOT', packaging='war', classifier='test', headers={})
        assert artifact_url == 'http://artifactory.example.com/artifactory/libs-snapshots/com/company/sampleapp/web-module/web/1.1_RC8-SNAPSHOT/web-1.1_RC8-20140418.150212-1-test.war'

def test_get_snapshot_url_without_classifier():
    if False:
        i = 10
        return i + 15
    '\n    test when classifier not set and packaging\n    does not match snapshot_versions in the metadata.\n    '
    with patch('salt.modules.artifactory._get_snapshot_version_metadata_xml', MagicMock(return_value='<?xml version="1.0" encoding="UTF-8"?>\n                <metadata>\n                  <groupId>com.company.sampleapp.web-module</groupId>\n                  <artifactId>web</artifactId>\n                  <version>1.1_RC8-SNAPSHOT</version>\n                  <versioning>\n                    <snapshot>\n                      <timestamp>20140418.150212</timestamp>\n                      <buildNumber>1</buildNumber>\n                    </snapshot>\n                    <lastUpdated>20140623104055</lastUpdated>\n                    <snapshotVersions>\n                      <snapshotVersion>\n                        <extension>pom</extension>\n                        <value>1.1_RC8-20140418.150212-1</value>\n                        <updated>20140418150212</updated>\n                      </snapshotVersion>\n                      <snapshotVersion>\n                        <classifier>test</classifier>\n                        <extension>war</extension>\n                        <value>1.1_RC8-20140418.150212-1</value>\n                        <updated>20140418150212</updated>\n                      </snapshotVersion>\n                    </snapshotVersions>\n                  </versioning>\n                </metadata>\n            ')):
        with pytest.raises(artifactory.ArtifactoryError):
            (artifact_url, file_name) = artifactory._get_snapshot_url(artifactory_url='http://artifactory.example.com/artifactory', repository='libs-snapshots', group_id='com.company.sampleapp.web-module', artifact_id='web', version='1.1_RC8-SNAPSHOT', packaging='war', headers={})

def test_get_latest_snapshot_username_password():
    if False:
        return 10
    with patch('salt.modules.artifactory._get_artifact_metadata', return_value={'latest_version': '1.1'}), patch('salt.modules.artifactory._get_snapshot_url', return_value=('http://artifactory.example.com/artifactory/snapshot', '/path/to/file')), patch('salt.modules.artifactory.__save_artifact', return_value={}) as save_artifact_mock:
        artifactory.get_latest_snapshot(artifactory_url='http://artifactory.example.com/artifactory', repository='libs-snapshots', group_id='com.company.sampleapp.web-module', artifact_id='web', packaging='war', username='user', password='password')
        save_artifact_mock.assert_called_with('http://artifactory.example.com/artifactory/snapshot', '/path/to/file', {'Authorization': 'Basic dXNlcjpwYXNzd29yZA==\n'})

def test_get_snapshot_username_password():
    if False:
        return 10
    with patch('salt.modules.artifactory._get_snapshot_url', return_value=('http://artifactory.example.com/artifactory/snapshot', '/path/to/file')), patch('salt.modules.artifactory.__save_artifact', return_value={}) as save_artifact_mock:
        artifactory.get_snapshot(artifactory_url='http://artifactory.example.com/artifactory', repository='libs-snapshots', group_id='com.company.sampleapp.web-module', artifact_id='web', packaging='war', version='1.1', username='user', password='password')
        save_artifact_mock.assert_called_with('http://artifactory.example.com/artifactory/snapshot', '/path/to/file', {'Authorization': 'Basic dXNlcjpwYXNzd29yZA==\n'})

def test_get_latest_release_username_password():
    if False:
        print('Hello World!')
    with patch('salt.modules.artifactory.__find_latest_version', return_value='1.1'), patch('salt.modules.artifactory._get_release_url', return_value=('http://artifactory.example.com/artifactory/release', '/path/to/file')), patch('salt.modules.artifactory.__save_artifact', return_value={}) as save_artifact_mock:
        artifactory.get_latest_release(artifactory_url='http://artifactory.example.com/artifactory', repository='libs-snapshots', group_id='com.company.sampleapp.web-module', artifact_id='web', packaging='war', username='user', password='password')
        save_artifact_mock.assert_called_with('http://artifactory.example.com/artifactory/release', '/path/to/file', {'Authorization': 'Basic dXNlcjpwYXNzd29yZA==\n'})

def test_get_release_username_password():
    if False:
        i = 10
        return i + 15
    with patch('salt.modules.artifactory._get_release_url', return_value=('http://artifactory.example.com/artifactory/release', '/path/to/file')), patch('salt.modules.artifactory.__save_artifact', return_value={}) as save_artifact_mock:
        artifactory.get_release(artifactory_url='http://artifactory.example.com/artifactory', repository='libs-snapshots', group_id='com.company.sampleapp.web-module', artifact_id='web', packaging='war', version='1.1', username='user', password='password')
        save_artifact_mock.assert_called_with('http://artifactory.example.com/artifactory/release', '/path/to/file', {'Authorization': 'Basic dXNlcjpwYXNzd29yZA==\n'})

def test_save_artifact_file_exists_checksum_equal():
    if False:
        while True:
            i = 10
    artifact_url = 'http://artifactory.example.com/artifactory/artifact'
    target_file = '/path/to/file'
    sum_str = '0123456789abcdef0123456789abcdef01234567'
    sum_bin = sum_str.encode()
    with patch('os.path.isfile', return_value=True), patch.dict(artifactory.__salt__, {'file.get_hash': MagicMock(return_value=sum_str)}):
        with patch('salt.modules.artifactory.__download', return_value=(True, sum_bin, None)):
            result = getattr(artifactory, '__save_artifact')(artifact_url=artifact_url, target_file=target_file, headers={})
            assert result == {'status': True, 'changes': {}, 'target_file': target_file, 'comment': 'File {} already exists, checksum matches with Artifactory.\nChecksum URL: {}.sha1'.format(target_file, artifact_url)}
        with patch('salt.modules.artifactory.__download', return_value=(True, sum_str, None)):
            result = getattr(artifactory, '__save_artifact')(artifact_url=artifact_url, target_file=target_file, headers={})
            assert result == {'status': True, 'changes': {}, 'target_file': target_file, 'comment': 'File {} already exists, checksum matches with Artifactory.\nChecksum URL: {}.sha1'.format(target_file, artifact_url)}