import json
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import call, Mock, patch
from mycroft.skills.settings import get_local_settings, save_settings, SkillSettingsDownloader, SettingsMetaUploader
from ..base import MycroftUnitTestBase

class TestSettingsMetaUploader(MycroftUnitTestBase):
    use_msm_mock = True
    mock_package = 'mycroft.skills.settings.'

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.uploader = SettingsMetaUploader(str(self.temp_dir), 'test_skill')
        self.uploader.api = Mock()
        self.is_paired_mock = self._mock_is_paired()
        self.timer_mock = self._mock_timer()
        self.skill_metadata = dict(skillMetadata=dict(sections=[dict(name='Test Section', fields=[dict(type='label', label='Test Field')])]))

    def _mock_is_paired(self):
        if False:
            for i in range(10):
                print('nop')
        is_paired_patch = patch(self.mock_package + 'is_paired')
        self.addCleanup(is_paired_patch.stop)
        is_paired_mock = is_paired_patch.start()
        is_paired_mock.return_value = True
        return is_paired_mock

    def _mock_timer(self):
        if False:
            i = 10
            return i + 15
        timer_patch = patch(self.mock_package + 'Timer')
        self.addCleanup(timer_patch.stop)
        timer_mock = timer_patch.start()
        return timer_mock

    def test_not_paired(self):
        if False:
            for i in range(10):
                print('nop')
        self.is_paired_mock.return_value = False
        self.uploader.upload()
        self._check_api_not_called()
        self._check_timer_called()

    @patch('mycroft.skills.settings.DeviceApi')
    def test_no_settingsmeta(self, mock_api):
        if False:
            for i in range(10):
                print('nop')
        api_instance = Mock()
        api_instance.identity.uuid = '42'
        mock_api.return_value = api_instance
        self.uploader.upload()
        self._check_settingsmeta()
        self._check_api_call()
        self._check_timer_not_called()

    @patch('mycroft.skills.settings.DeviceApi')
    def test_failed_upload(self, mock_api):
        if False:
            i = 10
            return i + 15
        'The API call to upload the settingsmeta fails.\n\n        This will cause a timer to be generated to retry the update.\n        '
        api_instance = Mock()
        api_instance.identity.uuid = '42'
        api_instance.upload_skill_metadata = Mock(side_effect=ValueError)
        mock_api.return_value = api_instance
        self.uploader.upload()
        self._check_settingsmeta()
        self._check_api_call()
        self._check_timer_called()

    @patch('mycroft.skills.settings.DeviceApi')
    def test_json_settingsmeta(self, mock_api):
        if False:
            print('Hello World!')
        api_instance = Mock()
        api_instance.identity.uuid = '42'
        mock_api.return_value = api_instance
        json_path = str(self.temp_dir.joinpath('settingsmeta.json'))
        with open(json_path, 'w') as json_file:
            json.dump(self.skill_metadata, json_file)
        self.uploader.upload()
        self._check_settingsmeta(self.skill_metadata)
        self._check_api_call()
        self._check_timer_not_called()

    @patch('mycroft.skills.settings.DeviceApi')
    def test_yaml_settingsmeta(self, mock_api):
        if False:
            return 10
        api_instance = Mock()
        api_instance.identity.uuid = '42'
        mock_api.return_value = api_instance
        skill_metadata = 'skillMetadata:\n  sections:\n    - name: "Test Section"\n      fields:\n      - type: "label"\n        label: "Test Field"'
        yaml_path = str(self.temp_dir.joinpath('settingsmeta.yaml'))
        with open(yaml_path, 'w') as yaml_file:
            yaml_file.write(skill_metadata)
        self.uploader.upload()
        self._check_settingsmeta(self.skill_metadata)
        self._check_api_call()
        self._check_timer_not_called()

    def _check_settingsmeta(self, skill_settings=None):
        if False:
            i = 10
            return i + 15
        expected_settings_meta = dict(skill_gid='test_skill|99.99', display_name='Test Skill')
        if skill_settings is not None:
            expected_settings_meta.update(skill_settings)
        self.assertDictEqual(expected_settings_meta, self.uploader.settings_meta)

    def _check_api_call(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertListEqual([call.upload_skill_metadata(self.uploader.settings_meta)], self.uploader.api.method_calls)

    def _check_api_not_called(self):
        if False:
            return 10
        self.assertListEqual([], self.uploader.api.method_calls)

    def _check_timer_called(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertListEqual([call.start()], self.timer_mock.return_value.method_calls)

    def _check_timer_not_called(self):
        if False:
            print('Hello World!')
        self.assertListEqual([], self.timer_mock.return_value.method_calls)

class TestSettingsDownloader(MycroftUnitTestBase):
    mock_package = 'mycroft.skills.settings.'

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.settings_path = self.temp_dir.joinpath('settings.json')
        self.downloader = SkillSettingsDownloader(self.message_bus_mock)
        self.downloader.api = Mock()
        self.is_paired_mock = self._mock_is_paired()
        self.timer_mock = self._mock_timer()

    def _mock_is_paired(self):
        if False:
            return 10
        is_paired_patch = patch(self.mock_package + 'is_paired')
        self.addCleanup(is_paired_patch.stop)
        is_paired_mock = is_paired_patch.start()
        is_paired_mock.return_value = True
        return is_paired_mock

    def _mock_timer(self):
        if False:
            i = 10
            return i + 15
        timer_patch = patch(self.mock_package + 'Timer')
        self.addCleanup(timer_patch.stop)
        timer_mock = timer_patch.start()
        return timer_mock

    def test_not_paired(self):
        if False:
            return 10
        self.is_paired_mock.return_value = False
        self.downloader.download()
        self._check_api_not_called()
        self._check_timer_called()

    def test_settings_not_changed(self):
        if False:
            i = 10
            return i + 15
        test_skill_settings = {'test_skill|99.99': {'test_setting': 'test_value'}}
        self.downloader.last_download_result = test_skill_settings
        self.downloader.api.get_skill_settings = Mock(return_value=test_skill_settings)
        self.downloader.download()
        self._check_api_called()
        self._check_timer_called()
        self._check_no_message_bus_events()

    def test_settings_changed(self):
        if False:
            i = 10
            return i + 15
        local_skill_settings = {'test_skill|99.99': {'test_setting': 'test_value'}}
        remote_skill_settings = {'test_skill|99.99': {'test_setting': 'foo'}}
        self.downloader.last_download_result = local_skill_settings
        self.downloader.api.get_skill_settings = Mock(return_value=remote_skill_settings)
        self.downloader.download()
        self._check_api_called()
        self._check_timer_called()
        self._check_message_bus_events(remote_skill_settings)

    def test_download_failed(self):
        if False:
            print('Hello World!')
        self.downloader.api.get_skill_settings = Mock(side_effect=ValueError)
        pre_download_local_settings = {'test_skill|99.99': {'test_setting': 'test_value'}}
        self.downloader.last_download_result = pre_download_local_settings
        self.downloader.download()
        self._check_api_called()
        self._check_timer_called()
        self._check_no_message_bus_events()
        self.assertEqual(pre_download_local_settings, self.downloader.last_download_result)

    def test_stop_downloading(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure that the timer is cancelled and the continue flag is lowered.\n        '
        self.is_paired_mock.return_value = False
        self.downloader.download()
        self.downloader.stop_downloading()
        self.assertFalse(self.downloader.continue_downloading)
        self.assertTrue(self.downloader.download_timer.cancel.called_once_with())

    def _check_api_called(self):
        if False:
            while True:
                i = 10
        self.assertListEqual([call.get_skill_settings()], self.downloader.api.method_calls)

    def _check_api_not_called(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertListEqual([], self.downloader.api.method_calls)

    def _check_timer_called(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertListEqual([call.start()], self.timer_mock.return_value.method_calls)

    def _check_no_message_bus_events(self):
        if False:
            return 10
        self.assertListEqual(self.message_bus_mock.message_types, [])
        self.assertListEqual(self.message_bus_mock.message_data, [])

    def _check_message_bus_events(self, remote_skill_settings):
        if False:
            while True:
                i = 10
        self.assertListEqual(['mycroft.skills.settings.changed'], self.message_bus_mock.message_types)
        self.assertListEqual([remote_skill_settings], self.message_bus_mock.message_data)

class TestSettings(TestCase):

    def setUp(self) -> None:
        if False:
            return 10
        temp_dir = tempfile.mkdtemp()
        self.temp_dir = Path(temp_dir)
        self.skill_mock = Mock()
        self.skill_mock.root_dir = str(self.temp_dir)
        self.skill_mock.name = 'test_skill'

    def test_empty_settings(self):
        if False:
            while True:
                i = 10
        settings = get_local_settings(self.skill_mock.root_dir, self.skill_mock.name)
        self.assertDictEqual(settings, {})

    def test_settings_file_exists(self):
        if False:
            return 10
        settings_path = str(self.temp_dir.joinpath('settings.json'))
        with open(settings_path, 'w') as settings_file:
            settings_file.write('{"foo": "bar"}\n')
        settings = get_local_settings(self.skill_mock.root_dir, self.skill_mock.name)
        self.assertDictEqual(settings, {'foo': 'bar'})
        self.assertEqual(settings['foo'], 'bar')
        self.assertNotIn('store', settings)
        self.assertIn('foo', settings)

    def test_store_settings(self):
        if False:
            while True:
                i = 10
        settings = dict(foo='bar')
        save_settings(self.skill_mock.root_dir, settings)
        settings_path = str(self.temp_dir.joinpath('settings.json'))
        with open(settings_path) as settings_file:
            file_contents = settings_file.read()
        self.assertEqual(file_contents, '{"foo": "bar"}')