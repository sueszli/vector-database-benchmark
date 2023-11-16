from os import path
from unittest import TestCase
from unittest.mock import Mock, patch
from mycroft.skills.skill_loader import SkillLoader
from mycroft.skills.skill_manager import SkillManager, UploadQueue
from ..base import MycroftUnitTestBase
from ..mocks import mock_msm

class TestUploadQueue(TestCase):

    def test_upload_queue_create(self):
        if False:
            while True:
                i = 10
        queue = UploadQueue()
        self.assertFalse(queue.started)
        queue.start()
        self.assertTrue(queue.started)

    def test_upload_queue_use(self):
        if False:
            for i in range(10):
                print('nop')
        queue = UploadQueue()
        queue.start()
        specific_loader = Mock(spec=SkillLoader, instance=Mock())
        loaders = [Mock(), specific_loader, Mock(), Mock()]
        for (i, l) in enumerate(loaders):
            queue.put(l)
            self.assertEqual(len(queue), i + 1)
        queue.put(specific_loader)
        self.assertEqual(len(queue), len(loaders))
        queue.send()
        self.assertEqual(len(queue), 0)

    def test_upload_queue_preloaded(self):
        if False:
            while True:
                i = 10
        queue = UploadQueue()
        loaders = [Mock(), Mock(), Mock(), Mock()]
        for (i, l) in enumerate(loaders):
            queue.put(l)
            self.assertEqual(len(queue), i + 1)
        queue.start()
        self.assertEqual(len(queue), 0)
        for l in loaders:
            l.instance.settings_meta.upload.assert_called_once_with()

class TestSkillManager(MycroftUnitTestBase):
    mock_package = 'mycroft.skills.skill_manager.'
    use_msm_mock = True

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self._mock_skill_updater()
        self._mock_skill_settings_downloader()
        self.skill_manager = SkillManager(self.message_bus_mock)
        self._mock_skill_loader_instance()

    def _mock_msm(self):
        if False:
            while True:
                i = 10
        if self.use_msm_mock:
            msm_patch = patch(self.mock_package + 'msm_creator')
            self.addCleanup(msm_patch.stop)
            self.create_msm_mock = msm_patch.start()
            self.msm_mock = mock_msm(str(self.temp_dir))
            self.create_msm_mock.return_value = self.msm_mock

    def _mock_skill_settings_downloader(self):
        if False:
            print('Hello World!')
        settings_download_patch = patch(self.mock_package + 'SkillSettingsDownloader', spec=True)
        self.addCleanup(settings_download_patch.stop)
        self.settings_download_mock = settings_download_patch.start()

    def _mock_skill_updater(self):
        if False:
            for i in range(10):
                print('nop')
        skill_updater_patch = patch(self.mock_package + 'SkillUpdater', spec=True)
        self.addCleanup(skill_updater_patch.stop)
        self.skill_updater_mock = skill_updater_patch.start()

    def _mock_skill_loader_instance(self):
        if False:
            for i in range(10):
                print('nop')
        self.skill_dir = self.temp_dir.joinpath('test_skill')
        self.skill_loader_mock = Mock(spec=SkillLoader)
        self.skill_loader_mock.instance = Mock()
        self.skill_loader_mock.instance.default_shutdown = Mock()
        self.skill_loader_mock.instance.converse = Mock()
        self.skill_loader_mock.instance.converse.return_value = True
        self.skill_loader_mock.skill_id = 'test_skill'
        self.skill_manager.skill_loaders = {str(self.skill_dir): self.skill_loader_mock}

    def test_instantiate(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.skill_manager.config['data_dir'], str(self.temp_dir))
        expected_result = ['skill.converse.request', 'mycroft.internet.connected', 'skillmanager.update', 'skillmanager.list', 'skillmanager.deactivate', 'skillmanager.keep', 'skillmanager.activate', 'mycroft.paired', 'mycroft.skills.settings.update']
        self.assertListEqual(expected_result, self.message_bus_mock.event_handlers)

    def test_remove_git_locks(self):
        if False:
            return 10
        git_dir = self.temp_dir.joinpath('foo/.git')
        git_dir.mkdir(parents=True)
        git_lock_file_path = str(git_dir.joinpath('index.lock'))
        with open(git_lock_file_path, 'w') as git_lock_file:
            git_lock_file.write('foo')
        self.skill_manager._remove_git_locks()
        self.assertFalse(path.exists(git_lock_file_path))

    def test_load_priority(self):
        if False:
            i = 10
            return i + 15
        load_mock = Mock()
        self.skill_manager._load_skill = load_mock
        (skill, self.skill_manager.msm.list) = self._build_mock_msm_skill_list()
        self.msm_mock.all_skills = [skill]
        self.skill_manager.load_priority()
        self.assertFalse(skill.install.called)
        load_mock.assert_called_once_with(skill.path)

    def test_install_priority(self):
        if False:
            i = 10
            return i + 15
        load_mock = Mock()
        self.skill_manager._load_skill = load_mock
        (skill, self.skill_manager.msm.list) = self._build_mock_msm_skill_list()
        skill.is_local = False
        self.msm_mock.all_skills = [skill]
        self.skill_manager.load_priority()
        self.assertTrue(self.msm_mock.install.called)
        load_mock.assert_called_once_with(skill.path)

    def test_priority_skill_not_recognized(self):
        if False:
            print('Hello World!')
        load_or_reload_mock = Mock()
        self.skill_manager._load_or_reload_skill = load_or_reload_mock
        (skill, self.skill_manager.msm.list) = self._build_mock_msm_skill_list()
        skill.name = 'barfoo'
        self.skill_manager.load_priority()
        self.assertFalse(skill.install.called)
        self.assertFalse(load_or_reload_mock.called)

    def test_priority_skill_install_failed(self):
        if False:
            while True:
                i = 10
        load_or_reload_mock = Mock()
        self.skill_manager._load_or_reload_skill = load_or_reload_mock
        (skill, self.skill_manager.msm.list) = self._build_mock_msm_skill_list()
        skill.is_local = False
        skill.install.side_effect = ValueError
        self.skill_manager.load_priority()
        self.assertRaises(ValueError, skill.install)
        self.assertFalse(load_or_reload_mock.called)

    def _build_mock_msm_skill_list(self):
        if False:
            while True:
                i = 10
        skill = Mock()
        skill.name = 'foobar'
        skill.is_local = True
        skill.install = Mock()
        skill.update = Mock()
        skill.update_deps = Mock()
        skill.path = str(self.temp_dir.joinpath('foobar'))
        skill_list_func = Mock(return_value=[skill])
        return (skill, skill_list_func)

    def test_no_skill_in_skill_dir(self):
        if False:
            print('Hello World!')
        self.skill_dir.mkdir(parents=True)
        skill_directories = self.skill_manager._get_skill_directories()
        self.assertListEqual([], skill_directories)

    def test_get_skill_directories(self):
        if False:
            i = 10
            return i + 15
        self.skill_dir.mkdir(parents=True)
        self.skill_dir.joinpath('__init__.py').touch()
        skill_directories = self.skill_manager._get_skill_directories()
        self.assertListEqual([str(self.skill_dir)], skill_directories)

    def test_unload_removed_skills(self):
        if False:
            i = 10
            return i + 15
        self.skill_manager._unload_removed_skills()
        self.assertDictEqual({}, self.skill_manager.skill_loaders)
        self.skill_loader_mock.unload.assert_called_once_with()

    def test_send_skill_list(self):
        if False:
            i = 10
            return i + 15
        self.skill_loader_mock.active = True
        self.skill_loader_mock.loaded = True
        self.skill_manager.send_skill_list(None)
        self.assertListEqual(['mycroft.skills.list'], self.message_bus_mock.message_types)
        message_data = self.message_bus_mock.message_data[0]
        self.assertIn('test_skill', message_data.keys())
        skill_data = message_data['test_skill']
        self.assertDictEqual(dict(active=True, id='test_skill'), skill_data)

    def test_stop(self):
        if False:
            while True:
                i = 10
        self.skill_manager.stop()
        self.assertTrue(self.skill_manager._stop_event.is_set())
        instance = self.skill_loader_mock.instance
        instance.default_shutdown.assert_called_once_with()

    def test_handle_converse_request(self):
        if False:
            return 10
        message = Mock()
        message.data = dict(skill_id='test_skill', utterances=['hey you'], lang='en-US')
        self.skill_loader_mock.loaded = True
        converse_response_mock = Mock()
        self.skill_manager._emit_converse_response = converse_response_mock
        converse_error_mock = Mock()
        self.skill_manager._emit_converse_error = converse_error_mock
        self.skill_manager.handle_converse_request(message)
        converse_response_mock.assert_called_once_with(True, message, self.skill_loader_mock)
        converse_error_mock.assert_not_called()

    def test_converse_request_missing_skill(self):
        if False:
            for i in range(10):
                print('nop')
        message = Mock()
        message.data = dict(skill_id='foo')
        self.skill_loader_mock.loaded = True
        converse_response_mock = Mock()
        self.skill_manager._emit_converse_response = converse_response_mock
        converse_error_mock = Mock()
        self.skill_manager._emit_converse_error = converse_error_mock
        self.skill_manager.handle_converse_request(message)
        converse_response_mock.assert_not_called()
        converse_error_mock.assert_called_once_with(message, 'foo', 'skill id does not exist')

    def test_converse_request_skill_not_loaded(self):
        if False:
            return 10
        message = Mock()
        message.data = dict(skill_id='test_skill')
        self.skill_loader_mock.loaded = False
        converse_response_mock = Mock()
        self.skill_manager._emit_converse_response = converse_response_mock
        converse_error_mock = Mock()
        self.skill_manager._emit_converse_error = converse_error_mock
        self.skill_manager.handle_converse_request(message)
        converse_response_mock.assert_not_called()
        converse_error_mock.assert_called_once_with(message, 'test_skill', 'converse requested but skill not loaded')

    def test_schedule_now(self):
        if False:
            print('Hello World!')
        with patch(self.mock_package + 'time') as time_mock:
            time_mock.return_value = 100
            self.skill_updater_mock.next_download = 0
            self.skill_manager.schedule_now(None)
        self.assertEqual(99, self.skill_manager.skill_updater.next_download)

    def test_handle_paired(self):
        if False:
            return 10
        self.skill_updater_mock.next_download = 0
        self.skill_manager.handle_paired(None)
        updater = self.skill_manager.skill_updater
        updater.post_manifest.assert_called_once_with(reload_skills_manifest=True)

    def test_deactivate_skill(self):
        if False:
            i = 10
            return i + 15
        message = Mock()
        message.data = dict(skill='test_skill')
        self.skill_manager.deactivate_skill(message)
        self.skill_loader_mock.deactivate.assert_called_once_with()

    def test_deactivate_except(self):
        if False:
            while True:
                i = 10
        message = Mock()
        message.data = dict(skill='test_skill')
        self.skill_loader_mock.active = True
        foo_skill_loader = Mock(spec=SkillLoader)
        foo_skill_loader.skill_id = 'foo'
        foo2_skill_loader = Mock(spec=SkillLoader)
        foo2_skill_loader.skill_id = 'foo2'
        test_skill_loader = Mock(spec=SkillLoader)
        test_skill_loader.skill_id = 'test_skill'
        self.skill_manager.skill_loaders['foo'] = foo_skill_loader
        self.skill_manager.skill_loaders['foo2'] = foo2_skill_loader
        self.skill_manager.skill_loaders['test_skill'] = test_skill_loader
        self.skill_manager.deactivate_except(message)
        foo_skill_loader.deactivate.assert_called_once_with()
        foo2_skill_loader.deactivate.assert_called_once_with()
        self.assertFalse(test_skill_loader.deactivate.called)

    def test_activate_skill(self):
        if False:
            while True:
                i = 10
        message = Mock()
        message.data = dict(skill='test_skill')
        test_skill_loader = Mock(spec=SkillLoader)
        test_skill_loader.skill_id = 'test_skill'
        test_skill_loader.active = False
        self.skill_manager.skill_loaders = {}
        self.skill_manager.skill_loaders['test_skill'] = test_skill_loader
        self.skill_manager.activate_skill(message)
        test_skill_loader.activate.assert_called_once_with()

    def test_load_on_startup(self):
        if False:
            i = 10
            return i + 15
        self.skill_dir.mkdir(parents=True)
        self.skill_dir.joinpath('__init__.py').touch()
        patch_obj = self.mock_package + 'SkillLoader'
        self.skill_manager.skill_loaders = {}
        with patch(patch_obj, spec=True) as loader_mock:
            self.skill_manager._load_on_startup()
            loader_mock.return_value.load.assert_called_once_with()
            self.assertEqual(loader_mock.return_value, self.skill_manager.skill_loaders[str(self.skill_dir)])
        self.assertListEqual(['mycroft.skills.initialized'], self.message_bus_mock.message_types)

    def test_load_newly_installed_skill(self):
        if False:
            while True:
                i = 10
        self.skill_dir.mkdir(parents=True)
        self.skill_dir.joinpath('__init__.py').touch()
        patch_obj = self.mock_package + 'SkillLoader'
        self.skill_manager.skill_loaders = {}
        with patch(patch_obj, spec=True) as loader_mock:
            self.skill_manager._load_new_skills()
            loader_mock.return_value.load.assert_called_once_with()
            self.assertEqual(loader_mock.return_value, self.skill_manager.skill_loaders[str(self.skill_dir)])

    def test_reload_modified(self):
        if False:
            return 10
        self.skill_dir.mkdir(parents=True)
        self.skill_dir.joinpath('__init__.py').touch()
        self.skill_loader_mock.reload_needed.return_value = True
        self.skill_manager._reload_modified_skills()
        self.skill_loader_mock.reload.assert_called_once_with()
        self.assertEqual(self.skill_loader_mock, self.skill_manager.skill_loaders[str(self.skill_dir)])

    def test_update_skills(self):
        if False:
            return 10
        updater_mock = Mock()
        updater_mock.update_skills = Mock()
        updater_mock.next_download = 0
        self.skill_manager.skill_updater = updater_mock
        self.skill_manager._update_skills()
        updater_mock.update_skills.assert_called_once_with()