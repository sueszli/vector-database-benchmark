import mock
from st2tests.base import BaseActionTestCase
from pack_mgmt.get_pack_warnings import GetPackWarnings
PACK_METADATA = {'py23': {'version': '0.4.0', 'name': 'py23', 'repo_url': 'https://github.com/StackStorm-Exchange/stackstorm-no_warnings', 'author': 'st2-dev', 'keywords': ['some', 'search', 'another', 'terms'], 'email': 'info@stackstorm.com', 'description': 'st2 pack to test package management pipeline', 'python_versions': ['2', '3']}, 'py3': {'version': '0.4.0', 'name': 'py3', 'repo_url': 'https://github.com/StackStorm-Exchange/stackstorm-no_warnings', 'author': 'st2-dev', 'keywords': ['some', 'search', 'another', 'terms'], 'email': 'info@stackstorm.com', 'description': 'st2 pack to test package management pipeline', 'python_versions': ['3']}, 'pynone': {'version': '0.4.0', 'name': 'pynone', 'repo_url': 'https://github.com/StackStorm-Exchange/stackstorm-no_warnings', 'author': 'st2-dev', 'keywords': ['some', 'search', 'another', 'terms'], 'email': 'info@stackstorm.com', 'description': 'st2 pack to test package management pipeline'}, 'py2': {'version': '0.5.0', 'name': 'py2', 'repo_url': 'https://github.com/StackStorm-Exchange/stackstorm-test2', 'author': 'stanley', 'keywords': ['some', 'special', 'terms'], 'email': 'info@stackstorm.com', 'description': 'another st2 pack to test package management pipeline', 'python_versions': ['2']}, 'py22': {'version': '0.5.0', 'name': 'py22', 'repo_url': 'https://github.com/StackStorm-Exchange/stackstorm-test2', 'author': 'stanley', 'keywords': ['some', 'special', 'terms'], 'email': 'info@stackstorm.com', 'description': 'another st2 pack to test package management pipeline', 'python_versions': ['2']}}

def mock_get_pack_basepath(pack):
    if False:
        while True:
            i = 10
    '\n    Mock get_pack_basepath function which just returns pack n ame\n    '
    return pack

def mock_get_pack_metadata(pack_dir):
    if False:
        return 10
    '\n    Mock get_pack_version function which return mocked pack version\n    '
    metadata = {}
    if pack_dir in PACK_METADATA:
        metadata = PACK_METADATA[pack_dir]
    return metadata

@mock.patch('pack_mgmt.get_pack_warnings.get_pack_base_path', mock_get_pack_basepath)
@mock.patch('pack_mgmt.get_pack_warnings.get_pack_metadata', mock_get_pack_metadata)
class GetPackWarningsTestCase(BaseActionTestCase):
    action_cls = GetPackWarnings

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(GetPackWarningsTestCase, self).setUp()

    def test_run_get_pack_warnings_py3_pack(self):
        if False:
            return 10
        action = self.get_action_instance()
        packs_status = {'py3': 'Success.'}
        result = action.run(packs_status=packs_status)
        self.assertEqual(result['warning_list'], [])

    def test_run_get_pack_warnings_py2_pack(self):
        if False:
            for i in range(10):
                print('nop')
        action = self.get_action_instance()
        packs_status = {'py2': 'Success.'}
        result = action.run(packs_status=packs_status)
        self.assertEqual(len(result['warning_list']), 1)
        warning = result['warning_list'][0]
        self.assertTrue('DEPRECATION WARNING' in warning)
        self.assertTrue('Pack py2 only supports Python 2' in warning)

    def test_run_get_pack_warnings_py23_pack(self):
        if False:
            for i in range(10):
                print('nop')
        action = self.get_action_instance()
        packs_status = {'py23': 'Success.'}
        result = action.run(packs_status=packs_status)
        self.assertEqual(result['warning_list'], [])

    def test_run_get_pack_warnings_pynone_pack(self):
        if False:
            print('Hello World!')
        action = self.get_action_instance()
        packs_status = {'pynone': 'Success.'}
        result = action.run(packs_status=packs_status)
        self.assertEqual(result['warning_list'], [])

    def test_run_get_pack_warnings_multiple_pack(self):
        if False:
            i = 10
            return i + 15
        action = self.get_action_instance()
        packs_status = {'py2': 'Success.', 'py23': 'Success.', 'py22': 'Success.'}
        result = action.run(packs_status=packs_status)
        self.assertEqual(len(result['warning_list']), 2)
        warning0 = result['warning_list'][0]
        warning1 = result['warning_list'][1]
        self.assertTrue('DEPRECATION WARNING' in warning0)
        self.assertTrue('DEPRECATION WARNING' in warning1)
        self.assertTrue('Pack py2 only supports Python 2' in warning0 and 'Pack py22 only supports Python 2' in warning1 or ('Pack py22 only supports Python 2' in warning0 and 'Pack py2 only supports Python 2' in warning1))