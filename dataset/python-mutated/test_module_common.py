from __future__ import annotations
import os.path
import pytest
import ansible.errors
from ansible.executor import module_common as amc
from ansible.executor.interpreter_discovery import InterpreterDiscoveryRequiredError

class TestStripComments:

    def test_no_changes(self):
        if False:
            i = 10
            return i + 15
        no_comments = u'def some_code():\n    return False'
        assert amc._strip_comments(no_comments) == no_comments

    def test_all_comments(self):
        if False:
            i = 10
            return i + 15
        all_comments = u'# This is a test\n            # Being as it is\n            # To be\n            '
        assert amc._strip_comments(all_comments) == u''

    def test_all_whitespace(self):
        if False:
            for i in range(10):
                print('nop')
        all_whitespace = '\n              \n\n                \n\t\t\r\n\n            '
        assert amc._strip_comments(all_whitespace) == u''

    def test_somewhat_normal(self):
        if False:
            for i in range(10):
                print('nop')
        mixed = u"#!/usr/bin/python\n\n# here we go\ndef test(arg):\n    # this is a thing\n    thing = '# test'\n    return thing\n# End\n"
        mixed_results = u"def test(arg):\n    thing = '# test'\n    return thing"
        assert amc._strip_comments(mixed) == mixed_results

class TestSlurp:

    def test_slurp_nonexistent(self, mocker):
        if False:
            for i in range(10):
                print('nop')
        mocker.patch('os.path.exists', side_effect=lambda x: False)
        with pytest.raises(ansible.errors.AnsibleError):
            amc._slurp('no_file')

    def test_slurp_file(self, mocker):
        if False:
            print('Hello World!')
        mocker.patch('os.path.exists', side_effect=lambda x: True)
        m = mocker.mock_open(read_data='This is a test')
        mocker.patch('builtins.open', m)
        assert amc._slurp('some_file') == 'This is a test'

    def test_slurp_file_with_newlines(self, mocker):
        if False:
            return 10
        mocker.patch('os.path.exists', side_effect=lambda x: True)
        m = mocker.mock_open(read_data='#!/usr/bin/python\ndef test(args):\nprint("hi")\n')
        mocker.patch('builtins.open', m)
        assert amc._slurp('some_file') == '#!/usr/bin/python\ndef test(args):\nprint("hi")\n'

class TestGetShebang:
    """Note: We may want to change the API of this function in the future.  It isn't a great API"""

    def test_no_interpreter_set(self, templar):
        if False:
            i = 10
            return i + 15
        with pytest.raises(InterpreterDiscoveryRequiredError):
            amc._get_shebang(u'/usr/bin/python', {}, templar)

    def test_python_interpreter(self, templar):
        if False:
            while True:
                i = 10
        assert amc._get_shebang(u'/usr/bin/python3.8', {}, templar) == ('#!/usr/bin/python3.8', u'/usr/bin/python3.8')

    def test_non_python_interpreter(self, templar):
        if False:
            print('Hello World!')
        assert amc._get_shebang(u'/usr/bin/ruby', {}, templar) == ('#!/usr/bin/ruby', u'/usr/bin/ruby')

    def test_interpreter_set_in_task_vars(self, templar):
        if False:
            return 10
        assert amc._get_shebang(u'/usr/bin/python', {u'ansible_python_interpreter': u'/usr/bin/pypy'}, templar) == (u'#!/usr/bin/pypy', u'/usr/bin/pypy')

    def test_non_python_interpreter_in_task_vars(self, templar):
        if False:
            return 10
        assert amc._get_shebang(u'/usr/bin/ruby', {u'ansible_ruby_interpreter': u'/usr/local/bin/ruby'}, templar) == (u'#!/usr/local/bin/ruby', u'/usr/local/bin/ruby')

    def test_with_args(self, templar):
        if False:
            i = 10
            return i + 15
        assert amc._get_shebang(u'/usr/bin/python', {u'ansible_python_interpreter': u'/usr/bin/python3'}, templar, args=('-tt', '-OO')) == (u'#!/usr/bin/python3 -tt -OO', u'/usr/bin/python3')

    def test_python_via_env(self, templar):
        if False:
            while True:
                i = 10
        assert amc._get_shebang(u'/usr/bin/python', {u'ansible_python_interpreter': u'/usr/bin/env python'}, templar) == (u'#!/usr/bin/env python', u'/usr/bin/env python')

class TestDetectionRegexes:
    ANSIBLE_MODULE_UTIL_STRINGS = (b'import ansible_collections.my_ns.my_col.plugins.module_utils.my_util', b'from ansible_collections.my_ns.my_col.plugins.module_utils import my_util', b'from ansible_collections.my_ns.my_col.plugins.module_utils.my_util import my_func', b'import ansible.module_utils.basic', b'from ansible.module_utils import basic', b'from ansible.module_utils.basic import AnsibleModule', b'from ..module_utils import basic', b'from .. module_utils import basic', b'from ....module_utils import basic', b'from ..module_utils.basic import AnsibleModule')
    NOT_ANSIBLE_MODULE_UTIL_STRINGS = (b'from ansible import release', b'from ..release import __version__', b'from .. import release', b'from ansible.modules.system import ping', b'from ansible_collecitons.my_ns.my_col.plugins.modules import function')
    OFFSET = os.path.dirname(os.path.dirname(amc.__file__))
    CORE_PATHS = (('%s/modules/from_role.py' % OFFSET, 'ansible/modules/from_role'), ('%s/modules/system/ping.py' % OFFSET, 'ansible/modules/system/ping'), ('%s/modules/cloud/amazon/s3.py' % OFFSET, 'ansible/modules/cloud/amazon/s3'))
    COLLECTION_PATHS = (('/root/ansible_collections/ns/col/plugins/modules/ping.py', 'ansible_collections/ns/col/plugins/modules/ping'), ('/root/ansible_collections/ns/col/plugins/modules/subdir/ping.py', 'ansible_collections/ns/col/plugins/modules/subdir/ping'))

    @pytest.mark.parametrize('testcase', ANSIBLE_MODULE_UTIL_STRINGS)
    def test_detect_new_style_python_module_re(self, testcase):
        if False:
            for i in range(10):
                print('nop')
        assert amc.NEW_STYLE_PYTHON_MODULE_RE.search(testcase)

    @pytest.mark.parametrize('testcase', NOT_ANSIBLE_MODULE_UTIL_STRINGS)
    def test_no_detect_new_style_python_module_re(self, testcase):
        if False:
            return 10
        assert not amc.NEW_STYLE_PYTHON_MODULE_RE.search(testcase)

    @pytest.mark.parametrize('testcase, result', CORE_PATHS)
    def test_detect_core_library_path_re(self, testcase, result):
        if False:
            i = 10
            return i + 15
        assert amc.CORE_LIBRARY_PATH_RE.search(testcase).group('path') == result

    @pytest.mark.parametrize('testcase', (p[0] for p in COLLECTION_PATHS))
    def test_no_detect_core_library_path_re(self, testcase):
        if False:
            return 10
        assert not amc.CORE_LIBRARY_PATH_RE.search(testcase)

    @pytest.mark.parametrize('testcase, result', COLLECTION_PATHS)
    def test_detect_collection_path_re(self, testcase, result):
        if False:
            return 10
        assert amc.COLLECTION_PATH_RE.search(testcase).group('path') == result

    @pytest.mark.parametrize('testcase', (p[0] for p in CORE_PATHS))
    def test_no_detect_collection_path_re(self, testcase):
        if False:
            while True:
                i = 10
        assert not amc.COLLECTION_PATH_RE.search(testcase)