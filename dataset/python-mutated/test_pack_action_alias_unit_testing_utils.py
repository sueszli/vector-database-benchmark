from __future__ import absolute_import
import mock
from st2tests.base import BaseActionAliasTestCase
from st2tests.fixtures.packs.pack_dir_name_doesnt_match_ref.fixture import PACK_NAME as PACK_NAME_NOT_THE_SAME_AS_DIR_NAME, PACK_PATH as PACK_PATH_1
from st2common.exceptions.content import ParseException
from st2common.models.db.actionalias import ActionAliasDB
__all__ = ['PackActionAliasUnitTestUtils']

class PackActionAliasUnitTestUtils(BaseActionAliasTestCase):
    action_alias_name = 'mock'
    mock_get_action_alias_db_by_name = True

    def test_assertExtractedParametersMatch_success(self):
        if False:
            i = 10
            return i + 15
        format_string = self.action_alias_db.formats[0]
        command = 'show last 3 metrics for my.host'
        expected_parameters = {'count': '3', 'server': 'my.host'}
        self.assertExtractedParametersMatch(format_string=format_string, command=command, parameters=expected_parameters)
        format_string = self.action_alias_db.formats[0]
        command = 'show last 10 metrics for my.host.example'
        expected_parameters = {'count': '10', 'server': 'my.host.example'}
        self.assertExtractedParametersMatch(format_string=format_string, command=command, parameters=expected_parameters)

    def test_assertExtractedParametersMatch_command_doesnt_match_format_string(self):
        if False:
            return 10
        format_string = self.action_alias_db.formats[0]
        command = 'show last foo'
        expected_parameters = {}
        expected_msg = 'Command "show last foo" doesn\'t match format string "show last {{count}} metrics for {{server}}"'
        self.assertRaisesRegexp(ParseException, expected_msg, self.assertExtractedParametersMatch, format_string=format_string, command=command, parameters=expected_parameters)

    def test_assertCommandMatchesExactlyOneFormatString(self):
        if False:
            return 10
        format_strings = ['foo bar {{bar}}', 'foo bar {{baz}} baz']
        command = 'foo bar a test=1'
        self.assertCommandMatchesExactlyOneFormatString(format_strings=format_strings, command=command)
        format_strings = ['foo bar {{bar}}', 'foo bar {{baz}}']
        command = 'foo bar a test=1'
        expected_msg = 'Command "foo bar a test=1" matched multiple format strings: foo bar {{bar}}, foo bar {{baz}}'
        self.assertRaisesRegexp(AssertionError, expected_msg, self.assertCommandMatchesExactlyOneFormatString, format_strings=format_strings, command=command)
        format_strings = ['foo bar {{bar}}', 'foo bar {{baz}}']
        command = 'does not match foo'
        expected_msg = 'Command "does not match foo" didn\'t match any of the provided format strings'
        self.assertRaisesRegexp(AssertionError, expected_msg, self.assertCommandMatchesExactlyOneFormatString, format_strings=format_strings, command=command)

    @mock.patch.object(BaseActionAliasTestCase, '_get_base_pack_path', mock.Mock(return_value=PACK_PATH_1))
    def test_base_class_works_when_pack_directory_name_doesnt_match_pack_name(self):
        if False:
            i = 10
            return i + 15
        self.mock_get_action_alias_db_by_name = False
        action_alias_db = self._get_action_alias_db_by_name(name='alias1')
        self.assertEqual(action_alias_db.name, 'alias1')
        self.assertEqual(action_alias_db.pack, PACK_NAME_NOT_THE_SAME_AS_DIR_NAME)

    def _get_action_alias_db_by_name(self, name):
        if False:
            for i in range(10):
                print('nop')
        if not self.mock_get_action_alias_db_by_name:
            return super(PackActionAliasUnitTestUtils, self)._get_action_alias_db_by_name(name)
        values = {'name': self.action_alias_name, 'pack': 'mock', 'formats': ['show last {{count}} metrics for {{server}}']}
        action_alias_db = ActionAliasDB(**values)
        return action_alias_db