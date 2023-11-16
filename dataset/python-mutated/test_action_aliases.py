from st2tests.base import BaseActionAliasTestCase

class PackGet(BaseActionAliasTestCase):
    action_alias_name = 'pack_get'

    def test_alias_pack_get(self):
        if False:
            for i in range(10):
                print('nop')
        format_string = self.action_alias_db.formats[0]['representation'][0]
        format_strings = self.action_alias_db.get_format_strings()
        command = 'pack get st2'
        expected_parameters = {'pack': 'st2'}
        self.assertExtractedParametersMatch(format_string=format_string, command=command, parameters=expected_parameters)
        self.assertCommandMatchesExactlyOneFormatString(format_strings=format_strings, command=command)

class PackInstall(BaseActionAliasTestCase):
    action_alias_name = 'pack_install'

    def test_alias_pack_install(self):
        if False:
            i = 10
            return i + 15
        format_string = self.action_alias_db.formats[0]['representation'][0]
        command = 'pack install st2'
        expected_parameters = {'packs': 'st2'}
        self.assertExtractedParametersMatch(format_string=format_string, command=command, parameters=expected_parameters)

class PackSearch(BaseActionAliasTestCase):
    action_alias_name = 'pack_search'

    def test_alias_pack_search(self):
        if False:
            print('Hello World!')
        format_string = self.action_alias_db.formats[0]['representation'][0]
        format_strings = self.action_alias_db.get_format_strings()
        command = 'pack search st2'
        expected_parameters = {'query': 'st2'}
        self.assertExtractedParametersMatch(format_string=format_string, command=command, parameters=expected_parameters)
        self.assertCommandMatchesExactlyOneFormatString(format_strings=format_strings, command=command)

class PackShow(BaseActionAliasTestCase):
    action_alias_name = 'pack_show'

    def test_alias_pack_show(self):
        if False:
            while True:
                i = 10
        format_string = self.action_alias_db.formats[0]['representation'][0]
        format_strings = self.action_alias_db.get_format_strings()
        command = 'pack show st2'
        expected_parameters = {'pack': 'st2'}
        self.assertExtractedParametersMatch(format_string=format_string, command=command, parameters=expected_parameters)
        self.assertCommandMatchesExactlyOneFormatString(format_strings=format_strings, command=command)