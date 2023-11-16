from __future__ import absolute_import
import os
import six
from st2common.content.loader import ContentPackLoader
from st2common.content.loader import MetaLoader
from st2common.constants.pack import MANIFEST_FILE_NAME
from st2common.util.pack import get_pack_ref_from_metadata
from st2common.exceptions.content import ParseException
from st2common.bootstrap.aliasesregistrar import AliasesRegistrar
from st2common.models.utils.action_alias_utils import extract_parameters_for_action_alias_db
from st2common.models.utils.action_alias_utils import extract_parameters
from st2tests.pack_resource import BasePackResourceTestCase
__all__ = ['BaseActionAliasTestCase']

class BaseActionAliasTestCase(BasePackResourceTestCase):
    """
    Base class for testing action aliases.
    """
    action_alias_name = None
    action_alias_db = None

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(BaseActionAliasTestCase, self).setUp()
        if not self.action_alias_name:
            raise ValueError('"action_alias_name" class attribute needs to be provided')
        self.action_alias_db = self._get_action_alias_db_by_name(name=self.action_alias_name)

    def assertCommandMatchesExactlyOneFormatString(self, format_strings, command):
        if False:
            for i in range(10):
                print('nop')
        '\n        Assert that the provided command matches exactly one format string from the provided list.\n        '
        matched_format_strings = []
        for format_string in format_strings:
            try:
                extract_parameters(format_str=format_string, param_stream=command)
            except ParseException:
                continue
            matched_format_strings.append(format_string)
        if len(matched_format_strings) == 0:
            msg = 'Command "%s" didn\'t match any of the provided format strings' % command
            raise AssertionError(msg)
        elif len(matched_format_strings) > 1:
            msg = 'Command "%s" matched multiple format strings: %s' % (command, ', '.join(matched_format_strings))
            raise AssertionError(msg)

    def assertExtractedParametersMatch(self, format_string, command, parameters):
        if False:
            for i in range(10):
                print('nop')
        '\n        Assert that the provided command matches the format string.\n\n        In addition to that, also assert that the parameters which have been extracted from the\n        user input (command) also match the provided parameters.\n        '
        extracted_params = extract_parameters_for_action_alias_db(action_alias_db=self.action_alias_db, format_str=format_string, param_stream=command)
        if extracted_params != parameters:
            msg = 'Extracted parameters from command string "%s" against format string "%s" didn\'t match the provided parameters: ' % (command, format_string)
            try:
                self.assertEqual(extracted_params, parameters)
            except AssertionError as e:
                msg += six.text_type(e)
            raise AssertionError(msg)

    def _get_action_alias_db_by_name(self, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieve ActionAlias DB object for the provided alias name.\n        '
        base_pack_path = self._get_base_pack_path()
        pack_yaml_path = os.path.join(base_pack_path, MANIFEST_FILE_NAME)
        if os.path.isfile(pack_yaml_path):
            meta_loader = MetaLoader()
            pack_metadata = meta_loader.load(pack_yaml_path)
            pack = get_pack_ref_from_metadata(metadata=pack_metadata)
        else:
            (_, pack) = os.path.split(base_pack_path)
        pack_loader = ContentPackLoader()
        registrar = AliasesRegistrar(use_pack_cache=False)
        aliases_path = pack_loader.get_content_from_pack(pack_dir=base_pack_path, content_type='aliases')
        aliases = registrar._get_aliases_from_pack(aliases_dir=aliases_path)
        for alias_path in aliases:
            (action_alias_db, altered) = registrar._get_action_alias_db(pack=pack, action_alias=alias_path, ignore_metadata_file_error=True)
            if action_alias_db.name == name:
                return action_alias_db
        raise ValueError('Alias with name "%s" not found' % name)