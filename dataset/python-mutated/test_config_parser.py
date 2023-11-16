from __future__ import absolute_import
from unittest2 import TestCase
from st2common.util.config_parser import ContentPackConfigParser
import st2tests.config as tests_config
from st2tests.fixtures.packs.dummy_pack_1.fixture import PACK_NAME as DUMMY_PACK_1
from st2tests.fixtures.packs.dummy_pack_2.fixture import PACK_NAME as DUMMY_PACK_2
from st2tests.fixtures.packs.dummy_pack_18.fixture import PACK_DIR_NAME as DUMMY_PACK_18

class ContentPackConfigParserTestCase(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(ContentPackConfigParserTestCase, self).setUp()
        tests_config.parse_args()

    def test_get_config_inexistent_pack(self):
        if False:
            while True:
                i = 10
        parser = ContentPackConfigParser(pack_name='inexistent')
        config = parser.get_config()
        self.assertEqual(config, None)

    def test_get_config_no_config(self):
        if False:
            return 10
        pack_name = DUMMY_PACK_1
        parser = ContentPackConfigParser(pack_name=pack_name)
        config = parser.get_config()
        self.assertEqual(config, None)

    def test_get_config_existing_config(self):
        if False:
            i = 10
            return i + 15
        pack_name = DUMMY_PACK_2
        parser = ContentPackConfigParser(pack_name=pack_name)
        config = parser.get_config()
        self.assertEqual(config.config['section1']['key1'], 'value1')
        self.assertEqual(config.config['section2']['key10'], 'value10')

    def test_get_config_for_unicode_char(self):
        if False:
            i = 10
            return i + 15
        pack_name = DUMMY_PACK_18
        parser = ContentPackConfigParser(pack_name=pack_name)
        config = parser.get_config()
        self.assertEqual(config.config['section1']['key1'], '测试')