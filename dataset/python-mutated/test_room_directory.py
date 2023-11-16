import yaml
from synapse.config.room_directory import RoomDirectoryConfig
from tests import unittest

class RoomDirectoryConfigTestCase(unittest.TestCase):

    def test_alias_creation_acl(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        config = yaml.safe_load('\n        alias_creation_rules:\n            - user_id: "*bob*"\n              alias: "*"\n              action: "deny"\n            - user_id: "*"\n              alias: "#unofficial_*"\n              action: "allow"\n            - user_id: "@foo*:example.com"\n              alias: "*"\n              action: "allow"\n            - user_id: "@gah:example.com"\n              alias: "#goo:example.com"\n              action: "allow"\n\n        room_list_publication_rules: []\n        ')
        rd_config = RoomDirectoryConfig()
        rd_config.read_config(config)
        self.assertFalse(rd_config.is_alias_creation_allowed(user_id='@bob:example.com', room_id='!test', alias='#test:example.com'))
        self.assertTrue(rd_config.is_alias_creation_allowed(user_id='@test:example.com', room_id='!test', alias='#unofficial_st:example.com'))
        self.assertTrue(rd_config.is_alias_creation_allowed(user_id='@foobar:example.com', room_id='!test', alias='#test:example.com'))
        self.assertTrue(rd_config.is_alias_creation_allowed(user_id='@gah:example.com', room_id='!test', alias='#goo:example.com'))
        self.assertFalse(rd_config.is_alias_creation_allowed(user_id='@test:example.com', room_id='!test', alias='#test:example.com'))

    def test_room_publish_acl(self) -> None:
        if False:
            return 10
        config = yaml.safe_load('\n        alias_creation_rules: []\n\n        room_list_publication_rules:\n            - user_id: "*bob*"\n              alias: "*"\n              action: "deny"\n            - user_id: "*"\n              alias: "#unofficial_*"\n              action: "allow"\n            - user_id: "@foo*:example.com"\n              alias: "*"\n              action: "allow"\n            - user_id: "@gah:example.com"\n              alias: "#goo:example.com"\n              action: "allow"\n            - room_id: "!test-deny"\n              action: "deny"\n        ')
        rd_config = RoomDirectoryConfig()
        rd_config.read_config(config)
        self.assertFalse(rd_config.is_publishing_room_allowed(user_id='@bob:example.com', room_id='!test', aliases=['#test:example.com']))
        self.assertTrue(rd_config.is_publishing_room_allowed(user_id='@test:example.com', room_id='!test', aliases=['#unofficial_st:example.com']))
        self.assertTrue(rd_config.is_publishing_room_allowed(user_id='@foobar:example.com', room_id='!test', aliases=[]))
        self.assertTrue(rd_config.is_publishing_room_allowed(user_id='@gah:example.com', room_id='!test', aliases=['#goo:example.com']))
        self.assertFalse(rd_config.is_publishing_room_allowed(user_id='@test:example.com', room_id='!test', aliases=['#test:example.com']))
        self.assertTrue(rd_config.is_publishing_room_allowed(user_id='@foobar:example.com', room_id='!test-deny', aliases=[]))
        self.assertFalse(rd_config.is_publishing_room_allowed(user_id='@gah:example.com', room_id='!test-deny', aliases=[]))
        self.assertTrue(rd_config.is_publishing_room_allowed(user_id='@test:example.com', room_id='!test', aliases=['#unofficial_st:example.com', '#blah:example.com']))