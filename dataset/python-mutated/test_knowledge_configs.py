import unittest
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.orm.session import Session
from superagi.models.knowledge_configs import KnowledgeConfigs

class TestKnowledgeConfigs(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.session = Mock(spec=Session)
        self.knowledge_id = 1
        self.test_configs = {'key1': 'value1', 'key2': 'value2'}

    @patch('requests.get')
    def test_fetch_knowledge_config_details_marketplace(self, mock_get):
        if False:
            for i in range(10):
                print('nop')
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{'key': 'key1', 'value': 'value1'}, {'key': 'key2', 'value': 'value2'}]
        mock_get.return_value = mock_response
        configs = KnowledgeConfigs.fetch_knowledge_config_details_marketplace(self.knowledge_id)
        self.assertEqual(configs, self.test_configs)

    def test_add_update_knowledge_config(self):
        if False:
            print('Hello World!')
        KnowledgeConfigs.add_update_knowledge_config(self.session, self.knowledge_id, self.test_configs)
        self.session.add.assert_called()
        self.session.commit.assert_called()

    def test_get_knowledge_config_from_knowledge_id(self):
        if False:
            i = 10
            return i + 15
        test_obj = Mock()
        test_obj.key = 'key1'
        test_obj.value = 'value1'
        self.session.query.return_value.filter.return_value.all.return_value = [test_obj]
        configs = KnowledgeConfigs.get_knowledge_config_from_knowledge_id(self.session, self.knowledge_id)
        self.assertEqual(configs, {'key1': 'value1'})

    def test_delete_knowledge_config(self):
        if False:
            return 10
        KnowledgeConfigs.delete_knowledge_config(self.session, self.knowledge_id)
        self.session.query.assert_called()
        self.session.commit.assert_called()

    def tearDown(self):
        if False:
            while True:
                i = 10
        pass
if __name__ == '__main__':
    unittest.main()