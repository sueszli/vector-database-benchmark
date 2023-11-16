import unittest
import time
import pytest
import azure.cosmos.cosmos_client as cosmos_client
import test_config
pytestmark = pytest.mark.cosmosEmulator

@pytest.mark.usefixtures('teardown')
class Test_session_container(unittest.TestCase):
    host = test_config._test_config.host
    masterkey = test_config._test_config.masterKey
    connectionPolicy = test_config._test_config.connectionPolicy

    def setUp(self):
        if False:
            print('Hello World!')
        self.client = cosmos_client.CosmosClient(self.host, self.masterkey, consistency_level='Session', connection_policy=self.connectionPolicy)
        self.session = self.client.client_connection.Session

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_create_collection(self):
        if False:
            return 10
        session_token = self.session.get_session_token('')
        assert session_token == ''
        create_collection_response_result = {u'_self': u'dbs/DdAkAA==/colls/DdAkAPS2rAA=/', u'_rid': u'DdAkAPS2rAA=', u'id': u'sample collection'}
        create_collection_response_header = {'x-ms-session-token': '0:0#409#24=-1#12=-1', 'x-ms-alt-content-path': 'dbs/sample%20database'}
        self.session.update_session(create_collection_response_result, create_collection_response_header)
        token = self.session.get_session_token(u'/dbs/sample%20database/colls/sample%20collection')
        assert token == '0:0#409#24=-1#12=-1'
        token = self.session.get_session_token(u'dbs/DdAkAA==/colls/DdAkAPS2rAA=/')
        assert token == '0:0#409#24=-1#12=-1'
        return True

    def test_document_requests(self):
        if False:
            for i in range(10):
                print('nop')
        create_document_response_result = {u'_self': u'dbs/DdAkAA==/colls/DdAkAPS2rAA=/docs/DdAkAPS2rAACAAAAAAAAAA==/', u'_rid': u'DdAkAPS2rAACAAAAAAAAAA==', u'id': u'eb391181-5c49-415a-ab27-848ce21d5d11'}
        create_document_response_header = {'x-ms-session-token': '0:0#406#24=-1#12=-1', 'x-ms-alt-content-path': 'dbs/sample%20database/colls/sample%20collection', 'x-ms-content-path': 'DdAkAPS2rAA='}
        self.session.update_session(create_document_response_result, create_document_response_header)
        token = self.session.get_session_token(u'dbs/DdAkAA==/colls/DdAkAPS2rAA=/docs/DdAkAPS2rAACAAAAAAAAAA==/')
        assert token == '0:0#406#24=-1#12=-1'
        token = self.session.get_session_token(u'dbs/sample%20database/colls/sample%20collection/docs/eb391181-5c49-415a-ab27-848ce21d5d11')
        assert token == '0:0#406#24=-1#12=-1'
if __name__ == '__main__':
    unittest.main()