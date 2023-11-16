from __future__ import absolute_import
import os
import uuid
from google.cloud import dialogflow_v2beta1
import pytest
import knowledge_base_management
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
KNOWLEDGE_BASE_NAME = 'knowledge_{}'.format(uuid.uuid4())
pytest.KNOWLEDGE_BASE_ID = None

@pytest.fixture(scope='function', autouse=True)
def teardown():
    if False:
        print('Hello World!')
    yield
    client = dialogflow_v2beta1.KnowledgeBasesClient()
    assert pytest.KNOWLEDGE_BASE_ID is not None
    knowledge_base_path = client.knowledge_base_path(PROJECT_ID, pytest.KNOWLEDGE_BASE_ID)
    client.delete_knowledge_base(name=knowledge_base_path)

def test_create_knowledge_base(capsys):
    if False:
        i = 10
        return i + 15
    knowledge_base_management.create_knowledge_base(PROJECT_ID, KNOWLEDGE_BASE_NAME)
    (out, _) = capsys.readouterr()
    assert KNOWLEDGE_BASE_NAME in out
    pytest.KNOWLEDGE_BASE_ID = out.split('/knowledgeBases/')[1].split('\n')[0]