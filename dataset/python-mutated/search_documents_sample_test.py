import os
from contentwarehouse.snippets import search_documents_sample
from contentwarehouse.snippets import test_utilities
import pytest
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
location = 'us'
document_query_text = 'document'
user_id = 'user:xxxx@example.com'

def test_search_documents(capsys: pytest.CaptureFixture) -> None:
    if False:
        return 10
    project_number = test_utilities.get_project_number(project_id)
    search_documents_sample.search_documents_sample(project_number=project_number, location=location, document_query_text=document_query_text, user_id=user_id)
    (out, _) = capsys.readouterr()
    assert 'document' in out