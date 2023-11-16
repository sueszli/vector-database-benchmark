import os
import uuid
import create_inspect_template as ct
import delete_inspect_template as dt
import google.api_core.exceptions
import google.cloud.storage
import list_inspect_templates as lt
import pytest
UNIQUE_STRING = str(uuid.uuid4()).split('-')[0]
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
TEST_TEMPLATE_ID = 'test-template' + UNIQUE_STRING

def test_create_list_and_delete_template(capsys: pytest.CaptureFixture) -> None:
    if False:
        return 10
    try:
        ct.create_inspect_template(GCLOUD_PROJECT, ['FIRST_NAME', 'EMAIL_ADDRESS', 'PHONE_NUMBER'], template_id=TEST_TEMPLATE_ID)
    except google.api_core.exceptions.InvalidArgument:
        dt.delete_inspect_template(GCLOUD_PROJECT, TEST_TEMPLATE_ID)
        (out, _) = capsys.readouterr()
        assert TEST_TEMPLATE_ID in out
        ct.create_inspect_template(GCLOUD_PROJECT, ['FIRST_NAME', 'EMAIL_ADDRESS', 'PHONE_NUMBER'], template_id=TEST_TEMPLATE_ID)
    (out, _) = capsys.readouterr()
    assert TEST_TEMPLATE_ID in out
    lt.list_inspect_templates(GCLOUD_PROJECT)
    (out, _) = capsys.readouterr()
    assert TEST_TEMPLATE_ID in out
    dt.delete_inspect_template(GCLOUD_PROJECT, TEST_TEMPLATE_ID)
    (out, _) = capsys.readouterr()
    assert TEST_TEMPLATE_ID in out