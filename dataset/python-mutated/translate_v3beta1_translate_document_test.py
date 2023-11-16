import os
import pytest
import translate_v3beta1_translate_document
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']
FILE_PATH = 'resources/fake_invoice.pdf'

def test_translate_document(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    response = translate_v3beta1_translate_document.translate_document(project_id=PROJECT_ID, file_path=FILE_PATH)
    (out, _) = capsys.readouterr()
    assert 'en' in response.document_translation.detected_language_code