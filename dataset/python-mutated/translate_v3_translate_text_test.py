import os
import pytest
import translate_v3_translate_text
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']

def test_translate_text(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    response = translate_v3_translate_text.translate_text('Hello World!', PROJECT_ID)
    (out, _) = capsys.readouterr()
    assert 'Bonjour le monde' in response.translations[0].translated_text