import os
import pytest
import translate_v3_translate_text_with_model
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']
MODEL_ID = 'TRL251293382528204800'

def test_translate_text_with_model(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        return 10
    response = translate_v3_translate_text_with_model.translate_text_with_model("That' il do it.", PROJECT_ID, MODEL_ID)
    (out, _) = capsys.readouterr()
    assert 'それはそうだ' or 'それじゃあ' in out
    assert response is not None