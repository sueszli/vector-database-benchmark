import os
from google.api_core.retry import Retry
import pytest
from beta_snippets import transcribe_file_with_auto_punctuation, transcribe_file_with_diarization, transcribe_file_with_enhanced_model, transcribe_file_with_metadata, transcribe_file_with_multichannel, transcribe_file_with_multilanguage, transcribe_file_with_spoken_punctuation_end_emojis, transcribe_file_with_word_level_confidence
RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

@Retry()
def test_transcribe_file_with_enhanced_model(capsys: pytest.CaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    result = transcribe_file_with_enhanced_model()
    (out, _) = capsys.readouterr()
    assert 'Chrome' in out
    assert result is not None

@Retry()
def test_transcribe_file_with_metadata(capsys: pytest.CaptureFixture) -> None:
    if False:
        return 10
    result = transcribe_file_with_metadata()
    (out, _) = capsys.readouterr()
    assert 'Chrome' in out
    assert result is not None

@Retry()
def test_transcribe_file_with_auto_punctuation(capsys: pytest.CaptureFixture) -> None:
    if False:
        while True:
            i = 10
    result = transcribe_file_with_auto_punctuation()
    (out, _) = capsys.readouterr()
    assert 'First alternative of result ' in out
    assert result is not None

@Retry()
def test_transcribe_diarization(capsys: pytest.CaptureFixture) -> None:
    if False:
        while True:
            i = 10
    result = transcribe_file_with_diarization()
    (out, err) = capsys.readouterr()
    assert 'word:' in out
    assert 'speaker_tag:' in out
    assert result is not None

@Retry()
def test_transcribe_multichannel_file(capsys: pytest.CaptureFixture) -> None:
    if False:
        return 10
    result = transcribe_file_with_multichannel()
    (out, err) = capsys.readouterr()
    assert 'OK Google stream stranger things from Netflix to my TV' in out
    assert result is not None

@Retry()
def test_transcribe_multilanguage_file(capsys: pytest.CaptureFixture) -> None:
    if False:
        i = 10
        return i + 15
    result = transcribe_file_with_multilanguage()
    (out, err) = capsys.readouterr()
    assert 'First alternative of result' in out
    assert 'Transcript' in out
    assert result is not None

@Retry()
def test_transcribe_word_level_confidence(capsys: pytest.CaptureFixture) -> None:
    if False:
        while True:
            i = 10
    result = transcribe_file_with_word_level_confidence()
    (out, err) = capsys.readouterr()
    assert 'OK Google stream stranger things from Netflix to my TV' in out
    assert result is not None

@Retry()
def test_transcribe_file_with_spoken_punctuation_end_emojis(capsys: pytest.CaptureFixture) -> None:
    if False:
        print('Hello World!')
    result = transcribe_file_with_spoken_punctuation_end_emojis()
    (out, err) = capsys.readouterr()
    assert 'First alternative of result ' in out
    assert result is not None