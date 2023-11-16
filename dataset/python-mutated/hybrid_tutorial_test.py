import os
import sys
import uuid
import pytest
import hybrid_tutorial
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']

def test_vision_standard_format() -> None:
    if False:
        i = 10
        return i + 15
    text = hybrid_tutorial.pic_to_text('resources/standard_format.jpeg')
    assert len(text) > 0

def test_create_and_delete_glossary() -> None:
    if False:
        i = 10
        return i + 15
    sys.path.insert(1, '../')
    from google.cloud import translate_v3 as translate
    languages = ['fr', 'en']
    glossary_name = f'test-glossary-{uuid.uuid4()}'
    glossary_uri = 'gs://cloud-samples-data/translation/bistro_glossary.csv'
    created_glossary_name = hybrid_tutorial.create_glossary(languages, PROJECT_ID, glossary_name, glossary_uri)
    client = translate.TranslationServiceClient()
    name = client.glossary_path(PROJECT_ID, 'us-central1', created_glossary_name)
    operation = client.delete_glossary(name=name)
    result = operation.result(timeout=180)
    assert created_glossary_name in result.name
    print(f'Deleted: {result.name}')

def test_translate_standard() -> None:
    if False:
        for i in range(10):
            print('nop')
    expected_text = 'Good morning'
    languages = ['fr', 'en']
    glossary_name = 'bistro-glossary'
    glossary_uri = f'gs://cloud-samples-data/translation/{glossary_name}.csv'
    created_glossary_name = hybrid_tutorial.create_glossary(languages, PROJECT_ID, glossary_name, glossary_uri)
    text = hybrid_tutorial.translate_text('Bonjour', 'fr', 'en', PROJECT_ID, created_glossary_name)
    assert text == expected_text

def test_translate_glossary() -> None:
    if False:
        i = 10
        return i + 15
    expected_text = 'I eat goat cheese'
    input_text = 'Je mange du chevre'
    languages = ['fr', 'en']
    glossary_name = 'bistro-glossary'
    glossary_uri = f'gs://cloud-samples-data/translation/{glossary_name}.csv'
    created_glossary_name = hybrid_tutorial.create_glossary(languages, PROJECT_ID, glossary_name, glossary_uri)
    text = hybrid_tutorial.translate_text(input_text, 'fr', 'en', PROJECT_ID, created_glossary_name)
    assert text == expected_text

def test_tts_standard(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        print('Hello World!')
    outfile = 'resources/test_standard_text.mp3'
    text = 'this is\na test!'
    generated_outfile = hybrid_tutorial.text_to_speech(text, outfile)
    assert os.path.isfile(generated_outfile)
    (out, err) = capsys.readouterr()
    assert 'Audio content written to file ' + generated_outfile in out
    os.remove(outfile)