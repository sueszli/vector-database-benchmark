import os
from ssml_addresses import ssml_to_audio, text_to_ssml

def test_text_to_ssml(capsys):
    if False:
        for i in range(10):
            print('nop')
    with open('resources/example.ssml') as f:
        expected_ssml = f.read()
    input_text = 'resources/example.txt'
    tested_ssml = text_to_ssml(input_text)
    assert expected_ssml == tested_ssml

def test_ssml_to_audio(capsys):
    if False:
        while True:
            i = 10
    with open('resources/example.ssml') as f:
        input_ssml = f.read()
    ssml_to_audio(input_ssml, 'test_example.mp3')
    (out, err) = capsys.readouterr()
    assert os.path.isfile('test_example.mp3')
    assert 'Audio content written to file test_example.mp3' in out
    os.remove('test_example.mp3')