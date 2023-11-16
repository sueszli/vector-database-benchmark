import os
import synthesize_file
TEXT_FILE = 'resources/hello.txt'
SSML_FILE = 'resources/hello.ssml'

def test_synthesize_text_file(capsys):
    if False:
        for i in range(10):
            print('nop')
    synthesize_file.synthesize_text_file(text_file=TEXT_FILE)
    (out, err) = capsys.readouterr()
    assert 'Audio content written to file' in out
    statinfo = os.stat('output.mp3')
    assert statinfo.st_size > 0

def test_synthesize_ssml_file(capsys):
    if False:
        while True:
            i = 10
    synthesize_file.synthesize_ssml_file(ssml_file=SSML_FILE)
    (out, err) = capsys.readouterr()
    assert 'Audio content written to file' in out
    statinfo = os.stat('output.mp3')
    assert statinfo.st_size > 0