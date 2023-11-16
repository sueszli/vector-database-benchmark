import os
import synthesize_text
TEXT = 'Hello there.'
SSML = '<speak>Hello there.</speak>'

def test_synthesize_text(capsys):
    if False:
        for i in range(10):
            print('nop')
    synthesize_text.synthesize_text(text=TEXT)
    (out, err) = capsys.readouterr()
    assert 'Audio content written to file' in out
    statinfo = os.stat('output.mp3')
    assert statinfo.st_size > 0

def test_synthesize_ssml(capsys):
    if False:
        print('Hello World!')
    synthesize_text.synthesize_ssml(ssml=SSML)
    (out, err) = capsys.readouterr()
    assert 'Audio content written to file' in out
    statinfo = os.stat('output.mp3')
    assert statinfo.st_size > 0