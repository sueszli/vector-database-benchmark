import list_voices

def test_list_voices(capsys):
    if False:
        print('Hello World!')
    list_voices.list_voices()
    (out, err) = capsys.readouterr()
    assert 'en-US' in out
    assert 'SSML Voice Gender: MALE' in out
    assert 'SSML Voice Gender: FEMALE' in out