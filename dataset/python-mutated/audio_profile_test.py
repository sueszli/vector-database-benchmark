import os
import os.path
import audio_profile
TEXT = 'hello'
OUTPUT = 'output.mp3'
EFFECTS_PROFILE_ID = 'telephony-class-application'

def test_audio_profile(capsys):
    if False:
        while True:
            i = 10
    if os.path.exists(OUTPUT):
        os.remove(OUTPUT)
    assert not os.path.exists(OUTPUT)
    audio_profile.synthesize_text_with_audio_profile(TEXT, OUTPUT, EFFECTS_PROFILE_ID)
    (out, err) = capsys.readouterr()
    assert 'Audio content written to file "%s"' % OUTPUT in out
    assert os.path.exists(OUTPUT)
    os.remove(OUTPUT)