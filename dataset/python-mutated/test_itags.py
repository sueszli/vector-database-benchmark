from pytube import itags

def test_get_format_profile():
    if False:
        while True:
            i = 10
    profile = itags.get_format_profile(22)
    assert profile['resolution'] == '720p'

def test_get_format_profile_non_existant():
    if False:
        print('Hello World!')
    profile = itags.get_format_profile(2239)
    assert profile['resolution'] is None