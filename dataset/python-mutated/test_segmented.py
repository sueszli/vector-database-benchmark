from streamlink.stream.segmented.segmented import log

def test_logger_name():
    if False:
        print('Hello World!')
    assert log.name == 'streamlink.stream.segmented'