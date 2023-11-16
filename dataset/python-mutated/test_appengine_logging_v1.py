from google.cloud import appengine_logging_v1

def test_log_line():
    if False:
        while True:
            i = 10
    appengine_logging_v1.LogLine()

def test_source_location():
    if False:
        return 10
    appengine_logging_v1.SourceLocation()

def test_source_reference():
    if False:
        return 10
    appengine_logging_v1.SourceReference()

def test_request_log():
    if False:
        i = 10
        return i + 15
    appengine_logging_v1.RequestLog()