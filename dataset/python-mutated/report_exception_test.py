import report_exception

def test_error_sends():
    if False:
        return 10
    report_exception.report_exception()

def test_manual_error_sends():
    if False:
        print('Hello World!')
    report_exception.report_manual_error()