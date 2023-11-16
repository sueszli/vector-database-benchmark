def report_exception():
    if False:
        while True:
            i = 10
    from google.cloud import error_reporting
    client = error_reporting.Client()
    try:
        raise Exception('Something went wrong')
    except Exception:
        client.report_exception()

def report_manual_error():
    if False:
        i = 10
        return i + 15
    from google.cloud import error_reporting
    client = error_reporting.Client()
    client.report('An error has occurred.')
if __name__ == '__main__':
    report_exception()
    report_manual_error()