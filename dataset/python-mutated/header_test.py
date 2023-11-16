import webtest
import header

def test_send_mail(testbed):
    if False:
        print('Hello World!')
    testbed.init_mail_stub()
    testbed.init_app_identity_stub()
    app = webtest.TestApp(header.app)
    response = app.post('/header', 'thread_id=42')
    assert response.status_int == 200
    assert 'Sent an email to Albert with Reference header set to 42.' in response.body