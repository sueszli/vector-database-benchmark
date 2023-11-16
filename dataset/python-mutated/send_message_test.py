import webtest
import send_message

def test_send_message(testbed):
    if False:
        print('Hello World!')
    testbed.init_mail_stub()
    testbed.init_app_identity_stub()
    app = webtest.TestApp(send_message.app)
    response = app.get('/send_message')
    assert response.status_int == 200
    assert 'Sent an email message to Albert.' in response.body