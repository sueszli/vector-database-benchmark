import webtest
import send_mail

def test_send_mail(testbed):
    if False:
        return 10
    testbed.init_mail_stub()
    testbed.init_app_identity_stub()
    app = webtest.TestApp(send_mail.app)
    response = app.get('/send_mail')
    assert response.status_int == 200
    assert 'Sent an email to Albert.' in response.body