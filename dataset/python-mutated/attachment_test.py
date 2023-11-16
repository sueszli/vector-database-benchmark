import webtest
import attachment

def test_send_mail(testbed):
    if False:
        for i in range(10):
            print('nop')
    testbed.init_mail_stub()
    testbed.init_app_identity_stub()
    app = webtest.TestApp(attachment.app)
    response = app.post('/attachment', upload_files=[('file', 'hello.txt', 'Good day!')])
    assert response.status_int == 200
    assert 'Sent hello.txt to Albert.' in response.body