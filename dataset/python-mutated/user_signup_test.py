import webtest
import user_signup

def test_user_signup(testbed):
    if False:
        for i in range(10):
            print('nop')
    testbed.init_mail_stub()
    testbed.init_app_identity_stub()
    testbed.init_datastore_v3_stub()
    app = webtest.TestApp(user_signup.app)
    response = app.post('/user/signup', 'email_address=alice@example.com')
    assert response.status_int == 200
    assert 'An email has been sent to alice@example.com.' in response.body
    records = user_signup.UserConfirmationRecord.query().fetch(1)
    response = app.get('/user/confirm?code={}'.format(records[0].key.id()))
    assert response.status_int == 200
    assert 'Confirmed alice@example.com.' in response.body

def test_bad_code(testbed):
    if False:
        return 10
    testbed.init_datastore_v3_stub()
    app = webtest.TestApp(user_signup.app)
    response = app.get('/user/confirm?code=garbage', status=404)
    assert response.status_int == 404