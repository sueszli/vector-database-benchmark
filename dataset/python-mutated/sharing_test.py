import webtest
import sharing

def test_get(testbed):
    if False:
        return 10
    app = webtest.TestApp(sharing.app)
    response = app.get('/')
    assert 'Previously incremented by ' in response.body