import webtest
import migration1

def test_get(testbed):
    if False:
        i = 10
        return i + 15
    app = webtest.TestApp(migration1.app)
    app.get('/')