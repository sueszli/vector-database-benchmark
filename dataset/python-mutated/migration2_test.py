import webtest
import migration2

def test_get(testbed):
    if False:
        while True:
            i = 10
    app = webtest.TestApp(migration2.app)
    app.get('/')