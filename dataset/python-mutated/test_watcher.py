import uiautomator2 as u2

def test_watch_context(sess: u2.Device):
    if False:
        return 10
    with sess.watch_context(builtin=True) as ctx:
        ctx.when('App').click()
        sess(text='Menu').click()
        assert sess(text='Inflate from XML').wait()

def teardown_function(d: u2.Device):
    if False:
        print('Hello World!')
    print('Teardown', d)