import threading
from functools import partial
from pprint import pprint
import uiautomator2 as u2
import pytest

def test_get_text(sess: u2.Session):
    if False:
        i = 10
        return i + 15
    assert sess.xpath('App').get_text() == 'App'

def test_click(sess: u2.Session):
    if False:
        return 10
    sess.xpath('App').click()
    assert sess.xpath('Alarm').wait()
    assert sess.xpath('Alarm').exists

def test_swipe(sess: u2.Session):
    if False:
        while True:
            i = 10
    d = sess
    d.xpath('App').click()
    d.xpath('Alarm').wait()
    d.xpath('@android:id/list').get().swipe('up', 0.5)
    assert d.xpath('Voice Recognition').wait()

def test_xpath_query(sess: u2.Session):
    if False:
        return 10
    assert sess.xpath('Accessibility').wait()
    assert sess.xpath('%ccessibility').wait()
    assert sess.xpath('Accessibilit%').wait()

def test_element_all(sess: u2.Session):
    if False:
        while True:
            i = 10
    app = sess.xpath('//*[@text="App"]')
    assert app.wait()
    assert len(app.all()) == 1
    assert app.exists
    elements = sess.xpath('//*[@resource-id="android:id/list"]/android.widget.TextView').all()
    assert len(elements) == 11
    el = elements[0]
    assert el.text == 'Accessibility'

def test_watcher(sess: u2.Session, request):
    if False:
        for i in range(10):
            print('nop')
    sess.xpath.when('App').click()
    sess.xpath.watch_background(interval=1.0)
    event = threading.Event()

    def _set_event(e):
        if False:
            return 10
        e.set()
    sess.xpath.when('Action Bar').call(partial(_set_event, event))
    assert event.wait(5.0), 'xpath not trigger callback'

@pytest.mark.skip('Deprecated')
def test_watcher_from_yaml(sess: u2.Session, request):
    if False:
        while True:
            i = 10
    yaml_content = '---\n- when: App\n  then: click\n- when: Action Bar\n  then: >\n    def callback(d):\n        print("D:", d)\n        d.xpath("Alarm").click()\n    \n    def hello():\n        print("World")\n'
    sess.xpath.apply_watch_from_yaml(yaml_content)
    sess.xpath.watch_background(interval=1.0)
    assert sess.xpath('Alarm Controller').wait(timeout=10)

def test_xpath_scroll_to(sess: u2.Session):
    if False:
        for i in range(10):
            print('nop')
    d = sess
    d.xpath('Graphics').click()
    d.xpath('@android:id/list').scroll_to('Pictures')
    assert d.xpath('Pictures').exists

def test_xpath_parent(sess: u2.Session):
    if False:
        while True:
            i = 10
    d = sess
    info = d.xpath('App').parent('@android:id/list').info
    assert info['resourceId'] == 'android:id/list'