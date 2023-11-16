import pytest
import pytest_bdd as bdd
bdd.scenarios('notifications.feature')
pytestmark = [pytest.mark.usefixtures('notification_server'), pytest.mark.qtwebkit_skip]

@bdd.given('the notification server supports body markup')
def supports_body_markup(notification_server, quteproc):
    if False:
        print('Hello World!')
    notification_server.supports_body_markup = True
    quteproc.send_cmd(":debug-pyeval -q __import__('qutebrowser').browser.webengine.notification.bridge._drop_adapter()")

@bdd.given("the notification server doesn't support body markup")
def doesnt_support_body_markup(notification_server, quteproc):
    if False:
        i = 10
        return i + 15
    notification_server.supports_body_markup = False
    quteproc.send_cmd(":debug-pyeval -q __import__('qutebrowser').browser.webengine.notification.bridge._drop_adapter()")

@bdd.given('I clean up the notification server')
def cleanup_notification_server(notification_server):
    if False:
        return 10
    notification_server.cleanup()

@bdd.then('1 notification should be presented')
def notification_presented_single(notification_server):
    if False:
        for i in range(10):
            print('nop')
    assert len(notification_server.messages) == 1

@bdd.then(bdd.parsers.cfparse('{count:d} notifications should be presented'))
def notification_presented_count(notification_server, count):
    if False:
        while True:
            i = 10
    assert len(notification_server.messages) == count

@bdd.then(bdd.parsers.parse('the notification should have body "{body}"'))
def notification_body(notification_server, body):
    if False:
        return 10
    msg = notification_server.last_msg()
    assert msg.body == body

@bdd.then(bdd.parsers.parse('the notification should have title "{title}"'))
def notification_title(notification_server, title):
    if False:
        while True:
            i = 10
    msg = notification_server.last_msg()
    assert msg.title == title

@bdd.then(bdd.parsers.cfparse('the notification should have image dimensions {width:d}x{height:d}'))
def notification_image_dimensions(notification_server, width, height):
    if False:
        return 10
    msg = notification_server.last_msg()
    assert (msg.img_width, msg.img_height) == (width, height)

@bdd.then('the notification should be closed via web')
def notification_closed(notification_server):
    if False:
        i = 10
        return i + 15
    msg = notification_server.last_msg()
    assert msg.closed_via_web

@bdd.when('I close the notification')
def close_notification(notification_server):
    if False:
        print('Hello World!')
    notification_server.close(notification_server.last_id)

@bdd.when(bdd.parsers.cfparse('I close the notification with id {id_:d}'))
def close_notification_id(notification_server, id_):
    if False:
        i = 10
        return i + 15
    notification_server.close(id_)

@bdd.when('I click the notification')
def click_notification(notification_server):
    if False:
        for i in range(10):
            print('nop')
    notification_server.click(notification_server.last_id)

@bdd.when(bdd.parsers.cfparse('I click the notification with id {id_:d}'))
def click_notification_id(notification_server, id_):
    if False:
        return 10
    notification_server.click(id_)

@bdd.when(bdd.parsers.cfparse('I trigger a {name} action on the notification with id {id_:d}'))
def custom_notification_action(notification_server, id_, name):
    if False:
        return 10
    notification_server.action(id_, name)