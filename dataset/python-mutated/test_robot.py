import hashlib
import time
import os
import pytest
from werobot import WeRoBot
from werobot.utils import generate_token, to_text

def _make_xml(content):
    if False:
        return 10
    return '\n        <xml>\n        <ToUserName><![CDATA[toUser]]></ToUserName>\n        <FromUserName><![CDATA[fromUser]]></FromUserName>\n        <CreateTime>1348831860</CreateTime>\n        <MsgType><![CDATA[text]]></MsgType>\n        <Content><![CDATA[%s]]></Content>\n        <MsgId>1234567890123456</MsgId>\n        </xml>\n    ' % content

def test_signature_checker():
    if False:
        while True:
            i = 10
    token = generate_token()
    robot = WeRoBot(token, SESSION_STORAGE=False)
    timestamp = str(int(time.time()))
    nonce = '12345678'
    sign = [token, timestamp, nonce]
    sign.sort()
    sign = ''.join(sign)
    sign = sign.encode()
    sign = hashlib.sha1(sign).hexdigest()
    assert robot.check_signature(timestamp, nonce, sign)

def test_register_handlers():
    if False:
        while True:
            i = 10
    robot = WeRoBot(SESSION_STORAGE=False)
    for type in robot.message_types:
        assert hasattr(robot, type) or hasattr(robot, type.replace('_event', ''))

    @robot.text
    def text_handler():
        if False:
            i = 10
            return i + 15
        return 'Hi'
    assert robot._handlers['text'] == [(text_handler, 0)]

    @robot.image
    def image_handler(message):
        if False:
            i = 10
            return i + 15
        return 'nice pic'
    assert robot._handlers['image'] == [(image_handler, 1)]
    assert robot.get_handlers('text') == [(text_handler, 0)]

    @robot.handler
    def handler(message, session):
        if False:
            return 10
        pass
    assert robot.get_handlers('text') == [(text_handler, 0), (handler, 2)]

    @robot.video
    def video_handler():
        if False:
            print('Hello World!')
        pass
    assert robot._handlers['video'] == [(video_handler, 0)]
    assert robot.get_handlers('video') == [(video_handler, 0), (handler, 2)]

    @robot.shortvideo
    def shortvideo_handler():
        if False:
            return 10
        pass
    assert robot._handlers['shortvideo'] == [(shortvideo_handler, 0)]
    assert robot.get_handlers('shortvideo') == [(shortvideo_handler, 0), (handler, 2)]

    @robot.location
    def location_handler():
        if False:
            while True:
                i = 10
        pass
    assert robot._handlers['location'] == [(location_handler, 0)]

    @robot.link
    def link_handler():
        if False:
            while True:
                i = 10
        pass
    assert robot._handlers['link'] == [(link_handler, 0)]

    @robot.subscribe
    def subscribe_handler():
        if False:
            while True:
                i = 10
        pass
    assert robot._handlers['subscribe_event'] == [(subscribe_handler, 0)]

    @robot.unsubscribe
    def unsubscribe_handler():
        if False:
            while True:
                i = 10
        pass
    assert robot._handlers['unsubscribe_event'] == [(unsubscribe_handler, 0)]

    @robot.voice
    def voice_handler():
        if False:
            i = 10
            return i + 15
        pass
    assert robot._handlers['voice'] == [(voice_handler, 0)]

    @robot.click
    def click_handler():
        if False:
            while True:
                i = 10
        pass
    assert robot._handlers['click_event'] == [(click_handler, 0)]

    @robot.key_click('MENU')
    def menu_handler():
        if False:
            print('Hello World!')
        pass
    assert len(robot._handlers['click_event']) == 2

    @robot.scan
    def scan_handler():
        if False:
            for i in range(10):
                print('nop')
        pass
    assert robot._handlers['scan_event'] == [(scan_handler, 0)]

    @robot.scancode_push
    def scancode_push_handler():
        if False:
            while True:
                i = 10
        pass
    assert robot._handlers['scancode_push_event'] == [(scancode_push_handler, 0)]

    @robot.scancode_waitmsg
    def scancode_waitmsg_handler():
        if False:
            for i in range(10):
                print('nop')
        pass
    assert robot._handlers['scancode_waitmsg_event'] == [(scancode_waitmsg_handler, 0)]

def test_filter():
    if False:
        i = 10
        return i + 15
    import re
    import werobot.testing
    robot = WeRoBot(SESSION_STORAGE=False)

    @robot.filter('喵')
    def _1():
        if False:
            return 10
        return '喵'
    assert len(robot._handlers['text']) == 1

    @robot.filter(re.compile(to_text('.*?呵呵.*?')))
    def _2():
        if False:
            print('Hello World!')
        return '哼'
    assert len(robot._handlers['text']) == 2

    @robot.text
    def _3():
        if False:
            for i in range(10):
                print('nop')
        return '汪'
    assert len(robot._handlers['text']) == 3
    tester = werobot.testing.WeTest(robot)
    assert tester.send_xml(_make_xml('啊'))._args['content'] == u'汪'
    assert tester.send_xml(_make_xml('啊呵呵'))._args['content'] == u'哼'
    assert tester.send_xml(_make_xml('喵'))._args['content'] == u'喵'
    try:
        os.remove(os.path.abspath('werobot_session'))
    except OSError:
        pass
    robot = WeRoBot(SESSION_STORAGE=False)

    @robot.filter('帮助', '跪求帮助', re.compile('(.*?)help.*?'))
    def _(message, session, match):
        if False:
            return 10
        if match and match.group(1) == u'小姐姐':
            return '本小姐就帮你一下'
        return '就不帮'
    assert len(robot._handlers['text']) == 3

    @robot.text
    def _4():
        if False:
            return 10
        return '哦'
    assert len(robot._handlers['text']) == 4
    tester = werobot.testing.WeTest(robot)
    assert tester.send_xml(_make_xml('啊'))._args['content'] == u'哦'
    assert tester.send_xml(_make_xml('帮助'))._args['content'] == u'就不帮'
    assert tester.send_xml(_make_xml('跪求帮助'))._args['content'] == u'就不帮'
    assert tester.send_xml(_make_xml('ooohelp'))._args['content'] == u'就不帮'
    assert tester.send_xml(_make_xml('小姐姐help'))._args['content'] == u'本小姐就帮你一下'

def test_register_not_callable_object():
    if False:
        i = 10
        return i + 15
    robot = WeRoBot(SESSION_STORAGE=False)
    with pytest.raises(ValueError):
        robot.add_handler('s')

def test_error_page():
    if False:
        while True:
            i = 10
    robot = WeRoBot()

    @robot.error_page
    def make_error_page(url):
        if False:
            print('Hello World!')
        return url
    assert robot.make_error_page('喵') == '喵'

def test_config_ignore():
    if False:
        return 10
    from werobot.config import Config
    config = Config(TOKEN='token from config')
    robot = WeRoBot(config=config, token='token2333')
    assert robot.token == 'token from config'

def test_add_filter():
    if False:
        i = 10
        return i + 15
    import werobot.testing
    import re
    robot = WeRoBot()

    def test_register():
        if False:
            print('Hello World!')
        return 'test'
    robot.add_filter(test_register, ['test', re.compile(u'.*?啦.*?')])
    tester = werobot.testing.WeTest(robot)
    assert tester.send_xml(_make_xml('test'))._args['content'] == 'test'
    assert tester.send_xml(_make_xml(u'我要测试啦'))._args['content'] == 'test'
    assert tester.send_xml(_make_xml(u'我要测试')) is None
    with pytest.raises(ValueError) as e:
        robot.add_filter('test', ['test'])
    assert e.value.args[0] == 'test is not callable'
    with pytest.raises(ValueError) as e:
        robot.add_filter(test_register, 'test')
    assert e.value.args[0] == 'test is not list'
    with pytest.raises(TypeError) as e:
        robot.add_filter(test_register, [['bazinga']])
    assert e.value.args[0] == "['bazinga'] is not a valid rule"