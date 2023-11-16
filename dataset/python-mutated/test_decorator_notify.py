from os.path import dirname
from os.path import join
from apprise.decorators import notify
from apprise import Apprise
from apprise import AppriseConfig
from apprise import AppriseAsset
from apprise import AppriseAttachment
from apprise import common
import logging
logging.disable(logging.CRITICAL)
TEST_VAR_DIR = join(dirname(__file__), 'var')

def test_notify_simple_decoration():
    if False:
        while True:
            i = 10
    'decorators: Test simple @notify\n    '
    assert 'utiltest' not in common.NOTIFY_SCHEMA_MAP
    verify_obj = {}

    @notify(on='utiltest', name='Apprise @notify Decorator Testing')
    def my_inline_notify_wrapper(body, title, notify_type, attach, *args, **kwargs):
        if False:
            print('Hello World!')
        verify_obj.update({'body': body, 'title': title, 'notify_type': notify_type, 'attach': attach, 'args': args, 'kwargs': kwargs})
    assert 'utiltest' in common.NOTIFY_SCHEMA_MAP
    aobj = Apprise()
    assert aobj.add('utiltest://') is True
    assert len(verify_obj) == 0
    assert aobj.notify('Hello World', title='My Title', attach=(join(TEST_VAR_DIR, 'apprise-test.gif'), join(TEST_VAR_DIR, 'apprise-test.png'))) is True
    assert len(verify_obj) > 0
    assert verify_obj['body'] == 'Hello World'
    assert verify_obj['title'] == 'My Title'
    assert verify_obj['notify_type'] == common.NotifyType.INFO
    assert isinstance(verify_obj['attach'], AppriseAttachment)
    assert len(verify_obj['attach']) == 2
    assert 'body_format' in verify_obj['kwargs']
    assert verify_obj['kwargs']['body_format'] is None
    assert isinstance(verify_obj['kwargs'], dict)
    assert 'meta' in verify_obj['kwargs']
    assert isinstance(verify_obj['kwargs']['meta'], dict)
    assert len(verify_obj['kwargs']['meta']) == 4
    assert 'tag' in verify_obj['kwargs']['meta']
    assert 'asset' in verify_obj['kwargs']['meta']
    assert isinstance(verify_obj['kwargs']['meta']['asset'], AppriseAsset)
    assert verify_obj['kwargs']['meta']['schema'] == 'utiltest'
    assert verify_obj['kwargs']['meta']['url'] == 'utiltest://'
    verify_obj = {}
    assert aobj.notify('Hello Another World', title='My Other Title', body_format=common.NotifyFormat.HTML, notify_type=common.NotifyType.WARNING) is True
    assert len(verify_obj) > 0
    assert verify_obj['body'] == 'Hello Another World'
    assert verify_obj['title'] == 'My Other Title'
    assert verify_obj['notify_type'] == common.NotifyType.WARNING
    assert verify_obj['attach'] is None
    assert 'body_format' in verify_obj['kwargs']
    assert verify_obj['kwargs']['body_format'] == common.NotifyFormat.HTML
    assert 'meta' in verify_obj['kwargs']
    assert isinstance(verify_obj['kwargs'], dict)
    assert len(verify_obj['kwargs']['meta']) == 4
    assert 'asset' in verify_obj['kwargs']['meta']
    assert isinstance(verify_obj['kwargs']['meta']['asset'], AppriseAsset)
    assert 'tag' in verify_obj['kwargs']['meta']
    assert isinstance(verify_obj['kwargs']['meta']['tag'], set)
    assert verify_obj['kwargs']['meta']['schema'] == 'utiltest'
    assert verify_obj['kwargs']['meta']['url'] == 'utiltest://'
    assert 'notexc' not in common.NOTIFY_SCHEMA_MAP

    @notify(on='notexc', name='Apprise @notify Exception Handling')
    def my_exception_inline_notify_wrapper(body, title, notify_type, attach, *args, **kwargs):
        if False:
            return 10
        raise ValueError('An exception was thrown!')
    assert 'notexc' in common.NOTIFY_SCHEMA_MAP
    aobj = Apprise()
    assert aobj.add('notexc://') is True
    assert aobj.notify('Exceptions will be thrown!') is False
    del common.NOTIFY_SCHEMA_MAP['utiltest']
    del common.NOTIFY_SCHEMA_MAP['notexc']

def test_notify_complex_decoration():
    if False:
        while True:
            i = 10
    'decorators: Test complex @notify\n    '
    assert 'utiltest' not in common.NOTIFY_SCHEMA_MAP
    verify_obj = {}

    @notify(on='utiltest://user@myhost:23?key=value&NOT=CaseSensitive', name='Apprise @notify Decorator Testing')
    def my_inline_notify_wrapper(body, title, notify_type, attach, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        verify_obj.update({'body': body, 'title': title, 'notify_type': notify_type, 'attach': attach, 'args': args, 'kwargs': kwargs})
    assert 'utiltest' in common.NOTIFY_SCHEMA_MAP
    aobj = Apprise()
    assert aobj.add('utiltest://') is True
    assert len(verify_obj) == 0
    assert aobj.notify('Hello World', title='My Title', attach=(join(TEST_VAR_DIR, 'apprise-test.gif'), join(TEST_VAR_DIR, 'apprise-test.png'))) is True
    assert len(verify_obj) > 0
    assert verify_obj['body'] == 'Hello World'
    assert verify_obj['title'] == 'My Title'
    assert verify_obj['notify_type'] == common.NotifyType.INFO
    assert isinstance(verify_obj['attach'], AppriseAttachment)
    assert len(verify_obj['attach']) == 2
    assert 'body_format' in verify_obj['kwargs']
    assert verify_obj['kwargs']['body_format'] is None
    assert isinstance(verify_obj['kwargs'], dict)
    assert 'meta' in verify_obj['kwargs']
    assert isinstance(verify_obj['kwargs']['meta'], dict)
    assert 'asset' in verify_obj['kwargs']['meta']
    assert isinstance(verify_obj['kwargs']['meta']['asset'], AppriseAsset)
    assert 'tag' in verify_obj['kwargs']['meta']
    assert isinstance(verify_obj['kwargs']['meta']['tag'], set)
    assert len(verify_obj['kwargs']['meta']) == 8
    assert verify_obj['kwargs']['meta']['schema'] == 'utiltest'
    assert verify_obj['kwargs']['meta']['url'].startswith('utiltest://user@myhost:23?')
    assert 'key=value' in verify_obj['kwargs']['meta']['url']
    assert 'not=CaseSensitive' in verify_obj['kwargs']['meta']['url']
    verify_obj = {}
    aobj = Apprise()
    assert aobj.add('utiltest://customhost?key=new&key2=another') is True
    assert len(verify_obj) == 0
    assert aobj.notify('Hello World', title='My Title') is True
    assert len(verify_obj) > 0
    assert verify_obj['body'] == 'Hello World'
    assert verify_obj['title'] == 'My Title'
    assert verify_obj['notify_type'] == common.NotifyType.INFO
    assert verify_obj['attach'] is None
    assert 'body_format' in verify_obj['kwargs']
    assert verify_obj['kwargs']['body_format'] is None
    assert 'meta' in verify_obj['kwargs']
    assert isinstance(verify_obj['kwargs'], dict)
    assert len(verify_obj['kwargs']['meta']) == 8
    assert verify_obj['kwargs']['meta']['schema'] == 'utiltest'
    assert verify_obj['kwargs']['meta']['host'] == 'customhost'
    assert verify_obj['kwargs']['meta']['user'] == 'user'
    assert verify_obj['kwargs']['meta']['port'] == 23
    assert isinstance(verify_obj['kwargs']['meta']['qsd'], dict)
    assert len(verify_obj['kwargs']['meta']['qsd']) == 3
    assert verify_obj['kwargs']['meta']['qsd']['key'] == 'new'
    assert verify_obj['kwargs']['meta']['qsd']['not'] == 'CaseSensitive'
    assert verify_obj['kwargs']['meta']['qsd']['key2'] == 'another'
    assert verify_obj['kwargs']['meta']['url'].startswith('utiltest://user@customhost:23?')
    assert 'key=new' in verify_obj['kwargs']['meta']['url']
    assert 'not=CaseSensitive' in verify_obj['kwargs']['meta']['url']
    assert 'key2=another' in verify_obj['kwargs']['meta']['url']
    del common.NOTIFY_SCHEMA_MAP['utiltest']

def test_notify_multi_instance_decoration(tmpdir):
    if False:
        i = 10
        return i + 15
    'decorators: Test multi-instance @notify\n    '
    assert 'multi' not in common.NOTIFY_SCHEMA_MAP
    verify_obj = []

    @notify(on='multi', name='Apprise @notify Decorator Testing')
    def my_inline_notify_wrapper(body, title, notify_type, attach, meta, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        verify_obj.append({'body': body, 'title': title, 'notify_type': notify_type, 'attach': attach, 'meta': meta, 'args': args, 'kwargs': kwargs})
    assert 'multi' in common.NOTIFY_SCHEMA_MAP
    t = tmpdir.mkdir('multi-test').join('apprise.yml')
    t.write('urls:\n    - multi://user1:pass@hostname\n    - multi://user2:pass2@hostname\n    ')
    ac = AppriseConfig(paths=str(t))
    aobj = Apprise()
    aobj.add(ac)
    assert len(ac) == 1
    assert len(ac.servers()) == 2
    assert len(verify_obj) == 0
    assert aobj.notify('Hello World', title='My Title') is True
    assert len(verify_obj) == 2
    verify_obj = sorted(verify_obj, key=lambda x: x['meta']['user'])
    obj = verify_obj[0]
    assert obj['body'] == 'Hello World'
    assert obj['title'] == 'My Title'
    assert obj['notify_type'] == common.NotifyType.INFO
    meta = obj['meta']
    assert isinstance(meta, dict)
    assert 'body_format' in obj['kwargs']
    assert obj['kwargs']['body_format'] is None
    assert isinstance(obj['kwargs'], dict)
    assert 'asset' in meta
    assert isinstance(meta['asset'], AppriseAsset)
    assert 'tag' in meta
    assert isinstance(meta['tag'], set)
    assert len(meta) == 7
    assert meta['schema'] == 'multi'
    assert meta['host'] == 'hostname'
    assert meta['user'] == 'user1'
    assert meta['password'] == 'pass'
    assert meta['url'] == 'multi://user1:pass@hostname'
    obj = verify_obj[1]
    assert obj['body'] == 'Hello World'
    assert obj['title'] == 'My Title'
    assert obj['notify_type'] == common.NotifyType.INFO
    meta = obj['meta']
    assert isinstance(meta, dict)
    assert 'body_format' in obj['kwargs']
    assert obj['kwargs']['body_format'] is None
    assert isinstance(obj['kwargs'], dict)
    assert 'asset' in meta
    assert isinstance(meta['asset'], AppriseAsset)
    assert 'tag' in meta
    assert isinstance(meta['tag'], set)
    assert len(meta) == 7
    assert meta['schema'] == 'multi'
    assert meta['host'] == 'hostname'
    assert meta['user'] == 'user2'
    assert meta['password'] == 'pass2'
    assert meta['url'] == 'multi://user2:pass2@hostname'
    del common.NOTIFY_SCHEMA_MAP['multi']