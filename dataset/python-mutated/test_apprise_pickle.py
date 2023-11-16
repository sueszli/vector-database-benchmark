from __future__ import print_function
import sys
import pickle
from apprise import Apprise, AppriseAsset, AppriseLocale
import logging
logging.disable(logging.CRITICAL)
sys.dont_write_bytecode = True

def test_apprise_pickle_asset(tmpdir):
    if False:
        print('Hello World!')
    'pickle: AppriseAsset\n    '
    asset = AppriseAsset()
    serialized = pickle.dumps(asset)
    new_asset = pickle.loads(serialized)
    keys = ('app_id', 'app_desc', 'app_url', 'html_notify_map', 'ascii_notify_map', 'default_html_color', 'default_extension', 'theme', 'image_url_mask', 'image_url_logo', 'image_path_mask', 'body_format', 'async_mode', 'interpret_escapes', 'encoding', 'secure_logging', '_recursion')
    for key in keys:
        assert getattr(asset, key) == getattr(new_asset, key)

def test_apprise_pickle_locale(tmpdir):
    if False:
        while True:
            i = 10
    'pickle: AppriseLocale\n    '
    _locale = AppriseLocale.AppriseLocale()
    serialized = pickle.dumps(_locale)
    new_locale = pickle.loads(serialized)
    assert _locale.lang == new_locale.lang
    new_locale.detect_language()

def test_apprise_pickle_core(tmpdir):
    if False:
        i = 10
        return i + 15
    'pickle: Apprise\n    '
    asset = AppriseAsset(app_id='default')
    apobj = Apprise(asset=asset)
    xml_asset = AppriseAsset(app_id='xml')
    apobj.add('json://localhost')
    apobj.add('xml://localhost', asset=xml_asset)
    apobj.add('form://localhost')
    apobj.add('mailto://user:pass@localhost', tag='email')
    serialized = pickle.dumps(apobj)
    new_apobj = pickle.loads(serialized)
    assert len(new_apobj) == 4
    assert apobj[0].app_id == 'default'
    assert apobj[1].app_id == 'xml'
    assert apobj[2].app_id == 'default'
    assert apobj[3].app_id == 'default'
    assert 'email' in apobj[3].tags