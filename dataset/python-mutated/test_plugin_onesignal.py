from apprise.plugins.NotifyOneSignal import NotifyOneSignal
from helpers import AppriseURLTester
from apprise import Apprise
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('onesignal://', {'instance': TypeError}), ('onesignal://:@/', {'instance': TypeError}), ('onesignal://apikey/', {'instance': TypeError}), ('onesignal://appid@%20%20/', {'instance': TypeError}), ('onesignal://appid@apikey/playerid/?lang=X', {'instance': TypeError}), ('onesignal://appid@apikey/', {'instance': NotifyOneSignal, 'notify_response': False}), ('onesignal://appid@apikey/playerid', {'instance': NotifyOneSignal, 'privacy_url': 'onesignal://a...d@a...y/playerid'}), ('onesignal://appid@apikey/player', {'instance': NotifyOneSignal, 'include_image': False}), ('onesignal://appid@apikey/@user?image=no', {'instance': NotifyOneSignal}), ('onesignal://appid@apikey/user@email.com/#seg/player/@user/%20/a', {'instance': NotifyOneSignal}), ('onesignal://appid@apikey?to=#segment,playerid', {'instance': NotifyOneSignal}), ('onesignal://appid@apikey/#segment/@user/?batch=yes', {'instance': NotifyOneSignal}), ('onesignal://appid@apikey/#segment/@user/?batch=no', {'instance': NotifyOneSignal}), ('onesignal://templateid:appid@apikey/playerid', {'instance': NotifyOneSignal}), ('onesignal://appid@apikey/playerid/?lang=es&subtitle=Sub', {'instance': NotifyOneSignal}), ('onesignal://?apikey=abc&template=tp&app=123&to=playerid', {'instance': NotifyOneSignal}), ('onesignal://appid@apikey/#segment/playerid/', {'instance': NotifyOneSignal, 'response': False, 'requests_response_code': 999}), ('onesignal://appid@apikey/#segment/playerid/', {'instance': NotifyOneSignal, 'test_requests_exceptions': True}))

def test_plugin_onesignal_urls():
    if False:
        i = 10
        return i + 15
    '\n    NotifyOneSignal() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

def test_plugin_onesignal_edge_cases():
    if False:
        i = 10
        return i + 15
    '\n    NotifyOneSignal() Batch Validation\n\n    '
    obj = Apprise.instantiate('onesignal://appid@apikey/#segment/@user/playerid/user@email.com/?batch=yes')
    assert isinstance(obj, NotifyOneSignal)
    assert len(obj) == 4
    obj = Apprise.instantiate('onesignal://appid@apikey/@user1/@user2/@user3/@user4/?batch=yes')
    assert isinstance(obj, NotifyOneSignal)
    assert len(obj) == 1
    obj = Apprise.instantiate('onesignal://appid@apikey/@user1/@user2/@user3/@user4/?batch=no')
    assert isinstance(obj, NotifyOneSignal)
    assert len(obj) == 4
    obj = Apprise.instantiate('onesignal://appid@apikey/#segment1/#seg2/#seg3/#seg4/?batch=yes')
    assert isinstance(obj, NotifyOneSignal)
    assert len(obj) == 1
    obj = Apprise.instantiate('onesignal://appid@apikey/#segment1/#seg2/#seg3/#seg4/?batch=no')
    assert isinstance(obj, NotifyOneSignal)
    assert len(obj) == 4
    obj = Apprise.instantiate('onesignal://appid@apikey/pid1/pid2/pid3/pid4/?batch=yes')
    assert isinstance(obj, NotifyOneSignal)
    assert len(obj) == 1
    obj = Apprise.instantiate('onesignal://appid@apikey/pid1/pid2/pid3/pid4/?batch=no')
    assert isinstance(obj, NotifyOneSignal)
    assert len(obj) == 4
    emails = ('abc@yahoo.ca', 'def@yahoo.ca', 'ghi@yahoo.ca', 'jkl@yahoo.ca')
    obj = Apprise.instantiate('onesignal://appid@apikey/{}/?batch=yes'.format('/'.join(emails)))
    assert isinstance(obj, NotifyOneSignal)
    assert len(obj) == 1
    obj = Apprise.instantiate('onesignal://appid@apikey/{}/?batch=no'.format('/'.join(emails)))
    assert isinstance(obj, NotifyOneSignal)
    assert len(obj) == 4
    emails = ('abc@yahoo.ca', 'def@yahoo.ca', 'ghi@yahoo.ca', 'jkl@yahoo.ca')
    users = ('@user1', '@user2', '@user3', '@user4')
    players = ('player1', 'player2', 'player3', 'player4')
    segments = ('#seg1', '#seg2', '#seg3', '#seg4')
    path = '{}/{}/{}/{}'.format('/'.join(emails), '/'.join(users), '/'.join(players), '/'.join(segments))
    obj = Apprise.instantiate('onesignal://appid@apikey/{}/?batch=yes'.format(path))
    assert isinstance(obj, NotifyOneSignal)
    assert len(obj) == 4
    obj = Apprise.instantiate('onesignal://appid@apikey/{}/?batch=no'.format(path))
    assert isinstance(obj, NotifyOneSignal)
    assert len(obj) == 16