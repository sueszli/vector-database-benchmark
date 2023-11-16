from libqtile.widget import idlerpg
online_response = {'player': {'ttl': 1000, 'online': '1', 'unused': '0'}}
offline_response = {'player': {'ttl': 10300, 'online': '0'}}

def test_idlerpg():
    if False:
        while True:
            i = 10
    idler = idlerpg.IdleRPG()
    assert idler.parse(online_response) == 'IdleRPG: online TTL: 0:16:40'
    assert idler.parse(offline_response) == 'IdleRPG: offline TTL: 2:51:40'