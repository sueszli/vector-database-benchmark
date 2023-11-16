from mitmproxy.test import taddons

def test_load_script(tdata):
    if False:
        for i in range(10):
            print('nop')
    with taddons.context() as tctx:
        s = tctx.script(tdata.path('mitmproxy/data/addonscripts/recorder/recorder.py'))
        assert s