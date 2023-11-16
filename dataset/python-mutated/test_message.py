from __future__ import annotations
import pytest
pytest
import bokeh.protocol.message as message

def test_create_header(monkeypatch: pytest.MonkeyPatch) -> None:
    if False:
        for i in range(10):
            print('nop')
    message.Message.msgtype = 'msgtype'
    monkeypatch.setattr('bokeh.util.serialization.make_id', lambda : 'msgid')
    header = message.Message.create_header(request_id='bar')
    assert set(header.keys()) == {'msgid', 'msgtype', 'reqid'}
    assert header['msgtype'] == 'msgtype'
    assert header['msgid'] == 'msgid'
    assert header['reqid'] == 'bar'