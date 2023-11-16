"""
Extension that verifies WPA key by precaptured handshake using cowpatty
"""
import subprocess
from collections import defaultdict
import shlex
import wifiphisher.common.extensions as extensions

def get_process_result(command_string):
    if False:
        print('Hello World!')
    command = shlex.split(command_string)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, universal_newlines=True)
    output = ''
    while True:
        output += process.stdout.readline().strip()
        code = process.poll()
        if code is not None:
            for lines in process.stdout.readlines():
                output += lines.strip()
            break
    return output

def is_valid_handshake_capture(filename):
    if False:
        for i in range(10):
            print('nop')
    command = '/bin/cowpatty -c -r {}'.format(filename)
    output = get_process_result(command)
    return 'Collected all necessary data' in output

class Handshakeverify(object):

    def __init__(self, data):
        if False:
            for i in range(10):
                print('nop')
        self.capt_file = data.args.handshake_capture
        self.essid = data.target_ap_essid
        self.key_file_path = '/tmp/keyfile.tmp'
        self.key = ''
        self.found = False

    def send_channels(self):
        if False:
            print('Hello World!')
        return []

    def get_packet(self, packet):
        if False:
            print('Hello World!')
        return defaultdict(list)

    def send_output(self):
        if False:
            while True:
                i = 10
        if self.key != '' and self.found:
            return ['VALID KEY: ' + self.key]
        elif self.key != '' and (not self.found):
            return ['INVALID KEY ({})'.format(self.key)]
        return ['WAITING FOR WPA KEY POST (ESSID: {})'.format(self.essid)]

    def on_exit(self):
        if False:
            while True:
                i = 10
        pass

    @extensions.register_backend_funcs
    def psk_verify(self, *list_data):
        if False:
            for i in range(10):
                print('nop')
        self.key = list_data[0]
        keyfile = open(self.key_file_path, 'w')
        keyfile.write(self.key + '\n')
        keyfile.close()
        command = '/bin/cowpatty -f "{}" -r "{}" -s "{}"'.format(self.key_file_path, self.capt_file, self.essid)
        self.found = False
        output = get_process_result(command)
        if 'The PSK is' in output:
            self.found = True
        if self.key != '' and self.found:
            return 'success'
        elif self.key != '' and (not self.found):
            return 'fail'
        return 'unknown'