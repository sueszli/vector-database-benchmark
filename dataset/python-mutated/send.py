import sys
import json
from .send_func import send

def main():
    if False:
        for i in range(10):
            print('nop')
    'Main function, will run if executed from command line.\n\n    Sends parameters from commandline.\n\n    Param 1:    message string\n    Param 2:    data (json string)\n    '
    if len(sys.argv) == 2:
        message_to_send = sys.argv[1]
        data_to_send = {}
    elif len(sys.argv) == 3:
        message_to_send = sys.argv[1]
        try:
            data_to_send = json.loads(sys.argv[2])
        except BaseException:
            print('Second argument must be a JSON string')
            print('Ex: python -m mycroft.messagebus.send speak \'{"utterance" : "hello"}\'')
            exit()
    else:
        print('Command line interface to the mycroft-core messagebus.')
        print('Usage: python -m mycroft.messagebus.send message')
        print('       python -m mycroft.messagebus.send message JSON-string\n')
        print('Examples: python -m mycroft.messagebus.send system.wifi.setup')
        print('Ex: python -m mycroft.messagebus.send speak \'{"utterance" : "hello"}\'')
        exit()
    send(message_to_send, data_to_send)
if __name__ == '__main__':
    try:
        main()
    except IOError:
        print('Could not connect to websocket, no message sent')