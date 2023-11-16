from os import getpid
from os.path import basename
import json
import websocket
from threading import Thread, Lock
from mycroft.messagebus.client import MessageBusClient
from mycroft.messagebus.message import Message
bus = None
buffer = None
msgs = []
loaded = []
skill = None
page = None
vars = {}

def start_qml_gui(messagebus, output_buf):
    if False:
        return 10
    global bus
    global buffer
    bus = messagebus
    buffer = output_buf
    log_message('Announcing CLI GUI')
    bus.on('mycroft.gui.port', handle_gui_ready)
    bus.emit(Message('mycroft.gui.connected', {'gui_id': 'cli_' + str(getpid())}))
    log_message('Announced CLI GUI')

def log_message(msg):
    if False:
        print('Hello World!')
    global msgs
    msgs.append(msg)
    if len(msgs) > 20:
        del msgs[0]
    build_output_buffer()

def build_output_buffer():
    if False:
        for i in range(10):
            print('nop')
    global buffer
    buffer.clear()
    try:
        if skill:
            buffer.append('Active Skill: {}'.format(skill))
            buffer.append('Page: {}'.format(basename(page)))
            buffer.append('vars: ')
            for v in vars[skill]:
                buffer.append('     {}: {}'.format(v, vars[skill][v]))
    except Exception as e:
        buffer.append(repr(e))
    buffer.append('-----------------')
    buffer.append('MESSAGES')
    buffer.append('-----------------')
    for m in msgs:
        if len(buffer) > 20:
            return
        buffer.append(m)

def handle_gui_ready(msg):
    if False:
        i = 10
        return i + 15
    gui_id = msg.data.get('gui_id')
    if not gui_id == 'cli_' + str(getpid()):
        return
    port = msg.data.get('port')
    if port:
        log_message('Connecting CLI GUI on ' + str(port))
        ws = websocket.WebSocketApp('ws://0.0.0.0:' + str(port) + '/gui', on_message=on_gui_message, on_error=on_gui_error, on_close=on_gui_close)
        log_message('WS = ' + str(ws))
        event_thread = Thread(target=gui_connect, args=[ws])
        event_thread.setDaemon(True)
        event_thread.start()

def gui_connect(ws):
    if False:
        for i in range(10):
            print('nop')
    log_message('GUI Connected' + str(ws))
    ws.on_open = on_gui_open
    ws.run_forever()

def on_gui_open(ws):
    if False:
        i = 10
        return i + 15
    log_message('GUI Opened')

def on_gui_message(ws, payload):
    if False:
        return 10
    global loaded
    global skill
    global page
    global vars
    try:
        msg = json.loads(payload)
        log_message('Msg: ' + str(payload))
        type = msg.get('type')
        if type == 'mycroft.session.set':
            skill = msg.get('namespace')
            data = msg.get('data')
            if skill not in vars:
                vars[skill] = {}
            for d in data:
                vars[skill][d] = data[d]
        elif type == 'mycroft.session.list.insert':
            skill = msg.get('data')[0]['skill_id']
            loaded.insert(0, [skill, []])
        elif type == 'mycroft.gui.list.insert':
            page = msg['data'][0]['url']
            pos = msg.get('position')
            loaded[0][1].insert(pos, page)
            skill = loaded[0][0]
        elif type == 'mycroft.session.list.move':
            pos = msg.get('from')
            loaded.insert(0, loaded.pop(pos))
        elif type == 'mycroft.events.triggered':
            skill = msg['namespace']
            pos = msg['data']['number']
            for n in loaded:
                if n[0] == skill:
                    page = n[1][pos]
        build_output_buffer()
    except Exception as e:
        log_message(repr(e))
        log_message('Invalid JSON: ' + str(payload))

def on_gui_close(ws):
    if False:
        return 10
    log_message('GUI closed')

def on_gui_error(ws, err):
    if False:
        while True:
            i = 10
    log_message('GUI error: ' + str(err))