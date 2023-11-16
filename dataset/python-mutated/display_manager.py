""" DisplayManager

This module provides basic "state" for the visual representation associated
with this Mycroft instance.  The current states are:
   ActiveSkill - The skill that last interacted with the display via the
                 Enclosure API.

Currently, a wakeword sets the ActiveSkill to "wakeword", which will auto
clear after 10 seconds.

A skill is set to Active when it matches an intent, outputs audio, or
changes the display via the EnclosureAPI()

A skill is automatically cleared from Active two seconds after audio
output is spoken, or 2 seconds after resetting the display.

So it is common to have '' as the active skill.
"""
import json
from threading import Thread, Timer
import os
from mycroft.messagebus.client import MessageBusClient
from mycroft.util import get_ipc_directory
from mycroft.util.log import LOG

def _write_data(dictionary):
    if False:
        return 10
    " Writes the dictionary of state data to the IPC directory.\n\n    Args:\n        dictionary (dict): information to place in the 'disp_info' file\n    "
    managerIPCDir = os.path.join(get_ipc_directory(), 'managers')
    path = os.path.join(managerIPCDir, 'disp_info')
    permission = 'r+' if os.path.isfile(path) else 'w+'
    if permission == 'w+' and os.path.isdir(managerIPCDir) is False:
        os.makedirs(managerIPCDir)
        os.chmod(managerIPCDir, 511)
    try:
        with open(path, permission) as dispFile:
            if os.stat(str(dispFile.name)).st_size != 0:
                data = json.load(dispFile)
            else:
                data = {}
                LOG.info('Display Manager is creating ' + dispFile.name)
            for key in dictionary:
                data[key] = dictionary[key]
            dispFile.seek(0)
            dispFile.write(json.dumps(data))
            dispFile.truncate()
        os.chmod(path, 511)
    except Exception as e:
        LOG.error(e)
        LOG.error('Error found in display manager file, deleting...')
        os.remove(path)
        _write_data(dictionary)

def _read_data():
    if False:
        i = 10
        return i + 15
    ' Writes the dictionary of state data from the IPC directory.\n    Returns:\n        dict: loaded state information\n    '
    managerIPCDir = os.path.join(get_ipc_directory(), 'managers')
    path = os.path.join(managerIPCDir, 'disp_info')
    permission = 'r' if os.path.isfile(path) else 'w+'
    if permission == 'w+' and os.path.isdir(managerIPCDir) is False:
        os.makedirs(managerIPCDir)
    data = {}
    try:
        with open(path, permission) as dispFile:
            if os.stat(str(dispFile.name)).st_size != 0:
                data = json.load(dispFile)
    except Exception as e:
        LOG.error(e)
        os.remove(path)
        _read_data()
    return data

class DisplayManager:
    """ The Display manager handles the basic state of the display,
    be it a mark-1 or a mark-2 or even a future Mark-3.
    """

    def __init__(self, name=None):
        if False:
            for i in range(10):
                print('nop')
        self.name = name or ''

    def set_active(self, skill_name=None):
        if False:
            while True:
                i = 10
        ' Sets skill name as active in the display Manager\n        Args:\n            string: skill_name\n        '
        name = skill_name if skill_name is not None else self.name
        _write_data({'active_skill': name})

    def get_active(self):
        if False:
            i = 10
            return i + 15
        " Get the currenlty active skill from the display manager\n        Returns:\n            string: The active skill's name\n        "
        data = _read_data()
        active_skill = ''
        if 'active_skill' in data:
            active_skill = data['active_skill']
        return active_skill

    def remove_active(self):
        if False:
            return 10
        ' Clears the active skill '
        LOG.debug('Removing active skill...')
        _write_data({'active_skill': ''})

def init_display_manager_bus_connection():
    if False:
        return 10
    ' Connects the display manager to the messagebus '
    LOG.info('Connecting display manager to messagebus')
    display_manager = DisplayManager()
    should_remove = [True]

    def check_flag(flag):
        if False:
            i = 10
            return i + 15
        if flag[0] is True:
            display_manager.remove_active()

    def set_delay(event=None):
        if False:
            print('Hello World!')
        should_remove[0] = True
        Timer(2, check_flag, [should_remove]).start()

    def set_remove_flag(event=None):
        if False:
            while True:
                i = 10
        should_remove[0] = False

    def connect():
        if False:
            i = 10
            return i + 15
        bus.run_forever()

    def remove_wake_word():
        if False:
            for i in range(10):
                print('nop')
        data = _read_data()
        if 'active_skill' in data and data['active_skill'] == 'wakeword':
            display_manager.remove_active()

    def set_wakeword_skill(event=None):
        if False:
            i = 10
            return i + 15
        display_manager.set_active('wakeword')
        Timer(10, remove_wake_word).start()
    bus = MessageBusClient()
    bus.on('recognizer_loop:audio_output_end', set_delay)
    bus.on('recognizer_loop:audio_output_start', set_remove_flag)
    bus.on('recognizer_loop:record_begin', set_wakeword_skill)
    event_thread = Thread(target=connect)
    event_thread.setDaemon(True)
    event_thread.start()