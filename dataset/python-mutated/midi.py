"""pygame.midi
pygame module for interacting with midi input and output.

The midi module can send output to midi devices, and get input
from midi devices.  It can also list midi devices on the system.

Including real midi devices, and virtual ones.

It uses the portmidi library.  Is portable to which ever platforms
portmidi supports (currently windows, OSX, and linux).

This uses pyportmidi for now, but may use its own bindings at some
point in the future.  The pyportmidi bindings are included with pygame.

New in pygame 1.9.0.
"""
import math
import atexit
import pygame
import pygame.locals
import pygame.pypm as _pypm
MIDIIN = pygame.locals.MIDIIN
MIDIOUT = pygame.locals.MIDIOUT
__all__ = ['Input', 'MIDIIN', 'MIDIOUT', 'MidiException', 'Output', 'get_count', 'get_default_input_id', 'get_default_output_id', 'get_device_info', 'init', 'midis2events', 'quit', 'get_init', 'time', 'frequency_to_midi', 'midi_to_frequency', 'midi_to_ansi_note']
__theclasses__ = ['Input', 'Output']

def _module_init(state=None):
    if False:
        print('Hello World!')
    if state is not None:
        _module_init.value = state
        return state
    try:
        _module_init.value
    except AttributeError:
        return False
    return _module_init.value

def init():
    if False:
        return 10
    'initialize the midi module\n    pygame.midi.init(): return None\n\n    Call the initialisation function before using the midi module.\n\n    It is safe to call this more than once.\n    '
    if not _module_init():
        _pypm.Initialize()
        _module_init(True)
        atexit.register(quit)

def quit():
    if False:
        return 10
    "uninitialize the midi module\n    pygame.midi.quit(): return None\n\n\n    Called automatically atexit if you don't call it.\n\n    It is safe to call this function more than once.\n    "
    if _module_init():
        _pypm.Terminate()
        _module_init(False)

def get_init():
    if False:
        i = 10
        return i + 15
    'returns True if the midi module is currently initialized\n    pygame.midi.get_init(): return bool\n\n    Returns True if the pygame.midi module is currently initialized.\n\n    New in pygame 1.9.5.\n    '
    return _module_init()

def _check_init():
    if False:
        print('Hello World!')
    if not _module_init():
        raise RuntimeError('pygame.midi not initialised.')

def get_count():
    if False:
        for i in range(10):
            print('nop')
    'gets the number of devices.\n    pygame.midi.get_count(): return num_devices\n\n\n    Device ids range from 0 to get_count() -1\n    '
    _check_init()
    return _pypm.CountDevices()

def get_default_input_id():
    if False:
        return 10
    'gets default input device number\n    pygame.midi.get_default_input_id(): return default_id\n\n\n    Return the default device ID or -1 if there are no devices.\n    The result can be passed to the Input()/Output() class.\n\n    On the PC, the user can specify a default device by\n    setting an environment variable. For example, to use device #1.\n\n        set PM_RECOMMENDED_INPUT_DEVICE=1\n\n    The user should first determine the available device ID by using\n    the supplied application "testin" or "testout".\n\n    In general, the registry is a better place for this kind of info,\n    and with USB devices that can come and go, using integers is not\n    very reliable for device identification. Under Windows, if\n    PM_RECOMMENDED_OUTPUT_DEVICE (or PM_RECOMMENDED_INPUT_DEVICE) is\n    *NOT* found in the environment, then the default device is obtained\n    by looking for a string in the registry under:\n        HKEY_LOCAL_MACHINE/SOFTWARE/PortMidi/Recommended_Input_Device\n    and HKEY_LOCAL_MACHINE/SOFTWARE/PortMidi/Recommended_Output_Device\n    for a string. The number of the first device with a substring that\n    matches the string exactly is returned. For example, if the string\n    in the registry is "USB", and device 1 is named\n    "In USB MidiSport 1x1", then that will be the default\n    input because it contains the string "USB".\n\n    In addition to the name, get_device_info() returns "interf", which\n    is the interface name. (The "interface" is the underlying software\n    system or API used by PortMidi to access devices. Examples are\n    MMSystem, DirectX (not implemented), ALSA, OSS (not implemented), etc.)\n    At present, the only Win32 interface is "MMSystem", the only Linux\n    interface is "ALSA", and the only Max OS X interface is "CoreMIDI".\n    To specify both the interface and the device name in the registry,\n    separate the two with a comma and a space, e.g.:\n        MMSystem, In USB MidiSport 1x1\n    In this case, the string before the comma must be a substring of\n    the "interf" string, and the string after the space must be a\n    substring of the "name" name string in order to match the device.\n\n    Note: in the current release, the default is simply the first device\n    (the input or output device with the lowest PmDeviceID).\n    '
    _check_init()
    return _pypm.GetDefaultInputDeviceID()

def get_default_output_id():
    if False:
        print('Hello World!')
    'gets default output device number\n    pygame.midi.get_default_output_id(): return default_id\n\n\n    Return the default device ID or -1 if there are no devices.\n    The result can be passed to the Input()/Output() class.\n\n    On the PC, the user can specify a default device by\n    setting an environment variable. For example, to use device #1.\n\n        set PM_RECOMMENDED_OUTPUT_DEVICE=1\n\n    The user should first determine the available device ID by using\n    the supplied application "testin" or "testout".\n\n    In general, the registry is a better place for this kind of info,\n    and with USB devices that can come and go, using integers is not\n    very reliable for device identification. Under Windows, if\n    PM_RECOMMENDED_OUTPUT_DEVICE (or PM_RECOMMENDED_INPUT_DEVICE) is\n    *NOT* found in the environment, then the default device is obtained\n    by looking for a string in the registry under:\n        HKEY_LOCAL_MACHINE/SOFTWARE/PortMidi/Recommended_Input_Device\n    and HKEY_LOCAL_MACHINE/SOFTWARE/PortMidi/Recommended_Output_Device\n    for a string. The number of the first device with a substring that\n    matches the string exactly is returned. For example, if the string\n    in the registry is "USB", and device 1 is named\n    "In USB MidiSport 1x1", then that will be the default\n    input because it contains the string "USB".\n\n    In addition to the name, get_device_info() returns "interf", which\n    is the interface name. (The "interface" is the underlying software\n    system or API used by PortMidi to access devices. Examples are\n    MMSystem, DirectX (not implemented), ALSA, OSS (not implemented), etc.)\n    At present, the only Win32 interface is "MMSystem", the only Linux\n    interface is "ALSA", and the only Max OS X interface is "CoreMIDI".\n    To specify both the interface and the device name in the registry,\n    separate the two with a comma and a space, e.g.:\n        MMSystem, In USB MidiSport 1x1\n    In this case, the string before the comma must be a substring of\n    the "interf" string, and the string after the space must be a\n    substring of the "name" name string in order to match the device.\n\n    Note: in the current release, the default is simply the first device\n    (the input or output device with the lowest PmDeviceID).\n    '
    _check_init()
    return _pypm.GetDefaultOutputDeviceID()

def get_device_info(an_id):
    if False:
        return 10
    "returns information about a midi device\n    pygame.midi.get_device_info(an_id): return (interf, name,\n                                                input, output,\n                                                opened)\n\n    interf - a byte string describing the device interface, eg b'ALSA'.\n    name - a byte string for the name of the device, eg b'Midi Through Port-0'\n    input - 0, or 1 if the device is an input device.\n    output - 0, or 1 if the device is an output device.\n    opened - 0, or 1 if the device is opened.\n\n    If the id is out of range, the function returns None.\n    "
    _check_init()
    return _pypm.GetDeviceInfo(an_id)

class Input:
    """Input is used to get midi input from midi devices.
    Input(device_id)
    Input(device_id, buffer_size)

    buffer_size - the number of input events to be buffered waiting to
      be read using Input.read()
    """

    def __init__(self, device_id, buffer_size=4096):
        if False:
            return 10
        '\n        The buffer_size specifies the number of input events to be buffered\n        waiting to be read using Input.read().\n        '
        _check_init()
        if device_id == -1:
            raise MidiException('Device id is -1, not a valid output id.  -1 usually means there were no default Output devices.')
        try:
            result = get_device_info(device_id)
        except TypeError:
            raise TypeError('an integer is required')
        except OverflowError:
            raise OverflowError('long int too large to convert to int')
        if result:
            (_, _, is_input, is_output, _) = result
            if is_input:
                try:
                    self._input = _pypm.Input(device_id, buffer_size)
                except TypeError:
                    raise TypeError('an integer is required')
                self.device_id = device_id
            elif is_output:
                raise MidiException('Device id given is not a valid input id, it is an output id.')
            else:
                raise MidiException('Device id given is not a valid input id.')
        else:
            raise MidiException('Device id invalid, out of range.')

    def _check_open(self):
        if False:
            while True:
                i = 10
        if self._input is None:
            raise MidiException('midi not open.')

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        'closes a midi stream, flushing any pending buffers.\n        Input.close(): return None\n\n        PortMidi attempts to close open streams when the application\n        exits -- this is particularly difficult under Windows.\n        '
        _check_init()
        if self._input is not None:
            self._input.Close()
        self._input = None

    def read(self, num_events):
        if False:
            print('Hello World!')
        'reads num_events midi events from the buffer.\n        Input.read(num_events): return midi_event_list\n\n        Reads from the Input buffer and gives back midi events.\n        [[[status,data1,data2,data3],timestamp],\n         [[status,data1,data2,data3],timestamp],...]\n        '
        _check_init()
        self._check_open()
        return self._input.Read(num_events)

    def poll(self):
        if False:
            for i in range(10):
                print('nop')
        "returns true if there's data, or false if not.\n        Input.poll(): return Bool\n\n        raises a MidiException on error.\n        "
        _check_init()
        self._check_open()
        result = self._input.Poll()
        if result == _pypm.TRUE:
            return True
        if result == _pypm.FALSE:
            return False
        err_text = _pypm.GetErrorText(result)
        raise MidiException((result, err_text))

class Output:
    """Output is used to send midi to an output device
    Output(device_id)
    Output(device_id, latency = 0)
    Output(device_id, buffer_size = 4096)
    Output(device_id, latency, buffer_size)

    The buffer_size specifies the number of output events to be
    buffered waiting for output.  (In some cases -- see below --
    PortMidi does not buffer output at all and merely passes data
    to a lower-level API, in which case buffersize is ignored.)

    latency is the delay in milliseconds applied to timestamps to determine
    when the output should actually occur. (If latency is < 0, 0 is
    assumed.)

    If latency is zero, timestamps are ignored and all output is delivered
    immediately. If latency is greater than zero, output is delayed until
    the message timestamp plus the latency. (NOTE: time is measured
    relative to the time source indicated by time_proc. Timestamps are
    absolute, not relative delays or offsets.) In some cases, PortMidi
    can obtain better timing than your application by passing timestamps
    along to the device driver or hardware. Latency may also help you
    to synchronize midi data to audio data by matching midi latency to
    the audio buffer latency.

    """

    def __init__(self, device_id, latency=0, buffer_size=256):
        if False:
            return 10
        'Output(device_id)\n        Output(device_id, latency = 0)\n        Output(device_id, buffer_size = 4096)\n        Output(device_id, latency, buffer_size)\n\n        The buffer_size specifies the number of output events to be\n        buffered waiting for output.  (In some cases -- see below --\n        PortMidi does not buffer output at all and merely passes data\n        to a lower-level API, in which case buffersize is ignored.)\n\n        latency is the delay in milliseconds applied to timestamps to determine\n        when the output should actually occur. (If latency is < 0, 0 is\n        assumed.)\n\n        If latency is zero, timestamps are ignored and all output is delivered\n        immediately. If latency is greater than zero, output is delayed until\n        the message timestamp plus the latency. (NOTE: time is measured\n        relative to the time source indicated by time_proc. Timestamps are\n        absolute, not relative delays or offsets.) In some cases, PortMidi\n        can obtain better timing than your application by passing timestamps\n        along to the device driver or hardware. Latency may also help you\n        to synchronize midi data to audio data by matching midi latency to\n        the audio buffer latency.\n        '
        _check_init()
        self._aborted = 0
        if device_id == -1:
            raise MidiException('Device id is -1, not a valid output id.  -1 usually means there were no default Output devices.')
        try:
            result = get_device_info(device_id)
        except TypeError:
            raise TypeError('an integer is required')
        except OverflowError:
            raise OverflowError('long int too large to convert to int')
        if result:
            (_, _, is_input, is_output, _) = result
            if is_output:
                try:
                    self._output = _pypm.Output(device_id, latency, buffer_size)
                except TypeError:
                    raise TypeError('an integer is required')
                self.device_id = device_id
            elif is_input:
                raise MidiException('Device id given is not a valid output id, it is an input id.')
            else:
                raise MidiException('Device id given is not a valid output id.')
        else:
            raise MidiException('Device id invalid, out of range.')

    def _check_open(self):
        if False:
            while True:
                i = 10
        if self._output is None:
            raise MidiException('midi not open.')
        if self._aborted:
            raise MidiException('midi aborted.')

    def close(self):
        if False:
            while True:
                i = 10
        'closes a midi stream, flushing any pending buffers.\n        Output.close(): return None\n\n        PortMidi attempts to close open streams when the application\n        exits -- this is particularly difficult under Windows.\n        '
        _check_init()
        if self._output is not None:
            self._output.Close()
        self._output = None

    def abort(self):
        if False:
            for i in range(10):
                print('nop')
        'terminates outgoing messages immediately\n        Output.abort(): return None\n\n        The caller should immediately close the output port;\n        this call may result in transmission of a partial midi message.\n        There is no abort for Midi input because the user can simply\n        ignore messages in the buffer and close an input device at\n        any time.\n        '
        _check_init()
        if self._output:
            self._output.Abort()
        self._aborted = 1

    def write(self, data):
        if False:
            print('Hello World!')
        'writes a list of midi data to the Output\n        Output.write(data)\n\n        writes series of MIDI information in the form of a list:\n             write([[[status <,data1><,data2><,data3>],timestamp],\n                    [[status <,data1><,data2><,data3>],timestamp],...])\n        <data> fields are optional\n        example: choose program change 1 at time 20000 and\n        send note 65 with velocity 100 500 ms later.\n             write([[[0xc0,0,0],20000],[[0x90,60,100],20500]])\n        notes:\n          1. timestamps will be ignored if latency = 0.\n          2. To get a note to play immediately, send MIDI info with\n             timestamp read from function Time.\n          3. understanding optional data fields:\n               write([[[0xc0,0,0],20000]]) is equivalent to\n               write([[[0xc0],20000]])\n\n        Can send up to 1024 elements in your data list, otherwise an\n         IndexError exception is raised.\n        '
        _check_init()
        self._check_open()
        self._output.Write(data)

    def write_short(self, status, data1=0, data2=0):
        if False:
            print('Hello World!')
        'write_short(status <, data1><, data2>)\n        Output.write_short(status)\n        Output.write_short(status, data1 = 0, data2 = 0)\n\n        output MIDI information of 3 bytes or less.\n        data fields are optional\n        status byte could be:\n             0xc0 = program change\n             0x90 = note on\n             etc.\n             data bytes are optional and assumed 0 if omitted\n        example: note 65 on with velocity 100\n             write_short(0x90,65,100)\n        '
        _check_init()
        self._check_open()
        self._output.WriteShort(status, data1, data2)

    def write_sys_ex(self, when, msg):
        if False:
            i = 10
            return i + 15
        "writes a timestamped system-exclusive midi message.\n        Output.write_sys_ex(when, msg)\n\n        msg - can be a *list* or a *string*\n        when - a timestamp in milliseconds\n        example:\n          (assuming o is an onput MIDI stream)\n            o.write_sys_ex(0,'\\xF0\\x7D\\x10\\x11\\x12\\x13\\xF7')\n          is equivalent to\n            o.write_sys_ex(pygame.midi.time(),\n                           [0xF0,0x7D,0x10,0x11,0x12,0x13,0xF7])\n        "
        _check_init()
        self._check_open()
        self._output.WriteSysEx(when, msg)

    def note_on(self, note, velocity, channel=0):
        if False:
            print('Hello World!')
        'turns a midi note on.  Note must be off.\n        Output.note_on(note, velocity, channel=0)\n\n        note is an integer from 0 to 127\n        velocity is an integer from 0 to 127\n        channel is an integer from 0 to 15\n\n        Turn a note on in the output stream.  The note must already\n        be off for this to work correctly.\n        '
        if not 0 <= channel <= 15:
            raise ValueError('Channel not between 0 and 15.')
        self.write_short(144 + channel, note, velocity)

    def note_off(self, note, velocity=0, channel=0):
        if False:
            return 10
        'turns a midi note off.  Note must be on.\n        Output.note_off(note, velocity=0, channel=0)\n\n        note is an integer from 0 to 127\n        velocity is an integer from 0 to 127 (release velocity)\n        channel is an integer from 0 to 15\n\n        Turn a note off in the output stream.  The note must already\n        be on for this to work correctly.\n        '
        if not 0 <= channel <= 15:
            raise ValueError('Channel not between 0 and 15.')
        self.write_short(128 + channel, note, velocity)

    def set_instrument(self, instrument_id, channel=0):
        if False:
            return 10
        'select an instrument for a channel, with a value between 0 and 127\n        Output.set_instrument(instrument_id, channel=0)\n\n        Also called "patch change" or "program change".\n        '
        if not 0 <= instrument_id <= 127:
            raise ValueError(f'Undefined instrument id: {instrument_id}')
        if not 0 <= channel <= 15:
            raise ValueError('Channel not between 0 and 15.')
        self.write_short(192 + channel, instrument_id)

    def pitch_bend(self, value=0, channel=0):
        if False:
            print('Hello World!')
        'modify the pitch of a channel.\n        Output.pitch_bend(value=0, channel=0)\n\n        Adjust the pitch of a channel.  The value is a signed integer\n        from -8192 to +8191.  For example, 0 means "no change", +4096 is\n        typically a semitone higher, and -8192 is 1 whole tone lower (though\n        the musical range corresponding to the pitch bend range can also be\n        changed in some synthesizers).\n\n        If no value is given, the pitch bend is returned to "no change".\n        '
        if not 0 <= channel <= 15:
            raise ValueError('Channel not between 0 and 15.')
        if not -8192 <= value <= 8191:
            raise ValueError(f'Pitch bend value must be between -8192 and +8191, not {value}.')
        value = value + 8192
        lsb = value & 127
        msb = value >> 7
        self.write_short(224 + channel, lsb, msb)

def time():
    if False:
        for i in range(10):
            print('nop')
    'returns the current time in ms of the PortMidi timer\n    pygame.midi.time(): return time\n\n    The time is reset to 0, when the module is inited.\n    '
    _check_init()
    return _pypm.Time()

def midis2events(midis, device_id):
    if False:
        print('Hello World!')
    'converts midi events to pygame events\n    pygame.midi.midis2events(midis, device_id): return [Event, ...]\n\n    Takes a sequence of midi events and returns list of pygame events.\n    '
    evs = []
    for midi in midis:
        ((status, data1, data2, data3), timestamp) = midi
        event = pygame.event.Event(MIDIIN, status=status, data1=data1, data2=data2, data3=data3, timestamp=timestamp, vice_id=device_id)
        evs.append(event)
    return evs

class MidiException(Exception):
    """exception that pygame.midi functions and classes can raise
    MidiException(errno)
    """

    def __init__(self, value):
        if False:
            while True:
                i = 10
        super().__init__(value)
        self.parameter = value

    def __str__(self):
        if False:
            print('Hello World!')
        return repr(self.parameter)

def frequency_to_midi(frequency):
    if False:
        while True:
            i = 10
    'converts a frequency into a MIDI note.\n\n    Rounds to the closest midi note.\n\n    ::Examples::\n\n    >>> frequency_to_midi(27.5)\n    21\n    >>> frequency_to_midi(36.7)\n    26\n    >>> frequency_to_midi(4186.0)\n    108\n    '
    return int(round(69 + 12 * math.log(frequency / 440.0) / math.log(2)))

def midi_to_frequency(midi_note):
    if False:
        print('Hello World!')
    'Converts a midi note to a frequency.\n\n    ::Examples::\n\n    >>> midi_to_frequency(21)\n    27.5\n    >>> midi_to_frequency(26)\n    36.7\n    >>> midi_to_frequency(108)\n    4186.0\n    '
    return round(440.0 * 2 ** ((midi_note - 69) * (1.0 / 12.0)), 1)

def midi_to_ansi_note(midi_note):
    if False:
        for i in range(10):
            print('nop')
    "returns the Ansi Note name for a midi number.\n\n    ::Examples::\n\n    >>> midi_to_ansi_note(21)\n    'A0'\n    >>> midi_to_ansi_note(102)\n    'F#7'\n    >>> midi_to_ansi_note(108)\n    'C8'\n    "
    notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    num_notes = 12
    note_name = notes[int((midi_note - 21) % num_notes)]
    note_number = (midi_note - 12) // num_notes
    return f'{note_name}{note_number}'