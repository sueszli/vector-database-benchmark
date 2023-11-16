import time
import sys
import struct
import math
import binascii
from bluetooth import set_l2cap_mtu
SX_SELECT = 1 << 0
SX_L3 = 1 << 1
SX_R3 = 1 << 2
SX_START = 1 << 3
SX_DUP = 1 << 4
SX_DRIGHT = 1 << 5
SX_DDOWN = 1 << 6
SX_DLEFT = 1 << 7
SX_L2 = 1 << 8
SX_R2 = 1 << 9
SX_L1 = 1 << 10
SX_R1 = 1 << 11
SX_TRIANGLE = 1 << 12
SX_CIRCLE = 1 << 13
SX_X = 1 << 14
SX_SQUARE = 1 << 15
SX_POWER = 1 << 16
SX_LSTICK_X = 0
SX_LSTICK_Y = 1
SX_RSTICK_X = 2
SX_RSTICK_Y = 3
keymap_sixaxis = {SX_X: ('XG', 'A', 0, 0), SX_CIRCLE: ('XG', 'B', 0, 0), SX_SQUARE: ('XG', 'X', 0, 0), SX_TRIANGLE: ('XG', 'Y', 0, 0), SX_DUP: ('XG', 'dpadup', 0, 0), SX_DDOWN: ('XG', 'dpaddown', 0, 0), SX_DLEFT: ('XG', 'dpadleft', 0, 0), SX_DRIGHT: ('XG', 'dpadright', 0, 0), SX_START: ('XG', 'start', 0, 0), SX_SELECT: ('XG', 'back', 0, 0), SX_R1: ('XG', 'white', 0, 0), SX_R2: ('XG', 'rightanalogtrigger', 6, 1), SX_L2: ('XG', 'leftanalogtrigger', 5, 1), SX_L1: ('XG', 'black', 0, 0), SX_L3: ('XG', 'leftthumbbutton', 0, 0), SX_R3: ('XG', 'rightthumbbutton', 0, 0)}
axismap_sixaxis = {SX_LSTICK_X: ('XG', 'leftthumbstickleft', 'leftthumbstickright'), SX_LSTICK_Y: ('XG', 'leftthumbstickup', 'leftthumbstickdown'), SX_RSTICK_X: ('XG', 'rightthumbstickleft', 'rightthumbstickright'), SX_RSTICK_Y: ('XG', 'rightthumbstickup', 'rightthumbstickdown')}
keymap_sixaxis_keys = keymap_sixaxis.keys()
keymap_sixaxis_keys.sort()
keymap_sixaxis_keys.reverse()

def getkeys(bflags):
    if False:
        while True:
            i = 10
    keys = []
    for k in keymap_sixaxis_keys:
        if k & bflags == k:
            keys.append(k)
            bflags = bflags & ~k
    return keys

def normalize(val):
    if False:
        print('Hello World!')
    upperlimit = 65281
    lowerlimit = 2
    val_range = upperlimit - lowerlimit
    offset = 10000
    val = (val + val_range / 2) % val_range
    upperlimit -= offset
    lowerlimit += offset
    if val < lowerlimit:
        val = lowerlimit
    if val > upperlimit:
        val = upperlimit
    val = (float(val) - offset) / (float(upperlimit) - lowerlimit) * 65535.0
    if val <= 0:
        val = 1
    return val

def normalize_axis(val, deadzone):
    if False:
        i = 10
        return i + 15
    val = float(val) - 127.5
    val = val / 127.5
    if abs(val) < deadzone:
        return 0.0
    if val > 0.0:
        val = (val - deadzone) / (1.0 - deadzone)
    else:
        val = (val + deadzone) / (1.0 - deadzone)
    return 65536.0 * val

def normalize_angle(val, valrange):
    if False:
        i = 10
        return i + 15
    valrange *= 2
    val = val / valrange
    if val > 1.0:
        val = 1.0
    if val < -1.0:
        val = -1.0
    return (val + 0.5) * 65535.0

def average(array):
    if False:
        return 10
    val = 0
    for i in array:
        val += i
    return val / len(array)

def smooth(arr, val):
    if False:
        i = 10
        return i + 15
    cnt = len(arr)
    arr.insert(0, val)
    arr.pop(cnt)
    return average(arr)

def set_l2cap_mtu2(sock, mtu):
    if False:
        for i in range(10):
            print('nop')
    SOL_L2CAP = 6
    L2CAP_OPTIONS = 1
    s = sock.getsockopt(SOL_L2CAP, L2CAP_OPTIONS, 12)
    o = list(struct.unpack('HHHBBBH', s))
    o[0] = o[1] = mtu
    s = struct.pack('HHHBBBH', *o)
    try:
        sock.setsockopt(SOL_L2CAP, L2CAP_OPTIONS, s)
    except:
        print('Warning: Unable to set mtu')

class sixaxis:

    def __init__(self, xbmc, control_sock, interrupt_sock):
        if False:
            return 10
        self.xbmc = xbmc
        self.num_samples = 16
        self.sumx = [0] * self.num_samples
        self.sumy = [0] * self.num_samples
        self.sumr = [0] * self.num_samples
        self.axis_amount = [0, 0, 0, 0]
        self.released = set()
        self.pressed = set()
        self.pending = set()
        self.held = set()
        self.psflags = 0
        self.psdown = 0
        self.mouse_enabled = 0
        set_l2cap_mtu2(control_sock, 64)
        set_l2cap_mtu2(interrupt_sock, 64)
        time.sleep(0.25)
        control_sock.send('SôB\x03\x00\x00')
        data = control_sock.recv(1)
        bytes = [82, 1]
        bytes.extend([0, 0, 0])
        bytes.extend([255, 114])
        bytes.extend([0, 0, 0, 0])
        bytes.extend([2])
        bytes.extend([255, 0, 1, 0, 1])
        bytes.extend([255, 0, 1, 0, 1])
        bytes.extend([255, 0, 1, 0, 1])
        bytes.extend([255, 0, 1, 0, 1])
        bytes.extend([0, 0, 0, 0, 0])
        bytes.extend([0, 0, 0, 0, 0])
        control_sock.send(struct.pack('42B', *bytes))
        data = control_sock.recv(1)

    def __del__(self):
        if False:
            return 10
        self.close()

    def close(self):
        if False:
            print('Hello World!')
        for key in self.held | self.pressed:
            (mapname, action, amount, axis) = keymap_sixaxis[key]
            self.xbmc.send_button_state(map=mapname, button=action, amount=0, down=0, axis=axis)
        self.held = set()
        self.pressed = set()

    def process_socket(self, isock):
        if False:
            i = 10
            return i + 15
        data = isock.recv(50)
        if data == None:
            return False
        return self.process_data(data)

    def process_data(self, data):
        if False:
            for i in range(10):
                print('nop')
        if len(data) < 3:
            return False
        if struct.unpack('BBB', data[0:3]) != (161, 1, 0):
            return False
        if len(data) >= 48:
            v1 = struct.unpack('h', data[42:44])
            v2 = struct.unpack('h', data[44:46])
            v3 = struct.unpack('h', data[46:48])
        else:
            v1 = [0, 0]
            v2 = [0, 0]
            v3 = [0, 0]
        if len(data) >= 50:
            v4 = struct.unpack('h', data[48:50])
        else:
            v4 = [0, 0]
        ax = float(v1[0])
        ay = float(v2[0])
        az = float(v3[0])
        rz = float(v4[0])
        at = math.sqrt(ax * ax + ay * ay + az * az)
        bflags = struct.unpack('<I', data[3:7])[0]
        if len(data) > 27:
            pressure = struct.unpack('BBBBBBBBBBBB', data[15:27])
        else:
            pressure = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        roll = -math.atan2(ax, math.sqrt(ay * ay + az * az))
        pitch = math.atan2(ay, math.sqrt(ax * ax + az * az))
        pitch -= math.radians(20)
        xpos = normalize_angle(roll, math.radians(30))
        ypos = normalize_angle(pitch, math.radians(30))
        axis = struct.unpack('BBBB', data[7:11])
        return self.process_input(bflags, pressure, axis, xpos, ypos)

    def process_input(self, bflags, pressure, axis, xpos, ypos):
        if False:
            return 10
        xval = smooth(self.sumx, xpos)
        yval = smooth(self.sumy, ypos)
        analog = False
        for i in range(4):
            config = axismap_sixaxis[i]
            self.axis_amount[i] = self.send_singleaxis(axis[i], self.axis_amount[i], config[0], config[1], config[2])
            if self.axis_amount[i] != 0:
                analog = True
        if self.mouse_enabled == 1:
            self.xbmc.send_mouse_position(xval, yval)
        if bflags & SX_POWER == SX_POWER:
            if self.psdown:
                if time.time() - self.psdown > 5:
                    for key in self.held | self.pressed:
                        (mapname, action, amount, axis) = keymap_sixaxis[key]
                        self.xbmc.send_button_state(map=mapname, button=action, amount=0, down=0, axis=axis)
                    raise Exception('PS3 Sixaxis powering off, user request')
            else:
                self.psdown = time.time()
        else:
            if self.psdown:
                self.mouse_enabled = 1 - self.mouse_enabled
            self.psdown = 0
        keys = set(getkeys(bflags))
        self.released = (self.pressed | self.held) - keys
        self.held = (self.pressed | self.held) - self.released
        self.pressed = keys - self.held & self.pending
        self.pending = keys - self.held
        for key in self.released:
            (mapname, action, amount, axis) = keymap_sixaxis[key]
            self.xbmc.send_button_state(map=mapname, button=action, amount=0, down=0, axis=axis)
        for key in self.held:
            (mapname, action, amount, axis) = keymap_sixaxis[key]
            if amount > 0:
                amount = pressure[amount - 1] * 256
                self.xbmc.send_button_state(map=mapname, button=action, amount=amount, down=1, axis=axis)
        for key in self.pressed:
            (mapname, action, amount, axis) = keymap_sixaxis[key]
            if amount > 0:
                amount = pressure[amount - 1] * 256
            self.xbmc.send_button_state(map=mapname, button=action, amount=amount, down=1, axis=axis)
        if analog or keys or self.mouse_enabled:
            return True
        else:
            return False

    def send_singleaxis(self, axis, last_amount, mapname, action_min, action_pos):
        if False:
            i = 10
            return i + 15
        amount = normalize_axis(axis, 0.3)
        if last_amount < 0:
            last_action = action_min
        elif last_amount > 0:
            last_action = action_pos
        else:
            last_action = None
        if amount < 0:
            new_action = action_min
        elif amount > 0:
            new_action = action_pos
        else:
            new_action = None
        if last_action and new_action != last_action:
            self.xbmc.send_button_state(map=mapname, button=last_action, amount=0, axis=1)
        if new_action and amount != last_amount:
            self.xbmc.send_button_state(map=mapname, button=new_action, amount=abs(amount), axis=1)
        return amount