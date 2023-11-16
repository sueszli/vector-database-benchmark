from pyboy.utils import WindowEvent
(P10, P11, P12, P13) = range(4)

def reset_bit(x, bit):
    if False:
        while True:
            i = 10
    return x & ~(1 << bit)

def set_bit(x, bit):
    if False:
        for i in range(10):
            print('nop')
    return x | 1 << bit

class Interaction:

    def __init__(self):
        if False:
            print('Hello World!')
        self.directional = 15
        self.standard = 15

    def key_event(self, key):
        if False:
            print('Hello World!')
        _directional = self.directional
        _standard = self.standard
        if key == WindowEvent.PRESS_ARROW_RIGHT:
            self.directional = reset_bit(self.directional, P10)
        elif key == WindowEvent.PRESS_ARROW_LEFT:
            self.directional = reset_bit(self.directional, P11)
        elif key == WindowEvent.PRESS_ARROW_UP:
            self.directional = reset_bit(self.directional, P12)
        elif key == WindowEvent.PRESS_ARROW_DOWN:
            self.directional = reset_bit(self.directional, P13)
        elif key == WindowEvent.PRESS_BUTTON_A:
            self.standard = reset_bit(self.standard, P10)
        elif key == WindowEvent.PRESS_BUTTON_B:
            self.standard = reset_bit(self.standard, P11)
        elif key == WindowEvent.PRESS_BUTTON_SELECT:
            self.standard = reset_bit(self.standard, P12)
        elif key == WindowEvent.PRESS_BUTTON_START:
            self.standard = reset_bit(self.standard, P13)
        elif key == WindowEvent.RELEASE_ARROW_RIGHT:
            self.directional = set_bit(self.directional, P10)
        elif key == WindowEvent.RELEASE_ARROW_LEFT:
            self.directional = set_bit(self.directional, P11)
        elif key == WindowEvent.RELEASE_ARROW_UP:
            self.directional = set_bit(self.directional, P12)
        elif key == WindowEvent.RELEASE_ARROW_DOWN:
            self.directional = set_bit(self.directional, P13)
        elif key == WindowEvent.RELEASE_BUTTON_A:
            self.standard = set_bit(self.standard, P10)
        elif key == WindowEvent.RELEASE_BUTTON_B:
            self.standard = set_bit(self.standard, P11)
        elif key == WindowEvent.RELEASE_BUTTON_SELECT:
            self.standard = set_bit(self.standard, P12)
        elif key == WindowEvent.RELEASE_BUTTON_START:
            self.standard = set_bit(self.standard, P13)
        return (_directional ^ self.directional) & _directional or (_standard ^ self.standard) & _standard

    def pull(self, joystickbyte):
        if False:
            return 10
        P14 = joystickbyte >> 4 & 1
        P15 = joystickbyte >> 5 & 1
        joystickByte = 255 & (joystickbyte | 207)
        if P14 and P15:
            pass
        elif not P14 and (not P15):
            pass
        elif not P14:
            joystickByte &= self.directional
        elif not P15:
            joystickByte &= self.standard
        return joystickByte

    def save_state(self, f):
        if False:
            print('Hello World!')
        f.write(self.directional)
        f.write(self.standard)

    def load_state(self, f, state_version):
        if False:
            return 10
        if state_version >= 7:
            self.directional = f.read()
            self.standard = f.read()
        else:
            self.directional = 15
            self.standard = 15