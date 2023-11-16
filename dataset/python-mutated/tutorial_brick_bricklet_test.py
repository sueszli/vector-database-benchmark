HOST = 'localhost'
PORT = 4223
UID_DC = 'XXYYZZ'
UID_POTI = 'XYZ'
from tinkerforge.ip_connection import IPConnection
from tinkerforge.brick_dc import BrickDC
from tinkerforge.bricklet_rotary_poti import BrickletRotaryPoti
dc = None

def cb_position(position):
    if False:
        while True:
            i = 10
    velocity = 32767 // 2 * position // 150
    print('Set Position/Velocity: ' + str(position) + '/' + str(velocity))
    dc.set_velocity(velocity)
if __name__ == '__main__':
    ipcon = IPConnection()
    dc = BrickDC(UID_DC, ipcon)
    poti = BrickletRotaryPoti(UID_POTI, ipcon)
    ipcon.connect(HOST, PORT)
    poti.set_position_callback_period(50)
    poti.register_callback(poti.CALLBACK_POSITION, cb_position)
    dc.enable()
    dc.set_acceleration(65535)
    input('Press Enter to exit\n')
    dc.set_acceleration(16384)
    dc.set_velocity(0)
    time.sleep(2)
    dc.disable()
    ipcon.disconnect()