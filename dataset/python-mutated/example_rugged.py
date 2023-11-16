from tinkerforge.ip_connection import IPConnection
from tinkerforge.bricklet_lcd_20x4 import LCD20x4
from tinkerforge.bricklet_temperature import Temperature

class ExampleRugged:
    HOST = 'localhost'
    PORT = 4223

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.lcd = None
        self.temp = None
        self.ipcon = IPConnection()
        self.ipcon.register_callback(IPConnection.CALLBACK_ENUMERATE, self.cb_enumerate)
        self.ipcon.register_callback(IPConnection.CALLBACK_CONNECTED, self.cb_connected)
        self.ipcon.connect(ExampleRugged.HOST, ExampleRugged.PORT)
        self.ipcon.enumerate()

    def cb_button_pressed(self, button):
        if False:
            for i in range(10):
                print('nop')
        if self.lcd:
            if button == 0:
                if self.lcd.is_backlight_on():
                    self.lcd.backlight_off()
                else:
                    self.lcd.backlight_on()

    def cb_temperature(self, temperature):
        if False:
            i = 10
            return i + 15
        if self.lcd:
            self.lcd.clear_display()
            s = 'Temperature: {0:.2f}{1:c}C'.format(temperature / 100.0, 223)
            self.lcd.write_line(0, 0, s)

    def cb_enumerate(self, uid, connected_uid, position, hardware_version, firmware_version, device_identifier, enumeration_type):
        if False:
            return 10
        if enumeration_type == IPConnection.ENUMERATION_TYPE_CONNECTED or enumeration_type == IPConnection.ENUMERATION_TYPE_AVAILABLE:
            if device_identifier == LCD20x4.DEVICE_IDENTIFIER:
                self.lcd = LCD20x4(uid, self.ipcon)
                self.lcd.register_callback(self.lcd.CALLBACK_BUTTON_PRESSED, self.cb_button_pressed)
                self.lcd.clear_display()
                self.lcd.backlight_on()
            if device_identifier == Temperature.DEVICE_IDENTIFIER:
                self.temp = Temperature(uid, self.ipcon)
                self.temp.register_callback(self.temp.CALLBACK_TEMPERATURE, self.cb_temperature)
                self.temp.set_temperature_callback_period(50)

    def cb_connected(self, connected_reason):
        if False:
            for i in range(10):
                print('nop')
        self.ipcon.enumerate()
if __name__ == '__main__':
    ExampleRugged()
    raw_input('Press key to exit\n')