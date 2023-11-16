import numpy as _np
import dbus as _dbus
from openrazer.client import constants as c

def clamp_ubyte(value):
    if False:
        print('Hello World!')
    '\n    Clamp a value to 0->255\n\n    Aka -3453\n    :param value: Integer\n    :type value: int\n\n    :return: Integer 0->255\n    :rtype: int\n    '
    if value > 255:
        value = 255
    elif value < 0:
        value = 0
    return value

class BaseRazerFX(object):

    def __init__(self, serial: str, capabilities: dict, daemon_dbus=None):
        if False:
            while True:
                i = 10
        self._capabilities = capabilities
        if daemon_dbus is None:
            session_bus = _dbus.SessionBus()
            daemon_dbus = session_bus.get_object('org.razer', '/org/razer/device/{0}'.format(serial))
        self._dbus = daemon_dbus

    def has(self, capability: str) -> bool:
        if False:
            i = 10
            return i + 15
        "\n        Convenience function to check capability\n\n        Uses the main device capability list and automatically prefixes 'lighting_'\n        :param capability: Device capability\n        :type capability: str\n\n        :return: True or False\n        :rtype: bool\n        "
        return self._capabilities.get('lighting_' + capability, False)

class RazerFX(BaseRazerFX):

    def __init__(self, serial: str, capabilities: dict, daemon_dbus=None, matrix_dims=(-1, -1)):
        if False:
            print('Hello World!')
        super().__init__(serial, capabilities, daemon_dbus)
        self._lighting_dbus = _dbus.Interface(self._dbus, 'razer.device.lighting.chroma')
        if self.has('led_matrix') and all([dim >= 1 for dim in matrix_dims]):
            self.advanced = RazerAdvancedFX(serial, capabilities, daemon_dbus=self._dbus, matrix_dims=matrix_dims)
        else:
            self.advanced = None
        if self.has('led_matrix') and self.has('ripple'):
            self._custom_lighting_dbus = _dbus.Interface(self._dbus, 'razer.device.lighting.custom')
        else:
            self._custom_lighting_dbus = None
        self.misc = MiscLighting(serial, capabilities, self._dbus)

    @property
    def effect(self) -> str:
        if False:
            print('Hello World!')
        '\n        Get current effect\n\n        :return: Effect name ("static", "spectrum", etc.)\n        :rtype: str\n        '
        return self._lighting_dbus.getEffect()

    @property
    def colors(self) -> bytearray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get current effect colors\n\n        :return: Effect colors (an array of 9 bytes, for 3 colors in RGB format)\n        :rtype: bytearray\n        '
        return bytes(self._lighting_dbus.getEffectColors())

    @property
    def speed(self) -> int:
        if False:
            while True:
                i = 10
        '\n        Get current effect speed\n\n        :return: Effect speed (a value between 0 and 3)\n        :rtype: int\n        '
        return self._lighting_dbus.getEffectSpeed()

    @property
    def wave_dir(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get current wave direction\n\n        :return: Wave direction (WAVE_LEFT or WAVE_RIGHT)\n        :rtype: int\n        '
        return self._lighting_dbus.getWaveDir()

    def none(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        No effect\n\n        :return: True if success, False otherwise\n        :rtype: bool\n        '
        if self.has('none'):
            self._lighting_dbus.setNone()
            return True
        return False

    def spectrum(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Spectrum effect\n\n        :return: True if success, False otherwise\n        :rtype: bool\n        '
        if self.has('spectrum'):
            self._lighting_dbus.setSpectrum()
            return True
        return False

    def wave(self, direction: int) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Wave effect\n\n        :param direction: Wave direction either WAVE_RIGHT (0x01) or WAVE_LEFT (0x02)\n        :type direction: int\n\n        :return: True if success, False otherwise\n        :rtype: bool\n\n        :raises ValueError: If direction is invalid\n        '
        if direction not in (c.WAVE_LEFT, c.WAVE_RIGHT):
            raise ValueError('Direction must be WAVE_RIGHT (0x01) or WAVE_LEFT (0x02)')
        if self.has('wave'):
            self._lighting_dbus.setWave(direction)
            return True
        return False

    def wheel(self, direction: int) -> bool:
        if False:
            print('Hello World!')
        '\n        Wheel effect\n\n        :param direction: Wheel direction either WHEEL_RIGHT or WHEEL_LEFT\n        :type direction: int\n\n        :return: True if success, False otherwise\n        :rtype: bool\n\n        :raises ValueError: If direction is invalid\n        '
        if direction not in (c.WHEEL_LEFT, c.WHEEL_RIGHT):
            raise ValueError('Direction must be WHEEL_RIGHT (0x01) or WHEEL_LEFT (0x02)')
        if self.has('wheel'):
            self._lighting_dbus.setWheel(direction)
            return True
        return False

    def static(self, red: int, green: int, blue: int) -> bool:
        if False:
            return 10
        '\n        Static effect\n\n        :param red: Red component. Must be 0->255\n        :type red: int\n\n        :param green: Green component. Must be 0->255\n        :type green: int\n\n        :param blue: Blue component. Must be 0->255\n        :type blue: int\n\n        :return: True if success, False otherwise\n        :rtype: bool\n\n        :raises ValueError: If parameters are invalid\n        '
        if not isinstance(red, int):
            raise ValueError('Red is not an integer')
        if not isinstance(green, int):
            raise ValueError('Green is not an integer')
        if not isinstance(blue, int):
            raise ValueError('Blue is not an integer')
        if self.has('static'):
            red = clamp_ubyte(red)
            green = clamp_ubyte(green)
            blue = clamp_ubyte(blue)
            self._lighting_dbus.setStatic(red, green, blue)
            return True
        return False

    def reactive(self, red: int, green: int, blue: int, time: int) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Reactive effect\n\n        :param time: Reactive speed. One of REACTIVE_500MS, REACTIVE_1000MS, REACTIVE_1500MS or REACTIVE_2000MS\n        :param time: int\n\n        :param red: Red component. Must be 0->255\n        :type red: int\n\n        :param green: Green component. Must be 0->255\n        :type green: int\n\n        :param blue: Blue component. Must be 0->255\n        :type blue: int\n\n        :return: True if success, False otherwise\n        :rtype: bool\n\n        :raises ValueError: If parameters are invalid\n        '
        if time not in (c.REACTIVE_500MS, c.REACTIVE_1000MS, c.REACTIVE_1500MS, c.REACTIVE_2000MS):
            raise ValueError('Time not one of REACTIVE_500MS, REACTIVE_1000MS, REACTIVE_1500MS or REACTIVE_2000MS')
        if not isinstance(red, int):
            raise ValueError('Red is not an integer')
        if not isinstance(green, int):
            raise ValueError('Green is not an integer')
        if not isinstance(blue, int):
            raise ValueError('Blue is not an integer')
        if self.has('reactive'):
            red = clamp_ubyte(red)
            green = clamp_ubyte(green)
            blue = clamp_ubyte(blue)
            self._lighting_dbus.setReactive(red, green, blue, time)
            return True
        return False

    def breath_single(self, red: int, green: int, blue: int) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Breath effect - single colour\n\n        :param red: Red component. Must be 0->255\n        :type red: int\n\n        :param green: Green component. Must be 0->255\n        :type green: int\n\n        :param blue: Blue component. Must be 0->255\n        :type blue: int\n\n        :return: True if success, False otherwise\n        :rtype: bool\n\n        :raises ValueError: If parameters are invalid\n        '
        if not isinstance(red, int):
            raise ValueError('Red is not an integer')
        if not isinstance(green, int):
            raise ValueError('Green is not an integer')
        if not isinstance(blue, int):
            raise ValueError('Blue is not an integer')
        if self.has('breath_single'):
            red = clamp_ubyte(red)
            green = clamp_ubyte(green)
            blue = clamp_ubyte(blue)
            self._lighting_dbus.setBreathSingle(red, green, blue)
            return True
        return False

    def breath_dual(self, red: int, green: int, blue: int, red2: int, green2: int, blue2: int) -> bool:
        if False:
            print('Hello World!')
        '\n        Breath effect - single colour\n\n        :param red: First red component. Must be 0->255\n        :type red: int\n\n        :param green: First green component. Must be 0->255\n        :type green: int\n\n        :param blue: First blue component. Must be 0->255\n        :type blue: int\n\n        :param red2: Second red component. Must be 0->255\n        :type red2: int\n\n        :param green2: Second green component. Must be 0->255\n        :type green2: int\n\n        :param blue2: Second blue component. Must be 0->255\n        :type blue2: int\n\n        :return: True if success, False otherwise\n        :rtype: bool\n\n        :raises ValueError: If parameters are invalid\n        '
        if not isinstance(red, int):
            raise ValueError('Primary red is not an integer')
        if not isinstance(green, int):
            raise ValueError('Primary green is not an integer')
        if not isinstance(blue, int):
            raise ValueError('Primary blue is not an integer')
        if not isinstance(red2, int):
            raise ValueError('Secondary red is not an integer')
        if not isinstance(green2, int):
            raise ValueError('Secondary green is not an integer')
        if not isinstance(blue2, int):
            raise ValueError('Secondary blue is not an integer')
        if self.has('breath_dual'):
            red = clamp_ubyte(red)
            green = clamp_ubyte(green)
            blue = clamp_ubyte(blue)
            red2 = clamp_ubyte(red2)
            green2 = clamp_ubyte(green2)
            blue2 = clamp_ubyte(blue2)
            self._lighting_dbus.setBreathDual(red, green, blue, red2, green2, blue2)
            return True
        return False

    def breath_triple(self, red: int, green: int, blue: int, red2: int, green2: int, blue2: int, red3: int, green3: int, blue3: int) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Breath effect - single colour\n\n        :param red: First red component. Must be 0->255\n        :type red: int\n\n        :param green: First green component. Must be 0->255\n        :type green: int\n\n        :param blue: First blue component. Must be 0->255\n        :type blue: int\n\n        :param red2: Second red component. Must be 0->255\n        :type red2: int\n\n        :param green2: Second green component. Must be 0->255\n        :type green2: int\n\n        :param blue2: Second blue component. Must be 0->255\n        :type blue2: int\n\n        :param red3: Second red component. Must be 0->255\n        :type red3: int\n\n        :param green3: Second green component. Must be 0->255\n        :type green3: int\n\n        :param blue3: Second blue component. Must be 0->255\n        :type blue3: int\n\n        :return: True if success, False otherwise\n        :rtype: bool\n\n        :raises ValueError: If parameters are invalid\n        '
        if not isinstance(red, int):
            raise ValueError('Primary red is not an integer')
        if not isinstance(green, int):
            raise ValueError('Primary green is not an integer')
        if not isinstance(blue, int):
            raise ValueError('Primary blue is not an integer')
        if not isinstance(red2, int):
            raise ValueError('Secondary red is not an integer')
        if not isinstance(green2, int):
            raise ValueError('Secondary green is not an integer')
        if not isinstance(blue2, int):
            raise ValueError('Secondary blue is not an integer')
        if not isinstance(red3, int):
            raise ValueError('Tertiary red is not an integer')
        if not isinstance(green3, int):
            raise ValueError('Tertiary green is not an integer')
        if not isinstance(blue3, int):
            raise ValueError('Tertiary blue is not an integer')
        if self.has('breath_triple'):
            red = clamp_ubyte(red)
            green = clamp_ubyte(green)
            blue = clamp_ubyte(blue)
            red2 = clamp_ubyte(red2)
            green2 = clamp_ubyte(green2)
            blue2 = clamp_ubyte(blue2)
            red3 = clamp_ubyte(red3)
            green3 = clamp_ubyte(green3)
            blue3 = clamp_ubyte(blue3)
            self._lighting_dbus.setBreathTriple(red, green, blue, red2, green2, blue2, red3, green3, blue3)
            return True
        return False

    def breath_random(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Breath effect - random colours\n\n        :return: True if success, False otherwise\n        :rtype: bool\n\n        :raises ValueError: If parameters are invalid\n        '
        if self.has('breath_random'):
            self._lighting_dbus.setBreathRandom()
            return True
        return False

    def ripple(self, red: int, green: int, blue: int, refreshrate: float=c.RIPPLE_REFRESH_RATE) -> bool:
        if False:
            return 10
        '\n        Set the Ripple Effect.\n\n        The refresh rate should be set to about 0.05 for a decent effect\n        :param red: Red RGB component\n        :rtype red: int\n\n        :param green: Green RGB component\n        :type green: int\n\n        :param blue: Blue RGB component\n        :type blue: int\n\n        :param refreshrate: Effect refresh rate\n        :type refreshrate: float\n\n        :return: True if success, False otherwise\n        :rtype: bool\n\n        :raises ValueError: If arguments are invalid\n        '
        if not isinstance(refreshrate, float):
            raise ValueError('Refresh rate is not a float')
        if not isinstance(red, int):
            raise ValueError('Red is not an integer')
        if not isinstance(green, int):
            raise ValueError('Green is not an integer')
        if not isinstance(blue, int):
            raise ValueError('Blue is not an integer')
        if self.has('ripple'):
            red = clamp_ubyte(red)
            green = clamp_ubyte(green)
            blue = clamp_ubyte(blue)
            self._custom_lighting_dbus.setRipple(red, green, blue, refreshrate)
            return True
        return False

    def ripple_random(self, refreshrate: float=c.RIPPLE_REFRESH_RATE):
        if False:
            return 10
        '\n        Set the Ripple Effect with random colours\n\n        The refresh rate should be set to about 0.05 for a decent effect\n        :param refreshrate: Effect refresh rate\n        :type refreshrate: float\n\n        :return: True if success, False otherwise\n        :rtype: bool\n\n        :raises ValueError: If arguments are invalid\n        '
        if not isinstance(refreshrate, float):
            raise ValueError('Refresh rate is not a float')
        if self.has('ripple_random'):
            self._custom_lighting_dbus.setRippleRandomColour(refreshrate)
            return True
        return False

    def starlight_single(self, red: int, green: int, blue: int, time: int) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Starlight effect\n\n        :param red: Red component. Must be 0->255\n        :type red: int\n\n        :param green: Green component. Must be 0->255\n        :type green: int\n\n        :param blue: Blue component. Must be 0->255\n        :type blue: int\n\n        :param time: Starlight speed\n        :type time: int\n\n        :return: True if success, False otherwise\n        :rtype: bool\n\n        :raises ValueError: If parameters are invalid\n        '
        if time not in (c.STARLIGHT_FAST, c.STARLIGHT_NORMAL, c.STARLIGHT_SLOW):
            raise ValueError('Time not one of STARLIGHT_FAST, STARLIGHT_NORMAL or STARLIGHT_SLOW')
        if not isinstance(red, int):
            raise ValueError('Red is not an integer')
        if not isinstance(green, int):
            raise ValueError('Green is not an integer')
        if not isinstance(blue, int):
            raise ValueError('Blue is not an integer')
        if self.has('starlight_single'):
            red = clamp_ubyte(red)
            green = clamp_ubyte(green)
            blue = clamp_ubyte(blue)
            self._lighting_dbus.setStarlightSingle(red, green, blue, time)
            return True
        return False

    def starlight_dual(self, red: int, green: int, blue: int, red2: int, green2: int, blue2: int, time: int) -> bool:
        if False:
            print('Hello World!')
        '\n        Starlight effect\n\n        :param red: Red component. Must be 0->255\n        :type red: int\n\n        :param green: Green component. Must be 0->255\n        :type green: int\n\n        :param blue: Blue component. Must be 0->255\n        :type blue: int\n\n        :param red2: Red component. Must be 0->255\n        :type red2: int\n\n        :param green2: Green component. Must be 0->255\n        :type green2: int\n\n        :param blue2: Blue component. Must be 0->255\n        :type blue2: int\n\n        :param time: Starlight speed\n        :type time: int\n\n        :return: True if success, False otherwise\n        :rtype: bool\n\n        :raises ValueError: If parameters are invalid\n        '
        if time not in (c.STARLIGHT_FAST, c.STARLIGHT_NORMAL, c.STARLIGHT_SLOW):
            raise ValueError('Time not one of STARLIGHT_FAST, STARLIGHT_NORMAL or STARLIGHT_SLOW')
        if not isinstance(red, int):
            raise ValueError('Red is not an integer')
        if not isinstance(green, int):
            raise ValueError('Green is not an integer')
        if not isinstance(blue, int):
            raise ValueError('Blue is not an integer')
        if not isinstance(red2, int):
            raise ValueError('Red 2 is not an integer')
        if not isinstance(green2, int):
            raise ValueError('Green 2 is not an integer')
        if not isinstance(blue2, int):
            raise ValueError('Blue 2 is not an integer')
        if self.has('starlight_dual'):
            red = clamp_ubyte(red)
            green = clamp_ubyte(green)
            blue = clamp_ubyte(blue)
            red2 = clamp_ubyte(red2)
            green2 = clamp_ubyte(green2)
            blue2 = clamp_ubyte(blue2)
            self._lighting_dbus.setStarlightDual(red, green, blue, red2, green2, blue2, time)
            return True
        return False

    def starlight_random(self, time: int) -> bool:
        if False:
            return 10
        '\n        Starlight effect\n\n        :param time: Starlight speed\n        :type time: int\n\n        :return: True if success, False otherwise\n        :rtype: bool\n\n        :raises ValueError: If parameters are invalid\n        '
        if time not in (c.STARLIGHT_FAST, c.STARLIGHT_NORMAL, c.STARLIGHT_SLOW):
            raise ValueError('Time not one of STARLIGHT_FAST, STARLIGHT_NORMAL or STARLIGHT_SLOW')
        if self.has('starlight_random'):
            self._lighting_dbus.setStarlightRandom(time)
            return True
        return False

class RazerAdvancedFX(BaseRazerFX):

    def __init__(self, serial: str, capabilities: dict, daemon_dbus=None, matrix_dims=(-1, -1)):
        if False:
            i = 10
            return i + 15
        super().__init__(serial, capabilities, daemon_dbus)
        self._capabilities = capabilities
        if not all([dim >= 1 for dim in matrix_dims]):
            raise ValueError('Matrix dimensions cannot contain -1')
        if daemon_dbus is None:
            session_bus = _dbus.SessionBus()
            daemon_dbus = session_bus.get_object('org.razer', '/org/razer/device/{0}'.format(serial))
        self._matrix_dims = matrix_dims
        self._lighting_dbus = _dbus.Interface(daemon_dbus, 'razer.device.lighting.chroma')
        self.matrix = Frame(matrix_dims)

    @property
    def cols(self):
        if False:
            i = 10
            return i + 15
        '\n        Number of columns in matrix\n\n        :return: Columns\n        :rtype: int\n        '
        return self._matrix_dims[1]

    @property
    def rows(self):
        if False:
            while True:
                i = 10
        '\n        Number of rows in matrix\n\n        :return: Rows\n        :rtype: int\n        '
        return self._matrix_dims[0]

    def _draw(self, ba):
        if False:
            for i in range(10):
                print('nop')
        self._lighting_dbus.setKeyRow(ba)
        self._lighting_dbus.setCustom()

    def draw(self):
        if False:
            return 10
        "\n        Draw what's in the current frame buffer\n        "
        self._draw(bytes(self.matrix))

    def draw_fb_or(self):
        if False:
            return 10
        self._draw(bytes(self.matrix.draw_with_fb_or()))

    def set_key(self, column_id, rgb, row_id=0):
        if False:
            i = 10
            return i + 15
        if self.has('led_single'):
            if isinstance(rgb, (tuple, list)) and len(rgb) == 3 and all([isinstance(component, int) for component in rgb]):
                if row_id < self._matrix_dims[0] and column_id < self._matrix_dims[1]:
                    self._lighting_dbus.setKey(row_id, column_id, [clamp_ubyte(component) for component in rgb])
                else:
                    raise ValueError('Row or column out of bounds. Max dimensions are: {0},{1}'.format(*self._matrix_dims))
            else:
                raise ValueError('RGB must be an RGB tuple')

    def restore(self):
        if False:
            print('Hello World!')
        '\n        Restore the device to the last effect\n        '
        self._lighting_dbus.restoreLastEffect()

class SingleLed(BaseRazerFX):

    def __init__(self, serial: str, capabilities: dict, daemon_dbus=None, led_name='logo'):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(serial, capabilities, daemon_dbus)
        self._led_name = led_name
        self._lighting_dbus = _dbus.Interface(self._dbus, 'razer.device.lighting.{0}'.format(led_name))

    def _shas(self, item):
        if False:
            return 10
        return self.has('{0}_{1}'.format(self._led_name, item))

    def _getattr(self, name):
        if False:
            print('Hello World!')
        attr = name.replace('#', self._led_name.title().replace('_', ''))
        return getattr(self._lighting_dbus, attr, None)

    @property
    def active(self) -> bool:
        if False:
            while True:
                i = 10
        func = self._getattr('get#Active')
        if func is not None:
            return func()
        else:
            return False

    @active.setter
    def active(self, value: bool):
        if False:
            for i in range(10):
                print('nop')
        func = self._getattr('set#Active')
        if func is not None:
            if value:
                func(True)
            else:
                func(False)

    @property
    def effect(self) -> str:
        if False:
            return 10
        '\n        Get current effect\n\n        :return: Effect name ("static", "spectrum", etc.)\n        :rtype: str\n        '
        return str(self._getattr('get#Effect')())

    @property
    def colors(self) -> bytearray:
        if False:
            return 10
        '\n        Get current effect colors\n\n        :return: Effect colors (an array of 9 bytes, for 3 colors in RGB format)\n        :rtype: bytearray\n        '
        return bytes(self._getattr('get#EffectColors')())

    @property
    def speed(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Get current effect speed\n\n        :return: Effect speed (a value between 0 and 3)\n        :rtype: int\n        '
        return int(self._getattr('get#EffectSpeed')())

    @property
    def wave_dir(self) -> int:
        if False:
            print('Hello World!')
        '\n        Get current wave direction\n\n        :return: Wave direction (WAVE_LEFT or WAVE_RIGHT)\n        :rtype: int\n        '
        return int(self._getattr('get#WaveDir')())

    @property
    def brightness(self):
        if False:
            return 10
        if self._shas('brightness'):
            return float(self._getattr('get#Brightness')())
        return 0.0

    @brightness.setter
    def brightness(self, brightness: float):
        if False:
            return 10
        if self._shas('brightness'):
            if not isinstance(brightness, (float, int)):
                raise ValueError('Brightness is not a float')
            if brightness > 100:
                brightness = 100.0
            elif brightness < 0:
                brightness = 0.0
            self._getattr('set#Brightness')(brightness)

    def blinking(self, red: int, green: int, blue: int) -> bool:
        if False:
            print('Hello World!')
        if not isinstance(red, int):
            raise ValueError('Red is not an integer')
        if not isinstance(green, int):
            raise ValueError('Green is not an integer')
        if not isinstance(blue, int):
            raise ValueError('Blue is not an integer')
        if self._shas('blinking'):
            red = clamp_ubyte(red)
            green = clamp_ubyte(green)
            blue = clamp_ubyte(blue)
            self._getattr('set#Blinking')(red, green, blue)
            return True
        return False

    def pulsate(self, red: int, green: int, blue: int) -> bool:
        if False:
            while True:
                i = 10
        if not isinstance(red, int):
            raise ValueError('Red is not an integer')
        if not isinstance(green, int):
            raise ValueError('Green is not an integer')
        if not isinstance(blue, int):
            raise ValueError('Blue is not an integer')
        if self._shas('pulsate'):
            red = clamp_ubyte(red)
            green = clamp_ubyte(green)
            blue = clamp_ubyte(blue)
            self._getattr('set#Pulsate')(red, green, blue)
            return True
        return False

    def static(self, red: int, green: int, blue: int) -> bool:
        if False:
            return 10
        if not isinstance(red, int):
            raise ValueError('Red is not an integer')
        if not isinstance(green, int):
            raise ValueError('Green is not an integer')
        if not isinstance(blue, int):
            raise ValueError('Blue is not an integer')
        if self._shas('static'):
            red = clamp_ubyte(red)
            green = clamp_ubyte(green)
            blue = clamp_ubyte(blue)
            self._getattr('set#Static')(red, green, blue)
            return True
        return False

    def wave(self, direction: int) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if direction not in (c.WAVE_LEFT, c.WAVE_RIGHT):
            raise ValueError('Direction must be WAVE_RIGHT (0x01) or WAVE_LEFT (0x02)')
        if self._shas('wave'):
            self._getattr('set#Wave')(direction)
            return True
        return False

    def none(self) -> bool:
        if False:
            print('Hello World!')
        if self._shas('none'):
            self._getattr('set#None')()
            return True
        return False

    def on(self) -> bool:
        if False:
            while True:
                i = 10
        if self._shas('on'):
            self._getattr('set#On')()
            return True
        return False

    def spectrum(self) -> bool:
        if False:
            i = 10
            return i + 15
        if self._shas('spectrum'):
            self._getattr('set#Spectrum')()
            return True
        return False

    def reactive(self, red: int, green: int, blue: int, time: int) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Reactive effect\n\n        :param time: Reactive speed. One of REACTIVE_500MS, REACTIVE_1000MS, REACTIVE_1500MS or REACTIVE_2000MS\n        :param time: int\n\n        :param red: Red component. Must be 0->255\n        :type red: int\n\n        :param green: Green component. Must be 0->255\n        :type green: int\n\n        :param blue: Blue component. Must be 0->255\n        :type blue: int\n\n        :return: True if success, False otherwise\n        :rtype: bool\n\n        :raises ValueError: If parameters are invalid\n        '
        if time not in (c.REACTIVE_500MS, c.REACTIVE_1000MS, c.REACTIVE_1500MS, c.REACTIVE_2000MS):
            raise ValueError('Time not one of REACTIVE_500MS, REACTIVE_1000MS, REACTIVE_1500MS or REACTIVE_2000MS')
        if not isinstance(red, int):
            raise ValueError('Red is not an integer')
        if not isinstance(green, int):
            raise ValueError('Green is not an integer')
        if not isinstance(blue, int):
            raise ValueError('Blue is not an integer')
        if self._shas('reactive'):
            red = clamp_ubyte(red)
            green = clamp_ubyte(green)
            blue = clamp_ubyte(blue)
            self._getattr('set#Reactive')(red, green, blue, time)
            return True
        return False

    def breath_single(self, red: int, green: int, blue: int) -> bool:
        if False:
            while True:
                i = 10
        '\n        Breath effect - single colour\n\n        :param red: Red component. Must be 0->255\n        :type red: int\n\n        :param green: Green component. Must be 0->255\n        :type green: int\n\n        :param blue: Blue component. Must be 0->255\n        :type blue: int\n\n        :return: True if success, False otherwise\n        :rtype: bool\n\n        :raises ValueError: If parameters are invalid\n        '
        if not isinstance(red, int):
            raise ValueError('Red is not an integer')
        if not isinstance(green, int):
            raise ValueError('Green is not an integer')
        if not isinstance(blue, int):
            raise ValueError('Blue is not an integer')
        if self._shas('breath_single'):
            red = clamp_ubyte(red)
            green = clamp_ubyte(green)
            blue = clamp_ubyte(blue)
            self._getattr('set#BreathSingle')(red, green, blue)
            return True
        return False

    def breath_dual(self, red: int, green: int, blue: int, red2: int, green2: int, blue2: int) -> bool:
        if False:
            while True:
                i = 10
        '\n        Breath effect - single colour\n\n        :param red: First red component. Must be 0->255\n        :type red: int\n\n        :param green: First green component. Must be 0->255\n        :type green: int\n\n        :param blue: First blue component. Must be 0->255\n        :type blue: int\n\n        :param red2: Second red component. Must be 0->255\n        :type red2: int\n\n        :param green2: Second green component. Must be 0->255\n        :type green2: int\n\n        :param blue2: Second blue component. Must be 0->255\n        :type blue2: int\n\n        :return: True if success, False otherwise\n        :rtype: bool\n\n        :raises ValueError: If parameters are invalid\n        '
        if not isinstance(red, int):
            raise ValueError('Primary red is not an integer')
        if not isinstance(green, int):
            raise ValueError('Primary green is not an integer')
        if not isinstance(blue, int):
            raise ValueError('Primary blue is not an integer')
        if not isinstance(red2, int):
            raise ValueError('Secondary red is not an integer')
        if not isinstance(green2, int):
            raise ValueError('Secondary green is not an integer')
        if not isinstance(blue2, int):
            raise ValueError('Secondary blue is not an integer')
        if self._shas('breath_dual'):
            red = clamp_ubyte(red)
            green = clamp_ubyte(green)
            blue = clamp_ubyte(blue)
            red2 = clamp_ubyte(red2)
            green2 = clamp_ubyte(green2)
            blue2 = clamp_ubyte(blue2)
            self._getattr('set#BreathDual')(red, green, blue, red2, green2, blue2)
            return True
        return False

    def breath_random(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Breath effect - random colours\n\n        :return: True if success, False otherwise\n        :rtype: bool\n\n        :raises ValueError: If parameters are invalid\n        '
        if self._shas('breath_random'):
            self._getattr('set#BreathRandom')()
            return True
        return False

    def breath_mono(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Breath effect - mono colour\n\n        :return: True if success, False otherwise\n        :rtype: bool\n\n        :raises ValueError: If parameters are invalid\n        '
        if self._shas('breath_mono'):
            self._getattr('set#BreathMono')()
            return True
        return False

class MiscLighting(BaseRazerFX):

    def __init__(self, serial: str, capabilities: dict, daemon_dbus=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(serial, capabilities, daemon_dbus)
        self._lighting_dbus = _dbus.Interface(self._dbus, 'razer.device.lighting.logo')
        if self.has('logo'):
            self._logo = SingleLed(serial, capabilities, daemon_dbus, 'logo')
        else:
            self._logo = None
        if self.has('scroll'):
            self._scroll = SingleLed(serial, capabilities, daemon_dbus, 'scroll')
        else:
            self._scroll = None
        if self.has('left'):
            self._left = SingleLed(serial, capabilities, daemon_dbus, 'left')
        else:
            self._left = None
        if self.has('right'):
            self._right = SingleLed(serial, capabilities, daemon_dbus, 'right')
        else:
            self._right = None
        if self.has('charging'):
            self._charging = SingleLed(serial, capabilities, daemon_dbus, 'charging')
        else:
            self._charging = None
        if self.has('fast_charging'):
            self._fast_charging = SingleLed(serial, capabilities, daemon_dbus, 'fast_charging')
        else:
            self._fast_charging = None
        if self.has('fully_charged'):
            self._fully_charged = SingleLed(serial, capabilities, daemon_dbus, 'fully_charged')
        else:
            self._fully_charged = None
        if self.has('backlight'):
            self._backlight = SingleLed(serial, capabilities, daemon_dbus, 'backlight')
        else:
            self._backlight = None

    @property
    def logo(self):
        if False:
            i = 10
            return i + 15
        return self._logo

    @property
    def scroll_wheel(self):
        if False:
            i = 10
            return i + 15
        return self._scroll

    @property
    def left(self):
        if False:
            while True:
                i = 10
        return self._left

    @property
    def right(self):
        if False:
            for i in range(10):
                print('nop')
        return self._right

    @property
    def charging(self):
        if False:
            for i in range(10):
                print('nop')
        return self._charging

    @property
    def fast_charging(self):
        if False:
            i = 10
            return i + 15
        return self._fast_charging

    @property
    def fully_charged(self):
        if False:
            while True:
                i = 10
        return self._fully_charged

    @property
    def backlight(self):
        if False:
            return 10
        return self._backlight

class Frame(object):
    """
    Class to represent the RGB matrix of the keyboard. So to animate you'd use multiple frames
    """

    def __init__(self, dimensions):
        if False:
            print('Hello World!')
        (self._rows, self._cols) = dimensions
        self._components = 3
        self._matrix = None
        self._fb1 = None
        self.reset()

    def __getitem__(self, key: tuple) -> tuple:
        if False:
            for i in range(10):
                print('nop')
        '\n        Method to allow a slice to get an RGB tuple\n\n        :param key: Key, must be y,x tuple\n        :type key: tuple\n\n        :return: RGB tuple\n        :rtype: tuple\n\n        :raises AssertionError: If key is invalid\n        '
        assert isinstance(key, tuple), 'Key is not a tuple'
        assert 0 <= key[0] < self._rows, 'Row out of bounds'
        assert 0 <= key[1] < self._cols, 'Column out of bounds'
        return tuple(self._matrix[:, key[0], key[1]])

    def __setitem__(self, key: tuple, rgb: tuple):
        if False:
            i = 10
            return i + 15
        '\n        Method to allow a slice to set an RGB tuple\n\n        :param key: Key, must be y,x tuple\n        :type key: tuple\n\n        :param rgb: RGB tuple\n        :type rgb: tuple\n\n        :raises AssertionError: If key is invalid\n        '
        assert isinstance(key, tuple), 'Key is not a tuple'
        assert 0 <= key[0] < self._rows, 'Row out of bounds'
        assert 0 <= key[1] < self._cols, 'Column out of bounds'
        assert isinstance(rgb, (list, tuple)) and len(rgb) == 3, 'Value must be a tuple,list of 3 RGB components'
        self._matrix[:, key[0], key[1]] = rgb

    def __bytes__(self) -> bytes:
        if False:
            print('Hello World!')
        '\n        When bytes() is ran on the class will return a binary capable of being sent to the driver\n\n        :return: Driver binary payload\n        :rtype: bytes\n        '
        return b''.join([self.row_binary(row_id) for row_id in range(0, self._rows)])

    def reset(self):
        if False:
            i = 10
            return i + 15
        '\n        Init/Clear the matrix\n        '
        if self._matrix is None:
            self._matrix = _np.zeros((self._components, self._rows, self._cols), 'uint8')
            self._fb1 = _np.copy(self._matrix)
        else:
            self._matrix.fill(0)

    def set(self, y: int, x: int, rgb: tuple):
        if False:
            return 10
        '\n        Method to allow a slice to set an RGB tuple\n\n        :param y: Row\n        :type y: int\n\n        :param x: Column\n        :type x: int\n\n        :param rgb: RGB tuple\n        :type rgb: tuple\n\n        :raises AssertionError: If key is invalid\n        '
        self.__setitem__((y, x), rgb)

    def get(self, y: int, x: int) -> list:
        if False:
            while True:
                i = 10
        '\n        Method to allow a slice to get an RGB tuple\n\n        :param y: Row\n        :type y: int\n\n        :param x: Column\n        :type x: int\n\n        :return rgb: RGB tuple\n        :return rgb: tuple\n\n        :raises AssertionError: If key is invalid\n        '
        return self.__getitem__((y, x))

    def row_binary(self, row_id: int) -> bytes:
        if False:
            print('Hello World!')
        '\n        Get binary payload for 1 row which is compatible with the driver\n\n        :param row_id: Row ID\n        :type row_id: int\n\n        :return: Binary payload\n        :rtype: bytes\n        '
        assert 0 <= row_id < self._rows, 'Row out of bounds'
        start = 0
        end = self._cols - 1
        return row_id.to_bytes(1, byteorder='big') + start.to_bytes(1, byteorder='big') + end.to_bytes(1, byteorder='big') + self._matrix[:, row_id].tobytes(order='F')

    def to_binary(self):
        if False:
            return 10
        '\n        Get the whole binary for the keyboard to be sent to the driver.\n\n        :return: Driver binary payload\n        :rtype: bytes\n        '
        return bytes(self)

    def to_framebuffer(self):
        if False:
            i = 10
            return i + 15
        self._fb1 = _np.copy(self._matrix)

    def to_framebuffer_or(self):
        if False:
            return 10
        self._fb1 = _np.bitwise_or(self._fb1, self._matrix)

    def draw_with_fb_or(self):
        if False:
            i = 10
            return i + 15
        self._matrix = _np.bitwise_or(self._fb1, self._matrix)
        return bytes(self)