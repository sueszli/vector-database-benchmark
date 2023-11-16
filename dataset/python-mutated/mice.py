import dbus as _dbus
from openrazer.client.devices import RazerDevice as __RazerDevice
from openrazer.client.macro import RazerMacro as _RazerMacro

class RazerMouse(__RazerDevice):
    _MACRO_CLASS = _RazerMacro

    @property
    def max_dpi(self) -> int:
        if False:
            i = 10
            return i + 15
        "\n        Gets max DPI\n\n        :return: Max DPI, if device does not have DPI it'll return None\n        :rtype: int or None\n        "
        if self.has('dpi'):
            return int(self._dbus_interfaces['dpi'].maxDPI())
        else:
            return None

    @property
    def available_dpi(self) -> list:
        if False:
            print('Hello World!')
        '\n        Gets the available DPI\n\n        :return: Available DPI, if device has only a couple of fixed possible DPI values\n        :rtype: list or None\n        '
        if self.has('available_dpi'):
            dbuslist = self._dbus_interfaces['dpi'].availableDPI()
            return [int(d) for d in dbuslist]
        else:
            return None

    @property
    def dpi(self) -> tuple:
        if False:
            i = 10
            return i + 15
        '\n        Get mouse DPI\n\n        Will return a tuple\n        :return: DPI (500, 500)\n        :rtype: tuple\n\n        :raises NotImplementedError: If function is not supported\n        '
        if self.has('available_dpi'):
            dpi_x = self._dbus_interfaces['dpi'].getDPI()[0]
            return (int(dpi_x), 0)
        elif self.has('dpi'):
            (dpi_x, dpi_y) = self._dbus_interfaces['dpi'].getDPI()
            return (int(dpi_x), int(dpi_y))
        else:
            raise NotImplementedError()

    @dpi.setter
    def dpi(self, value: tuple):
        if False:
            i = 10
            return i + 15
        "\n        Set mouse dpi\n\n        Daemon does type validation but can't be too careful\n        :param value: DPI X, Y tuple\n        :type value: tuple\n\n        :raises ValueError: If the tuple isn't long enough or contains invalid crap\n        :raises NotImplementedError: If function is not supported\n        "
        if self.has('dpi'):
            if len(value) != 2:
                raise ValueError('DPI tuple is not of length 2. Length: {0}'.format(len(value)))
            max_dpi = self.max_dpi
            (dpi_x, dpi_y) = value
            dpi_x_only = self.has('available_dpi')
            if not isinstance(dpi_x, int) or not isinstance(dpi_y, int):
                raise ValueError('DPI X or Y is not an integer, X:{0} Y:{1}'.format(type(dpi_x), type(dpi_y)))
            if dpi_x < 0 or dpi_x > max_dpi:
                raise ValueError('DPI X either too small or too large, X:{0}'.format(dpi_x))
            if dpi_x_only and (not dpi_y == 0):
                raise ValueError('DPI Y is not supported for this device')
            elif dpi_y < 0 or dpi_y > max_dpi:
                raise ValueError('DPI Y either too small or too large, Y:{0}'.format(dpi_y))
            self._dbus_interfaces['dpi'].setDPI(dpi_x, dpi_y)
        else:
            raise NotImplementedError()

    @property
    def dpi_stages(self) -> (int, list):
        if False:
            print('Hello World!')
        '\n        Get mouse DPI stages\n\n        Will return a tuple containing the active DPI stage number and the list\n        of DPI stages as tuples.\n        The active DPI stage number must be: >= 1 and <= nr of DPI stages.\n        :return: active DPI stage number and DPI stages\n                 (1, [(500, 500), (1000, 1000), (2000, 2000) ...]\n        :rtype: (int, list)\n\n        :raises NotImplementedError: if function is not supported\n        '
        if self.has('dpi_stages'):
            response = self._dbus_interfaces['dpi'].getDPIStages()
            dpi_stages = []
            active_stage = int(response[0])
            for (dpi_x, dpi_y) in response[1]:
                dpi_stages.append((int(dpi_x), int(dpi_y)))
            return (active_stage, dpi_stages)
        else:
            raise NotImplementedError()

    @dpi_stages.setter
    def dpi_stages(self, value: (int, list)):
        if False:
            for i in range(10):
                print('nop')
        "\n        Set mouse DPI stages\n\n        Daemon does type validation but can't be too careful\n        :param value: active DPI stage number and list of DPI X, Y tuples\n        :type value: (int, list)\n\n        :raises ValueError: when the input is invalid\n        :raises NotImplementedError: If function is not supported\n        "
        if self.has('dpi_stages'):
            max_dpi = self.max_dpi
            dpi_stages = []
            active_stage = value[0]
            if not isinstance(active_stage, int):
                raise ValueError('Active DPI stage is not an integer: {0}'.format(type(active_stage)))
            if active_stage < 1:
                raise ValueError('Active DPI stage has invalid value: {0} < 1'.format(active_stage))
            for stage in value[1]:
                if len(stage) != 2:
                    raise ValueError('DPI tuple is not of length 2. Length: {0}'.format(len(stage)))
                (dpi_x, dpi_y) = stage
                if not isinstance(dpi_x, int) or not isinstance(dpi_y, int):
                    raise ValueError('DPI X or Y is not an integer, X:{0} Y:{1}'.format(type(dpi_x), type(dpi_y)))
                if dpi_x < 0 or dpi_x > max_dpi:
                    raise ValueError('DPI X either too small or too large, X:{0}'.format(dpi_x))
                if dpi_y < 0 or dpi_y > max_dpi:
                    raise ValueError('DPI Y either too small or too large, Y:{0}'.format(dpi_y))
                dpi_stages.append((dpi_x, dpi_y))
            if active_stage > len(dpi_stages):
                raise ValueError('Active DPI stage has invalid value: {0} > {1}'.format(active_stage, len(dpi_stages)))
            self._dbus_interfaces['dpi'].setDPIStages(active_stage, dpi_stages)
        else:
            raise NotImplementedError()

    @property
    def scroll_mode(self) -> int:
        if False:
            print('Hello World!')
        "\n        Get the scroll wheel mode of the device\n\n        :return: The device's current scroll mode (0 = tactile, 1 = free spin)\n        :rtype: int\n\n        :raises NotImplementedError: If function is not supported\n        "
        if self.has('scroll_mode'):
            return int(self._dbus_interfaces['scroll'].getScrollMode())
        else:
            raise NotImplementedError()

    @scroll_mode.setter
    def scroll_mode(self, mode: int):
        if False:
            while True:
                i = 10
        '\n        Set the scroll mode of the device\n\n        :param mode: The mode to set (0 = tactile, 1 = free spin)\n        :type mode: int\n\n        :raises NotImplementedError: If function is not supported\n        '
        if self.has('scroll_mode'):
            self._dbus_interfaces['scroll'].setScrollMode(mode)
        else:
            raise NotImplementedError()

    @property
    def scroll_acceleration(self) -> bool:
        if False:
            print('Hello World!')
        "\n        Get the device's scroll acceleration state\n\n        :return: true if acceleration enabled, false otherwise\n        :rtype: bool\n\n        :raises NotImplementedError: If function is not supported\n        "
        if self.has('scroll_acceleration'):
            return bool(int(self._dbus_interfaces['scroll'].getScrollAcceleration()))
        else:
            raise NotImplementedError()

    @scroll_acceleration.setter
    def scroll_acceleration(self, enabled: bool):
        if False:
            while True:
                i = 10
        "\n        Set the device's scroll acceleration state\n\n        :param enabled: true to enable acceleration, false to disable it\n        :type enabled: bool\n\n        :raises NotImplementedError: If function is not supported\n        "
        if self.has('scroll_acceleration'):
            self._dbus_interfaces['scroll'].setScrollAcceleration(enabled)
        else:
            raise NotImplementedError()

    @property
    def scroll_smart_reel(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Get the device\'s "smart reel" state\n\n        :return: true if smart reel enabled, false otherwise\n        :rtype: bool\n\n        :raises NotImplementedError: If function is not supported\n        '
        if self.has('scroll_smart_reel'):
            return bool(int(self._dbus_interfaces['scroll'].getScrollSmartReel()))
        else:
            raise NotImplementedError()

    @scroll_smart_reel.setter
    def scroll_smart_reel(self, enabled: bool):
        if False:
            while True:
                i = 10
        '\n        Set the device\'s "smart reel" state\n\n        :param enabled: true to enable smart reel, false to disable it\n        :type enabled: bool\n\n        :raises NotImplementedError: If function is not supported\n        '
        if self.has('scroll_smart_reel'):
            self._dbus_interfaces['scroll'].setScrollSmartReel(enabled)
        else:
            raise NotImplementedError()