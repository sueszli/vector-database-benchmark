#!/usr/bin/env python
# -*- coding: utf-8 -*-

HOST = "localhost"
PORT = 4223
UID = "XYZ" # Change XYZ to the UID of your Temperature Bricklet 2.0

import time
from tinkerforge.ip_connection import IPConnection
from tinkerforge.bricklet_temperature_v2 import BrickletTemperatureV2

if __name__ == "__main__":
    ipcon = IPConnection() # Create IP connection
    t = BrickletTemperatureV2(UID, ipcon) # Create device object

    ipcon.connect(HOST, PORT) # Connect to brickd
    # Don't use device before ipcon is connected

    # Get current temperature (unit is °C/100)
    temperature = t.get_temperature() / 100.0
    print('Temperature of ' + str(temperature) + ' °C on ' + time.ctime())

    ipcon.disconnect()
