/*
 * Author: Jon Trulson <jtrulson@ics.com>
 * Copyright (c) 2016 Intel Corporation.
 *
 * This program and the accompanying materials are made available under the
 * terms of the The MIT License which is available at
 * https://opensource.org/licenses/MIT.
 *
 * SPDX-License-Identifier: MIT
 */

#include <iostream>
#include <signal.h>

#include "lsm6ds3h.hpp"
#include "upm_utilities.h"

using namespace std;

int shouldRun = true;

void
sig_handler(int signo)
{
    if (signo == SIGINT)
        shouldRun = false;
}

int
main(int argc, char** argv)
{
    signal(SIGINT, sig_handler);
    //! [Interesting]

    // Instantiate an LSM6DS3H using default I2C parameters
    upm::LSM6DS3H sensor;

    // For SPI, bus 0, you would pass -1 as the address, and a valid pin
    // for CS: LSM6DS3H(0, -1, 10);

    // now output data every 250 milliseconds
    while (shouldRun) {
        float x, y, z;

        sensor.update();

        sensor.getAccelerometer(&x, &y, &z);
        cout << "Accelerometer x: " << x << " y: " << y << " z: " << z << " g" << endl;

        sensor.getGyroscope(&x, &y, &z);
        cout << "Gyroscope x: " << x << " y: " << y << " z: " << z << " dps" << endl;

        // we show both C and F for temperature
        cout << "Compensation Temperature: " << sensor.getTemperature() << " C / "
             << sensor.getTemperature(true) << " F" << endl;

        cout << endl;

        upm_delay_us(250000);
    }

    //! [Interesting]

    cout << "Exiting..." << endl;

    return 0;
}
