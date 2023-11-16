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
#include <string>

#include "aeotecsdg2.hpp"
#include "upm_utilities.h"

using namespace std;

bool shouldRun = true;

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

    string defaultDev = "/dev/ttyACM0";

    // if an argument was specified, use it as the device instead
    if (argc > 1)
        defaultDev = string(argv[1]);

    // Instantiate an Aeotec Smart Dimmer Gen2 instance, on device node
    // 9.  You will almost certainly need to change this to reflect your
    // own network.  Use the ozwdump example to see what nodes are
    // available.
    upm::AeotecSDG2 sensor(9);

    // The first thing to do is create options, then lock them when done.
    sensor.optionsCreate();
    sensor.optionsLock();

    // Next, initialize it.
    cout << "Initializing, this may take awhile depending on your ZWave network" << endl;

    sensor.init(defaultDev);
    cout << "Initialization complete" << endl;

    // turn light on
    cout << "Turning switch on, then upm_delaying for 5 secs" << endl;
    sensor.on();
    upm_delay(5);

    cout << "Querying data..." << endl;
    bool dim = false;
    while (shouldRun) {
        // put on a light show...
        if (dim)
            sensor.setLevel(25);
        else
            sensor.on();

        dim = !dim;

        sensor.update();

        cout << "Current Level: " << sensor.getLevel() << endl;

        cout << "Volts: " << sensor.getVolts() << " volts" << endl;

        cout << "Energy Consumption: " << sensor.getEnergy() << " kWh" << endl;

        cout << "Watts: " << sensor.getWatts() << endl;

        cout << "Current: " << sensor.getCurrent() << " amps" << endl;

        cout << endl;

        upm_delay(5);
    }

    cout << "Turning switch off and upm_delaying for 5 seconds..." << endl;
    sensor.off();
    upm_delay(5);

    cout << "Exiting..." << endl;

    //! [Interesting]

    return 0;
}
