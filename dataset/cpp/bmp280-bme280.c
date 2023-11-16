/*
 * Author: Jon Trulson <jtrulson@ics.com>
 * Copyright (c) 2017 Intel Corporation.
 *
 * The MIT License
 *
 * This program and the accompanying materials are made available under the
 * terms of the The MIT License which is available at
 * https://opensource.org/licenses/MIT.
 *
 * SPDX-License-Identifier: MIT
 */

#include <unistd.h>
#include <stdio.h>
#include <signal.h>

#include "upm_utilities.h"
#include "bmp280.h"

bool shouldRun = true;

void sig_handler(int signo)
{
    if (signo == SIGINT)
        shouldRun = false;
}

int main(int argc, char **argv)
{
    signal(SIGINT, sig_handler);
//! [Interesting]

    // Instantiate a BME280 instance using default i2c bus and
    // address.  We use the BMP280 driver to do all of our work, since
    // the BMP280 and the BME280 are identical except for the fact
    // that the BME280 includes a humidity sensor.
    bmp280_context sensor = bmp280_init(BME280_DEFAULT_I2C_BUS,
                                        BME280_DEFAULT_ADDR, -1);

    if (!sensor)
    {
        printf("bmp280_init() failed\n");
        return 1;
    }

    // For SPI, bus 0, you would pass -1 as the address, and a valid pin for CS:
    // bmp280_init(BME280_DEFAULT_SPI_BUS,
    //             -1, 10)

    while (shouldRun)
    {
        // update our values from the sensor
        if (bmp280_update(sensor))
        {
            printf("bmp280_update() failed\n");
            bmp280_close(sensor);
            return 1;
        }

        printf("Compensation Temperature: %f C\n",
               bmp280_get_temperature(sensor));
        printf("Pressure: %f Pa\n", bmp280_get_pressure(sensor));
        printf("Computed Altitude: %f m\n", bmp280_get_altitude(sensor));
        printf("Relative Humidity: %f %%\n\n", bmp280_get_humidity(sensor));

        upm_delay(1);
    }
//! [Interesting]

    printf("Exiting...\n");

    bmp280_close(sensor);

    return 0;
}
