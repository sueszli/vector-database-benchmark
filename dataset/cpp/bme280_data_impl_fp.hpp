/*
 * Copyright (c) 2016-2017, Sascha Schade
 * Copyright (c) 2017, Fabian Greif
 * Copyright (c) 2017, Niklas Hauser
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#include <stdio.h>
#include <cmath>
#include <inttypes.h>

#include <modm/debug/logger/logger.hpp>

#ifndef MODM_BME280_DATA_HPP
#	error  "Don't include this file directly, use 'bme280_data.hpp' instead!"
#endif

#undef  MODM_LOG_LEVEL
#define MODM_LOG_LEVEL modm::log::DISABLED

namespace modm
{
namespace bme280data
{

// Fixed point implementation

Data::Data() :
	calibratedPressure(0),
	t_fine(0)
{
}

void
Data::calculateCalibratedTemperature()
{
	int32_t adc = (((int32_t(raw[3])) << 16) | (raw[4] << 8) | (raw[5] << 0));
	adc >>= 4;

	MODM_LOG_DEBUG.printf("adc = 0x%05" PRIx32 "\n", adc);

	int32_t T1 = calibration.T1;
	int32_t T2 = calibration.T2;
	int32_t T3 = calibration.T3;

	int32_t var1 = ((((adc >> 3) - (T1 << 1))) * (T2)) >> 11;


	int32_t var2 = (  (  ( ((adc >> 4) - (T1)) *
		                   ((adc >> 4) - (T1))
		                 ) >> 12
				 	  ) * (T3)
                   ) >> 14;

	t_fine = var1 + var2; // global

	calibratedTemperature = (t_fine * 5 + 128) >> 8;

	meta |= TEMPERATURE_CALCULATED;
}

void
Data::calculateCalibratedPressure()
{
	// Before pressure can be calculated, temperature must be.
	if (not (meta & TEMPERATURE_CALCULATED)) {
		calculateCalibratedTemperature();
	}

	int32_t adc = (((int32_t(raw[0])) << 16) | (raw[1] << 8) | (raw[2] << 0));
	adc >>= 4;

	int64_t P1 = calibration.P1;
	int64_t P2 = calibration.P2;
	int64_t P3 = calibration.P3;
	int64_t P4 = calibration.P4;
	int64_t P5 = calibration.P5;
	int64_t P6 = calibration.P6;
	int64_t P7 = calibration.P7;
	int64_t P8 = calibration.P8;
	int64_t P9 = calibration.P9;

	int64_t var1 = t_fine - 128'000;
	int64_t var2 = var1 * var1 * P6;
	var2 = var2 + ((var1 * P5) << 17);
	var2 = var2 + (P4 << 35);
	var1 = ((var1 * var1 * P3) >> 8) + ((var1 * P2) << 12);
	var1 = ((((int64_t(1)) << 47) + var1)) * (P1) >> 33;

	if (var1 == 0) {
    	return;  // avoid exception caused by division by zero
  	}
  	int64_t p = 1048576 - adc;
  	p = (((p << 31) - var2) * 3125) / var1;
  	var1 = ((P9) * (p >> 13) * (p >> 13)) >> 25;
  	var2 = (P8 * p) >> 19;

	p = ((p + var1 + var2) >> 8) + (P7 << 4);

	calibratedPressure = ((float)p/256)*1000;

	meta |= PRESSURE_CALCULATED;
}

void
Data::calculateCalibratedHumidity()
{
	// Before humidity can be calculated, temperature must be.
	if (not (meta & TEMPERATURE_CALCULATED)) {
		calculateCalibratedTemperature();
	}

	int32_t adc = ((raw[6] << 8) | (raw[7] << 0));

	int32_t H1 = calibration.H1;
	int32_t H2 = calibration.H2;
	int32_t H3 = calibration.H3;
	int32_t H4 = calibration.H4;
	int32_t H5 = calibration.H5;
	int32_t H6 = calibration.H6;

  	int32_t v = (t_fine - int32_t(76'800));

    v = (((((adc << 14) - (H4 << 20) - (H5 * v)) + (int32_t(16'384))) >> 15) *
		  (((((((v * H6) >> 10) * (((v * (H3)) >> 11) + (int32_t(32'768)))) >> 10) +
		  	(int32_t(2'097'152))) * H2 + 8'192) >> 14));

	v = (v - (((((v >> 15) * (v >> 15)) >> 7) * H1) >> 4));

	v = (v < 0) ? 0 : v;
	v = (v > 419'430'400) ? 419'430'400 : v;

	calibratedHumidity = (v >> 12);
	meta |= HUMIDITY_CALCULATED;
}

int32_t
Data::getTemperature()
{
	if (not (meta & TEMPERATURE_CALCULATED)) {
		calculateCalibratedTemperature();
	}
	return calibratedTemperature;
}

int32_t
Data::getPressure()
{
	if (not (meta & PRESSURE_CALCULATED)) {
		calculateCalibratedPressure();
	}
	return calibratedPressure;
}

int32_t
Data::getHumidity()
{
	if (not (meta & HUMIDITY_CALCULATED)) {
		calculateCalibratedHumidity();
	}
	return calibratedHumidity;
}

} // bme280data namespace

} // modm namespace
