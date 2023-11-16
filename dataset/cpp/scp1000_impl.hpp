/*
 * Copyright (c) 2011, Fabian Greif
 * Copyright (c) 2011-2012, 2017, Niklas Hauser
 * Copyright (c) 2014, Sascha Schade
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#ifndef MODM_SCP1000_HPP
	#error	"Don't include this file directly, use 'scp1000.hpp' instead!"
#endif

template < typename Spi, typename Cs, typename Int >
Spi modm::Scp1000<Spi, Cs, Int>::spi;

template < typename Spi, typename Cs, typename Int >
Cs modm::Scp1000<Spi, Cs, Int>::chipSelect;

template < typename Spi, typename Cs, typename Int >
Int modm::Scp1000<Spi, Cs, Int>::interruptPin;

template < typename Spi, typename Cs, typename Int >
bool modm::Scp1000<Spi, Cs, Int>::newTemperature(false);

template < typename Spi, typename Cs, typename Int >
bool modm::Scp1000<Spi, Cs, Int>::newPressure(false);

template < typename Spi, typename Cs, typename Int >
uint8_t modm::Scp1000<Spi, Cs, Int>::temperature[2];

template < typename Spi, typename Cs, typename Int >
uint8_t modm::Scp1000<Spi, Cs, Int>::pressure[3];

// ----------------------------------------------------------------------------
template < typename Spi, typename Cs, typename Int >
bool
modm::Scp1000<Spi, Cs, Int>::initialize(scp1000::Operation opMode)
{
	chipSelect.setOutput();
	chipSelect.set();
#if defined MODM_CPU_ATXMEGA
	interruptPin.setInput(::modm::platform::PULLDOWN);
#else
	interruptPin.setInput();
#endif
	bool result;
	// Reset the chip
	result = reset();
	if (result) {
		result &= setOperation(opMode);
	}
	return result;
}

template < typename Spi, typename Cs, typename Int >
void
modm::Scp1000<Spi, Cs, Int>::readTemperature()
{
	read16BitRegister(scp1000::REGISTER_TEMPOUT, temperature);
	newTemperature = true;
}

template < typename Spi, typename Cs, typename Int >
void
modm::Scp1000<Spi, Cs, Int>::readPressure()
{
	pressure[0] = read8BitRegister(scp1000::REGISTER_DATARD8);
	read16BitRegister(scp1000::REGISTER_DATARD16, &pressure[1]);

	newPressure = true;
}

template < typename Spi, typename Cs, typename Int >
uint8_t*
modm::Scp1000<Spi, Cs, Int>::getTemperature()
{
	newTemperature = false;
	return temperature;
}

template < typename Spi, typename Cs, typename Int >
uint8_t*
modm::Scp1000<Spi, Cs, Int>::getPressure()
{
	newPressure = false;
	return pressure;
}


template < typename Spi, typename Cs, typename Int >
bool
modm::Scp1000<Spi, Cs, Int>::setOperation(scp1000::Operation opMode)
{
	writeRegister(scp1000::REGISTER_OPERATION, opMode);

	uint8_t retries = 16;
	// wait for the sensor to complete setting the operation
	while (--retries && (readStatus(true) & scp1000::OPERATION_STATUS_RUNNING)) {
		modm::delay_ms(1);
	}

	// The sensor took too long to complete the operation
	if (retries)
		return true;

	return false;
}

template < typename Spi, typename Cs, typename Int >
uint8_t
modm::Scp1000<Spi, Cs, Int>::readStatus(bool opStatus)
{
	Register address = scp1000::REGISTER_STATUS;
	if (opStatus) {
		address = scp1000::REGISTER_OPSTATUS;
	}
	return read8BitRegister(address);
}

template < typename Spi, typename Cs, typename Int >
bool
modm::Scp1000<Spi, Cs, Int>::reset(uint8_t timeout=50)
{
	writeRegister(scp1000::REGISTER_RSTR, scp1000::RESET);

	// wait a bit to give the Scp1000 some time to restart
	modm::delay_ms(151);

	uint8_t retries = timeout;
	// wait for the sensor to complete start up, this should take 160ms
	while (--retries && (readStatus() & scp1000::STATUS_STARTUP_RUNNING_bm)) {
		modm::delay_ms(1);
	}

	if (retries > 0) {
		return true;
	}

	// The sensor took too long to start up
	return false;
}

template < typename Spi, typename Cs, typename Int >
bool
modm::Scp1000<Spi, Cs, Int>::isNewTemperatureAvailable()
{
	return newTemperature;
}

template < typename Spi, typename Cs, typename Int >
bool
modm::Scp1000<Spi, Cs, Int>::isNewPressureAvailable()
{
	return newPressure;
}

template < typename Spi, typename Cs, typename Int >
bool
modm::Scp1000<Spi, Cs, Int>::isNewDataReady()
{
	return interruptPin.read();
}

template < typename Spi, typename Cs, typename Int >
void
modm::Scp1000<Spi, Cs, Int>::writeRegister(scp1000::Register reg, uint8_t data)
{
	chipSelect.reset();
	spi.write(reg<<2|0x02);
	spi.write(data);
	chipSelect.set();
}

template < typename Spi, typename Cs, typename Int >
uint8_t
modm::Scp1000<Spi, Cs, Int>::read8BitRegister(scp1000::Register reg)
{
	uint8_t result;
	chipSelect.reset();
	spi.write(reg<<2);
	result = spi.write(0x00);
	chipSelect.set();
	return result;
}

template < typename Spi, typename Cs, typename Int >
void
modm::Scp1000<Spi, Cs, Int>::read16BitRegister(scp1000::Register reg, uint8_t *buffer)
{
	chipSelect.reset();
	spi.write(reg<<2);
	buffer[0] = spi.write(0x00);
	buffer[1] = spi.write(0x00);
	chipSelect.set();
}
