/*
 * Copyright (c) 2009-2010, Martin Rosekeit
 * Copyright (c) 2009-2011, Fabian Greif
 * Copyright (c) 2011-2015, Niklas Hauser
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#ifndef MODM_TMP175_HPP
#	error  "Don't include this file directly, use 'tmp175.hpp' instead!"
#endif

// ----------------------------------------------------------------------------
template < typename I2cMaster >
modm::Tmp175<I2cMaster>::Tmp175(Data &data, uint8_t address)
:	Lm75<I2cMaster>(reinterpret_cast<lm75::Data&>(data), address),
	updateTime(250), conversionTime(232),
	periodTimeout(updateTime), conversionTimeout(conversionTime)
{
	this->stop();
}

template < typename I2cMaster >
bool
modm::Tmp175<I2cMaster>::run()
{
	PT_BEGIN();

	while(true)
	{
		PT_WAIT_UNTIL(periodTimeout.isExpired());
		periodTimeout.restart(updateTime);

		PT_CALL(startConversion());

		conversionTimeout.restart(conversionTime);
		PT_WAIT_UNTIL(conversionTimeout.isExpired());

		PT_CALL(this->readTemperature());
	}

	PT_END();
}

template < typename I2cMaster >
modm::tmp175::Data&
modm::Tmp175<I2cMaster>::getData()
{
	return reinterpret_cast<Data&>(this->data);
}

// MARK: - tasks
template < typename I2cMaster >
void
modm::Tmp175<I2cMaster>::setUpdateRate(uint8_t rate)
{
	// clamp conversion rate to max 33Hz (=~30ms)
	if (rate == 0) rate = 1;
	if (rate > 33) rate = 33;

	updateTime = modm::ShortDuration(1000/rate - 29);
	periodTimeout.restart(updateTime);

	this->restart();
}

template < typename I2cMaster >
modm::ResumableResult<bool>
modm::Tmp175<I2cMaster>::setResolution(Resolution resolution)
{
	RF_BEGIN();

	Resolution_t::set(reinterpret_cast<Config1_t&>(this->config_msb), resolution);

	conversionTime = modm::ShortDuration((uint8_t(resolution) + 1) * 29);

	RF_END_RETURN_CALL( writeConfiguration() );
}

// MARK: conversion
template < typename I2cMaster >
modm::ResumableResult<bool>
modm::Tmp175<I2cMaster>::startConversion()
{
	RF_BEGIN();

	reinterpret_cast<Config1_t&>(this->config_msb).set(Config1::OneShot);

	if ( RF_CALL(writeConfiguration()) )
	{
		reinterpret_cast<Config1_t&>(this->config_msb).reset(Config1::OneShot);
		RF_RETURN(true);
	}

	RF_END_RETURN(false);
}

// MARK: configuration
template < typename I2cMaster >
modm::ResumableResult<bool>
modm::Tmp175<I2cMaster>::writeConfiguration()
{
	RF_BEGIN();

	this->buffer[0] = uint8_t(Register::Configuration);
	this->buffer[1] = reinterpret_cast<Config1_t&>(this->config_msb).value;

	this->transaction.configureWrite(this->buffer, 2);

	RF_END_RETURN_CALL( this->runTransaction() );
}

template < typename I2cMaster >
modm::ResumableResult<bool>
modm::Tmp175<I2cMaster>::setLimitRegister(Register reg, float temperature)
{
	RF_BEGIN();

	{
		uint8_t res = uint8_t(Resolution_t::get(reinterpret_cast<Config1_t&>(this->config_msb)));

		int16_t temp = temperature * (2 << res);
		temp <<= 4 + res;

		this->buffer[0] = uint8_t(reg);
		this->buffer[1] = (temp >> 8);
		this->buffer[2] = temp;
	}

	this->transaction.configureWrite(this->buffer, 3);

	RF_END_RETURN_CALL( this->runTransaction() );
}

