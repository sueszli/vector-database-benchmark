/*
 * Copyright (c) 2016, Niklas Hauser
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#ifndef MODM_FT6X06_HPP
#	error "Don't include this file directly, use 'ft6x06.hpp' instead!"
#endif

// ----------------------------------------------------------------------------
template < typename I2cMaster >
modm::Ft6x06<I2cMaster>::Ft6x06(Data &data, uint8_t address)
:	I2cDevice<I2cMaster,3>(address), data(data)
{
}

// MARK: - Tasks
template < typename I2cMaster >
modm::ResumableResult<bool>
modm::Ft6x06<I2cMaster>::configure(InterruptMode mode, uint8_t activeRate, uint8_t monitorRate)
{
	RF_BEGIN();

	if (RF_CALL(write(Register::G_MODE, uint8_t(mode))))
	{
		if (RF_CALL(write(Register::PERIOD_ACTIVE, activeRate)))
		{
			RF_RETURN_CALL(write(Register::PERIOD_MONITOR, monitorRate));
		}
	}

	RF_END_RETURN(false);
}

template < typename I2cMaster >
modm::ResumableResult<bool>
modm::Ft6x06<I2cMaster>::readTouches()
{
	RF_BEGIN();

	if (RF_CALL(read(Register::GEST_ID, buffer, 14)))
	{
		std::memcpy(data.data, buffer, 14);
		RF_RETURN(true);
	}

	RF_END_RETURN(false);
}

// ----------------------------------------------------------------------------
// MARK: write register
template < class I2cMaster >
modm::ResumableResult<bool>
modm::Ft6x06<I2cMaster>::write(Register reg, uint8_t value)
{
	RF_BEGIN();

	buffer[0] = uint8_t(reg);
	buffer[1] = value;
	this->transaction.configureWrite(buffer, 2);

	RF_END_RETURN_CALL( this->runTransaction() );
}

// MARK: read multilength register
template < class I2cMaster >
modm::ResumableResult<bool>
modm::Ft6x06<I2cMaster>::read(Register reg, uint8_t *buffer, uint8_t length)
{
	RF_BEGIN();

	this->buffer[0] = uint8_t(reg);
	this->transaction.configureWriteRead(this->buffer, 1, buffer, length);

	RF_END_RETURN_CALL( this->runTransaction() );
}
