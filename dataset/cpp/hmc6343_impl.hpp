/*
 * Copyright (c) 2013-2015, Niklas Hauser
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#ifndef MODM_HMC6343_HPP
#	error  "Don't include this file directly, use 'hmc6343.hpp' instead!"
#endif
#include <modm/math/utils/endianness.hpp>

// ----------------------------------------------------------------------------
template < class I2cMaster >
modm::Hmc6343<I2cMaster>::Hmc6343(Data &data, uint8_t address)
:	I2cDevice<I2cMaster,2>(address), data(data), timeout(500ms)
{
}

// ----------------------------------------------------------------------------
// MARK: - register access
// MARK: write command
template < class I2cMaster >
modm::ResumableResult<bool>
modm::Hmc6343<I2cMaster>::writeCommand(Command command, uint16_t timeout)
{
	RF_BEGIN();

	buffer[0] = i(command);
	RF_WAIT_UNTIL( this->timeout.isExpired() and this->startWrite(buffer, 1) );

	this->timeout.restart(std::chrono::milliseconds(timeout));
	RF_WAIT_WHILE( this->isTransactionRunning() );

	RF_END_RETURN( this->wasTransactionSuccessful() );
}

// MARK: write register
template < class I2cMaster >
modm::ResumableResult<bool>
modm::Hmc6343<I2cMaster>::writeRegister(Register reg, uint8_t value)
{
	RF_BEGIN();

	buffer[0] = i(Command::WriteEeprom);
	buffer[1] = i(reg);
	buffer[2] = value;

	RF_WAIT_UNTIL( timeout.isExpired() and this->startWrite(buffer, 3) );

	timeout.restart(10ms);
	RF_WAIT_WHILE( this->isTransactionRunning() );

	RF_END_RETURN( this->wasTransactionSuccessful() );
}

// MARK: write 16bit register
template < class I2cMaster >
modm::ResumableResult<bool>
modm::Hmc6343<I2cMaster>::writeRegister(Register16 reg, uint16_t value)
{
	RF_BEGIN();

	// for little endian machines this endianness "conversion" does nothing
	*reinterpret_cast<uint16_t*>(buffer+2) = modm::toLittleEndian(value);
	// for big endian machines, the bytes are swapped, so that the following is always true!
	// buffer[2] has LSB, buffer[3] has MSB

	// LSB
	if ( RF_CALL( writeRegister(static_cast<Register>(reg), buffer[2]) ) )
	{
		// MSB
		RF_RETURN_CALL( writeRegister(static_cast<Register>(i(reg)+1), buffer[3]) );
	}

	RF_END_RETURN(false);
}

// MARK: read register
template < class I2cMaster >
modm::ResumableResult<bool>
modm::Hmc6343<I2cMaster>::readRegister(Register reg, uint8_t &value)
{
	RF_BEGIN();

	buffer[0] = i(Command::ReadEeprom);
	buffer[1] = i(reg);
	RF_WAIT_UNTIL( timeout.isExpired() and this->startWrite(buffer, 2) );

	timeout.restart(10ms);
	RF_WAIT_WHILE( this->isTransactionRunning() );

	if( this->wasTransactionSuccessful() )
	{
		RF_WAIT_UNTIL( timeout.isExpired() and this->startRead(&value, 1) );

		RF_WAIT_WHILE( this->isTransactionRunning() );

		RF_RETURN( this->wasTransactionSuccessful() );
	}

	RF_END_RETURN(false);
}

// MARK: read 16bit register
template < class I2cMaster >
modm::ResumableResult<bool>
modm::Hmc6343<I2cMaster>::readRegister(Register16 reg, uint16_t &value)
{
	RF_BEGIN();

	// LSB
	if ( RF_CALL( readRegister(static_cast<Register>(reg), buffer[2]) ) )
	{
		// MSB
		if ( RF_CALL( readRegister(static_cast<Register>(i(reg)+1), buffer[3]) ) )
		{
			// buffer[2] has LSB, buffer[3] has MSB
			// bytes get swapped on big endian machines
			value = modm::fromLittleEndian(*reinterpret_cast<uint16_t*>(buffer+2));
			RF_RETURN(true);
		}
	}

	RF_END_RETURN(false);
}

// MARK: read 6 or 1 bytes of data
template < class I2cMaster >
modm::ResumableResult<bool>
modm::Hmc6343<I2cMaster>::readPostData(Command command, uint8_t offset, uint8_t readSize)
{
	RF_BEGIN();

	if (RF_CALL(writeCommand(command, 1)))
	{
		RF_WAIT_UNTIL( timeout.isExpired() and this->startRead(data.data + offset, readSize) );

		RF_WAIT_WHILE( this->isTransactionRunning() );

		RF_RETURN( this->wasTransactionSuccessful() );
	}

	RF_END_RETURN(false);
}
