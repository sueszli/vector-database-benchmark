/*
 * Copyright (c) 2014-2016, Niklas Hauser
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#ifndef MODM_MCP23_TRANSPORT_HPP
#	error  "Don't include this file directly, use 'mcp23_transport.hpp' instead!"
#endif

// ============================================================================
// MARK: I2C TRANSPORT
template < class I2cMaster >
modm::Mcp23TransportI2c<I2cMaster>::Mcp23TransportI2c(uint8_t address)
:	I2cDevice<I2cMaster, 2>(address)
{
}

// MARK: - register access
// MARK: write register
template < class I2cMaster >
modm::ResumableResult<bool>
modm::Mcp23TransportI2c<I2cMaster>::write(uint8_t reg, uint8_t value)
{
	RF_BEGIN();

	buffer[0] = reg;
	buffer[1] = value;

	this->transaction.configureWrite(buffer, 2);

	RF_END_RETURN_CALL( this->runTransaction() );
}

template < class I2cMaster >
modm::ResumableResult<bool>
modm::Mcp23TransportI2c<I2cMaster>::write16(uint8_t reg, uint16_t value)
{
	RF_BEGIN();

	buffer[0] = reg;
	buffer[1] = value;
	buffer[2] = (value >> 8);

	this->transaction.configureWrite(buffer, 3);

	RF_END_RETURN_CALL( this->runTransaction() );
}

// MARK: read register
template < class I2cMaster >
modm::ResumableResult<bool>
modm::Mcp23TransportI2c<I2cMaster>::read(uint8_t reg, uint8_t *buffer, uint8_t length)
{
	RF_BEGIN();

	this->buffer[0] = reg;
	this->transaction.configureWriteRead(this->buffer, 1, buffer, length);

	RF_END_RETURN_CALL( this->runTransaction() );
}

// ============================================================================
// MARK: SPI TRANSPORT
template < class SpiMaster, class Cs >
modm::Mcp23TransportSpi<SpiMaster, Cs>::Mcp23TransportSpi(uint8_t address)
:	address(address << 1)
{
	Cs::setOutput(modm::Gpio::High);
}

// MARK: ping
template < class SpiMaster, class Cs >
modm::ResumableResult<bool>
modm::Mcp23TransportSpi<SpiMaster, Cs>::ping()
{
	RF_BEGIN();
	RF_END_RETURN(true);
}

// MARK: - register access
// MARK: write register
template < class SpiMaster, class Cs >
modm::ResumableResult<bool>
modm::Mcp23TransportSpi<SpiMaster, Cs>::write(uint8_t reg, uint8_t value)
{
	RF_BEGIN();

	RF_WAIT_UNTIL(this->acquireMaster());
	Cs::reset();

	RF_CALL(SpiMaster::transfer(address | Write));
	RF_CALL(SpiMaster::transfer(reg));
	RF_CALL(SpiMaster::transfer(value));

	if (this->releaseMaster())
		Cs::set();

	RF_END_RETURN(true);
}

template < class SpiMaster, class Cs >
modm::ResumableResult<bool>
modm::Mcp23TransportSpi<SpiMaster, Cs>::write16(uint8_t reg, uint16_t value)
{
	RF_BEGIN();

	RF_WAIT_UNTIL(this->acquireMaster());
	Cs::reset();

	RF_CALL(SpiMaster::transfer(address | Write));
	RF_CALL(SpiMaster::transfer(reg));
	RF_CALL(SpiMaster::transfer(value));
	RF_CALL(SpiMaster::transfer(value >> 8));

	if (this->releaseMaster())
		Cs::set();

	RF_END_RETURN(true);
}

// MARK: read register
template < class SpiMaster, class Cs >
modm::ResumableResult<bool>
modm::Mcp23TransportSpi<SpiMaster, Cs>::read(uint8_t reg, uint8_t *buffer, uint8_t length)
{
	RF_BEGIN();

	RF_WAIT_UNTIL(this->acquireMaster());
	Cs::reset();

	RF_CALL(SpiMaster::transfer(address | Read));
	RF_CALL(SpiMaster::transfer(reg));

	RF_CALL(SpiMaster::transfer(nullptr, buffer, length));

	if (this->releaseMaster())
		Cs::set();

	RF_END_RETURN(true);
}

