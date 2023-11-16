/*
 * Copyright (c) 2015, 2018, Niklas Hauser
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#ifndef MODM_MCP23X17_HPP
#	error	"Don't include this file directly, use 'mcp23x17.hpp' instead!"
#endif

// ----------------------------------------------------------------------------
template < class Transport >
modm::Mcp23x17<Transport>::Mcp23x17(uint8_t address) :
	Transport(address), memory()
{
}

template < class Transport >
modm::ResumableResult<bool>
modm::Mcp23x17<Transport>::initialize()
{
	RF_BEGIN();

	memory = Memory(); // reset local register cache
	RF_CALL(this->write(i(Register::IOCON), memory.controlA.value));

	RF_CALL(this->write16(i(Register::IODIR), memory.direction.value));
	RF_CALL(this->write16(i(Register::IPOL), memory.polarity.value));
	RF_CALL(this->write16(i(Register::GPINTEN), memory.interruptEnable.value));
	RF_CALL(this->write16(i(Register::DEFVAL), memory.interruptDefault.value));
	RF_CALL(this->write16(i(Register::INTCON), memory.interruptControl.value));
	RF_CALL(this->write16(i(Register::GPPU), memory.pullup.value));
	RF_CALL(this->write16(i(Register::GPIO), memory.gpio.value));
	RF_CALL(this->write16(i(Register::OLAT), memory.outputLatch.value));
	RF_CALL(Transport::read(i(Register::INTF), buffer + 14, 4));

	RF_END_RETURN_CALL( Transport::read(i(Register::IODIR), buffer, sizeof(buffer)) );
}

// MARK: - Tasks
template < class Transport >
modm::ResumableResult<bool>
modm::Mcp23x17<Transport>::setOutput(Pins pins)
{
	RF_BEGIN();

	// output is 0, input is 1
	memory.direction.reset(pins);

	RF_END_RETURN_CALL( this->write16(i(Register::IODIR), memory.direction.value) );
}

template < class Transport >
modm::ResumableResult<bool>
modm::Mcp23x17<Transport>::set(Pins pins)
{
	RF_BEGIN();

	// high is 1, low is 0
	// set output latches locally, but only those that are output
	memory.outputLatch.set(pins & ~memory.direction);

	RF_END_RETURN_CALL( this->write16(i(Register::GPIO), memory.outputLatch.value) );
}

template < class Transport >
modm::ResumableResult<bool>
modm::Mcp23x17<Transport>::reset(Pins pins)
{
	RF_BEGIN();

	// high is 1, low is 0
	// reset reset output latches locally, but only those that are output
	memory.outputLatch.reset(pins & ~memory.direction);

	RF_END_RETURN_CALL( this->write16(i(Register::GPIO), memory.outputLatch.value) );
}

template < class Transport >
modm::ResumableResult<bool>
modm::Mcp23x17<Transport>::toggle(Pins pins)
{
	RF_BEGIN();

	// high is 1, low is 0
	// toggle output latches locally, but only those that are output
	memory.outputLatch.toggle(pins & ~memory.direction);

	RF_END_RETURN_CALL( this->write16(i(Register::GPIO), memory.outputLatch.value) );
}

template < class Transport >
modm::ResumableResult<bool>
modm::Mcp23x17<Transport>::set(Pins pins, bool value)
{
	RF_BEGIN();

	// high is 1, low is 0
	// update output latches locally, but only those that are output
	memory.outputLatch.update(pins & ~memory.direction, value);

	RF_END_RETURN_CALL( this->write16(i(Register::GPIO), memory.outputLatch.value) );
}

template < class Transport >
modm::ResumableResult<bool>
modm::Mcp23x17<Transport>::setInput(Pins pins)
{
	RF_BEGIN();

	// output is 0, input is 1
	memory.direction.set(pins);
	memory.outputLatch.reset(pins);

	RF_END_RETURN_CALL( this->write16(i(Register::IODIR), memory.direction.value) );
}

template < class Transport >
modm::ResumableResult<bool>
modm::Mcp23x17<Transport>::setPullUp(Pins pins)
{
	RF_BEGIN();

	// inverted is 1, normal is 0
	memory.pullup.set(pins);

	RF_END_RETURN_CALL( this->write16(i(Register::GPPU), memory.pullup.value) );
}

template < class Transport >
modm::ResumableResult<bool>
modm::Mcp23x17<Transport>::resetPullUp(Pins pins)
{
	RF_BEGIN();

	// inverted is 1, normal is 0
	memory.pullup.reset(pins);

	RF_END_RETURN_CALL( this->write16(i(Register::GPPU), memory.pullup.value) );
}

template < class Transport >
modm::ResumableResult<bool>
modm::Mcp23x17<Transport>::setInvertInput(Pins pins)
{
	RF_BEGIN();

	// inverted is 1, normal is 0
	memory.polarity.set(pins);

	RF_END_RETURN_CALL( this->write16(i(Register::IPOL), memory.polarity.value) );
}

template < class Transport >
modm::ResumableResult<bool>
modm::Mcp23x17<Transport>::resetInvertInput(Pins pins)
{
	RF_BEGIN();

	// inverted is 1, normal is 0
	memory.polarity.reset(pins);

	RF_END_RETURN_CALL( this->write16(i(Register::IPOL), memory.polarity.value) );
}

template < class Transport >
modm::ResumableResult<bool>
modm::Mcp23x17<Transport>::writePort(uint16_t data)
{
	RF_BEGIN();

	// high is 1, low is 0
	// output is 0, input is 1
	// set output latches locally, but only those that are output
	// clear all outputs
	memory.outputLatch.clear(~memory.direction);
	// set masked output values
	memory.outputLatch.set(Pins(data) & ~memory.direction);

	RF_END_RETURN_CALL( this->write16(i(Register::GPIO), memory.outputLatch.value) );
}

template < class Transport >
modm::ResumableResult<bool>
modm::Mcp23x17<Transport>::readPort(uint16_t &data)
{
	RF_BEGIN();

	if (RF_CALL( readInput() ))
	{
		data = memory.gpio.value;
		RF_RETURN( true );
	}

	RF_END_RETURN( false );
}
