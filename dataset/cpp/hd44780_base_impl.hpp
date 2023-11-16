/*
 * Copyright (c) 2013, 2017, Niklas Hauser
 * Copyright (c) 2014-2015, Sascha Schade
 * Copyright (c) 2018, Antal Szabó
 * Copyright (c) 2018, Álan Crístoffer
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#ifndef MODM_HD44780_BASE_HPP
	#error	"Don't include this file directly, use 'hd44780_base.hpp' instead!"
#endif

#include <modm/architecture/interface/delay.hpp>

// ----------------------------------------------------------------------------
template <typename DATA, typename RW, typename RS, typename E>
void
modm::Hd44780Base<DATA, RW, RS, E>::initialize(LineMode lineMode)
{
	E::setOutput(E_Disable);
	RW::setOutput(RW_Write);
	RS::setOutput(RS_Command);

	Bus<DATA, E, DATA::width>::writeHighNibble(Set8BitBus);

	modm::delay_ms(5);

	Bus<DATA, E, DATA::width>::writeHighNibble(Set8BitBus);

	modm::delay_us(100);

	Bus<DATA, E, DATA::width>::writeHighNibble(Set8BitBus);

	if constexpr (DATA::width == 4)
	{
		while(isBusy())
			;
		RW::set(RW_Write);
		Bus<DATA, E, DATA::width>::writeHighNibble(Set4BitBus);
	}

	while(!writeCommand(InternalCommand(lineMode) |
			Bus<DATA, E, DATA::width>::Mode))
		;
	while(!writeCommand(DisableDisplay))
		;
	while(!writeCommand(ClearDisplay))
		;
	while(!writeCommand(IncrementCursor))
		;
	while(!writeCommand(EnableDisplay | DisableCursor | DisableCursorBlink))
		;
}

template <typename DATA, typename RW, typename RS, typename E>
bool
modm::Hd44780Base<DATA, RW, RS, E>::clear()
{
	return writeCommand(ClearDisplay);
}

template <typename DATA, typename RW, typename RS, typename E>
bool
modm::Hd44780Base<DATA, RW, RS, E>::resetCursor()
{
	return writeCommand(ResetCursor);
}

// MARK: write
template <typename DATA, typename RW, typename RS, typename E>
void
modm::Hd44780Base<DATA, RW, RS, E>::write(uint8_t data)
{
	RW::set(RW_Write);
	Bus<DATA, E, DATA::width>::write(data);
}

template <typename DATA, typename RW, typename RS, typename E>
bool
modm::Hd44780Base<DATA, RW, RS, E>::writeAddress(uint8_t address)
{
	if (isBusy())
		return false;

	RS::set(RS_Command);
	RW::set(RW_Write);

	write((SetDDRAM_Address | (address & DDRAM_AddressMask)));

	return true;
}

template <typename DATA, typename RW, typename RS, typename E>
bool
modm::Hd44780Base<DATA, RW, RS, E>::writeCommand(uint8_t command)
{
	if (isBusy())
		return false;

	RS::set(RS_Command);
	write(command);

	return true;
}

template <typename DATA, typename RW, typename RS, typename E>
bool
modm::Hd44780Base<DATA, RW, RS, E>::writeRAM(uint8_t data)
{
	if (isBusy())
		return false;

	RS::set(RS_RAM);
	write(data);

	return true;
}

// MARK: read

template <typename DATA, typename RW, typename RS, typename E>
uint8_t
modm::Hd44780Base<DATA, RW, RS, E>::read()
{
	RW::set(RW_Read);
	return Bus<DATA, E, DATA::width>::read();
}

template <typename DATA, typename RW, typename RS, typename E>
bool
modm::Hd44780Base<DATA, RW, RS, E>::readAddress(uint8_t &address)
{
	RS::set(RS_Command);
	address = (read() & CGRAM_AddressMask);

	return true;
}

template <typename DATA, typename RW, typename RS, typename E>
bool
modm::Hd44780Base<DATA, RW, RS, E>::readRAM(uint8_t &data)
{
	if (isBusy())
		return false;

	RS::set(RS_RAM);
	data = read();

	return true;
}

// MARK: status
template <typename DATA, typename RW, typename RS, typename E>
bool
modm::Hd44780Base<DATA, RW, RS, E>::isBusy()
{
	RS::set(RS_Command);

	if (read() & BusyFlagMask)
	{
		modm::delay_us(2);
		return true;
	}
	return false;
}

template <typename DATA, typename RW, typename RS, typename E>
bool
modm::Hd44780Base<DATA, RW, RS, E>::writeCGRAM(uint8_t character, uint8_t *cg)
{
	while(not writeCommand(SetCGRAM_Address | (character << 3)))
		;
	for (std::size_t ii = 0; ii < 8; ++ii) {
		modm::delay(50us);
		writeRAM(cg[ii]);
	}
	return true;
}


// MARK: bus specialisation of 4bit port
template <typename DATA, typename RW, typename RS, typename E>
template <typename Data, typename Enable>
void
modm::Hd44780Base<DATA, RW, RS, E>::Bus<Data, Enable, 4>::write(uint8_t data)
{
	DATA::setOutput();
	DATA::write(data >> 4);

	E::set();
	modm::delay_us(1);
	E::reset();
	modm::delay_ns(10);

	DATA::write(data);

	E::set();
	modm::delay_us(1);
	E::reset();
	modm::delay_ns(10);
}

template <typename DATA, typename RW, typename RS, typename E>
template <typename Data, typename Enable>
void
modm::Hd44780Base<DATA, RW, RS, E>::Bus<Data, Enable, 4>::writeHighNibble(uint8_t data)
{
	Bus<DATA, E, 8>::write(data >> 4);
}

template <typename DATA, typename RW, typename RS, typename E>
template <typename Data, typename Enable>
uint8_t
modm::Hd44780Base<DATA, RW, RS, E>::Bus<Data, Enable, 4>::read()
{
	uint8_t data;
	DATA::setInput();

	E::set();
	modm::delay_us(1);
	data = DATA::read();
	E::reset();
	modm::delay_ns(10);

	data <<= 4;

	E::set();
	modm::delay_us(1);
	data |= DATA::read();
	E::reset();
	modm::delay_ns(10);

	return data;
}

// MARK: bus specialisation of 8bit port
template <typename DATA, typename RW, typename RS, typename E>
template <typename Data, typename Enable>
void
modm::Hd44780Base<DATA, RW, RS, E>::Bus<Data, Enable, 8>::write(uint8_t data)
{
	DATA::setOutput();
	DATA::write(data);

	E::set();
	modm::delay_us(1);
	E::reset();
	modm::delay_ns(500);
}

template <typename DATA, typename RW, typename RS, typename E>
template <typename Data, typename Enable>
void
modm::Hd44780Base<DATA, RW, RS, E>::Bus<Data, Enable, 8>::writeHighNibble(uint8_t data)
{
	write(data);
}

template <typename DATA, typename RW, typename RS, typename E>
template <typename Data, typename Enable>
uint8_t
modm::Hd44780Base<DATA, RW, RS, E>::Bus<Data, Enable, 8>::read()
{
	uint8_t data;
	DATA::setInput();

	E::set();
	modm::delay_us(1);
	data = DATA::read();
	E::reset();
	modm::delay_ns(500);

	return data;
}



