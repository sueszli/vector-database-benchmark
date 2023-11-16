/*
 * Copyright (c) 2011, Georgi Grinshpun
 * Copyright (c) 2012-2013, Niklas Hauser
 * Copyright (c) 2012, 2014, Sascha Schade
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#ifndef MODM_NOKIA6610_HPP
#error	"Don't include this file directly, use 'nokia6610.hpp' instead!"
#endif

#include "nokia6610_defines.hpp"

// ----------------------------------------------------------------------------
template <typename SPI, typename CS, typename Reset, bool GE12>
void
modm::Nokia6610<SPI, CS, Reset, GE12>::initialize()
{

	// CS pin
	CS::set();
	CS::setOutput();


	// Reset pin
	Reset::setOutput();
	Reset::reset();
	modm::delay_ms(1);
	Reset::set();
	modm::delay_ms(10);


	lcdSettings();

	this->clear();
//	this->update();
}

template <typename SPI, typename CS, typename Reset, bool GE12>
void
modm::Nokia6610<SPI, CS, Reset, GE12>::writeSpi9Bit(uint16_t data){
	// add new bits to temp
	temp = temp | ((data&0x1ff)<< (7-countValidBits));
	countValidBits += 9;

	while(countValidBits >= 8){
		SPI::write(temp>>8);
		countValidBits -= 8;
		temp <<= 8;
	}
}

template <typename SPI, typename CS, typename Reset, bool GE12>
void
modm::Nokia6610<SPI, CS, Reset, GE12>::writeSpi9BitFlush(){
	if (countValidBits > 0){
		SPI::write(temp>>8);
	}
	countValidBits = 0;
	temp = 0;
}


template <typename SPI, typename CS, typename Reset, bool GE12>
void
modm::Nokia6610<SPI, CS, Reset, GE12>::writeSpiCommand(uint16_t data){
	writeSpi9Bit(data & ~0x0100);
}

template <typename SPI, typename CS, typename Reset, bool GE12>
void
modm::Nokia6610<SPI, CS, Reset, GE12>::writeSpiData(uint16_t data){
	writeSpi9Bit(data | 0x0100);
}

template <typename SPI, typename CS, typename Reset, bool GE12>
void
modm::Nokia6610<SPI, CS, Reset, GE12>::setContrast(uint8_t contrast) {
	CS::reset();

	if (GE12){
	}
	else{
		writeSpiCommand(nokia::NOKIA_GE8_VOLCTR);
		writeSpiData(contrast); // contrast
		writeSpiData(3);
	}

	writeSpi9BitFlush();

	CS::set();
}

template <typename SPI, typename CS, typename Reset, bool GE12>
void
modm::Nokia6610<SPI, CS, Reset, GE12>::lcdSettings()
{
	CS::reset();

	if( GE12){
	}
	else{
		// Hardware reset
		Reset::reset();
		modm::delay_ms(50);
		Reset::set();
		modm::delay_ms(50);

		// Display vontrol
		writeSpiCommand(nokia::NOKIA_GE8_DISCTL);
		writeSpiData(0x00); // default
		writeSpiData(0x20); // (32 + 1) * 4 = 132 lines (of which 130 are visible)
		writeSpiData(0x0a); // default

		// COM scan
		writeSpiCommand(nokia::NOKIA_GE8_COMSCN);
		writeSpiData(0x00);  // Scan 1-80

		// Internal oscilator ON
		writeSpiCommand(nokia::NOKIA_GE8_OSCON);

		writeSpi9BitFlush();

		CS::set();
		// wait aproximetly 100ms
		modm::delay_ms(100);
		CS::reset();

		// Sleep out
		writeSpiCommand(nokia::NOKIA_GE8_SLPOUT);

		// Voltage control
		writeSpiCommand(nokia::NOKIA_GE8_VOLCTR);
		writeSpiData(43); // middle value of V1
		writeSpiData(0x03); // middle value of resistance value

		// Power control
		writeSpiCommand(nokia::NOKIA_GE8_PWRCTR);
		writeSpiData(0x0f);   // referance voltage regulator on, circuit voltage follower on, BOOST ON

		// Data control
		writeSpiCommand(nokia::NOKIA_GE8_DATCTL);
		writeSpiData(0x04); // page scan
		writeSpiData(0x00); // RGB sequence
		writeSpiData(0x02); // 12bit per pixel mode A others may not work

		// Page Address set
		writeSpiCommand(nokia::NOKIA_GE8_PASET);
		writeSpiData(2); // start at 2 others corrupt display settings in a unpredictable way
		writeSpiData(2 + this->getHeight()-1 + 2); // 2 more for filling, but not handled by display

		// Page Column set
		writeSpiCommand(nokia::NOKIA_GE8_CASET);
		writeSpiData(0);
		writeSpiData(0 + this->getWidth()-1);

		// Display On
		writeSpiCommand(nokia::NOKIA_GE8_DISON);

	}

	writeSpi9BitFlush();
	CS::set();
}

template <typename SPI, typename CS, typename Reset, bool GE12>
void
modm::Nokia6610<SPI, CS, Reset, GE12>::update()
{
	CS::reset();

	if (GE12)
	{
		return;
	}
	else
	{
		// Display OFF
		//writeSpiCommand(nokia::NOKIA_GE8_DISOFF);

		// WRITE MEMORY
		writeSpiCommand(nokia::NOKIA_GE8_RAMWR);
	}

	static const uint32_t mask1Blank  = 0x00f000; // RGB000
	static const uint32_t mask1Filled = 0xff0000; // RGB000

	static const uint32_t mask2Blank = mask1Blank>>12; // 000RGB
	static const uint32_t mask2Filled = mask1Filled>>12; // 000RGB

	for(uint8_t x = 0; x < this->getWidth(); ++x)
	{
		for(uint8_t y = 0; y < this->getHeight()/8; ++y) {
			uint8_t group = this->buffer[x][y];
			for (uint8_t pix = 0; pix < 8; pix+=2, group>>=2){
				uint32_t data =
						((group&1)?mask1Filled:mask1Blank)|
						((group&2)?mask2Filled:mask2Blank);

				writeSpiData(data>>16);
				writeSpiData(data>>8);
				writeSpiData(data);
			}
		}
		// fill pixel not handled by  display
		uint32_t data = mask1Blank | mask2Blank;
		writeSpiData(data>>16);
		writeSpiData(data>>8);
		writeSpiData(data);
	}

	if (GE12)
	{
	}
	else
	{
		// wait approximately 100ms
		modm::delay_ms(100);

		// Display On
		writeSpiCommand(nokia::NOKIA_GE8_DISON);
	}

	writeSpi9BitFlush();
	CS::reset();
}
