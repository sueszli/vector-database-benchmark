/*
 * Copyright (c) 2012-2014, Sascha Schade
 * Copyright (c) 2013, Fabian Greif
 * Copyright (c) 2013, Hans Schily
 * Copyright (c) 2013, Niklas Hauser
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#ifndef MODM_SIEMENS_S75_HPP
#error	"Don't include this file directly, use 'siemens_s75.hpp' instead!"
#endif

// ----------------------------------------------------------------------------

template <typename MEMORY, typename RESET, uint16_t WIDTH, uint16_t HEIGHT, modm::Orientation ORIENTATION>
void
modm::SiemensS75Common<MEMORY, RESET, WIDTH, HEIGHT, ORIENTATION>::update()
{
	uint8_t width;
	uint8_t height;

	switch (ORIENTATION)
	{
	case modm::Orientation::LandscapeLeft:
	case modm::Orientation::LandscapeRight:
	{
		// Set CGRAM Address to height - 1 = upper left corner
		interface.writeRegister(0x21, 0xAF00);
		// size of the MODM Display buffer, not the hardware pixels
		width  = 176;
		height = 136 / 8; // Display is only 132 pixels high.
	}
	break;
	case modm::Orientation::Portrait:
	{
		// Set CGRAM Address to 0 = upper left corner
		interface.writeRegister(0x21, 0x0000);
		width  = 132;
		height = 176 / 8;
	}
	break;
	case modm::Orientation::PortraitUpsideDown:
	{
		interface.writeRegister(0x21, 131);
		width  = 132;
		height = 176 / 8;
	}
	break;
	}

	// Set instruction register to "RAM Data write"
	interface.writeIndex(0x22);

	const uint16_t maskBlank  = 0x0000; // RRRR RGGG GGGB BBBB
	const uint16_t maskFilled = 0x37e0; // RRRR RGGG GGGB BBBB

	for (int_fast16_t x = 0; x < width; ++x)
	{
		for (int_fast16_t y = 0; y < height; ++y)
		{
			// group of 8 black-and-white pixels
			uint_fast8_t group = this->buffer[x][y];
			uint16_t PortBuffer[8];
			uint_fast8_t PortIdx = 0;

			uint_fast8_t pixels;
			if ((ORIENTATION == modm::Orientation::LandscapeLeft) ||
				(ORIENTATION == modm::Orientation::LandscapeRight))
			{
				// Only 4 pixels at the lower end of the display in landscape mode
				if (y == (height - 1)) {
					// The last pixels
					pixels = 4;
				}
				else {
					pixels = 8;
				}
			} else {
				pixels = 8;
			}

			for (uint_fast8_t pix = 0; pix < pixels; ++pix, group >>= 1) {
				if (group & 1) {
					PortBuffer[PortIdx++] = maskFilled;
				}
				else {
					PortBuffer[PortIdx++] = maskBlank;
				}
			} // pix

			for (uint_fast16_t ii = 0; ii < PortIdx; ++ii) {
				interface.writeData(PortBuffer[ii]);
			}
		} // y
	} // x
}


template <typename MEMORY, typename RESET, uint16_t WIDTH, uint16_t HEIGHT, modm::Orientation ORIENTATION>
void
modm::SiemensS75Common<MEMORY, RESET, WIDTH, HEIGHT, ORIENTATION>::initialize()
{
	// Reset pin
	RESET::setOutput(false);

	SiemensS75Common<MEMORY, RESET, WIDTH, HEIGHT, ORIENTATION>::lcdSettings();

	this->clear();
}

template <typename MEMORY, typename RESET, uint16_t WIDTH, uint16_t HEIGHT, modm::Orientation ORIENTATION>
void
modm::SiemensS75Common<MEMORY, RESET, WIDTH, HEIGHT, ORIENTATION>::lcdCls(const uint16_t colour)
{
	// Set CGRAM Address to 0 = upper left corner
	interface.writeRegister(0x21, 0x0000);

	// Set instruction register to "RAM Data write"
	interface.writeIndex(0x22);

	// Write all pixels to the same colour
	for (uint_fast16_t ii = 0; ii < 132 * 176; ++ii) {
		interface.writeData(colour);
	}
}

template <typename MEMORY, typename RESET, uint16_t WIDTH, uint16_t HEIGHT, modm::Orientation ORIENTATION>
void
modm::SiemensS75Common<MEMORY, RESET, WIDTH, HEIGHT, ORIENTATION>::lcdSettings()
{
	// Hardware reset is low from initialize
	modm::delay_ms(50);
	RESET::set();
	modm::delay_ms(50);

	interface.writeRegister(0x00, 0x0001); // R00: Start oscillation
	modm::delay_ms(10);

	//power on sequence
	interface.writeRegister(0x10, 0x1f92);	// R10: Power Control 1
	interface.writeRegister(0x11, 0x0014);	// R11: Power Control 2
	interface.writeRegister(0x00, 0x0001);	// R00: Start oscillation
	interface.writeRegister(0x10, 0x1f92);	// R10: Power Control 1
	interface.writeRegister(0x11, 0x001c);	// R11: Power Control 2
	interface.writeRegister(0x28, 0x0006);	// R28: VCOM OTP (1)
	interface.writeRegister(0x02, 0x0000);	// R02: LCD drive AC control
	interface.writeRegister(0x12, 0x040f);	// R12: Power Control 2

	modm::delay_ms(100);

	// R03: Entry mode
	switch(ORIENTATION)
	{
	case modm::Orientation::LandscapeLeft:
		interface.writeRegister(0x03, 0x7820);
		break;
	case modm::Orientation::LandscapeRight:
		interface.writeRegister(0x03, 0x7810);
		break;
	case modm::Orientation::Portrait:
		interface.writeRegister(0x03, 0x7838);
		break;
	case modm::Orientation::PortraitUpsideDown:
		interface.writeRegister(0x03, 0x7808);
		break;
	}

	/**
	 * Bit 0 set: stopped working
	 * Bit 1 set: no change
	 * 0x7830 | 0x0003: no change
	 * 0x7830 | 0x0004: stopped working
	 * 0x7830 | 0x0008:	landscape
	 *
	 * 0x7838 | 0x0001:	stopped working
	 * 0x7838 | 0x0002:	no change
	 * 0x7838 | 0x0003:	colour inverted
	 * 0x7838 | 0x0004:	no change
	 * 0x7800
	 */

	interface.writeRegister(0x01, 0x31af);	// R01: Driver output control
	interface.writeRegister(0x07, 0x0033);	// R07: Display Control

	interface.writeRegister(0x44, 0x8300); // Horizontal RAM Address
	interface.writeRegister(0x45, 0xaf00); // Vertical RAM Address
	modm::delay_ms(10);

	// colourful test
	lcdCls(0x0000); // black
}
