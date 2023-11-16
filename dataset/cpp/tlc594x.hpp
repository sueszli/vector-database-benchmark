/*
 * Copyright (c) 2012-2017, Niklas Hauser
 * Copyright (c) 2015, Sascha Schade
 * Copyright (c) 2019, Linas Nikiperavicius
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#ifndef MODM_TLC594X_HPP
#define MODM_TLC594X_HPP

#include <cstdint>
#include <modm/architecture/interface/delay.hpp>
#include <modm/architecture/interface/gpio.hpp>
#include <modm/architecture/interface/spi.hpp>

namespace modm
{

/**
 * TLC594* multi-channel, daisy-chainable, constant-current sink, 12bit PWM LED driver.
 *
 * This class does not use the DCPRG pin, as writing to the EEPROM requires
 * 22V to be applied to Vprog, requiring additional external circuits.
 * Use of any EEPROM functions is therefore not implemented here.
 * Make sure that pin is connected to Vcc, otherwise Dot Correction does not work.
 *
 * This driver can be used for the 16-channel TLC5940 and 24-channel TLC5947
 * and probably similar TLC59s as well, simply by adjusting the number of
 * CHANNELS.
 * Therefore this class can also be used with daisy-chained TLC59s, e.g.
 * to control two TLC5940s set CHANNELS to 2*16.
 *
 * ####  WARNING  ####
 *
 * Each channel in the TLC594x chip drives a transistor using a feedback loop
 * to make it meet a particular current requirement.
 * If a channel is disconnected, the feedback loop will fully drive the transistor.
 * If most of the channels are disconnected (quite common in a testing
 * environment if not in production), this will end up pulling quite a bit
 * of power from the chip's 3.3 or 5v supply.
 * This can significantly heat up the chip and cause power supply issues.
 *
 *
 * @tparam CHANNELS	Number of channels must be multiples of 4, adjust for daisy-chained chips
 * @tparam	Spi		Spi interface
 * @tparam	Xlat	Level triggered latch pin
 * @tparam	Xblank	Level triggered blank pin, use modm::platform::GpioUnused if not connected
 * @tparam	Vprog	Vprog pin, use modm::platform::GpioUnused if not connected
 * @tparam	Xerr	Error pin, use modm::platform::GpioUnused if not connected
 *
 * @author	Niklas Hauser
 * @ingroup	modm_driver_tlc594x
 */
template<
	uint16_t CHANNELS,
	typename Spi,
	typename Xlat,
	typename Xblank=modm::platform::GpioUnused,
	typename Vprog=modm::platform::GpioUnused,
	typename Xerr=modm::platform::GpioUnused >
class TLC594X : private modm::NestedResumable< 1 >
{
public:
	static_assert(CHANNELS % 4 ==  0, "Template parameter CHANNELS must be a multiple of 4.");
	static_assert(CHANNELS     >= 16, "Template parameter CHANNELS must be larger than 16 (one TLC chip).");

	/**
	 * @param channels	initialize channels buffer with value, disable with -1
	 * @param dots		initialize dot correction buffer with value, disable with -1
	 * @param writeCH	write channels value to chip
	 * @param writeDC	write dots value to chip
	 * @param enable set blank pin low after initialization
	 */
	void
	initialize(uint16_t channels=0, uint8_t dots=63, bool writeCH=true, bool writeDC=true, bool enable=true);

	/// set the 12bit value of a channel
	/// call writeChannels() to update the chip
	void
	setChannel(uint16_t channel, uint16_t value);

	/// @param value	the 12bit value of all channels
	void
	setAllChannels(uint16_t value);

	/// get the stored 12bit value of a channel
	/// this does reflect the actual value in the chip
	uint16_t
	getChannel(uint16_t channel);

	/// set the 6bit dot correction value of a channel
	/// call writeChannels() to update the chip
	void
	setDotCorrection(uint16_t channel, uint8_t value);

	/// @param value the 6bit dot correction value of all channels
	/// @param update write data to chip
	void
	setAllDotCorrection(uint8_t value);

	/// get the stored 6bit dot correction value of a channel
	/// this does reflect the actual value in the chip
	uint8_t
	getDotCorrection(uint16_t channel);

	/// transfer channel data to driver chip
	modm::ResumableResult<void>
	writeChannels(bool flush=true);

	/// transfer dot correction data to driver chip
	modm::ResumableResult<void>
	writeDotCorrection(bool flush=true);

	/// writes data from the input shift register to either GS or DC register.
	void
	latch();

	/// sets the blank pin low
	void
	enable()
	{
		enabled=true;
		Xblank::reset();
	}

	/// sets the blank pin high
	void
	disable()
	{
		enabled=false;
		Xblank::set();
	}

	/// @return true if LOD or TEF is detected
	bool
	isError()
	{
		return !Xerr::read();
	}

	uint8_t*
	getGrayscaleData()
	{
		return gs;
	}

	uint8_t*
	getDotCorrectionData()
	{
		return dc;
	}

	uint8_t*
	getStatusData()
	{
		return status;
	}

	bool
	isEnabled()
	{
		return enabled;
	}

private:
	uint8_t status[CHANNELS*3/2];
	uint8_t gs[CHANNELS*3/2];
	uint8_t dc[CHANNELS*3/4];
	bool enabled = false;
};

} //namespace modm

#include "tlc594x_impl.hpp"

#endif // MODM_TLC594X_HPP
