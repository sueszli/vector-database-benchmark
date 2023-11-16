/*
 * Copyright (c) 2015, Niklas Hauser
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#ifndef MODM_L3GD20_HPP
#	error  "Don't include this file directly, use 'l3gd20.hpp' instead!"
#endif

#include <cstring>

// ----------------------------------------------------------------------------
// MARK: LIS302 DRIVER
template < class Transport >
modm::L3gd20<Transport>::L3gd20(Data &data, uint8_t address)
:	Transport(address), data(data), rawBuffer{7,0,0,0,0, 0,0,0, 0,0,0,0,0,0, 0,0, 0,0}
{
}

template < class Transport >
modm::ResumableResult<bool>
modm::L3gd20<Transport>::configure(Scale scale, MeasurementRate rate)
{
	RF_BEGIN();

	// MeasurementRate must be set in Control1
	rawBuffer[0] = i(rate) | 0x0F;
	rawBuffer[0] = 0x0F;
	// Scale must be set in Control4
	rawBuffer[3] = i(scale);
	data.scale = (i(scale) >> 4) + 1;

	if (RF_CALL(this->write(i(Register::CTRL_REG1), rawBuffer[0])))
	{
		RF_RETURN_CALL(this->write(i(Register::CTRL_REG4), rawBuffer[3]));
	}
	RF_END_RETURN(false);
}

template < class Transport >
modm::ResumableResult<bool>
modm::L3gd20<Transport>::updateControlRegister(uint8_t index, Control_t setMask, Control_t clearMask)
{
	RF_BEGIN();

	rawBuffer[index] = (rawBuffer[index] & ~clearMask.value) | setMask.value;
	// update the scale in the data object, if we update CTRL_REG4 (index 3)
	if (index == 3)
	{
		data.scale = ((Control4_t(rawBuffer[3]) & (Control4::FS1 | Control4::FS0)).value >> 4) + 1;
	}

	RF_END_RETURN_CALL(this->write(0x20 + index, rawBuffer[index]));
}

template < class Transport >
modm::ResumableResult<bool>
modm::L3gd20<Transport>::readRotation()
{
	RF_BEGIN();

	if (RF_CALL(this->read(i(Register::OUT_TEMP) | Transport::AddressIncrement, rawBuffer + 6, 12)))
	{
		// copy the memory
		std::memcpy(data.data, rawBuffer + 8, 6);
		data.temperature = rawBuffer[6];
		RF_RETURN(true);
	}

	RF_END_RETURN(false);
}

// ----------------------------------------------------------------------------
template < class Transport >
modm::ResumableResult<bool>
modm::L3gd20<Transport>::updateRegister(uint8_t reg, uint8_t setMask, uint8_t clearMask)
{
	RF_BEGIN();

	if (RF_CALL(this->read(reg, rawBuffer[6])))
	{
		rawBuffer[6] = (rawBuffer[6] & ~clearMask) | setMask;
		RF_RETURN_CALL(this->write(reg, rawBuffer[6]));
	}

	RF_END_RETURN(false);
}
