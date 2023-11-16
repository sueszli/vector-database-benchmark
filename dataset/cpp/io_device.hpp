/*
 * Copyright (c) 2009, Martin Rosekeit
 * Copyright (c) 2009-2012, Fabian Greif
 * Copyright (c) 2012-2013, Niklas Hauser
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#ifndef FAKE_IO_DEVICE_HPP
#define FAKE_IO_DEVICE_HPP

#include <stdint.h>

namespace modm_test
{

/// @ingroup modm_test_mock_io_device
class FakeIODevice
{
public:
	static void
	setBaudrate(uint32_t);

	static void
	write(uint8_t data);

	static bool
	read(uint8_t& byte);

	static void
	reset();


	static void
	moveSendToReceiveBuffer();

	static inline uint8_t sendBuffer[40];
	static inline uint8_t bytesSend{0};

	static inline uint8_t receiveBuffer[40];
	static inline uint8_t receivePosition{0};
	static inline uint8_t bytesReceived{0};
};

}

#endif	// FAKE_IO_DEVICE_HPP
