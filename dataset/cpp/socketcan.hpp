/*
 * Copyright (c) 2016, Sascha Schade
 * Copyright (c) 2017, Niklas Hauser
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#ifndef MODM_HOSTED_SOCKETCAN_HPP
#define MODM_HOSTED_SOCKETCAN_HPP

#include <iostream>

#include <modm/architecture/interface/can.hpp>

namespace modm
{

namespace platform
{

/// @ingroup modm_platform_socketcan
class SocketCan : public ::modm::Can
{
public:
	SocketCan() = default;

	~SocketCan();

	bool
	open(std::string deviceName);

	void
	close();

	bool
	isMessageAvailable();

	bool
	getMessage(can::Message& message);

	inline bool
	isReadyToSend() { return true; }

	BusState
	getBusState();

	bool
	sendMessage(const can::Message& message);

private:
	int skt{-1};
};

} // namespace platform
} // modm namespace

#endif // MODM_HOSTED_SOCKETCAN_HPP
