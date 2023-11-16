/*
 * Copyright (c) 2009-2012, Fabian Greif
 * Copyright (c) 2012-2013, Niklas Hauser
 * Copyright (c) 2013, Sascha Schade
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#ifndef	MODM_SAB_SLAVE_HPP
	#error	"Don't include this file directly, use 'slave.hpp' instead!"
#endif

// ----------------------------------------------------------------------------
template <typename T>
void
modm::sab::Response::send(const T& payload)
{
	triggered = true;
	transmitter->send(true, reinterpret_cast<const void *>(&payload), sizeof(T));
}

// ----------------------------------------------------------------------------
inline void
modm::sab::Action::call(Response& response, const void *payload)
{
	// redirect call to the actual object
	(object->*function)(response, payload);
}

// Disable warnings for Visual Studio about using 'this' in a base member
// initializer list.
// In this case though it is totally safe so it is ok to disable this warning.
#ifdef MODM_COMPILER_MSVC
#	pragma warning(disable:4355)
#endif

// ----------------------------------------------------------------------------
template <typename Interface>
modm::sab::Slave<Interface>::Slave(uint8_t address,
		modm::accessor::Flash<Action> list,
		uint8_t count) :
	ownAddress(address), actionList(list), actionCount(count),
	response(this)
{
	Interface::initialize();
}

// ----------------------------------------------------------------------------
template <typename Interface>
void
modm::sab::Slave<Interface>::update()
{
	Interface::update();
	if (Interface::isMessageAvailable())
	{
		if (Interface::getAddress() == this->ownAddress &&
				!Interface::isResponse())
		{
			this->response.triggered = false;
			this->currentCommand = Interface::getCommand();

			modm::accessor::Flash<Action> list = actionList;
			for (uint_fast8_t i = 0; i < actionCount; ++i, ++list)
			{
				Action action(*list);
				if (this->currentCommand == action.command)
				{
					if (Interface::getPayloadLength() == action.payloadLength)
					{
						// execute callback function
						action.call(this->response, Interface::getPayload());

						if (!this->response.triggered) {
							this->response.error(ERROR_NO_RESPONSE);
						}
					}
					else {
						this->response.error(ERROR_WRONG_PAYLOAD_LENGTH);
					}
					break;
				}
			}

			if (!this->response.triggered) {
				this->response.error(ERROR_NO_ACTION);
			}
		}
		Interface::dropMessage();
	}
}

// ----------------------------------------------------------------------------
template <typename Interface>
void
modm::sab::Slave<Interface>::send(bool acknowledge,
			const void *payload, std::size_t payloadLength)
{
	Flags flags;
	if (acknowledge) {
		flags = modm::sab::ACK;
	}
	else {
		flags = modm::sab::NACK;
	}

	Interface::sendMessage(this->ownAddress, flags, this->currentCommand,
			payload, payloadLength);
}
