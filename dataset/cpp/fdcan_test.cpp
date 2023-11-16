/*
 * Copyright (c) 2021, Christopher Durand
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#include "fdcan_test.hpp"

#include <modm/platform.hpp>
#include <modm/board.hpp>

using namespace modm::platform;

void
FdcanTest::testSendReceive()
{
	Fdcan1::initialize<Board::SystemClock, 500_kbps, 1_pct>(9, Fdcan1::Mode::TestInternalLoopback);
	// receive all extended messages
	Fdcan1::setExtendedFilter(0, Fdcan1::FilterConfig::Fifo1,
			modm::can::ExtendedIdentifier(0),
			modm::can::ExtendedMask(0));

	modm::can::Message message{0x12345678, 7};
	constexpr std::string_view data = "\xDE\xAD\xBE\xEF\x12\x34\x56";
	std::copy(std::begin(data), std::begin(data) + 7, message.data);

	TEST_ASSERT_FALSE(Fdcan1::isMessageAvailable());
	TEST_ASSERT_TRUE(Fdcan1::sendMessage(message));

	modm::delay_ms(1);

	modm::can::Message receivedMessage;
	TEST_ASSERT_TRUE(Fdcan1::getMessage(receivedMessage));
	TEST_ASSERT_EQUALS(receivedMessage.getIdentifier(), 0x12345678u);
	TEST_ASSERT_EQUALS(receivedMessage.getLength(), 7);
	TEST_ASSERT_TRUE(receivedMessage.isExtended());
	TEST_ASSERT_FALSE(receivedMessage.isRemoteTransmitRequest());
	TEST_ASSERT_TRUE(std::equal(std::begin(data), std::begin(data) + 7, message.data));
}

void
FdcanTest::testFilters()
{
	Fdcan1::initialize<Board::SystemClock, 500_kbps, 1_pct>(9, Fdcan1::Mode::TestInternalLoopback);
	// receive all extended messages
	Fdcan1::setExtendedFilter(0, Fdcan1::FilterConfig::Fifo1,
			modm::can::ExtendedIdentifier(0),
			modm::can::ExtendedMask(0));

	Fdcan1::setStandardFilter(27, Fdcan1::FilterConfig::Fifo0,
			modm::can::StandardIdentifier(0x108),
			modm::can::StandardMask(0x1F8));

	modm::can::Message message{0x188, 0};
	message.setExtended(false);
	Fdcan1::sendMessage(message);
	modm::delay_ms(1);
	TEST_ASSERT_FALSE(Fdcan1::isMessageAvailable());

	message.setIdentifier(0xF09);
	Fdcan1::sendMessage(message);
	modm::delay_ms(1);
	TEST_ASSERT_TRUE(Fdcan1::isMessageAvailable());
	TEST_ASSERT_TRUE(Fdcan1::getMessage(message));
	TEST_ASSERT_FALSE(message.isExtended());
}

void
FdcanTest::testBuffers()
{
	Fdcan1::initialize<Board::SystemClock, 500_kbps, 1_pct>(9, Fdcan1::Mode::TestInternalLoopback);
	// receive all extended messages
	Fdcan1::setExtendedFilter(0, Fdcan1::FilterConfig::Fifo1,
			modm::can::ExtendedIdentifier(0),
			modm::can::ExtendedMask(0));

	// send (RxBufferSize + 2) messages, exceeds internal peripheral queue size (3
	// msgs) as well as the software queue size, but not both added together. So
	// no message should get lost.
	const uint_fast16_t numberOfMsgs = Fdcan1::RxBufferSize + 2;

	modm::can::Message message{0x4711, 0};
	for (uint_fast16_t i = 0; i <= numberOfMsgs; ++i) {
		uint_fast8_t length = i % 8;
		message.setLength(length);
		for (uint_fast8_t dataIndex = 0; dataIndex < length; ++dataIndex) {
			message.data[dataIndex] = i;
		}
		Fdcan1::sendMessage(message);
	}

	modm::delay_ms(10);

	// try to receive same messages
	modm::can::Message receivedMessage;
	for (uint_fast16_t i = 0; i <= numberOfMsgs; ++i) {
		TEST_ASSERT_TRUE(Fdcan1::getMessage(receivedMessage));

		TEST_ASSERT_EQUALS(receivedMessage.getIdentifier(), 0x4711u);
		uint_fast8_t length = i % 8;
		TEST_ASSERT_EQUALS(receivedMessage.getLength(), length);
		for (uint_fast8_t dataIndex = 0; dataIndex < length; ++dataIndex) {
			TEST_ASSERT_EQUALS(receivedMessage.data[dataIndex], i);
		}
	}
	TEST_ASSERT_FALSE(Fdcan1::isMessageAvailable());
	TEST_ASSERT_FALSE(Fdcan1::getMessage(message));
}
