/*
 * Copyright 2014-2023 Real Logic Limited.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <array>

#include <gtest/gtest.h>

#include "MockAtomicBuffer.h"
#include "concurrent/logbuffer/TermReader.h"

using namespace aeron::concurrent::logbuffer;
using namespace aeron::concurrent::mock;
using namespace aeron::concurrent;
using namespace aeron;

#define TERM_BUFFER_CAPACITY (LogBufferDescriptor::TERM_MIN_LENGTH)
#define META_DATA_BUFFER_CAPACITY (LogBufferDescriptor::LOG_META_DATA_LENGTH)
#define HDR_LENGTH (DataFrameHeader::LENGTH)
#define TERM_BUFFER_UNALIGNED_CAPACITY (LogBufferDescriptor::TERM_MIN_LENGTH + FrameDescriptor::FRAME_ALIGNMENT - 1)
#define INITIAL_TERM_ID 7

typedef std::array<std::uint8_t, TERM_BUFFER_CAPACITY> term_buffer_t;
typedef std::array<std::uint8_t, META_DATA_BUFFER_CAPACITY> meta_data_buffer_t;
typedef std::array<std::uint8_t, HDR_LENGTH> hdr_t;
typedef std::array<std::uint8_t, TERM_BUFFER_UNALIGNED_CAPACITY> log_buffer_unaligned_t;

void rethrowHandler(const std::exception &ex)
{
    throw ex;
}

class MockDataHandler
{
public:
    MOCK_CONST_METHOD4(onData, void(AtomicBuffer&, util::index_t, util::index_t, Header &));
};

using namespace std::placeholders;

class TermReaderTest : public testing::Test
{
public:
    TermReaderTest() :
        m_log(&m_logBuffer[0], m_logBuffer.size()),
        m_fragmentHeader(INITIAL_TERM_ID, TERM_BUFFER_CAPACITY, nullptr),
        m_handler_func(std::bind(&MockDataHandler::onData, &m_handler, _1, _2, _3, _4))
    {
        m_logBuffer.fill(0);
    }

    void SetUp() override
    {
        m_logBuffer.fill(0);
    }

protected:
    AERON_DECL_ALIGNED(term_buffer_t m_logBuffer, 16) = {};
    MockAtomicBuffer m_log;
    Header m_fragmentHeader;
    MockDataHandler m_handler;
    fragment_handler_t m_handler_func;
};

TEST_F(TermReaderTest, shouldReadFirstMessage)
{
    const util::index_t msgLength = 1;
    const util::index_t frameLength = DataFrameHeader::LENGTH + msgLength;
    const util::index_t alignedFrameLength = util::BitUtil::align(frameLength, FrameDescriptor::FRAME_ALIGNMENT);
    const std::int32_t termOffset = 0;
    testing::Sequence sequence;

    EXPECT_CALL(m_log, getInt32Volatile(FrameDescriptor::lengthOffset(0)))
        .Times(1)
        .InSequence(sequence)
        .WillOnce(testing::Return(frameLength));
    EXPECT_CALL(m_log, getUInt16(FrameDescriptor::typeOffset(0)))
        .Times(1)
        .InSequence(sequence)
        .WillOnce(testing::Return(DataFrameHeader::HDR_TYPE_DATA));
    EXPECT_CALL(m_handler, onData(testing::Ref(m_log), DataFrameHeader::LENGTH, msgLength, testing::_))
        .Times(1)
        .InSequence(sequence);
    EXPECT_CALL(m_log, getInt32Volatile(FrameDescriptor::lengthOffset(alignedFrameLength)))
        .Times(1)
        .InSequence(sequence)
        .WillOnce(testing::Return(0));

    TermReader::ReadOutcome readOutcome{};

    TermReader::read(readOutcome, m_log, termOffset, m_handler_func, INT_MAX, m_fragmentHeader, rethrowHandler);

    EXPECT_EQ(readOutcome.offset, alignedFrameLength);
    EXPECT_EQ(readOutcome.fragmentsRead, 1);
}

TEST_F(TermReaderTest, shouldNotReadPastTail)
{
    const std::int32_t termOffset = 0;
    testing::Sequence sequence;

    EXPECT_CALL(m_log, getInt32Volatile(FrameDescriptor::lengthOffset(0)))
        .Times(1)
        .InSequence(sequence)
        .WillOnce(testing::Return(0));
    EXPECT_CALL(m_log, getUInt16(FrameDescriptor::typeOffset(0)))
        .Times(0);
    EXPECT_CALL(m_handler, onData(testing::Ref(m_log), DataFrameHeader::LENGTH, testing::_, testing::_))
        .Times(0);

    TermReader::ReadOutcome readOutcome{};

    TermReader::read(readOutcome, m_log, termOffset, m_handler_func, INT_MAX, m_fragmentHeader, rethrowHandler);

    EXPECT_EQ(readOutcome.offset, termOffset);
    EXPECT_EQ(readOutcome.fragmentsRead, 0);
}

TEST_F(TermReaderTest, shouldReadOneLimitedMessage)
{
    const util::index_t msgLength = 1;
    const util::index_t frameLength = DataFrameHeader::LENGTH + msgLength;
    const util::index_t alignedFrameLength = util::BitUtil::align(frameLength, FrameDescriptor::FRAME_ALIGNMENT);
    const std::int32_t termOffset = 0;
    testing::Sequence sequence;

    EXPECT_CALL(m_log, getInt32Volatile(testing::_))
        .Times(1)
        .InSequence(sequence)
        .WillOnce(testing::Return(frameLength));
    EXPECT_CALL(m_log, getUInt16(testing::_))
        .Times(1)
        .InSequence(sequence)
        .WillOnce(testing::Return(DataFrameHeader::HDR_TYPE_DATA));
    EXPECT_CALL(m_handler, onData(testing::Ref(m_log), DataFrameHeader::LENGTH, msgLength, testing::_))
        .Times(1)
        .InSequence(sequence);

    TermReader::ReadOutcome readOutcome{};

    TermReader::read(readOutcome, m_log, termOffset, m_handler_func, 1, m_fragmentHeader, rethrowHandler);

    EXPECT_EQ(readOutcome.offset, alignedFrameLength);
    EXPECT_EQ(readOutcome.fragmentsRead, 1);
}

TEST_F(TermReaderTest, shouldReadMultipleMessages)
{
    const util::index_t msgLength = 1;
    const util::index_t frameLength = DataFrameHeader::LENGTH + msgLength;
    const util::index_t alignedFrameLength = util::BitUtil::align(frameLength, FrameDescriptor::FRAME_ALIGNMENT);
    const std::int32_t termOffset = 0;
    testing::Sequence sequence;

    EXPECT_CALL(m_log, getInt32Volatile(FrameDescriptor::lengthOffset(0)))
        .Times(1)
        .InSequence(sequence)
        .WillOnce(testing::Return(frameLength));
    EXPECT_CALL(m_log, getUInt16(FrameDescriptor::typeOffset(0)))
        .Times(1)
        .InSequence(sequence)
        .WillOnce(testing::Return(DataFrameHeader::HDR_TYPE_DATA));
    EXPECT_CALL(m_handler, onData(testing::Ref(m_log), DataFrameHeader::LENGTH, msgLength, testing::_))
        .Times(1)
        .InSequence(sequence);
    EXPECT_CALL(m_log, getInt32Volatile(FrameDescriptor::lengthOffset(alignedFrameLength)))
        .Times(1)
        .InSequence(sequence)
        .WillOnce(testing::Return(frameLength));
    EXPECT_CALL(m_log, getUInt16(FrameDescriptor::typeOffset(alignedFrameLength)))
        .Times(1)
        .InSequence(sequence)
        .WillOnce(testing::Return(DataFrameHeader::HDR_TYPE_DATA));
    EXPECT_CALL(m_handler,
                onData(testing::Ref(m_log), alignedFrameLength + DataFrameHeader::LENGTH, msgLength, testing::_))
        .Times(1)
        .InSequence(sequence);
    EXPECT_CALL(m_log, getInt32Volatile(FrameDescriptor::lengthOffset(alignedFrameLength * 2)))
        .Times(1)
        .InSequence(sequence)
        .WillOnce(testing::Return(0));

    TermReader::ReadOutcome readOutcome{};

    TermReader::read(readOutcome, m_log, termOffset, m_handler_func, INT_MAX, m_fragmentHeader, rethrowHandler);

    EXPECT_EQ(readOutcome.offset, alignedFrameLength * 2);
    EXPECT_EQ(readOutcome.fragmentsRead, 2);
}

TEST_F(TermReaderTest, shouldReadLastMessage)
{
    const util::index_t msgLength = 1;
    const util::index_t frameLength = DataFrameHeader::LENGTH + msgLength;
    const util::index_t alignedFrameLength = util::BitUtil::align(frameLength, FrameDescriptor::FRAME_ALIGNMENT);
    const std::int32_t startOfMessage = TERM_BUFFER_CAPACITY - alignedFrameLength;
    testing::Sequence sequence;

    EXPECT_CALL(m_log, getInt32Volatile(FrameDescriptor::lengthOffset(startOfMessage)))
        .Times(1)
        .InSequence(sequence)
        .WillOnce(testing::Return(frameLength));
    EXPECT_CALL(m_log, getUInt16(FrameDescriptor::typeOffset(startOfMessage)))
        .Times(1)
        .InSequence(sequence)
        .WillOnce(testing::Return(DataFrameHeader::HDR_TYPE_DATA));
    EXPECT_CALL(m_handler, onData(testing::Ref(m_log), startOfMessage + DataFrameHeader::LENGTH, msgLength, testing::_))
        .Times(1)
        .InSequence(sequence);

    TermReader::ReadOutcome readOutcome{};

    TermReader::read(readOutcome, m_log, startOfMessage, m_handler_func, INT_MAX, m_fragmentHeader, rethrowHandler);

    EXPECT_EQ(readOutcome.offset, TERM_BUFFER_CAPACITY);
    EXPECT_EQ(readOutcome.fragmentsRead, 1);
}

TEST_F(TermReaderTest, shouldNotReadLastMessageWhenPadding)
{
    const util::index_t msgLength = 1;
    const util::index_t frameLength = DataFrameHeader::LENGTH + msgLength;
    const util::index_t alignedFrameLength = util::BitUtil::align(frameLength, FrameDescriptor::FRAME_ALIGNMENT);
    const util::index_t startOfMessage = TERM_BUFFER_CAPACITY - alignedFrameLength;
    testing::Sequence sequence;

    EXPECT_CALL(m_log, getInt32Volatile(FrameDescriptor::lengthOffset(startOfMessage)))
        .Times(1)
        .InSequence(sequence)
        .WillOnce(testing::Return(frameLength));
    EXPECT_CALL(m_log, getUInt16(FrameDescriptor::typeOffset(startOfMessage)))
        .Times(1)
        .InSequence(sequence)
        .WillOnce(testing::Return(DataFrameHeader::HDR_TYPE_PAD));
    EXPECT_CALL(m_handler, onData(testing::Ref(m_log), startOfMessage + DataFrameHeader::LENGTH, msgLength, testing::_))
        .Times(0);

    TermReader::ReadOutcome readOutcome{};

    TermReader::read(readOutcome, m_log, startOfMessage, m_handler_func, INT_MAX, m_fragmentHeader, rethrowHandler);

    EXPECT_EQ(readOutcome.offset, TERM_BUFFER_CAPACITY);
    EXPECT_EQ(readOutcome.fragmentsRead, 0);
}
