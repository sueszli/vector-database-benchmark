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
#include <thread>
#include <atomic>
#include <vector>

#include <gtest/gtest.h>

#include "MockAtomicBuffer.h"
#include "concurrent/ringbuffer/RingBufferDescriptor.h"
#include "concurrent/ringbuffer/ManyToOneRingBuffer.h"

using namespace aeron;
using namespace aeron::concurrent::ringbuffer;
using namespace aeron::concurrent;
using namespace aeron::concurrent::mock;

#define CAPACITY (1024)
#define BUFFER_SZ (CAPACITY + RingBufferDescriptor::TRAILER_LENGTH)
#define ODD_BUFFER_SZ ((CAPACITY - 1) + RingBufferDescriptor::TRAILER_LENGTH)

typedef std::array<std::uint8_t, BUFFER_SZ> buffer_t;
typedef std::array<std::uint8_t, ODD_BUFFER_SZ> odd_sized_buffer_t;

const static std::int32_t MSG_TYPE_ID = 101;
const static util::index_t HEAD_COUNTER_INDEX = 1024 + RingBufferDescriptor::HEAD_POSITION_OFFSET;
const static util::index_t TAIL_COUNTER_INDEX = 1024 + RingBufferDescriptor::TAIL_POSITION_OFFSET;

class ManyToOneRingBufferTest : public testing::Test
{
public:

    ManyToOneRingBufferTest() :
        m_ab(&m_buffer[0], m_buffer.size()),
        m_srcAb(&m_srcBuffer[0], m_srcBuffer.size()),
        m_ringBuffer(m_ab),
        m_mockAb(&m_buffer[0], m_buffer.size()),
        m_mockRingBuffer(m_mockAb)
    {
        clear();
    }

protected:
    AERON_DECL_ALIGNED(buffer_t m_buffer, 16) = {};
    AERON_DECL_ALIGNED(buffer_t m_srcBuffer, 16) = {};
    AtomicBuffer m_ab;
    AtomicBuffer m_srcAb;
    ManyToOneRingBuffer m_ringBuffer;
    MockAtomicBuffer m_mockAb;
    ManyToOneRingBuffer m_mockRingBuffer;

    inline void clear()
    {
        m_buffer.fill(0);
        m_srcBuffer.fill(0);
    }
};

TEST_F(ManyToOneRingBufferTest, shouldCalculateCapacityForBuffer)
{
    EXPECT_EQ(m_ab.capacity(), BUFFER_SZ);
    EXPECT_EQ(m_ringBuffer.capacity(), BUFFER_SZ - RingBufferDescriptor::TRAILER_LENGTH);
}

TEST_F(ManyToOneRingBufferTest, shouldThrowForCapacityNotPowerOfTwo)
{
    AERON_DECL_ALIGNED(odd_sized_buffer_t testBuffer, 16);

    testBuffer.fill(0);
    AtomicBuffer ab(&testBuffer[0], testBuffer.size());

    ASSERT_THROW(
        {
            ManyToOneRingBuffer ringBuffer(ab);
        }, util::IllegalArgumentException);
}

TEST_F(ManyToOneRingBufferTest, shouldThrowWhenMaxMessageSizeExceeded)
{
    ASSERT_THROW(
        {
            m_ringBuffer.write(MSG_TYPE_ID, m_srcAb, 0, m_ringBuffer.maxMsgLength() + 1);
        }, util::IllegalArgumentException);
}

TEST_F(ManyToOneRingBufferTest, shouldWriteToEmptyBuffer)
{
    util::index_t tail = 0;
    util::index_t tailIndex = 0;
    util::index_t length = 8;
    util::index_t recordLength = length + RecordDescriptor::HEADER_LENGTH;
    util::index_t srcIndex = 0;
    util::index_t alignedRecordLength = util::BitUtil::align(recordLength, RecordDescriptor::ALIGNMENT);

    ASSERT_TRUE(m_ringBuffer.write(MSG_TYPE_ID, m_srcAb, srcIndex, length));

    EXPECT_EQ(m_ab.getInt32(RecordDescriptor::lengthOffset(tailIndex)), recordLength);
    EXPECT_EQ(m_ab.getInt32(RecordDescriptor::typeOffset(tailIndex)), MSG_TYPE_ID);
    EXPECT_EQ(m_ab.getInt64(TAIL_COUNTER_INDEX), tail + alignedRecordLength);
}

TEST_F(ManyToOneRingBufferTest, shouldRejectWriteWhenInsufficientSpace)
{
    util::index_t length = 100;
    util::index_t head = 0;
    util::index_t tail =
        head + (CAPACITY - util::BitUtil::align(length - RecordDescriptor::ALIGNMENT, RecordDescriptor::ALIGNMENT));
    util::index_t srcIndex = 0;

    m_ab.putInt64(HEAD_COUNTER_INDEX, head);
    m_ab.putInt64(TAIL_COUNTER_INDEX, tail);

    ASSERT_FALSE(m_ringBuffer.write(MSG_TYPE_ID, m_srcAb, srcIndex, length));

    EXPECT_EQ(m_ab.getInt64(TAIL_COUNTER_INDEX), tail);
}

TEST_F(ManyToOneRingBufferTest, shouldRejectWriteWhenBufferFull)
{
    util::index_t length = 8;
    util::index_t head = 0;
    util::index_t tail = head + CAPACITY;
    util::index_t srcIndex = 0;

    m_ab.putInt64(HEAD_COUNTER_INDEX, head);
    m_ab.putInt64(TAIL_COUNTER_INDEX, tail);

    ASSERT_FALSE(m_ringBuffer.write(MSG_TYPE_ID, m_srcAb, srcIndex, length));

    EXPECT_EQ(m_ab.getInt64(TAIL_COUNTER_INDEX), tail);
}

TEST_F(ManyToOneRingBufferTest, shouldInsertPaddingRecordPlusMessageOnBufferWrap)
{
    util::index_t length = 100;
    util::index_t recordLength = length + RecordDescriptor::HEADER_LENGTH;
    util::index_t alignedRecordLength = util::BitUtil::align(recordLength, RecordDescriptor::ALIGNMENT);
    util::index_t tail = CAPACITY - RecordDescriptor::ALIGNMENT;
    util::index_t head = tail - (RecordDescriptor::ALIGNMENT * 4);
    util::index_t srcIndex = 0;

    m_ab.putInt64(HEAD_COUNTER_INDEX, head);
    m_ab.putInt64(TAIL_COUNTER_INDEX, tail);

    ASSERT_TRUE(m_ringBuffer.write(MSG_TYPE_ID, m_srcAb, srcIndex, length));

    EXPECT_EQ(m_ab.getInt32(RecordDescriptor::typeOffset(tail)), RecordDescriptor::PADDING_MSG_TYPE_ID);
    EXPECT_EQ(m_ab.getInt32(RecordDescriptor::lengthOffset(tail)), RecordDescriptor::ALIGNMENT);

    EXPECT_EQ(m_ab.getInt32(RecordDescriptor::lengthOffset(0)), recordLength);
    EXPECT_EQ(m_ab.getInt32(RecordDescriptor::typeOffset(0)), MSG_TYPE_ID);
    EXPECT_EQ(m_ab.getInt64(TAIL_COUNTER_INDEX), tail + alignedRecordLength + RecordDescriptor::ALIGNMENT);
}

TEST_F(ManyToOneRingBufferTest, shouldInsertPaddingRecordPlusMessageOnBufferWrapWithHeadEqualToTail)
{
    util::index_t length = 100;
    util::index_t recordLength = length + RecordDescriptor::HEADER_LENGTH;
    util::index_t alignedRecordLength = util::BitUtil::align(recordLength, RecordDescriptor::ALIGNMENT);
    util::index_t tail = CAPACITY - RecordDescriptor::ALIGNMENT;
    util::index_t head = tail;
    util::index_t srcIndex = 0;

    m_ab.putInt64(HEAD_COUNTER_INDEX, head);
    m_ab.putInt64(TAIL_COUNTER_INDEX, tail);

    ASSERT_TRUE(m_ringBuffer.write(MSG_TYPE_ID, m_srcAb, srcIndex, length));

    EXPECT_EQ(m_ab.getInt32(RecordDescriptor::typeOffset(tail)), RecordDescriptor::PADDING_MSG_TYPE_ID);
    EXPECT_EQ(m_ab.getInt32(RecordDescriptor::lengthOffset(tail)), RecordDescriptor::ALIGNMENT);

    EXPECT_EQ(m_ab.getInt32(RecordDescriptor::lengthOffset(0)), recordLength);
    EXPECT_EQ(m_ab.getInt32(RecordDescriptor::typeOffset(0)), MSG_TYPE_ID);
    EXPECT_EQ(m_ab.getInt64(TAIL_COUNTER_INDEX), tail + alignedRecordLength + RecordDescriptor::ALIGNMENT);
}

TEST_F(ManyToOneRingBufferTest, shouldReadNothingFromEmptyBuffer)
{
    util::index_t tail = 0;
    util::index_t head = 0;

    m_ab.putInt64(HEAD_COUNTER_INDEX, head);
    m_ab.putInt64(TAIL_COUNTER_INDEX, tail);

    int timesCalled = 0;
    const int messagesRead = m_ringBuffer.read(
        [&](std::int32_t, concurrent::AtomicBuffer &, util::index_t, util::index_t)
        {
            timesCalled++;
        });

    EXPECT_EQ(messagesRead, 0);
    EXPECT_EQ(timesCalled, 0);
}

TEST_F(ManyToOneRingBufferTest, shouldReadSingleMessage)
{
    util::index_t length = 8;
    util::index_t head = 0;
    util::index_t recordLength = length + RecordDescriptor::HEADER_LENGTH;
    util::index_t alignedRecordLength = util::BitUtil::align(recordLength, RecordDescriptor::ALIGNMENT);
    util::index_t tail = alignedRecordLength;

    m_ab.putInt64(HEAD_COUNTER_INDEX, head);
    m_ab.putInt64(TAIL_COUNTER_INDEX, tail);

    m_ab.putInt32(RecordDescriptor::typeOffset(0), MSG_TYPE_ID);
    m_ab.putInt32(RecordDescriptor::lengthOffset(0), recordLength);

    int timesCalled = 0;
    const int messagesRead = m_ringBuffer.read(
        [&](std::int32_t, concurrent::AtomicBuffer &, util::index_t, util::index_t)
        {
            timesCalled++;
        });

    EXPECT_EQ(messagesRead, 1);
    EXPECT_EQ(timesCalled, 1);
    EXPECT_EQ(m_ab.getInt64(HEAD_COUNTER_INDEX), head + alignedRecordLength);

    for (int i = 0; i < RecordDescriptor::ALIGNMENT; i += 4)
    {
        EXPECT_EQ(m_ab.getInt32(i), 0) << "buffer has not been zeroed between indexes " << i << "-" << i + 3;
    }
}

TEST_F(ManyToOneRingBufferTest, shouldNotReadSingleMessagePartWayThroughWriting)
{
    util::index_t length = 8;
    util::index_t head = 0;
    util::index_t recordLength = length + RecordDescriptor::HEADER_LENGTH;
    util::index_t alignedRecordLength = util::BitUtil::align(recordLength, RecordDescriptor::ALIGNMENT);
    util::index_t endTail = alignedRecordLength;

    m_ab.putInt64(TAIL_COUNTER_INDEX, endTail);
    m_ab.putInt32(RecordDescriptor::typeOffset(0), MSG_TYPE_ID);
    m_ab.putInt32(RecordDescriptor::lengthOffset(0), -recordLength);

    int timesCalled = 0;
    const int messagesRead = m_ringBuffer.read(
        [&](std::int32_t, concurrent::AtomicBuffer &, util::index_t, util::index_t)
        {
            timesCalled++;
        });

    EXPECT_EQ(messagesRead, 0);
    EXPECT_EQ(timesCalled, 0);
    EXPECT_EQ(m_ab.getInt64(HEAD_COUNTER_INDEX), head);
}

TEST_F(ManyToOneRingBufferTest, shouldReadTwoMessages)
{
    util::index_t length = 8;
    util::index_t head = 0;
    util::index_t recordLength = length + RecordDescriptor::HEADER_LENGTH;
    util::index_t alignedRecordLength = util::BitUtil::align(recordLength, RecordDescriptor::ALIGNMENT);
    util::index_t tail = alignedRecordLength * 2;

    m_ab.putInt64(HEAD_COUNTER_INDEX, head);
    m_ab.putInt64(TAIL_COUNTER_INDEX, tail);

    m_ab.putInt32(RecordDescriptor::typeOffset(0), MSG_TYPE_ID);
    m_ab.putInt32(RecordDescriptor::lengthOffset(0), recordLength);

    m_ab.putInt32(RecordDescriptor::typeOffset(0 + alignedRecordLength), MSG_TYPE_ID);
    m_ab.putInt32(RecordDescriptor::lengthOffset(0 + alignedRecordLength), recordLength);

    int timesCalled = 0;
    const int messagesRead = m_ringBuffer.read(
        [&](std::int32_t, concurrent::AtomicBuffer &, util::index_t, util::index_t)
        {
            timesCalled++;
        });

    EXPECT_EQ(messagesRead, 2);
    EXPECT_EQ(timesCalled, 2);
    EXPECT_EQ(m_ab.getInt64(HEAD_COUNTER_INDEX), head + alignedRecordLength + alignedRecordLength);

    for (int i = 0; i < RecordDescriptor::ALIGNMENT * 2; i += 4)
    {
        EXPECT_EQ(m_ab.getInt32(i), 0) << "buffer has not been zeroed between indexes " << i << "-" << i + 3;
    }
}

TEST_F(ManyToOneRingBufferTest, shouldLimitReadOfMessages)
{
    util::index_t length = 8;
    util::index_t head = 0;
    util::index_t recordLength = length + RecordDescriptor::HEADER_LENGTH;
    util::index_t alignedRecordLength = util::BitUtil::align(recordLength, RecordDescriptor::ALIGNMENT);
    util::index_t tail = alignedRecordLength * 2;

    m_ab.putInt64(HEAD_COUNTER_INDEX, head);
    m_ab.putInt64(TAIL_COUNTER_INDEX, tail);

    m_ab.putInt32(RecordDescriptor::typeOffset(0), MSG_TYPE_ID);
    m_ab.putInt32(RecordDescriptor::lengthOffset(0), recordLength);

    m_ab.putInt32(RecordDescriptor::typeOffset(0 + alignedRecordLength), MSG_TYPE_ID);
    m_ab.putInt32(RecordDescriptor::lengthOffset(0 + alignedRecordLength), recordLength);

    int timesCalled = 0;
    const int messagesRead = m_ringBuffer.read(
        [&](std::int32_t, concurrent::AtomicBuffer &, util::index_t, util::index_t)
        {
            timesCalled++;
        },
        1);

    EXPECT_EQ(messagesRead, 1);
    EXPECT_EQ(timesCalled, 1);
    EXPECT_EQ(m_ab.getInt64(HEAD_COUNTER_INDEX), head + alignedRecordLength);

    for (int i = 0; i < RecordDescriptor::ALIGNMENT; i += 4)
    {
        EXPECT_EQ(m_ab.getInt32(i), 0) << "buffer has not been zeroed between indexes " << i << "-" << i + 3;
    }
    EXPECT_EQ(m_ab.getInt32(RecordDescriptor::lengthOffset(alignedRecordLength)), recordLength);
}

TEST_F(ManyToOneRingBufferTest, shouldCopeWithExceptionFromHandler)
{
    util::index_t length = 8;
    util::index_t head = 0;
    util::index_t recordLength = length + RecordDescriptor::HEADER_LENGTH;
    util::index_t alignedRecordLength = util::BitUtil::align(recordLength, RecordDescriptor::ALIGNMENT);
    util::index_t tail = alignedRecordLength * 2;

    m_ab.putInt64(HEAD_COUNTER_INDEX, head);
    m_ab.putInt64(TAIL_COUNTER_INDEX, tail);

    m_ab.putInt32(RecordDescriptor::typeOffset(0), MSG_TYPE_ID);
    m_ab.putInt32(RecordDescriptor::lengthOffset(0), recordLength);

    m_ab.putInt32(RecordDescriptor::typeOffset(0 + alignedRecordLength), MSG_TYPE_ID);
    m_ab.putInt32(RecordDescriptor::lengthOffset(0 + alignedRecordLength), recordLength);

    int timesCalled = 0;
    auto handler =
        [&](std::int32_t, concurrent::AtomicBuffer &, util::index_t, util::index_t)
        {
            timesCalled++;
            if (2 == timesCalled)
            {
                throw std::runtime_error("expected exception");
            }
        };

    bool exceptionReceived = false;

    try
    {
        m_ringBuffer.read(handler);
    }
    catch (const std::runtime_error &)
    {
        exceptionReceived = true;
    }

    EXPECT_EQ(timesCalled, 2);
    EXPECT_TRUE(exceptionReceived);
    EXPECT_EQ(m_ab.getInt64(HEAD_COUNTER_INDEX), head + alignedRecordLength + alignedRecordLength);

    for (int i = 0; i < RecordDescriptor::ALIGNMENT * 2; i += 4)
    {
        EXPECT_EQ(m_ab.getInt32(i), 0) << "buffer has not been zeroed between indexes " << i << "-" << i + 3;
    }
}

TEST_F(ManyToOneRingBufferTest, shouldNotUnblockWhenEmpty)
{
    util::index_t tail = RecordDescriptor::ALIGNMENT * 4;
    util::index_t head = tail;

    m_ab.putInt64(TAIL_COUNTER_INDEX, tail);
    m_ab.putInt64(HEAD_COUNTER_INDEX, head);

    EXPECT_FALSE(m_ringBuffer.unblock());
}

TEST_F(ManyToOneRingBufferTest, shouldUnblockMessageWithHeader)
{
    util::index_t messageLength = RecordDescriptor::ALIGNMENT * 4;
    util::index_t head = messageLength;
    util::index_t tail = messageLength * 2;

    m_ab.putInt64(HEAD_COUNTER_INDEX, head);
    m_ab.putInt64(TAIL_COUNTER_INDEX, tail);

    m_ab.putInt32(RecordDescriptor::typeOffset(head), MSG_TYPE_ID);
    m_ab.putInt32(RecordDescriptor::lengthOffset(head), -messageLength);

    EXPECT_TRUE(m_ringBuffer.unblock());

    EXPECT_EQ(m_ab.getInt32(RecordDescriptor::typeOffset(head)), RecordDescriptor::PADDING_MSG_TYPE_ID);
    EXPECT_EQ(m_ab.getInt32(RecordDescriptor::lengthOffset(head)), messageLength);

    EXPECT_EQ(m_ab.getInt64(HEAD_COUNTER_INDEX), messageLength);
    EXPECT_EQ(m_ab.getInt64(TAIL_COUNTER_INDEX), messageLength * 2);
}

TEST_F(ManyToOneRingBufferTest, shouldUnblockGapWithZeros)
{
    util::index_t messageLength = RecordDescriptor::ALIGNMENT * 4;
    util::index_t head = messageLength;
    util::index_t tail = messageLength * 3;

    m_ab.putInt64(HEAD_COUNTER_INDEX, head);
    m_ab.putInt64(TAIL_COUNTER_INDEX, tail);

    m_ab.putInt32(RecordDescriptor::lengthOffset(messageLength * 2), messageLength);

    EXPECT_TRUE(m_ringBuffer.unblock());

    EXPECT_EQ(m_ab.getInt32(RecordDescriptor::typeOffset(head)), RecordDescriptor::PADDING_MSG_TYPE_ID);
    EXPECT_EQ(m_ab.getInt32(RecordDescriptor::lengthOffset(head)), messageLength);

    EXPECT_EQ(m_ab.getInt64(HEAD_COUNTER_INDEX), messageLength);
    EXPECT_EQ(m_ab.getInt64(TAIL_COUNTER_INDEX), messageLength * 3);
}

TEST_F(ManyToOneRingBufferTest, shouldNotUnblockGapWithMessageRaceOnSecondMessageIncreasingTailThenInterrupting)
{
    util::index_t messageLength = RecordDescriptor::ALIGNMENT * 4;
    util::index_t head = messageLength;
    util::index_t tail = messageLength * 3;

    EXPECT_CALL(m_mockAb, getInt64Volatile(HEAD_COUNTER_INDEX))
        .Times(1)
        .WillRepeatedly(testing::Return(head));

    EXPECT_CALL(m_mockAb, getInt64Volatile(TAIL_COUNTER_INDEX))
        .Times(1)
        .WillRepeatedly(testing::Return(tail));

    EXPECT_CALL(m_mockAb, getInt32Volatile(testing::_))
        .WillRepeatedly(testing::Return(0));

    EXPECT_CALL(m_mockAb, getInt32Volatile(messageLength * 2))
        .Times(1)
        .WillOnce(testing::Return(0));

    EXPECT_FALSE(m_mockRingBuffer.unblock());
}

TEST_F(ManyToOneRingBufferTest, shouldNotUnblockGapWithMessageRaceWhenScanForwardTakesAnInterrupt)
{
    util::index_t messageLength = RecordDescriptor::ALIGNMENT * 4;
    util::index_t head = messageLength;
    util::index_t tail = messageLength * 3;

    EXPECT_CALL(m_mockAb, getInt64Volatile(HEAD_COUNTER_INDEX))
        .Times(1)
        .WillRepeatedly(testing::Return(head));

    EXPECT_CALL(m_mockAb, getInt64Volatile(TAIL_COUNTER_INDEX))
        .Times(1)
        .WillRepeatedly(testing::Return(tail));

    EXPECT_CALL(m_mockAb, getInt32Volatile(testing::_))
        .WillRepeatedly(testing::Return(0));

    EXPECT_CALL(m_mockAb, getInt32Volatile(messageLength * 2))
        .Times(2)
        .WillOnce(testing::Return(0))
        .WillOnce(testing::Return(messageLength));

    EXPECT_CALL(m_mockAb, getInt32Volatile(messageLength * 2 + RecordDescriptor::ALIGNMENT))
        .Times(1)
        .WillOnce(testing::Return(7));

    EXPECT_FALSE(m_mockRingBuffer.unblock());
}

#define NUM_MESSAGES_PER_PUBLISHER (10 * 1000 * 1000)
#define NUM_IDS_PER_THREAD (10 * 1000 * 1000)
#define NUM_PUBLISHERS (2)

TEST(ManyToOneRingBufferConcurrentTest, shouldProvideCorrelationIds)
{
    AERON_DECL_ALIGNED(buffer_t mpscBuffer, 16);
    mpscBuffer.fill(0);
    AtomicBuffer mpscAb(&mpscBuffer[0], mpscBuffer.size());
    ManyToOneRingBuffer ringBuffer(mpscAb);

    std::atomic<int> countDown(NUM_PUBLISHERS);

    std::vector<std::thread> threads;

    for (int i = 0; i < NUM_PUBLISHERS; i++)
    {
        threads.push_back(std::thread(
            [&]()
            {
                countDown--;
                while (countDown > 0)
                {
                    std::this_thread::yield(); // spin until we is ready
                }

                for (int m = 0; m < NUM_IDS_PER_THREAD; m++)
                {
                    ringBuffer.nextCorrelationId();
                }
            }));
    }

    for (std::thread &t: threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }

    ASSERT_EQ(ringBuffer.nextCorrelationId(), NUM_IDS_PER_THREAD * NUM_PUBLISHERS);
}

TEST(ManyToOneRingBufferConcurrentTest, shouldExchangeMessages)
{
    AERON_DECL_ALIGNED(buffer_t mpscBuffer, 16);
    mpscBuffer.fill(0);
    AtomicBuffer mpscAb(&mpscBuffer[0], mpscBuffer.size());
    ManyToOneRingBuffer ringBuffer(mpscAb);

    std::atomic<int> countDown(NUM_PUBLISHERS);
    std::atomic<int> publisherId(0);

    std::vector<std::thread> threads;

    for (int i = 0; i < NUM_PUBLISHERS; i++)
    {
        threads.push_back(std::thread(
            [&]()
            {
                AERON_DECL_ALIGNED(buffer_t srcBuffer, 16);
                srcBuffer.fill(0);
                AtomicBuffer srcAb(&srcBuffer[0], srcBuffer.size());
                int id = publisherId.fetch_add(1);

                countDown--;
                while (countDown > 0)
                {
                    std::this_thread::yield(); // spin until we is ready
                }

                const int messageLength = 4 + 4;
                const int messageNumOffset = 4;

                srcAb.putInt32(0, id);

                for (int m = 0; m < NUM_MESSAGES_PER_PUBLISHER; m++)
                {
                    srcAb.putInt32(messageNumOffset, m);
                    while (!ringBuffer.write(MSG_TYPE_ID, srcAb, 0, messageLength))
                    {
                        std::this_thread::yield();
                    }
                }
            }));
    }

    try
    {
        int msgCount = 0;
        int counts[NUM_PUBLISHERS];

        for (int &count : counts)
        {
            count = 0;
        }

        while (msgCount < (NUM_MESSAGES_PER_PUBLISHER * NUM_PUBLISHERS))
        {
            const int readCount = ringBuffer.read(
                [&counts](
                    std::int32_t msgTypeId,
                    concurrent::AtomicBuffer &buffer,
                    util::index_t index,
                    util::index_t length)
                {
                    const std::int32_t id = buffer.getInt32(index);
                    const std::int32_t messageNumber = buffer.getInt32(index + 4);

                    EXPECT_EQ(length, 4 + 4);
                    ASSERT_EQ(msgTypeId, MSG_TYPE_ID);

                    EXPECT_EQ(counts[id], messageNumber);
                    counts[id]++;
                });

            if (0 == readCount)
            {
                std::this_thread::yield();
            }

            msgCount += readCount;
        }

        for (std::thread &t: threads)
        {
            if (t.joinable())
            {
                t.join();
            }
        }
    }
    catch (const util::OutOfBoundsException &e)
    {
        printf("EXCEPTION %s at %s\n", e.what(), e.where());
    }
    catch (...)
    {
        printf("EXCEPTION unknown\n");
    }
}
