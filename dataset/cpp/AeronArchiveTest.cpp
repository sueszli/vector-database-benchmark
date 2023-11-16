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

#include <chrono>
#include <thread>
#include <iostream>
#include <iosfwd>
#include <vector>
#include <cstring>

#include <gtest/gtest.h>

#include "client/AeronArchive.h"
#include "client/RecordingPos.h"
#include "client/ReplayMerge.h"
#include "concurrent/YieldingIdleStrategy.h"
#include "concurrent/SleepingIdleStrategy.h"
#include "ChannelUriStringBuilder.h"
#include "CncFileReader.h"
#include "aeron_archive_client/RecordingSignal.h"

#if defined(__linux__) || defined(Darwin)
#include <unistd.h>
#include <ftw.h>
#include <cstdio>
#include <spawn.h>
#include <pthread.h>
#elif defined(_WIN32)
#include <windows.h>
typedef intptr_t pid_t;
#else
#error "must spawn Java Archive per test"
#endif

#include "TestArchive.h"

using namespace aeron;
using namespace aeron::util;
using namespace aeron::concurrent;
using namespace aeron::archive::client;

class AeronArchiveTestBase
{
public:
    ~AeronArchiveTestBase()
    {
        if (m_debug)
        {
            std::cout << m_stream.str();
        }
    }

    void DoSetUp()
    {
        std::string sourceArchiveDir = m_archiveDir + AERON_FILE_SEP + "source";
        m_archive = std::make_shared<TestArchive>(m_context.aeronDirectoryName(), sourceArchiveDir, m_stream);

        setCredentials(m_context);
    }

    void DoTearDown()
    {
    }

    static std::shared_ptr<Publication> addPublication(Aeron &aeron, const std::string &channel, std::int32_t streamId)
    {
        std::int64_t publicationId = aeron.addPublication(channel, streamId);
        std::shared_ptr<Publication> publication = aeron.findPublication(publicationId);
        YieldingIdleStrategy idleStrategy;
        while (!publication)
        {
            idleStrategy.idle();
            publication = aeron.findPublication(publicationId);
        }

        return publication;
    }

    static std::shared_ptr<Subscription> addSubscription(
        Aeron &aeron, const std::string &channel, std::int32_t streamId)
    {
        std::int64_t subscriptionId = aeron.addSubscription(channel, streamId);
        std::shared_ptr<Subscription> subscription = aeron.findSubscription(subscriptionId);
        YieldingIdleStrategy idleStrategy;
        while (!subscription)
        {
            idleStrategy.idle();
            subscription = aeron.findSubscription(subscriptionId);
        }

        return subscription;
    }

    static std::int32_t getRecordingCounterId(std::int32_t sessionId, CountersReader &countersReader)
    {
        std::int32_t counterId;
        while (CountersReader::NULL_COUNTER_ID ==
            (counterId = RecordingPos::findCounterIdBySessionId(countersReader, sessionId)))
        {
            std::this_thread::yield();
        }

        return counterId;
    }

    static void offerMessages(Publication &publication, std::size_t messageCount, const std::string &messagePrefix)
    {
        BufferClaim bufferClaim;
        YieldingIdleStrategy idleStrategy;

        for (std::size_t i = 0; i < messageCount; i++)
        {
            const std::string message = messagePrefix + std::to_string(i);
            while (publication.tryClaim(static_cast<util::index_t>(message.length()), bufferClaim) < 0)
            {
                idleStrategy.idle();
            }

            bufferClaim.buffer().putStringWithoutLength(bufferClaim.offset(), message);
            bufferClaim.commit();
        }
    }

    void consumeMessages(Subscription &subscription, std::size_t messageCount, const std::string &messagePrefix) const
    {
        std::size_t received = 0;
        YieldingIdleStrategy idleStrategy;

        fragment_handler_t handler =
            [&](AtomicBuffer &buffer, util::index_t offset, util::index_t length, Header &header)
            {
                const std::string expected = messagePrefix + std::to_string(received);
                const std::string actual = buffer.getStringWithoutLength(offset, static_cast<std::size_t>(length));

                EXPECT_EQ(expected, actual);

                received++;
            };

        while (received < messageCount)
        {
            if (0 == subscription.poll(handler, m_fragmentLimit))
            {
                idleStrategy.idle();
            }
        }

        ASSERT_EQ(received, messageCount);
    }

    std::int64_t consumeMessagesExpectingBound(
        Subscription &subscription,
        std::int64_t boundPosition,
        const std::string &messagePrefix,
        std::int64_t timeoutMs) const
    {
        std::size_t received = 0;
        std::int64_t position = 0;
        YieldingIdleStrategy idleStrategy;

        fragment_handler_t handler =
            [&](AtomicBuffer &buffer, util::index_t offset, util::index_t length, Header &header)
            {
                const std::string expected = messagePrefix + std::to_string(received);
                const std::string actual = buffer.getStringWithoutLength(offset, static_cast<std::size_t>(length));
                EXPECT_EQ(expected, actual);

                received++;
                position = header.position();
            };

        const long long deadlineMs = currentTimeMillis() + timeoutMs;

        while (currentTimeMillis() < deadlineMs)
        {
            if (0 == subscription.poll(handler, m_fragmentLimit))
            {
                idleStrategy.idle();
            }
        }

        return position;
    }

    bool attemptReplayMerge(
        ReplayMerge &replayMerge,
        Publication &publication,
        fragment_handler_t &handler,
        const std::string &messagePrefix,
        std::size_t totalMessageCount,
        std::size_t &messagesPublished,
        std::size_t &receivedMessageCount) const
    {
        YieldingIdleStrategy idleStrategy;

        for (std::size_t i = messagesPublished; i < totalMessageCount; i++)
        {
            BufferClaim bufferClaim;
            const std::string message = messagePrefix + std::to_string(i);

            idleStrategy.reset();
            while (publication.tryClaim(static_cast<util::index_t>(message.length()), bufferClaim) < 0)
            {
                idleStrategy.idle();
                int fragments = replayMerge.poll(handler, m_fragmentLimit);
                if (0 == fragments && replayMerge.hasFailed())
                {
                    return false;
                }
            }

            bufferClaim.buffer().putStringWithoutLength(bufferClaim.offset(), message);
            bufferClaim.commit();
            ++messagesPublished;

            int fragments = replayMerge.poll(handler, m_fragmentLimit);
            if (0 == fragments && replayMerge.hasFailed())
            {
                return false;
            }
        }

        while (!replayMerge.isMerged())
        {
            int fragments = replayMerge.poll(handler, m_fragmentLimit);
            if (0 == fragments && replayMerge.hasFailed())
            {
                return false;
            }
            idleStrategy.idle(fragments);
        }

        Image &image = *replayMerge.image();
        while (receivedMessageCount < totalMessageCount)
        {
            int fragments = image.poll(handler, m_fragmentLimit);
            if (0 == fragments && image.isClosed())
            {
                return false;
            }
            idleStrategy.idle(fragments);
        }

        return true;
    }

    void startDestArchive()
    {
        const std::string aeronDir = aeron::Context::defaultAeronPath() + "_dest";
        const std::string archiveDir = m_archiveDir + AERON_FILE_SEP + "dest";
        const std::string controlChannel = "aeron:udp?endpoint=localhost:8011";
        const std::string replicationChannel = "aeron:udp?endpoint=localhost:8012";
        m_destArchive = std::make_shared<TestArchive>(
            aeronDir, archiveDir, m_stream, controlChannel, replicationChannel);
        m_destContext.controlRequestChannel(controlChannel);
        setCredentials(m_destContext);
    }

protected:
    const std::string m_java = JAVA_EXECUTABLE;
    const std::string m_aeronAllJar = AERON_ALL_JAR;
    const std::string m_archiveDir = ARCHIVE_DIR;

    const std::string m_recordingChannel = "aeron:udp?endpoint=localhost:3333";
    const std::int32_t m_recordingStreamId = 33;
    const std::string m_replayChannel = "aeron:udp?endpoint=localhost:6666";
    const std::int32_t m_replayStreamId = 66;

    const int m_fragmentLimit = 10;
    std::ostringstream m_stream;

    std::shared_ptr<TestArchive> m_destArchive;
    std::shared_ptr<TestArchive> m_archive;
    AeronArchive::Context_t m_destContext;
    AeronArchive::Context_t m_context;

    pid_t m_pid = -1;
    bool m_debug = true;

private:

    static void setCredentials(AeronArchive::Context_t &context)
    {
        auto onEncodedCredentials =
            []() -> std::pair<const char *, std::uint32_t>
            {
                std::string credentials("admin:admin");

                char *arr = new char[credentials.length() + 1];
                std::memcpy(arr, credentials.data(), credentials.length());
                arr[credentials.length()] = '\0';

                return { arr, static_cast<std::uint32_t>(credentials.length()) };
            };

        context.credentialsSupplier(CredentialsSupplier(onEncodedCredentials));
    }
};

class AeronArchiveTest : public AeronArchiveTestBase, public testing::Test
{
public:
    void SetUp() final
    {
        DoSetUp();
    }

    void TearDown() final
    {
        DoTearDown();
    }
};

class AeronArchiveParamTest : public AeronArchiveTestBase, public testing::TestWithParam<bool>
{
public:
    void SetUp() final
    {
        DoSetUp();
    }

    void TearDown() final
    {
        DoTearDown();
    }
};

INSTANTIATE_TEST_SUITE_P(AeronArchive, AeronArchiveParamTest, testing::Values(true, false));

TEST_F(AeronArchiveTest, shouldAsyncConnectToArchive)
{
    SleepingIdleStrategy idleStrategy(IDLE_SLEEP_MS_1);
    std::shared_ptr<AeronArchive::AsyncConnect> asyncConnect = AeronArchive::asyncConnect(m_context);
    std::uint8_t previousStep = asyncConnect->step();

    std::shared_ptr<AeronArchive> aeronArchive = asyncConnect->poll();
    while (!aeronArchive)
    {
        if (asyncConnect->step() == previousStep)
        {
            idleStrategy.idle();
        }
        else
        {
            idleStrategy.reset();
            previousStep = asyncConnect->step();
        }

        aeronArchive = asyncConnect->poll();
    }

    EXPECT_TRUE(aeronArchive->controlResponsePoller().subscription()->isConnected());
}

TEST_F(AeronArchiveTest, shouldConnectToArchive)
{
    std::shared_ptr<AeronArchive> aeronArchive = AeronArchive::connect(m_context);

    EXPECT_TRUE(aeronArchive->controlResponsePoller().subscription()->isConnected());
}

TEST_F(AeronArchiveTest, shouldRecordPublicationAndFindRecording)
{
    const std::string messagePrefix = "Message ";
    const std::size_t messageCount = 10;
    std::int32_t sessionId;
    std::int64_t recordingIdFromCounter;
    std::int64_t stopPosition;

    std::shared_ptr<AeronArchive> aeronArchive = AeronArchive::connect(m_context);

    const std::int64_t subscriptionId = aeronArchive->startRecording(
        m_recordingChannel, m_recordingStreamId, AeronArchive::SourceLocation::LOCAL);

    {
        std::shared_ptr<Subscription> subscription = addSubscription(
            *aeronArchive->context().aeron(), m_recordingChannel, m_recordingStreamId);
        std::shared_ptr<Publication> publication = addPublication(
            *aeronArchive->context().aeron(), m_recordingChannel, m_recordingStreamId);

        sessionId = publication->sessionId();

        CountersReader &countersReader = aeronArchive->context().aeron()->countersReader();
        const std::int32_t counterId = getRecordingCounterId(sessionId, countersReader);
        recordingIdFromCounter = RecordingPos::getRecordingId(countersReader, counterId);

        offerMessages(*publication, messageCount, messagePrefix);
        consumeMessages(*subscription, messageCount, messagePrefix);

        stopPosition = publication->position();

        YieldingIdleStrategy idleStrategy;
        while (countersReader.getCounterValue(counterId) < stopPosition)
        {
            idleStrategy.idle();
        }

        EXPECT_EQ(aeronArchive->getRecordingPosition(recordingIdFromCounter), stopPosition);
        EXPECT_EQ(aeronArchive->getStopPosition(recordingIdFromCounter), aeron::NULL_VALUE);
    }

    aeronArchive->stopRecording(subscriptionId);

    const std::int64_t recordingId = aeronArchive->findLastMatchingRecording(
        0, "endpoint=localhost:3333", m_recordingStreamId, sessionId);

    EXPECT_EQ(recordingIdFromCounter, recordingId);
    EXPECT_EQ(aeronArchive->getStopPosition(recordingIdFromCounter), stopPosition);

    const std::int32_t count = aeronArchive->listRecording(
        recordingId,
        [&](
            std::int64_t controlSessionId,
            std::int64_t correlationId,
            std::int64_t recordingId1,
            std::int64_t startTimestamp,
            std::int64_t stopTimestamp,
            std::int64_t startPosition,
            std::int64_t newStopPosition,
            std::int32_t initialTermId,
            std::int32_t segmentFileLength,
            std::int32_t termBufferLength,
            std::int32_t mtuLength,
            std::int32_t sessionId1,
            std::int32_t streamId,
            const std::string &strippedChannel,
            const std::string &originalChannel,
            const std::string &sourceIdentity)
        {
            EXPECT_EQ(recordingId, recordingId1);
            EXPECT_EQ(streamId, m_recordingStreamId);
        });

    EXPECT_EQ(count, 1);
}

TEST_P(AeronArchiveParamTest, shouldRecordThenReplay)
{
    const bool useParams = GetParam();
    const std::string messagePrefix = "Message ";
    const std::size_t messageCount = 10;
    std::int32_t sessionId;
    std::int64_t recordingId;
    std::int64_t stopPosition;

    std::shared_ptr<AeronArchive> aeronArchive = AeronArchive::connect(m_context);

    const std::int64_t subscriptionId = aeronArchive->startRecording(
        m_recordingChannel, m_recordingStreamId, AeronArchive::SourceLocation::LOCAL);

    {
        std::shared_ptr<Subscription> subscription = addSubscription(
            *aeronArchive->context().aeron(), m_recordingChannel, m_recordingStreamId);
        std::shared_ptr<Publication> publication = addPublication(
            *aeronArchive->context().aeron(), m_recordingChannel, m_recordingStreamId);

        sessionId = publication->sessionId();

        CountersReader &countersReader = aeronArchive->context().aeron()->countersReader();
        const std::int32_t counterId = getRecordingCounterId(sessionId, countersReader);
        recordingId = RecordingPos::getRecordingId(countersReader, counterId);
        EXPECT_TRUE(RecordingPos::isActive(countersReader, counterId, recordingId));
        EXPECT_EQ(counterId, RecordingPos::findCounterIdByRecordingId(countersReader, recordingId));
        EXPECT_EQ("aeron:ipc", RecordingPos::getSourceIdentity(countersReader, counterId));

        offerMessages(*publication, messageCount, messagePrefix);
        consumeMessages(*subscription, messageCount, messagePrefix);

        stopPosition = publication->position();

        YieldingIdleStrategy idleStrategy;
        while (countersReader.getCounterValue(counterId) < stopPosition)
        {
            idleStrategy.idle();
        }
    }

    aeronArchive->stopRecording(subscriptionId);

    YieldingIdleStrategy idleStrategy;
    while (aeronArchive->getStopPosition(recordingId) != stopPosition)
    {
        idleStrategy.idle();
    }

    {
        const std::int64_t position = 0L;
        const std::int64_t length = stopPosition - position;
        std::shared_ptr<Subscription> subscription = addSubscription(
            *aeronArchive->context().aeron(), m_replayChannel, m_replayStreamId);

        if (useParams)
        {
            aeronArchive->startReplay(
                recordingId,
                m_replayChannel,
                m_replayStreamId,
                ReplayParams().position(position).length(length).fileIoMaxLength(4096));
        }
        else
        {
            aeronArchive->startReplay(recordingId, position, length, m_replayChannel, m_replayStreamId);
        }

        consumeMessages(*subscription, messageCount, messagePrefix);
        EXPECT_EQ(stopPosition, subscription->imageByIndex(0)->position());
    }
}

TEST_P(AeronArchiveParamTest, shouldRecordThenBoundedReplay)
{
    const bool useParams = GetParam();

    const std::string messagePrefix = "Message ";
    const std::size_t messageCount = 10;
    std::int32_t sessionId;
    std::int64_t recordingIdFromCounter;
    std::int64_t stopPosition;
    YieldingIdleStrategy idleStrategy;

    std::shared_ptr<AeronArchive> aeronArchive = AeronArchive::connect(m_context);

    const std::int64_t subscriptionId = aeronArchive->startRecording(
        m_recordingChannel, m_recordingStreamId, AeronArchive::SourceLocation::LOCAL);

    {
        std::shared_ptr<Subscription> subscription = addSubscription(
            *aeronArchive->context().aeron(), m_recordingChannel, m_recordingStreamId);
        std::shared_ptr<Publication> publication = addPublication(
            *aeronArchive->context().aeron(), m_recordingChannel, m_recordingStreamId);

        sessionId = publication->sessionId();

        CountersReader &countersReader = aeronArchive->context().aeron()->countersReader();
        const std::int32_t counterId = getRecordingCounterId(sessionId, countersReader);
        recordingIdFromCounter = RecordingPos::getRecordingId(countersReader, counterId);

        offerMessages(*publication, messageCount, messagePrefix);
        consumeMessages(*subscription, messageCount, messagePrefix);

        stopPosition = publication->position();

        while (countersReader.getCounterValue(counterId) < stopPosition)
        {
            idleStrategy.idle();
        }
    }

    aeronArchive->stopRecording(subscriptionId);
    std::string counterName = "BoundedTestCounter";
    int64_t counterId = aeronArchive->context().aeron()->addCounter(
        10001, reinterpret_cast<const uint8_t *>(counterName.c_str()), counterName.length(), counterName);
    std::shared_ptr<Counter> counter;
    while (nullptr == (counter = aeronArchive->context().aeron()->findCounter(counterId)))
    {
        idleStrategy.idle();
    }

    while (aeronArchive->getStopPosition(recordingIdFromCounter) != stopPosition)
    {
        idleStrategy.idle();
    }

    {
        const std::int64_t position = 0L;
        const std::int64_t length = stopPosition - position;
        const std::int64_t boundedLength = (length / 4) * 3;
        counter->set(boundedLength);

        std::shared_ptr<Subscription> subscription = addSubscription(
            *aeronArchive->context().aeron(), m_replayChannel, m_replayStreamId);

        if (useParams)
        {
            aeronArchive->startReplay(
                recordingIdFromCounter,
                m_replayChannel,
                m_replayStreamId,
                ReplayParams()
                    .position(position)
                    .length(length)
                    .boundingLimitCounterId(counter->id())
                    .fileIoMaxLength(4096));
        }
        else
        {
            aeronArchive->startBoundedReplay(
                recordingIdFromCounter, position, length, counter->id(), m_replayChannel, m_replayStreamId);
        }

        const std::int64_t positionConsumed = consumeMessagesExpectingBound(
            *subscription, position + boundedLength, messagePrefix, 1000);

        EXPECT_LT(position + (length / 2), positionConsumed);
        EXPECT_LE(positionConsumed, position + boundedLength);
    }
}


TEST_P(AeronArchiveParamTest, shouldRecordThenReplayThenTruncate)
{
    const bool useParams = GetParam();
    const std::string messagePrefix = "Message ";
    const std::size_t messageCount = 10;
    std::int32_t sessionId;
    std::int64_t recordingIdFromCounter;
    std::int64_t stopPosition;

    std::shared_ptr<AeronArchive> aeronArchive = AeronArchive::connect(m_context);

    const std::int64_t subscriptionId = aeronArchive->startRecording(
        m_recordingChannel, m_recordingStreamId, AeronArchive::SourceLocation::LOCAL);

    {
        std::shared_ptr<Subscription> subscription = addSubscription(
            *aeronArchive->context().aeron(), m_recordingChannel, m_recordingStreamId);
        std::shared_ptr<Publication> publication = addPublication(
            *aeronArchive->context().aeron(), m_recordingChannel, m_recordingStreamId);

        sessionId = publication->sessionId();

        CountersReader &countersReader = aeronArchive->context().aeron()->countersReader();
        const std::int32_t counterId = getRecordingCounterId(sessionId, countersReader);
        recordingIdFromCounter = RecordingPos::getRecordingId(countersReader, counterId);

        offerMessages(*publication, messageCount, messagePrefix);
        consumeMessages(*subscription, messageCount, messagePrefix);

        stopPosition = publication->position();

        YieldingIdleStrategy idleStrategy;
        while (countersReader.getCounterValue(counterId) < stopPosition)
        {
            idleStrategy.idle();
        }

        EXPECT_EQ(aeronArchive->getRecordingPosition(recordingIdFromCounter), stopPosition);
        EXPECT_EQ(aeronArchive->getStopPosition(recordingIdFromCounter), aeron::NULL_VALUE);
    }

    aeronArchive->stopRecording(subscriptionId);

    const std::int64_t recordingId = aeronArchive->findLastMatchingRecording(
        0, "endpoint=localhost:3333", m_recordingStreamId, sessionId);

    EXPECT_EQ(recordingIdFromCounter, recordingId);
    EXPECT_EQ(aeronArchive->getStopPosition(recordingIdFromCounter), stopPosition);

    const std::int64_t position = 0L;

    {
        const std::int64_t length = stopPosition - position;
        std::shared_ptr<Subscription> subscription;
        if (useParams)
        {
            subscription = aeronArchive->replay(
                recordingId,
                m_replayChannel,
                m_replayStreamId,
                ReplayParams().position(position).length(length).fileIoMaxLength(4096));
        }
        else
        {
            subscription = aeronArchive->replay(
                recordingId, position, length, m_replayChannel, m_replayStreamId);
        }

        consumeMessages(*subscription, messageCount, messagePrefix);
        EXPECT_EQ(stopPosition, subscription->imageByIndex(0)->position());
    }

    aeronArchive->truncateRecording(recordingId, position);

    const std::int32_t count = aeronArchive->listRecording(
        recordingId,
        [](
            std::int64_t controlSessionId,
            std::int64_t correlationId,
            std::int64_t recordingId1,
            std::int64_t startTimestamp,
            std::int64_t stopTimestamp,
            std::int64_t startPosition,
            std::int64_t newStopPosition,
            std::int32_t initialTermId,
            std::int32_t segmentFileLength,
            std::int32_t termBufferLength,
            std::int32_t mtuLength,
            std::int32_t sessionId1,
            std::int32_t streamId,
            const std::string &strippedChannel,
            const std::string &originalChannel,
            const std::string &sourceIdentity)
        {
            EXPECT_EQ(startPosition, newStopPosition);
        });

    EXPECT_EQ(count, 1);
}

TEST_P(AeronArchiveParamTest, shouldRecordAndCancelReplayEarly)
{
    const bool useParams = GetParam();
    const std::string messagePrefix = "Message ";
    const std::size_t messageCount = 10;
    std::int64_t recordingId;
    std::int64_t stopPosition;

    std::shared_ptr<AeronArchive> aeronArchive = AeronArchive::connect(m_context);

    {
        std::shared_ptr<Subscription> subscription = addSubscription(
            *aeronArchive->context().aeron(), m_recordingChannel, m_recordingStreamId);
        std::shared_ptr<Publication> publication = aeronArchive->addRecordedPublication(
            m_recordingChannel, m_recordingStreamId);

        CountersReader &countersReader = aeronArchive->context().aeron()->countersReader();
        const std::int32_t counterId = getRecordingCounterId(publication->sessionId(), countersReader);
        recordingId = RecordingPos::getRecordingId(countersReader, counterId);

        offerMessages(*publication, messageCount, messagePrefix);
        consumeMessages(*subscription, messageCount, messagePrefix);

        stopPosition = publication->position();

        YieldingIdleStrategy idleStrategy;
        while (countersReader.getCounterValue(counterId) < stopPosition)
        {
            idleStrategy.idle();
        }

        EXPECT_EQ(aeronArchive->getRecordingPosition(recordingId), stopPosition);

        aeronArchive->stopRecording(publication);

        idleStrategy.reset();
        while (NULL_POSITION != aeronArchive->getRecordingPosition(recordingId))
        {
            idleStrategy.idle();
        }
    }

    const std::int64_t position = 0L;
    const std::int64_t length = stopPosition - position;

    std::int64_t replaySessionId;
    if (useParams)
    {
        replaySessionId = aeronArchive->startReplay(
            recordingId,
            m_replayChannel,
            m_replayStreamId,
            ReplayParams().position(position).length(length).fileIoMaxLength(4096));
    }
    else
    {
        replaySessionId = aeronArchive->startReplay(recordingId, position, length, m_replayChannel, m_replayStreamId);
    }

    aeronArchive->stopReplay(replaySessionId);
}

TEST_P(AeronArchiveParamTest, shouldReplayRecordingFromLateJoinPosition)
{
    const bool useParams = GetParam();
    const std::string messagePrefix = "Message ";
    const std::size_t messageCount = 10;

    std::shared_ptr<AeronArchive> aeronArchive = AeronArchive::connect(m_context);

    aeronArchive->startRecording(
        m_recordingChannel, m_recordingStreamId, AeronArchive::SourceLocation::LOCAL, true);

    {
        std::shared_ptr<Subscription> subscription = addSubscription(
            *aeronArchive->context().aeron(), m_recordingChannel, m_recordingStreamId);
        std::shared_ptr<Publication> publication = addPublication(
            *aeronArchive->context().aeron(), m_recordingChannel, m_recordingStreamId);

        CountersReader &countersReader = aeronArchive->context().aeron()->countersReader();
        const std::int32_t counterId = getRecordingCounterId(publication->sessionId(), countersReader);
        const std::int64_t recordingId = RecordingPos::getRecordingId(countersReader, counterId);

        offerMessages(*publication, messageCount, messagePrefix);
        consumeMessages(*subscription, messageCount, messagePrefix);

        const std::int64_t currentPosition = publication->position();

        YieldingIdleStrategy idleStrategy;
        while (countersReader.getCounterValue(counterId) < currentPosition)
        {
            idleStrategy.idle();
        }

        {
            std::shared_ptr<Subscription> replaySubscription;

            if (useParams)
            {
                replaySubscription = aeronArchive->replay(
                    recordingId,
                    m_replayChannel,
                    m_replayStreamId,
                    ReplayParams().position(currentPosition).fileIoMaxLength(4096));
            }
            else
            {
                replaySubscription = aeronArchive->replay(
                    recordingId, currentPosition, NULL_LENGTH, m_replayChannel, m_replayStreamId);
            }
            offerMessages(*publication, messageCount, messagePrefix);
            consumeMessages(*subscription, messageCount, messagePrefix);
            consumeMessages(*replaySubscription, messageCount, messagePrefix);

            const std::int64_t endPosition = publication->position();
            EXPECT_EQ(endPosition, replaySubscription->imageByIndex(0)->position());
        }
    }
}

struct SubscriptionDescriptor
{
    const std::int64_t m_controlSessionId;
    const std::int64_t m_correlationId;
    const std::int64_t m_subscriptionId;
    const std::int32_t m_streamId;

    SubscriptionDescriptor(
        std::int64_t controlSessionId,
        std::int64_t correlationId,
        std::int64_t subscriptionId,
        std::int32_t streamId) :
        m_controlSessionId(controlSessionId),
        m_correlationId(correlationId),
        m_subscriptionId(subscriptionId),
        m_streamId(streamId)
    {
    }
};

TEST_F(AeronArchiveTest, shouldListRegisteredRecordingSubscriptions)
{
    std::vector<SubscriptionDescriptor> descriptors;
    recording_subscription_descriptor_consumer_t consumer =
        [&descriptors](
            std::int64_t controlSessionId,
            std::int64_t correlationId,
            std::int64_t subscriptionId,
            std::int32_t streamId,
            const std::string &strippedChannel)
        {
            descriptors.emplace_back(controlSessionId, correlationId, subscriptionId, streamId);
        };

    const std::int32_t expectedStreamId = 7;
    const std::string channelOne = "aeron:ipc";
    const std::string channelTwo = "aeron:udp?endpoint=localhost:5678";
    const std::string channelThree = "aeron:udp?endpoint=localhost:4321";

    std::shared_ptr<AeronArchive> aeronArchive = AeronArchive::connect(m_context);

    const std::int64_t subIdOne = aeronArchive->startRecording(
        channelOne, expectedStreamId, AeronArchive::SourceLocation::LOCAL);
    const std::int64_t subIdTwo = aeronArchive->startRecording(
        channelTwo, expectedStreamId + 1, AeronArchive::SourceLocation::LOCAL);
    const std::int64_t subIdThree = aeronArchive->startRecording(
        channelThree, expectedStreamId + 2, AeronArchive::SourceLocation::LOCAL);

    const std::int32_t countOne = aeronArchive->listRecordingSubscriptions(
        0, 5, "ipc", expectedStreamId, true, consumer);

    EXPECT_EQ(1uL, descriptors.size());
    EXPECT_EQ(1L, countOne);

    descriptors.clear();

    const std::int32_t countTwo = aeronArchive->listRecordingSubscriptions(
        0, 5, "", expectedStreamId, false, consumer);

    EXPECT_EQ(3uL, descriptors.size());
    EXPECT_EQ(3L, countTwo);

    aeronArchive->stopRecording(subIdTwo);
    descriptors.clear();

    const std::int32_t countThree = aeronArchive->listRecordingSubscriptions(
        0, 5, "", expectedStreamId, false, consumer);

    EXPECT_EQ(2uL, descriptors.size());
    EXPECT_EQ(2L, countThree);

    EXPECT_EQ(1L, std::count_if(
        descriptors.begin(),
        descriptors.end(),
        [=](const SubscriptionDescriptor &descriptor){ return descriptor.m_subscriptionId == subIdOne; }));

    EXPECT_EQ(1L, std::count_if(
        descriptors.begin(),
        descriptors.end(),
        [=](const SubscriptionDescriptor &descriptor){ return descriptor.m_subscriptionId == subIdThree; }));
}

TEST_F(AeronArchiveTest, shouldMergeFromReplayToLive)
{
    const std::size_t termLength = 64 * 1024;
    const std::string messagePrefix = "Message ";
    const std::size_t minMessagesPerTerm = termLength / (messagePrefix.length() + DataFrameHeader::LENGTH);
    const std::string controlEndpoint = "localhost:23265";
    const std::string recordingEndpoint = "localhost:23266";
    const std::string liveEndpoint = "localhost:23267";
    const std::string replayEndpoint = "localhost:0";

    const std::string publicationChannel = ChannelUriStringBuilder()
        .media(UDP_MEDIA)
        .controlEndpoint(controlEndpoint)
        .controlMode(MDC_CONTROL_MODE_DYNAMIC)
        .flowControl("tagged,g:99901/1,t:5s")
        .termLength(termLength)
        .build();

    const std::string liveDestination = ChannelUriStringBuilder()
        .media(UDP_MEDIA)
        .endpoint(liveEndpoint)
        .controlEndpoint(controlEndpoint)
        .build();

    const std::string replayDestination = ChannelUriStringBuilder()
        .media(UDP_MEDIA)
        .endpoint(replayEndpoint)
        .build();

    const std::size_t initialMessageCount = minMessagesPerTerm * 3;
    const std::size_t subsequentMessageCount = minMessagesPerTerm * 3;
    const std::size_t totalMessageCount = initialMessageCount + subsequentMessageCount;
    YieldingIdleStrategy idleStrategy;

    std::shared_ptr<AeronArchive> aeronArchive = AeronArchive::connect(m_context);
    std::shared_ptr<Publication> publication = addPublication(
        *aeronArchive->context().aeron(), publicationChannel, m_recordingStreamId);

    const std::int32_t sessionId = publication->sessionId();

    const std::string recordingChannel = ChannelUriStringBuilder()
        .media(UDP_MEDIA)
        .groupTag(99901)
        .sessionId(sessionId)
        .endpoint(recordingEndpoint)
        .controlEndpoint(controlEndpoint)
        .build();

    const std::string subscriptionChannel = ChannelUriStringBuilder()
        .media(UDP_MEDIA)
        .controlMode(MDC_CONTROL_MODE_MANUAL)
        .sessionId(sessionId)
        .build();

    aeronArchive->startRecording(
        recordingChannel, m_recordingStreamId, AeronArchive::SourceLocation::REMOTE, true);

    CountersReader &countersReader = aeronArchive->context().aeron()->countersReader();
    const std::int32_t counterId = getRecordingCounterId(sessionId, countersReader);
    const std::int64_t recordingId = RecordingPos::getRecordingId(countersReader, counterId);
    EXPECT_TRUE(RecordingPos::isActive(countersReader, counterId, recordingId));
    EXPECT_EQ(counterId, RecordingPos::findCounterIdByRecordingId(countersReader, recordingId));
    EXPECT_EQ("127.0.0.1:23265", RecordingPos::getSourceIdentity(countersReader, counterId));

    offerMessages(*publication, initialMessageCount, messagePrefix);
    while (countersReader.getCounterValue(counterId) < publication->position())
    {
        idleStrategy.idle();
    }

    std::size_t messagesPublished = initialMessageCount;
    std::size_t receivedMessageCount = 0;
    std::int64_t receivedPosition = 0;

    fragment_handler_t fragment_handler =
        [&](AtomicBuffer &buffer, util::index_t offset, util::index_t length, Header &header)
        {
            const std::string expected = messagePrefix + std::to_string(receivedMessageCount);
            const std::string actual = buffer.getStringWithoutLength(offset, static_cast<std::size_t>(length));

            EXPECT_EQ(expected, actual);

            receivedMessageCount++;
            receivedPosition = header.position();
        };

    while (true)
    {
        std::shared_ptr<Subscription> subscription = addSubscription(
            *aeronArchive->context().aeron(), subscriptionChannel, m_recordingStreamId);

        const std::string replayChannel = ChannelUriStringBuilder()
            .media(UDP_MEDIA)
            .sessionId(publication->sessionId())
            .build();

        ReplayMerge replayMerge(
            subscription,
            aeronArchive,
            replayChannel,
            replayDestination,
            liveDestination,
            recordingId,
            receivedPosition);

        if (attemptReplayMerge(
            replayMerge,
            *publication,
            fragment_handler,
            messagePrefix,
            totalMessageCount,
            messagesPublished,
            receivedMessageCount))
        {
            break;
        }

        idleStrategy.reset();
        idleStrategy.idle();
    }

    EXPECT_EQ(receivedMessageCount, totalMessageCount);
    EXPECT_EQ(receivedPosition, publication->position());
}

TEST_F(AeronArchiveTest, shouldExceptionForIncorrectInitialCredentials)
{
    auto onEncodedCredentials =
        []() -> std::pair<const char *, std::uint32_t>
        {
            std::string credentials("admin:NotAdmin");

            char *arr = new char[credentials.length() + 1];
            std::memcpy(arr, credentials.data(), credentials.length());
            arr[credentials.length()] = '\0';

            return { arr, static_cast<std::uint32_t>(credentials.length()) };
        };

    m_context.credentialsSupplier(CredentialsSupplier(onEncodedCredentials));

    ASSERT_THROW(
        {
            std::shared_ptr<AeronArchive> aeronArchive = AeronArchive::connect(m_context);
        },
        ArchiveException);
}

TEST_F(AeronArchiveTest, shouldBeAbleToHandleBeingChallenged)
{
    auto onEncodedCredentials =
        []() -> std::pair<const char *, std::uint32_t>
        {
            std::string credentials("admin:adminC");

            char *arr = new char[credentials.length() + 1];
            std::memcpy(arr, credentials.data(), credentials.length());
            arr[credentials.length()] = '\0';

            return { arr, static_cast<std::uint32_t>(credentials.length()) };
        };

    auto onChallenge =
        [](std::pair<const char *, std::uint32_t> encodedChallenge) -> std::pair<const char *, std::uint32_t>
        {
            std::string credentials("admin:CSadmin");

            char *arr = new char[credentials.length() + 1];
            std::memcpy(arr, credentials.data(), credentials.length());
            arr[credentials.length()] = '\0';

            return { arr, static_cast<std::uint32_t>(credentials.length()) };
        };

    m_context.credentialsSupplier(CredentialsSupplier(onEncodedCredentials, onChallenge));

    ASSERT_NO_THROW(
        {
            std::shared_ptr<AeronArchive> aeronArchive = AeronArchive::connect(m_context);
        });
}

TEST_F(AeronArchiveTest, shouldExceptionForIncorrectChallengeCredentials)
{
    auto onEncodedCredentials =
        []() -> std::pair<const char *, std::uint32_t>
        {
            std::string credentials("admin:adminC");

            char *arr = new char[credentials.length() + 1];
            std::memcpy(arr, credentials.data(), credentials.length());
            arr[credentials.length()] = '\0';

            return { arr, static_cast<std::uint32_t>(credentials.length()) };
        };

    auto onChallenge =
        [](std::pair<const char *, std::uint32_t> encodedChallenge) -> std::pair<const char *, std::uint32_t>
        {
            std::string credentials("admin:adminNoCS");

            char *arr = new char[credentials.length() + 1];
            std::memcpy(arr, credentials.data(), credentials.length());
            arr[credentials.length()] = '\0';

            return { arr, static_cast<std::uint32_t>(credentials.length()) };
        };

    m_context.credentialsSupplier(CredentialsSupplier(onEncodedCredentials, onChallenge));

    ASSERT_THROW(
        {
            std::shared_ptr<AeronArchive> aeronArchive = AeronArchive::connect(m_context);
        },
        ArchiveException);
}

TEST_F(AeronArchiveTest, shouldPurgeStoppedRecording)
{
    const std::string messagePrefix = "Message ";
    const std::size_t messageCount = 10;
    std::int32_t sessionId;
    std::int64_t recordingIdFromCounter;
    std::int64_t stopPosition;

    std::shared_ptr<AeronArchive> aeronArchive = AeronArchive::connect(m_context);

    const std::int64_t subscriptionId = aeronArchive->startRecording(
        m_recordingChannel, m_recordingStreamId, AeronArchive::SourceLocation::LOCAL);

    {
        std::shared_ptr<Subscription> subscription = addSubscription(
            *aeronArchive->context().aeron(), m_recordingChannel, m_recordingStreamId);
        std::shared_ptr<Publication> publication = addPublication(
            *aeronArchive->context().aeron(), m_recordingChannel, m_recordingStreamId);

        sessionId = publication->sessionId();

        CountersReader &countersReader = aeronArchive->context().aeron()->countersReader();
        const std::int32_t counterId = getRecordingCounterId(sessionId, countersReader);
        recordingIdFromCounter = RecordingPos::getRecordingId(countersReader, counterId);

        offerMessages(*publication, messageCount, messagePrefix);
        consumeMessages(*subscription, messageCount, messagePrefix);

        stopPosition = publication->position();

        YieldingIdleStrategy idleStrategy;
        while (countersReader.getCounterValue(counterId) < stopPosition)
        {
            idleStrategy.idle();
        }

        EXPECT_EQ(aeronArchive->getRecordingPosition(recordingIdFromCounter), stopPosition);
        EXPECT_EQ(aeronArchive->getStopPosition(recordingIdFromCounter), aeron::NULL_VALUE);
    }

    aeronArchive->stopRecording(subscriptionId);

    const std::int64_t recordingId = aeronArchive->findLastMatchingRecording(
        0, "endpoint=localhost:3333", m_recordingStreamId, sessionId);

    EXPECT_EQ(recordingIdFromCounter, recordingId);
    EXPECT_EQ(aeronArchive->getStopPosition(recordingIdFromCounter), stopPosition);

    aeronArchive->purgeRecording(recordingId);

    const std::int32_t count = aeronArchive->listRecording(
        recordingId,
        [](
            std::int64_t controlSessionId,
            std::int64_t correlationId,
            std::int64_t recordingId1,
            std::int64_t startTimestamp,
            std::int64_t stopTimestamp,
            std::int64_t startPosition,
            std::int64_t newStopPosition,
            std::int32_t initialTermId,
            std::int32_t segmentFileLength,
            std::int32_t termBufferLength,
            std::int32_t mtuLength,
            std::int32_t sessionId1,
            std::int32_t streamId,
            const std::string &strippedChannel,
            const std::string &originalChannel,
            const std::string &sourceIdentity)
        {
            FAIL();
        });

    EXPECT_EQ(count, 0);
}

TEST_F(AeronArchiveTest, shouldReadRecordingDescriptor)
{
    std::shared_ptr<AeronArchive> aeronArchive = AeronArchive::connect(m_context);
    std::shared_ptr<Aeron> aeron = aeronArchive->context().aeron();

    std::shared_ptr<Publication> publication = addPublication(
        *aeron, m_recordingChannel, m_recordingStreamId);

    const std::int32_t sessionId = publication->sessionId();

    const std::int64_t subscriptionId = aeronArchive->startRecording(
        m_recordingChannel, m_recordingStreamId, AeronArchive::SourceLocation::LOCAL);

    const std::int64_t recordingId =
        [&]
        {
            CountersReader &countersReader = aeron->countersReader();
            const std::int32_t counterId = getRecordingCounterId(sessionId, countersReader);
            return RecordingPos::getRecordingId(countersReader, counterId);
        }();

    aeronArchive->stopRecording(subscriptionId);

    const std::int32_t count = aeronArchive->listRecording(
        recordingId,
        [&](
            std::int64_t controlSessionId,
            std::int64_t correlationId,
            std::int64_t recordingId1,
            std::int64_t startTimestamp,
            std::int64_t stopTimestamp,
            std::int64_t startPosition,
            std::int64_t newStopPosition,
            std::int32_t initialTermId,
            std::int32_t segmentFileLength,
            std::int32_t termBufferLength,
            std::int32_t mtuLength,
            std::int32_t sessionId1,
            std::int32_t streamId,
            const std::string &strippedChannel,
            const std::string &originalChannel,
            const std::string &sourceIdentity)
        {
            EXPECT_EQ(sessionId, sessionId1);
            EXPECT_EQ(recordingId, recordingId1);
            EXPECT_EQ(streamId, m_recordingStreamId);
            EXPECT_EQ(originalChannel, m_recordingChannel);
        });

    EXPECT_EQ(count, 1);
}

TEST_F(AeronArchiveTest, shouldReadJumboRecordingDescriptor)
{
    const std::string messagePrefix = "Message ";
    const std::size_t messageCount = 10;
    std::int32_t sessionId;
    std::int64_t recordingId;
    std::int64_t stopPosition;
    std::string recordingChannel = "aeron:udp?endpoint=localhost:3333|term-length=64k|alias=";
    recordingChannel.append(2000, 'X');

    std::shared_ptr<AeronArchive> aeronArchive = AeronArchive::connect(m_context);

    const std::int64_t subscriptionId = aeronArchive->startRecording(
        recordingChannel, m_recordingStreamId, AeronArchive::SourceLocation::LOCAL);

    {
        std::shared_ptr<Subscription> subscription = addSubscription(
            *aeronArchive->context().aeron(), recordingChannel, m_recordingStreamId);
        std::shared_ptr<Publication> publication = addPublication(
            *aeronArchive->context().aeron(), recordingChannel, m_recordingStreamId);

        sessionId = publication->sessionId();

        CountersReader &countersReader = aeronArchive->context().aeron()->countersReader();
        const std::int32_t counterId = getRecordingCounterId(sessionId, countersReader);
        recordingId = RecordingPos::getRecordingId(countersReader, counterId);

        offerMessages(*publication, messageCount, messagePrefix);
        consumeMessages(*subscription, messageCount, messagePrefix);

        stopPosition = publication->position();

        YieldingIdleStrategy idleStrategy;
        while (countersReader.getCounterValue(counterId) < stopPosition)
        {
            idleStrategy.idle();
        }

        EXPECT_EQ(aeronArchive->getRecordingPosition(recordingId), stopPosition);
        EXPECT_EQ(aeronArchive->getStopPosition(recordingId), aeron::NULL_VALUE);
    }

    aeronArchive->stopRecording(subscriptionId);

    EXPECT_EQ(aeronArchive->getStopPosition(recordingId), stopPosition);

    const std::int32_t count = aeronArchive->listRecording(
        recordingId,
        [&](
            std::int64_t controlSessionId,
            std::int64_t correlationId,
            std::int64_t recordingId1,
            std::int64_t startTimestamp,
            std::int64_t stopTimestamp,
            std::int64_t startPosition,
            std::int64_t newStopPosition,
            std::int32_t initialTermId,
            std::int32_t segmentFileLength,
            std::int32_t termBufferLength,
            std::int32_t mtuLength,
            std::int32_t sessionId1,
            std::int32_t streamId,
            const std::string &strippedChannel,
            const std::string &originalChannel,
            const std::string &sourceIdentity)
        {
            EXPECT_EQ(recordingId, recordingId1);
            EXPECT_EQ(streamId, m_recordingStreamId);
            EXPECT_EQ(originalChannel, recordingChannel);
        });

    EXPECT_EQ(count, 1);
}

TEST_F(AeronArchiveTest, shouldRecordReplicateThenReplay)
{
    const std::string messagePrefix = "Message ";
    const std::size_t messageCount = 10;
    std::int32_t sessionId;
    std::int64_t recordingId;
    std::int64_t stopPosition;

    startDestArchive();

    std::set<std::int32_t> signals;

    auto signalConsumer = [&](
        std::int64_t controlSessionId,
        std::int64_t recordingId,
        std::int64_t subscriptionId,
        std::int64_t position,
        std::int32_t recordingSignalCode) -> void
    {
        signals.insert(recordingSignalCode);
    };

    m_destContext.recordingSignalConsumer(signalConsumer);

    std::shared_ptr<AeronArchive> srcAeronArchive = AeronArchive::connect(m_context);
    std::shared_ptr<AeronArchive> dstAeronArchive = AeronArchive::connect(m_destContext);

    const std::int64_t subscriptionId = srcAeronArchive->startRecording(
        m_recordingChannel, m_recordingStreamId, AeronArchive::SourceLocation::LOCAL);

    {
        std::shared_ptr<Subscription> subscription = addSubscription(
            *srcAeronArchive->context().aeron(), m_recordingChannel, m_recordingStreamId);
        std::shared_ptr<Publication> publication = addPublication(
            *srcAeronArchive->context().aeron(), m_recordingChannel, m_recordingStreamId);

        sessionId = publication->sessionId();

        CountersReader &countersReader = srcAeronArchive->context().aeron()->countersReader();
        const std::int32_t counterId = getRecordingCounterId(sessionId, countersReader);
        recordingId = RecordingPos::getRecordingId(countersReader, counterId);
        EXPECT_TRUE(RecordingPos::isActive(countersReader, counterId, recordingId));
        EXPECT_EQ(counterId, RecordingPos::findCounterIdByRecordingId(countersReader, recordingId));
        EXPECT_EQ("aeron:ipc", RecordingPos::getSourceIdentity(countersReader, counterId));

        offerMessages(*publication, messageCount, messagePrefix);
        consumeMessages(*subscription, messageCount, messagePrefix);

        stopPosition = publication->position();

        YieldingIdleStrategy idleStrategy;
        while (countersReader.getCounterValue(counterId) < stopPosition)
        {
            idleStrategy.idle();
        }
    }

    srcAeronArchive->stopRecording(subscriptionId);

    YieldingIdleStrategy idleStrategy;
    while (srcAeronArchive->getStopPosition(recordingId) != stopPosition)
    {
        idleStrategy.idle();
    }

    auto credentials = std::make_pair("admin:admin", 11);

    ReplicationParams params;
    params.encodedCredentials(credentials);

    dstAeronArchive->replicate(
        recordingId, m_context.controlRequestStreamId(), m_context.controlRequestChannel(), params);

    while (0 == signals.count(aeron::archive::client::RecordingSignal::Value::SYNC))
    {
        dstAeronArchive->pollForRecordingSignals();
        idleStrategy.idle();
    }

    const std::int64_t position = 0L;
    const std::int64_t length = stopPosition - position;
    std::shared_ptr<Subscription> subscription = addSubscription(
        *srcAeronArchive->context().aeron(), m_replayChannel, m_replayStreamId);

    srcAeronArchive->startReplay(
        recordingId,
        m_replayChannel,
        m_replayStreamId,
        ReplayParams().position(position).length(length).fileIoMaxLength(4096));

    consumeMessages(*subscription, messageCount, messagePrefix);
    EXPECT_EQ(stopPosition, subscription->imageByIndex(0)->position());
}

TEST_F(AeronArchiveTest, shouldRecordReplicateTwice)
{
    const std::string messagePrefix = "Message ";
    const std::size_t messageCount = 10;
    std::int32_t sessionId;
    std::int64_t recordingId;
    std::int64_t stopPosition;
    YieldingIdleStrategy idleStrategy;

    startDestArchive();

    std::set<std::int32_t> signals;

    auto signalConsumer = [&](
        std::int64_t controlSessionId,
        std::int64_t recordingId,
        std::int64_t subscriptionId,
        std::int64_t position,
        std::int32_t recordingSignalCode) -> void
    {
        signals.insert(recordingSignalCode);
    };

    m_destContext.recordingSignalConsumer(signalConsumer);

    std::shared_ptr<AeronArchive> srcAeronArchive = AeronArchive::connect(m_context);
    std::shared_ptr<AeronArchive> dstAeronArchive = AeronArchive::connect(m_destContext);

    const std::int64_t subscriptionId = srcAeronArchive->startRecording(
        m_recordingChannel, m_recordingStreamId, AeronArchive::SourceLocation::LOCAL);

    std::int64_t halfwayPosition;

    {
        std::shared_ptr<Subscription> subscription = addSubscription(
            *srcAeronArchive->context().aeron(), m_recordingChannel, m_recordingStreamId);
        std::shared_ptr<Publication> publication = addPublication(
            *srcAeronArchive->context().aeron(), m_recordingChannel, m_recordingStreamId);

        sessionId = publication->sessionId();

        CountersReader &countersReader = srcAeronArchive->context().aeron()->countersReader();
        const std::int32_t counterId = getRecordingCounterId(sessionId, countersReader);
        recordingId = RecordingPos::getRecordingId(countersReader, counterId);
        EXPECT_TRUE(RecordingPos::isActive(countersReader, counterId, recordingId));
        EXPECT_EQ(counterId, RecordingPos::findCounterIdByRecordingId(countersReader, recordingId));
        EXPECT_EQ("aeron:ipc", RecordingPos::getSourceIdentity(countersReader, counterId));

        offerMessages(*publication, messageCount, messagePrefix);
        consumeMessages(*subscription, messageCount, messagePrefix);
        halfwayPosition = publication->position();
        while (countersReader.getCounterValue(counterId) < halfwayPosition)
        {
            idleStrategy.idle();
        }

        offerMessages(*publication, messageCount, messagePrefix);
        consumeMessages(*subscription, messageCount, messagePrefix);
        stopPosition = publication->position();

        while (countersReader.getCounterValue(counterId) < stopPosition)
        {
            idleStrategy.idle();
        }
    }

    srcAeronArchive->stopRecording(subscriptionId);
    while (srcAeronArchive->getStopPosition(recordingId) != stopPosition)
    {
        idleStrategy.idle();
    }

    auto credentials = std::make_pair("admin:admin", 11);

    ReplicationParams params1;
    params1
        .encodedCredentials(credentials)
        .stopPosition(halfwayPosition)
        .replicationSessionId(1);

    dstAeronArchive->replicate(
        recordingId, m_context.controlRequestStreamId(), m_context.controlRequestChannel(), params1);

    while (0 == signals.count(aeron::archive::client::RecordingSignal::Value::REPLICATE_END))
    {
        dstAeronArchive->pollForRecordingSignals();
        idleStrategy.idle();
    }

    ReplicationParams params2;
    params2
        .encodedCredentials(credentials)
        .replicationSessionId(2);

    dstAeronArchive->replicate(
        recordingId, m_context.controlRequestStreamId(), m_context.controlRequestChannel(), params2);

    signals.clear();

    while (0 == signals.count(aeron::archive::client::RecordingSignal::Value::REPLICATE_END))
    {
        dstAeronArchive->pollForRecordingSignals();
        idleStrategy.idle();
    }
}

