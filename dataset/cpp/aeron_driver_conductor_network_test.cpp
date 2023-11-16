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

#include <cinttypes>
#include "aeron_driver_conductor_test.h"

using testing::_;
using testing::Eq;
using testing::Ne;
using testing::Args;
using testing::Mock;
using testing::AnyNumber;

class DriverConductorNetworkTest : public DriverConductorTest, public testing::Test
{
protected:
    aeron_publication_image_t *aeron_driver_conductor_find_publication_image(
        aeron_driver_conductor_t *conductor, aeron_receive_channel_endpoint_t *endpoint, int32_t stream_id)
    {
        for (size_t i = 0, length = conductor->publication_images.length; i < length; i++)
        {
            aeron_publication_image_t *image = conductor->publication_images.array[i].image;

            if (endpoint == image->endpoint && stream_id == image->stream_id)
            {
                return image;
            }
        }

        return NULL;
    }

    inline size_t aeron_publication_image_subscriber_count(aeron_publication_image_t *image)
    {
        return image->conductor_fields.subscribable.length;
    }
};

TEST_F(DriverConductorNetworkTest, shouldBeAbleToAddMultipleNetworkSubscriptionsWithDifferentChannelSameStreamId)
{
    int64_t client_id = nextCorrelationId();
    int64_t sub_id_1 = nextCorrelationId();
    int64_t sub_id_2 = nextCorrelationId();
    int64_t sub_id_3 = nextCorrelationId();
    int64_t sub_id_4 = nextCorrelationId();

    ASSERT_EQ(addNetworkSubscription(client_id, sub_id_1, CHANNEL_1, STREAM_ID_1), 0);
    ASSERT_EQ(addNetworkSubscription(client_id, sub_id_2, CHANNEL_2, STREAM_ID_1), 0);
    ASSERT_EQ(addNetworkSubscription(client_id, sub_id_3, CHANNEL_3, STREAM_ID_1), 0);
    ASSERT_EQ(addNetworkSubscription(client_id, sub_id_4, CHANNEL_4, STREAM_ID_1), 0);

    doWorkUntilDone();

    aeron_receive_channel_endpoint_t *endpoint_1 = aeron_driver_conductor_find_receive_channel_endpoint(
        &m_conductor.m_conductor, CHANNEL_1);
    aeron_receive_channel_endpoint_t *endpoint_2 = aeron_driver_conductor_find_receive_channel_endpoint(
        &m_conductor.m_conductor, CHANNEL_2);
    aeron_receive_channel_endpoint_t *endpoint_3 = aeron_driver_conductor_find_receive_channel_endpoint(
        &m_conductor.m_conductor, CHANNEL_3);
    aeron_receive_channel_endpoint_t *endpoint_4 = aeron_driver_conductor_find_receive_channel_endpoint(
        &m_conductor.m_conductor, CHANNEL_4);

    ASSERT_NE(endpoint_1, (aeron_receive_channel_endpoint_t *)nullptr);
    ASSERT_NE(endpoint_2, (aeron_receive_channel_endpoint_t *)nullptr);
    ASSERT_NE(endpoint_3, (aeron_receive_channel_endpoint_t *)nullptr);
    ASSERT_NE(endpoint_4, (aeron_receive_channel_endpoint_t *)nullptr);
    ASSERT_EQ(aeron_driver_conductor_num_receive_channel_endpoints(&m_conductor.m_conductor), 4u);
    ASSERT_EQ(aeron_driver_conductor_num_network_subscriptions(&m_conductor.m_conductor), 4u);

    EXPECT_CALL(m_mockCallbacks, broadcastToClient(_, _, _));
    EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_SUBSCRIPTION_READY, _, _))
        .With(IsSubscriptionReady(sub_id_1));
    EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_SUBSCRIPTION_READY, _, _))
        .With(IsSubscriptionReady(sub_id_2));
    EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_SUBSCRIPTION_READY, _, _))
        .With(IsSubscriptionReady(sub_id_3));
    EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_SUBSCRIPTION_READY, _, _))
        .With(IsSubscriptionReady(sub_id_4));

    readAllBroadcastsFromConductor(mock_broadcast_handler);
}

TEST_F(DriverConductorNetworkTest, shouldKeepSubscriptionMediaEndpointUponRemovalOfAllButOneSubscriber)
{
    int64_t client_id = nextCorrelationId();
    int64_t sub_id_1 = nextCorrelationId();
    int64_t sub_id_2 = nextCorrelationId();
    int64_t sub_id_3 = nextCorrelationId();
    int64_t sub_id_4 = nextCorrelationId();

    ASSERT_EQ(addNetworkSubscription(client_id, sub_id_1, CHANNEL_1, STREAM_ID_1), 0);
    ASSERT_EQ(addNetworkSubscription(client_id, sub_id_2, CHANNEL_1, STREAM_ID_2), 0);
    ASSERT_EQ(addNetworkSubscription(client_id, sub_id_3, CHANNEL_1, STREAM_ID_3), 0);
    ASSERT_EQ(addNetworkSubscription(client_id, sub_id_4, CHANNEL_1, STREAM_ID_4), 0);

    doWorkUntilDone();

    int64_t remove_correlation_id_1 = nextCorrelationId();
    int64_t remove_correlation_id_2 = nextCorrelationId();
    int64_t remove_correlation_id_3 = nextCorrelationId();

    ASSERT_EQ(removeSubscription(client_id, remove_correlation_id_1, sub_id_1), 0);
    ASSERT_EQ(removeSubscription(client_id, remove_correlation_id_2, sub_id_2), 0);
    ASSERT_EQ(removeSubscription(client_id, remove_correlation_id_3, sub_id_3), 0);

    doWorkUntilDone();

    aeron_receive_channel_endpoint_t *endpoint = aeron_driver_conductor_find_receive_channel_endpoint(
        &m_conductor.m_conductor, CHANNEL_1);

    ASSERT_NE(endpoint, (aeron_receive_channel_endpoint_t *)nullptr);
    ASSERT_EQ(aeron_driver_conductor_num_receive_channel_endpoints(&m_conductor.m_conductor), 1u);
    ASSERT_EQ(aeron_driver_conductor_num_network_subscriptions(&m_conductor.m_conductor), 1u);

    EXPECT_CALL(m_mockCallbacks, broadcastToClient(_, _, _)).Times(AnyNumber());
    EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_SUBSCRIPTION_READY, _, _)).Times(4);
    EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_OPERATION_SUCCESS, _, _)).Times(3);

    readAllBroadcastsFromConductor(mock_broadcast_handler);
}

TEST_F(DriverConductorNetworkTest, shouldCreatePublicationImageForActiveNetworkSubscription)
{
    int64_t client_id = nextCorrelationId();
    int64_t sub_id = nextCorrelationId();

    ASSERT_EQ(addNetworkSubscription(client_id, sub_id, CHANNEL_1, STREAM_ID_1), 0);
    doWorkUntilDone();
    readAllBroadcastsFromConductor(null_broadcast_handler);

    aeron_receive_channel_endpoint_t *endpoint = aeron_driver_conductor_find_receive_channel_endpoint(
        &m_conductor.m_conductor, CHANNEL_1);

    createPublicationImage(endpoint, STREAM_ID_1, 1000);

    EXPECT_EQ(aeron_driver_conductor_num_images(&m_conductor.m_conductor), 1u);

    aeron_publication_image_t *image = aeron_driver_conductor_find_publication_image(
        &m_conductor.m_conductor, endpoint, STREAM_ID_1);

    EXPECT_NE(image, (aeron_publication_image_t *)nullptr);
    EXPECT_EQ(aeron_publication_image_subscriber_count(image), 1u);

    int64_t image_registration_id = aeron_publication_image_registration_id(image);
    const char *log_file_name = aeron_publication_image_log_file_name(image);

    EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_AVAILABLE_IMAGE, _, _))
        .With(IsAvailableImage(image_registration_id, sub_id, STREAM_ID_1, SESSION_ID, log_file_name, SOURCE_IDENTITY));

    readAllBroadcastsFromConductor(mock_broadcast_handler);
}

TEST_F(DriverConductorNetworkTest, shouldNotCreatePublicationImageForNonActiveNetworkSubscription)
{
    int64_t client_id = nextCorrelationId();
    int64_t sub_id = nextCorrelationId();

    ASSERT_EQ(addNetworkSubscription(client_id, sub_id, CHANNEL_1, STREAM_ID_1), 0);
    doWorkUntilDone();
    readAllBroadcastsFromConductor(null_broadcast_handler);

    aeron_receive_channel_endpoint_t *endpoint = aeron_driver_conductor_find_receive_channel_endpoint(
        &m_conductor.m_conductor, CHANNEL_1);

    createPublicationImage(endpoint, STREAM_ID_2, 1000);

    EXPECT_EQ(aeron_driver_conductor_num_images(&m_conductor.m_conductor), 0u);

    readAllBroadcastsFromConductor(null_broadcast_handler);
}

TEST_F(DriverConductorNetworkTest, shouldRemoveSubscriptionFromImageWhenRemoveSubscription)
{
    int64_t client_id = nextCorrelationId();
    int64_t sub_id = nextCorrelationId();

    ASSERT_EQ(addNetworkSubscription(client_id, sub_id, CHANNEL_1, STREAM_ID_1), 0);
    doWorkUntilDone();
    readAllBroadcastsFromConductor(null_broadcast_handler);

    aeron_receive_channel_endpoint_t *endpoint = aeron_driver_conductor_find_receive_channel_endpoint(
        &m_conductor.m_conductor, CHANNEL_1);

    createPublicationImage(endpoint, STREAM_ID_1, 1000);

    EXPECT_EQ(aeron_driver_conductor_num_images(&m_conductor.m_conductor), 1u);

    aeron_publication_image_t *image =
        aeron_driver_conductor_find_publication_image(&m_conductor.m_conductor, endpoint, STREAM_ID_1);

    EXPECT_NE(image, (aeron_publication_image_t *)nullptr);
    EXPECT_EQ(aeron_publication_image_subscriber_count(image), 1u);

    int64_t remove_correlation_id = nextCorrelationId();
    ASSERT_EQ(removeSubscription(client_id, remove_correlation_id, sub_id), 0);
    doWork();

    EXPECT_EQ(aeron_publication_image_subscriber_count(image), 0u);

    EXPECT_CALL(m_mockCallbacks, broadcastToClient(_, _, _)).Times(AnyNumber());
    EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_OPERATION_SUCCESS, _, _))
        .With(IsOperationSuccess(remove_correlation_id));
    readAllBroadcastsFromConductor(mock_broadcast_handler);
}

TEST_F(DriverConductorNetworkTest, shouldTimeoutImageAndSendUnavailableImageWhenNoActivity)
{
    int64_t client_id = nextCorrelationId();
    int64_t sub_id = nextCorrelationId();

    ASSERT_EQ(addNetworkSubscription(client_id, sub_id, CHANNEL_1, STREAM_ID_1), 0);
    doWorkUntilDone();
    readAllBroadcastsFromConductor(null_broadcast_handler);

    aeron_receive_channel_endpoint_t *endpoint = aeron_driver_conductor_find_receive_channel_endpoint(
        &m_conductor.m_conductor, CHANNEL_1);

    createPublicationImage(endpoint, STREAM_ID_1, 1000);

    EXPECT_EQ(aeron_driver_conductor_num_images(&m_conductor.m_conductor), 1u);

    aeron_publication_image_t *image = aeron_driver_conductor_find_publication_image(
        &m_conductor.m_conductor, endpoint, STREAM_ID_1);

    EXPECT_NE(image, (aeron_publication_image_t *)nullptr);
    EXPECT_EQ(aeron_publication_image_subscriber_count(image), 1u);
    readAllBroadcastsFromConductor(null_broadcast_handler);

    int64_t image_correlation_id = image->conductor_fields.managed_resource.registration_id;

    int64_t timeout = m_context.m_context->image_liveness_timeout_ns +
        (m_context.m_context->client_liveness_timeout_ns * 2) + (m_context.m_context->timer_interval_ns * 3);

    doWorkForNs(
        timeout,
        100,
        [&]()
        {
            clientKeepalive(client_id);
        });

    EXPECT_EQ(aeron_driver_conductor_num_images(&m_conductor.m_conductor), 0u);

    EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_UNAVAILABLE_IMAGE, _, _))
        .With(IsUnavailableImage(STREAM_ID_1, image_correlation_id, sub_id, CHANNEL_1));
    readAllBroadcastsFromConductor(mock_broadcast_handler);
}

TEST_F(DriverConductorNetworkTest, shouldRemoveSubscriptionAfterImageTimeout)
{
    int64_t client_id = nextCorrelationId();
    int64_t sub_id = nextCorrelationId();
    int64_t remove_correlation_id = nextCorrelationId();

    ASSERT_EQ(addNetworkSubscription(client_id, sub_id, CHANNEL_1, STREAM_ID_1), 0);
    doWorkUntilDone();

    aeron_receive_channel_endpoint_t *endpoint = aeron_driver_conductor_find_receive_channel_endpoint(
        &m_conductor.m_conductor, CHANNEL_1);

    createPublicationImage(endpoint, STREAM_ID_1, 1000);

    int64_t timeout = m_context.m_context->image_liveness_timeout_ns +
        (m_context.m_context->client_liveness_timeout_ns * 2) + (m_context.m_context->timer_interval_ns * 3);

    doWorkForNs(
        timeout,
        100,
        [&]()
        {
            clientKeepalive(client_id);
        });

    readAllBroadcastsFromConductor(null_broadcast_handler);
    EXPECT_EQ(aeron_driver_conductor_num_images(&m_conductor.m_conductor), 0u);
    EXPECT_EQ(aeron_driver_conductor_num_active_network_subscriptions(&m_conductor.m_conductor, CHANNEL_1, STREAM_ID_1), 0u);
    ASSERT_EQ(removeSubscription(client_id, remove_correlation_id, sub_id), 0);

    doWorkUntilDone();
    EXPECT_EQ(aeron_driver_conductor_num_network_subscriptions(&m_conductor.m_conductor), 0u);
}

TEST_F(DriverConductorNetworkTest, shouldRetryFreeOperationsAfterSubscrptionIsClosed)
{
    int64_t client_id = nextCorrelationId();
    int64_t sub_id = nextCorrelationId();
    int64_t remove_correlation_id = nextCorrelationId();

    ASSERT_EQ(addNetworkSubscription(client_id, sub_id, CHANNEL_1, STREAM_ID_1), 0);
    doWorkUntilDone();

    aeron_receive_channel_endpoint_t *endpoint = aeron_driver_conductor_find_receive_channel_endpoint(
        &m_conductor.m_conductor, CHANNEL_1);

    createPublicationImage(endpoint, STREAM_ID_1, 1000);

    const int64_t image_timeout = m_context.m_context->image_liveness_timeout_ns +
        (m_context.m_context->client_liveness_timeout_ns * 2) + (m_context.m_context->timer_interval_ns * 3);

    free_map_raw_log = false;
    int64_t *free_fails_counter = aeron_system_counter_addr(
        &m_conductor.m_conductor.system_counters, AERON_SYSTEM_COUNTER_FREE_FAILS);
    EXPECT_EQ(aeron_counter_get(free_fails_counter), 0);

    doWorkForNs(
        image_timeout,
        100,
        [&]()
        {
            clientKeepalive(client_id);
        });

    readAllBroadcastsFromConductor(null_broadcast_handler);
    const int64_t free_fails = aeron_counter_get(free_fails_counter);
    EXPECT_GT(free_fails, 0);
    EXPECT_EQ(aeron_driver_conductor_num_images(&m_conductor.m_conductor), 0u);
    EXPECT_EQ(aeron_driver_conductor_num_active_network_subscriptions(&m_conductor.m_conductor, CHANNEL_1, STREAM_ID_1), 0u);
    free_map_raw_log = true;

    doWorkUntilDone();

    readAllBroadcastsFromConductor(null_broadcast_handler);
    EXPECT_EQ(aeron_counter_get(free_fails_counter), free_fails);
    EXPECT_EQ(aeron_driver_conductor_num_images(&m_conductor.m_conductor), 0u);
    EXPECT_EQ(aeron_driver_conductor_num_active_network_subscriptions(&m_conductor.m_conductor, CHANNEL_1, STREAM_ID_1), 0u);
    ASSERT_EQ(removeSubscription(client_id, remove_correlation_id, sub_id), 0);

    doWorkUntilDone();
    EXPECT_EQ(aeron_driver_conductor_num_network_subscriptions(&m_conductor.m_conductor), 0u);
}

TEST_F(DriverConductorNetworkTest, shouldSendAvailableImageForMultipleSubscriptions)
{
    int64_t client_id = nextCorrelationId();
    int64_t sub_id_1 = nextCorrelationId();
    int64_t sub_id_2 = nextCorrelationId();

    ASSERT_EQ(addNetworkSubscription(client_id, sub_id_1, CHANNEL_1, STREAM_ID_1), 0);
    ASSERT_EQ(addNetworkSubscription(client_id, sub_id_2, CHANNEL_1, STREAM_ID_1), 0);
    doWorkUntilDone();
    readAllBroadcastsFromConductor(null_broadcast_handler);

    aeron_receive_channel_endpoint_t *endpoint = aeron_driver_conductor_find_receive_channel_endpoint(
        &m_conductor.m_conductor, CHANNEL_1);

    createPublicationImage(endpoint, STREAM_ID_1, 1000);

    aeron_publication_image_t *image = aeron_driver_conductor_find_publication_image(
        &m_conductor.m_conductor, endpoint, STREAM_ID_1);

    EXPECT_NE(image, (aeron_publication_image_t *)nullptr);
    EXPECT_EQ(aeron_publication_image_subscriber_count(image), 2u);

    int64_t image_registration_id = aeron_publication_image_registration_id(image);
    const char *log_file_name = aeron_publication_image_log_file_name(image);

    EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_AVAILABLE_IMAGE, _, _))
        .With(IsAvailableImage(image_registration_id, sub_id_1, STREAM_ID_1, SESSION_ID, log_file_name, SOURCE_IDENTITY));
    EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_AVAILABLE_IMAGE, _, _))
        .With(IsAvailableImage(image_registration_id, sub_id_2, STREAM_ID_1, SESSION_ID, log_file_name, SOURCE_IDENTITY));

    readAllBroadcastsFromConductor(mock_broadcast_handler);
}

TEST_F(DriverConductorNetworkTest, shouldSendAvailableImageForSecondSubscriptionAfterCreatingImage)
{
    int64_t client_id = nextCorrelationId();
    int64_t sub_id_1 = nextCorrelationId();
    int64_t sub_id_2 = nextCorrelationId();

    ASSERT_EQ(addNetworkSubscription(client_id, sub_id_1, CHANNEL_1, STREAM_ID_1), 0);
    doWorkUntilDone();

    aeron_receive_channel_endpoint_t *endpoint = aeron_driver_conductor_find_receive_channel_endpoint(
        &m_conductor.m_conductor, CHANNEL_1);

    createPublicationImage(endpoint, STREAM_ID_1, 1000);

    aeron_publication_image_t *image = aeron_driver_conductor_find_publication_image(
        &m_conductor.m_conductor, endpoint, STREAM_ID_1);

    EXPECT_NE(image, (aeron_publication_image_t *)nullptr);

    ASSERT_EQ(addNetworkSubscription(client_id, sub_id_2, CHANNEL_1, STREAM_ID_1), 0);
    doWork();

    int64_t image_registration_id = aeron_publication_image_registration_id(image);
    const char *log_file_name = aeron_publication_image_log_file_name(image);

    EXPECT_CALL(m_mockCallbacks, broadcastToClient(_, _, _));
    {
        testing::InSequence sequenced;

        EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_SUBSCRIPTION_READY, _, _))
            .With(IsSubscriptionReady(sub_id_1));
        EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_AVAILABLE_IMAGE, _, _))
            .With(IsAvailableImage(image_registration_id, sub_id_1, STREAM_ID_1, SESSION_ID, log_file_name, SOURCE_IDENTITY));
        EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_SUBSCRIPTION_READY, _, _))
            .With(IsSubscriptionReady(sub_id_2));
        EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_AVAILABLE_IMAGE, _, _))
            .With(IsAvailableImage(image_registration_id, sub_id_2, STREAM_ID_1, SESSION_ID, log_file_name, SOURCE_IDENTITY));
    }
    readAllBroadcastsFromConductor(mock_broadcast_handler);
}

TEST_F(DriverConductorNetworkTest, shouldTimeoutImageAndSendUnavailableImageWhenNoActivityForMultipleSubscriptions)
{
    int64_t client_id = nextCorrelationId();
    int64_t sub_id_1 = nextCorrelationId();
    int64_t sub_id_2 = nextCorrelationId();

    ASSERT_EQ(addNetworkSubscription(client_id, sub_id_1, CHANNEL_1, STREAM_ID_1), 0);
    doWorkUntilDone();

    aeron_receive_channel_endpoint_t *endpoint = aeron_driver_conductor_find_receive_channel_endpoint(
        &m_conductor.m_conductor, CHANNEL_1);

    createPublicationImage(endpoint, STREAM_ID_1, 1000);

    aeron_publication_image_t *image = aeron_driver_conductor_find_publication_image(
        &m_conductor.m_conductor, endpoint, STREAM_ID_1);

    EXPECT_NE(image, (aeron_publication_image_t *)nullptr);

    ASSERT_EQ(addNetworkSubscription(client_id, sub_id_2, CHANNEL_1, STREAM_ID_1), 0);
    doWorkUntilDone();

    int64_t image_correlation_id = aeron_publication_image_registration_id(image);
    int64_t timeout = m_context.m_context->image_liveness_timeout_ns +
        (m_context.m_context->client_liveness_timeout_ns * 2);

    doWorkForNs(
        timeout,
        100,
        [&]()
        {
            clientKeepalive(client_id);
        });

    EXPECT_CALL(m_mockCallbacks, broadcastToClient(_, _, _));
    {
        testing::InSequence sequenced;

        EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_SUBSCRIPTION_READY, _, _))
            .With(IsSubscriptionReady(sub_id_1));
        EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_AVAILABLE_IMAGE, _, _));
        EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_SUBSCRIPTION_READY, _, _))
            .With(IsSubscriptionReady(sub_id_2));
        EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_AVAILABLE_IMAGE, _, _));
        EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_UNAVAILABLE_IMAGE, _, _))
            .With(IsUnavailableImage(STREAM_ID_1, image_correlation_id, sub_id_1, CHANNEL_1));
        EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_UNAVAILABLE_IMAGE, _, _))
            .With(IsUnavailableImage(STREAM_ID_1, image_correlation_id, sub_id_2, CHANNEL_1));
    }
    readAllBroadcastsFromConductor(mock_broadcast_handler);
}

TEST_F(DriverConductorNetworkTest, shouldUseExistingChannelEndpointOnAddPublicationWithSameTagIdAndSameStreamId)
{
    int64_t client_id = nextCorrelationId();
    int64_t pub_id_1 = nextCorrelationId();
    int64_t pub_id_2 = nextCorrelationId();

    ASSERT_EQ(addPublication(client_id, pub_id_1, CHANNEL_1 "|tags=1001", STREAM_ID_1, false), 0);
    ASSERT_EQ(addPublication(client_id, pub_id_2, "aeron:udp?tags=1001", STREAM_ID_1, false), 0);
    doWorkUntilDone();
    EXPECT_EQ(aeron_driver_conductor_num_send_channel_endpoints(&m_conductor.m_conductor), 1u);
    EXPECT_EQ(aeron_driver_conductor_num_network_publications(&m_conductor.m_conductor), 1u);
    readAllBroadcastsFromConductor(null_broadcast_handler);

    doWorkForNs(
        m_context.m_context->publication_linger_timeout_ns + (m_context.m_context->client_liveness_timeout_ns * 2));
    EXPECT_EQ(aeron_driver_conductor_num_clients(&m_conductor.m_conductor), 0u);
    EXPECT_EQ(aeron_driver_conductor_num_network_publications(&m_conductor.m_conductor), 0u);
    EXPECT_EQ(aeron_driver_conductor_num_send_channel_endpoints(&m_conductor.m_conductor), 0u);
}

TEST_F(DriverConductorNetworkTest, shouldUseExistingChannelEndpointOnAddPublicationWithSameTagIdDifferentStreamId)
{
    int64_t client_id = nextCorrelationId();
    int64_t pub_id_1 = nextCorrelationId();
    int64_t pub_id_2 = nextCorrelationId();

    ASSERT_EQ(addPublication(client_id, pub_id_1, CHANNEL_1 "|tags=1001", STREAM_ID_1, false), 0);
    ASSERT_EQ(addPublication(client_id, pub_id_2, "aeron:udp?tags=1001", STREAM_ID_2, false), 0);
    doWorkUntilDone();
    EXPECT_EQ(aeron_driver_conductor_num_send_channel_endpoints(&m_conductor.m_conductor), 1u);
    EXPECT_EQ(aeron_driver_conductor_num_network_publications(&m_conductor.m_conductor), 2u);
    readAllBroadcastsFromConductor(null_broadcast_handler);

    doWorkForNs(
        m_context.m_context->publication_linger_timeout_ns + (m_context.m_context->client_liveness_timeout_ns * 2));
    EXPECT_EQ(aeron_driver_conductor_num_clients(&m_conductor.m_conductor), 0u);
    EXPECT_EQ(aeron_driver_conductor_num_network_publications(&m_conductor.m_conductor), 0u);
    EXPECT_EQ(aeron_driver_conductor_num_send_channel_endpoints(&m_conductor.m_conductor), 0u);
}

TEST_F(DriverConductorNetworkTest, shouldUseExistingChannelEndpointOnAddSubscriptionWithSameTagId)
{
    int64_t client_id = nextCorrelationId();
    int64_t sub_id_1 = nextCorrelationId();
    int64_t sub_id_2 = nextCorrelationId();

    ASSERT_EQ(addNetworkSubscription(client_id, sub_id_1, CHANNEL_1 "|tags=1001", STREAM_ID_1), 0);
    ASSERT_EQ(addNetworkSubscription(client_id, sub_id_2, "aeron:udp?tags=1001", STREAM_ID_1), 0);
    doWorkUntilDone();
    EXPECT_EQ(aeron_driver_conductor_num_receive_channel_endpoints(&m_conductor.m_conductor), 1u);
    EXPECT_EQ(aeron_driver_conductor_num_network_subscriptions(&m_conductor.m_conductor), 2u);
    readAllBroadcastsFromConductor(null_broadcast_handler);

    doWorkForNs(
        m_context.m_context->publication_linger_timeout_ns + (m_context.m_context->client_liveness_timeout_ns * 2));
    EXPECT_EQ(aeron_driver_conductor_num_clients(&m_conductor.m_conductor), 0u);
    EXPECT_EQ(aeron_driver_conductor_num_network_subscriptions(&m_conductor.m_conductor), 0u);
    EXPECT_EQ(aeron_driver_conductor_num_receive_channel_endpoints(&m_conductor.m_conductor), 0u);
}

TEST_F(DriverConductorNetworkTest, shouldUseExistingPublicationOnAddPublicationWithSameSessionTagIdAndSameStreamId)
{
    int64_t client_id = nextCorrelationId();
    int64_t pub_id_1 = nextCorrelationId();
    int64_t pub_id_2 = nextCorrelationId();

    ASSERT_EQ(addPublication(client_id, pub_id_1, CHANNEL_1 "|tags=1001,1002", STREAM_ID_1, false), 0);
    ASSERT_EQ(addPublication(client_id, pub_id_2, CHANNEL_2 "|session-id=tag:1002", STREAM_ID_1, false), 0);
    doWorkUntilDone();
    EXPECT_EQ(aeron_driver_conductor_num_send_channel_endpoints(&m_conductor.m_conductor), 2u);
    EXPECT_EQ(aeron_driver_conductor_num_network_publications(&m_conductor.m_conductor), 2u);
    readAllBroadcastsFromConductor(null_broadcast_handler);

    aeron_network_publication_t *pub_1 = aeron_driver_conductor_find_network_publication(
        &m_conductor.m_conductor, pub_id_1);
    aeron_network_publication_t *pub_2 = aeron_driver_conductor_find_network_publication(
        &m_conductor.m_conductor, pub_id_2);

    EXPECT_EQ(pub_1->session_id, pub_2->session_id);

    doWorkForNs(
        m_context.m_context->publication_linger_timeout_ns + (m_context.m_context->client_liveness_timeout_ns * 2));
    EXPECT_EQ(aeron_driver_conductor_num_clients(&m_conductor.m_conductor), 0u);
    EXPECT_EQ(aeron_driver_conductor_num_network_publications(&m_conductor.m_conductor), 0u);
    EXPECT_EQ(aeron_driver_conductor_num_send_channel_endpoints(&m_conductor.m_conductor), 0u);
}

TEST_F(DriverConductorNetworkTest, shouldErrorWithUnknownSessionIdTag)
{
    int64_t client_id = nextCorrelationId();
    int64_t pub_id_1 = nextCorrelationId();

    ASSERT_EQ(addPublication(client_id, pub_id_1, CHANNEL_2 "|session-id=tag:1002", STREAM_ID_1, false), 0);
    doWork();

    EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_ERROR, _, _));
    readAllBroadcastsFromConductor(mock_broadcast_handler);
}

TEST_F(DriverConductorNetworkTest, shouldErrorWithInvalidSessionIdTag)
{
    int64_t client_id = nextCorrelationId();
    int64_t pub_id_1 = nextCorrelationId();

    ASSERT_EQ(addPublication(client_id, pub_id_1, CHANNEL_1 "|tags=1001,1002", STREAM_ID_1, false), 0);
    doWorkUntilDone();
    readAllBroadcastsFromConductor(null_broadcast_handler);

    ASSERT_EQ(addPublication(client_id, pub_id_1, CHANNEL_2 "|session-id=tag:1002a", STREAM_ID_1, false), 0);
    doWorkUntilDone();

    EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_ERROR, _, _));
    readAllBroadcastsFromConductor(mock_broadcast_handler);
}

TEST_F(DriverConductorNetworkTest, shouldBeAbleToAddAndRemoveDestinationToManualMdcPublication)
{
    int64_t client_id = nextCorrelationId();
    int64_t pub_id = nextCorrelationId();
    int64_t add_destination_id = nextCorrelationId();
    int64_t remove_destination_id = nextCorrelationId();

    ASSERT_EQ(addPublication(client_id, pub_id, CHANNEL_MDC_MANUAL, STREAM_ID_1, false), 0);
    doWorkUntilDone();
    EXPECT_EQ(aeron_driver_conductor_num_network_publications(&m_conductor.m_conductor), 1u);
    readAllBroadcastsFromConductor(null_broadcast_handler);

    ASSERT_EQ(addDestination(client_id, add_destination_id, pub_id, CHANNEL_1), 0);
    doWorkUntilDone();

    EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_OPERATION_SUCCESS, _, _))
        .With(IsOperationSuccess(add_destination_id));
    readAllBroadcastsFromConductor(mock_broadcast_handler);

    ASSERT_EQ(removeDestination(client_id, remove_destination_id, pub_id, CHANNEL_1), 0);
    doWork();

    EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_OPERATION_SUCCESS, _, _))
        .With(IsOperationSuccess(remove_destination_id));
    readAllBroadcastsFromConductor(mock_broadcast_handler);
}

TEST_F(DriverConductorNetworkTest, shouldFailToAddSubscriptionWithDifferentReliabilityParameter)
{
    int64_t client_id = nextCorrelationId();
    int64_t sub_id = nextCorrelationId();

    ASSERT_EQ(addNetworkSubscription(client_id, sub_id, CHANNEL_1, STREAM_ID_1), 0);
    doWorkUntilDone();
    readAllBroadcastsFromConductor(null_broadcast_handler);

    ASSERT_EQ(addNetworkSubscription(client_id, sub_id, CHANNEL_1_UNRELIABLE, STREAM_ID_1), 0);
    doWorkUntilDone();

    EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_ERROR, _, _));
    readAllBroadcastsFromConductor(mock_broadcast_handler);
}

TEST_F(DriverConductorNetworkTest, shouldAllowDifferentReliabilityParameterWithSpecificSession)
{
    int64_t client_id = nextCorrelationId();
    int64_t sub_id = nextCorrelationId();

    ASSERT_EQ(addNetworkSubscription(client_id, sub_id, CHANNEL_1_UNRELIABLE, STREAM_ID_1), 0);
    doWorkUntilDone();
    readAllBroadcastsFromConductor(null_broadcast_handler);

    ASSERT_EQ(addNetworkSubscription(client_id, sub_id, CHANNEL_1_WITH_SESSION_ID_1, STREAM_ID_1), 0);
    doWorkUntilDone();

    EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_SUBSCRIPTION_READY, _, _));
    readAllBroadcastsFromConductor(mock_broadcast_handler);
}

TEST_F(DriverConductorNetworkTest, shouldAddMdsWithSingleUnicastSubscription)
{
    int64_t client_id = nextCorrelationId();
    int64_t sub_id = nextCorrelationId();

    ASSERT_EQ(addNetworkSubscription(client_id, sub_id, CHANNEL_MDC_MANUAL, STREAM_ID_1), 0);
    doWorkUntilDone();

    EXPECT_CALL(m_mockCallbacks, broadcastToClient(_, _, _));
    EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_SUBSCRIPTION_READY, _, _));
    readAllBroadcastsFromConductor(mock_broadcast_handler);

    int64_t dest_correlation_id = nextCorrelationId();

    ASSERT_EQ(addReceiveDestination(client_id, dest_correlation_id, sub_id, CHANNEL_1), 0);
    doWorkUntilDone();

    EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_OPERATION_SUCCESS, _, _));
    readAllBroadcastsFromConductor(mock_broadcast_handler);
}

TEST_F(DriverConductorNetworkTest, shouldFailToAddMdsWithSingleUnicastSubscriptionWithInvalidRegistrationId)
{
    int64_t client_id = nextCorrelationId();
    int64_t sub_id = nextCorrelationId();
    int64_t invalid_sub_id = sub_id + 1000000;

    ASSERT_EQ(addNetworkSubscription(client_id, sub_id, CHANNEL_MDC_MANUAL, STREAM_ID_1), 0);
    doWorkUntilDone();

    EXPECT_CALL(m_mockCallbacks, broadcastToClient(_, _, _));
    EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_SUBSCRIPTION_READY, _, _));
    readAllBroadcastsFromConductor(mock_broadcast_handler);

    int64_t dest_correlation_id = nextCorrelationId();

    ASSERT_EQ(addReceiveDestination(client_id, dest_correlation_id, invalid_sub_id, CHANNEL_1), 0);
    doWorkUntilDone();

    EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_ERROR, _, _));
    readAllBroadcastsFromConductor(mock_broadcast_handler);
}

TEST_F(DriverConductorNetworkTest, shouldFailToAddMdsWithSingleUnicastSubscriptionWithInvalidNonManualControlEndpoint)
{
    int64_t client_id = nextCorrelationId();
    int64_t sub_id = nextCorrelationId();

    ASSERT_EQ(addNetworkSubscription(client_id, sub_id, CHANNEL_2, STREAM_ID_1), 0);
    doWorkUntilDone();

    EXPECT_CALL(m_mockCallbacks, broadcastToClient(_, _, _));
    EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_SUBSCRIPTION_READY, _, _));
    readAllBroadcastsFromConductor(mock_broadcast_handler);

    int64_t dest_correlation_id = nextCorrelationId();

    ASSERT_EQ(addReceiveDestination(client_id, dest_correlation_id, sub_id, CHANNEL_1), 0);
    doWorkUntilDone();

    EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_ERROR, _, _));
    readAllBroadcastsFromConductor(mock_broadcast_handler);
}

TEST_F(DriverConductorNetworkTest, shouldAddAndRemoveMdsWithSingleUnicastSubscription)
{
    int64_t client_id = nextCorrelationId();
    int64_t sub_id = nextCorrelationId();

    ASSERT_EQ(addNetworkSubscription(client_id, sub_id, CHANNEL_MDC_MANUAL, STREAM_ID_1), 0);
    doWorkUntilDone();
    readAllBroadcastsFromConductor(null_broadcast_handler);

    int64_t dest_correlation_id = nextCorrelationId();

    ASSERT_EQ(addReceiveDestination(client_id, dest_correlation_id, sub_id, CHANNEL_1), 0);
    doWorkUntilDone();
    readAllBroadcastsFromConductor(null_broadcast_handler);

    int64_t remove_correlation_id = nextCorrelationId();

    ASSERT_EQ(removeReceiveDestination(client_id, remove_correlation_id, sub_id, CHANNEL_1), 0);
    doWorkUntilDone();

    EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_OPERATION_SUCCESS, _, _));
    readAllBroadcastsFromConductor(mock_broadcast_handler);
}

TEST_F(DriverConductorNetworkTest, shouldFailToRemoveMdsWithSingleUnicastSubscriptionWithInvalidSusbcriptionId)
{
    int64_t client_id = nextCorrelationId();
    int64_t sub_id = nextCorrelationId();
    int64_t invalid_sub_id = sub_id + 1000000;

    ASSERT_EQ(addNetworkSubscription(client_id, sub_id, CHANNEL_MDC_MANUAL, STREAM_ID_1), 0);
    doWorkUntilDone();
    readAllBroadcastsFromConductor(null_broadcast_handler);

    int64_t dest_correlation_id = nextCorrelationId();

    ASSERT_EQ(addReceiveDestination(client_id, dest_correlation_id, sub_id, CHANNEL_1), 0);
    doWorkUntilDone();
    readAllBroadcastsFromConductor(null_broadcast_handler);

    int64_t remove_correlation_id = nextCorrelationId();

    ASSERT_EQ(removeReceiveDestination(client_id, remove_correlation_id, invalid_sub_id, CHANNEL_1), 0);
    doWorkUntilDone();

    EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_ERROR, _, _));
    readAllBroadcastsFromConductor(mock_broadcast_handler);
}

TEST_F(DriverConductorNetworkTest, shouldFailToAddPublicationWithAtsEnabled)
{
    int64_t client_id = nextCorrelationId();
    int64_t pub_id = nextCorrelationId();

    ASSERT_EQ(addPublication(client_id, pub_id, CHANNEL_1 "|ats=true", STREAM_ID_1, false), 0);
    doWorkUntilDone();
    EXPECT_CALL(m_mockCallbacks, broadcastToClient(_, _, _));
    EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_ERROR, _, _));
    readAllBroadcastsFromConductor(mock_broadcast_handler);
}

TEST_F(DriverConductorNetworkTest, shouldFailToAddSubscriptionWithAtsEnabled)
{
    int64_t client_id = nextCorrelationId();
    int64_t pub_id = nextCorrelationId();

    ASSERT_EQ(addNetworkSubscription(client_id, pub_id, CHANNEL_1 "|ats=true", STREAM_ID_1), 0);
    doWorkUntilDone();
    EXPECT_CALL(m_mockCallbacks, broadcastToClient(_, _, _));
    EXPECT_CALL(m_mockCallbacks, broadcastToClient(AERON_RESPONSE_ON_ERROR, _, _));
    readAllBroadcastsFromConductor(mock_broadcast_handler);
}
