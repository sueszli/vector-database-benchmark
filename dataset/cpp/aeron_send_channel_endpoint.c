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

#if defined(__linux__)
#define _BSD_SOURCE
#define _GNU_SOURCE
#endif

#include <string.h>
#include "aeron_socket.h"
#include "uri/aeron_uri.h"
#include "aeron_driver_sender.h"
#include "aeron_driver_context.h"
#include "aeron_alloc.h"
#include "aeron_position.h"
#include "aeron_timestamps.h"

#if !defined(HAVE_STRUCT_MMSGHDR)
struct mmsghdr
{
    struct msghdr msg_hdr;
    unsigned int msg_len;
};
#endif

int aeron_send_channel_endpoint_create(
    aeron_send_channel_endpoint_t **endpoint,
    aeron_udp_channel_t *channel,
    aeron_driver_uri_publication_params_t *params,
    aeron_driver_context_t *context,
    aeron_counters_manager_t *counters_manager,
    int64_t registration_id)
{
    aeron_send_channel_endpoint_t *_endpoint = NULL;
    char bind_addr_and_port[AERON_NETUTIL_FORMATTED_MAX_LENGTH];
    int bind_addr_and_port_length;

    if (aeron_alloc((void **)&_endpoint, sizeof(aeron_send_channel_endpoint_t)) < 0)
    {
        aeron_udp_channel_delete(channel);
        return -1;
    }

    _endpoint->destination_tracker = NULL;
    _endpoint->data_paths = &context->sender_proxy->sender->data_paths;

    struct sockaddr_storage *connect_addr = NULL;
    if (aeron_udp_channel_is_multi_destination(channel))
    {
        if (aeron_alloc((void **)&_endpoint->destination_tracker, sizeof(aeron_udp_destination_tracker_t)) < 0 ||
            aeron_udp_destination_tracker_init(
                _endpoint->destination_tracker,
                _endpoint->data_paths,
                context->sender_cached_clock,
                AERON_UDP_CHANNEL_CONTROL_MODE_MANUAL == channel->control_mode,
                AERON_UDP_DESTINATION_TRACKER_DESTINATION_TIMEOUT_NS) < 0)
        {
            aeron_udp_channel_delete(channel);
            aeron_free(_endpoint);
            return -1;
        }
    }
    else if (context->connect_enabled)
    {
        connect_addr = &channel->remote_data;
    }

    _endpoint->conductor_fields.refcnt = 0;
    _endpoint->conductor_fields.udp_channel = channel;
    _endpoint->conductor_fields.managed_resource.incref = aeron_send_channel_endpoint_incref;
    _endpoint->conductor_fields.managed_resource.decref = aeron_send_channel_endpoint_decref;
    _endpoint->conductor_fields.managed_resource.clientd = _endpoint;
    _endpoint->conductor_fields.managed_resource.registration_id = -1;
    _endpoint->conductor_fields.status = AERON_SEND_CHANNEL_ENDPOINT_STATUS_ACTIVE;
    _endpoint->conductor_fields.socket_sndbuf = 0 != channel->socket_sndbuf_length ?
        channel->socket_sndbuf_length : context->socket_sndbuf;
    _endpoint->conductor_fields.socket_rcvbuf = 0 != channel->socket_rcvbuf_length ?
        channel->socket_rcvbuf_length : context->socket_rcvbuf;
    _endpoint->transport.fd = -1;
    _endpoint->channel_status.counter_id = -1;
    _endpoint->local_sockaddr_indicator.counter_id = -1;
    _endpoint->tracker_num_destinations.counter_id = -1;
    _endpoint->transport_bindings = context->udp_channel_transport_bindings;
    _endpoint->data_paths = &context->sender_proxy->sender->data_paths;
    _endpoint->transport.data_paths = _endpoint->data_paths;

    if (context->sender_port_manager->get_managed_port(
        context->sender_port_manager->state,
        &_endpoint->bind_addr,
        channel,
        channel->is_multicast ? &channel->remote_control : &channel->local_control) < 0)
    {
        AERON_APPEND_ERR("uri=%s", channel->original_uri);
        aeron_send_channel_endpoint_delete(counters_manager, _endpoint);
        return -1;
    }

    _endpoint->port_manager = context->sender_port_manager;

    aeron_udp_channel_transport_params_t transport_params = {
        _endpoint->conductor_fields.socket_rcvbuf,
        _endpoint->conductor_fields.socket_sndbuf,
        params->mtu_length,
        channel->interface_index,
        0 != channel->multicast_ttl ? channel->multicast_ttl : context->multicast_ttl,
        false,
    };

    if (context->udp_channel_transport_bindings->init_func(
        &_endpoint->transport,
        &_endpoint->bind_addr,
        channel->is_multicast ? &channel->local_control : &channel->remote_control,
        connect_addr,
        &transport_params,
        context,
        AERON_UDP_CHANNEL_TRANSPORT_AFFINITY_SENDER) < 0)
    {
        AERON_APPEND_ERR("uri=%s", channel->original_uri);
        aeron_send_channel_endpoint_delete(counters_manager, _endpoint);
        return -1;
    }

    if (aeron_udp_channel_is_channel_snd_timestamps_enabled(channel))
    {
        _endpoint->transport.timestamp_flags |= AERON_UDP_CHANNEL_TRANSPORT_CHANNEL_SND_TIMESTAMP;
    }

    if (aeron_int64_to_ptr_hash_map_init(
        &_endpoint->publication_dispatch_map, 8, AERON_MAP_DEFAULT_LOAD_FACTOR) < 0)
    {
        aeron_send_channel_endpoint_delete(counters_manager, _endpoint);
        return -1;
    }

    if ((bind_addr_and_port_length = aeron_send_channel_endpoint_bind_addr_and_port(
        _endpoint, bind_addr_and_port, sizeof(bind_addr_and_port))) < 0)
    {
        aeron_send_channel_endpoint_delete(counters_manager, _endpoint);
        return -1;
    }

    _endpoint->transport.dispatch_clientd = _endpoint;
    _endpoint->has_sender_released = false;

    _endpoint->channel_status.counter_id = aeron_counter_send_channel_status_allocate(
        counters_manager, registration_id, channel->uri_length, channel->original_uri);
    _endpoint->channel_status.value_addr = aeron_counters_manager_addr(
        counters_manager, _endpoint->channel_status.counter_id);

    if (_endpoint->channel_status.counter_id < 0)
    {
        aeron_send_channel_endpoint_delete(counters_manager, _endpoint);
        return -1;
    }

    if (NULL != _endpoint->destination_tracker)
    {
        _endpoint->tracker_num_destinations.counter_id = aeron_channel_endpoint_status_allocate(
            counters_manager,
            AERON_COUNTER_CHANNEL_MDC_NUM_DESTINATIONS_NAME,
            AERON_COUNTER_CHANNEL_NUM_DESTINATIONS_TYPE_ID,
            registration_id,
            channel->uri_length,
            channel->original_uri);
        _endpoint->tracker_num_destinations.value_addr = aeron_counters_manager_addr(
            counters_manager, _endpoint->tracker_num_destinations.counter_id);

        if (_endpoint->tracker_num_destinations.counter_id < 0)
        {
            aeron_send_channel_endpoint_delete(counters_manager, _endpoint);
            return -1;
        }

        aeron_udp_destination_tracker_set_counter(
            _endpoint->destination_tracker, &_endpoint->tracker_num_destinations);
    }

    // TODO: Remove the update and just create in a single shot.
    aeron_channel_endpoint_status_update_label(
        counters_manager,
        _endpoint->channel_status.counter_id,
        AERON_COUNTER_SEND_CHANNEL_STATUS_NAME,
        channel->uri_length,
        channel->original_uri,
        bind_addr_and_port_length,
        bind_addr_and_port);

    _endpoint->local_sockaddr_indicator.counter_id = aeron_counter_local_sockaddr_indicator_allocate(
        counters_manager,
        AERON_COUNTER_SND_LOCAL_SOCKADDR_NAME,
        registration_id,
        _endpoint->channel_status.counter_id,
        bind_addr_and_port);

    _endpoint->local_sockaddr_indicator.value_addr = aeron_counters_manager_addr(
        counters_manager, _endpoint->local_sockaddr_indicator.counter_id);

    if (_endpoint->local_sockaddr_indicator.counter_id < 0)
    {
        aeron_send_channel_endpoint_delete(counters_manager, _endpoint);
        return -1;
    }

    aeron_counter_set_ordered(
        _endpoint->local_sockaddr_indicator.value_addr, AERON_COUNTER_CHANNEL_ENDPOINT_STATUS_ACTIVE);

    _endpoint->sender_proxy = context->sender_proxy;
    _endpoint->cached_clock = context->sender_cached_clock;
    _endpoint->time_of_last_sm_ns = aeron_clock_cached_nano_time(_endpoint->cached_clock);
    memcpy(&_endpoint->current_data_addr, &channel->remote_data, sizeof(_endpoint->current_data_addr));

    *endpoint = _endpoint;
    return 0;
}

int aeron_send_channel_endpoint_delete(
    aeron_counters_manager_t *counters_manager, aeron_send_channel_endpoint_t *endpoint)
{
    if (NULL != counters_manager)
    {
        if (-1 != endpoint->channel_status.counter_id)
        {
            aeron_counters_manager_free(counters_manager, endpoint->channel_status.counter_id);
        }

        if (-1 != endpoint->local_sockaddr_indicator.counter_id)
        {
            aeron_counters_manager_free(counters_manager, endpoint->local_sockaddr_indicator.counter_id);
        }

        if (-1 != endpoint->tracker_num_destinations.counter_id)
        {
            aeron_counters_manager_free(counters_manager, endpoint->tracker_num_destinations.counter_id);
        }
    }

    aeron_int64_to_ptr_hash_map_delete(&endpoint->publication_dispatch_map);
    aeron_udp_channel_delete(endpoint->conductor_fields.udp_channel);
    endpoint->transport_bindings->close_func(&endpoint->transport);

    if (NULL != endpoint->port_manager)
    {
        endpoint->port_manager->free_managed_port(endpoint->port_manager->state, &endpoint->bind_addr);
    }

    if (NULL != endpoint->destination_tracker)
    {
        aeron_udp_destination_tracker_close(endpoint->destination_tracker);
        aeron_free(endpoint->destination_tracker);
    }

    aeron_free(endpoint);

    return 0;
}

void aeron_send_channel_endpoint_incref(void *clientd)
{
    aeron_send_channel_endpoint_t *endpoint = (aeron_send_channel_endpoint_t *)clientd;

    endpoint->conductor_fields.refcnt++;
}

void aeron_send_channel_endpoint_decref(void *clientd)
{
    aeron_send_channel_endpoint_t *endpoint = (aeron_send_channel_endpoint_t *)clientd;

    if (0 == --endpoint->conductor_fields.refcnt)
    {
        /* mark as CLOSING to be aware not to use again (to be receiver_released and deleted) */
        endpoint->conductor_fields.status = AERON_SEND_CHANNEL_ENDPOINT_STATUS_CLOSING;
        aeron_driver_sender_proxy_on_remove_endpoint(endpoint->sender_proxy, endpoint);
    }
}

static void aeron_send_channel_apply_timestamps(
    aeron_send_channel_endpoint_t *endpoint,
    struct iovec *iov,
    size_t iov_length)
{
    if (AERON_UDP_CHANNEL_TRANSPORT_CHANNEL_SND_TIMESTAMP & endpoint->transport.timestamp_flags)
    {
        struct timespec send_timestamp;
        if (0 == aeron_clock_gettime_realtime(&send_timestamp))
        {
            int32_t offset = endpoint->conductor_fields.udp_channel->channel_snd_timestamp_offset;

            for (size_t i = 0; i < iov_length; i++)
            {
                aeron_timestamps_set_timestamp(
                    &send_timestamp,
                    offset,
                    (uint8_t *)iov[i].iov_base,
                    iov[i].iov_len);
            }
        }
    }
}

int aeron_send_channel_send(
    aeron_send_channel_endpoint_t *endpoint,
    struct iovec *iov,
    size_t iov_length,
    int64_t *bytes_sent)
{
    int result;

    aeron_send_channel_apply_timestamps(endpoint, iov, iov_length);

    if (NULL == endpoint->destination_tracker)
    {
        result = endpoint->data_paths->send_func(
            endpoint->data_paths, &endpoint->transport, &endpoint->current_data_addr, iov, iov_length, bytes_sent);
    }
    else
    {
        result = aeron_udp_destination_tracker_send(
            endpoint->destination_tracker, &endpoint->transport, iov, iov_length, bytes_sent);
    }

    return result;
}

int aeron_send_channel_endpoint_add_publication(
    aeron_send_channel_endpoint_t *endpoint, aeron_network_publication_t *publication)
{
    int64_t key_value = aeron_map_compound_key(publication->stream_id, publication->session_id);

    int result = aeron_int64_to_ptr_hash_map_put(&endpoint->publication_dispatch_map, key_value, publication);
    if (result < 0)
    {
        AERON_APPEND_ERR("%s", "Failed to add publication to publication_dispatch_map");
    }

    return result;
}

int aeron_send_channel_endpoint_remove_publication(
    aeron_send_channel_endpoint_t *endpoint, aeron_network_publication_t *publication)
{
    int64_t key_value = aeron_map_compound_key(publication->stream_id, publication->session_id);

    aeron_int64_to_ptr_hash_map_remove(&endpoint->publication_dispatch_map, key_value);
    return 0;
}

void aeron_send_channel_endpoint_dispatch(
    aeron_udp_channel_data_paths_t *data_paths,
    aeron_udp_channel_transport_t *transport,
    void *sender_clientd,
    void *endpoint_clientd,
    void *destination_clientd,
    uint8_t *buffer,
    size_t length,
    struct sockaddr_storage *addr,
    struct timespec *media_timestamp)
{
    aeron_driver_sender_t *sender = (aeron_driver_sender_t *)sender_clientd;
    aeron_frame_header_t *frame_header = (aeron_frame_header_t *)buffer;
    aeron_send_channel_endpoint_t *endpoint = (aeron_send_channel_endpoint_t *)endpoint_clientd;

    if ((length < sizeof(aeron_frame_header_t)) || (frame_header->version != AERON_FRAME_HEADER_VERSION))
    {
        aeron_counter_increment(sender->invalid_frames_counter, 1);
        return;
    }

    switch (frame_header->type)
    {
        case AERON_HDR_TYPE_NAK:
            if (length >= sizeof(aeron_nak_header_t))
            {
                aeron_send_channel_endpoint_on_nak(endpoint, buffer, length, addr);
                aeron_counter_ordered_increment(sender->nak_messages_received_counter, 1);
            }
            else
            {
                aeron_counter_increment(sender->invalid_frames_counter, 1);
            }
            break;

        case AERON_HDR_TYPE_SM:
            if (length >= sizeof(aeron_status_message_header_t))
            {
                aeron_send_channel_endpoint_on_status_message(endpoint, buffer, length, addr);
                aeron_counter_ordered_increment(sender->status_messages_received_counter, 1);
            }
            else
            {
                aeron_counter_increment(sender->invalid_frames_counter, 1);
            }
            break;

        case AERON_HDR_TYPE_RTTM:
            if (length >= sizeof(aeron_rttm_header_t))
            {
                aeron_send_channel_endpoint_on_rttm(endpoint, buffer, length, addr);
            }
            else
            {
                aeron_counter_increment(sender->invalid_frames_counter, 1);
            }
            break;

        default:
            break;
    }
}

void aeron_send_channel_endpoint_on_nak(
    aeron_send_channel_endpoint_t *endpoint, uint8_t *buffer, size_t length, struct sockaddr_storage *addr)
{
    aeron_nak_header_t *nak_header = (aeron_nak_header_t *)buffer;
    int64_t key_value = aeron_map_compound_key(nak_header->stream_id, nak_header->session_id);
    aeron_network_publication_t *publication = aeron_int64_to_ptr_hash_map_get(
        &endpoint->publication_dispatch_map, key_value);

    if (NULL != publication)
    {
        aeron_network_publication_on_nak(publication, nak_header->term_id, nak_header->term_offset, nak_header->length);
    }
}

void aeron_send_channel_endpoint_publication_trigger_send_setup_frame(void *clientd, int64_t key, void *value)
{
    aeron_network_publication_t *publication = (aeron_network_publication_t *)value;

    aeron_network_publication_trigger_send_setup_frame(publication);
}

void aeron_send_channel_endpoint_on_status_message(
    aeron_send_channel_endpoint_t *endpoint, uint8_t *buffer, size_t length, struct sockaddr_storage *addr)
{
    aeron_status_message_header_t *sm_header = (aeron_status_message_header_t *)buffer;
    int64_t key_value = aeron_map_compound_key(sm_header->stream_id, sm_header->session_id);
    aeron_network_publication_t *publication = aeron_int64_to_ptr_hash_map_get(
        &endpoint->publication_dispatch_map, key_value);

    if (NULL != endpoint->destination_tracker)
    {
        aeron_udp_destination_tracker_on_status_message(endpoint->destination_tracker, buffer, length, addr);
    }

    if (NULL != publication)
    {
        if (sm_header->frame_header.flags & AERON_STATUS_MESSAGE_HEADER_SEND_SETUP_FLAG)
        {
            aeron_network_publication_trigger_send_setup_frame(publication);
        }
        else
        {
            aeron_network_publication_on_status_message(publication, buffer, length, addr);
        }

        endpoint->time_of_last_sm_ns = aeron_clock_cached_nano_time(endpoint->cached_clock);
    }
}

void aeron_send_channel_endpoint_on_rttm(
    aeron_send_channel_endpoint_t *endpoint, uint8_t *buffer, size_t length, struct sockaddr_storage *addr)
{
    aeron_rttm_header_t *rttm_header = (aeron_rttm_header_t *)buffer;
    int64_t key_value = aeron_map_compound_key(rttm_header->stream_id, rttm_header->session_id);
    aeron_network_publication_t *publication = aeron_int64_to_ptr_hash_map_get(
        &endpoint->publication_dispatch_map, key_value);

    if (NULL != publication)
    {
        aeron_network_publication_on_rttm(publication, buffer, length, addr);
    }
}

int aeron_send_channel_endpoint_check_for_re_resolution(
    aeron_send_channel_endpoint_t *endpoint, int64_t now_ns, aeron_driver_conductor_proxy_t *conductor_proxy)
{
    if (AERON_UDP_CHANNEL_CONTROL_MODE_MANUAL == endpoint->conductor_fields.udp_channel->control_mode)
    {
        aeron_udp_destination_tracker_check_for_re_resolution(
            endpoint->destination_tracker, endpoint, now_ns, conductor_proxy);
    }
    else if (!endpoint->conductor_fields.udp_channel->is_multicast &&
        endpoint->conductor_fields.udp_channel->has_explicit_endpoint &&
        now_ns > (endpoint->time_of_last_sm_ns + AERON_SEND_CHANNEL_ENDPOINT_DESTINATION_TIMEOUT_NS))
    {
        const char *endpoint_name = endpoint->conductor_fields.udp_channel->uri.params.udp.endpoint;

        aeron_driver_conductor_proxy_on_re_resolve_endpoint(
            conductor_proxy, endpoint_name, endpoint, &endpoint->current_data_addr);
    }

    return 0;
}

int aeron_send_channel_endpoint_resolution_change(
    aeron_driver_context_t *context,
    aeron_send_channel_endpoint_t *endpoint,
    const char *endpoint_name,
    struct sockaddr_storage *new_addr)
{
    if (NULL != endpoint->destination_tracker)
    {
        aeron_udp_destination_tracker_resolution_change(endpoint->destination_tracker, endpoint_name, new_addr);
    }
    else
    {
        memcpy(&endpoint->current_data_addr, new_addr, sizeof(endpoint->current_data_addr));
        if (context->udp_channel_transport_bindings->reconnect_func(&endpoint->transport, &endpoint->current_data_addr) < 0)
        {
            char addr_str[AERON_NETUTIL_FORMATTED_MAX_LENGTH];
            aeron_format_source_identity(addr_str, sizeof(addr_str), &endpoint->current_data_addr);
            AERON_APPEND_ERR("failed to reconnect transport with re-resolved address: %s", addr_str);
            return -1;
        }
    }

    return 0;
}

extern void aeron_send_channel_endpoint_sender_release(aeron_send_channel_endpoint_t *endpoint);

extern bool aeron_send_channel_endpoint_has_sender_released(aeron_send_channel_endpoint_t *endpoint);

extern int aeron_send_channel_endpoint_add_destination(
    aeron_send_channel_endpoint_t *endpoint,
    aeron_uri_t *uri,
    struct sockaddr_storage *addr);

extern int aeron_send_channel_endpoint_remove_destination(
    aeron_send_channel_endpoint_t *endpoint,
    struct sockaddr_storage *addr,
    aeron_uri_t **removed_uri);

extern bool aeron_send_channel_endpoint_tags_match(
    aeron_send_channel_endpoint_t *endpoint, aeron_udp_channel_t *channel);

extern int aeron_send_channel_endpoint_bind_addr_and_port(
    aeron_send_channel_endpoint_t *endpoint, char *buffer, size_t length);

extern bool aeron_send_channel_is_unicast(aeron_send_channel_endpoint_t *endpoint);
