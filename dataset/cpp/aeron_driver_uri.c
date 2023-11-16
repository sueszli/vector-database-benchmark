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

#include <inttypes.h>
#include "uri/aeron_driver_uri.h"
#include "util/aeron_arrayutil.h"
#include "util/aeron_math.h"
#include "util/aeron_parse_util.h"
#include "aeron_driver_context.h"
#include "aeron_driver_conductor.h"

int aeron_uri_get_term_length_param(aeron_uri_params_t *uri_params, aeron_driver_uri_publication_params_t *params)
{
    const char *value_str;

    if ((value_str = aeron_uri_find_param_value(uri_params, AERON_URI_TERM_LENGTH_KEY)) != NULL)
    {
        uint64_t value;

        if (-1 == aeron_parse_size64(value_str, &value))
        {
            AERON_SET_ERR(EINVAL, "could not parse %s=%s in URI", AERON_URI_TERM_LENGTH_KEY, value_str);
            return -1;
        }

        if (aeron_logbuffer_check_term_length(value) < 0)
        {
            return -1;
        }

        params->term_length = value;
        params->has_term_length = true;
    }

    return 0;
}

int aeron_uri_get_mtu_length_param(aeron_uri_params_t *uri_params, aeron_driver_uri_publication_params_t *params)
{
    const char *value_str;

    if ((value_str = aeron_uri_find_param_value(uri_params, AERON_URI_MTU_LENGTH_KEY)) != NULL)
    {
        uint64_t value;

        if (-1 == aeron_parse_size64(value_str, &value))
        {
            AERON_SET_ERR(EINVAL, "could not parse %s=%s in URI", AERON_URI_MTU_LENGTH_KEY, value_str);
            return -1;
        }

        if (aeron_driver_context_validate_mtu_length(value) < 0)
        {
            return -1;
        }

        params->mtu_length = value;
        params->has_mtu_length = true;
    }

    return 0;
}

int aeron_uri_linger_timeout_param(aeron_uri_params_t *uri_params, aeron_driver_uri_publication_params_t *params)
{
    const char *value_str;

    if ((value_str = aeron_uri_find_param_value(uri_params, AERON_URI_LINGER_TIMEOUT_KEY)) != NULL)
    {
        uint64_t value;

        if (-1 == aeron_parse_duration_ns(value_str, &value))
        {
            AERON_SET_ERR(EINVAL, "could not parse %s=%s in URI", AERON_URI_LINGER_TIMEOUT_KEY, value_str);
            return -1;
        }

        params->linger_timeout_ns = value;
    }

    return 0;
}

int aeron_uri_publication_session_id_param(
    aeron_uri_params_t *uri_params, aeron_driver_conductor_t *conductor, aeron_driver_uri_publication_params_t *params)
{
    const char *session_id_str = aeron_uri_find_param_value(uri_params, AERON_URI_SESSION_ID_KEY);
    if (NULL != session_id_str)
    {
        if (0 == strncmp("tag:", session_id_str, strlen("tag:")))
        {
            char *end_ptr;
            errno = 0;

            long long tag = strtoll(&session_id_str[4], &end_ptr, 0);
            if (0 != errno || '\0' != *end_ptr)
            {
                AERON_SET_ERR(
                    EINVAL,
                    "could not parse %s=%s as int64_t in URI due to: %s",
                    AERON_URI_SESSION_ID_KEY, session_id_str, strerror(errno));
                return -1;
            }

            aeron_network_publication_t *publication = aeron_driver_conductor_find_network_publication_by_tag(
                conductor, (int64_t)tag);

            if (NULL == publication)
            {
                AERON_SET_ERR(
                    EINVAL, "%s=%s must reference a network publication", AERON_URI_SESSION_ID_KEY, session_id_str);
                return -1;
            }

            params->has_session_id = true;
            params->session_id = publication->session_id;
            params->mtu_length = publication->mtu_length;
            params->term_length = publication->term_buffer_length;
        }
        else
        {
            int result = aeron_uri_get_int32(uri_params, AERON_URI_SESSION_ID_KEY, &params->session_id);
            params->has_session_id = 1 == result;

            return result < 0 ? -1 : 0;
        }
    }

    return 0;
}

int aeron_uri_subscription_session_id_param(aeron_uri_params_t *uri_params, aeron_driver_uri_subscription_params_t *params)
{
    int result = aeron_uri_get_int32(uri_params, AERON_URI_SESSION_ID_KEY, &params->session_id);
    params->has_session_id = 1 == result;

    return result < 0 ? -1 : 0;
}

int aeron_diver_uri_publication_params(
    aeron_uri_t *uri,
    aeron_driver_uri_publication_params_t *params,
    aeron_driver_conductor_t *conductor,
    bool is_exclusive)
{
    aeron_driver_context_t *context = conductor->context;

    params->linger_timeout_ns = context->publication_linger_timeout_ns;
    params->term_length = AERON_URI_IPC == uri->type ? context->ipc_term_buffer_length : context->term_buffer_length;
    params->has_term_length = false;
    params->mtu_length = AERON_URI_IPC == uri->type ? context->ipc_mtu_length : context->mtu_length;
    params->has_mtu_length = false;
    params->initial_term_id = 0;
    params->term_offset = 0;
    params->term_id = 0;
    params->has_position = false;
    params->is_sparse = context->term_buffer_sparse_file;
    params->signal_eos = true;
    params->spies_simulate_connection = context->spies_simulate_connection;
    params->has_session_id = false;
    params->session_id = 0;
    params->entity_tag = AERON_URI_INVALID_TAG;

    aeron_uri_params_t *uri_params = AERON_URI_IPC == uri->type ?
        &uri->params.ipc.additional_params : &uri->params.udp.additional_params;

    if (aeron_uri_publication_session_id_param(uri_params, conductor, params) < 0)
    {
        return -1;
    }

    const char *entity_tag_str = AERON_URI_IPC == uri->type ? uri->params.ipc.entity_tag : uri->params.udp.entity_tag;
    if (NULL != entity_tag_str)
    {
        errno = 0;
        char *end_ptr;
        long long entity_tag = strtoll(entity_tag_str, &end_ptr, 10);
        if (0 != errno || *end_ptr != '\0')
        {
            AERON_SET_ERR(EINVAL, "Entity tag invalid: %s", entity_tag_str);
            return -1;
        }

        params->entity_tag = (int64_t)entity_tag;
    }

    if (aeron_uri_linger_timeout_param(uri_params, params) < 0)
    {
        return -1;
    }

    if (aeron_uri_get_term_length_param(uri_params, params) < 0)
    {
        return -1;
    }

    if (aeron_uri_get_mtu_length_param(uri_params, params) < 0)
    {
        return -1;
    }

    int count = 0;

    int32_t initial_term_id;
    int32_t term_id;
    int parse_result;

    parse_result = aeron_uri_get_int32(uri_params, AERON_URI_INITIAL_TERM_ID_KEY, &initial_term_id);
    if (parse_result < 0)
    {
        return -1;
    }
    count += parse_result > 0 ? 1 : 0;

    parse_result = aeron_uri_get_int32(uri_params, AERON_URI_TERM_ID_KEY, &term_id);
    if (parse_result < 0)
    {
        return -1;
    }
    count += parse_result > 0 ? 1 : 0;

    const char *term_offset_str = aeron_uri_find_param_value(uri_params, AERON_URI_TERM_OFFSET_KEY);
    count += term_offset_str ? 1 : 0;

    if (count > 0)
    {
        char *end_ptr = NULL;

        if (count < 3)
        {
            AERON_SET_ERR(
                EINVAL,
                "params: %s %s %s must be used as a complete set",
                AERON_URI_INITIAL_TERM_ID_KEY,
                AERON_URI_TERM_ID_KEY,
                AERON_URI_TERM_OFFSET_KEY);
            return -1;
        }

        errno = 0;
        end_ptr = NULL;
        uint64_t term_offset = strtoull(term_offset_str, &end_ptr, 0);
        if ((term_offset == 0 && 0 != errno) || end_ptr == term_offset_str)
        {
            AERON_SET_ERR(
                EINVAL,
                "could not parse %s=%s in URI: %s",
                AERON_URI_TERM_OFFSET_KEY,
                term_offset_str,
                strerror(errno));
            return -1;
        }

        if (aeron_sub_wrap_i32(term_id, initial_term_id) < 0)
        {
            AERON_SET_ERR(
                EINVAL,
                "Param difference greater than 2^31 - 1: %s=%" PRId32 " %s=%" PRId32,
                AERON_URI_INITIAL_TERM_ID_KEY,
                initial_term_id,
                AERON_URI_TERM_OFFSET_KEY,
                term_id);
            return -1;
        }

        if (term_offset > params->term_length)
        {
            AERON_SET_ERR(
                EINVAL,
                "Param %s=%" PRIu64 " > %s=%" PRIu64,
                AERON_URI_TERM_OFFSET_KEY,
                term_offset,
                AERON_URI_TERM_LENGTH_KEY,
                params->term_length);
            return -1;
        }

        if ((term_offset & (AERON_LOGBUFFER_FRAME_ALIGNMENT - 1u)) != 0)
        {
            AERON_SET_ERR(
                EINVAL,
                "Param %s=%" PRIu64 " must be multiple of FRAME_ALIGNMENT",
                AERON_URI_TERM_OFFSET_KEY,
                params->term_offset);
            return -1;
        }

        params->term_offset = term_offset;
        params->initial_term_id = initial_term_id;
        params->term_id = term_id;
        params->has_position = true;
    }

    if (aeron_uri_get_bool(uri_params, AERON_URI_SPARSE_TERM_KEY, &params->is_sparse) < 0)
    {
        return -1;
    }

    if (aeron_uri_get_bool(uri_params, AERON_URI_EOS_KEY, &params->signal_eos) < 0)
    {
        return -1;
    }

    if (aeron_uri_get_bool(uri_params, AERON_URI_SPIES_SIMULATE_CONNECTION_KEY, &params->spies_simulate_connection) < 0)
    {
        return -1;
    }

    return 0;
}

int aeron_driver_uri_subscription_params(
    aeron_uri_t *uri, aeron_driver_uri_subscription_params_t *params, aeron_driver_conductor_t *conductor)
{
    aeron_driver_context_t *context = conductor->context;

    params->is_reliable = context->reliable_stream;
    params->is_sparse = context->term_buffer_sparse_file;
    params->is_tether = context->tether_subscriptions;
    params->is_rejoin = context->rejoin_stream;
    params->initial_window_length = context->initial_window_length;

    aeron_uri_params_t *uri_params = AERON_URI_IPC == uri->type ?
        &uri->params.ipc.additional_params : &uri->params.udp.additional_params;

    if (aeron_uri_get_bool(uri_params, AERON_UDP_CHANNEL_RELIABLE_KEY, &params->is_reliable) < 0)
    {
        return -1;
    }

    if (aeron_uri_get_bool(uri_params, AERON_URI_SPARSE_TERM_KEY, &params->is_sparse) < 0)
    {
        return -1;
    }

    if (aeron_uri_get_bool(uri_params, AERON_URI_TETHER_KEY, &params->is_tether) < 0)
    {
        return -1;
    }

    if (aeron_uri_get_bool(uri_params, AERON_URI_REJOIN_KEY, &params->is_rejoin) < 0)
    {
        return -1;
    }

    params->group = aeron_config_parse_inferable_boolean(
        aeron_uri_find_param_value(uri_params, AERON_URI_GROUP_KEY), context->receiver_group_consideration);

    if (aeron_uri_subscription_session_id_param(uri_params, params) < 0)
    {
        return -1;
    }

    if (aeron_uri_get_receiver_window_length(uri_params, &params->initial_window_length) < 0)
    {
        return -1;
    }

    return 0;
}

static int aeron_publication_params_validate_mtu(size_t socket_sndbuf, size_t mtu_length, const char *name)
{
    if (socket_sndbuf < mtu_length)
    {
        AERON_SET_ERR(
            EINVAL,
            "MTU greater than SO_SNDBUF for %s: mtu=%" PRIu64 " so-sndbuf=%" PRIu64,
            name, mtu_length, socket_sndbuf);
        return -1;
    }
    else
    {
        return 0;
    }
}

int aeron_publication_params_validate_mtu_for_sndbuf(
    aeron_driver_uri_publication_params_t *params,
    size_t endpoint_socket_sndbuf,
    size_t channel_socket_sndbuf,
    size_t context_socket_sndbuf,
    size_t os_default_socket_sndbuf)
{
    if (0 != endpoint_socket_sndbuf)
    {
        return aeron_publication_params_validate_mtu(endpoint_socket_sndbuf, params->mtu_length, "endpoint");
    }

    if (0 != channel_socket_sndbuf)
    {
        return aeron_publication_params_validate_mtu(channel_socket_sndbuf, params->mtu_length, "channel");
    }

    if (0 != context_socket_sndbuf)
    {
        return aeron_publication_params_validate_mtu(context_socket_sndbuf, params->mtu_length, "context");
    }

    if (0 != os_default_socket_sndbuf)
    {
        return aeron_publication_params_validate_mtu(os_default_socket_sndbuf, params->mtu_length, "os default");
    }

    return 0;
}

int aeron_subscription_params_validate_initial_window_for_rcvbuf(
    aeron_driver_uri_subscription_params_t *params,
    size_t endpoint_socket_rcvbuf,
    size_t os_default_socket_rcvbuf)
{
    if (0 != endpoint_socket_rcvbuf && endpoint_socket_rcvbuf < params->initial_window_length)
    {
        AERON_SET_ERR(
            EINVAL,
            "Initial window greater than SO_SNDBUF for channel: rcv-wnd=%" PRIu64 " so-rcvbuf=%" PRIu64,
            params->initial_window_length, endpoint_socket_rcvbuf);
        return -1;
    }

    if (0 == endpoint_socket_rcvbuf && os_default_socket_rcvbuf < params->initial_window_length)
    {
        AERON_SET_ERR(
            EINVAL,
            "Initial window greater than SO_SNDBUF for channel: rcv-wnd=%" PRIu64 " so-rcvbuf=%" PRIu64 " (OS default)",
            params->initial_window_length, endpoint_socket_rcvbuf);
        return -1;
    }

    return 0;
}

int aeron_driver_uri_get_timestamp_offset(aeron_uri_t *uri, const char *key, int32_t *offset)
{
    *offset = AERON_NULL_VALUE;

    if (AERON_URI_UDP != uri->type)
    {
        return 0;
    }

    const char *offset_str = aeron_uri_find_param_value(&uri->params.udp.additional_params, key);

    if (NULL == offset_str)
    {
        return 0;
    }

    if (0 == strcmp(AERON_URI_TIMESTAMP_OFFSET_RESERVED, offset_str))
    {
        *offset = AERON_UDP_CHANNEL_RESERVED_VALUE_OFFSET;
        return 0;
    }

    char *end_ptr = NULL;
    errno = 0;
    long parse_offset = strtol(offset_str, &end_ptr, 0);
    errno = 0 == errno && '\0' != *end_ptr ? EINVAL : 0;
    if (0 != errno)
    {
        AERON_SET_ERR(errno, "Invalid %s: %s", AERON_URI_MEDIA_RCV_TIMESTAMP_OFFSET_KEY, offset_str);
        return -1;
    }

    *offset = (int32_t)parse_offset;

    return 0;
}

const char *aeron_driver_uri_get_offset_info(int32_t offset)
{
    if (AERON_NULL_VALUE == offset)
    {
        return "(not specified)";
    }
    else if (AERON_UDP_CHANNEL_RESERVED_VALUE_OFFSET == offset)
    {
        return "(reserved)";
    }

    return "";
}
