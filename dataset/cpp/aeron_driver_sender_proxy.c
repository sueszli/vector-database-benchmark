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

#include "aeron_driver_sender.h"
#include "aeron_alloc.h"

void aeron_driver_sender_proxy_offer(aeron_driver_sender_proxy_t *sender_proxy, void *cmd, size_t length)
{
    aeron_rb_write_result_t result;
    while (AERON_RB_FULL == (result = aeron_mpsc_rb_write(sender_proxy->command_queue, 1, cmd, length)))
    {
        aeron_counter_ordered_increment(sender_proxy->fail_counter, 1);
        sched_yield();
    }

    if (AERON_RB_ERROR == result)
    {
        aeron_distinct_error_log_record(
            sender_proxy->sender->error_log, EINVAL, "Error writing to receiver proxy ring buffer");
    }
}

void aeron_driver_sender_proxy_on_add_endpoint(
    aeron_driver_sender_proxy_t *sender_proxy, aeron_send_channel_endpoint_t *endpoint)
{
    sender_proxy->on_add_endpoint_func(endpoint->conductor_fields.udp_channel);
    aeron_command_base_t cmd =
        {
            .func = aeron_driver_sender_on_add_endpoint,
            .item = endpoint
        };

    if (AERON_THREADING_MODE_IS_SHARED_OR_INVOKER(sender_proxy->threading_mode))
    {
        aeron_driver_sender_on_add_endpoint(sender_proxy->sender, &cmd);
    }
    else
    {
        aeron_driver_sender_proxy_offer(sender_proxy, &cmd, sizeof(cmd));
    }
}

void aeron_driver_sender_proxy_on_remove_endpoint(
    aeron_driver_sender_proxy_t *sender_proxy, aeron_send_channel_endpoint_t *endpoint)
{
    sender_proxy->on_remove_endpoint_func(endpoint->conductor_fields.udp_channel);
    aeron_command_base_t cmd =
        {
            .func = aeron_driver_sender_on_remove_endpoint,
            .item = endpoint
        };

    if (AERON_THREADING_MODE_IS_SHARED_OR_INVOKER(sender_proxy->threading_mode))
    {
        aeron_driver_sender_on_remove_endpoint(sender_proxy->sender, &cmd);
    }
    else
    {
        aeron_driver_sender_proxy_offer(sender_proxy, &cmd, sizeof(cmd));
    }
}

void aeron_driver_sender_proxy_on_add_publication(
    aeron_driver_sender_proxy_t *sender_proxy, aeron_network_publication_t *publication)
{
    aeron_command_base_t cmd =
        {
            .func = aeron_driver_sender_on_add_publication,
            .item = publication
        };

    if (AERON_THREADING_MODE_IS_SHARED_OR_INVOKER(sender_proxy->threading_mode))
    {
        aeron_driver_sender_on_add_publication(sender_proxy->sender, &cmd);
    }
    else
    {
        aeron_driver_sender_proxy_offer(sender_proxy, &cmd, sizeof(cmd));
    }
}

void aeron_driver_sender_proxy_on_remove_publication(
    aeron_driver_sender_proxy_t *sender_proxy, aeron_network_publication_t *publication)
{
    aeron_command_base_t cmd =
        {
            .func = aeron_driver_sender_on_remove_publication,
            .item = publication
        };

    if (AERON_THREADING_MODE_IS_SHARED_OR_INVOKER(sender_proxy->threading_mode))
    {
        aeron_driver_sender_on_remove_publication(sender_proxy->sender, &cmd);
    }
    else
    {
        aeron_driver_sender_proxy_offer(sender_proxy, &cmd, sizeof(cmd));
    }
}

void aeron_driver_sender_proxy_on_add_destination(
    aeron_driver_sender_proxy_t *sender_proxy,
    aeron_send_channel_endpoint_t *endpoint,
    aeron_uri_t *uri,
    struct sockaddr_storage *addr)
{
    aeron_command_destination_t cmd =
        {
            .base = { .func = aeron_driver_sender_on_add_destination, .item = NULL },
            .endpoint = endpoint,
            .uri = uri
        };
    memcpy(&cmd.control_address, addr, sizeof(cmd.control_address));

    if (AERON_THREADING_MODE_IS_SHARED_OR_INVOKER(sender_proxy->threading_mode))
    {
        aeron_driver_sender_on_add_destination(sender_proxy->sender, &cmd);
    }
    else
    {
        aeron_driver_sender_proxy_offer(sender_proxy, &cmd, sizeof(cmd));
    }
}

void aeron_driver_sender_proxy_on_remove_destination(
    aeron_driver_sender_proxy_t *sender_proxy, aeron_send_channel_endpoint_t *endpoint, struct sockaddr_storage *addr)
{
    aeron_command_destination_t cmd =
        {
            .base = { .func = aeron_driver_sender_on_remove_destination, .item = NULL },
            .endpoint = endpoint
        };
    memcpy(&cmd.control_address, addr, sizeof(cmd.control_address));

    if (AERON_THREADING_MODE_IS_SHARED_OR_INVOKER(sender_proxy->threading_mode))
    {
        aeron_driver_sender_on_remove_destination(sender_proxy->sender, &cmd);
    }
    else
    {
        aeron_driver_sender_proxy_offer(sender_proxy, &cmd, sizeof(cmd));
    }
}

void aeron_driver_sender_proxy_on_resolution_change(
    aeron_driver_sender_proxy_t *sender_proxy,
    const char *endpoint_name,
    aeron_send_channel_endpoint_t *endpoint,
    struct sockaddr_storage *new_addr)
{
    aeron_command_sender_resolution_change_t cmd =
        {
            .base = { .func = aeron_driver_sender_on_resolution_change, .item = NULL },
            .endpoint = endpoint,
            .endpoint_name = endpoint_name,
        };
    memcpy(&cmd.new_addr, new_addr, sizeof(cmd.new_addr));

    if (AERON_THREADING_MODE_IS_SHARED_OR_INVOKER(sender_proxy->threading_mode))
    {
        aeron_driver_sender_on_resolution_change(sender_proxy->sender, &cmd);
    }
    else
    {
        aeron_driver_sender_proxy_offer(sender_proxy, &cmd, sizeof(cmd));
    }
}

void aeron_driver_sender_proxy_on_delete_cmd(aeron_driver_sender_proxy_t *sender_proxy, aeron_command_base_t *cmd)
{
}
