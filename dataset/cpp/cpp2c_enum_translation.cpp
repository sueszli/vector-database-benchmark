// Copyright (c) 2020 by Robert Bosch GmbH. All rights reserved.
// Copyright (c) 2020 - 2022 by Apex.AI Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include "iceoryx_binding_c/internal/cpp2c_enum_translation.hpp"
#include "iceoryx_binding_c/enums.h"

using namespace iox;
using namespace iox::popo;

namespace cpp2c
{
iox_SubscribeState subscribeState(const iox::SubscribeState value) noexcept
{
    switch (value)
    {
    case iox::SubscribeState::NOT_SUBSCRIBED:
        return iox_SubscribeState::SubscribeState_NOT_SUBSCRIBED;
    case iox::SubscribeState::SUBSCRIBE_REQUESTED:
        return iox_SubscribeState::SubscribeState_SUBSCRIBE_REQUESTED;
    case iox::SubscribeState::SUBSCRIBED:
        return iox_SubscribeState::SubscribeState_SUBSCRIBED;
    case iox::SubscribeState::UNSUBSCRIBE_REQUESTED:
        return iox_SubscribeState::SubscribeState_UNSUBSCRIBE_REQUESTED;
    case iox::SubscribeState::WAIT_FOR_OFFER:
        return iox_SubscribeState::SubscribeState_WAIT_FOR_OFFER;
    }
    return iox_SubscribeState::SubscribeState_UNDEFINED_ERROR;
}

iox_ChunkReceiveResult chunkReceiveResult(const iox::popo::ChunkReceiveResult value) noexcept
{
    switch (value)
    {
    case iox::popo::ChunkReceiveResult::NO_CHUNK_AVAILABLE:
        return ChunkReceiveResult_NO_CHUNK_AVAILABLE;
    case iox::popo::ChunkReceiveResult::TOO_MANY_CHUNKS_HELD_IN_PARALLEL:
        return ChunkReceiveResult_TOO_MANY_CHUNKS_HELD_IN_PARALLEL;
    }
    return ChunkReceiveResult_UNDEFINED_ERROR;
}

iox_AllocationResult allocationResult(const iox::popo::AllocationError value) noexcept
{
    switch (value)
    {
    case AllocationError::UNDEFINED_ERROR:
        return AllocationResult_UNDEFINED_ERROR;
    case AllocationError::NO_MEMPOOLS_AVAILABLE:
        return AllocationResult_NO_MEMPOOLS_AVAILABLE;
    case AllocationError::RUNNING_OUT_OF_CHUNKS:
        return AllocationResult_RUNNING_OUT_OF_CHUNKS;
    case AllocationError::TOO_MANY_CHUNKS_ALLOCATED_IN_PARALLEL:
        return AllocationResult_TOO_MANY_CHUNKS_ALLOCATED_IN_PARALLEL;
    case AllocationError::INVALID_PARAMETER_FOR_USER_PAYLOAD_OR_USER_HEADER:
        return AllocationResult_INVALID_PARAMETER_FOR_USER_PAYLOAD_OR_USER_HEADER;
    case AllocationError::INVALID_PARAMETER_FOR_REQUEST_HEADER:
        return AllocationResult_INVALID_PARAMETER_FOR_REQUEST_HEADER;
    }
    return AllocationResult_UNDEFINED_ERROR;
}

iox_ClientSendResult clientSendResult(const iox::popo::ClientSendError value) noexcept
{
    switch (value)
    {
    case ClientSendError::NO_CONNECT_REQUESTED:
        return ClientSendResult_NO_CONNECT_REQUESTED;
    case ClientSendError::SERVER_NOT_AVAILABLE:
        return ClientSendResult_SERVER_NOT_AVAILABLE;
    case ClientSendError::INVALID_REQUEST:
        return ClientSendResult_INVALID_REQUEST;
    }
    return ClientSendResult_UNDEFINED_ERROR;
}

iox_ServerSendResult serverSendResult(const iox::popo::ServerSendError value) noexcept
{
    switch (value)
    {
    case ServerSendError::NOT_OFFERED:
        return ServerSendResult_NOT_OFFERED;
    case ServerSendError::CLIENT_NOT_AVAILABLE:
        return ServerSendResult_CLIENT_NOT_AVAILABLE;
    case ServerSendError::INVALID_RESPONSE:
        return ServerSendResult_INVALID_RESPONSE;
    }
    return ServerSendResult_UNDEFINED_ERROR;
}

iox_WaitSetResult waitSetResult(const iox::popo::WaitSetError value) noexcept
{
    switch (value)
    {
    case WaitSetError::WAIT_SET_FULL:
        return WaitSetResult_WAIT_SET_FULL;
    case WaitSetError::ALREADY_ATTACHED:
        return WaitSetResult_ALREADY_ATTACHED;
    }
    return WaitSetResult_UNDEFINED_ERROR;
}

iox_ListenerResult listenerResult(const iox::popo::ListenerError value) noexcept
{
    switch (value)
    {
    case ListenerError::EVENT_ALREADY_ATTACHED:
        return ListenerResult_EVENT_ALREADY_ATTACHED;
    case ListenerError::LISTENER_FULL:
        return ListenerResult_LISTENER_FULL;
    case ListenerError::EMPTY_EVENT_CALLBACK:
        return ListenerResult_EMPTY_EVENT_CALLBACK;
    }
    return ListenerResult_UNDEFINED_ERROR;
}

iox_ConsumerTooSlowPolicy consumerTooSlowPolicy(const iox::popo::ConsumerTooSlowPolicy policy) noexcept
{
    switch (policy)
    {
    case ConsumerTooSlowPolicy::WAIT_FOR_CONSUMER:
        return ConsumerTooSlowPolicy_WAIT_FOR_CONSUMER;
    case ConsumerTooSlowPolicy::DISCARD_OLDEST_DATA:
        return ConsumerTooSlowPolicy_DISCARD_OLDEST_DATA;
    }
    return ConsumerTooSlowPolicy_DISCARD_OLDEST_DATA;
}
iox_ConsumerTooSlowPolicy subscriberTooSlowPolicy(const iox::popo::ConsumerTooSlowPolicy policy) noexcept
{
    return consumerTooSlowPolicy(policy);
}
iox_QueueFullPolicy queueFullPolicy(const iox::popo::QueueFullPolicy policy) noexcept
{
    switch (policy)
    {
    case QueueFullPolicy::BLOCK_PRODUCER:
        return QueueFullPolicy_BLOCK_PRODUCER;
    case QueueFullPolicy::DISCARD_OLDEST_DATA:
        return QueueFullPolicy_DISCARD_OLDEST_DATA;
    }
    return QueueFullPolicy_DISCARD_OLDEST_DATA;
}

iox_ClientEvent clientEvent(const iox::popo::ClientEvent value) noexcept
{
    switch (value)
    {
    case ClientEvent::RESPONSE_RECEIVED:
        return ClientEvent_RESPONSE_RECEIVED;
    }

    return ClientEvent_RESPONSE_RECEIVED;
}

iox_ClientState clientState(const iox::popo::ClientState value) noexcept
{
    switch (value)
    {
    case ClientState::HAS_RESPONSE:
        return ClientState_HAS_RESPONSE;
    }

    return ClientState_HAS_RESPONSE;
}

iox_ServerEvent serverEvent(const iox::popo::ServerEvent value) noexcept
{
    switch (value)
    {
    case ServerEvent::REQUEST_RECEIVED:
        return ServerEvent_REQUEST_RECEIVED;
    }

    return ServerEvent_REQUEST_RECEIVED;
}

iox_ServerState serverState(const iox::popo::ServerState value) noexcept
{
    switch (value)
    {
    case ServerState::HAS_REQUEST:
        return ServerState_HAS_REQUEST;
    }

    return ServerState_HAS_REQUEST;
}

iox_ConnectionState connectionState(const iox::ConnectionState value) noexcept
{
    switch (value)
    {
    case iox::ConnectionState::CONNECTED:
        return ConnectionState_CONNECTED;
    case iox::ConnectionState::NOT_CONNECTED:
        return ConnectionState_NOT_CONNECTED;
    case iox::ConnectionState::CONNECT_REQUESTED:
        return ConnectionState_CONNECT_REQUESTED;
    case iox::ConnectionState::DISCONNECT_REQUESTED:
        return ConnectionState_DISCONNECT_REQUESTED;
    case iox::ConnectionState::WAIT_FOR_OFFER:
        return ConnectionState_WAIT_FOR_OFFER;
    }

    return ConnectionState_NOT_CONNECTED;
}

iox_ServerRequestResult serverRequestResult(const iox::popo::ServerRequestResult value) noexcept
{
    switch (value)
    {
    case iox::popo::ServerRequestResult::TOO_MANY_REQUESTS_HELD_IN_PARALLEL:
        return ServerRequestResult_TOO_MANY_REQUESTS_HELD_IN_PARALLEL;
    case iox::popo::ServerRequestResult::NO_PENDING_REQUESTS:
        return ServerRequestResult_NO_PENDING_REQUESTS;
    case iox::popo::ServerRequestResult::UNDEFINED_CHUNK_RECEIVE_ERROR:
        return ServerRequestResult_UNDEFINED_CHUNK_RECEIVE_ERROR;
    case iox::popo::ServerRequestResult::NO_PENDING_REQUESTS_AND_SERVER_DOES_NOT_OFFER:
        return ServerRequestResult_NO_PENDING_REQUESTS_AND_SERVER_DOES_NOT_OFFER;
    }

    return ServerRequestResult_SUCCESS;
}
} // namespace cpp2c
