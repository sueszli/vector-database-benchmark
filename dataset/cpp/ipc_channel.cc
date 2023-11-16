//
// Aspia Project
// Copyright (C) 2016-2023 Dmitry Chapyshev <dmitry@aspia.ru>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//

#include "base/ipc/ipc_channel.h"

#include "base/location.h"
#include "base/logging.h"
#include "base/ipc/ipc_channel_proxy.h"
#include "base/message_loop/message_loop.h"
#include "base/message_loop/message_pump_asio.h"
#include "base/strings/unicode.h"

#include <asio/read.hpp>
#include <asio/write.hpp>

#include <functional>

#if defined(OS_WIN)
#include "base/win/scoped_object.h"
#include <Psapi.h>
#endif // defined(OS_WIN)

namespace base {

namespace {

const uint32_t kMaxMessageSize = 16 * 1024 * 1024; // 16MB

#if defined(OS_POSIX)
const char16_t kLocalSocketPrefix[] = u"/tmp/aspia_";
#endif // defined(OS_POSIX)

#if defined(OS_WIN)

const char16_t kPipeNamePrefix[] = u"\\\\.\\pipe\\aspia.";
const DWORD kConnectTimeout = 5000; // ms

//--------------------------------------------------------------------------------------------------
ProcessId clientProcessIdImpl(HANDLE pipe_handle)
{
    ULONG process_id = kNullProcessId;

    if (!GetNamedPipeClientProcessId(pipe_handle, &process_id))
    {
        PLOG(LS_ERROR) << "GetNamedPipeClientProcessId failed";
        return kNullProcessId;
    }

    return process_id;
}

//--------------------------------------------------------------------------------------------------
ProcessId serverProcessIdImpl(HANDLE pipe_handle)
{
    ULONG process_id = kNullProcessId;

    if (!GetNamedPipeServerProcessId(pipe_handle, &process_id))
    {
        PLOG(LS_ERROR) << "GetNamedPipeServerProcessId failed";
        return kNullProcessId;
    }

    return process_id;
}

//--------------------------------------------------------------------------------------------------
SessionId clientSessionIdImpl(HANDLE pipe_handle)
{
    ULONG session_id = kInvalidSessionId;

    if (!GetNamedPipeClientSessionId(pipe_handle, &session_id))
    {
        PLOG(LS_ERROR) << "GetNamedPipeClientSessionId failed";
        return kInvalidSessionId;
    }

    return session_id;
}

//--------------------------------------------------------------------------------------------------
SessionId serverSessionIdImpl(HANDLE pipe_handle)
{
    ULONG session_id = kInvalidSessionId;

    if (!GetNamedPipeServerSessionId(pipe_handle, &session_id))
    {
        PLOG(LS_ERROR) << "GetNamedPipeServerSessionId failed";
        return kInvalidSessionId;
    }

    return session_id;
}

#endif // defined(OS_WIN)

} // namespace

//--------------------------------------------------------------------------------------------------
IpcChannel::IpcChannel()
    : stream_(MessageLoop::current()->pumpAsio()->ioContext()),
      proxy_(new IpcChannelProxy(MessageLoop::current()->taskRunner(), this))
{
    // Nothing
}

//--------------------------------------------------------------------------------------------------
IpcChannel::IpcChannel(std::u16string_view channel_name, Stream&& stream)
    : channel_name_(channel_name),
      stream_(std::move(stream)),
      proxy_(new IpcChannelProxy(MessageLoop::current()->taskRunner(), this)),
      is_connected_(true)
{
    LOG(LS_INFO) << "Ctor";

#if defined(OS_WIN)
    peer_process_id_ = clientProcessIdImpl(stream_.native_handle());
    peer_session_id_ = clientSessionIdImpl(stream_.native_handle());
#endif // defined(OS_WIN)
}

//--------------------------------------------------------------------------------------------------
IpcChannel::~IpcChannel()
{
    LOG(LS_INFO) << "Dtor";
    DCHECK_CALLED_ON_VALID_THREAD(thread_checker_);

    proxy_->willDestroyCurrentChannel();
    proxy_ = nullptr;

    listener_ = nullptr;

    disconnect();
}

//--------------------------------------------------------------------------------------------------
std::shared_ptr<IpcChannelProxy> IpcChannel::channelProxy()
{
    return proxy_;
}

//--------------------------------------------------------------------------------------------------
void IpcChannel::setListener(Listener* listener)
{
    LOG(LS_INFO) << "Listener changed (" << (listener != nullptr) << ")";
    listener_ = listener;
}

//--------------------------------------------------------------------------------------------------
bool IpcChannel::connect(std::u16string_view channel_id)
{
    DCHECK_CALLED_ON_VALID_THREAD(thread_checker_);

    if (channel_id.empty())
    {
        LOG(LS_ERROR) << "Empty channel id";
        return false;
    }

    channel_name_ = channelName(channel_id);

#if defined(OS_WIN)
    const DWORD flags = SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION | FILE_FLAG_OVERLAPPED;

    win::ScopedHandle handle;

    while (true)
    {
        handle.reset(CreateFileW(reinterpret_cast<const wchar_t*>(channel_name_.c_str()),
                                 GENERIC_WRITE | GENERIC_READ,
                                 0,
                                 nullptr,
                                 OPEN_EXISTING,
                                 flags,
                                 nullptr));
        if (handle.isValid())
            break;

        DWORD error_code = GetLastError();

        if (error_code != ERROR_PIPE_BUSY)
        {
            LOG(LS_ERROR) << "Failed to connect to the named pipe: "
                          << SystemError::toString(error_code);
            return false;
        }

        if (!WaitNamedPipeW(reinterpret_cast<const wchar_t*>(channel_name_.c_str()),
                            kConnectTimeout))
        {
            PLOG(LS_ERROR) << "WaitNamedPipeW failed";
            return false;
        }
    }

    std::error_code error_code;
    stream_.assign(handle.release(), error_code);
    if (error_code)
        return false;

    peer_process_id_ = serverProcessIdImpl(stream_.native_handle());
    peer_session_id_ = serverSessionIdImpl(stream_.native_handle());

    is_connected_ = true;
    return true;
#else // defined(OS_WIN)
    asio::local::stream_protocol::endpoint endpoint(base::local8BitFromUtf16(channel_name_));
    std::error_code error_code;
    stream_.connect(endpoint, error_code);
    if (error_code)
    {
        LOG(LS_ERROR) << "Unable to connect: " << base::utf16FromLocal8Bit(error_code.message());
        return false;
    }

    is_connected_ = true;
    return true;
#endif // !defined(OS_WIN)
}

//--------------------------------------------------------------------------------------------------
void IpcChannel::disconnect()
{
    DCHECK_CALLED_ON_VALID_THREAD(thread_checker_);

    if (!is_connected_)
        return;

    LOG(LS_INFO) << "disconnect called";
    is_connected_ = false;

    std::error_code ignored_code;

    stream_.cancel(ignored_code);
    stream_.close(ignored_code);
}

//--------------------------------------------------------------------------------------------------
bool IpcChannel::isConnected() const
{
    DCHECK_CALLED_ON_VALID_THREAD(thread_checker_);
    return is_connected_;
}

//--------------------------------------------------------------------------------------------------
bool IpcChannel::isPaused() const
{
    DCHECK_CALLED_ON_VALID_THREAD(thread_checker_);
    return is_paused_;
}

//--------------------------------------------------------------------------------------------------
void IpcChannel::pause()
{
    DCHECK_CALLED_ON_VALID_THREAD(thread_checker_);
    is_paused_ = true;
}

//--------------------------------------------------------------------------------------------------
void IpcChannel::resume()
{
    DCHECK_CALLED_ON_VALID_THREAD(thread_checker_);

    if (!is_connected_ || !is_paused_)
        return;

    LOG(LS_INFO) << "resume called";
    is_paused_ = false;

    // If we have a message that was received before the pause command.
    if (read_size_)
        onMessageReceived();

    DCHECK_EQ(read_size_, 0);
    doReadMessage();
}

//--------------------------------------------------------------------------------------------------
void IpcChannel::send(ByteArray&& buffer)
{
    DCHECK_CALLED_ON_VALID_THREAD(thread_checker_);

    const bool schedule_write = write_queue_.empty();

    // Add the buffer to the queue for sending.
    write_queue_.emplace(std::move(buffer));

    if (schedule_write)
        doWrite();
}

//--------------------------------------------------------------------------------------------------
std::filesystem::path IpcChannel::peerFilePath() const
{
#if defined(OS_WIN)
    win::ScopedHandle process(
        OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, peer_process_id_));
    if (!process.isValid())
    {
        PLOG(LS_ERROR) << "OpenProcess failed";
        return std::filesystem::path();
    }

    wchar_t buffer[MAX_PATH] = { 0 };

    if (!GetModuleFileNameExW(process.get(), nullptr, buffer, static_cast<DWORD>(std::size(buffer))))
    {
        PLOG(LS_ERROR) << "GetModuleFileNameExW failed";
        return std::filesystem::path();
    }

    return buffer;
#else // defined(OS_WIN)
    NOTIMPLEMENTED();
    return std::filesystem::path();
#endif // !defined(OS_WIN)
}

//--------------------------------------------------------------------------------------------------
// static
std::u16string IpcChannel::channelName(std::u16string_view channel_id)
{
#if defined(OS_WIN)
    std::u16string name(kPipeNamePrefix);
    name.append(channel_id);
    return name;
#else // defined(OS_WIN)
    std::u16string name(kLocalSocketPrefix);
    name.append(channel_id);
    return name;
#endif // !defined(OS_WIN)
}

//--------------------------------------------------------------------------------------------------
void IpcChannel::onErrorOccurred(const Location& location, const std::error_code& error_code)
{
    if (error_code == asio::error::operation_aborted)
        return;

    LOG(LS_ERROR) << "Error in IPC channel '" << channel_name_ << "': "
                  << utf16FromLocal8Bit(error_code.message())
                  << " (code: " << error_code.value()
                  << ", location: " << location.toString() << ")";

    disconnect();

    if (listener_)
    {
        listener_->onIpcDisconnected();
        listener_ = nullptr;
    }
    else
    {
        LOG(LS_ERROR) << "No listener";
    }
}

//--------------------------------------------------------------------------------------------------
void IpcChannel::doWrite()
{
    write_size_ = static_cast<uint32_t>(write_queue_.front().size());

    if (!write_size_ || write_size_ > kMaxMessageSize)
    {
        onErrorOccurred(FROM_HERE, asio::error::message_size);
        return;
    }

    asio::async_write(stream_, asio::buffer(&write_size_, sizeof(write_size_)),
        [this](const std::error_code& error_code, size_t bytes_transferred)
    {
        if (error_code)
        {
            onErrorOccurred(FROM_HERE, error_code);
            return;
        }

        DCHECK_EQ(bytes_transferred, sizeof(write_size_));
        DCHECK(!write_queue_.empty());

        const ByteArray& buffer = write_queue_.front();

        // Send the buffer to the recipient.
        asio::async_write(stream_, asio::buffer(buffer.data(), buffer.size()),
            [this](const std::error_code& error_code, size_t bytes_transferred)
        {
            if (error_code)
            {
                onErrorOccurred(FROM_HERE, error_code);
                return;
            }

            DCHECK_EQ(bytes_transferred, write_size_);
            DCHECK(!write_queue_.empty());

            // Delete the sent message from the queue.
            write_queue_.pop();

            // If the queue is not empty, then we send the following message.
            if (write_queue_.empty() && !proxy_->reloadWriteQueue(&write_queue_))
                return;

            doWrite();
        });
    });
}

//--------------------------------------------------------------------------------------------------
void IpcChannel::doReadMessage()
{
    asio::async_read(stream_, asio::buffer(&read_size_, sizeof(read_size_)),
        [this](const std::error_code& error_code, size_t bytes_transferred)
    {
        if (error_code)
        {
            onErrorOccurred(FROM_HERE, error_code);
            return;
        }

        DCHECK_EQ(bytes_transferred, sizeof(read_size_));

        if (!read_size_ || read_size_ > kMaxMessageSize)
        {
            onErrorOccurred(FROM_HERE, asio::error::message_size);
            return;
        }

        if (read_buffer_.capacity() < read_size_)
        {
            read_buffer_.clear();
            read_buffer_.reserve(read_size_);
        }

        read_buffer_.resize(read_size_);

        asio::async_read(stream_, asio::buffer(read_buffer_.data(), read_buffer_.size()),
            [this](const std::error_code& error_code, size_t bytes_transferred)
        {
            if (error_code)
            {
                onErrorOccurred(FROM_HERE, error_code);
                return;
            }

            DCHECK_EQ(bytes_transferred, read_size_);

            if (is_paused_)
                return;

            onMessageReceived();

            if (is_paused_)
                return;

            DCHECK_EQ(read_size_, 0);
            doReadMessage();
        });
    });
}

//--------------------------------------------------------------------------------------------------
void IpcChannel::onMessageReceived()
{
    if (listener_)
    {
        listener_->onIpcMessageReceived(read_buffer_);
    }
    else
    {
        LOG(LS_ERROR) << "No listener";
    }

    read_size_ = 0;
}

} // namespace base
