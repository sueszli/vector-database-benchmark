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

#include "client/client_desktop.h"

#include "base/logging.h"
#include "base/task_runner.h"
#include "base/audio/audio_player.h"
#include "base/codec/audio_decoder_opus.h"
#include "base/codec/cursor_decoder.h"
#include "base/codec/video_decoder.h"
#include "base/codec/webm_file_writer.h"
#include "base/codec/webm_video_encoder.h"
#include "base/desktop/mouse_cursor.h"
#include "client/desktop_control_proxy.h"
#include "client/desktop_window.h"
#include "client/desktop_window_proxy.h"
#include "client/config_factory.h"
#include "common/desktop_session_constants.h"

namespace client {

namespace {

//--------------------------------------------------------------------------------------------------
int calculateFps(int last_fps, const std::chrono::milliseconds& duration, int64_t count)
{
    static const double kAlpha = 0.1;
    return static_cast<int>(
        (kAlpha * ((1000.0 / static_cast<double>(duration.count())) * static_cast<double>(count))) +
        ((1.0 - kAlpha) * static_cast<double>(last_fps)));
}

//--------------------------------------------------------------------------------------------------
size_t calculateAvgSize(size_t last_avg_size, size_t bytes)
{
    static const double kAlpha = 0.1;
    return static_cast<size_t>(
        (kAlpha * static_cast<double>(bytes)) +
        ((1.0 - kAlpha) * static_cast<double>(last_avg_size)));
}

//--------------------------------------------------------------------------------------------------
const char* audioEncodingToString(proto::AudioEncoding encoding)
{
    switch (encoding)
    {
        case proto::AUDIO_ENCODING_OPUS:
            return "AUDIO_ENCODING_OPUS";

        case proto::AUDIO_ENCODING_RAW:
            return "AUDIO_ENCODING_RAW";

        default:
            return "AUDIO_ENCODING_UNKNOWN";
    }
}

//--------------------------------------------------------------------------------------------------
const char* videoEncodingToString(proto::VideoEncoding encoding)
{
    switch (encoding)
    {
        case proto::VIDEO_ENCODING_VP8:
            return "VIDEO_ENCODING_VP8";

        case proto::VIDEO_ENCODING_VP9:
            return "VIDEO_ENCODING_VP9";

        case proto::VIDEO_ENCODING_ZSTD:
            return "VIDEO_ENCODING_ZSTD";

        default:
            return "VIDEO_ENCODING_UNKNOWN";
    }
}

//--------------------------------------------------------------------------------------------------
const char* videoErrorCodeToString(proto::VideoErrorCode error_code)
{
    switch (error_code)
    {
        case proto::VIDEO_ERROR_CODE_OK:
            return "VIDEO_ERROR_CODE_OK";

        case proto::VIDEO_ERROR_CODE_TEMPORARY:
            return "VIDEO_ERROR_CODE_TEMPORARY";

        case proto::VIDEO_ERROR_CODE_PERMANENT:
            return "VIDEO_ERROR_CODE_PERMANENT";

        case proto::VIDEO_ERROR_CODE_PAUSED:
            return "VIDEO_ERROR_CODE_PAUSED";

        default:
            return "VIDEO_ERROR_CODE_UNKNOWN";
    }
}

} // namespace

//--------------------------------------------------------------------------------------------------
ClientDesktop::ClientDesktop(std::shared_ptr<base::TaskRunner> io_task_runner)
    : Client(io_task_runner),
      desktop_control_proxy_(std::make_shared<DesktopControlProxy>(io_task_runner, this)),
      incoming_message_(std::make_unique<proto::HostToClient>()),
      outgoing_message_(std::make_unique<proto::ClientToHost>())
{
    LOG(LS_INFO) << "Ctor";
}

//--------------------------------------------------------------------------------------------------
ClientDesktop::~ClientDesktop()
{
    LOG(LS_INFO) << "Dtor";
    desktop_control_proxy_->dettach();
}

//--------------------------------------------------------------------------------------------------
void ClientDesktop::setDesktopWindow(std::shared_ptr<DesktopWindowProxy> desktop_window_proxy)
{
    LOG(LS_INFO) << "Desktop window installed";
    desktop_window_proxy_ = std::move(desktop_window_proxy);
}

//--------------------------------------------------------------------------------------------------
void ClientDesktop::onSessionStarted(const base::Version& peer_version)
{
    LOG(LS_INFO) << "Desktop session started";

    start_time_ = Clock::now();
    started_ = true;

    input_event_filter_.setSessionType(sessionType());
    desktop_window_proxy_->showWindow(desktop_control_proxy_, peer_version);

    clipboard_monitor_ = std::make_unique<common::ClipboardMonitor>();
    clipboard_monitor_->start(ioTaskRunner(), this);

    audio_player_ = base::AudioPlayer::create();
}

//--------------------------------------------------------------------------------------------------
void ClientDesktop::onSessionMessageReceived(uint8_t /* channel_id */, const base::ByteArray& buffer)
{
    incoming_message_->Clear();

    if (!base::parse(buffer, incoming_message_.get()))
    {
        LOG(LS_ERROR) << "Invalid message from host";
        return;
    }

    if (incoming_message_->has_video_packet() || incoming_message_->has_cursor_shape())
    {
        if (incoming_message_->has_video_packet())
            readVideoPacket(incoming_message_->video_packet());

        if (incoming_message_->has_cursor_shape())
            readCursorShape(incoming_message_->cursor_shape());
    }
    else if (incoming_message_->has_audio_packet())
    {
        readAudioPacket(incoming_message_->audio_packet());
    }
    else if (incoming_message_->has_cursor_position())
    {
        readCursorPosition(incoming_message_->cursor_position());
    }
    else if (incoming_message_->has_clipboard_event())
    {
        readClipboardEvent(incoming_message_->clipboard_event());
    }
    else if (incoming_message_->has_capabilities())
    {
        readCapabilities(incoming_message_->capabilities());
    }
    else if (incoming_message_->has_extension())
    {
        readExtension(incoming_message_->extension());
    }
    else
    {
        // Unknown messages are ignored.
        LOG(LS_ERROR) << "Unhandled message from host";
    }
}

//--------------------------------------------------------------------------------------------------
void ClientDesktop::onSessionMessageWritten(uint8_t /* channel_id */, size_t pending)
{
    if (pending >= 2)
        input_event_filter_.setNetworkOverflow(true);
    else
        input_event_filter_.setNetworkOverflow(false);
}

//--------------------------------------------------------------------------------------------------
void ClientDesktop::onClipboardEvent(const proto::ClipboardEvent& event)
{
    std::optional<proto::ClipboardEvent> out_event = input_event_filter_.sendClipboardEvent(event);
    if (!out_event.has_value())
        return;

    outgoing_message_->Clear();
    outgoing_message_->mutable_clipboard_event()->CopyFrom(*out_event);
    sendMessage(proto::HOST_CHANNEL_ID_SESSION, *outgoing_message_);
}

//--------------------------------------------------------------------------------------------------
void ClientDesktop::setDesktopConfig(const proto::DesktopConfig& desktop_config)
{
    LOG(LS_INFO) << "setDesktopConfig called";
    desktop_config_ = desktop_config;

    ConfigFactory::fixupDesktopConfig(&desktop_config_);

    // If the session is not already running, then we do not need to send the configuration.
    if (!started_)
    {
        LOG(LS_INFO) << "Session not started yet";
        return;
    }

    if (!(desktop_config_.flags() & proto::ENABLE_CURSOR_SHAPE))
    {
        LOG(LS_INFO) << "Cursor shape disabled";
        cursor_decoder_.reset();
    }

    input_event_filter_.setClipboardEnabled(desktop_config_.flags() & proto::ENABLE_CLIPBOARD);

    outgoing_message_->Clear();
    outgoing_message_->mutable_config()->CopyFrom(desktop_config_);

    LOG(LS_INFO) << "Send new config to host";
    sendMessage(proto::HOST_CHANNEL_ID_SESSION, *outgoing_message_);
}

//--------------------------------------------------------------------------------------------------
void ClientDesktop::setCurrentScreen(const proto::Screen& screen)
{
    LOG(LS_INFO) << "Current screen changed: " << screen.id();

    outgoing_message_->Clear();
    proto::DesktopExtension* extension = outgoing_message_->mutable_extension();

    extension->set_name(common::kSelectScreenExtension);
    extension->set_data(screen.SerializeAsString());

    sendMessage(proto::HOST_CHANNEL_ID_SESSION, *outgoing_message_);
}

//--------------------------------------------------------------------------------------------------
void ClientDesktop::setPreferredSize(int width, int height)
{
    LOG(LS_INFO) << "Preferred size changed: " << width << "x" << height;

    outgoing_message_->Clear();

    proto::PreferredSize preferred_size;
    preferred_size.set_width(width);
    preferred_size.set_height(height);

    proto::DesktopExtension* extension = outgoing_message_->mutable_extension();

    extension->set_name(common::kPreferredSizeExtension);
    extension->set_data(preferred_size.SerializeAsString());

    sendMessage(proto::HOST_CHANNEL_ID_SESSION, *outgoing_message_);
}

//--------------------------------------------------------------------------------------------------
void ClientDesktop::setVideoPause(bool enable)
{
    LOG(LS_INFO) << "Video pause changed: " << enable;

    if (enable)
    {
        ++video_pause_count_;
    }
    else
    {
        if (!video_pause_count_)
            return;
        ++video_resume_count_;
    }

    outgoing_message_->Clear();

    proto::Pause pause;
    pause.set_enable(enable);

    proto::DesktopExtension* extension = outgoing_message_->mutable_extension();

    extension->set_name(common::kVideoPauseExtension);
    extension->set_data(pause.SerializeAsString());

    sendMessage(proto::HOST_CHANNEL_ID_SESSION, *outgoing_message_);
}

//--------------------------------------------------------------------------------------------------
void ClientDesktop::setAudioPause(bool enable)
{
    LOG(LS_INFO) << "Audio pause changed: " << enable;

    if (enable)
    {
        ++audio_pause_count_;
    }
    else
    {
        if (!audio_pause_count_)
            return;
        ++audio_resume_count_;
    }

    outgoing_message_->Clear();

    proto::Pause pause;
    pause.set_enable(enable);

    proto::DesktopExtension* extension = outgoing_message_->mutable_extension();

    extension->set_name(common::kAudioPauseExtension);
    extension->set_data(pause.SerializeAsString());

    sendMessage(proto::HOST_CHANNEL_ID_SESSION, *outgoing_message_);
}

//--------------------------------------------------------------------------------------------------
void ClientDesktop::setVideoRecording(bool enable, const std::filesystem::path& file_path)
{
    proto::VideoRecording video_recording;

    if (enable)
    {
        LOG(LS_INFO) << "Video recording enabled (file: " << file_path << ")";

        video_recording.set_action(proto::VideoRecording::ACTION_STARTED);

        webm_file_writer_ = std::make_unique<base::WebmFileWriter>(file_path, computerName());
        webm_video_encoder_ = std::make_unique<base::WebmVideoEncoder>();

        webm_video_encode_timer_ = std::make_unique<base::WaitableTimer>(
            base::WaitableTimer::Type::REPEATED, ioTaskRunner());
        webm_video_encode_timer_->start(std::chrono::milliseconds(60), [this]()
        {
            if (!webm_video_encoder_ || !webm_file_writer_ || !desktop_frame_)
                return;

            proto::VideoPacket packet;

            if (webm_video_encoder_->encode(*desktop_frame_, &packet))
                webm_file_writer_->addVideoPacket(packet);
        });
    }
    else
    {
        LOG(LS_INFO) << "Video recording disabled";

        video_recording.set_action(proto::VideoRecording::ACTION_STOPPED);

        webm_video_encode_timer_.reset();
        webm_video_encoder_.reset();
        webm_file_writer_.reset();
    }

    outgoing_message_->Clear();
    proto::DesktopExtension* extension = outgoing_message_->mutable_extension();

    extension->set_name(common::kVideoRecordingExtension);
    extension->set_data(video_recording.SerializeAsString());

    sendMessage(proto::HOST_CHANNEL_ID_SESSION, *outgoing_message_);
}

//--------------------------------------------------------------------------------------------------
void ClientDesktop::onKeyEvent(const proto::KeyEvent& event)
{
    std::optional<proto::KeyEvent> out_event = input_event_filter_.keyEvent(event);
    if (!out_event.has_value())
        return;

    outgoing_message_->Clear();
    outgoing_message_->mutable_key_event()->CopyFrom(*out_event);

    sendMessage(proto::HOST_CHANNEL_ID_SESSION, *outgoing_message_);
}

//--------------------------------------------------------------------------------------------------
void ClientDesktop::onTextEvent(const proto::TextEvent& event)
{
    std::optional<proto::TextEvent> out_event = input_event_filter_.textEvent(event);
    if (!out_event.has_value())
        return;

    outgoing_message_->Clear();
    outgoing_message_->mutable_text_event()->CopyFrom(*out_event);

    sendMessage(proto::HOST_CHANNEL_ID_SESSION, *outgoing_message_);
}

//--------------------------------------------------------------------------------------------------
void ClientDesktop::onMouseEvent(const proto::MouseEvent& event)
{
    std::optional<proto::MouseEvent> out_event = input_event_filter_.mouseEvent(event);
    if (!out_event.has_value())
        return;

    outgoing_message_->Clear();
    outgoing_message_->mutable_mouse_event()->CopyFrom(*out_event);

    sendMessage(proto::HOST_CHANNEL_ID_SESSION, *outgoing_message_);
}

//--------------------------------------------------------------------------------------------------
void ClientDesktop::onPowerControl(proto::PowerControl::Action action)
{
    if (sessionType() != proto::SESSION_TYPE_DESKTOP_MANAGE)
    {
        LOG(LS_INFO) << "Power control supported only for desktop manage session";
        return;
    }

    outgoing_message_->Clear();
    proto::DesktopExtension* extension = outgoing_message_->mutable_extension();

    proto::PowerControl power_control;
    power_control.set_action(action);

    extension->set_name(common::kPowerControlExtension);
    extension->set_data(power_control.SerializeAsString());

    sendMessage(proto::HOST_CHANNEL_ID_SESSION, *outgoing_message_);
}

//--------------------------------------------------------------------------------------------------
void ClientDesktop::onRemoteUpdate()
{
    outgoing_message_->Clear();
    outgoing_message_->mutable_extension()->set_name(common::kRemoteUpdateExtension);
    sendMessage(proto::HOST_CHANNEL_ID_SESSION, *outgoing_message_);
}

//--------------------------------------------------------------------------------------------------
void ClientDesktop::onSystemInfoRequest(const proto::system_info::SystemInfoRequest& request)
{
    outgoing_message_->Clear();
    proto::DesktopExtension* extension = outgoing_message_->mutable_extension();
    extension->set_name(common::kSystemInfoExtension);
    extension->set_data(request.SerializeAsString());

    sendMessage(proto::HOST_CHANNEL_ID_SESSION, *outgoing_message_);
}

//--------------------------------------------------------------------------------------------------
void ClientDesktop::onTaskManager(const proto::task_manager::ClientToHost& message)
{
    outgoing_message_->Clear();
    proto::DesktopExtension* extension = outgoing_message_->mutable_extension();
    extension->set_name(common::kTaskManagerExtension);
    extension->set_data(message.SerializeAsString());

    sendMessage(proto::HOST_CHANNEL_ID_SESSION, *outgoing_message_);
}

//--------------------------------------------------------------------------------------------------
void ClientDesktop::onMetricsRequest()
{
    TimePoint current_time = Clock::now();

    if (fps_time_ != TimePoint())
    {
        std::chrono::milliseconds fps_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(current_time - fps_time_);
        fps_ = calculateFps(fps_, fps_duration, fps_frame_count_);
    }
    else
    {
        fps_ = 0;
    }

    fps_time_ = current_time;
    fps_frame_count_ = 0;

    std::chrono::seconds session_duration =
        std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time_);

    DesktopWindow::Metrics metrics;
    metrics.duration = session_duration;
    metrics.total_rx = totalRx();
    metrics.total_tx = totalTx();
    metrics.speed_rx = speedRx();
    metrics.speed_tx = speedTx();

    if (min_video_packet_ != std::numeric_limits<size_t>::max())
        metrics.min_video_packet = min_video_packet_;

    metrics.max_video_packet = max_video_packet_;
    metrics.avg_video_packet = avg_video_packet_;
    metrics.video_packet_count = video_packet_count_;
    metrics.video_pause_count = video_pause_count_;
    metrics.video_resume_count = video_resume_count_;

    if (min_audio_packet_ != std::numeric_limits<size_t>::max())
        metrics.min_audio_packet = min_audio_packet_;

    metrics.max_audio_packet = max_audio_packet_;
    metrics.avg_audio_packet = avg_audio_packet_;
    metrics.audio_packet_count = audio_packet_count_;
    metrics.audio_pause_count = audio_pause_count_;
    metrics.audio_resume_count = audio_resume_count_;

    metrics.video_capturer_type = video_capturer_type_;
    metrics.fps = fps_;
    metrics.send_mouse = input_event_filter_.sendMouseCount();
    metrics.drop_mouse = input_event_filter_.dropMouseCount();
    metrics.send_key   = input_event_filter_.sendKeyCount();
    metrics.send_text  = input_event_filter_.sendTextCount();
    metrics.read_clipboard = input_event_filter_.readClipboardCount();
    metrics.send_clipboard = input_event_filter_.sendClipboardCount();
    metrics.cursor_shape_count = cursor_shape_count_;
    metrics.cursor_pos_count   = cursor_pos_count_;

    if (cursor_decoder_)
    {
        metrics.cursor_cached = cursor_decoder_->cachedCursors();
        metrics.cursor_taken_from_cache = cursor_decoder_->takenCursorsFromCache();
    }

    desktop_window_proxy_->setMetrics(metrics);
}

//--------------------------------------------------------------------------------------------------
void ClientDesktop::readCapabilities(const proto::DesktopCapabilities& capabilities)
{
    LOG(LS_INFO) << "Capabilities received";

    // We notify the window about changes in the list of extensions and video encodings.
    // A window can disable/enable some of its capabilities in accordance with this information.
    desktop_window_proxy_->setCapabilities(capabilities);

    // If current video encoding not supported.
    if (!(capabilities.video_encodings() & static_cast<uint32_t>(desktop_config_.video_encoding())))
    {
        LOG(LS_ERROR) << "Current video encoding not supported";

        // We tell the window about the need to change the encoding.
        desktop_window_proxy_->configRequired();
    }
    else
    {
        // Everything is fine, we send the current configuration.
        setDesktopConfig(desktop_config_);
    }
}

//--------------------------------------------------------------------------------------------------
void ClientDesktop::readVideoPacket(const proto::VideoPacket& packet)
{
    proto::VideoErrorCode error_code = packet.error_code();
    if (error_code != proto::VIDEO_ERROR_CODE_OK)
    {
        LOG(LS_ERROR) << "Video error detected: " << videoErrorCodeToString(error_code);
        desktop_window_proxy_->setFrameError(error_code);
        return;
    }

    if (video_encoding_ != packet.encoding())
    {
        LOG(LS_INFO) << "Video encoding changed from: " << videoEncodingToString(video_encoding_)
                     << " to: " << videoEncodingToString(packet.encoding());

        video_decoder_ = base::VideoDecoder::create(packet.encoding());
        video_encoding_ = packet.encoding();
    }

    if (!video_decoder_)
    {
        LOG(LS_ERROR) << "Video decoder not initialized";
        return;
    }

    if (packet.has_format())
    {
        const proto::VideoPacketFormat& format = packet.format();
        base::Size video_size(format.video_rect().width(), format.video_rect().height());
        base::Size screen_size = video_size;

        static const int kMaxValue = std::numeric_limits<uint16_t>::max();

        if (video_size.width()  <= 0 || video_size.width()  >= kMaxValue ||
            video_size.height() <= 0 || video_size.height() >= kMaxValue)
        {
            LOG(LS_ERROR) << "Wrong video frame size: "
                          << video_size.width() << "x" << video_size.height();
            return;
        }

        if (format.has_screen_size())
        {
            screen_size = base::Size(
                format.screen_size().width(), format.screen_size().height());

            if (screen_size.width() <= 0 || screen_size.width() >= kMaxValue ||
                screen_size.height() <= 0 || screen_size.height() >= kMaxValue)
            {
                LOG(LS_ERROR) << "Wrong screen size: "
                              << screen_size.width() << "x" << screen_size.height();
                return;
            }
        }

        video_capturer_type_ = format.capturer_type();

        LOG(LS_INFO) << "New video size: " << video_size.width() << "x" << video_size.height();
        LOG(LS_INFO) << "New screen size: " << screen_size.width() << "x" << screen_size.height();
        LOG(LS_INFO) << "New video capturer: " << video_capturer_type_;

        desktop_frame_ = desktop_window_proxy_->allocateFrame(video_size);
        desktop_window_proxy_->setFrame(screen_size, desktop_frame_);
    }

    if (!desktop_frame_)
    {
        LOG(LS_ERROR) << "The desktop frame is not initialized";
        return;
    }

    if (!video_decoder_->decode(packet, desktop_frame_.get()))
    {
        LOG(LS_ERROR) << "The video packet could not be decoded";
        return;
    }

    ++video_packet_count_;
    ++fps_frame_count_;

    size_t packet_size = packet.ByteSizeLong();

    avg_video_packet_ = calculateAvgSize(avg_video_packet_, packet_size);
    min_video_packet_ = std::min(min_video_packet_, packet_size);
    max_video_packet_ = std::max(max_video_packet_, packet_size);

    desktop_window_proxy_->drawFrame();
}

//--------------------------------------------------------------------------------------------------
void ClientDesktop::readAudioPacket(const proto::AudioPacket& packet)
{
    if (webm_file_writer_)
        webm_file_writer_->addAudioPacket(packet);

    if (!audio_player_)
    {
        LOG(LS_ERROR) << "Audio packet received but audio player not initialized";
        return;
    }

    if (packet.encoding() != audio_encoding_)
    {
        audio_decoder_ = base::AudioDecoder::create(packet.encoding());
        audio_encoding_ = packet.encoding();

        LOG(LS_INFO) << "Audio encoding changed to: " << audioEncodingToString(audio_encoding_);
    }

    if (!audio_decoder_)
    {
        LOG(LS_INFO) << "Audio decoder not initialized now";
        return;
    }

    size_t packet_size = packet.ByteSizeLong();

    avg_audio_packet_ = calculateAvgSize(avg_audio_packet_, packet_size);
    min_audio_packet_ = std::min(min_audio_packet_, packet_size);
    max_audio_packet_ = std::max(max_audio_packet_, packet_size);

    ++audio_packet_count_;

    std::unique_ptr<proto::AudioPacket> decoded_packet = audio_decoder_->decode(packet);
    if (decoded_packet)
        audio_player_->addPacket(std::move(decoded_packet));
}

//--------------------------------------------------------------------------------------------------
void ClientDesktop::readCursorShape(const proto::CursorShape& cursor_shape)
{
    if (sessionType() != proto::SESSION_TYPE_DESKTOP_MANAGE)
    {
        LOG(LS_ERROR) << "Cursor shape received not session type not desktop manage";
        return;
    }

    if (!(desktop_config_.flags() & proto::ENABLE_CURSOR_SHAPE))
    {
        LOG(LS_ERROR) << "Cursor shape received not disabled in client";
        return;
    }

    ++cursor_shape_count_;

    if (!cursor_decoder_)
    {
        LOG(LS_INFO) << "Cursor decoder initialization";
        cursor_decoder_ = std::make_unique<base::CursorDecoder>();
    }

    std::shared_ptr<base::MouseCursor> mouse_cursor = cursor_decoder_->decode(cursor_shape);
    if (!mouse_cursor)
        return;

    desktop_window_proxy_->setMouseCursor(mouse_cursor);
}

//--------------------------------------------------------------------------------------------------
void ClientDesktop::readCursorPosition(const proto::CursorPosition& cursor_position)
{
    if (!(desktop_config_.flags() & proto::CURSOR_POSITION))
    {
        LOG(LS_ERROR) << "Cursor position received not disabled in client";
        return;
    }

    ++cursor_pos_count_;

    desktop_window_proxy_->setCursorPosition(cursor_position);
}

//--------------------------------------------------------------------------------------------------
void ClientDesktop::readClipboardEvent(const proto::ClipboardEvent& event)
{
    if (!clipboard_monitor_)
    {
        LOG(LS_ERROR) << "Clipboard received not disabled in client";
        return;
    }

    std::optional<proto::ClipboardEvent> out_event = input_event_filter_.readClipboardEvent(event);
    if (!out_event.has_value())
        return;

    clipboard_monitor_->injectClipboardEvent(*out_event);
}

//--------------------------------------------------------------------------------------------------
void ClientDesktop::readExtension(const proto::DesktopExtension& extension)
{
    if (extension.name() == common::kTaskManagerExtension)
    {
        proto::task_manager::HostToClient message;

        if (!message.ParseFromString(extension.data()))
        {
            LOG(LS_ERROR) << "Unable to parse task manager extension data";
            return;
        }

        desktop_window_proxy_->setTaskManager(message);
    }
    else if (extension.name() == common::kSelectScreenExtension)
    {
        proto::ScreenList screen_list;

        if (!screen_list.ParseFromString(extension.data()))
        {
            LOG(LS_ERROR) << "Unable to parse select screen extension data";
            return;
        }

        LOG(LS_INFO) << "Screen list received";
        LOG(LS_INFO) << "Primary screen: " << screen_list.primary_screen();
        LOG(LS_INFO) << "Current screen: " << screen_list.current_screen();

        for (int i = 0; i < screen_list.screen_size(); ++i)
        {
            const proto::Screen& screen = screen_list.screen(i);
            const proto::Point& dpi = screen.dpi();
            const proto::Point& position = screen.position();
            const proto::Resolution& resolution = screen.resolution();

            LOG(LS_INFO) << "Screen #" << i << ": id=" << screen.id()
                         << " title=" << screen.title()
                         << " dpi=" << dpi.x() << "x" << dpi.y()
                         << " pos=" << position.x() << "x" << position.y()
                         << " res=" << resolution.width() << "x" << resolution.height();
        }

        desktop_window_proxy_->setScreenList(screen_list);
    }
    else if (extension.name() == common::kScreenTypeExtension)
    {
        proto::ScreenType screen_type;

        if (!screen_type.ParseFromString(extension.data()))
        {
            LOG(LS_ERROR) << "Unable to parse screen type extension data";
            return;
        }

        LOG(LS_INFO) << "Screen type received (type=" << screen_type.type()
                     << " name=" << screen_type.name() << ")";

        desktop_window_proxy_->setScreenType(screen_type);
    }
    else if (extension.name() == common::kSystemInfoExtension)
    {
        proto::system_info::SystemInfo system_info;

        if (!system_info.ParseFromString(extension.data()))
        {
            LOG(LS_ERROR) << "Unable to parse system info extension data";
            return;
        }

        desktop_window_proxy_->setSystemInfo(system_info);
    }
    else
    {
        LOG(LS_ERROR) << "Unknown extension: " << extension.name();
    }
}

} // namespace client
