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

#include "host/desktop_session_proxy.h"

#include "base/logging.h"
#include "base/environment.h"
#include "base/strings/string_number_conversions.h"

#include <thread>

namespace host {

//--------------------------------------------------------------------------------------------------
DesktopSessionProxy::DesktopSessionProxy()
{
    std::string default_fps_string;
    if (base::Environment::get("ASPIA_DEFAULT_FPS", &default_fps_string))
    {
        int default_fps = kDefaultScreenCaptureFps;

        if (base::stringToInt(default_fps_string, &default_fps))
        {
            LOG(LS_INFO) << "Default FPS specified by environment variable";

            if (default_fps < 1 || default_fps > 60)
            {
                LOG(LS_INFO) << "Environment variable contains an incorrect default FPS: " << default_fps;
            }
            else
            {
                default_capture_fps_ = default_fps;
            }
        }
    }

    std::string min_fps_string;
    if (base::Environment::get("ASPIA_MIN_FPS", &min_fps_string))
    {
        int min_fps = kMinScreenCaptureFps;

        if (base::stringToInt(min_fps_string, &min_fps))
        {
            LOG(LS_INFO) << "Minimum FPS specified by environment variable";

            if (min_fps < 1 || min_fps > 60)
            {
                LOG(LS_INFO) << "Environment variable contains an incorrect minimum FPS: " << min_fps;
            }
            else
            {
                min_capture_fps_ = min_fps;
            }
        }
    }

    bool max_fps_from_env = false;
    std::string max_fps_string;
    if (base::Environment::get("ASPIA_MAX_FPS", &max_fps_string))
    {
        int max_fps = kMaxScreenCaptureFpsHighEnd;

        if (base::stringToInt(max_fps_string, &max_fps))
        {
            LOG(LS_INFO) << "Maximum FPS specified by environment variable";

            if (max_fps < 1 || max_fps > 60)
            {
                LOG(LS_INFO) << "Environment variable contains an incorrect maximum FPS: " << max_fps;
            }
            else
            {
                max_capture_fps_ = max_fps;
                max_fps_from_env = true;
            }
        }
    }

    if (!max_fps_from_env)
    {
        uint32_t threads = std::thread::hardware_concurrency();
        if (threads <= 2)
        {
            LOG(LS_INFO) << "Low-end CPU detected. Maximum capture FPS: " << kMaxScreenCaptureFpsLowEnd;
            max_capture_fps_ = kMaxScreenCaptureFpsLowEnd;
        }
    }
}

//--------------------------------------------------------------------------------------------------
DesktopSessionProxy::~DesktopSessionProxy()
{
    DCHECK(!desktop_session_);
}

//--------------------------------------------------------------------------------------------------
void DesktopSessionProxy::control(proto::internal::DesktopControl::Action action)
{
    if (desktop_session_)
        desktop_session_->control(action);
}

//--------------------------------------------------------------------------------------------------
void DesktopSessionProxy::configure(const DesktopSession::Config& config)
{
    if (is_paused_)
        return;

    if (desktop_session_)
        desktop_session_->configure(config);
}

//--------------------------------------------------------------------------------------------------
void DesktopSessionProxy::selectScreen(const proto::Screen& screen)
{
    if (is_paused_)
        return;

    if (desktop_session_)
        desktop_session_->selectScreen(screen);
}

//--------------------------------------------------------------------------------------------------
void DesktopSessionProxy::captureScreen()
{
    if (is_paused_)
        return;

    if (desktop_session_)
        desktop_session_->captureScreen();
}

//--------------------------------------------------------------------------------------------------
void DesktopSessionProxy::setScreenCaptureFps(int fps)
{
    screen_capture_fps_ = fps;

    if (desktop_session_)
        desktop_session_->setScreenCaptureFps(fps);
}

//--------------------------------------------------------------------------------------------------
int DesktopSessionProxy::defaultScreenCaptureFps() const
{
    return default_capture_fps_;
}

//--------------------------------------------------------------------------------------------------
int DesktopSessionProxy::minScreenCaptureFps() const
{
    return min_capture_fps_;
}

//--------------------------------------------------------------------------------------------------
int DesktopSessionProxy::maxScreenCaptureFps() const
{
    return max_capture_fps_;
}

//--------------------------------------------------------------------------------------------------
int DesktopSessionProxy::screenCaptureFps() const
{
    return screen_capture_fps_;
}

//--------------------------------------------------------------------------------------------------
void DesktopSessionProxy::injectKeyEvent(const proto::KeyEvent& event)
{
    if (is_keyboard_locked_ || is_paused_)
        return;

    if (desktop_session_)
        desktop_session_->injectKeyEvent(event);
}

//--------------------------------------------------------------------------------------------------
void DesktopSessionProxy::injectTextEvent(const proto::TextEvent& event)
{
    if (is_keyboard_locked_ || is_paused_)
        return;

    if (desktop_session_)
        desktop_session_->injectTextEvent(event);
}

//--------------------------------------------------------------------------------------------------
void DesktopSessionProxy::injectMouseEvent(const proto::MouseEvent& event)
{
    if (is_mouse_locked_ || is_paused_)
        return;

    if (desktop_session_)
        desktop_session_->injectMouseEvent(event);
}

//--------------------------------------------------------------------------------------------------
void DesktopSessionProxy::injectClipboardEvent(const proto::ClipboardEvent& event)
{
    if (is_paused_)
        return;

    if (desktop_session_)
        desktop_session_->injectClipboardEvent(event);
}

//--------------------------------------------------------------------------------------------------
void DesktopSessionProxy::setMouseLock(bool enable)
{
    is_mouse_locked_ = enable;
}

//--------------------------------------------------------------------------------------------------
void DesktopSessionProxy::setKeyboardLock(bool enable)
{
    is_keyboard_locked_ = enable;
}

//--------------------------------------------------------------------------------------------------
void DesktopSessionProxy::setPaused(bool enable)
{
    is_paused_ = enable;
}

//--------------------------------------------------------------------------------------------------
void DesktopSessionProxy::attachAndStart(DesktopSession* desktop_session)
{
    LOG(LS_INFO) << "Desktop session attach";

    desktop_session_ = desktop_session;
    DCHECK(desktop_session_);

    desktop_session_->setScreenCaptureFps(screen_capture_fps_);
    desktop_session_->start();
}

//--------------------------------------------------------------------------------------------------
void DesktopSessionProxy::stopAndDettach()
{
    LOG(LS_INFO) << "Desktop session dettach";

    if (desktop_session_)
    {
        desktop_session_->stop();
        desktop_session_ = nullptr;
    }
}

} // namespace host
