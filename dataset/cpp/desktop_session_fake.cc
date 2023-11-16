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

#include "host/desktop_session_fake.h"

#include "base/logging.h"
#include "base/task_runner.h"

namespace host {

//--------------------------------------------------------------------------------------------------
DesktopSessionFake::DesktopSessionFake(
    std::shared_ptr<base::TaskRunner> /* task_runner */, Delegate* delegate)
    : delegate_(delegate)
{
    LOG(LS_INFO) << "Ctor";
    DCHECK(delegate_);
}

//--------------------------------------------------------------------------------------------------
DesktopSessionFake::~DesktopSessionFake()
{
    LOG(LS_INFO) << "Dtor";
}

//--------------------------------------------------------------------------------------------------
void DesktopSessionFake::start()
{
    LOG(LS_INFO) << "Start called for fake session";

    if (delegate_)
    {
        delegate_->onDesktopSessionStarted();
    }
    else
    {
        LOG(LS_ERROR) << "Invalid delegate";
    }
}

//--------------------------------------------------------------------------------------------------
void DesktopSessionFake::stop()
{
    LOG(LS_INFO) << "Stop called for fake session";
    delegate_ = nullptr;
}

//--------------------------------------------------------------------------------------------------
void DesktopSessionFake::control(proto::internal::DesktopControl::Action action)
{
    LOG(LS_INFO) << "CONTROL with action: " << controlActionToString(action);

    switch (action)
    {
        case proto::internal::DesktopControl::ENABLE:
            if (delegate_)
                delegate_->onScreenCaptureError(proto::VIDEO_ERROR_CODE_TEMPORARY);
            break;

        default:
            break;
    }
}

//--------------------------------------------------------------------------------------------------
void DesktopSessionFake::configure(const Config& /* config */)
{
    // Nothing
}

//--------------------------------------------------------------------------------------------------
void DesktopSessionFake::selectScreen(const proto::Screen& /* screen */)
{
    // Nothing
}

//--------------------------------------------------------------------------------------------------
void DesktopSessionFake::captureScreen()
{
    if (delegate_)
        delegate_->onScreenCaptureError(proto::VIDEO_ERROR_CODE_TEMPORARY);
}

//--------------------------------------------------------------------------------------------------
void DesktopSessionFake::setScreenCaptureFps(int /* fps */)
{
    // Nothing
}

//--------------------------------------------------------------------------------------------------
void DesktopSessionFake::injectKeyEvent(const proto::KeyEvent& /* event */)
{
    // Nothing
}

//--------------------------------------------------------------------------------------------------
void DesktopSessionFake::injectTextEvent(const proto::TextEvent& /* event */)
{
    // Nothing
}

//--------------------------------------------------------------------------------------------------
void DesktopSessionFake::injectMouseEvent(const proto::MouseEvent& /* event */)
{
    // Nothing
}

//--------------------------------------------------------------------------------------------------
void DesktopSessionFake::injectClipboardEvent(const proto::ClipboardEvent& /* event */)
{
    // Nothing
}

} // namespace host
