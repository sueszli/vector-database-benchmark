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

#include "host/desktop_session.h"

namespace host {

//--------------------------------------------------------------------------------------------------
// static
const char* DesktopSession::controlActionToString(proto::internal::DesktopControl::Action action)
{
    switch (action)
    {
        case proto::internal::DesktopControl::ENABLE:
            return "ENABLE";

        case proto::internal::DesktopControl::DISABLE:
            return "DISABLE";

        case proto::internal::DesktopControl::LOGOFF:
            return "LOGOFF";

        case proto::internal::DesktopControl::LOCK:
            return "LOCK";

        default:
            return "UNKNOWN";
    }
}

} // namespace host
