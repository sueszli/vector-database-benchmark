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

#include "common/clipboard_win.h"

#include "base/logging.h"
#include "base/strings/string_util.h"
#include "base/strings/unicode.h"
#include "base/win/message_window.h"
#include "base/win/scoped_clipboard.h"
#include "base/win/scoped_hglobal.h"

namespace common {

//--------------------------------------------------------------------------------------------------
ClipboardWin::ClipboardWin() = default;

//--------------------------------------------------------------------------------------------------
ClipboardWin::~ClipboardWin()
{
    if (!window_)
        return;

    RemoveClipboardFormatListener(window_->hwnd());
    window_.reset();
}

//--------------------------------------------------------------------------------------------------
void ClipboardWin::init()
{
    if (window_)
        return;

    window_ = std::make_unique<base::win::MessageWindow>();

    if (!window_->create(std::bind(&ClipboardWin::onMessage,
                                   this,
                                   std::placeholders::_1, std::placeholders::_2,
                                   std::placeholders::_3, std::placeholders::_4)))
    {
        LOG(LS_ERROR) << "Couldn't create clipboard window.";
        return;
    }

    if (!AddClipboardFormatListener(window_->hwnd()))
    {
        PLOG(LS_ERROR) << "AddClipboardFormatListener failed";
        return;
    }
}

//--------------------------------------------------------------------------------------------------
void ClipboardWin::setData(const std::string& data)
{
    if (!window_)
        return;

    std::wstring text;
    if (!base::utf8ToWide(base::replaceLfByCrLf(data), &text))
    {
        LOG(LS_ERROR) << "Couldn't convert data to unicode";
        return;
    }

    base::win::ScopedClipboard clipboard;
    if (!clipboard.init(window_->hwnd()))
    {
        PLOG(LS_ERROR) << "Couldn't open the clipboard";
        return;
    }

    clipboard.empty();

    if (data.empty())
        return;

    HGLOBAL text_global = GlobalAlloc(GMEM_MOVEABLE, (text.size() + 1) * sizeof(wchar_t));
    if (!text_global)
    {
        PLOG(LS_ERROR) << "GlobalAlloc failed";
        return;
    }

    LPWSTR text_global_locked = reinterpret_cast<LPWSTR>(GlobalLock(text_global));
    if (!text_global_locked)
    {
        PLOG(LS_ERROR) << "GlobalLock failed";
        GlobalFree(text_global);
        return;
    }

    memcpy(text_global_locked, text.data(), text.size() * sizeof(wchar_t));
    text_global_locked[text.size()] = 0;

    GlobalUnlock(text_global);

    clipboard.setData(CF_UNICODETEXT, text_global);
}

//--------------------------------------------------------------------------------------------------
bool ClipboardWin::onMessage(UINT message, WPARAM /* wParam */, LPARAM /* lParam */, LRESULT& result)
{
    switch (message)
    {
        case WM_CLIPBOARDUPDATE:
            onClipboardUpdate();
            break;

        default:
            return false;
    }

    result = 0;
    return true;
}

//--------------------------------------------------------------------------------------------------
void ClipboardWin::onClipboardUpdate()
{
    if (!IsClipboardFormatAvailable(CF_UNICODETEXT))
        return;

    std::string data;

    // Add a scope, so that we keep the clipboard open for as short a time as possible.
    {
        base::win::ScopedClipboard clipboard;

        if (!clipboard.init(window_->hwnd()))
        {
            PLOG(LS_ERROR) << "Couldn't open the clipboard";
            return;
        }

        HGLOBAL text_global = clipboard.data(CF_UNICODETEXT);
        if (!text_global)
        {
            PLOG(LS_ERROR) << "Couldn't get data from the clipboard";
            return;
        }

        {
            base::win::ScopedHGLOBAL<wchar_t> text_lock(text_global);
            if (!text_lock.get())
            {
                PLOG(LS_ERROR) << "Couldn't lock clipboard data";
                return;
            }

            if (!base::wideToUtf8(text_lock.get(), &data))
            {
                LOG(LS_ERROR) << "Couldn't convert data to utf8";
                return;
            }
        }
    }

    if (!data.empty())
        onData(base::replaceCrLfByLf(data));
}

} // namespace common
