// clipboard.cpp
/*
  neogfx C++ App/Game Engine
  Copyright (c) 2015, 2020 Leigh Johnston.  All Rights Reserved.
  
  This program is free software: you can redistribute it and / or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <neogfx/neogfx.hpp>
#include <neogfx/app/clipboard.hpp>
#include "native/i_native_clipboard.hpp"

namespace neogfx
{
    clipboard::clipboard(i_native_clipboard& aSystemClipboard) : iSystemClipboard(aSystemClipboard), iActiveSink(nullptr)
    {
    }

    bool clipboard::sink_active() const
    {
        return iActiveSink != nullptr;
    }

    i_clipboard_sink& clipboard::active_sink()
    {
        if (!sink_active())
            throw no_active_sink();
        return *iActiveSink;
    }

    void clipboard::activate(i_clipboard_sink& aSink)
    {
        if (sink_active())
            deactivate(*iActiveSink);
        iActiveSink = &aSink;
        SinkActivated.trigger();
    }

    void clipboard::deactivate(i_clipboard_sink& aSink)
    {
        if (iActiveSink != &aSink)
            throw sink_not_active();
        iActiveSink = nullptr;
        SinkDeactivated.trigger();
    }

    bool clipboard::has_text() const
    {
        return iSystemClipboard.has_text();
    }

    i_string const& clipboard::text() const
    {
       return iSystemClipboard.text();
    }

    void clipboard::set_text(i_string const& aText)
    {
        iSystemClipboard.set_text(aText);
    }

    bool clipboard::has_image() const
    {
        return iSystemClipboard.has_image();
    }

    neogfx::image clipboard::image() const
    {
        return iSystemClipboard.image();
    }

    void clipboard::set_image(const neogfx::image& aImage)
    {
        iSystemClipboard.set_image(aImage);
    }

    void clipboard::cut()
    {
        if (sink_active())
            active_sink().cut(*this);
    }

    void clipboard::copy()
    {
        if (sink_active())
            active_sink().copy(*this);
    }

    void clipboard::paste()
    {
        if (sink_active())
            active_sink().paste(*this);
    }

    void clipboard::delete_selected()
    {
        if (sink_active())
            active_sink().delete_selected();
    }

    void clipboard::select_all()
    {
        if (sink_active())
            active_sink().select_all();
    }
}