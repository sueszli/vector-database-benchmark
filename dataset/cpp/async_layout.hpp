// async_layout.hpp
/*
  neogfx C++ App/Game Engine
  Copyright (c) 2021 Leigh Johnston.  All Rights Reserved.
  
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

#pragma once

#include <neogfx/neogfx.hpp>
#include <vector>
#include <neogfx/gui/widget/timer.hpp>
#include <neogfx/gui/window/i_window.hpp>
#include <neogfx/gui/layout/i_async_layout.hpp>

namespace neogfx
{
    class async_layout : public i_async_layout
    {
    private:
        struct entry
        {
            destroyed_flag destroyed;
            i_widget* widget;
            bool validated;
            std::optional<pause_rendering> pauseRendering;

            entry(destroyed_flag&& destroyed, i_widget* widget) :
                destroyed{ std::move(destroyed) }, widget{ widget }, validated{ false }
            {
                if (!widget->is_root())
                    pauseRendering.emplace(widget->root());
            }
            entry(entry&&) = default;
            entry& operator=(entry&&) = default;
        };
        typedef std::vector<entry> entry_queue;
    public:
        async_layout();
    public:
        bool exists(i_widget& aWidget) const noexcept override;
        bool defer_layout(i_widget& aWidget) override;
        void validate(i_widget& aWidget) override;
        void invalidate(i_widget& aWidget) override;
    private:
        std::optional<entry_queue::const_iterator> pending(i_widget& aWidget) const noexcept;
        std::optional<entry_queue::iterator> pending(i_widget& aWidget) noexcept;
        std::optional<entry_queue::const_iterator> processing(i_widget& aWidget) const noexcept;
        std::optional<entry_queue::iterator> processing(i_widget& aWidget) noexcept;
        void process();
    private:
        neolib::callback_timer iTimer;
        entry_queue iPending;
        entry_queue iProcessing;
    };
}