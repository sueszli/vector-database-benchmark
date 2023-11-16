// i_display.hpp
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

#pragma once

#include <neogfx/neogfx.hpp>
#include <neolib/core/i_enum.hpp>
#include <neogfx/core/device_metrics.hpp>
#include <neogfx/core/units.hpp>
#include <neogfx/gfx/color.hpp>
#include <neogfx/hid/video_mode.hpp>
#include <neogfx/gui/window/window_bits.hpp>

namespace neogfx
{
    enum class subpixel_format : uint32_t
    {
        None,
        RGBHorizontal,
        BGRHorizontal,
        RGBVertical,
        BGRVertical
    };
}

begin_declare_enum(neogfx::subpixel_format)
declare_enum_string(neogfx::subpixel_format, None)
declare_enum_string(neogfx::subpixel_format, RGBHorizontal)
declare_enum_string(neogfx::subpixel_format, BGRHorizontal)
declare_enum_string(neogfx::subpixel_format, RGBVertical)
declare_enum_string(neogfx::subpixel_format, BGRVertical)
end_declare_enum(neogfx::subpixel_format)

namespace neogfx
{
    class i_display : public i_units_context
    {
    public:
        struct failed_to_get_monitor_dpi : std::runtime_error { failed_to_get_monitor_dpi() : std::runtime_error("neogfx::i_display::failed_to_get_monitor_dpi") {} };
        struct fullscreen_not_active : std::logic_error { fullscreen_not_active() : std::logic_error("neogfx::i_display::fullscreen_not_active: Fullscreen not currently active") {} };
        struct failed_to_enter_fullscreen : std::runtime_error { failed_to_enter_fullscreen(std::string const& aReason) : 
            std::runtime_error("neogfx::i_displayy::failed_to_enter_fullscreen: Failed to enter fullscreen, reason: " + aReason) {} };
        struct failed_to_leave_fullscreen : std::runtime_error { failed_to_leave_fullscreen(std::string const& aReason) :
            std::runtime_error("neogfx::i_displayy::failed_to_leave_fullscreen: Failed to leave fullscreen, reason: " + aReason) {} };
    public:
        virtual ~i_display() = default;
    public:
        virtual uint32_t index() const = 0;
    public:
        virtual const i_device_metrics& metrics() const = 0;
        virtual void update_dpi() = 0;
    public:
        virtual bool is_fullscreen() const = 0;
        virtual const video_mode& fullscreen_video_mode() const = 0;
        virtual void enter_fullscreen(const video_mode& aVideoMode) = 0;
        virtual void leave_fullscreen() = 0;
        virtual neogfx::rect rect() const = 0;
        virtual neogfx::rect desktop_rect() const = 0;
        virtual window_placement default_window_placement() const = 0;
        virtual neogfx::color_space color_space() const = 0;
        virtual color read_pixel(const point& aPosition) const = 0;
    public:
        virtual neogfx::subpixel_format subpixel_format() const = 0;
    };
}