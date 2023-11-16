// i_surface.hpp
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
#include <neolib/core/variant.hpp>
#include <neogfx/core/event.hpp>
#include <neogfx/core/i_property.hpp>
#include <neogfx/core/geometrical.hpp>
#include <neogfx/gui/window/window_bits.hpp>
#include <neogfx/gfx/primitives.hpp>
#include <neogfx/hid/mouse.hpp>

namespace neogfx
{
    class i_widget;
    class i_window;
    class i_surface_window;

    enum class surface_type
    {
        Window,
        Touchscreen,
        Paper        // Printing support
    };

    class i_rendering_engine;
    class i_native_surface;

    typedef window_style surface_style;

    class i_surface : public i_device_metrics, public i_units_context, public i_property_owner, public i_reference_counted
    {
    public:
        declare_event(dpi_changed)
        declare_event(rendering)
        declare_event(rendering_finished)
        declare_event(closing)
        declare_event(closed)
    public:
        typedef i_surface abstract_type;
    public:
        struct no_native_surface : std::logic_error { no_native_surface() : std::logic_error("neogfx::i_surface::no_native_surface") {} };
        struct not_a_window : std::logic_error { not_a_window() : std::logic_error("neogfx::i_surface::not_a_window") {} };
    public:
        virtual ~i_surface() = default;
    public:
        virtual i_rendering_engine& rendering_engine() const = 0;
    public:
        virtual bool has_parent_surface() const = 0;
        virtual const i_surface& parent_surface() const = 0;
        virtual i_surface& parent_surface() = 0;
        virtual bool is_owner_of(const i_surface& aChildSurface) const = 0;
    public:
        virtual bool is_strong() const = 0;
        virtual bool is_weak() const = 0;
        virtual bool can_close() const = 0;
        virtual bool is_closed() const = 0;
        virtual void close() = 0;
    public:
        virtual bool is_window() const = 0;
        virtual bool is_nested_window() const = 0;
        virtual const i_surface_window& as_surface_window() const = 0;
        virtual i_surface_window& as_surface_window() = 0;
    public:
        virtual neogfx::surface_type surface_type() const = 0;
        virtual surface_style style() const = 0;
        virtual void set_style(surface_style aStyle) = 0;
        virtual neogfx::logical_coordinate_system logical_coordinate_system() const = 0;
        virtual void set_logical_coordinate_system(neogfx::logical_coordinate_system aSystem) = 0;
        virtual neogfx::logical_coordinates logical_coordinates() const = 0;
        virtual void set_logical_coordinates(const neogfx::logical_coordinates& aCoordinates) = 0;
        virtual double z_order() const = 0;
        virtual void layout_surface() = 0;
        virtual void invalidate_surface(const rect& aInvalidatedRect, bool aInternal = true) = 0;
        virtual bool has_invalidated_area() const = 0;
        virtual const rect& invalidated_area() const = 0;
        virtual rect validate() = 0;
        virtual double rendering_priority() const = 0;
        virtual void render_surface() = 0;
        virtual void pause_rendering() = 0;
        virtual void resume_rendering() = 0;
        virtual bool has_native_surface() const = 0;
        virtual const i_native_surface& native_surface() const = 0;
        virtual i_native_surface& native_surface() = 0;
    public:
        virtual point surface_position() const = 0;
        virtual void move_surface(const point& aPosition) = 0;
        virtual size surface_extents() const = 0;
        virtual void resize_surface(const size& aExtents) = 0;
        virtual double surface_opacity() const = 0;
        virtual void set_surface_opacity(double aOpacity) = 0;
        virtual double surface_transparency() const = 0;
        virtual void set_surface_transparency(double aTransparency) = 0;
    };
}