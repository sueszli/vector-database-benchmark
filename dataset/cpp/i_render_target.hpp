// i_render_target.hpp
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
#include <neogfx/core/device_metrics.hpp>
#include <neogfx/gfx/primitives.hpp>

namespace neogfx
{
    class i_texture;
    class i_rendering_context;

    enum class render_target_type
    {
        Surface,
        Texture
    };

    typedef int pixel_format_t; // todo: enum?

    class i_render_target : public i_device_metrics
    {
    public:
        declare_event(target_activating)
        declare_event(target_activated)
        declare_event(target_deactivating)
        declare_event(target_deactivated)
    public:
        struct failed_to_create_framebuffer : std::runtime_error { failed_to_create_framebuffer(std::string const& aReason) : std::runtime_error("neogfx::i_render_target::failed_to_create_framebuffer: Failed to create frame buffer, reason: " + aReason) {} };
        struct already_active : std::logic_error { already_active() : std::logic_error("neogfx::i_render_target::already_active") {} };
        struct not_active : std::logic_error { not_active() : std::logic_error("neogfx::i_render_target::not_active") {} };
        struct logical_coordinates_not_specified : std::logic_error { logical_coordinates_not_specified() : std::logic_error("neogfx::i_render_target::logical_coordinates_not_specified") {} };
    public:
        virtual render_target_type target_type() const = 0;
        virtual void* target_handle() const = 0;
        virtual void* target_device_handle() const = 0;
        virtual pixel_format_t pixel_format() const = 0;
        virtual const i_texture& target_texture() const = 0;
        virtual point target_origin() const = 0;
        virtual size target_extents() const = 0;
    public:
        virtual neogfx::logical_coordinate_system logical_coordinate_system() const = 0;
        virtual void set_logical_coordinate_system(neogfx::logical_coordinate_system aSystem) = 0;
        virtual neogfx::logical_coordinates logical_coordinates() const = 0;
        virtual void set_logical_coordinates(const neogfx::logical_coordinates& aCoordinates) = 0;
    public:
        virtual rect_i32 viewport() const = 0;
        virtual rect_i32 set_viewport(const rect_i32& aViewport) const = 0;
    public:
        virtual bool target_active() const = 0;
        virtual void activate_target() const = 0;
        virtual void deactivate_target() const = 0;
    public:
        virtual neogfx::color_space color_space() const = 0;
        virtual color read_pixel(const point& aPosition) const = 0;
    public:
        virtual std::unique_ptr<i_rendering_context> create_graphics_context(blending_mode aBlendingMode = blending_mode::Default) const = 0; // todo: use ref_ptr
    };

    class scoped_render_target
    {
    public:
        scoped_render_target();
        scoped_render_target(const i_render_target& aRenderTarget);
        scoped_render_target(const i_rendering_context& aRenderingContext);
        ~scoped_render_target();
    private:
        const i_render_target* iRenderTarget;
        const i_render_target* iPreviouslyActivatedTarget;
    };
}