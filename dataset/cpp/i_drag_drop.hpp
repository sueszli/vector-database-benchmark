// i_drag_drop.hpp
/*
  neogfx C++ App/Game Engine
  Copyright (c) 2020 Leigh Johnston.  All Rights Reserved.
  
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
#include <neolib/core/i_vector.hpp>
#include <neogfx/core/event.hpp>
#include <neogfx/hid/i_surface.hpp>
#include <neogfx/gui/widget/i_widget.hpp>
#include <neogfx/gui/widget/item_index.hpp>
#include <neogfx/gui/widget/i_item_presentation_model.hpp>
#include <neogfx/game/i_ecs.hpp>

namespace neogfx
{
    typedef uuid drag_drop_object_type_id;

    class i_drag_drop_source;

    class i_drag_drop_object
    {
    public:
        virtual ~i_drag_drop_object() = default;
    public:
        virtual i_drag_drop_source& source() const = 0;
        virtual drag_drop_object_type_id ddo_type() const = 0;
    public:
        virtual bool can_render() const = 0;
        virtual size render_extents() const = 0;
        virtual void render(i_graphics_context& aGc, point const& aPosition = {}) const = 0;
    public:
        static const drag_drop_object_type_id otid()
        {
            static drag_drop_object_type_id sId{ 0x00000000, 0x0000, 0x0000, 0x0000, { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 } };
            return sId;
        }
    };

    class i_drag_drop_file_list : public i_drag_drop_object
    {
    public:
        virtual neolib::i_vector<i_string> const& file_paths() const = 0;
    public:
        static const drag_drop_object_type_id otid()
        {
            static drag_drop_object_type_id sId{ 0xfaa77f8e, 0xfabc, 0x413c, 0xacdb, { 0x5b, 0xc1, 0x74, 0xc5, 0xd2, 0xbf } };
            return sId;
        }
    };

    class i_drag_drop_item : public i_drag_drop_object
    {
    public:
        virtual i_item_presentation_model const& presentation_model() const = 0;
        virtual item_presentation_model_index const& index() const = 0;
    public:
        static const drag_drop_object_type_id otid()
        {
            static drag_drop_object_type_id sId{ 0x53f26bc6, 0x1b0, 0x4c19, 0x955b, { 0x1c, 0x6d, 0xa, 0x3f, 0x3b, 0xaf } };
            return sId;
        }
    };

    class i_drag_drop_entity : public i_drag_drop_object
    {
    public:
        virtual game::i_ecs const& ecs() const = 0;
        virtual game::entity_id entity() const = 0;
    public:
        static const drag_drop_object_type_id otid()
        {
            static drag_drop_object_type_id sId{ 0x7d9e2908, 0x7b2a, 0x45e2, 0xa97, { 0xa0, 0x18, 0x6d, 0x97, 0x7e, 0x83 } };
            return sId;
        }
    };

    struct failed_drag_drop_registration : std::logic_error { failed_drag_drop_registration(std::string const& aReason) : std::logic_error{ "neogfx::failed_drag_drop_registration: " + aReason } {} };
    struct failed_drag_drop_unregistration : std::logic_error { failed_drag_drop_unregistration(std::string const& aReason) : std::logic_error{ "neogfx::failed_drag_drop_unregistration: " + aReason } {} };
    struct drag_drop_already_active : std::logic_error { drag_drop_already_active() : std::logic_error{ "neogfx::drag_drop_already_active" } {} };
    struct drag_drop_not_active : std::logic_error { drag_drop_not_active() : std::logic_error{ "neogfx::drag_drop_not_active" } {} };
    struct no_drag_drop_event_monitor : std::logic_error { no_drag_drop_event_monitor() : std::logic_error{ "neogfx::no_drag_drop_event_monitor" } {} };

    class i_drag_drop_source
    {
    public:
        declare_event(dragging_object, i_drag_drop_object const&)
        declare_event(dragging_cancelled, i_drag_drop_object const&)
        declare_event(object_dropped_on_target, i_drag_drop_object const&, i_drag_drop_target&)
    public:
        virtual ~i_drag_drop_source() = default;
    public:
        virtual bool drag_drop_source_enabled() const = 0;
        virtual void enable_drag_drop_source(bool aEnable = true) = 0;
        virtual bool drag_drop_active() const = 0;
        virtual i_drag_drop_object const& object_being_dragged() const = 0;
        virtual void start_drag_drop(i_drag_drop_object const& aObject) = 0;
        virtual void cancel_drag_drop() = 0;
        virtual void end_drag_drop(i_drag_drop_target& aTarget) = 0;
    public:
        virtual point const& drag_drop_tracking_position() const = 0;
        virtual i_ref_ptr<i_widget> const& drag_drop_widget() const = 0;
        virtual void set_drag_drop_widget(i_ref_ptr<i_widget> const& aWidget) = 0;
    public:
        virtual i_widget& drag_drop_event_monitor() const = 0;
        virtual void monitor_drag_drop_events(i_widget& aWidget) = 0;
        virtual void stop_monitoring_drag_drop_events() = 0;
    public:
        void enable_drag_drop_source(i_widget& aWidget)
        {
            enable_drag_drop_source(true);
            monitor_drag_drop_events(aWidget);
        }
    };

    struct drag_drop_target_not_a_widget : std::logic_error { drag_drop_target_not_a_widget() : std::logic_error{ "neogfx::drag_drop_target_not_a_widget" } {} };

    enum class drop_operation : uint32_t
    {
        None = 0x00000000,
        Copy = 0x00000001,
        Move = 0x00000002,
        Link = 0x00000003
    };

    class i_drag_drop_target
    {
    public:
        declare_event(object_acceptable, i_drag_drop_object const&, optional_point const&, drop_operation&)
        declare_event(object_dropped, i_drag_drop_object const&, optional_point const&)
    public:
        virtual ~i_drag_drop_target() = default;
    public:
        virtual bool drag_drop_target_enabled() const = 0;
        virtual void enable_drag_drop_target(bool aEnable = true) = 0;
    public:
        virtual bool can_accept(i_drag_drop_object const& aObject, optional_point const& aDropPosition = {}) const = 0;
        virtual drop_operation accepted_as(i_drag_drop_object const& aObject, optional_point const& aDropPosition = {}) const = 0;
        virtual bool accept(i_drag_drop_object const& aObject, optional_point const& aDropPosition = {}) = 0;
    public:
        virtual bool is_widget() const = 0;
        virtual i_widget const& as_widget() const = 0;
        virtual i_widget& as_widget() = 0;
    };

    struct drag_drop_target_not_found : std::logic_error { drag_drop_target_not_found() : std::logic_error{ "neogfx::drag_drop_target_not_found" } {} };

    class i_drag_drop : public i_service
    {
    public:
        declare_event(source_registered, i_drag_drop_source&)
        declare_event(source_unregistered, i_drag_drop_source&)
        declare_event(target_registered, i_drag_drop_target&)
        declare_event(target_unregistered, i_drag_drop_target&)
    public:
        virtual ~i_drag_drop() = default;
    public:
        virtual void register_source(i_drag_drop_source& aSource) = 0;
        virtual void unregister_source(i_drag_drop_source& aSource) = 0;
        virtual void register_target(i_drag_drop_target& aTarget) = 0;
        virtual void unregister_target(i_drag_drop_target& aTarget) = 0;
    public:
        virtual bool is_target_for(i_drag_drop_object const& aObject) const = 0;
        virtual bool is_target_at(i_drag_drop_object const& aObject, point const& aPosition) const = 0;
        virtual i_drag_drop_target& target_for(i_drag_drop_object const& aObject) const = 0;
        virtual i_drag_drop_target& target_at(i_drag_drop_object const& aObject, point const& aPosition) const = 0;
    public:
        static uuid const& iid() { static uuid const sIid{ 0x393fd9c4, 0x6db8, 0x4c04, 0x87f6, { 0x39, 0x87, 0x6a, 0x30, 0x35, 0xd4 } }; return sIid; }
    };
}