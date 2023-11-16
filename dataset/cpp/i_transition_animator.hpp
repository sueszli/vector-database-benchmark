// i_transition_animator.hpp
/*
  neogfx C++ App/Game Engine
  Copyright (c) 2018, 2020 Leigh Johnston.  All Rights Reserved.
  
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
#include <neolib/core/i_jar.hpp>
#include <neogfx/core/easing.hpp>

namespace neogfx
{
    typedef neolib::cookie transition_id;

    class i_animator;

    class i_transition : public i_reference_counted
    {
    public:
        struct cannot_apply : std::logic_error { cannot_apply() : std::logic_error{ "neogfx::i_transition::cannot_apply" } {} };
    public:
        typedef i_transition abstract_type;
    public:
        virtual ~i_transition() = default;
    public:
        virtual transition_id id() const = 0;
        virtual i_animator& animator() const = 0;
        virtual bool enabled() const = 0;
        virtual bool disabled() const = 0;
        virtual bool disable_when_finished() const = 0;
        virtual void enable(bool aDisableWhenFinished = false) = 0;
        virtual void disable() = 0;
        virtual easing easing_function() const = 0;
        virtual double duration() const = 0;
        virtual double start_time() const = 0;
        virtual double mix_value() const = 0;
        virtual bool animation_finished() const = 0;
        virtual bool finished() const = 0;
    public:
        virtual bool active() const = 0;
        virtual bool paused() const = 0;
        virtual void pause() = 0;
        virtual void resume() = 0;
    public:
        virtual void clear() = 0;
        virtual void sync(bool aIgnorePrevious = false) = 0;
        virtual void reset(bool aEnable = true, bool aDisableWhenFinished = false, bool aResetStartTime = true) = 0;
        virtual void reset(easing aNewEasingFunction, bool aEnable = true, bool aDisableWhenFinished = false, bool aResetStartTime = true) = 0;
        virtual bool can_apply() const = 0;
        virtual void apply() = 0;
    };

    class transition;

    class i_animator : public i_service
    {
        friend class neogfx::transition;
    public:
        virtual i_transition& transition(transition_id aTransitionId) = 0;
        virtual transition_id add_transition(i_transition& aTransition) = 0;
        virtual void remove_transition(transition_id aTransitionId) = 0;
        virtual void stop() = 0;
    public:
        virtual double animation_time() const = 0;
    protected:
        virtual transition_id allocate_id() = 0;
    public:
        static uuid const& iid() { static uuid const sIid{ 0x182ccd2e, 0xd7dd, 0x4a6c, 0x9ac, { 0x72, 0xfd, 0x81, 0x24, 0x7a, 0xe9 } }; return sIid; }
    };
}