// simple_physics.hpp
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
#include <neogfx/core/event.hpp>
#include <neogfx/game/system.hpp>
#include <neogfx/game/box_collider.hpp>
#include <neogfx/game/mesh_filter.hpp>
#include <neogfx/game/rigid_body.hpp>
#include <neogfx/game/mesh_render_cache.hpp>

namespace neogfx::game
{
    class simple_physics : public game::system<entity_info, box_collider, box_collider_2d, mesh_filter, rigid_body, mesh_render_cache>
    {
    public:
        simple_physics(i_ecs& aEcs);
        ~simple_physics();
    public:
        const system_id& id() const override;
        const i_string& name() const override;
    public:
        bool apply() override;
    public:
        bool universal_gravitation_enabled() const;
        void enable_universal_gravitation();
        void disable_universal_gravitation();
    public:
        void yield_after(std::chrono::duration<double, std::milli> aTime);
    public:
        struct meta
        {
            static const neolib::uuid& id()
            {
                static const neolib::uuid sId = { 0x49443e26, 0x762e, 0x4517, 0xbbb8,{ 0xc3, 0xd6, 0x95, 0x7b, 0xe9, 0xd4 } };
                return sId;
            }
            static const i_string& name()
            {
                static const string sName = "Simple Physics";
                return sName;
            }
        };
    private:
        std::chrono::duration<double, std::milli> iYieldTime = std::chrono::duration<double, std::milli>{ 1.0 };
    };
}