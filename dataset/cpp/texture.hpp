// texture.hpp
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
#include <neolib/core/uuid.hpp>
#include <neolib/core/string.hpp>
#include <neogfx/gfx/i_texture.hpp>
#include <neogfx/gfx/i_sub_texture.hpp>
#include <neogfx/gfx/i_texture_manager.hpp>
#include <neogfx/game/ecs_ids.hpp>
#include <neogfx/game/component.hpp>
#include <neogfx/game/i_component_data.hpp>
#include <neogfx/game/image.hpp>

namespace neogfx::game
{
    struct texture
    {
        neolib::cookie_ref_ptr id;
        texture_type type;
        std::optional<texture_sampling> sampling;
        scalar dpiScalingFactor;
        vec2 extents;
        optional_aabb_2d subTexture;

        struct meta : i_component_data::meta
        {
            static const neolib::uuid& id()
            {
                static const neolib::uuid sId = { 0x9f08230d, 0x25f, 0x4466, 0x9aab, { 0x1, 0x8d, 0x3, 0x29, 0x2e, 0xdc } };
                return sId;
            }
            static const i_string& name()
            {
                static const string sName = "Texture";
                return sName;
            }
            static uint32_t field_count()
            {
                return 6;
            }
            static component_data_field_type field_type(uint32_t aFieldIndex)
            {
                switch (aFieldIndex)
                {
                case 0:
                    return component_data_field_type::Id;
                case 1:
                    return component_data_field_type::Enum | component_data_field_type::Uint32;
                case 2:
                    return component_data_field_type::Enum | component_data_field_type::Uint32 | component_data_field_type::Optional;
                case 3:
                    return component_data_field_type::Scalar;
                case 4:
                    return component_data_field_type::Vec2;
                case 5:
                    return component_data_field_type::Aabb2d | component_data_field_type::Optional;
                default:
                    throw invalid_field_index();
                }
            }
            static neolib::uuid field_type_id(uint32_t aFieldIndex)
            {
                switch (aFieldIndex)
                {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                    return neolib::uuid{};
                default:
                    throw invalid_field_index();
                }
            }
            static const i_string& field_name(uint32_t aFieldIndex)
            {
                static const string sFieldNames[] =
                {
                    "Id",
                    "Type",
                    "Sampling",
                    "DPI Scale Factor",
                    "Extents",
                    "Sub Texture"
                };
                return sFieldNames[aFieldIndex];
            }
        };
    };

    inline bool batchable(const texture& lhs, const texture& rhs)
    {
        if (lhs.sampling != rhs.sampling)
            return false;
        auto const& lhsTexture = *service<i_texture_manager>().find_texture(lhs.id.cookie());
        auto const& rhsTexture = *service<i_texture_manager>().find_texture(rhs.id.cookie());
        return lhsTexture.native_handle() == rhsTexture.native_handle();
    }
}