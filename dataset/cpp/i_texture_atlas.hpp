// i_texture_atlas.hpp
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
#include <neogfx/gfx/i_image.hpp>
#include <neogfx/gfx/i_sub_texture.hpp>

namespace neogfx
{
    class i_texture_atlas
    {
    public:
        struct sub_texture_not_found : std::logic_error { sub_texture_not_found() : std::logic_error("neogfx::i_texture_atlas::sub_texture_not_found") {} };
        struct texture_too_big_for_atlas : std::logic_error { texture_too_big_for_atlas() : std::logic_error("neogfx::i_texture_atlas::texture_too_big_for_atlas") {} };
    public:
        virtual ~i_texture_atlas() = default;
    public:
        virtual const i_sub_texture& sub_texture(texture_id aSubTextureId) const = 0;
        virtual i_sub_texture& sub_texture(texture_id aSubTextureId) = 0;
        virtual i_sub_texture& create_sub_texture(const size& aSize, dimension aDpiScaleFactor, texture_sampling aSampling, texture_data_format aDataFormat = texture_data_format::RGBA) = 0;
        virtual i_sub_texture& create_sub_texture(const i_image& aImage) = 0;
        virtual i_sub_texture& create_sub_texture(const i_image& aImage, const rect& aImagePart) = 0;
        virtual void destroy_sub_texture(i_sub_texture& aSubTexture) = 0;
    };
}