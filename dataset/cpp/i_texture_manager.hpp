// i_texture_manager.hpp
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
#include <neolib/core/jar.hpp>
#include <neogfx/gfx/i_image.hpp>
#include <neogfx/gfx/i_texture.hpp>
#include <neogfx/gfx/i_sub_texture.hpp>
#include <neogfx/gfx/i_texture_atlas.hpp>

namespace neogfx
{
    class i_native_texture;

    class i_texture_manager : public neolib::i_cookie_consumer, public i_service
    {
        friend class texture_atlas;
    public:
        struct texture_not_found : std::logic_error { texture_not_found() : std::logic_error("neogfx::i_texture_manager::texture_not_found") {} };
    private:
        virtual texture_id allocate_texture_id() = 0;
    public:
        ref_ptr<i_texture> find_texture(texture_id aId) const
        {
            ref_ptr<i_texture> result;
            find_texture(aId, result);
            return result;
        }
        virtual void find_texture(texture_id aId, i_ref_ptr<i_texture>& aResult) const = 0;
        ref_ptr<i_texture> create_texture(neogfx::size const& aExtents, dimension aDpiScaleFactor = 1.0, texture_sampling aSampling = texture_sampling::NormalMipmap, texture_data_format aDataFormat = texture_data_format::RGBA, texture_data_type aDataType = texture_data_type::UnsignedByte, color_space aColorSpace = color_space::sRGB, optional_color const& aColor = optional_color())
        {
            ref_ptr<i_texture> result;
            create_texture(aExtents, aDpiScaleFactor, aSampling, aDataFormat, aDataType, aColorSpace, aColor, result);
            return result;
        }
        virtual void create_texture(neogfx::size const& aExtents, dimension aDpiScaleFactor, texture_sampling aSampling, texture_data_format aDataFormat, texture_data_type aDataType, color_space aColorSpace, optional_color const& aColor, i_ref_ptr<i_texture>& aResult) = 0;
        ref_ptr<i_texture> create_texture(i_image const& aImage, texture_data_format aDataFormat = texture_data_format::RGBA, texture_data_type aDataType = texture_data_type::UnsignedByte)
        {
            return create_texture(aImage, rect{ point{}, aImage.extents() }, aDataFormat, aDataType);
        }
        ref_ptr<i_texture> create_texture(i_image const& aImage, const rect& aImagePart, texture_data_format aDataFormat = texture_data_format::RGBA, texture_data_type aDataType = texture_data_type::UnsignedByte)
        {
            ref_ptr<i_texture> result;
            create_texture(aImage, aImagePart, aDataFormat, aDataType, result);
            return result;
        }
        virtual void create_texture(i_image const& aImage, const rect& aImagePart, texture_data_format aDataFormat, texture_data_type aDataType, i_ref_ptr<i_texture>& aResult) = 0;
        virtual void clear_textures() = 0;
    public:
        virtual std::unique_ptr<i_texture_atlas> create_texture_atlas(const size& aSize = size{ 1024.0, 1024.0 }) = 0;
    private:
        virtual void add_sub_texture(i_sub_texture& aSubTexture) = 0;
        virtual void remove_sub_texture(i_sub_texture& aSubTexture) = 0;
    public:
        static uuid const& iid() { static uuid const sIid{ 0xbc995572, 0x980e, 0x40cd, 0xa13e,{ 0x83, 0x66, 0xc1, 0x73, 0x50, 0xf4 } }; return sIid; }
    };
}