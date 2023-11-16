// texture_atlas.hpp
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
#include <unordered_map>
#include <set>
#include <tuple>
#include "i_texture_atlas.hpp"
#include "i_texture_manager.hpp"
#include "texture.hpp"
#include "sub_texture.hpp"
#include "rect_pack.hpp"

namespace neogfx
{
    class i_native_texture;

    class texture_atlas : public i_texture_atlas
    {
    private:
        struct fragments
        {
            struct fragment_less_than
            {
                bool operator()(const rect& aLhs, const rect& aRhs) const
                {
                    return std::forward_as_tuple(aLhs.width() * aLhs.height(), aLhs.top_left()) <
                        std::forward_as_tuple(aRhs.width() * aRhs.height(), aRhs.top_left());
                }
            };
            rect_pack pack;
            std::set<rect, fragment_less_than> used;
            std::set<rect, fragment_less_than> freed; // todo: use this when bin pack is full
            bool insert(const size& aSize, rect& aResult)
            {
                if (pack.insert(aSize, aResult))
                {
                    used.insert(aResult);
                    return true;
                }
                else
                    return false;
            }
        };
        typedef std::pair<texture, fragments> page;
        typedef std::list<page> pages;
        typedef std::pair<pages::iterator, neogfx::sub_texture> entry;
        typedef std::unordered_map<texture_id, entry> entries;
    public:
        texture_atlas(const size& aPageSize);
    public:
        const i_sub_texture& sub_texture(texture_id aSubTextureId) const override;
        i_sub_texture& sub_texture(texture_id aSubTextureId) override;
        i_sub_texture& create_sub_texture(const size& aSize, dimension aDpiScaleFactor, texture_sampling aSampling, texture_data_format aDataFormat = texture_data_format::RGBA) override;
        i_sub_texture& create_sub_texture(const i_image& aImage) override;
        i_sub_texture& create_sub_texture(const i_image& aImage, const rect& aImagePart) override;
        void destroy_sub_texture(i_sub_texture& aSubTexture) override;
    private:
        const size& page_size() const;
        pages::iterator create_page(dimension aDpiScaleFactor, texture_sampling aSampling, texture_data_format aDataFormat);
        std::pair<pages::iterator, rect> allocate_space(const size& aSize, dimension aDpiScaleFactor, texture_sampling aSampling, texture_data_format aDataFormat);
    private:
        i_texture_manager& iTextureManager;
        size iPageSize;
        pages iPages;
        entries iEntries;
    };
}