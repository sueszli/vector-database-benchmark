// i_glyph.hpp
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
#include <neogfx/core/geometrical.hpp>
#include <neogfx/gfx/i_sub_texture.hpp>

namespace neogfx
{
    enum class glyph_pixel_mode : uint32_t
    {
        None,
        Mono,
        Gray2Bit,
        Gray4Bit,
        Gray8Bit,
        Gray = Gray8Bit,
        LCD,
        LCD_V,
        BGRA
    };

    struct glyph_metrics
    {
        vec2 extents;
        vec2 bearing;
    };

    class i_glyph
    {
    public:
        virtual ~i_glyph() = default;
    public:
        virtual const i_sub_texture& texture() const = 0;
        virtual bool subpixel() const = 0;
        virtual const glyph_metrics& metrics() const = 0;
        virtual glyph_pixel_mode pixel_mode() const = 0;
    };
}