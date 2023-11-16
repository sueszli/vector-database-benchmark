// i_sub_texture.hpp
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
#include <neogfx/gfx/i_texture.hpp>

namespace neogfx
{
    class i_sub_texture : public i_texture
    {
    public:
        struct bad_rectangle : std::logic_error { bad_rectangle() : std::logic_error("neogfx::i_sub_texture::bad_rectangle") {} };
    public:
        virtual texture_id atlas_id() const = 0;
        virtual i_texture& atlas_texture() const = 0;
        virtual const rect& atlas_location() const = 0;
    };
}