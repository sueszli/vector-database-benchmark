// opengl.hpp
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
#include <type_traits>
#include <GL/glew.h>
#include <GL/GL.h>
#include "opengl_error.hpp"

template <typename T>
inline T to_gl_handle(void* aOpaqueHandle)
{
    return static_cast<T>(reinterpret_cast<std::ptrdiff_t>(aOpaqueHandle));
}

template <typename T1, typename T2>
inline std::enable_if_t<std::is_integral_v<T2>, T1> gl_handle_cast(T2 aGlHandle)
{
    return static_cast<T1>(aGlHandle);
}

template <typename T>
inline void* to_opaque_handle(T aGlHandle)
{
    return reinterpret_cast<void*>(static_cast<std::ptrdiff_t>(aGlHandle));
}
