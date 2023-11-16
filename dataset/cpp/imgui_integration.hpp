/* Copyright (C) 2019, Nikolai Wuttke. All rights reserved.
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "base/warnings.hpp"

RIGEL_DISABLE_WARNINGS
#include <SDL.h>
RIGEL_RESTORE_WARNINGS

#include <filesystem>
#include <optional>


namespace rigel::ui::imgui_integration
{

void init(
  SDL_Window* pWindow,
  void* pGlContext,
  const std::optional<std::filesystem::path>& preferencesPath);
void shutdown();

bool handleEvent(const SDL_Event& event);
void beginFrame(SDL_Window* pWindow);
void endFrame();

} // namespace rigel::ui::imgui_integration
