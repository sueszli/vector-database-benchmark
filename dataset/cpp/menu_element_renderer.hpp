/* Copyright (C) 2016, Nikolai Wuttke. All rights reserved.
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

#include "assets/palette.hpp"
#include "base/color.hpp"
#include "base/warnings.hpp"
#include "engine/tiled_texture.hpp"
#include "engine/timing.hpp"
#include "renderer/texture.hpp"

RIGEL_DISABLE_WARNINGS
#include <SDL.h>
RIGEL_RESTORE_WARNINGS

#include <optional>
#include <string>


namespace rigel::assets
{
class ResourceLoader;
}


namespace rigel::ui
{


class MenuElementRenderer
{
public:
  MenuElementRenderer(
    engine::TiledTexture* pSpriteSheet,
    renderer::Renderer* pRenderer,
    const assets::ResourceLoader& resources);

  void drawText(int x, int y, std::string_view text) const;
  void drawSmallWhiteText(int x, int y, std::string_view text) const;
  void drawMultiLineText(int x, int y, std::string_view text) const;
  void
    drawBigText(int x, int y, std::string_view text, const base::Color& color)
      const;

  void drawCheckBox(int x, int y, bool isChecked) const;

  void drawBonusScreenText(int x, int y, std::string_view text) const;

  /** Draw message box frame at given position, with slide-in animation
   *
   * elapsedTime should be the total elapsed time since the message box frame
   * is being drawn.
   *
   * Returns true if the animation is finished.
   */
  bool drawMessageBox(
    int x,
    int y,
    int width,
    int height,
    engine::TimeDelta elapsedTime) const;

  /** Draw text entry cursor icon at given position/state.
   *
   * elapsedTime should be the total elapsed time since the text entry cursor
   * is being drawn.
   */
  void drawTextEntryCursor(int x, int y, engine::TimeDelta elapsedTime) const;

  /** Draw menu selection indicator (spinning arrow) at given position.
   *
   * elapsedTime should be the total elapsed time since the selection indicator
   * is being drawn.
   */
  void
    drawSelectionIndicator(int x, int y, engine::TimeDelta elapsedTime) const;

private:
  void drawTextEntryCursor(int x, int y, int state) const;
  void drawSelectionIndicator(int x, int y, int state) const;

  renderer::Renderer* mpRenderer;
  engine::TiledTexture* mpSpriteSheet;
  engine::TiledTexture mBigTextTexture;
};

} // namespace rigel::ui
