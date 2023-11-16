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

#include "data/tile_attributes.hpp"
#include "engine/base_components.hpp"
#include "engine/map_renderer.hpp"
#include "game_logic/global_dependencies.hpp"

#include <optional>
#include <vector>


namespace rigel::game_logic
{
namespace components
{

struct DynamicGeometrySection
{
  // This represents the area below a piece of dynamic geometry, which
  // will be erased as the piece is falling down.
  struct ExtraSection
  {
    std::vector<std::uint32_t> mMapData;
    int mTop;
    int mHeight;
  };

  explicit DynamicGeometrySection(
    engine::components::BoundingBox geometrySection)
    : mLinkedGeometrySection(geometrySection)
    , mPreviousHeight(geometrySection.size.height)
  {
  }

  std::optional<base::Rect<int>> extraSectionRect() const
  {
    if (mExtraSection)
    {
      return base::Rect<int>{
        {mLinkedGeometrySection.left(), mExtraSection->mTop},
        {mLinkedGeometrySection.size.width, mExtraSection->mHeight}};
    }

    return std::nullopt;
  }

  engine::components::BoundingBox mLinkedGeometrySection;
  std::optional<ExtraSection> mExtraSection;

  // These two are for interpolation while sinking
  std::vector<engine::PackedTileData> mBottomRowCopy;
  int mPreviousHeight;
};


struct TileDebris
{
  data::map::TileIndex mTileIndex;
};


struct ShootableWall
{
};

} // namespace components

namespace behaviors
{

struct DynamicGeometryController
{
  enum class Type : std::uint8_t
  {
    FallDownAfterDelayThenSinkIntoGround,
    BlueKeyDoor,
    FallDownWhileEarthQuakeActiveThenExplode,
    FallDownImmediatelyThenStayOnGround,
    FallDownWhileEarthQuakeActiveThenStayOnGround, // TODO: Not implemented yet
    FallDownImmediatelyThenExplode,
    FallDownAfterDelayThenStayOnGround
  };

  enum class State : std::uint8_t
  {
    Waiting,
    Falling,
    Sinking
  };

  explicit DynamicGeometryController(const Type type)
    : mType(type)
  {
  }

  void update(
    GlobalDependencies& dependencies,
    GlobalState& state,
    bool isOnScreen,
    entityx::Entity entity);

  int mFramesElapsed = 0;
  Type mType;
  State mState = State::Waiting;
};

} // namespace behaviors
} // namespace rigel::game_logic
