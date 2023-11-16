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

#include "floating_laser_bot.hpp"

#include "base/match.hpp"
#include "data/sound_ids.hpp"
#include "engine/entity_tools.hpp"
#include "engine/movement.hpp"
#include "engine/random_number_generator.hpp"
#include "engine/sprite_tools.hpp"
#include "engine/visual_components.hpp"
#include "frontend/game_service_provider.hpp"
#include "game_logic/ientity_factory.hpp"
#include "game_logic/player.hpp"

#include <algorithm>


namespace rigel::game_logic::behaviors
{

namespace
{

struct GunSpec
{
  base::Vec2 mOffset;
  engine::components::Orientation mOrientation;
};


// clang-format off
constexpr GunSpec GUN_SPECS[]{
  {{-1, -1}, engine::components::Orientation::Left},
  {{-1,  0}, engine::components::Orientation::Left},
  {{ 2,  0}, engine::components::Orientation::Right},
  {{ 2, -1}, engine::components::Orientation::Right}};
// clang-format on

} // namespace


void FloatingLaserBot::update(
  GlobalDependencies& d,
  GlobalState& s,
  const bool isOnScreen,
  entityx::Entity entity)
{
  using engine::components::Sprite;
  using engine::components::WorldPosition;

  const auto& position = *entity.component<WorldPosition>();
  auto& animationFrame = entity.component<Sprite>()->mFramesToRender[0];

  auto moveTowardsPlayer = [&, this]() {
    const auto offsetToPlayer =
      s.mpPlayer->orientedPosition() - position + base::Vec2{1, -2};
    const auto movementX = std::clamp(offsetToPlayer.x, -1, 1);
    const auto movementY = std::clamp(offsetToPlayer.y, -1, 1);

    engine::moveHorizontally(*d.mpCollisionChecker, entity, movementX);
    engine::moveVertically(*d.mpCollisionChecker, entity, movementY);
  };

  auto attack = [&, this](int gunIndex) {
    d.mpServiceProvider->playSound(data::SoundId::EnemyLaserShot);
    spawnEnemyLaserShot(
      *d.mpEntityFactory,
      position + GUN_SPECS[gunIndex].mOffset,
      GUN_SPECS[gunIndex].mOrientation);
  };


  base::match(
    mState,
    [&, this](Waiting& state) {
      if (!isOnScreen)
      {
        // De-activate until sighted again
        engine::resetActivation(entity);
      }

      ++state.mFramesElapsed;
      if (state.mFramesElapsed == 10)
      {
        mState = Active{};
      }
    },


    [&, this](Active& state) {
      if (state.mFramesElapsed < 40)
      {
        if (d.mpRandomGenerator->gen() % 4 == 0)
        {
          moveTowardsPlayer();
        }
      }
      else if (state.mFramesElapsed < 50)
      {
        // unfold
        if (animationFrame < 5)
        {
          ++animationFrame;
        }
      }
      else if (state.mFramesElapsed < 80)
      {
        // while in this stage, only update on even frames
        if (s.mpPerFrameState->mIsOddFrame)
        {
          return;
        }

        attack(state.mFramesElapsed % 4);
      }
      else
      {
        // fold back in
        if (animationFrame > 0)
        {
          --animationFrame;
        }
        else
        {
          mState = Waiting{};
          return;
        }
      }

      ++state.mFramesElapsed;
    });

  engine::synchronizeBoundingBoxToSprite(entity);
}

} // namespace rigel::game_logic::behaviors
