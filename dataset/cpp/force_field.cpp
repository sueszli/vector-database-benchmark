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

#include "force_field.hpp"

#include "engine/random_number_generator.hpp"
#include "engine/sprite_tools.hpp"
#include "engine/visual_components.hpp"
#include "frontend/game_service_provider.hpp"
#include "game_logic/actor_tag.hpp"
#include "game_logic/global_dependencies.hpp"
#include "game_logic/player/components.hpp"

namespace ex = entityx;


namespace rigel::game_logic::interaction
{

using namespace engine::components;
using namespace game_logic::components;

void configureForceField(entityx::Entity entity, const int spawnIndex)
{
  engine::startAnimationLoop(entity, 1, 2, 4);
  entity.assign<BoundingBox>(BoundingBox{{0, -4}, {2, 10}});
  entity.assign<ActorTag>(ActorTag::Type::ForceField, spawnIndex);
}


void configureKeyCardSlot(
  entityx::Entity entity,
  const BoundingBox& boundingBox)
{
  entity.assign<Interactable>(InteractableType::ForceFieldCardReader);
  entity.assign<AnimationLoop>(1);
  entity.assign<BoundingBox>(boundingBox);
}


void disableKeyCardSlot(entityx::Entity entity)
{
  entity.remove<Interactable>();
  entity.remove<AnimationLoop>();
  entity.remove<BoundingBox>();
}


void disableNextForceField(entityx::EntityManager& es)
{
  auto nextForceField =
    findFirstMatchInSpawnOrder(es, ActorTag::Type::ForceField);
  if (nextForceField)
  {
    nextForceField.destroy();
  }
}

} // namespace rigel::game_logic::interaction


namespace rigel::game_logic::behaviors
{

void ForceField::update(
  GlobalDependencies& d,
  GlobalState& s,
  bool,
  entityx::Entity entity)
{
  auto& sprite = *entity.component<engine::components::Sprite>();

  const auto fizzle = (d.mpRandomGenerator->gen() / 32) % 2 != 0;
  if (fizzle)
  {
    d.mpServiceProvider->playSound(data::SoundId::ForceFieldFizzle);
    sprite.flashWhite();
  }
}

} // namespace rigel::game_logic::behaviors
