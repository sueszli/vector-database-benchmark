/* Copyright (C) 2017, Nikolai Wuttke. All rights reserved.
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

#include "effects_system.hpp"

#include "base/match.hpp"
#include "data/game_traits.hpp"
#include "engine/base_components.hpp"
#include "engine/life_time_components.hpp"
#include "engine/particle_system.hpp"
#include "engine/physical_components.hpp"
#include "engine/random_number_generator.hpp"
#include "engine/visual_components.hpp"
#include "frontend/game_service_provider.hpp"
#include "game_logic/damage_components.hpp"
#include "game_logic/ientity_factory.hpp"


namespace rigel::game_logic
{

namespace
{

using components::DestructionEffects;
using components::SpriteCascadeSpawner;

} // namespace


void triggerEffects(
  entityx::Entity entity,
  entityx::EntityManager& entityManager)
{
  if (!entity.has_component<DestructionEffects>())
  {
    return;
  }

  spawnEffects(
    *entity.component<DestructionEffects>(),
    *entity.component<engine::components::WorldPosition>(),
    entityManager);
}


void spawnEffects(
  const DestructionEffects& effects,
  const base::Vec2& position,
  entityx::EntityManager& entityManager)
{
  using namespace engine::components;

  auto effectSpawner = entityManager.create();
  effectSpawner.assign<DestructionEffects>(effects);
  effectSpawner.assign<WorldPosition>(position);
  effectSpawner.component<DestructionEffects>()->mActivated = true;

  const auto iHighestDelaySpec = std::max_element(
    std::begin(effects.mEffectSpecs),
    std::end(effects.mEffectSpecs),
    [](const auto& a, const auto& b) { return a.mDelay < b.mDelay; });
  const auto timeToLive = iHighestDelaySpec->mDelay;
  effectSpawner.assign<AutoDestroy>(AutoDestroy::afterTimeout(timeToLive));
}


EffectsSystem::EffectsSystem(
  IGameServiceProvider* pServiceProvider,
  engine::RandomNumberGenerator* pRandomGenerator,
  entityx::EntityManager* pEntityManager,
  IEntityFactory* pEntityFactory,
  engine::ParticleSystem* pParticles,
  entityx::EventManager& events)
  : mpServiceProvider(pServiceProvider)
  , mpRandomGenerator(pRandomGenerator)
  , mpEntityManager(pEntityManager)
  , mpEntityFactory(pEntityFactory)
  , mpParticles(pParticles)
{
  events.subscribe<events::ShootableKilled>(*this);
  events.subscribe<engine::events::CollidedWithWorld>(*this);
}


void EffectsSystem::update(entityx::EntityManager& es)
{
  using namespace engine::components;

  es.each<DestructionEffects, WorldPosition>([this](
                                               entityx::Entity entity,
                                               DestructionEffects& effects,
                                               const WorldPosition& position) {
    if (effects.mActivated)
    {
      processEffectsAndAdvance(position, effects);
    }
  });

  es.each<SpriteCascadeSpawner>(
    [this](entityx::Entity, SpriteCascadeSpawner& spawner) {
      if (!spawner.mSpawnedLastFrame)
      {
        const auto xOffset =
          mpRandomGenerator->gen() % spawner.mCoveredArea.width;
        const auto yOffset =
          mpRandomGenerator->gen() % spawner.mCoveredArea.height;
        const auto spawnPosition =
          spawner.mBasePosition + base::Vec2{xOffset, -yOffset};

        spawnFloatingOneShotSprite(
          *mpEntityFactory, spawner.mActorId, spawnPosition);
      }

      spawner.mSpawnedLastFrame = !spawner.mSpawnedLastFrame;
    });
}


void EffectsSystem::receive(const events::ShootableKilled& event)
{
  triggerEffectsIfConditionMatches(
    event.mEntity, DestructionEffects::TriggerCondition::OnKilled);
}


void EffectsSystem::receive(const engine::events::CollidedWithWorld& event)
{
  triggerEffectsIfConditionMatches(
    event.mEntity, DestructionEffects::TriggerCondition::OnCollision);
}


void EffectsSystem::triggerEffectsIfConditionMatches(
  entityx::Entity entity,
  const DestructionEffects::TriggerCondition expectedCondition)
{
  if (!entity.has_component<DestructionEffects>())
  {
    return;
  }

  const auto triggerCondition =
    entity.component<DestructionEffects>()->mTriggerCondition;
  if (triggerCondition == expectedCondition)
  {
    triggerEffects(entity, *mpEntityManager);
  }
}


void EffectsSystem::processEffectsAndAdvance(
  const base::Vec2& position,
  DestructionEffects& effects)
{
  using namespace effects;
  using namespace engine::components;
  using namespace engine::components::parameter_aliases;

  for (auto& spec : effects.mEffectSpecs)
  {
    if (effects.mFramesElapsed != spec.mDelay)
    {
      continue;
    }

    base::match(
      spec.mEffect,
      [this](const Sound& sound) { mpServiceProvider->playSound(sound.mId); },

      [this](const RandomExplosionSound&) {
        const auto randomChoice = mpRandomGenerator->gen();
        const auto soundId = randomChoice % 2 == 0
          ? data::SoundId::AlternateExplosion
          : data::SoundId::Explosion;
        mpServiceProvider->playSound(soundId);
      },

      [&, this](const Particles& particles) {
        const auto color = particles.mColor
          ? *particles.mColor
          : data::GameTraits::INGAME_PALETTE[mpRandomGenerator->gen() % 16];

        mpParticles->spawnParticles(
          position + particles.mOffset, color, particles.mVelocityScaleX);
      },

      [&, this](const EffectSprite& sprite) {
        if (sprite.mMovement == EffectSprite::Movement::None)
        {
          spawnOneShotSprite(
            *mpEntityFactory, sprite.mActorId, position + sprite.mOffset);
        }
        else if (sprite.mMovement == EffectSprite::Movement::FloatUp)
        {
          spawnFloatingOneShotSprite(
            *mpEntityFactory, sprite.mActorId, position + sprite.mOffset);
        }
        else
        {
          using M = EffectSprite::Movement;
          static_assert(
            int(M::FlyRight) == int(SpriteMovement::FlyRight) &&
            int(M::FlyUpperRight) == int(SpriteMovement::FlyUpperRight) &&
            int(M::FlyUp) == int(SpriteMovement::FlyUp) &&
            int(M::FlyUpperLeft) == int(SpriteMovement::FlyUpperLeft) &&
            int(M::FlyLeft) == int(SpriteMovement::FlyLeft) &&
            int(M::FlyDown) == int(SpriteMovement::FlyDown));

          const auto movementType =
            static_cast<SpriteMovement>(static_cast<int>(sprite.mMovement));
          spawnMovingEffectSprite(
            *mpEntityFactory,
            sprite.mActorId,
            movementType,
            position + sprite.mOffset);
        }
      },

      [&, this](const SpriteCascade& cascade) {
        auto coveredArea = effects.mCascadePlacementBox
          ? *effects.mCascadePlacementBox
          : BoundingBox{};

        spawnFireEffect(
          *mpEntityManager, position, coveredArea, cascade.mActorId);
      },

      [&, this](const ScoreNumber& scoreNumber) {
        spawnFloatingScoreNumber(
          *mpEntityFactory, scoreNumber.mType, position + scoreNumber.mOffset);
      });
  }

  ++effects.mFramesElapsed;
}

} // namespace rigel::game_logic
