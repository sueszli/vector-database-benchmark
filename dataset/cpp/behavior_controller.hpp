/* Copyright (C) 2018, Nikolai Wuttke. All rights reserved.
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
#include "game_logic/global_dependencies.hpp"

RIGEL_DISABLE_WARNINGS
#include <entityx/entityx.h>
RIGEL_RESTORE_WARNINGS

#include <memory>
#include <type_traits>

namespace rigel::engine::events
{
struct CollidedWithWorld;
}


namespace rigel::game_logic::components
{

namespace detail
{

template <typename...>
using void_t = void;

template <typename T, typename = void>
struct hasOnHit : std::false_type
{
};

template <typename T>
struct hasOnHit<T, void_t<decltype(&T::onHit)>> : std::true_type
{
};


template <typename T, typename = void>
struct hasOnKilled : std::false_type
{
};

template <typename T>
struct hasOnKilled<T, void_t<decltype(&T::onKilled)>> : std::true_type
{
};


template <typename T, typename = void>
struct hasOnCollision : std::false_type
{
};

template <typename T>
struct hasOnCollision<T, void_t<decltype(&T::onCollision)>> : std::true_type
{
};
} // namespace detail


template <typename T>
void updateBehaviorController(
  T& self,
  GlobalDependencies& dependencies,
  GlobalState& state,
  const bool isOnScreen,
  entityx::Entity entity)
{
  self.update(dependencies, state, isOnScreen, entity);
}


template <typename T>
std::enable_if_t<detail::hasOnHit<T>::value> behaviorControllerOnHit(
  T& self,
  GlobalDependencies& dependencies,
  GlobalState& state,
  entityx::Entity inflictorEntity,
  entityx::Entity entity)
{
  self.onHit(dependencies, state, inflictorEntity, entity);
}


template <typename T>
std::enable_if_t<!detail::hasOnHit<T>::value> behaviorControllerOnHit(
  T&,
  GlobalDependencies&,
  GlobalState&,
  entityx::Entity,
  entityx::Entity)
{
}


template <typename T>
std::enable_if_t<detail::hasOnKilled<T>::value> behaviorControllerOnKilled(
  T& self,
  GlobalDependencies& dependencies,
  GlobalState& state,
  const base::Vec2f& inflictorVelocity,
  entityx::Entity entity)
{
  self.onKilled(dependencies, state, inflictorVelocity, entity);
}


template <typename T>
std::enable_if_t<!detail::hasOnKilled<T>::value> behaviorControllerOnKilled(
  T&,
  GlobalDependencies&,
  GlobalState& state,
  const base::Vec2f&,
  entityx::Entity)
{
}


template <typename T>
std::enable_if_t<detail::hasOnCollision<T>::value>
  behaviorControllerOnCollision(
    T& self,
    GlobalDependencies& dependencies,
    GlobalState& state,
    const engine::events::CollidedWithWorld& event,
    entityx::Entity entity)
{
  self.onCollision(dependencies, state, event, entity);
}


template <typename T>
std::enable_if_t<!detail::hasOnCollision<T>::value>
  behaviorControllerOnCollision(
    T&,
    GlobalDependencies&,
    GlobalState&,
    const engine::events::CollidedWithWorld&,
    entityx::Entity)
{
}


class BehaviorController
{
public:
  template <typename T>
  explicit BehaviorController(T controller)
    : mpSelf(std::make_unique<Model<T>>(std::move(controller)))
  {
  }

  BehaviorController(const BehaviorController& other)
    : mpSelf(other.mpSelf->clone())
  {
  }

  BehaviorController& operator=(const BehaviorController& other)
  {
    auto copy = other;
    std::swap(mpSelf, copy.mpSelf);
    return *this;
  }

  BehaviorController(BehaviorController&&) = default;
  BehaviorController& operator=(BehaviorController&&) = default;

  void update(
    GlobalDependencies& dependencies,
    GlobalState& state,
    const bool isOnScreen,
    entityx::Entity entity)
  {
    mpSelf->update(dependencies, state, isOnScreen, entity);
  }

  void onHit(
    GlobalDependencies& dependencies,
    GlobalState& state,
    entityx::Entity inflictorEntity,
    entityx::Entity entity)
  {
    mpSelf->onHit(dependencies, state, inflictorEntity, entity);
  }

  void onKilled(
    GlobalDependencies& dependencies,
    GlobalState& state,
    const base::Vec2f& inflictorVelocity,
    entityx::Entity entity)
  {
    mpSelf->onKilled(dependencies, state, inflictorVelocity, entity);
  }

  void onCollision(
    GlobalDependencies& dependencies,
    GlobalState& state,
    const engine::events::CollidedWithWorld& event,
    entityx::Entity entity)
  {
    mpSelf->onCollision(dependencies, state, event, entity);
  }

  template <typename T>
  T& get()
  {
    auto pSelf = mpSelf.get();
    return dynamic_cast<Model<T>*>(pSelf)->mData;
  }

private:
  struct Concept
  {
    virtual ~Concept() = default;

    virtual std::unique_ptr<Concept> clone() const = 0;

    virtual void update(
      GlobalDependencies& dependencies,
      GlobalState& state,
      bool isOnScreen,
      entityx::Entity entity) = 0;

    virtual void onHit(
      GlobalDependencies& dependencies,
      GlobalState& state,
      entityx::Entity inflictorEntity,
      entityx::Entity entity) = 0;

    virtual void onKilled(
      GlobalDependencies& dependencies,
      GlobalState& state,
      const base::Vec2f& inflictorVelocity,
      entityx::Entity entity) = 0;

    virtual void onCollision(
      GlobalDependencies& dependencies,
      GlobalState& state,
      const engine::events::CollidedWithWorld& event,
      entityx::Entity entity) = 0;
  };

  template <typename T>
  struct Model : public Concept
  {
    explicit Model(T data_)
      : mData(std::move(data_))
    {
    }

    std::unique_ptr<Concept> clone() const override
    {
      return std::make_unique<Model>(mData);
    }

    void update(
      GlobalDependencies& dependencies,
      GlobalState& state,
      bool isOnScreen,
      entityx::Entity entity) override
    {
      updateBehaviorController(mData, dependencies, state, isOnScreen, entity);
    }

    void onHit(
      GlobalDependencies& dependencies,
      GlobalState& state,
      entityx::Entity inflictorEntity,
      entityx::Entity entity) override
    {
      behaviorControllerOnHit(
        mData, dependencies, state, inflictorEntity, entity);
    }

    void onKilled(
      GlobalDependencies& dependencies,
      GlobalState& state,
      const base::Vec2f& inflictorVelocity,
      entityx::Entity entity) override
    {
      behaviorControllerOnKilled(
        mData, dependencies, state, inflictorVelocity, entity);
    }

    void onCollision(
      GlobalDependencies& dependencies,
      GlobalState& state,
      const engine::events::CollidedWithWorld& event,
      entityx::Entity entity) override
    {
      behaviorControllerOnCollision(mData, dependencies, state, event, entity);
    }

    T mData;
  };

  std::unique_ptr<Concept> mpSelf;
};

} // namespace rigel::game_logic::components
