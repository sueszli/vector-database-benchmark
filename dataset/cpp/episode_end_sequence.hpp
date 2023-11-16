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

#include "frontend/game_mode.hpp"
#include "renderer/texture.hpp"
#include "ui/bonus_screen.hpp"
#include "ui/duke_3d_teaser_screen.hpp"

#include <variant>
#include <vector>


namespace rigel::ui
{

class EpisodeEndScreen
{
public:
  EpisodeEndScreen(GameMode::Context context, int episode);

  void updateAndRender(engine::TimeDelta dt);
  void handleEvent(const SDL_Event& event);

  bool finished() const;

private:
  std::vector<renderer::Texture> mScreenImages;
  std::size_t mCurrentImage = 0;
  IGameServiceProvider* mpServiceProvider;
};


class EpisodeEndSequence
{
public:
  EpisodeEndSequence(
    GameMode::Context context,
    int episode,
    std::set<data::Bonus> achievedBonuses,
    int scoreWithoutBonuses);

  void updateAndRender(engine::TimeDelta dt);
  void handleEvent(const SDL_Event& event);

  bool finished() const;

private:
  template <typename T>
  void startNewStage(T&& newStage);

  struct InitialWait
  {
    engine::TimeDelta mElapsedTime = {};
  };

  using Stage = std::variant<
    InitialWait,
    ui::EpisodeEndScreen,
    ui::Duke3DTeaserScreen,
    ui::BonusScreen>;

  Stage mStage = InitialWait{};
  GameMode::Context mContext;
  std::set<data::Bonus> mAchievedBonuses;
  int mEpisode;
  int mScoreWithoutBonuses;
};

} // namespace rigel::ui
