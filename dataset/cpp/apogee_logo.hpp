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

#include "engine/timing.hpp"
#include "frontend/game_mode.hpp"
#include "ui/movie_player.hpp"


namespace rigel::ui
{

class ApogeeLogo
{
public:
  explicit ApogeeLogo(GameMode::Context context);

  void start();

  void updateAndRender(engine::TimeDelta dt);

  bool isFinished() const;

private:
  ui::MoviePlayer mMoviePlayer;
  IGameServiceProvider* mpServiceProvider;
  data::Movie mLogoMovie;

  engine::TimeDelta mElapsedTime;
};

} // namespace rigel::ui
