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

#include "assets/duke_script_loader.hpp"
#include "data/duke_script.hpp"
#include "frontend/game_mode.hpp"
#include "ui/duke_script_runner.hpp"
#include "ui/options_menu.hpp"

#include <optional>


namespace rigel
{

class MenuMode : public GameMode
{
public:
  explicit MenuMode(Context context);

  std::unique_ptr<GameMode> updateAndRender(
    engine::TimeDelta dt,
    const std::vector<SDL_Event>& events) override;

private:
  enum class MenuState
  {
    AskIfQuit,
    ChooseInstructionsOrStory,
    EpisodeNotAvailableMessage,
    EpisodeNotAvailableMessageHighScores,
    NoSavedGameInSlotMessage,
    Instructions,
    MainMenu,
    OrderingInformation,
    RestoreGame,
    SelectHighscoresEpisode,
    SelectNewGameEpisode,
    SelectNewGameSkill,
    ShowCredits,
    ShowHiscores,
    Story
  };

  void handleEvent(const SDL_Event& event);
  void enterMainMenu();
  std::unique_ptr<GameMode>
    navigateToNextMenu(const ui::DukeScriptRunner::ExecutionResult& result);

private:
  Context mContext;
  std::optional<ui::OptionsMenu> mOptionsMenu;
  MenuState mMenuState = MenuState::MainMenu;
  int mChosenEpisode = 0;
};

} // namespace rigel
