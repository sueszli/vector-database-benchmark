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

#include "game_main.hpp"

#include "version_info.hpp"

#include "base/defer.hpp"
#include "frontend/game.hpp"
#include "renderer/opengl.hpp"
#include "sdl_utils/error.hpp"
#include "ui/game_path_browser.hpp"
#include "ui/imgui_integration.hpp"
#include "ui/utils.hpp"

#include "platform.hpp"

RIGEL_DISABLE_WARNINGS
#include <SDL.h>
#include <SDL_mixer.h>
#include <imgui.h>
#include <loguru.hpp>
RIGEL_RESTORE_WARNINGS

#include <filesystem>
#include <fstream>


namespace rigel
{

using namespace sdl_utils;

namespace
{

void loadGameControllerDbForOldSdl()
{
  // SDL versions before 2.0.10 didn't check the SDL_GAMECONTROLLERCONFIG_FILE
  // env var. To make working with game controllers more consistent across
  // SDL versions, we implement this ourselves in case the SDL version being
  // used is older.
  SDL_version version;
  SDL_GetVersion(&version);

  if (version.patch < 10)
  {
    LOG_F(
      INFO,
      "SDL older than 2.0.10, manually checking SDL_GAMECONTROLLERCONFIG_FILE env var");
    if (const auto pMappingsFile = SDL_getenv("SDL_GAMECONTROLLERCONFIG_FILE"))
    {
      SDL_GameControllerAddMappingsFromFile(pMappingsFile);
    }
  }
}


bool isValidGamePath(const std::filesystem::path& path)
{
  namespace fs = std::filesystem;

  std::error_code ec;
  return fs::exists(path / "NUKEM2.CMP", ec) && !ec;
}


void showLoadingScreen(SDL_Window* pWindow)
{
  glClear(GL_COLOR_BUFFER_BIT);
  ui::imgui_integration::beginFrame(pWindow);

  ui::drawLoadingScreenText();

  ui::imgui_integration::endFrame();
  SDL_GL_SwapWindow(pWindow);
}


void setupForFirstLaunch(
  SDL_Window* pWindow,
  UserProfile& userProfile,
  std::string_view commandLineGamePath)
{
  namespace fs = std::filesystem;

  auto gamePath = fs::path{};

  // Case 1: A path is given on the command line on first launch. Use that.
  if (!commandLineGamePath.empty())
  {
    gamePath = fs::u8path(commandLineGamePath);
  }

  // Case 2: The current working directory is set to a Duke Nukem II
  // installation, most likely because the RigelEngine executable has been
  // copied there. Use the current working directory as game path.
  if (gamePath.empty())
  {
    const auto currentWorkingDir = fs::current_path();
    if (commandLineGamePath.empty() && isValidGamePath(currentWorkingDir))
    {
      gamePath = currentWorkingDir;
    }
  }

  // Case 3: Neither case 1 nor case 2 apply. Show a folder browser to let
  // the user select their Duke Nukem II installation.
  if (gamePath.empty())
  {
    gamePath = ui::runFolderBrowser(pWindow);
  }

  // If we still don't have a game path, stop here.
  if (gamePath.empty())
  {
    throw std::runtime_error(
      R"(No game path given. RigelEngine needs the original Duke Nukem II data files in order to function.
You can download the Shareware version for free, see
https://github.com/lethal-guitar/RigelEngine/blob/master/README.md#acquiring-the-game-data
for more info.)");
  }

  // Make sure there is a data file at the game path.
  if (!isValidGamePath(gamePath))
  {
    throw std::runtime_error(
      "No game data (NUKEM2.CMP file) found in game path");
  }

  // Import original game's profile data, if our profile is still 'empty'
  if (!userProfile.hasProgressData())
  {
    importOriginalGameProfileData(userProfile, gamePath.u8string() + "/");
  }

  // Finally, persist the chosen game path in the user profile for the next
  // launch.
  userProfile.mGamePath = fs::canonical(gamePath);
  userProfile.saveToDisk();
}


void createModsDirIfNoneFound(const std::filesystem::path& gamePath)
{
  namespace fs = std::filesystem;

  const auto modsDirPath = gamePath / data::MODS_PATH;

  std::error_code ec;
  if (fs::exists(modsDirPath, ec))
  {
    return;
  }

  if (fs::create_directory(modsDirPath, ec); ec)
  {
    LOG_F(ERROR, "Couldn't create mods directory: %s", ec.message().c_str());
    return;
  }

  std::ofstream instructionsFile(modsDirPath / "instructions.txt");
  instructionsFile <<
    R"(Place mods into this directory, and they will show up in the 'Modding' section in the game's options menu.
Each mod needs to be in a single directory (i.e., subdirectory of the 'mods' directory).

If you have a mod which consists of loose files, create a directory for it
first, and then place the files into that directory.

For more information, see https://github.com/lethal-guitar/RigelEngine/wiki/Modding-support

You can find some mods to get you started at https://github.com/lethal-guitar/RigelEngine/wiki/List-of-mods

You've made a mod and would like to share it with the community?
Stop by the 'modding' channel on the RigelEngine Discord: https://discord.gg/dtJQPFYD
)";

  LOG_F(INFO, "Created empty 'mods' directory and instructions.txt");
}


void initAndRunGame(
  SDL_Window* pWindow,
  UserProfile& userProfile,
  const CommandLineOptions& commandLineOptions)
{
  auto run = [&](const CommandLineOptions& options, const bool isFirstLaunch) {
    showLoadingScreen(pWindow);

    const auto gamePath = effectiveGamePath(commandLineOptions, userProfile);

    createModsDirIfNoneFound(gamePath);

    // Set up mod library with effective game path. This will automatically do
    // a rescan, which is important in case available mods have changed since
    // the last run.
    LOG_F(INFO, "Setting up mod library");
    userProfile.mModLibrary.updateGamePath(gamePath);

    // The mod library might now have the changed flag set, but we don't want
    // the game to see the flag since that would cause the game to immediately
    // exit again requesting a restart.  Since the game hasn't been
    // instantiated yet, the changed flag is meaningless anyway since the game
    // will use the current up-to-date state of the mod library during
    // Initialization.
    userProfile.mModLibrary.clearSelectionChangedFlag();

    // Now initialize and run the game until it tells us that it's done
    LOG_F(INFO, "Starting game");
    Game game(options, &userProfile, pWindow, isFirstLaunch);

    for (;;)
    {
      auto maybeStopReason = game.runOneFrame();
      if (maybeStopReason)
      {
        return *maybeStopReason;
      }
    }
  };

  const auto needsProfileSetup = !userProfile.mGamePath.has_value() ||
    !isValidGamePath(*userProfile.mGamePath);
  if (needsProfileSetup)
  {
    setupForFirstLaunch(pWindow, userProfile, commandLineOptions.mGamePath);
  }

  auto result = run(
    commandLineOptions, needsProfileSetup && !userProfile.hasProgressData());

  // Some game option changes (like choosing a new game path) require
  // restarting the game to make the change effective. If the first game run
  // ended with a result of RestartNeeded, launch a new game, but start from
  // the main menu and discard most command line options.
  if (result == Game::StopReason::RestartNeeded)
  {
    LOG_F(INFO, "Game requested restart");

    auto optionsForRestartedGame = CommandLineOptions{};
    optionsForRestartedGame.mSkipIntro = true;
    optionsForRestartedGame.mDebugModeEnabled =
      commandLineOptions.mDebugModeEnabled;
    optionsForRestartedGame.mDisableAudio = commandLineOptions.mDisableAudio;

    while (result == Game::StopReason::RestartNeeded)
    {
      result = run(optionsForRestartedGame, false);
    }
  }

  // We're exiting, save the user profile
  LOG_F(INFO, "Game ended");
  userProfile.saveToDisk();
}


void logVersionAndSystemInfo()
{
  LOG_F(
    INFO,
    "RigelEngine v%d.%d.%d (commit %s) - %s renderer",
    VERSION_MAJOR,
    VERSION_MINOR,
    VERSION_PATCH,
    COMMIT_HASH,
    renderer::OPENGL_VARIANT_NAME);

  SDL_version sdlVersion;
  SDL_GetVersion(&sdlVersion);

  const auto pSdlMixerVersion = Mix_Linked_Version();

  LOG_F(
    INFO,
    "Using SDL v%d.%d.%d - SDL Mixer v%d.%d.%d",
    sdlVersion.major,
    sdlVersion.minor,
    sdlVersion.patch,
    pSdlMixerVersion->major,
    pSdlMixerVersion->minor,
    pSdlMixerVersion->patch);
}

} // namespace


int gameMain(const CommandLineOptions& options)
{
  using base::defer;

  logVersionAndSystemInfo();

  loadGameControllerDbForOldSdl();

  LOG_F(INFO, "Initializing SDL");
  sdl_utils::check(
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_GAMECONTROLLER));
  auto sdlGuard = defer([]() { SDL_Quit(); });

  LOG_F(INFO, "Initializing SDL_mixer");
  Mix_Init(MIX_INIT_FLAC | MIX_INIT_OGG | MIX_INIT_MP3 | MIX_INIT_MOD);
  auto sdlMixerGuard = defer([]() { Mix_Quit(); });

  LOG_F(
    INFO,
    "SDL backends: %s, %s",
    SDL_GetCurrentVideoDriver(),
    SDL_GetCurrentAudioDriver());

  sdl_utils::check(SDL_GL_LoadLibrary(nullptr));
  platform::setGLAttributes();

  LOG_F(INFO, "Loading user profile");
  auto userProfile = loadOrCreateUserProfile();

  LOG_F(INFO, "Creating window");
  auto pWindow = platform::createWindow(userProfile.mOptions);

  LOG_F(INFO, "Initializing OpenGL context");
  SDL_GLContext pGlContext =
    sdl_utils::check(SDL_GL_CreateContext(pWindow.get()));
  auto glGuard = defer([pGlContext]() { SDL_GL_DeleteContext(pGlContext); });

  LOG_F(INFO, "Loading OpenGL function pointers");
  renderer::loadGlFunctions();

  // On some platforms, an initial swap is necessary in order for the next
  // frame (in our case, the loading screen) to show up on screen.
  SDL_GL_SetSwapInterval(data::ENABLE_VSYNC_DEFAULT ? 1 : 0);
  SDL_GL_SwapWindow(pWindow.get());

  SDL_DisableScreenSaver();
  SDL_ShowCursor(SDL_DISABLE);

  LOG_F(INFO, "Initializing Dear ImGui");
  ui::imgui_integration::init(
    pWindow.get(), pGlContext, createOrGetPreferencesPath());
  auto imGuiGuard = defer([]() { ui::imgui_integration::shutdown(); });

  try
  {
    initAndRunGame(pWindow.get(), userProfile, options);
  }
  catch (const std::exception& error)
  {
    LOG_F(ERROR, "%s", error.what());
    ui::showErrorMessage(pWindow.get(), error.what());
    return -2;
  }

  LOG_F(INFO, "Exiting");

  return 0;
}

} // namespace rigel
