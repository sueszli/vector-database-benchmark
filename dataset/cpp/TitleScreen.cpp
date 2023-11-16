/*****************************************************************************
 * Copyright (c) 2014-2023 OpenRCT2 developers
 *
 * For a complete list of all authors, please refer to contributors.md
 * Interested in contributing? Visit https://github.com/OpenRCT2/OpenRCT2
 *
 * OpenRCT2 is licensed under the GNU General Public License version 3.
 *****************************************************************************/

#include "TitleScreen.h"

#include "../Context.h"
#include "../Game.h"
#include "../GameState.h"
#include "../Input.h"
#include "../OpenRCT2.h"
#include "../Version.h"
#include "../audio/audio.h"
#include "../config/Config.h"
#include "../core/Console.hpp"
#include "../drawing/Drawing.h"
#include "../interface/Screenshot.h"
#include "../interface/Viewport.h"
#include "../interface/Window.h"
#include "../localisation/Localisation.h"
#include "../network/NetworkBase.h"
#include "../network/network.h"
#include "../scenario/Scenario.h"
#include "../scenario/ScenarioRepository.h"
#include "../ui/UiContext.h"
#include "../util/Util.h"
#include "TitleSequence.h"
#include "TitleSequenceManager.h"
#include "TitleSequencePlayer.h"

using namespace OpenRCT2;

// TODO Remove when no longer required.
bool gPreviewingTitleSequenceInGame;
static TitleScreen* _singleton = nullptr;

TitleScreen::TitleScreen(GameState& gameState)
    : _gameState(gameState)
{
    _singleton = this;
}

TitleScreen::~TitleScreen()
{
    _singleton = nullptr;
}

ITitleSequencePlayer* TitleScreen::GetSequencePlayer()
{
    return _sequencePlayer;
}

size_t TitleScreen::GetCurrentSequence()
{
    return _currentSequence;
}

bool TitleScreen::PreviewSequence(size_t value)
{
    _currentSequence = value;
    _previewingSequence = TryLoadSequence(true);
    if (_previewingSequence)
    {
        if (!(gScreenFlags & SCREEN_FLAGS_TITLE_DEMO))
        {
            gPreviewingTitleSequenceInGame = true;
        }
    }
    else
    {
        _currentSequence = TitleGetConfigSequence();
        if (gScreenFlags & SCREEN_FLAGS_TITLE_DEMO)
        {
            TryLoadSequence();
        }
    }
    return _previewingSequence;
}

void TitleScreen::StopPreviewingSequence()
{
    if (_previewingSequence)
    {
        WindowBase* mainWindow = WindowGetMain();
        if (mainWindow != nullptr)
        {
            WindowUnfollowSprite(*mainWindow);
        }
        _previewingSequence = false;
        _currentSequence = TitleGetConfigSequence();
        gPreviewingTitleSequenceInGame = false;
    }
}

bool TitleScreen::IsPreviewingSequence()
{
    return _previewingSequence;
}

bool TitleScreen::ShouldHideVersionInfo()
{
    return _hideVersionInfo;
}

void TitleScreen::SetHideVersionInfo(bool value)
{
    _hideVersionInfo = value;
}

void TitleScreen::Load()
{
    LOG_VERBOSE("TitleScreen::Load()");

    if (GameIsPaused())
    {
        PauseToggle();
    }

    gScreenFlags = SCREEN_FLAGS_TITLE_DEMO;
    gScreenAge = 0;
    gCurrentLoadedPath.clear();

#ifndef DISABLE_NETWORK
    GetContext()->GetNetwork().Close();
#endif
    OpenRCT2::Audio::StopAll();
    GetContext()->GetGameState()->InitAll(DEFAULT_MAP_SIZE);
    ViewportInitAll();
    ContextOpenWindow(WindowClass::MainWindow);
    CreateWindows();
    TitleInitialise();
    OpenRCT2::Audio::PlayTitleMusic();

    if (gOpenRCT2ShowChangelog)
    {
        gOpenRCT2ShowChangelog = false;
        ContextOpenWindow(WindowClass::Changelog);
    }

    if (_sequencePlayer != nullptr)
    {
        _sequencePlayer->Begin(_currentSequence);

        // Force the title sequence to load / update so we
        // don't see a blank screen for a split second.
        TryLoadSequence();
        _sequencePlayer->Update();
    }

    LOG_VERBOSE("TitleScreen::Load() finished");
}

void TitleScreen::Tick()
{
    gInUpdateCode = true;

    ScreenshotCheck();
    TitleHandleKeyboardInput();

    if (GameIsNotPaused())
    {
        TryLoadSequence();
        _sequencePlayer->Update();

        int32_t numUpdates = 1;
        if (gGameSpeed > 1)
        {
            numUpdates = 1 << (gGameSpeed - 1);
        }
        for (int32_t i = 0; i < numUpdates; i++)
        {
            _gameState.UpdateLogic();
        }
        UpdatePaletteEffects();
        // update_weather_animation();
    }

    InputSetFlag(INPUT_FLAG_VIEWPORT_SCROLLING, false);

    ContextUpdateMapTooltip();
    WindowDispatchUpdateAll();

    gSavedAge++;

    ContextHandleInput();

    gInUpdateCode = false;
}

void TitleScreen::ChangePresetSequence(size_t preset)
{
    size_t count = TitleSequenceManager::GetCount();
    if (preset >= count)
    {
        return;
    }

    const utf8* configId = TitleSequenceManagerGetConfigID(preset);
    gConfigInterface.CurrentTitleSequencePreset = configId;

    if (!_previewingSequence)
        _currentSequence = preset;
    WindowInvalidateAll();
}

/**
 * Creates the windows shown on the title screen; New game, load game,
 * tutorial, toolbox and exit.
 */
void TitleScreen::CreateWindows()
{
    ContextOpenWindow(WindowClass::TitleMenu);
    ContextOpenWindow(WindowClass::TitleExit);
    ContextOpenWindow(WindowClass::TitleOptions);
    ContextOpenWindow(WindowClass::TitleLogo);
    WindowResizeGui(ContextGetWidth(), ContextGetHeight());
    _hideVersionInfo = false;
}

void TitleScreen::TitleInitialise()
{
    if (_sequencePlayer == nullptr)
    {
        _sequencePlayer = GetContext()->GetUiContext()->GetTitleSequencePlayer();
    }
    if (gConfigInterface.RandomTitleSequence)
    {
        const size_t total = TitleSequenceManager::GetCount();
        if (total > 0)
        {
            bool RCT1Installed = false, RCT1AAInstalled = false, RCT1LLInstalled = false;
            uint32_t RCT1Count = 0;
            const size_t scenarioCount = ScenarioRepositoryGetCount();

            for (size_t s = 0; s < scenarioCount; s++)
            {
                const ScenarioSource sourceGame = ScenarioRepositoryGetByIndex(s)->SourceGame;
                switch (sourceGame)
                {
                    case ScenarioSource::RCT1:
                        RCT1Count++;
                        break;
                    case ScenarioSource::RCT1_AA:
                        RCT1AAInstalled = true;
                        break;
                    case ScenarioSource::RCT1_LL:
                        RCT1LLInstalled = true;
                        break;
                    default:
                        break;
                }
            }

            // Mega Park can show up in the scenario list even if RCT1 has been uninstalled, so it must be greater than 1
            RCT1Installed = RCT1Count > 1;

            int32_t random = 0;
            bool safeSequence = false;
            const std::string RCT1String = FormatStringID(STR_TITLE_SEQUENCE_RCT1, nullptr);
            const std::string RCT1AAString = FormatStringID(STR_TITLE_SEQUENCE_RCT1_AA, nullptr);
            const std::string RCT1LLString = FormatStringID(STR_TITLE_SEQUENCE_RCT1_AA_LL, nullptr);

            // Ensure the random sequence chosen isn't from RCT1 or expansion if the player doesn't have it installed
            while (!safeSequence)
            {
                random = UtilRand() % static_cast<int32_t>(total);
                const utf8* scName = TitleSequenceManagerGetName(random);
                if (scName == RCT1String)
                {
                    safeSequence = RCT1Installed;
                }
                else if (scName == RCT1AAString)
                {
                    safeSequence = RCT1AAInstalled;
                }
                else if (scName == RCT1LLString)
                {
                    safeSequence = RCT1LLInstalled;
                }
                else
                {
                    safeSequence = true;
                }
            }
            ChangePresetSequence(random);
        }
    }
    size_t seqId = TitleGetConfigSequence();
    if (seqId == SIZE_MAX)
    {
        seqId = TitleSequenceManagerGetIndexForConfigID("*OPENRCT2");
        if (seqId == SIZE_MAX)
        {
            seqId = 0;
        }
    }
    ChangePresetSequence(static_cast<int32_t>(seqId));
}

bool TitleScreen::TryLoadSequence(bool loadPreview)
{
    if (_loadedTitleSequenceId != _currentSequence || loadPreview)
    {
        if (_sequencePlayer == nullptr)
        {
            _sequencePlayer = GetContext()->GetUiContext()->GetTitleSequencePlayer();
        }

        size_t numSequences = TitleSequenceManager::GetCount();
        if (numSequences > 0)
        {
            size_t targetSequence = _currentSequence;
            do
            {
                if (_sequencePlayer->Begin(targetSequence) && _sequencePlayer->Update())
                {
                    _loadedTitleSequenceId = targetSequence;
                    if (targetSequence != _currentSequence && !loadPreview)
                    {
                        // Forcefully change the preset to a preset that works.
                        const utf8* configId = TitleSequenceManagerGetConfigID(targetSequence);
                        gConfigInterface.CurrentTitleSequencePreset = configId;
                    }
                    _currentSequence = targetSequence;
                    GfxInvalidateScreen();
                    return true;
                }
                targetSequence = (targetSequence + 1) % numSequences;
            } while (targetSequence != _currentSequence && !loadPreview);
        }
        Console::Error::WriteLine("Unable to play any title sequences.");
        _sequencePlayer->Eject();
        _currentSequence = SIZE_MAX;
        _loadedTitleSequenceId = SIZE_MAX;
        if (!loadPreview)
        {
            GetContext()->GetGameState()->InitAll(DEFAULT_MAP_SIZE);
            GameNotifyMapChanged();
        }
        return false;
    }
    return true;
}

void TitleLoad()
{
    if (_singleton != nullptr)
    {
        _singleton->Load();
    }
}

void TitleCreateWindows()
{
    if (_singleton != nullptr)
    {
        _singleton->CreateWindows();
    }
}

void* TitleGetSequencePlayer()
{
    void* result = nullptr;
    if (_singleton != nullptr)
    {
        result = _singleton->GetSequencePlayer();
    }
    return result;
}

void TitleSequenceChangePreset(size_t preset)
{
    if (_singleton != nullptr)
    {
        _singleton->ChangePresetSequence(preset);
    }
}

bool TitleShouldHideVersionInfo()
{
    bool result = false;
    if (_singleton != nullptr)
    {
        result = _singleton->ShouldHideVersionInfo();
    }
    return result;
}

void TitleSetHideVersionInfo(bool value)
{
    if (_singleton != nullptr)
    {
        _singleton->SetHideVersionInfo(value);
    }
}

size_t TitleGetConfigSequence()
{
    return TitleSequenceManagerGetIndexForConfigID(gConfigInterface.CurrentTitleSequencePreset.c_str());
}

size_t TitleGetCurrentSequence()
{
    size_t result = 0;
    if (_singleton != nullptr)
    {
        result = _singleton->GetCurrentSequence();
    }
    return result;
}

bool TitlePreviewSequence(size_t value)
{
    if (_singleton != nullptr)
    {
        return _singleton->PreviewSequence(value);
    }
    return false;
}

void TitleStopPreviewingSequence()
{
    if (_singleton != nullptr)
    {
        _singleton->StopPreviewingSequence();
    }
}

bool TitleIsPreviewingSequence()
{
    if (_singleton != nullptr)
    {
        return _singleton->IsPreviewingSequence();
    }
    return false;
}

void DrawOpenRCT2(DrawPixelInfo& dpi, const ScreenCoordsXY& screenCoords)
{
    thread_local std::string buffer;
    buffer.clear();
    buffer.assign("{OUTLINE}{WHITE}");

    // Write name and version information
    buffer += gVersionInfoFull;

    GfxDrawString(dpi, screenCoords + ScreenCoordsXY(5, 5 - 13), buffer.c_str(), { COLOUR_BLACK });
    int16_t width = static_cast<int16_t>(GfxGetStringWidth(buffer, FontStyle::Medium));

    // Write platform information
    buffer.assign("{OUTLINE}{WHITE}");
    buffer.append(OPENRCT2_PLATFORM);
    buffer.append(" (");
    buffer.append(OPENRCT2_ARCHITECTURE);
    buffer.append(")");

    GfxDrawString(dpi, screenCoords + ScreenCoordsXY(5, 5), buffer.c_str(), { COLOUR_BLACK });
    width = std::max(width, static_cast<int16_t>(GfxGetStringWidth(buffer, FontStyle::Medium)));

    // Invalidate screen area
    GfxSetDirtyBlocks({ screenCoords - ScreenCoordsXY(0, 13),
                        screenCoords + ScreenCoordsXY{ width + 5, 30 } }); // 30 is an arbitrary height to catch both strings
}
