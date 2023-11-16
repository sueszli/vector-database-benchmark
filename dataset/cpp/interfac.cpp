/**
 * @file interfac.cpp
 *
 * Implementation of load screens.
 */

#include <cstdint>
#include <optional>

#include <SDL.h>

#include "control.h"
#include "engine.h"
#include "engine/clx_sprite.hpp"
#include "engine/demomode.h"
#include "engine/dx.h"
#include "engine/events.hpp"
#include "engine/load_cel.hpp"
#include "engine/load_clx.hpp"
#include "engine/load_pcx.hpp"
#include "engine/palette.h"
#include "engine/render/clx_render.hpp"
#include "hwcursor.hpp"
#include "init.h"
#include "loadsave.h"
#include "pfile.h"
#include "plrmsg.h"
#include "utils/sdl_geometry.h"

namespace devilution {

namespace {

constexpr uint32_t MaxProgress = 534;

OptionalOwnedClxSpriteList sgpBackCel;

bool IsProgress;
uint32_t sgdwProgress;
int progress_id;

/** The color used for the progress bar as an index into the palette. */
const uint8_t BarColor[3] = { 138, 43, 254 };
/** The screen position of the top left corner of the progress bar. */
const int BarPos[3][2] = { { 53, 37 }, { 53, 421 }, { 53, 37 } };

OptionalOwnedClxSpriteList ArtCutsceneWidescreen;

uint16_t CustomEventsBegin = SDL_USEREVENT;
constexpr uint16_t NumCustomEvents = WM_LAST - WM_FIRST + 1;

Cutscenes GetCutSceneFromLevelType(dungeon_type type)
{
	switch (type) {
	case DTYPE_TOWN:
		return CutTown;
	case DTYPE_CATHEDRAL:
		return CutLevel1;
	case DTYPE_CATACOMBS:
		return CutLevel2;
	case DTYPE_CAVES:
		return CutLevel3;
	case DTYPE_HELL:
		return CutLevel4;
	case DTYPE_NEST:
		return CutLevel6;
	case DTYPE_CRYPT:
		return CutLevel5;
	default:
		return CutLevel1;
	}
}

Cutscenes PickCutscene(interface_mode uMsg)
{
	switch (uMsg) {
	case WM_DIABLOADGAME:
	case WM_DIABNEWGAME:
		return CutStart;
	case WM_DIABRETOWN:
		return CutTown;
	case WM_DIABNEXTLVL:
	case WM_DIABPREVLVL:
	case WM_DIABTOWNWARP:
	case WM_DIABTWARPUP: {
		int lvl = MyPlayer->plrlevel;
		if (lvl == 1 && uMsg == WM_DIABNEXTLVL)
			return CutTown;
		if (lvl == 16 && uMsg == WM_DIABNEXTLVL)
			return CutGate;
		return GetCutSceneFromLevelType(GetLevelType(lvl));
	}
	case WM_DIABWARPLVL:
		return CutPortal;
	case WM_DIABSETLVL:
	case WM_DIABRTNLVL:
		if (setlvlnum == SL_BONECHAMB)
			return CutLevel2;
		if (setlvlnum == SL_VILEBETRAYER)
			return CutPortalRed;
		if (IsArenaLevel(setlvlnum)) {
			if (uMsg == WM_DIABSETLVL)
				return GetCutSceneFromLevelType(setlvltype);
			return CutTown;
		}
		return CutLevel1;
	default:
		app_fatal("Unknown progress mode");
	}
}

void LoadCutsceneBackground(interface_mode uMsg)
{
	const char *celPath;
	const char *palPath;

	switch (PickCutscene(uMsg)) {
	case CutStart:
		ArtCutsceneWidescreen = LoadOptionalClx("gendata\\cutstartw.clx");
		celPath = "gendata\\cutstart";
		palPath = "gendata\\cutstart.pal";
		progress_id = 1;
		break;
	case CutTown:
		ArtCutsceneWidescreen = LoadOptionalClx("gendata\\cutttw.clx");
		celPath = "gendata\\cuttt";
		palPath = "gendata\\cuttt.pal";
		progress_id = 1;
		break;
	case CutLevel1:
		ArtCutsceneWidescreen = LoadOptionalClx("gendata\\cutl1dw.clx");
		celPath = "gendata\\cutl1d";
		palPath = "gendata\\cutl1d.pal";
		progress_id = 0;
		break;
	case CutLevel2:
		ArtCutsceneWidescreen = LoadOptionalClx("gendata\\cut2w.clx");
		celPath = "gendata\\cut2";
		palPath = "gendata\\cut2.pal";
		progress_id = 2;
		break;
	case CutLevel3:
		ArtCutsceneWidescreen = LoadOptionalClx("gendata\\cut3w.clx");
		celPath = "gendata\\cut3";
		palPath = "gendata\\cut3.pal";
		progress_id = 1;
		break;
	case CutLevel4:
		ArtCutsceneWidescreen = LoadOptionalClx("gendata\\cut4w.clx");
		celPath = "gendata\\cut4";
		palPath = "gendata\\cut4.pal";
		progress_id = 1;
		break;
	case CutLevel5:
		ArtCutsceneWidescreen = LoadOptionalClx("nlevels\\cutl5w.clx");
		celPath = "nlevels\\cutl5";
		palPath = "nlevels\\cutl5.pal";
		progress_id = 1;
		break;
	case CutLevel6:
		ArtCutsceneWidescreen = LoadOptionalClx("nlevels\\cutl6w.clx");
		celPath = "nlevels\\cutl6";
		palPath = "nlevels\\cutl6.pal";
		progress_id = 1;
		break;
	case CutPortal:
		ArtCutsceneWidescreen = LoadOptionalClx("gendata\\cutportlw.clx");
		celPath = "gendata\\cutportl";
		palPath = "gendata\\cutportl.pal";
		progress_id = 1;
		break;
	case CutPortalRed:
		ArtCutsceneWidescreen = LoadOptionalClx("gendata\\cutportrw.clx");
		celPath = "gendata\\cutportr";
		palPath = "gendata\\cutportr.pal";
		progress_id = 1;
		break;
	case CutGate:
		ArtCutsceneWidescreen = LoadOptionalClx("gendata\\cutgatew.clx");
		celPath = "gendata\\cutgate";
		palPath = "gendata\\cutgate.pal";
		progress_id = 1;
		break;
	}

	assert(!sgpBackCel);
	sgpBackCel = LoadCel(celPath, 640);
	LoadPalette(palPath);

	sgdwProgress = 0;
}

void FreeCutsceneBackground()
{
	sgpBackCel = std::nullopt;
	ArtCutsceneWidescreen = std::nullopt;
}

void DrawCutsceneBackground()
{
	const Rectangle &uiRectangle = GetUIRectangle();
	const Surface &out = GlobalBackBuffer();
	SDL_FillRect(out.surface, nullptr, 0x000000);
	if (ArtCutsceneWidescreen) {
		const ClxSprite sprite = (*ArtCutsceneWidescreen)[0];
		RenderClxSprite(out, sprite, { uiRectangle.position.x - (sprite.width() - uiRectangle.size.width) / 2, uiRectangle.position.y });
	}
	ClxDraw(out, { uiRectangle.position.x, 480 - 1 + uiRectangle.position.y }, (*sgpBackCel)[0]);
}

void DrawCutsceneForeground()
{
	const Rectangle &uiRectangle = GetUIRectangle();
	const Surface &out = GlobalBackBuffer();
	constexpr int ProgressHeight = 22;
	SDL_Rect rect = MakeSdlRect(
	    out.region.x + BarPos[progress_id][0] + uiRectangle.position.x,
	    out.region.y + BarPos[progress_id][1] + uiRectangle.position.y,
	    sgdwProgress,
	    ProgressHeight);
	SDL_FillRect(out.surface, &rect, BarColor[progress_id]);

	if (DiabloUiSurface() == PalSurface)
		BltFast(&rect, &rect);
	RenderPresent();
}

} // namespace

void RegisterCustomEvents()
{
#ifndef USE_SDL1
	CustomEventsBegin = SDL_RegisterEvents(NumCustomEvents);
#endif
}

bool IsCustomEvent(uint16_t eventType)
{
	return eventType >= CustomEventsBegin && eventType < CustomEventsBegin + NumCustomEvents;
}

interface_mode GetCustomEvent(uint16_t eventType)
{
	return static_cast<interface_mode>(eventType - CustomEventsBegin);
}

uint16_t CustomEventToSdlEvent(interface_mode eventType)
{
	return CustomEventsBegin + eventType;
}

void interface_msg_pump()
{
	SDL_Event event;
	uint16_t modState;
	while (FetchMessage(&event, &modState)) {
		if (event.type != SDL_QUIT) {
			HandleMessage(event, modState);
		}
	}
}

void IncProgress()
{
	if (!HeadlessMode && !demo::IsRunning())
		interface_msg_pump();
	if (!IsProgress)
		return;
	sgdwProgress += 23;
	if (sgdwProgress > MaxProgress)
		sgdwProgress = MaxProgress;
	if (!HeadlessMode && !demo::IsRunning())
		DrawCutsceneForeground();
}

void CompleteProgress()
{
	if (HeadlessMode)
		return;
	if (!IsProgress)
		return;
	while (sgdwProgress < MaxProgress)
		IncProgress();
}

void ShowProgress(interface_mode uMsg)
{
	IsProgress = true;

	gbSomebodyWonGameKludge = false;
	plrmsg_delay(true);

	EventHandler previousHandler = SetEventHandler(DisableInputEventHandler);

	if (!HeadlessMode) {
		assert(ghMainWnd);

		interface_msg_pump();
		ClearScreenBuffer();
		scrollrt_draw_game_screen();

		if (IsHardwareCursor())
			SetHardwareCursorVisible(false);

		BlackPalette();

		// Blit the background once and then free it.
		LoadCutsceneBackground(uMsg);
		DrawCutsceneBackground();
		if (RenderDirectlyToOutputSurface && PalSurface != nullptr) {
			// Render into all the backbuffers if there are multiple.
			const void *initialPixels = PalSurface->pixels;
			if (DiabloUiSurface() == PalSurface)
				BltFast(nullptr, nullptr);
			RenderPresent();
			while (PalSurface->pixels != initialPixels) {
				DrawCutsceneBackground();
				if (DiabloUiSurface() == PalSurface)
					BltFast(nullptr, nullptr);
				RenderPresent();
			}
		}
		FreeCutsceneBackground();

		PaletteFadeIn(8);
		IncProgress();
		sound_init();
		IncProgress();
	}

	Player &myPlayer = *MyPlayer;

	switch (uMsg) {
	case WM_DIABLOADGAME:
		IncProgress();
		IncProgress();
		LoadGame(true);
		IncProgress();
		IncProgress();
		break;
	case WM_DIABNEWGAME:
		myPlayer.pOriginalCathedral = !gbIsHellfire;
		IncProgress();
		FreeGameMem();
		IncProgress();
		pfile_remove_temp_files();
		IncProgress();
		LoadGameLevel(true, ENTRY_MAIN);
		IncProgress();
		break;
	case WM_DIABNEXTLVL:
		IncProgress();
		if (!gbIsMultiplayer) {
			pfile_save_level();
		} else {
			DeltaSaveLevel();
		}
		IncProgress();
		FreeGameMem();
		setlevel = false;
		currlevel = myPlayer.plrlevel;
		leveltype = GetLevelType(currlevel);
		IncProgress();
		LoadGameLevel(false, ENTRY_MAIN);
		IncProgress();
		break;
	case WM_DIABPREVLVL:
		IncProgress();
		if (!gbIsMultiplayer) {
			pfile_save_level();
		} else {
			DeltaSaveLevel();
		}
		IncProgress();
		FreeGameMem();
		currlevel--;
		leveltype = GetLevelType(currlevel);
		assert(myPlayer.isOnActiveLevel());
		IncProgress();
		LoadGameLevel(false, ENTRY_PREV);
		IncProgress();
		break;
	case WM_DIABSETLVL:
		// Note: ReturnLevel, ReturnLevelType and ReturnLvlPosition is only set to ensure vanilla compatibility
		ReturnLevel = GetMapReturnLevel();
		ReturnLevelType = GetLevelType(ReturnLevel);
		ReturnLvlPosition = GetMapReturnPosition();
		IncProgress();
		if (!gbIsMultiplayer) {
			pfile_save_level();
		} else {
			DeltaSaveLevel();
		}
		IncProgress();
		setlevel = true;
		leveltype = setlvltype;
		currlevel = static_cast<uint8_t>(setlvlnum);
		FreeGameMem();
		IncProgress();
		LoadGameLevel(false, ENTRY_SETLVL);
		IncProgress();
		break;
	case WM_DIABRTNLVL:
		IncProgress();
		if (!gbIsMultiplayer) {
			pfile_save_level();
		} else {
			DeltaSaveLevel();
		}
		IncProgress();
		setlevel = false;
		FreeGameMem();
		IncProgress();
		currlevel = GetMapReturnLevel();
		leveltype = GetLevelType(currlevel);
		LoadGameLevel(false, ENTRY_RTNLVL);
		IncProgress();
		break;
	case WM_DIABWARPLVL:
		IncProgress();
		if (!gbIsMultiplayer) {
			pfile_save_level();
		} else {
			DeltaSaveLevel();
		}
		IncProgress();
		FreeGameMem();
		GetPortalLevel();
		IncProgress();
		LoadGameLevel(false, ENTRY_WARPLVL);
		IncProgress();
		break;
	case WM_DIABTOWNWARP:
		IncProgress();
		if (!gbIsMultiplayer) {
			pfile_save_level();
		} else {
			DeltaSaveLevel();
		}
		IncProgress();
		FreeGameMem();
		setlevel = false;
		currlevel = myPlayer.plrlevel;
		leveltype = GetLevelType(currlevel);
		IncProgress();
		LoadGameLevel(false, ENTRY_TWARPDN);
		IncProgress();
		break;
	case WM_DIABTWARPUP:
		IncProgress();
		if (!gbIsMultiplayer) {
			pfile_save_level();
		} else {
			DeltaSaveLevel();
		}
		IncProgress();
		FreeGameMem();
		currlevel = myPlayer.plrlevel;
		leveltype = GetLevelType(currlevel);
		IncProgress();
		LoadGameLevel(false, ENTRY_TWARPUP);
		IncProgress();
		break;
	case WM_DIABRETOWN:
		IncProgress();
		if (!gbIsMultiplayer) {
			pfile_save_level();
		} else {
			DeltaSaveLevel();
		}
		IncProgress();
		FreeGameMem();
		setlevel = false;
		currlevel = myPlayer.plrlevel;
		leveltype = GetLevelType(currlevel);
		IncProgress();
		LoadGameLevel(false, ENTRY_MAIN);
		IncProgress();
		break;
	}

	if (!HeadlessMode) {
		assert(ghMainWnd);

		if (RenderDirectlyToOutputSurface && PalSurface != nullptr) {
			// Ensure that all back buffers have the full progress bar.
			const void *initialPixels = PalSurface->pixels;
			do {
				DrawCutsceneForeground();
				if (DiabloUiSurface() == PalSurface)
					BltFast(nullptr, nullptr);
				RenderPresent();
			} while (PalSurface->pixels != initialPixels);
		}

		PaletteFadeOut(8);
	}

	previousHandler = SetEventHandler(previousHandler);
	assert(previousHandler == DisableInputEventHandler);
	IsProgress = false;

	NetSendCmdLocParam2(true, CMD_PLAYER_JOINLEVEL, myPlayer.position.tile, myPlayer.plrlevel, myPlayer.plrIsOnSetLevel ? 1 : 0);
	plrmsg_delay(false);

	if (gbSomebodyWonGameKludge && myPlayer.isOnLevel(16)) {
		PrepDoEnding();
	}

	gbSomebodyWonGameKludge = false;
}

} // namespace devilution
