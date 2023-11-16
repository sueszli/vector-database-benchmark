#include <optional>

#include "DiabloUI/diabloui.h"
#include "control.h"
#include "controls/input.h"
#include "controls/menu_controls.h"
#include "discord/discord.h"
#include "engine/load_clx.hpp"
#include "engine/load_pcx.hpp"
#include "utils/algorithm/container.hpp"
#include "utils/language.h"
#include "utils/sdl_geometry.h"

namespace devilution {
namespace {

OptionalOwnedClxSpriteList DiabloTitleLogo;

std::vector<std::unique_ptr<UiItemBase>> vecTitleScreen;

void TitleLoad()
{
	if (gbIsHellfire) {
		LoadBackgroundArt("ui_art\\hf_logo1", 16);
		ArtBackgroundWidescreen = LoadOptionalClx("ui_art\\hf_titlew.clx");
	} else {
		LoadBackgroundArt("ui_art\\title");
		DiabloTitleLogo = LoadPcxSpriteList("ui_art\\logo", /*numFrames=*/15, /*transparentColor=*/250);
	}
}

void TitleFree()
{
	ArtBackground = std::nullopt;
	ArtBackgroundWidescreen = std::nullopt;
	DiabloTitleLogo = std::nullopt;

	vecTitleScreen.clear();
}

} // namespace

void UiTitleDialog()
{
	TitleLoad();
	const Point uiPosition = GetUIRectangle().position;
	if (gbIsHellfire) {
		SDL_Rect rect = MakeSdlRect(0, uiPosition.y, 0, 0);
		if (ArtBackgroundWidescreen)
			vecTitleScreen.push_back(std::make_unique<UiImageClx>((*ArtBackgroundWidescreen)[0], rect, UiFlags::AlignCenter));
		vecTitleScreen.push_back(std::make_unique<UiImageAnimatedClx>(*ArtBackground, rect, UiFlags::AlignCenter));
	} else {
		UiAddBackground(&vecTitleScreen);

		vecTitleScreen.push_back(std::make_unique<UiImageAnimatedClx>(
		    *DiabloTitleLogo, MakeSdlRect(0, uiPosition.y + 182, 0, 0), UiFlags::AlignCenter));

		SDL_Rect rect = MakeSdlRect(uiPosition.x, uiPosition.y + 410, 640, 26);
		vecTitleScreen.push_back(std::make_unique<UiArtText>(_("Copyright © 1996-2001 Blizzard Entertainment").data(), rect, UiFlags::AlignCenter | UiFlags::FontSize24 | UiFlags::ColorUiSilver));
	}

	bool endMenu = false;
	Uint32 timeOut = SDL_GetTicks() + 7000;

	SDL_Event event;
	while (!endMenu && SDL_GetTicks() < timeOut) {
		UiRenderItems(vecTitleScreen);
		UiFadeIn();

		discord_manager::UpdateMenu();

		while (PollEvent(&event) != 0) {
			if (c_any_of(GetMenuActions(event), [](MenuAction menuAction) { return menuAction != MenuAction_NONE; })) {
				endMenu = true;
				break;
			}
			switch (event.type) {
			case SDL_KEYDOWN:
			case SDL_MOUSEBUTTONUP:
				endMenu = true;
				break;
			}
			UiHandleEvents(&event);
		}
	}

	TitleFree();
}

} // namespace devilution
