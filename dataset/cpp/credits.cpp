#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "DiabloUI/credits_lines.h"
#include "DiabloUI/diabloui.h"
#include "DiabloUI/support_lines.h"
#include "control.h"
#include "controls/input.h"
#include "controls/menu_controls.h"
#include "engine/load_clx.hpp"
#include "engine/render/clx_render.hpp"
#include "engine/render/text_render.hpp"
#include "hwcursor.hpp"
#include "utils/display.h"
#include "utils/language.h"
#include "utils/log.hpp"
#include "utils/sdl_compat.h"

namespace devilution {

namespace {

const SDL_Rect VIEWPORT = { 0, 114, 640, 251 };
const int LINE_H = 22;

// The maximum number of visible lines is the number of whole lines
// (VIEWPORT.h / LINE_H) rounded up, plus one extra line for when
// a line is leaving the screen while another one is entering.
#define MAX_VISIBLE_LINES ((VIEWPORT.h - 1) / LINE_H + 2)

class CreditsRenderer {

public:
	CreditsRenderer(char const *const *text, std::size_t textLines)
	{
		for (size_t i = 0; i < textLines; i++) {
			std::string_view orgText = _(text[i]);

			uint16_t offset = 0;
			size_t indexFirstNotTab = 0;
			while (indexFirstNotTab < orgText.size() && orgText[indexFirstNotTab] == '\t') {
				offset += 40;
				indexFirstNotTab++;
			}

			const std::string paragraphs = WordWrapString(orgText.substr(indexFirstNotTab), 580 - offset, FontSizeDialog);

			size_t previous = 0;
			while (true) {
				size_t next = paragraphs.find('\n', previous);
				linesToRender.emplace_back(LineContent { offset, paragraphs.substr(previous, next - previous) });
				if (next == std::string::npos)
					break;
				previous = next + 1;
			}
		}

		ticks_begin_ = SDL_GetTicks();
		prev_offset_y_ = 0;
		finished_ = false;
	}

	~CreditsRenderer()
	{
		ArtBackgroundWidescreen = std::nullopt;
		ArtBackground = std::nullopt;
	}

	void Render();

	[[nodiscard]] bool Finished() const
	{
		return finished_;
	}

private:
	struct LineContent {
		uint16_t offset;
		std::string text;
	};

	std::vector<LineContent> linesToRender;
	bool finished_;
	Uint32 ticks_begin_;
	int prev_offset_y_;
};

void CreditsRenderer::Render()
{
	const int offsetY = -VIEWPORT.h + (SDL_GetTicks() - ticks_begin_) / 40;
	if (offsetY == prev_offset_y_)
		return;
	prev_offset_y_ = offsetY;

	SDL_FillRect(DiabloUiSurface(), nullptr, 0x000000);
	const Point uiPosition = GetUIRectangle().position;
	if (ArtBackgroundWidescreen)
		RenderClxSprite(Surface(DiabloUiSurface()), (*ArtBackgroundWidescreen)[0], uiPosition - Displacement { 320, 0 });
	RenderClxSprite(Surface(DiabloUiSurface()), (*ArtBackground)[0], uiPosition);

	const std::size_t linesBegin = std::max(offsetY / LINE_H, 0);
	const std::size_t linesEnd = std::min(linesBegin + MAX_VISIBLE_LINES, linesToRender.size());

	if (linesBegin >= linesEnd) {
		if (linesEnd == linesToRender.size())
			finished_ = true;
		return;
	}

	SDL_Rect viewport = VIEWPORT;
	viewport.x += uiPosition.x;
	viewport.y += uiPosition.y;
	ScaleOutputRect(&viewport);

	// We use unscaled coordinates for calculation throughout.
	Sint16 destY = uiPosition.y + VIEWPORT.y - (offsetY - linesBegin * LINE_H);
	for (std::size_t i = linesBegin; i < linesEnd; ++i, destY += LINE_H) {
		Sint16 destX = uiPosition.x + VIEWPORT.x + 31;

		auto &lineContent = linesToRender[i];

		SDL_Rect dstRect = MakeSdlRect(destX + lineContent.offset, destY, 0, 0);
		ScaleOutputRect(&dstRect);
		dstRect.x -= viewport.x;
		dstRect.y -= viewport.y;

		const Surface &out = Surface(DiabloUiSurface(), viewport);
		DrawString(out, lineContent.text, Point { dstRect.x, dstRect.y },
		    { .flags = UiFlags::FontSizeDialog | UiFlags::ColorDialogWhite, .spacing = -1 });
	}
}

bool TextDialog(char const *const *text, std::size_t textLines)
{
	CreditsRenderer creditsRenderer(text, textLines);
	bool endMenu = false;

	if (IsHardwareCursor())
		SetHardwareCursorVisible(false);

	SDL_Event event;
	do {
		creditsRenderer.Render();
		UiFadeIn();
		while (PollEvent(&event) != 0) {
			switch (event.type) {
			case SDL_KEYDOWN:
			case SDL_MOUSEBUTTONUP:
				endMenu = true;
				break;
			default:
				for (MenuAction menuAction : GetMenuActions(event)) {
					if (IsNoneOf(menuAction, MenuAction_BACK, MenuAction_SELECT))
						continue;
					endMenu = true;
					break;
				}
				break;
			}
			UiHandleEvents(&event);
		}
	} while (!endMenu && !creditsRenderer.Finished());

	return true;
}

} // namespace

bool UiCreditsDialog()
{
	ArtBackgroundWidescreen = LoadOptionalClx("ui_art\\creditsw.clx");
	LoadBackgroundArt("ui_art\\credits");

	return TextDialog(CreditLines, CreditLinesSize);
}

bool UiSupportDialog()
{
	if (gbIsHellfire) {
		ArtBackgroundWidescreen = LoadOptionalClx("ui_art\\supportw.clx");
		LoadBackgroundArt("ui_art\\support");
	} else {
		ArtBackgroundWidescreen = LoadOptionalClx("ui_art\\creditsw.clx");
		LoadBackgroundArt("ui_art\\credits");
	}

	return TextDialog(SupportLines, SupportLinesSize);
}

} // namespace devilution
