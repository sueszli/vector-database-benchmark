/**
 * @file plrmsg.cpp
 *
 * Implementation of functionality for printing the ingame chat messages.
 */
#include "plrmsg.h"

#include <algorithm>
#include <cstdint>

#include <fmt/format.h>

#include "control.h"
#include "engine/render/text_render.hpp"
#include "inv.h"
#include "qol/chatlog.h"
#include "qol/stash.h"
#include "utils/algorithm/container.hpp"
#include "utils/language.h"
#include "utils/utf8.hpp"

namespace devilution {

namespace {

struct PlayerMessage {
	/** Time message was received */
	Uint32 time;
	/** The default text color */
	UiFlags style;
	/** The text message to display on screen */
	std::string text;
	/** Length of first portion of text that should be rendered in gold */
	size_t prefixLength;
	/** The line height of the text */
	int lineHeight;
};

std::array<PlayerMessage, 8> Messages;

int CountLinesOfText(std::string_view text)
{
	return 1 + c_count(text, '\n');
}

PlayerMessage &GetNextMessage()
{
	std::move_backward(Messages.begin(), Messages.end() - 1, Messages.end()); // Push back older messages

	return Messages.front();
}

} // namespace

void plrmsg_delay(bool delay)
{
	static uint32_t plrmsgTicks;

	if (delay) {
		plrmsgTicks = -SDL_GetTicks();
		return;
	}

	plrmsgTicks += SDL_GetTicks();
	for (PlayerMessage &message : Messages)
		message.time += plrmsgTicks;
}

void EventPlrMsg(std::string_view text, UiFlags style)
{
	PlayerMessage &message = GetNextMessage();

	message.style = style;
	message.time = SDL_GetTicks();
	message.text = std::string(text);
	message.prefixLength = 0;
	message.lineHeight = GetLineHeight(message.text, GameFont12) + 3;
	AddMessageToChatLog(text);
}

void SendPlrMsg(Player &player, std::string_view text)
{
	PlayerMessage &message = GetNextMessage();

	std::string from = fmt::format(fmt::runtime(_("{:s} (lvl {:d}): ")), player._pName, player.getCharacterLevel());

	message.style = UiFlags::ColorWhite;
	message.time = SDL_GetTicks();
	message.text = from + std::string(text);
	message.prefixLength = from.size();
	message.lineHeight = GetLineHeight(message.text, GameFont12) + 3;
	AddMessageToChatLog(text, &player);
}

void InitPlrMsg()
{
	Messages = {};
}

void DrawPlrMsg(const Surface &out)
{
	if (ChatLogFlag)
		return;

	int x = 10;
	int y = GetMainPanel().position.y - 13;
	int width = gnScreenWidth - 20;

	if (!talkflag && IsLeftPanelOpen()) {
		x += GetLeftPanel().position.x + GetLeftPanel().size.width;
		width -= GetLeftPanel().size.width;
	}
	if (!talkflag && IsRightPanelOpen())
		width -= gnScreenWidth - GetRightPanel().position.x;

	if (width < 300)
		return;

	width = std::min(540, width);

	for (PlayerMessage &message : Messages) {
		if (message.text.empty())
			break;
		if (!talkflag && SDL_GetTicks() - message.time >= 10000)
			break;

		std::string text = WordWrapString(message.text, width);
		int chatlines = CountLinesOfText(text);
		y -= message.lineHeight * chatlines;

		DrawHalfTransparentRectTo(out, x - 3, y, width + 6, message.lineHeight * chatlines);

		std::vector<DrawStringFormatArg> args {
			{ std::string_view(text.data(), message.prefixLength), UiFlags::ColorWhitegold },
			{ std::string_view(text.data() + message.prefixLength, text.size() - message.prefixLength), message.style }
		};
		DrawStringWithColors(out, "{:s}{:s}", args, { { x, y }, { width, 0 } },
		    { .flags = UiFlags::None, .lineHeight = message.lineHeight });
	}
}

} // namespace devilution
