#ifdef _DEBUG
#pragma once

#include <string_view>

#include <SDL.h>

#include "engine/surface.hpp"

namespace devilution {

void InitConsole();
bool IsConsoleOpen();
void OpenConsole();
bool ConsoleHandleEvent(const SDL_Event &event);
void DrawConsole(const Surface &out);
void RunInConsole(std::string_view code);
void PrintToConsole(std::string_view text);
void PrintWarningToConsole(std::string_view text);

} // namespace devilution
#endif // _DEBUG
