
#include <libtcod/console_drawing.h>

#include <catch2/catch_all.hpp>

#include "common.hpp"

TEST_CASE("Console rect") {
  auto console = tcod::Console{32, 32};
  tcod::draw_rect(console, {2, 2, 24, 24}, 0, std::nullopt, {{255, 0, 0}});
  tcod::draw_rect(console, {8, 8, 16, 1}, '-', tcod::ColorRGB{255, 255, 255}, std::nullopt);
}
