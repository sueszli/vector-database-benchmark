/* BSD 3-Clause License
 *
 * Copyright © 2008-2023, Jice and the libtcod contributors.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include <cstdarg>
#include <cstdio>
#include <cstring>

#include "console.hpp"
#include "console_init.h"
#include "console_printing.hpp"
#include "error.hpp"
#include "image.hpp"
#include "libtcod_int.h"

/***************************************************************************
    @brief Covert variable arguments into a std::string object.

    This function can throw an exception, so care should be taken handing va_list objects.

    @param fmt A format input string, must not be nullptr.
    @param varags A va_list to be used with the fmt string.
    @return std::string The formatted output.
 */
static auto vstring(const char* fmt, va_list varags) -> std::string {
  va_list varags_clone;
  va_copy(varags_clone, varags);
  const int str_length = vsnprintf(nullptr, 0, fmt, varags_clone);
  va_end(varags_clone);
  if (str_length < 0) throw std::runtime_error("Failed to format string.");
  std::string out(str_length, '\0');
  vsnprintf(&out[0], str_length + 1, fmt, varags);
  return out;
};

TCODConsole* TCODConsole::root = new TCODConsole();

TCODConsole::TCODConsole(int w, int h) : TCODConsole{TCOD_console_new(w, h)} {}

TCODConsole::TCODConsole(const char* filename) : TCODConsole{TCOD_console_from_file(filename)} {}

bool TCODConsole::loadAsc(const char* filename) { return TCOD_console_load_asc(data.get(), filename) != 0; }
bool TCODConsole::saveAsc(const char* filename) const { return TCOD_console_save_asc(data.get(), filename) != 0; }
bool TCODConsole::saveApf(const char* filename) const { return TCOD_console_save_apf(data.get(), filename) != 0; }
bool TCODConsole::loadApf(const char* filename) { return TCOD_console_load_apf(data.get(), filename) != 0; }

void TCODConsole::setCustomFont(const char* fontFile, int flags, int nbCharHoriz, int nbCharVertic) {
  TCOD_console_set_custom_font(fontFile, flags, nbCharHoriz, nbCharVertic);
}

void TCODConsole::mapAsciiCodeToFont(int asciiCode, int fontCharX, int fontCharY) {
  TCOD_console_map_ascii_code_to_font(asciiCode, fontCharX, fontCharY);
}

void TCODConsole::mapAsciiCodesToFont(int firstAsciiCode, int nbCodes, int fontCharX, int fontCharY) {
  TCOD_console_map_ascii_codes_to_font(firstAsciiCode, nbCodes, fontCharX, fontCharY);
}

void TCODConsole::mapStringToFont(const char* s, int fontCharX, int fontCharY) {
  TCOD_console_map_string_to_font(s, fontCharX, fontCharY);
}

void TCODConsole::setDirty(int, int, int, int) {}

#ifndef NO_SDL
TCOD_key_t TCODConsole::checkForKeypress(int flags) { return TCOD_sys_check_for_keypress(flags); }
TCOD_key_t TCODConsole::waitForKeypress(bool flush) { return TCOD_sys_wait_for_keypress(flush); }
bool TCODConsole::isWindowClosed() { return TCOD_console_is_window_closed() != 0; }
bool TCODConsole::hasMouseFocus() { return TCOD_console_has_mouse_focus() != 0; }
bool TCODConsole::isActive() { return TCOD_console_is_active() != 0; }
#endif  // NO_SDL

int TCODConsole::getWidth() const { return TCOD_console_get_width(data.get()); }

int TCODConsole::getHeight() const { return TCOD_console_get_height(data.get()); }

void TCODConsole::setColorControl(TCOD_colctrl_t con, const TCODColor& fore, const TCODColor& back) {
  TCOD_color_t b = {back.r, back.g, back.b}, f = {fore.r, fore.g, fore.b};
  TCOD_console_set_color_control(con, f, b);
}

TCODColor TCODConsole::getDefaultBackground() const {
  TCOD_color_t c = TCOD_console_get_default_background(data.get());
  TCODColor ret;
  ret.r = c.r;
  ret.g = c.g;
  ret.b = c.b;
  return ret;
}
TCODColor TCODConsole::getDefaultForeground() const { return TCOD_console_get_default_foreground(data.get()); }
void TCODConsole::setDefaultBackground(TCODColor back) {
  TCOD_color_t b = {back.r, back.g, back.b};
  TCOD_console_set_default_background(data.get(), b);
}
void TCODConsole::setDefaultForeground(TCODColor fore) {
  TCOD_color_t b = {fore.r, fore.g, fore.b};
  TCOD_console_set_default_foreground(data.get(), b);
}

#ifndef NO_SDL
void TCODConsole::setWindowTitle(const char* title) { TCOD_console_set_window_title(title); }
void TCODConsole::initRoot(int w, int h, const char* title, bool fullscreen, TCOD_renderer_t renderer) {
  tcod::check_throw_error(TCOD_console_init_root(w, h, title, fullscreen, renderer));
}
void TCODConsole::setFullscreen(bool fullscreen) { TCOD_console_set_fullscreen(fullscreen); }
bool TCODConsole::isFullscreen() { return TCOD_console_is_fullscreen() != 0; }
#endif  // NO_SDL

void TCODConsole::setBackgroundFlag(TCOD_bkgnd_flag_t bkgnd_flag) {
  TCOD_console_set_background_flag(data.get(), bkgnd_flag);
}

TCOD_bkgnd_flag_t TCODConsole::getBackgroundFlag() const { return TCOD_console_get_background_flag(data.get()); }

void TCODConsole::setAlignment(TCOD_alignment_t alignment) { TCOD_console_set_alignment(data.get(), alignment); }

TCOD_alignment_t TCODConsole::getAlignment() const { return TCOD_console_get_alignment(data.get()); }

void TCODConsole::blit(
    const TCODConsole* srcCon,
    int xSrc,
    int ySrc,
    int wSrc,
    int hSrc,
    TCODConsole* dstCon,
    int xDst,
    int yDst,
    float foreground_alpha,
    float background_alpha) {
  TCOD_console_blit(
      srcCon->get(), xSrc, ySrc, wSrc, hSrc, dstCon->get(), xDst, yDst, foreground_alpha, background_alpha);
}

void TCODConsole::flush() { tcod::check_throw_error(TCOD_console_flush()); }

void TCODConsole::setFade(uint8_t val, const TCODColor& fade) {
  TCOD_color_t f = {fade.r, fade.g, fade.b};
  TCOD_console_set_fade(val, f);
}

uint8_t TCODConsole::getFade() { return TCOD_console_get_fade(); }

TCODColor TCODConsole::getFadingColor() { return TCOD_console_get_fading_color(); }

void TCODConsole::putChar(int x, int y, int c, TCOD_bkgnd_flag_t flag) {
  TCOD_console_put_char(data.get(), x, y, c, flag);
}

void TCODConsole::putCharEx(int x, int y, int c, const TCODColor& fore, const TCODColor& back) {
  TCOD_color_t f = {fore.r, fore.g, fore.b};
  TCOD_color_t b = {back.r, back.g, back.b};
  TCOD_console_put_char_ex(data.get(), x, y, c, f, b);
}

void TCODConsole::clear() { TCOD_console_clear(data.get()); }

TCODColor TCODConsole::getCharBackground(int x, int y) const {
  return TCOD_console_get_char_background(data.get(), x, y);
}
void TCODConsole::setCharForeground(int x, int y, const TCODColor& col) {
  TCOD_color_t c = {col.r, col.g, col.b};
  TCOD_console_set_char_foreground(data.get(), x, y, c);
}
TCODColor TCODConsole::getCharForeground(int x, int y) const {
  return TCOD_console_get_char_foreground(data.get(), x, y);
}

int TCODConsole::getChar(int x, int y) const { return TCOD_console_get_char(data.get(), x, y); }

void TCODConsole::setCharBackground(int x, int y, const TCODColor& col, TCOD_bkgnd_flag_t flag) {
  TCOD_color_t c = {col.r, col.g, col.b};
  TCOD_console_set_char_background(data.get(), x, y, c, flag);
}

void TCODConsole::setChar(int x, int y, int c) { TCOD_console_set_char(data.get(), x, y, c); }

void TCODConsole::rect(int x, int y, int rw, int rh, bool clear, TCOD_bkgnd_flag_t flag) {
  TCOD_console_rect(data.get(), x, y, rw, rh, clear, flag);
}

void TCODConsole::hline(int x, int y, int l, TCOD_bkgnd_flag_t flag) { TCOD_console_hline(data.get(), x, y, l, flag); }

void TCODConsole::vline(int x, int y, int l, TCOD_bkgnd_flag_t flag) { TCOD_console_vline(data.get(), x, y, l, flag); }

void TCODConsole::printFrame(int x, int y, int w, int h, bool empty, TCOD_bkgnd_flag_t flag, const char* fmt, ...) {
  if (fmt) {
    va_list ap;
    va_start(ap, fmt);
    TCOD_console_print_frame(data.get(), x, y, w, h, empty, flag, TCOD_console_vsprint(fmt, ap));
    va_end(ap);
  } else {
    TCOD_console_print_frame(data.get(), x, y, w, h, empty, flag, NULL);
  }
}
/** Deprecated EASCII function. */
void TCODConsole::print(int x, int y, const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  TCOD_console_print_internal(
      data.get(), x, y, 0, 0, get()->bkgnd_flag, get()->alignment, TCOD_console_vsprint(fmt, ap), false, false);
  va_end(ap);
}
#ifndef TCOD_NO_UNICODE
void TCODConsole::print(int x, int y, const std::string& str) {
  tcod::print(
      *this,
      {x, y},
      str,
      tcod::ColorRGB{get()->fore},
      tcod::ColorRGB{get()->back},
      get()->alignment,
      get()->bkgnd_flag);
}
void TCODConsole::print(int x, int y, const std::string& str, TCOD_alignment_t alignment, TCOD_bkgnd_flag_t flag) {
  tcod::print(*this, {x, y}, str, tcod::ColorRGB{get()->fore}, tcod::ColorRGB{get()->back}, alignment, flag);
}
void TCODConsole::printf(int x, int y, const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  try {
    tcod::print(
        *this,
        {x, y},
        vstring(fmt, ap),
        tcod::ColorRGB{get()->fore},
        tcod::ColorRGB{get()->back},
        get()->alignment,
        get()->bkgnd_flag);
  } catch (...) {
    va_end(ap);
    throw;
  }
  va_end(ap);
}
void TCODConsole::printf(int x, int y, TCOD_bkgnd_flag_t flag, TCOD_alignment_t alignment, const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  auto str = vstring(fmt, ap);
  try {
    tcod::print(
        *this, {x, y}, vstring(fmt, ap), tcod::ColorRGB{get()->fore}, tcod::ColorRGB{get()->back}, alignment, flag);
  } catch (...) {
    va_end(ap);
    throw;
  }
  va_end(ap);
}
#endif  // TCOD_NO_UNICODE
/** Deprecated EASCII function. */
void TCODConsole::printEx(int x, int y, TCOD_bkgnd_flag_t flag, TCOD_alignment_t alignment, const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  TCOD_console_print_internal(data.get(), x, y, 0, 0, flag, alignment, TCOD_console_vsprint(fmt, ap), false, false);
  va_end(ap);
}

int TCODConsole::printRect(int x, int y, int w, int h, const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int ret = TCOD_console_print_internal(
      data.get(), x, y, w, h, get()->bkgnd_flag, get()->alignment, TCOD_console_vsprint(fmt, ap), true, false);
  va_end(ap);
  return ret;
}

int TCODConsole::printRectEx(
    int x, int y, int w, int h, TCOD_bkgnd_flag_t flag, TCOD_alignment_t alignment, const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int ret =
      TCOD_console_print_internal(data.get(), x, y, w, h, flag, alignment, TCOD_console_vsprint(fmt, ap), true, false);
  va_end(ap);
  return ret;
}

int TCODConsole::getHeightRect(int x, int y, int w, int h, const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int ret = TCOD_console_print_internal(
      data.get(), x, y, w, h, TCOD_BKGND_NONE, TCOD_LEFT, TCOD_console_vsprint(fmt, ap), true, true);
  va_end(ap);
  return ret;
}

bool TCODConsole::isKeyPressed(TCOD_keycode_t key) { return TCOD_console_is_key_pressed(key) != 0; }
void TCODConsole::setKeyColor(const TCODColor& col) {
  TCOD_color_t c = {col.r, col.g, col.b};
  TCOD_console_set_key_color(data.get(), c);
}

#ifndef NO_SDL
void TCODConsole::credits() { TCOD_console_credits(); }
void TCODConsole::resetCredits() { TCOD_console_credits_reset(); }
bool TCODConsole::renderCredits(int x, int y, bool alpha) { return TCOD_console_credits_render(x, y, alpha) != 0; }
#endif  // NO_SDL

#ifndef NO_UNICODE
void TCODConsole::mapStringToFont(const wchar_t* s, int fontCharX, int fontCharY) {
  TCOD_console_map_string_to_font_utf(s, fontCharX, fontCharY);
}

void TCODConsole::print(int x, int y, const wchar_t* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  TCOD_console_print_internal_utf(
      data.get(), x, y, 0, 0, get()->bkgnd_flag, get()->alignment, TCOD_console_vsprint_utf(fmt, ap), false, false);
  va_end(ap);
}

void TCODConsole::printEx(int x, int y, TCOD_bkgnd_flag_t flag, TCOD_alignment_t alignment, const wchar_t* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  TCOD_console_print_internal_utf(
      data.get(), x, y, 0, 0, flag, alignment, TCOD_console_vsprint_utf(fmt, ap), false, false);
  va_end(ap);
}

int TCODConsole::printRect(int x, int y, int w, int h, const wchar_t* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int ret = TCOD_console_print_internal_utf(
      data.get(), x, y, w, h, get()->bkgnd_flag, get()->alignment, TCOD_console_vsprint_utf(fmt, ap), true, false);
  va_end(ap);
  return ret;
}

int TCODConsole::printRectEx(
    int x, int y, int w, int h, TCOD_bkgnd_flag_t flag, TCOD_alignment_t alignment, const wchar_t* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int ret = TCOD_console_print_internal_utf(
      data.get(), x, y, w, h, flag, alignment, TCOD_console_vsprint_utf(fmt, ap), true, false);
  va_end(ap);
  return ret;
}

int TCODConsole::getHeightRect(int x, int y, int w, int h, const wchar_t* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int ret = TCOD_console_print_internal_utf(
      data.get(), x, y, w, h, TCOD_BKGND_NONE, TCOD_LEFT, TCOD_console_vsprint_utf(fmt, ap), true, true);
  va_end(ap);
  return ret;
}

// color control string formatting utilities for swigged language

// ctrl = TCOD_COLCTRL_1...TCOD_COLCTRL_5 or TCOD_COLCTRL_STOP
#define NB_BUFFERS 10
const char* TCODConsole::getColorControlString(TCOD_colctrl_t ctrl) {
  static char buf[NB_BUFFERS][2];
  static int buf_nb = 0;
  buf[buf_nb][0] = (char)ctrl;
  buf[buf_nb][1] = 0;
  const char* ret = buf[buf_nb];
  buf_nb = (buf_nb + 1) % NB_BUFFERS;
  return ret;
}

// ctrl = TCOD_COLCTRL_FORE_RGB or TCOD_COLCTRL_BACK_RGB
const char* TCODConsole::getRGBColorControlString(TCOD_colctrl_t ctrl, const TCODColor& col) {
  static char buf[NB_BUFFERS][5];
  static int buf_nb = 0;
  buf[buf_nb][0] = (char)ctrl;
  buf[buf_nb][1] = col.r;
  buf[buf_nb][2] = col.g;
  buf[buf_nb][3] = col.b;
  buf[buf_nb][4] = 0;
  const char* ret = buf[buf_nb];
  buf_nb = (buf_nb + 1) % NB_BUFFERS;
  return ret;
}
#endif  // NO_UNICODE

// Deprecated keyboard functions.
void TCODConsole::setKeyboardRepeat(int, int) {}
void TCODConsole::disableKeyboardRepeat() {}
