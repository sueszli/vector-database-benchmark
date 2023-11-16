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
#ifndef TCOD_GUI_FLATLIST_HPP
#define TCOD_GUI_FLATLIST_HPP
#ifndef TCOD_NO_UNICODE
#include <math.h>
#include <stdio.h>

#include "textbox.hpp"

namespace tcod::gui {
class FlatList : public TextBox {
 public:
  FlatList(int x, int y, int w, const char** list, const char* label, const char* tip = nullptr)
      : TextBox(x, y, w, 10, label, nullptr, tip), value{list}, list{list} {
    valueToText();
    this->w += 2;
  }
  void render() override {
    w--;
    box_x++;
    TextBox::render();
    box_x--;
    w++;
    auto& console = static_cast<TCOD_Console&>(*con);
    if (console.in_bounds({x + box_x, y})) {
      console.at({x + box_x, y}) = {
          0x2190,  // ←
          TCOD_ColorRGBA(onLeftArrow ? foreFocus : fore),
          TCOD_ColorRGBA(onLeftArrow ? backFocus : back)};
    }
    if (console.in_bounds({x + w - 1, y})) {
      console.at({x + w - 1, y}) = {
          0x2192,  // →
          TCOD_ColorRGBA(onRightArrow ? foreFocus : fore),
          TCOD_ColorRGBA(onRightArrow ? backFocus : back)};
    }
  }
  void update(const TCOD_key_t k) override {
    onLeftArrow = onRightArrow = false;
    if (mouse.cx == x + box_x && mouse.cy == y)
      onLeftArrow = true;
    else if (mouse.cx == x + w - 1 && mouse.cy == y)
      onRightArrow = true;
    Widget::update(k);
  }
  void update(const SDL_Event& ev_tile, const SDL_Event& ev_pixel) override {
    onLeftArrow = onRightArrow = false;
    switch (ev_tile.type) {
      case SDL_MOUSEMOTION:
        onLeftArrow = ev_tile.motion.x == x + box_x && ev_tile.motion.y == y;
        onRightArrow = ev_tile.motion.x == x + w - 1 && ev_tile.motion.y == y;
        break;
      default:
        break;
    }
    Widget::update(ev_tile, ev_pixel);
  }
  void setCallback(void (*cbk_)(Widget* wid, const char* val, void* data), void* data_) {
    this->cbk = cbk_;
    this->data = data_;
  }
  void setValue(const char* v) {
    const char** ptr = list;
    while (*ptr) {
      if (strcmp(v, *ptr) == 0) {
        value = ptr;
        valueToText();
        break;
      }
      ptr++;
    }
  }
  void setList(const char** l) {
    value = list = l;
    valueToText();
  }

 protected:
  void valueToText() { setText(*value); }
  void textToValue() {
    const char** ptr = list;
    while (*ptr) {
      if (strcmp(text_.c_str(), *ptr) == 0) {
        value = ptr;
        break;
      }
      ptr++;
    }
  }
  void onButtonClick() override {
    const char** oldValue = value;
    if (onLeftArrow) {
      if (value == list) {
        while (*value) {
          value++;
        }
      }
      value--;
    } else if (onRightArrow) {
      value++;
      if (*value == nullptr) value = list;
    }
    if (value != oldValue && cbk) {
      valueToText();
      cbk(this, *value, data);
    }
  }

  const char** value{};
  const char** list{};
  bool onLeftArrow{false};
  bool onRightArrow{false};
  void (*cbk)(Widget* wid, const char* val, void* data){};
  void* data{};
};
}  // namespace tcod::gui
#endif  // TCOD_NO_UNICODE
#endif /* TCOD_GUI_FLATLIST_HPP */
