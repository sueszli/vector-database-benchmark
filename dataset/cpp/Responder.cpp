/*
 * Copyright (C) 2020-2022 The opuntiaOS Project Authors.
 *  + Contributed by Nikita Melekhin <nimelehin@gmail.com>
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include <libfoundation/EventLoop.h>
#include <libui/App.h>
#include <libui/Responder.h>
#include <libui/Window.h>

namespace UI {

bool Responder::send_invalidate_message_to_server(const LG::Rect& rect) const
{
    auto& app = App::the();
    InvalidateMessage msg(Connection::the().key(), app.window().id(), rect);
    return app.connection().send_async_message(msg);
}

void Responder::send_layout_message(Window& win, UI::View* for_view)
{
    LFoundation::EventLoop::the().add(win, new LayoutEvent(for_view));
    m_display_message_sent = false;
}

void Responder::send_display_message_to_self(Window& win, const LG::Rect& display_rect)
{
    if (!m_display_message_sent || m_prev_display_message != display_rect) {
        LFoundation::EventLoop::the().add(win, new DisplayEvent(display_rect));
        m_display_message_sent = true;
        m_prev_display_message = display_rect;
    }
}

void Responder::receive_event(std::unique_ptr<LFoundation::Event> event)
{
    switch (event->type()) {
    case Event::Type::MouseEvent: {
        MouseEvent& own_event = *(MouseEvent*)event.get();
        receive_mouse_move_event(own_event);
        break;
    }

    case Event::Type::DisplayEvent: {
        DisplayEvent& own_event = *(DisplayEvent*)event.get();
        receive_display_event(own_event);
        break;
    }

    case Event::Type::LayoutEvent: {
        LayoutEvent& own_event = *(LayoutEvent*)event.get();
        receive_layout_event(own_event);
        break;
    }
    }
}

} // namespace UI