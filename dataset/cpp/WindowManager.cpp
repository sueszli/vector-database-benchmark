/*
 * Copyright (C) 2020-2022 The opuntiaOS Project Authors.
 *  + Contributed by Nikita Melekhin <nimelehin@gmail.com>
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "WindowManager.h"
#include "../Devices/Screen.h"
#include "../Managers/CursorManager.h"
#include <libapi/window_server/MessageContent/MouseAction.h>
#include <libfoundation/KeyboardMapping.h>
#include <libfoundation/Logger.h>

// #define WM_DEBUG

namespace WinServer {

static PopupData WindowPopupData {};

WindowManager* s_WinServer_WindowManager_the = nullptr;

WindowManager::WindowManager()
    : m_screen(Screen::the())
    , m_connection(Connection::the())
    , m_compositor(Compositor::the())
    , m_cursor_manager(CursorManager::the())
    , m_event_loop(LFoundation::EventLoop::the())
    , m_std_menubar_content()
    , m_visible_area(m_screen.bounds())
{
    s_WinServer_WindowManager_the = this;
    shrink_visible_area(menu_bar().height(), 0);
#ifdef TARGET_DESKTOP
    menu_bar().set_background_color(LG::Color::LightSystemOpaque128);
#endif // TARGET_DESKTOP
}

void WindowManager::setup_dock(Window* window)
{
#ifdef TARGET_DESKTOP
    window->make_frameless();
    window->bounds().set_y(m_screen.bounds().max_y() - window->bounds().height() + 1);
    window->content_bounds().set_y(m_screen.bounds().max_y() - window->bounds().height() + 1);
    shrink_visible_area(0, window->bounds().height());
#endif // TARGET_DESKTOP
    window->set_event_mask(WindowEvent::IconChange | WindowEvent::WindowStatus | WindowEvent::WindowCreation | WindowEvent::TitleChange);
    m_dock.set_window(window);
}

void WindowManager::setup_applist(Window* window)
{
#ifdef TARGET_DESKTOP
    window->make_frameless();
    const size_t coorx = (visible_area().max_x() - window->bounds().width()) / 2;
    const size_t coory = visible_area().max_y() - window->bounds().height() - 8;
    window->bounds().set_x(coorx);
    window->content_bounds().set_x(coorx);
    window->bounds().set_y(coory);
    window->content_bounds().set_y(coory);
#endif // TARGET_DESKTOP
    m_applist.set_window(window);
    minimize_window(*window);
}

void WindowManager::add_system_window(Window* window)
{
    switch (window->type()) {
    case WindowType::Homescreen:
        setup_dock(window);
        break;

    case WindowType::AppList:
        setup_applist(window);
        break;

    default:
        break;
    }
}

void WindowManager::add_window(Window* window)
{
    m_windows.push_back(window);
    set_active_window(window);

    if (window->type() != WindowType::Standard) {
        add_system_window(window);
    }
    notify_window_creation(window->id());
}

void WindowManager::remove_attention_from_window(Window* window)
{
    if (movable_window() == window) {
        m_movable_window = nullptr;
    }
    if (active_window() == window) {
        set_active_window(nullptr);
    }
    if (hovered_window() == window) {
        set_hovered_window(nullptr);
    }
}

void WindowManager::on_window_became_invisible(Window* window)
{
    if (window->type() == WindowType::Standard && active_window() == window) {
#ifdef TARGET_DESKTOP
        menu_bar().set_menubar_content(nullptr, m_compositor);
#elif TARGET_MOBILE
        menu_bar().set_style(StatusBarStyle::StandardOpaque);
        m_compositor.invalidate(menu_bar().bounds());
#endif
    }
    m_compositor.invalidate(window->bounds());
}

void WindowManager::remove_window(Window* window_ptr)
{
    notify_window_status_changed(window_ptr->id(), WindowStatusUpdateType::Removed);
    m_windows.erase(std::find(m_windows.begin(), m_windows.end(), window_ptr));
    on_window_became_invisible(window_ptr);
    remove_attention_from_window(window_ptr);
    delete window_ptr;
}

void WindowManager::minimize_window(Window& window)
{
    Window* window_ptr = &window;
    notify_window_status_changed(window.id(), WindowStatusUpdateType::Minimized);
    window.set_visible(false);
    m_windows.erase(std::find(m_windows.begin(), m_windows.end(), window_ptr));
    m_windows.push_back(window_ptr);
    on_window_became_invisible(window_ptr);
    remove_attention_from_window(window_ptr);
}

void WindowManager::resize_window(Window& window, const LG::Size& size)
{
    window.did_size_change(size);
    send_event(new ResizeMessage(window.connection_id(), window.id(), LG::Rect(0, 0, size.width(), size.height())));
    m_compositor.invalidate(window.bounds());
}

void WindowManager::maximize_window(Window& window)
{
    size_t fullscreen_h = m_screen.height();
    fullscreen_h = visible_area().height();

    const size_t vertical_borders = Desktop::WindowFrame::std_top_border_size() + Desktop::WindowFrame::std_bottom_border_size();
    const size_t horizontal_borders = Desktop::WindowFrame::std_left_border_size() + Desktop::WindowFrame::std_right_border_size();

    move_window(&window, -window.bounds().min_x(), -(window.bounds().min_y() - menu_bar().height()));
    resize_window(window, { m_screen.width() - horizontal_borders, fullscreen_h - vertical_borders });
}

void WindowManager::start_window_move(Window& window)
{
    m_movable_window = &window;
}

WindowManager::Window* WindowManager::top_window_in_view(WindowType type) const
{
    for (auto it = m_windows.begin(); it != m_windows.end(); it++) {
        auto* window = (*it);
        if (window->type() == type) {
            return window;
        }
    }
    return nullptr;
}

#ifdef TARGET_DESKTOP
void WindowManager::on_active_window_will_change()
{
    if (!m_active_window) {
        return;
    }

    if (m_active_window->type() == WindowType::AppList) {
        m_active_window->set_visible(false);
        on_window_became_invisible(m_active_window);
    }
}

void WindowManager::on_active_window_did_change()
{
}

void WindowManager::bring_system_windows_to_front()
{
    if (m_dock.has_value()) {
        do_bring_to_front(*m_dock.window());
    }
}

void WindowManager::bring_to_front(Window& window)
{
    auto* prev_window = top_window_in_view(WindowType::Standard);
    do_bring_to_front(window);
    bring_system_windows_to_front();

    window.set_visible(true);
    window.frame().set_active(true);
    m_compositor.invalidate(window.bounds());
    if (prev_window && prev_window->id() != window.id()) {
        prev_window->frame().set_active(false);
        prev_window->frame().invalidate(m_compositor);
    }
    if (window.type() == WindowType::Standard) {
        menu_bar().set_menubar_content(&window.menubar_content(), m_compositor);
    } else {
        menu_bar().set_menubar_content(nullptr, m_compositor);
    }
}
#elif TARGET_MOBILE
void WindowManager::on_active_window_will_change()
{
}

void WindowManager::on_active_window_did_change()
{
    // If current active_window has become NULL, try to restore the lastest.
    if (active_window() == nullptr) {
        if (auto top_window = m_windows.begin(); top_window != m_windows.end()) {
            m_active_window = *top_window;
        }
    }
}

void WindowManager::bring_system_windows_to_front()
{
}

void WindowManager::bring_to_front(Window& window)
{
    do_bring_to_front(window);
    bring_system_windows_to_front();
    window.set_visible(true);
    m_active_window = &window;
    m_compositor.invalidate(window.bounds());
    if (window.type() == WindowType::Standard) {
        menu_bar().set_style(window.style());
        m_compositor.invalidate(menu_bar().bounds());
    }
}
#endif

#ifdef TARGET_DESKTOP
bool WindowManager::continue_window_move()
{
    if (!movable_window()) {
        return false;
    }

    if (!m_cursor_manager.pressed<CursorManager::Params::LeftButton>()) {
        m_movable_window = nullptr;
        return true;
    }

    auto bounds = movable_window()->bounds();
    m_compositor.invalidate(movable_window()->bounds());
    move_window(movable_window(), m_cursor_manager.get<CursorManager::Params::OffsetX>(), m_cursor_manager.get<CursorManager::Params::OffsetY>());
    bounds.unite(movable_window()->bounds());
    m_compositor.invalidate(bounds);
    return true;
}
#endif // TARGET_DESKTOP

void WindowManager::update_mouse_position(std::unique_ptr<LFoundation::Event> mouse_event)
{
    auto invalidate_bounds = m_cursor_manager.current_cursor().bounds();
    invalidate_bounds.origin().set(m_cursor_manager.draw_position());
    m_compositor.invalidate(invalidate_bounds);

    m_cursor_manager.update_position((WinServer::MouseEvent*)mouse_event.get());

    invalidate_bounds.origin().set(m_cursor_manager.draw_position());
    m_compositor.invalidate(invalidate_bounds);
}

#ifdef TARGET_DESKTOP
void WindowManager::receive_mouse_event(std::unique_ptr<LFoundation::Event> event)
{
    update_mouse_position(std::move(event));
    if (continue_window_move()) {
        return;
    }

    // Checking and dispatching mouse move for Popup.
    if (popup().visible() && popup().bounds().contains(m_cursor_manager.x(), m_cursor_manager.y())) {
        popup().on_mouse_move(m_cursor_manager);
        if (m_cursor_manager.is_changed<CursorManager::Params::Buttons>()) {
            popup().on_mouse_status_change(m_cursor_manager);
        }
        return;
    } else {
        popup().on_mouse_leave(m_cursor_manager);
        if (m_cursor_manager.pressed<CursorManager::Params::LeftButton>()) {
            popup().set_visible(false);
        }
    }

    // Checking and dispatching mouse move for MenuBar.
    if (menu_bar().bounds().contains(m_cursor_manager.x(), m_cursor_manager.y())) {
        menu_bar().on_mouse_move(m_cursor_manager);
        if (m_cursor_manager.is_changed<CursorManager::Params::Buttons>()) {
            menu_bar().on_mouse_status_change(m_cursor_manager);
        }
        return;
    } else if (menu_bar().is_hovered()) {
        menu_bar().on_mouse_leave(m_cursor_manager);
    }

    Window* curr_hovered_window = nullptr;
    Window* window_under_mouse_ptr = nullptr;

    for (auto* window_ptr : m_windows) {
        auto& window = *window_ptr;
        if (!window.visible()) {
            continue;
        }

        if (window.bounds().contains(m_cursor_manager.x(), m_cursor_manager.y())) {
            window_under_mouse_ptr = window_ptr;
            break;
        }
    }

    if (m_cursor_manager.pressed<CursorManager::Params::LeftButton>() || m_cursor_manager.pressed<CursorManager::Params::RightButton>()) {
        if (!window_under_mouse_ptr && m_active_window) {
            menu_bar().set_menubar_content(nullptr, m_compositor);
            m_compositor.invalidate(m_active_window->bounds());
            m_active_window->frame().set_active(false);
            set_active_window(nullptr);
        } else if (m_active_window != window_under_mouse_ptr) {
            set_active_window(window_under_mouse_ptr);
        }
    }

    if (!window_under_mouse_ptr) {
        if (hovered_window()) {
            send_event(new MouseLeaveMessage(hovered_window()->connection_id(), hovered_window()->id(), 0, 0));
        }
        return;
    }
    auto& window = *window_under_mouse_ptr;

    if (window.content_bounds().contains(m_cursor_manager.x(), m_cursor_manager.y())) {
        if (window.type() == WindowType::Standard && active_window() != &window) {
            curr_hovered_window = nullptr;
        } else {
            if (m_cursor_manager.is_changed<CursorManager::Params::Coords>()) {
                LG::Point<int> point(m_cursor_manager.x(), m_cursor_manager.y());
                point.offset_by(-window.content_bounds().origin());
                send_event(new MouseMoveMessage(window.connection_id(), window.id(), point.x(), point.y()));
            }
            curr_hovered_window = &window;
        }
    } else if (m_cursor_manager.is_changed<CursorManager::Params::LeftButton>() && m_cursor_manager.pressed<CursorManager::Params::LeftButton>()) {
        auto tap_point = LG::Point<int>(m_cursor_manager.x() - window.frame().bounds().min_x(), m_cursor_manager.y() - window.frame().bounds().min_y());
        window.frame().receive_tap_event(tap_point);
        start_window_move(window);
    }

    if (hovered_window() && hovered_window() != curr_hovered_window) {
        send_event(new MouseLeaveMessage(hovered_window()->connection_id(), hovered_window()->id(), 0, 0));
    }
    set_hovered_window(curr_hovered_window);

    if (hovered_window() && m_cursor_manager.is_changed<CursorManager::Params::Buttons>()) {
        LG::Point<int> point(m_cursor_manager.x(), m_cursor_manager.y());
        point.offset_by(-window.content_bounds().origin());

        auto buttons_state = MouseActionState();
        if (m_cursor_manager.is_changed<CursorManager::Params::LeftButton>()) {
            // TODO: May be remove if?
            if (m_cursor_manager.pressed<CursorManager::Params::LeftButton>()) {
                buttons_state.set(MouseActionType::LeftMouseButtonPressed);
            } else {
                buttons_state.set(MouseActionType::LeftMouseButtonReleased);
            }
        }

        send_event(new MouseActionMessage(window.connection_id(), window.id(), buttons_state.state(), point.x(), point.y()));
    }

    if (hovered_window() && m_cursor_manager.is_changed<CursorManager::Params::Wheel>()) {
        auto* window = hovered_window();
        auto data = m_cursor_manager.get<CursorManager::Params::Wheel>();
        send_event(new MouseWheelMessage(window->connection_id(), window->id(), data, m_cursor_manager.x(), m_cursor_manager.y()));
    }
}
#elif TARGET_MOBILE
void WindowManager::receive_mouse_event(std::unique_ptr<LFoundation::Event> event)
{
    update_mouse_position(std::move(event));

    if (m_compositor.control_bar().control_button_bounds().contains(m_cursor_manager.x(), m_cursor_manager.y()) && active_window()) {
        if (m_cursor_manager.pressed<CursorManager::Params::LeftButton>()) {
            switch (active_window()->type()) {
            case WindowType::Standard:
                remove_window(active_window());
                break;
            case WindowType::AppList:
                minimize_window(*active_window());
                break;

            case WindowType::Homescreen:
            default:
                break;
            }
        }
        return;
    }

    // Tap emulation
    if (m_cursor_manager.is_changed<CursorManager::Params::Buttons>() && active_window()) {
        auto window = active_window();
        auto buttons_state = MouseActionState();
        LG::Point<int> point(m_cursor_manager.x(), m_cursor_manager.y());
        point.offset_by(-window->content_bounds().origin());
        if (m_cursor_manager.is_changed<CursorManager::Params::LeftButton>()) {
            // TODO: May be remove if?
            if (m_cursor_manager.pressed<CursorManager::Params::LeftButton>()) {
                buttons_state.set(MouseActionType::LeftMouseButtonPressed);
            } else {
                buttons_state.set(MouseActionType::LeftMouseButtonReleased);
            }
        }
        send_event(new MouseActionMessage(window->connection_id(), window->id(), buttons_state.state(), point.x(), point.y()));
    }

    if (active_window()) {
        auto window = active_window();
        if (m_cursor_manager.pressed<CursorManager::Params::LeftButton>()) {
            if (m_cursor_manager.is_changed<CursorManager::Params::Coords>()) {
                LG::Point<int> point(m_cursor_manager.x(), m_cursor_manager.y());
                point.offset_by(-window->content_bounds().origin());
                send_event(new MouseMoveMessage(window->connection_id(), window->id(), point.x(), point.y()));
            }
        }
    }
}
#endif // TARGET_MOBILE

void WindowManager::receive_keyboard_event(std::unique_ptr<LFoundation::Event> event)
{
    auto* keyboard_event = reinterpret_cast<KeyboardEvent*>(event.release());
    if (active_window()) {
        auto window = active_window();
        send_event(new KeyboardMessage(window->connection_id(), window->id(), keyboard_event->packet().key));
    }
    delete keyboard_event;
}

void WindowManager::receive_event(std::unique_ptr<LFoundation::Event> event)
{
    switch (event->type()) {
    case WinServer::Event::Type::MouseEvent:
        receive_mouse_event(std::move(event));
        break;
    case WinServer::Event::Type::KeyboardEvent:
        receive_keyboard_event(std::move(event));
        break;
    }
}

// Notifiers

bool WindowManager::notify_listner_about_window_creation(const Window& win, int changed_window_id)
{
#ifdef WM_DEBUG
    Logger::debug << "notify_listner_about_window_status " << win.id() << " that " << changed_window_id << " " << type << std::endl;
#endif
    auto* changed_window_ptr = window(changed_window_id);
    if (!changed_window_ptr) {
        return false;
    }
    send_event(new NotifyWindowCreateMessage(win.connection_id(), win.id(), changed_window_ptr->bundle_id(), changed_window_ptr->icon_path(), changed_window_id, changed_window_ptr->type()));
    return true;
}

bool WindowManager::notify_listner_about_window_status(const Window& win, int changed_window_id, WindowStatusUpdateType type)
{
#ifdef WM_DEBUG
    Logger::debug << "notify_listner_about_window_status " << win.id() << " that " << changed_window_id << " " << type << std::endl;
#endif
    auto* changed_window_ptr = window(changed_window_id);
    if (!changed_window_ptr) {
        return false;
    }
    send_event(new NotifyWindowStatusChangedMessage(win.connection_id(), win.id(), changed_window_id, (int)type));
    return true;
}

bool WindowManager::notify_listner_about_changed_icon(const Window& win, int changed_window_id)
{
#ifdef WM_DEBUG
    Logger::debug << "notify_listner_about_changed_icon " << win.id() << " that " << changed_window_id << std::endl;
#endif
    auto* changed_window_ptr = window(changed_window_id);
    if (!changed_window_ptr) {
        return false;
    }
    send_event(new NotifyWindowIconChangedMessage(win.connection_id(), win.id(), changed_window_id, changed_window_ptr->icon_path()));
    return true;
}

bool WindowManager::notify_listner_about_changed_title(const Window& win, int changed_window_id)
{
#ifdef WM_DEBUG
    Logger::debug << "notify_listner_about_changed_title " << win.id() << " that " << changed_window_id << std::endl;
#endif
    auto* changed_window_ptr = window(changed_window_id);
    if (!changed_window_ptr) {
        return false;
    }
    send_event(new NotifyWindowTitleChangedMessage(win.connection_id(), win.id(), changed_window_id, changed_window_ptr->app_title()));
    return true;
}

void WindowManager::notify_window_creation(int changed_window_id)
{
    for (auto* window_ptr : m_windows) {
        auto& window = *window_ptr;
        if (window.event_mask() & WindowEvent::WindowCreation) {
            notify_listner_about_window_creation(window, changed_window_id);
        }
    }
}

void WindowManager::notify_window_status_changed(int changed_window_id, WindowStatusUpdateType type)
{
    for (auto* window_ptr : m_windows) {
        auto& window = *window_ptr;
        if (window.event_mask() & WindowEvent::WindowStatus) {
            notify_listner_about_window_status(window, changed_window_id, type);
        }
    }
}

void WindowManager::notify_window_icon_changed(int changed_window_id)
{
    for (auto* window_ptr : m_windows) {
        auto& window = *window_ptr;
        if (window.event_mask() & WindowEvent::IconChange) {
            notify_listner_about_changed_icon(window, changed_window_id);
        }
    }
}

void WindowManager::notify_window_title_changed(int changed_window_id)
{
    for (auto* window_ptr : m_windows) {
        auto& window = *window_ptr;
        if (window.event_mask() & WindowEvent::TitleChange) {
            notify_listner_about_changed_title(window, changed_window_id);
        }
    }
}

#ifdef TARGET_DESKTOP
void WindowManager::on_window_style_change(Window& window)
{
    if (window.visible()) {
        window.frame().invalidate(m_compositor);
    }
}
#elif TARGET_MOBILE
void WindowManager::on_window_style_change(Window& window)
{
    if (active_window() == &window && window.type() == WindowType::Standard) {
        menu_bar().set_style(window.style());
        m_compositor.invalidate(menu_bar().bounds());
    }
}
#endif

void WindowManager::on_window_menubar_change(Window& window)
{
    if (m_active_window == &window) {
        menu_bar().invalidate_menubar_panel(m_compositor);
    }
}

void WindowManager::on_window_misbehave(Window& window, ViolationClass viocls)
{
    switch (viocls) {
    case ViolationClass::Ignorable:
    case ViolationClass::Moderate:
        break;

    case ViolationClass::Serious:
        // TODO: Currently we only remove the window, but all apps
        // should be stopped with with a signal.
        remove_window(&window);
        break;

    default:
        break;
    }
}

} // namespace WinServer