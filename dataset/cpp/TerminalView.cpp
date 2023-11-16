#include "TerminalView.h"
#include <algorithm>
#include <libfoundation/EventLoop.h>
#include <libfoundation/KeyboardMapping.h>
#include <libg/Color.h>
#include <libui/Context.h>

TerminalView::TerminalView(UI::View* superview, const LG::Rect& frame, int ptmx)
    : UI::View(superview, frame)
    , m_ptmx(ptmx)
{
    recalc_dimensions(frame);
}

TerminalView::TerminalView(UI::View* superview, UI::Window* window, const LG::Rect& frame, int ptmx)
    : UI::View(superview, window, frame)
    , m_ptmx(ptmx)
{
    recalc_dimensions(frame);
    LFoundation::EventLoop::the().add(LFoundation::Timer([this] {
        this->m_cursor_visible = !this->m_cursor_visible;
        this->invalidate_cursor_glyph();
    },
        400, LFoundation::Timer::Repeat));
}

void TerminalView::recalc_dimensions(const LG::Rect& frame)
{
    m_max_rows = (frame.height() - padding() - UI::SafeArea::Bottom) / glyph_height();
    m_max_cols = (frame.width() - 2 * padding()) / glyph_width();
    // FIXME: Add copy and resize on window resize.
    char* new_data = (char*)malloc(m_max_rows * m_max_cols);
    memset(new_data, 0, m_max_rows * m_max_cols);
    if (m_display_data) {
        free(m_display_data);
    }
    m_display_data = new_data;
}

void TerminalView::display(const LG::Rect& rect)
{
    LG::Context ctx = UI::graphics_current_context();
    ctx.add_clip(rect);

    ctx.set_fill_color(background_color());
    ctx.fill(bounds());

    ctx.set_fill_color(cursor_color());
    auto cursor_left_corner = pos_on_screen();
    ctx.fill(LG::Rect(cursor_left_corner.x(), cursor_left_corner.y(), cursor_width(), glyph_height()));

    auto& f = font();
    ctx.set_fill_color(font_color());
    LG::Point<int> text_start { padding(), padding() };

    for (int i = 0; i < m_max_rows; i++) {
        for (int j = 0; j < m_max_cols; j++) {
            int idx = i * m_max_cols + j;
            ctx.draw(text_start, f.glyph(m_display_data[idx]));
            text_start.offset_by(glyph_width(), 0);
        }
        text_start.set_x(padding());
        text_start.offset_by(0, glyph_height());
    }
}

void TerminalView::scroll_line()
{
    data_do_new_line();
    set_needs_display();
}

void TerminalView::data_do_new_line()
{
    char* data_plus_line = m_display_data + (m_max_cols);
    char* data_end_minus_line = m_display_data + (m_max_rows - 1) * m_max_cols;
    memmove(m_display_data, data_plus_line, (m_max_rows - 1) * m_max_cols);
    memset(data_end_minus_line, 0, m_max_cols);
}

WindowStatus TerminalView::cursor_positions_do_new_line()
{
    m_col = 0;
    m_row++;
    if (m_row == m_max_rows) {
        m_row--;
        return DoNewLine;
    }
    return Normal;
}

WindowStatus TerminalView::cursor_position_move_right()
{
    m_col++;
    if (m_col == m_max_cols) {
        return cursor_positions_do_new_line();
    }
    return Normal;
}

WindowStatus TerminalView::cursor_position_move_left()
{
    m_col--;
    if (m_col == 0 && m_row > 0) {
        m_row--;
    }
    return Normal;
}

void TerminalView::new_line()
{
    will_move_cursor();
    WindowStatus status = cursor_positions_do_new_line();
    switch (status) {
    case DoNewLine:
        scroll_line();
        break;
    case Normal:
        break;
    }
    did_move_cursor();
}

void TerminalView::increment_counter()
{
    will_move_cursor();
    m_col++;
    if (m_col == m_max_cols) {
        new_line();
    }
    did_move_cursor();
}

void TerminalView::decrement_counter()
{
    will_move_cursor();
    m_col--;
    if (m_col == 0) {
        m_row--;
    }
    did_move_cursor();
}

void TerminalView::put_char(char c)
{
    auto pt = pos_on_screen();
    set_needs_display(LG::Rect(pt.x(), pt.y(), glyph_width(), glyph_height()));
    m_display_data[pos_in_data()] = c;
}

void TerminalView::push_back_char(char c)
{
    if (c == '\n') {
        new_line();
        return;
    }
    put_char(c);
    increment_counter();
}

void TerminalView::put_text(const std::string& data)
{
    auto current_pos = pos_on_screen();
    LG::Point<int> top_left_update_location { current_pos.x(), current_pos.y() };
    LG::Point<int> bottom_right_update_location { current_pos.x(), current_pos.y() };
    auto set_to_redraw_full_screen = [&]() {
        data_do_new_line();
        top_left_update_location = { bounds().min_x(), bounds().min_y() };
        bottom_right_update_location = { bounds().max_x(), bounds().max_y() };
    };

    will_move_cursor();
    int n = data.size();
    for (int i = 0; i < n; i++) {
        char c = data[i];
        if (c == '\n') {
            auto status = cursor_positions_do_new_line();
            if (status == DoNewLine) {
                set_to_redraw_full_screen();
            }
        } else {
            auto pt = pos_on_screen();
            data_set_char(c);
            auto status = cursor_position_move_right();
            if (status == DoNewLine) {
                set_to_redraw_full_screen();
            } else {
                top_left_update_location.set_x(std::min(top_left_update_location.x(), pt.x()));
                top_left_update_location.set_y(std::min(top_left_update_location.y(), pt.y()));
                bottom_right_update_location.set_x(std::max(bottom_right_update_location.x(), pt.x() + glyph_width()));
                bottom_right_update_location.set_y(std::max(bottom_right_update_location.y(), pt.y() + glyph_height()));
            }
        }
    }

    auto w = bottom_right_update_location.x() - top_left_update_location.x() + 1;
    auto h = bottom_right_update_location.y() - top_left_update_location.y() + 1;
    set_needs_display(LG::Rect(top_left_update_location.x(), top_left_update_location.y(), w, h));
    did_move_cursor();
}

void TerminalView::send_input()
{
    write(ptmx(), m_input.c_str(), m_input.size());
    m_input.clear();
}

void TerminalView::receive_keyup_event(UI::KeyUpEvent&)
{
}

void TerminalView::receive_keydown_event(UI::KeyDownEvent& event)
{
    // FIXME: More symbols and static size of font
    if (event.key() == LFoundation::Keycode::KEY_BACKSPACE) {
        if (m_input.size()) {
            m_input.pop_back();
            decrement_counter();
            put_char(' ');
        }
    } else if (event.key() == LFoundation::Keycode::KEY_RETURN) {
        m_input.push_back('\n');
        push_back_char('\n');
        send_input();
    } else if (event.key() < 128) {
        m_input.push_back(char(event.key()));
        push_back_char(char(event.key()));
    }
}
