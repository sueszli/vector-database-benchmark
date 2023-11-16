#include "HomeScreenView.h"
#include "AppListView.h"
#include "IconView.h"
#include <algorithm>
#include <cstdlib>
#include <libfoundation/EventLoop.h>
#include <libfoundation/KeyboardMapping.h>
#include <libg/Color.h>
#include <libg/ImageLoaders/PNGLoader.h>
#include <libui/App.h>
#include <libui/Context.h>
#include <libui/StackView.h>
#include <unistd.h>

static HomeScreenView* this_view;

HomeScreenView::HomeScreenView(UI::View* superview, const LG::Rect& frame)
    : UI::View(superview, frame)
{
}

HomeScreenView::HomeScreenView(UI::View* superview, UI::Window* window, const LG::Rect& frame)
    : UI::View(superview, window, frame)
{
    LG::Rect homegrid_frame = LG::Rect(0, grid_padding(), bounds().width(), bounds().height() - grid_padding() - dock_height_with_padding());
    auto& vstack_view = add_subview<UI::StackView>(homegrid_frame);
    vstack_view.set_background_color(LG::Color::Opaque);
    vstack_view.set_axis(UI::LayoutConstraints::Axis::Vertical);
    vstack_view.set_distribution(UI::StackView::Distribution::EqualSpacing);
    vstack_view.set_alignment(UI::StackView::Alignment::Center);

    for (int i = 0; i < grid_entities_per_column(); i++) {
        auto& hstack_view = vstack_view.add_arranged_subview<UI::StackView>();
        hstack_view.set_background_color(LG::Color::Opaque);
        hstack_view.set_distribution(UI::StackView::Distribution::EqualSpacing);
        vstack_view.add_constraint(UI::Constraint(hstack_view, UI::Constraint::Attribute::Left, UI::Constraint::Relation::Equal, vstack_view, UI::Constraint::Attribute::Left, 1, grid_padding()));
        vstack_view.add_constraint(UI::Constraint(hstack_view, UI::Constraint::Attribute::Right, UI::Constraint::Relation::Equal, vstack_view, UI::Constraint::Attribute::Right, 1, -grid_padding()));
        vstack_view.add_constraint(UI::Constraint(hstack_view, UI::Constraint::Attribute::Height, UI::Constraint::Relation::Equal, icon_view_size()));
        m_grid_stackviews.push_back(&hstack_view);
    }

    int dock_offsety = bounds().height() - dock_height_with_padding();
    auto& dock_subview = add_subview<UI::View>(LG::Rect(0, dock_offsety, bounds().width(), dock_height_with_padding()));
    dock_subview.layer().set_corner_mask(LG::CornerMask(16, LG::CornerMask::Masked, LG::CornerMask::NonMasked));
    dock_subview.set_background_color(LG::Color::LightSystemOpaque128);
    dock_subview.add_gesture_recognizer<UI::SwipeGestureRecognizer>([this](const UI::GestureRecognizer* recon) {
        if (recon->state() == UI::GestureRecognizer::State::Ended) {
            if (m_applist_view) {
                m_applist_view->on_click();
            }
        }
    });

    LG::Rect applist_frame = homegrid_frame;
    applist_frame.set_x(grid_padding());
    applist_frame.set_width(bounds().width() - 2 * grid_padding());
    applist_frame.set_y(bounds().height() - dock_height_with_padding() + grid_padding());
    applist_frame.set_height(applist_height());
    auto& applist_view = add_subview<AppListView>(applist_frame);
    m_applist_view = &applist_view;

    LG::Rect dock_frame = homegrid_frame;
    dock_frame.set_y(applist_frame.max_y() + (dock_height() - icon_size()) / 2);
    dock_frame.set_height(dock_height());
    auto& dock_stack_view = add_subview<UI::StackView>(dock_frame);
    dock_stack_view.set_spacing(12); // TODO: Set spacing which depends on screen width.
    dock_stack_view.set_background_color(LG::Color::Opaque);
    dock_stack_view.set_axis(UI::LayoutConstraints::Axis::Horizontal);
    dock_stack_view.set_distribution(UI::StackView::Distribution::EqualCentering);
    m_dock_stackview = &dock_stack_view;

    vstack_view.set_needs_layout();
}

void HomeScreenView::display(const LG::Rect& rect)
{
    LG::Context ctx = UI::graphics_current_context();
    ctx.add_clip(rect);

    ctx.set_fill_color(LG::Color(0, 0, 0, 0));
    ctx.fill(bounds());
}

void HomeScreenView::new_grid_entity(const std::string& title, const std::string& icon_path, std::string&& exec_path)
{
    // TODO: Add pages.
    LG::PNG::PNGLoader loader;
    int row_to_put_to = 0;
    for (int i = 0; i < grid_entities_per_column(); i++) {
        if (m_grid_stackviews[i]->subviews().size() < grid_entities_per_row()) {
            row_to_put_to = i;
            break;
        }
    }
    auto& icon_view = m_grid_stackviews[row_to_put_to]->add_arranged_subview<IconView>();
    icon_view.add_constraint(UI::Constraint(icon_view, UI::Constraint::Attribute::Height, UI::Constraint::Relation::Equal, icon_view_size()));
    icon_view.add_constraint(UI::Constraint(icon_view, UI::Constraint::Attribute::Width, UI::Constraint::Relation::Equal, icon_view_size()));
    icon_view.set_title(title);
    icon_view.entity().set_icon(loader.load_from_file(icon_path + "/48x48.png"));
    icon_view.entity().set_path_to_exec(std::move(exec_path));
    set_needs_layout();
}

void HomeScreenView::new_fast_launch_entity(const std::string& title, const std::string& icon_path, std::string&& exec_path)
{
    if (m_dock_stackview->subviews().size() >= grid_entities_per_row()) {
        return;
    }

    LG::PNG::PNGLoader loader;
    auto& icon_view = m_dock_stackview->add_arranged_subview<IconView>();
    icon_view.add_constraint(UI::Constraint(icon_view, UI::Constraint::Attribute::Height, UI::Constraint::Relation::Equal, icon_view_size()));
    icon_view.add_constraint(UI::Constraint(icon_view, UI::Constraint::Attribute::Width, UI::Constraint::Relation::Equal, icon_view_size()));
    icon_view.entity().set_icon(loader.load_from_file(icon_path + "/48x48.png"));
    icon_view.entity().set_path_to_exec(std::move(exec_path));
    set_needs_layout();
}

void HomeScreenView::on_window_create(const std::string& bundle_id, const std::string& icon_path, int window_id, int window_type)
{
    if (window_type == WindowType::AppList) {
        if (m_applist_view) {
            m_applist_view->set_target_window_id(window_id);
        }
        return;
    }
}