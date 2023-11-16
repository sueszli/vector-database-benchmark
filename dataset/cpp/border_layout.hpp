// border_layout.hpp
/*
  neogfx C++ App/Game Engine
  Copyright (c) 2018, 2020 Leigh Johnston.  All Rights Reserved.
  
  This program is free software: you can redistribute it and / or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <neogfx/neogfx.hpp>
#include <neogfx/core/geometrical.hpp>
#include <neogfx/gui/layout/grid_layout.hpp>
#include <neogfx/gui/layout/horizontal_layout.hpp>
#include <neogfx/gui/layout/vertical_layout.hpp>
#include <neogfx/gui/layout/stack_layout.hpp>

namespace neogfx
{
    class border_layout : public layout
    {
        meta_object(layout)
    public:
        struct not_implemented : std::logic_error { not_implemented() : std::logic_error("neogfx::border_layout::not_implemented") {} };
    public:
        border_layout(neogfx::alignment aAlignment = neogfx::alignment::VCenter);
        border_layout(i_widget& aParent, neogfx::alignment aAlignment = neogfx::alignment::VCenter);
        border_layout(i_layout& aParent, neogfx::alignment aAlignment = neogfx::alignment::VCenter);
        ~border_layout();
    public:
        const i_layout& part(layout_position aPosition) const;
        i_layout& part(layout_position aPosition);
        const i_layout& top() const;
        i_layout& top();
        const i_layout& left() const;
        i_layout& left();
        const i_layout& center() const;
        i_layout& center();
        const i_layout& right() const;
        i_layout& right();
        const i_layout& bottom() const;
        i_layout& bottom();
    public:
        i_spacer& add_spacer() override;
        i_spacer& add_spacer_at(layout_item_index aPosition) override;
    public:
        void invalidate(bool aDeferLayout = true) override;
    public:
        void layout_items(const point& aPosition, const size& aSize) override;
        void fix_weightings(bool aRecalculate = true) override;
    public:
        size minimum_size(optional_size const& aAvailableSpace = optional_size{}) const override;
        size maximum_size(optional_size const& aAvailableSpace = optional_size{}) const override;
    public:
        void set_autoscale(neogfx::autoscale aAutoscale, bool aUpdateLayout = true) override;
    private:
        void init();
    private:
        vertical_layout iRows;
        horizontal_layout iTop;
        horizontal_layout iMiddle;
        vertical_layout iLeft;
        stack_layout iCenter;
        vertical_layout iRight;
        horizontal_layout iBottom;
    };
}