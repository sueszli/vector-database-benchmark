// border_layout.cpp
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
#include <neogfx/gui/layout/border_layout.hpp>
#include <neogfx/gui/layout/spacer.hpp>
#include <neogfx/gui/widget/i_widget.hpp>

namespace neogfx
{
    border_layout::border_layout(neogfx::alignment aAlignment) :
        layout{ aAlignment },
        iRows{ *this, aAlignment },
        iTop{ iRows, aAlignment },
        iMiddle{ iRows, aAlignment },
        iLeft{ iMiddle, aAlignment },
        iCenter{ iMiddle, aAlignment },
        iRight{ iMiddle, aAlignment },
        iBottom{ iRows, aAlignment }
    {
        init();
    }

    border_layout::border_layout(i_widget& aParent, neogfx::alignment aAlignment) :
        layout{ aParent, aAlignment },
        iRows{ *this, aAlignment },
        iTop{ iRows, aAlignment },
        iMiddle{ iRows, aAlignment },
        iLeft{ iMiddle, aAlignment },
        iCenter{ iMiddle, aAlignment },
        iRight{ iMiddle, aAlignment },
        iBottom{ iRows, aAlignment }
    {
        init();
    }

    border_layout::border_layout(i_layout& aParent, neogfx::alignment aAlignment) :
        layout{ aParent, aAlignment },
        iRows{ *this, aAlignment },
        iTop{ iRows, aAlignment },
        iMiddle{ iRows, aAlignment },
        iLeft{ iMiddle, aAlignment },
        iCenter{ iMiddle, aAlignment },
        iRight{ iMiddle, aAlignment },
        iBottom{ iRows, aAlignment }
    {
        init();
    }

    border_layout::~border_layout()
    {
        set_destroying();
    }

    const i_layout& border_layout::part(layout_position aPosition) const
    {
        switch (aPosition)
        {
        case layout_position::Top:
            return top();
        case layout_position::Left:
            return left();
        case layout_position::Center:
            return center();
        case layout_position::Right:
            return right();
        case layout_position::Bottom:
            return bottom();
        }
        return center();
    }

    i_layout& border_layout::part(layout_position aPosition)
    {
        return const_cast<i_layout&>(to_const(*this).part(aPosition));
    }

    const i_layout& border_layout::top() const
    {
        return iTop;
    }

    i_layout& border_layout::top()
    {
        return iTop;
    }

    const i_layout& border_layout::left() const
    {
        return iLeft;
    }

    i_layout& border_layout::left()
    {
        return iLeft;
    }

    const i_layout& border_layout::center() const
    {
        return iCenter;
    }

    i_layout& border_layout::center()
    {
        return iCenter;
    }

    const i_layout& border_layout::right() const
    {
        return iRight;
    }

    i_layout& border_layout::right()
    {
        return iRight;
    }

    const i_layout& border_layout::bottom() const
    {
        return iBottom;
    }

    i_layout& border_layout::bottom()
    {
        return iBottom;
    }

    i_spacer & border_layout::add_spacer()
    {
        throw not_implemented();
    }

    i_spacer& border_layout::add_spacer_at(layout_item_index)
    {
        throw not_implemented();
    }

    void border_layout::invalidate(bool aDeferLayout)
    {
        if (!is_alive())
            return;
        iRows.invalidate(true);
        iMiddle.invalidate(true);
        iTop.invalidate(true);
        iLeft.invalidate(true);
        iCenter.invalidate(true);
        iRight.invalidate(true);
        iBottom.invalidate(true);
        layout::invalidate(aDeferLayout);
    }

    void border_layout::layout_items(const point& aPosition, const size& aSize)
    {
        scoped_layout_items layoutItems;
        validate();
        set_position(aPosition);
        set_extents(aSize);
        iRows.layout_items(aPosition, aSize);
    }

    void border_layout::fix_weightings(bool aRecalculate)
    {
        iRows.fix_weightings(aRecalculate);
        iMiddle.fix_weightings(aRecalculate);
        invalidate(false);
    }

    size border_layout::minimum_size(optional_size const& aAvailableSpace) const
    {
        size result = iRows.minimum_size(aAvailableSpace);
#ifdef NEOGFX_DEBUG
        if (debug::layoutItem == this)
            service<debug::logger>() << neolib::logger::severity::Debug << "border_layout::minimum_size(...) --> " << result << endl;
#endif // NEOGFX_DEBUG
        return result;
    }

    size border_layout::maximum_size(optional_size const& aAvailableSpace) const
    {
        size result = iRows.maximum_size(aAvailableSpace);
#ifdef NEOGFX_DEBUG
        if (debug::layoutItem == this)
            service<debug::logger>() << neolib::logger::severity::Debug << "border_layout::maximum_size(...) --> " << result << endl;
#endif // NEOGFX_DEBUG
        return result;
    }

    void border_layout::set_autoscale(neogfx::autoscale aAutoscale, bool aUpdateLayout)
    {
        iRows.set_autoscale(aAutoscale, aUpdateLayout);
        iMiddle.set_autoscale(aAutoscale, aUpdateLayout);
    }

    void border_layout::init()
    {
        set_padding(neogfx::padding{});
        iRows.set_padding(neogfx::padding{});
        iTop.set_padding(neogfx::padding{});
        iMiddle.set_padding(neogfx::padding{});
        iLeft.set_padding(neogfx::padding{});
        iCenter.set_padding(neogfx::padding{});
        iRight.set_padding(neogfx::padding{});
        iBottom.set_padding(neogfx::padding{});

        set_spacing(neogfx::size{});
        iRows.set_spacing(neogfx::size{});
        iTop.set_spacing(neogfx::size{});
        iMiddle.set_spacing(neogfx::size{});
        iLeft.set_spacing(neogfx::size{});
        iCenter.set_spacing(neogfx::size{});
        iRight.set_spacing(neogfx::size{});
        iBottom.set_spacing(neogfx::size{});

        set_size_policy(size_constraint::Expanding);
        iRows.set_size_policy(size_constraint::Expanding);
        iTop.set_size_policy(neogfx::size_policy{ size_constraint::Expanding, size_constraint::Minimum });
        iMiddle.set_size_policy(size_constraint::Expanding);
        iCenter.set_size_policy(size_constraint::Expanding);
        iBottom.set_size_policy(neogfx::size_policy{ size_constraint::Expanding, size_constraint::Minimum });

        iMiddle.set_minimum_size(neogfx::size{});

        set_alive();
        invalidate();
    }
}