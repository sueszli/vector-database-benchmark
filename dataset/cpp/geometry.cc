//
// Aspia Project
// Copyright (C) 2016-2023 Dmitry Chapyshev <dmitry@aspia.ru>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//

#include "base/desktop/geometry.h"

#include <algorithm>
#include <cmath>

namespace base {

//--------------------------------------------------------------------------------------------------
Rect::Rect(const Rect& other)
    : left_(other.left_),
      top_(other.top_),
      right_(other.right_),
      bottom_(other.bottom_)
{
    // Nothing
}

//--------------------------------------------------------------------------------------------------
void Rect::setTopLeft(const Point& top_left)
{
    left_ = top_left.x();
    top_ = top_left.y();
}

//--------------------------------------------------------------------------------------------------
void Rect::setSize(const Size& size)
{
    right_ = left_ + size.width();
    bottom_ = top_ + size.height();
}

//--------------------------------------------------------------------------------------------------
bool Rect::contains(int32_t x, int32_t y) const
{
    return (x >= left_ && x < right_ && y >= top_ && y < bottom_);
}

//--------------------------------------------------------------------------------------------------
bool Rect::contains(const Point& pos) const
{
    return contains(pos.x(), pos.y());
}

//--------------------------------------------------------------------------------------------------
bool Rect::containsRect(const Rect& rect) const
{
    return (rect.left_ >= left_ && rect.right_  <= right_ &&
            rect.top_  >= top_  && rect.bottom_ <= bottom_);
}

//--------------------------------------------------------------------------------------------------
void Rect::translate(int32_t dx, int32_t dy)
{
    left_   += dx;
    right_  += dx;
    top_    += dy;
    bottom_ += dy;
}

//--------------------------------------------------------------------------------------------------
Rect Rect::translated(int32_t dx, int32_t dy) const
{
    Rect result(*this);
    result.translate(dx, dy);
    return result;
}

//--------------------------------------------------------------------------------------------------
void Rect::intersectWith(const Rect& rect)
{
    left_   = std::max(left(),   rect.left());
    top_    = std::max(top(),    rect.top());
    right_  = std::min(right(),  rect.right());
    bottom_ = std::min(bottom(), rect.bottom());

    if (isEmpty())
    {
        left_   = 0;
        top_    = 0;
        right_  = 0;
        bottom_ = 0;
    }
}

//--------------------------------------------------------------------------------------------------
void Rect::unionWith(const Rect& rect)
{
    if (isEmpty())
    {
        *this = rect;
        return;
    }

    if (rect.isEmpty())
        return;

    left_ = std::min(left(), rect.left());
    top_ = std::min(top(), rect.top());
    right_ = std::max(right(), rect.right());
    bottom_ = std::max(bottom(), rect.bottom());
}

//--------------------------------------------------------------------------------------------------
void Rect::extend(int32_t left_offset,
                         int32_t top_offset,
                         int32_t right_offset,
                         int32_t bottom_offset)
{
    left_   -= left_offset;
    top_    -= top_offset;
    right_  += right_offset;
    bottom_ += bottom_offset;
}

//--------------------------------------------------------------------------------------------------
void Rect::scale(double horizontal, double vertical)
{
    right_ += static_cast<int32_t>(std::round(width() * (horizontal - 1)));
    bottom_ += static_cast<int32_t>(std::round(height() * (vertical - 1)));
}

//--------------------------------------------------------------------------------------------------
void Rect::move(int32_t x, int32_t y)
{
    right_  += x - left_;
    bottom_ += y - top_;
    left_ = x;
    top_  = y;
}

//--------------------------------------------------------------------------------------------------
Rect Rect::moved(int32_t x, int32_t y) const
{
    Rect moved_rect(*this);
    moved_rect.move(x, y);
    return moved_rect;
}

//--------------------------------------------------------------------------------------------------
Rect& Rect::operator=(const Rect& other)
{
    left_   = other.left_;
    top_    = other.top_;
    right_  = other.right_;
    bottom_ = other.bottom_;

    return *this;
}

//--------------------------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& stream, const Rect& rect)
{
    return stream << "Rect("
                  << rect.left() << ' ' << rect.top() << ' '
                  << rect.right() << ' ' << rect.bottom()
                  << ')';
}

//--------------------------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& stream, const Point& point)
{
    return stream << "Point(" << point.x() << ' ' << point.y() << ')';
}

//--------------------------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& stream, const Size& size)
{
    return stream << "Size(" << size.width() << ' ' << size.height() << ')';
}

} // namespace base
