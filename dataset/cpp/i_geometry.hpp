// i_geometry.hpp
/*
  neogfx C++ App/Game Engine
  Copyright (c) 2015, 2020 Leigh Johnston.  All Rights Reserved.
  
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
#include <iostream>
#include <neolib/core/i_enum.hpp>
#include <neogfx/core/units.hpp>

namespace neogfx
{
    enum class size_constraint : uint32_t
    {
        Fixed,
        Minimum,
        Maximum,
        Expanding,
        MinimumExpanding,           // minimum unless a weight is defined
        DefaultMinimumExpanding,    // default minimum unless a weight is defined
        ExpandingUniform,           // leftover pixels are unwanted to ensure siblings are the same (pixel perfect) size after weighting
        Manual
    };

    enum class visibility_constraint : uint32_t
    {
        Ignore,
        Consider
    };
    
    enum class aspect_ratio : uint32_t
    {
        Ignore,
        Stretch,
        Keep,
        KeepExpanding
    };

    enum class cardinal : uint32_t
    {
        NorthWest,
        North,
        NorthEast,
        West,
        Center,
        East,
        SouthWest,
        South,
        SouthEast
    };

    enum class corner : uint32_t
    {
        TopLeft,
        TopRight,
        BottomRight,
        BottomLeft
    };
}

begin_declare_enum(neogfx::size_constraint)
declare_enum_string(neogfx::size_constraint, Fixed)
declare_enum_string(neogfx::size_constraint, Minimum)
declare_enum_string(neogfx::size_constraint, Maximum)
declare_enum_string(neogfx::size_constraint, Expanding)
declare_enum_string(neogfx::size_constraint, MinimumExpanding)
declare_enum_string(neogfx::size_constraint, DefaultMinimumExpanding)
declare_enum_string(neogfx::size_constraint, ExpandingUniform)
declare_enum_string(neogfx::size_constraint, Manual)
end_declare_enum(neogfx::size_constraint)

begin_declare_enum(neogfx::visibility_constraint)
declare_enum_string(neogfx::visibility_constraint, Ignore)
declare_enum_string(neogfx::visibility_constraint, Consider)
end_declare_enum(neogfx::visibility_constraint)

begin_declare_enum(neogfx::aspect_ratio)
declare_enum_string(neogfx::aspect_ratio, Ignore)
declare_enum_string(neogfx::aspect_ratio, Stretch)
declare_enum_string(neogfx::aspect_ratio, Keep)
declare_enum_string(neogfx::aspect_ratio, KeepExpanding)
end_declare_enum(neogfx::aspect_ratio)

begin_declare_enum(neogfx::cardinal)
declare_enum_string(neogfx::cardinal, NorthWest)
declare_enum_string(neogfx::cardinal, North)
declare_enum_string(neogfx::cardinal, NorthEast)
declare_enum_string(neogfx::cardinal, West)
declare_enum_string(neogfx::cardinal, Center)
declare_enum_string(neogfx::cardinal, East)
declare_enum_string(neogfx::cardinal, SouthWest)
declare_enum_string(neogfx::cardinal, South)
declare_enum_string(neogfx::cardinal, SouthEast)
end_declare_enum(neogfx::cardinal)

begin_declare_enum(neogfx::corner)
declare_enum_string(neogfx::corner, TopLeft)
declare_enum_string(neogfx::corner, TopRight)
declare_enum_string(neogfx::corner, BottomRight)
declare_enum_string(neogfx::corner, BottomLeft)
end_declare_enum(neogfx::corner)

namespace neogfx
{
    class size_policy
    {
    public:
        typedef size_policy abstract_type; // todo
    public:
        struct no_aspect_ratio : std::logic_error { no_aspect_ratio() : std::logic_error("neogfx::size_policy::no_aspect_ratio") {} };
    public:
        size_policy(
            size_constraint aSizeConstraint, 
            visibility_constraint aVisibility = visibility_constraint::Consider,
            optional_size const& aAspectRatio = {}) :
            iHorizontalConstraint{ aSizeConstraint }, 
            iVerticalConstraint{ aSizeConstraint }, 
            iVisibility{ aVisibility },
            iAspectRatio{ aAspectRatio }
        {
        }
        size_policy(
            size_constraint aHorizontalConstraint, 
            size_constraint aVerticalConstraint, 
            visibility_constraint aVisibility = visibility_constraint::Consider,
            optional_size const& aAspectRatio = {}) :
            iHorizontalConstraint{ aHorizontalConstraint }, 
            iVerticalConstraint{ aVerticalConstraint }, 
            iVisibility{ aVisibility },
            iAspectRatio{ aAspectRatio }
        {
        }
    public:
        bool operator==(const size_policy& aRhs) const
        {
            return iHorizontalConstraint == aRhs.iHorizontalConstraint &&
                iVerticalConstraint == aRhs.iVerticalConstraint &&
                iVisibility == aRhs.iVisibility &&
                iAspectRatio == aRhs.iAspectRatio;
        }
        bool operator!=(const size_policy& aRhs) const
        {
            return !(*this == aRhs);
        }
    public:
        size_constraint horizontal_constraint(bool aIgnoreUniformity = true) const
        {
            if (iHorizontalConstraint != size_constraint::ExpandingUniform)
                return iHorizontalConstraint;
            return aIgnoreUniformity ? size_constraint::Expanding : size_constraint::ExpandingUniform;
        }
        size_constraint vertical_constraint(bool aIgnoreUniformity = true) const
        {
            if (iVerticalConstraint != size_constraint::ExpandingUniform)
                return iVerticalConstraint;
            return aIgnoreUniformity ? size_constraint::Expanding : size_constraint::ExpandingUniform;
        }
        size_policy& set_constraint(size_constraint aConstraint)
        {
            iHorizontalConstraint = aConstraint;
            iVerticalConstraint = aConstraint;
            return *this;
        }
        size_policy& set_horizontal_constraint(size_constraint aHorizontalConstraint)
        {
            iHorizontalConstraint = aHorizontalConstraint;
            return *this;
        }
        size_policy& set_vertical_constraint(size_constraint aVerticalConstraint)
        {
            iVerticalConstraint = aVerticalConstraint;
            return *this;
        }
        visibility_constraint visibility() const
        {
            return iVisibility;
        }
        bool ignore_visibility() const
        {
            return iVisibility == visibility_constraint::Ignore;
        }
        size_policy& set_ignore_visibility(bool aIgnoreVisibility)
        {
            iVisibility = (aIgnoreVisibility ? visibility_constraint::Ignore : visibility_constraint::Consider);
            return *this;
        }
        bool maintain_aspect_ratio() const
        {
            return iAspectRatio != std::nullopt;
        }
        size aspect_ratio() const
        {
            if (maintain_aspect_ratio())
                return *iAspectRatio;
            throw no_aspect_ratio();
        }
        size_policy& set_aspect_ratio(optional_size const& aAspectRatio)
        {
            iAspectRatio = aAspectRatio;
            return *this;
        }
    public:
        static size_policy from_string(std::string const& aHorizontalConstraint, std::string const& aVerticalConstraint)
        {
            return size_policy{
                neolib::string_to_enum<size_constraint>(aHorizontalConstraint),
                neolib::string_to_enum<size_constraint>(aVerticalConstraint) };
        }
        void to_string(std::string& aHorizontalConstraint, std::string& aVerticalConstraint) const
        {
            aHorizontalConstraint = enum_to_string(horizontal_constraint());
            aVerticalConstraint = enum_to_string(vertical_constraint());
        }
    private:
        size_constraint iHorizontalConstraint;
        size_constraint iVerticalConstraint;
        visibility_constraint iVisibility;
        optional_size iAspectRatio;
    };

    template <typename Elem, typename Traits>
    inline std::basic_ostream<Elem, Traits>& operator<<(std::basic_ostream<Elem, Traits>& aStream, const size_policy& aPolicy)
    {
        aStream << "{" << aPolicy.horizontal_constraint() << " " << aPolicy.vertical_constraint() << " " << aPolicy.visibility() << " " << (aPolicy.maintain_aspect_ratio() ? aPolicy.aspect_ratio() : optional_size{}) << "}";
        return aStream;
    }

    using optional_size_policy = optional<size_policy>;

    template <typename Elem, typename Traits>
    inline std::basic_ostream<Elem, Traits>& operator<<(std::basic_ostream<Elem, Traits>& aStream, const optional_size_policy& aPolicy)
    {
        if (aPolicy != std::nullopt)
            return aStream << *aPolicy;
        return aStream << "{ nullopt }";
    }

    class i_geometry : public i_units_context
    {
        template <typename>
        friend class layout_item;
    public:
        virtual point origin() const = 0;
        virtual void reset_origin() const = 0;
        virtual point position() const = 0;
        virtual void set_position(const point& aPosition) = 0;
        virtual size extents() const = 0;
        virtual void set_extents(const size& aExtents) = 0;
        virtual bool has_size_policy() const noexcept = 0;
        virtual neogfx::size_policy size_policy() const = 0;
        virtual void set_size_policy(const optional_size_policy& aSizePolicy, bool aUpdateLayout = true) = 0;
        virtual bool has_weight() const noexcept = 0;
        virtual size weight() const = 0;
        virtual void set_weight(optional_size const& aWeight, bool aUpdateLayout = true) = 0;
        virtual bool has_ideal_size() const noexcept = 0;
        virtual bool is_ideal_size_constrained() const noexcept = 0; /// @todo remove when abstract anchor support added
        virtual size ideal_size(optional_size const& aAvailableSpace = {}) const = 0;
        virtual void set_ideal_size(optional_size const& aIdealSize, bool aUpdateLayout = true) = 0;
        virtual bool has_minimum_size() const noexcept = 0;
        virtual bool is_minimum_size_constrained() const noexcept = 0; /// @todo remove when abstract anchor support added
        virtual size minimum_size(optional_size const& aAvailableSpace = {}) const = 0;
        virtual void set_minimum_size(optional_size const& aMinimumSize, bool aUpdateLayout = true) = 0;
        virtual bool has_maximum_size() const noexcept = 0;
        virtual bool is_maximum_size_constrained() const noexcept = 0; /// @todo remove when abstract anchor support added
        virtual size maximum_size(optional_size const& aAvailableSpace = {}) const = 0;
        virtual void set_maximum_size(optional_size const& aMaximumSize, bool aUpdateLayout = true) = 0;
        virtual bool has_fixed_size() const noexcept = 0;
        virtual size fixed_size(optional_size const& aAvailableSpace = {}) const = 0;
        virtual void set_fixed_size(optional_size const& aFixedSize, bool aUpdateLayout = true) = 0;
        virtual bool has_transformation() const noexcept = 0;
        virtual mat33 const& transformation(bool aCombineAncestorTransformations = false) const = 0;
        virtual void set_transformation(optional_mat33 const& aTransformation, bool aUpdateLayout = true) = 0;
    public:
        virtual bool has_margin() const noexcept = 0;
        virtual neogfx::margin margin() const = 0;
        virtual void set_margin(optional_margin const& aMargin, bool aUpdateLayout = true) = 0;
        virtual bool has_border() const noexcept = 0;
        virtual neogfx::border border() const = 0;
        virtual void set_border(optional_border const& aBorder, bool aUpdateLayout = true) = 0;
        virtual bool has_padding() const noexcept = 0;
        virtual neogfx::padding padding() const = 0;
        virtual void set_padding(optional_padding const& aPadding, bool aUpdateLayout = true) = 0;
    protected:
        virtual point unconstrained_origin() const = 0;
        virtual point unconstrained_position() const = 0;
        // helpers
    public:
        size apply_fixed_size(size const& aResult) const
        {
            auto newResult = aResult;
            if (size_policy().horizontal_constraint() == size_constraint::Fixed && has_fixed_size())
                newResult.cx = fixed_size().cx;
            if (size_policy().vertical_constraint() == size_constraint::Fixed && has_fixed_size())
                newResult.cy = fixed_size().cy;
            return newResult;
        }
        neogfx::size_policy effective_size_policy() const
        {
            auto effectivePolicy = size_policy();
            if (effectivePolicy.horizontal_constraint() == size_constraint::MinimumExpanding)
                effectivePolicy.set_horizontal_constraint(has_weight() ? size_constraint::Expanding : size_constraint::Minimum);
            else if (effectivePolicy.horizontal_constraint() == size_constraint::DefaultMinimumExpanding)
                effectivePolicy.set_horizontal_constraint(has_weight() ? size_constraint::Expanding : size_constraint::Minimum);
            if (effectivePolicy.vertical_constraint() == size_constraint::MinimumExpanding)
                effectivePolicy.set_vertical_constraint(has_weight() ? size_constraint::Expanding : size_constraint::Minimum);
            else if (effectivePolicy.vertical_constraint() == size_constraint::DefaultMinimumExpanding)
                effectivePolicy.set_vertical_constraint(has_weight() ? size_constraint::Expanding : size_constraint::Minimum);
            return effectivePolicy;
        }
        void set_size_policy(size_constraint aConstraint, bool aUpdateLayout = true)
        {
            set_size_policy(neogfx::size_policy{ aConstraint }, aUpdateLayout);
        }
        void set_size_policy(size_constraint aConstraint, visibility_constraint aVisibility, bool aUpdateLayout = true)
        {
            set_size_policy(neogfx::size_policy{ aConstraint, aVisibility }, aUpdateLayout);
        }
        void set_size_policy(size_constraint aConstraint, visibility_constraint aVisibility, const size& aAspectRatio, bool aUpdateLayout = true)
        {
            set_size_policy(neogfx::size_policy{ aConstraint, aVisibility, aAspectRatio }, aUpdateLayout);
        }
        void set_size_policy(size_constraint aHorizontalConstraint, size_constraint aVerticalConstraint, bool aUpdateLayout = true)
        {
            set_size_policy(neogfx::size_policy{ aHorizontalConstraint, aVerticalConstraint }, aUpdateLayout);
        }
        void set_size_policy(size_constraint aHorizontalConstraint, size_constraint aVerticalConstraint, visibility_constraint aVisibility, bool aUpdateLayout = true)
        {
            set_size_policy(neogfx::size_policy{ aHorizontalConstraint, aVerticalConstraint, aVisibility }, aUpdateLayout);
        }
        void set_size_policy(size_constraint aHorizontalConstraint, size_constraint aVerticalConstraint, visibility_constraint aVisibility, const size& aAspectRatio, bool aUpdateLayout = true)
        {
            set_size_policy(neogfx::size_policy{ aHorizontalConstraint, aVerticalConstraint, aVisibility, aAspectRatio }, aUpdateLayout);
        }
        void set_minimum_width(dimension aWidth, bool aUpdateLayout = true)
        {
            auto newSize = minimum_size();
            newSize.cx = aWidth;
            set_minimum_size(newSize, aUpdateLayout);
        }
        void set_minimum_height(dimension aHeight, bool aUpdateLayout = true)
        {
            auto newSize = minimum_size();
            newSize.cy = aHeight;
            set_minimum_size(newSize, aUpdateLayout);
        }
        void set_maximum_width(dimension aWidth, bool aUpdateLayout = true)
        {
            auto newSize = maximum_size();
            newSize.cx = aWidth;
            set_maximum_size(newSize, aUpdateLayout);
        }
        void set_maximum_height(dimension aHeight, bool aUpdateLayout = true)
        {
            auto newSize = maximum_size();
            newSize.cy = aHeight;
            set_maximum_size(newSize, aUpdateLayout);
        }
    public:
        box_areas internal_spacing(bool aExtendIntoPadding = true) const
        {
            return margin() + border() + (aExtendIntoPadding ? padding() : box_areas{});
        }
    };

    class i_size_hint
    {
    public:
        typedef i_size_hint abstract_type;
    public:
        virtual ~i_size_hint() = default;
    public:
        virtual i_string const& primary_hint() const = 0;
        virtual i_string const& secondary_hint() const = 0;
    };

    class size_hint : public i_size_hint
    {
    public:
        size_hint(i_size_hint const& aOther) :
            iPrimaryHint{ aOther.primary_hint() },
            iSecondaryHint{ aOther.secondary_hint() }
        {
        }
        size_hint(string const& aPrimaryHint = {}, string const& aSecondaryHint = {}) :
            iPrimaryHint{ aPrimaryHint },
            iSecondaryHint{ aSecondaryHint }
        {
        }
    public:
        operator bool() const
        {
            return !primary_hint().empty() || !secondary_hint().empty();
        }
        bool operator==(const size_hint& aOther) const
        {
            return primary_hint() == aOther.primary_hint() && secondary_hint() == aOther.secondary_hint();
        }
        bool operator!=(const size_hint& aOther) const
        {
            return !(*this == aOther);
        }
    public:
        string const& primary_hint() const final
        {
            return iPrimaryHint;
        }
        string const& secondary_hint() const final
        {
            return iSecondaryHint;
        }
    private:
        string iPrimaryHint;
        string iSecondaryHint;
    };
}