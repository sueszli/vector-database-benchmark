// i_gradient.hpp
/*
  neogfx C++ App/Game Engine
  Copyright (c) 2020 Leigh Johnston.  All Rights Reserved.
  
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
#include <tuple>
#include <neogfx/core/geometrical.hpp>
#include <neogfx/gfx/color.hpp>
#include <neogfx/gfx/i_shader_array.hpp>
#include <neogfx/game/i_ecs.hpp>

namespace neogfx
{
    typedef game::id_t gradient_id;

    typedef neolib::variant<corner, scalar> gradient_orientation;

    enum class gradient_direction : uint32_t
    {
        Vertical,
        Horizontal,
        Diagonal,
        Rectangular,
        Radial
    };

    struct gradient_tile
    {
        size extents;
        bool aligned;
        auto operator<=>(const gradient_tile&) const = default;
    };

    enum class gradient_shape : uint32_t
    {
        Ellipse,
        Circle
    };

    enum class gradient_size : uint32_t
    {
        ClosestSide,
        FarthestSide,
        ClosestCorner,
        FarthestCorner
    };
}

begin_declare_enum(neogfx::gradient_direction)
declare_enum_string(neogfx::gradient_direction, Vertical)
declare_enum_string(neogfx::gradient_direction, Horizontal)
declare_enum_string(neogfx::gradient_direction, Diagonal)
declare_enum_string(neogfx::gradient_direction, Rectangular)
declare_enum_string(neogfx::gradient_direction, Radial)
end_declare_enum(neogfx::gradient_direction)

begin_declare_enum(neogfx::gradient_shape)
declare_enum_string(neogfx::gradient_shape, Ellipse)
declare_enum_string(neogfx::gradient_shape, Circle)
end_declare_enum(neogfx::gradient_shape)

begin_declare_enum(neogfx::gradient_size)
declare_enum_string(neogfx::gradient_size, ClosestSide)
declare_enum_string(neogfx::gradient_size, FarthestSide)
declare_enum_string(neogfx::gradient_size, ClosestCorner)
declare_enum_string(neogfx::gradient_size, FarthestCorner)
end_declare_enum(neogfx::gradient_size)

namespace neogfx
{
    class i_gradient_sampler
    {
        friend class gradient_manager;
    public:
        virtual ~i_gradient_sampler() = default;
    public:
        virtual i_shader_array<avec4u8> const& sampler() const = 0;
        virtual uint32_t sampler_row() const = 0;
        virtual bool used_by(gradient_id aGradient) const = 0;
        virtual void add_ref(gradient_id aGradient) const = 0;
        virtual void release(gradient_id aGradient) const = 0;
        virtual void release_all() const = 0;
    };

    class i_gradient_filter
    {
        friend class gradient_manager;
    public:
        virtual ~i_gradient_filter() = default;
    public:
        virtual i_shader_array<float> const& sampler() const = 0;
    private:
        virtual i_shader_array<float>& sampler() = 0;
    };

    enum class gradient_sharing
    {
        Shared,
        Unique
    };

    class i_gradient : public i_reference_counted
    {
        template <gradient_sharing>
        friend class basic_gradient;
        // exceptions
    public:
        struct bad_position : std::logic_error { bad_position() : std::logic_error("neogfx::i_gradient::bad_position") {} };
        // constants
    public:
        static const std::uint32_t MaxStops = 256;
        // types
    public:
        typedef i_gradient abstract_type;
        typedef neolib::i_pair<scalar, sRGB_color::abstract_type> color_stop;
        typedef neolib::i_pair<scalar, sRGB_color::view_component> alpha_stop;
        typedef neolib::i_vector<color_stop> color_stop_list;
        typedef neolib::i_vector<alpha_stop> alpha_stop_list;
        // construction
    public:
        virtual ~i_gradient() = default;
        virtual void clone(neolib::i_ref_ptr<i_gradient>& aResult) const = 0;
        // assignment
    public:
        virtual i_gradient& operator=(const i_gradient& aOther) = 0;
        // meta
    public:
        virtual gradient_id id() const = 0;
        virtual bool is_singular() const = 0;
        // operations
    public:
        virtual color_stop_list const& color_stops() const = 0;
        virtual color_stop_list& color_stops() = 0;
        virtual alpha_stop_list const& alpha_stops() const = 0;
        virtual alpha_stop_list& alpha_stops() = 0;
        virtual color_stop_list::const_iterator find_color_stop(scalar aPos, bool aToInsert = false) const = 0;
        virtual color_stop_list::const_iterator find_color_stop(scalar aPos, scalar aStart, scalar aEnd, bool aToInsert = false) const = 0;
        virtual alpha_stop_list::const_iterator find_alpha_stop(scalar aPos, bool aToInsert = false) const = 0;
        virtual alpha_stop_list::const_iterator find_alpha_stop(scalar aPos, scalar aStart, scalar aEnd, bool aToInsert = false) const = 0;
        virtual color_stop_list::iterator find_color_stop(scalar aPos, bool aToInsert = false) = 0;
        virtual color_stop_list::iterator find_color_stop(scalar aPos, scalar aStart, scalar aEnd, bool aToInsert = false) = 0;
        virtual alpha_stop_list::iterator find_alpha_stop(scalar aPos, bool aToInsert = false) = 0;
        virtual alpha_stop_list::iterator find_alpha_stop(scalar aPos, scalar aStart, scalar aEnd, bool aToInsert = false) = 0;
        virtual color_stop_list::iterator insert_color_stop(scalar aPos) = 0;
        virtual color_stop_list::iterator insert_color_stop(scalar aPos, scalar aStart, scalar aEnd) = 0;
        virtual alpha_stop_list::iterator insert_alpha_stop(scalar aPos) = 0;
        virtual alpha_stop_list::iterator insert_alpha_stop(scalar aPos, scalar aStart, scalar aEnd) = 0;
        virtual sRGB_color at(scalar aPos) const = 0;
        virtual sRGB_color at(scalar aPos, scalar aStart, scalar aEnd) const = 0;
        virtual sRGB_color color_at(scalar aPos) const = 0;
        virtual sRGB_color color_at(scalar aPos, scalar aStart, scalar aEnd) const = 0;
        virtual sRGB_color::view_component alpha_at(scalar aPos) const = 0;
        virtual sRGB_color::view_component alpha_at(scalar aPos, scalar aStart, scalar aEnd) const = 0;
        virtual i_gradient& reverse() = 0;
        virtual i_gradient& set_alpha(sRGB_color::view_component aAlpha) = 0;
        virtual i_gradient& set_combined_alpha(sRGB_color::view_component aAlpha) = 0;
        virtual gradient_direction direction() const = 0;
        virtual i_gradient& set_direction(gradient_direction aDirection) = 0;
        virtual gradient_orientation orientation() const = 0;
        virtual i_gradient& set_orientation(gradient_orientation aOrientation) = 0;
        virtual gradient_shape shape() const = 0;
        virtual i_gradient& set_shape(gradient_shape aShape) = 0;
        virtual gradient_size size() const = 0;
        virtual i_gradient& set_size(gradient_size aSize) = 0;
        virtual const optional_vec2& exponents() const = 0;
        virtual i_gradient& set_exponents(const optional_vec2& aExponents) = 0;
        virtual const optional_point& center() const = 0;
        virtual i_gradient& set_center(const optional_point& aCenter) = 0;
        virtual const std::optional<gradient_tile>& tile() const = 0;
        virtual i_gradient& set_tile(const std::optional<gradient_tile>& aTile) = 0;
        virtual scalar smoothness() const = 0;
        virtual i_gradient& set_smoothness(scalar aSmoothness) = 0;
        virtual const optional_rect& bounding_box() const = 0;
        virtual i_gradient& set_bounding_box(const optional_rect& aBoundingBox) = 0;
        virtual i_gradient& set_bounding_box_if_none(const optional_rect& aBoundingBox) = 0;
        // shader
    public:
        virtual const i_gradient_sampler& colors() const = 0;
        virtual const i_gradient_filter& filter() const = 0;
        // object
    private:
        virtual void share_object(i_ref_ptr<i_gradient>& aRef) const = 0;
        // helpers
    public:
        neolib::ref_ptr<i_gradient> clone() const
        {
            neolib::ref_ptr<i_gradient> result;
            clone(result);
            return result;
        }
        neolib::ref_ptr<i_gradient> reversed() const
        {
            auto result = clone();
            result->reverse();
            return result;
        }
        neolib::ref_ptr<i_gradient> with_alpha(sRGB_color::view_component aAlpha) const
        {
            auto result = clone();
            result->set_alpha(aAlpha);
            return result;
        }
        template <typename T>
        neolib::ref_ptr<i_gradient> with_alpha(T aAlpha, std::enable_if_t<!std::is_same_v<T, sRGB_color::view_component>, sfinae> = {})
        {
            return with_alpha(sRGB_color::convert<sRGB_color::view_component>(aAlpha));
        }
        neolib::ref_ptr<i_gradient> with_combined_alpha(sRGB_color::view_component aAlpha) const
        {
            auto result = clone();
            result->set_combined_alpha(aAlpha);
            return result;
        }
        template <typename T>
        neolib::ref_ptr<i_gradient> with_combined_alpha(T aAlpha, std::enable_if_t<!std::is_same_v<T, sRGB_color::view_component>, sfinae> = {}) const
        {
            return with_combined_alpha(sRGB_color::convert<sRGB_color::view_component>(aAlpha));
        }
        neolib::ref_ptr<i_gradient> with_direction(gradient_direction aDirection) const
        {
            auto result = clone();
            result->set_direction(aDirection);
            return result;
        }
        neolib::ref_ptr<i_gradient> with_orientation(gradient_orientation aOrientation) const
        {
            auto result = clone();
            result->set_orientation(aOrientation);
            return result;
        }
        neolib::ref_ptr<i_gradient> with_shape(gradient_shape aShape) const
        {
            auto result = clone();
            result->set_shape(aShape);
            return result;
        }
        neolib::ref_ptr<i_gradient> with_size(gradient_size aSize) const
        {
            auto result = clone();
            result->set_size(aSize);
            return result;
        }
        neolib::ref_ptr<i_gradient> with_exponents(const optional_vec2& aExponents) const
        {
            auto result = clone();
            result->set_exponents(aExponents);
            return result;
        }
        neolib::ref_ptr<i_gradient> with_center(const optional_point& aCenter) const
        {
            auto result = clone();
            result->set_center(aCenter);
            return result;
        }
        neolib::ref_ptr<i_gradient> with_tile(const std::optional<gradient_tile>& aTile) const
        {
            auto result = clone();
            result->set_tile(aTile);
            return result;
        }
        neolib::ref_ptr<i_gradient> with_smoothness(scalar aSmoothness) const
        {
            auto result = clone();
            result->set_smoothness(aSmoothness);
            return result;
        }
    public:
        static scalar normalized_position(scalar aPos, scalar aStart, scalar aEnd)
        {
            if (aStart != aEnd)
                return std::max(0.0, std::min(1.0, (aPos - aStart) / (aEnd - aStart)));
            else
                return 0.0;
        }
    };

    inline bool operator==(const i_gradient& aLhs, const i_gradient& aRhs)
    {
        if (aLhs.is_singular() != aRhs.is_singular())
            return false;
        else if (aLhs.is_singular())
            return false;
        else if (aLhs.id() == aRhs.id())
            return true;
        else
            return std::forward_as_tuple(aLhs.color_stops(), aLhs.alpha_stops(), aLhs.direction(), aLhs.orientation(), aLhs.shape(), aLhs.size(), aLhs.exponents(), aLhs.center(), aLhs.tile(), aLhs.smoothness()) ==
                std::forward_as_tuple(aRhs.color_stops(), aRhs.alpha_stops(), aRhs.direction(), aRhs.orientation(), aRhs.shape(), aRhs.size(), aRhs.exponents(), aRhs.center(), aRhs.tile(), aRhs.smoothness());
    }

    inline std::partial_ordering operator<=>(const i_gradient& aLhs, const i_gradient& aRhs)
    {
        if (aLhs.is_singular() || aRhs.is_singular())
        {
            if (aLhs.is_singular() == aRhs.is_singular())
                return std::partial_ordering::unordered;
            if (aLhs.is_singular() < aRhs.is_singular())
                return std::partial_ordering::less;
            else
                return std::partial_ordering::greater;
        }
        else if (aLhs.id() == aRhs.id())
            return std::partial_ordering::equivalent;
        else
            return std::forward_as_tuple(aLhs.color_stops(), aLhs.alpha_stops(), aLhs.direction(), aLhs.orientation(), aLhs.shape(), aLhs.size(), aLhs.exponents(), aLhs.center(), aLhs.tile(), aLhs.smoothness()) <=>
                std::forward_as_tuple(aRhs.color_stops(), aRhs.alpha_stops(), aRhs.direction(), aRhs.orientation(), aRhs.shape(), aRhs.size(), aRhs.exponents(), aRhs.center(), aRhs.tile(), aRhs.smoothness());
    }
}