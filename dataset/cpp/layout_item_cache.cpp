// layout_item_cache.cpp
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

#include <neogfx/neogfx.hpp>
#include <neogfx/gui/layout/layout_item_cache.hpp>
#include <neogfx/gui/layout/layout_item.hpp>
#include <neogfx/gui/layout/i_layout.hpp>
#include <neogfx/gui/layout/i_spacer.hpp>
#include <neogfx/gui/widget/i_widget.hpp>

namespace neogfx
{
    layout_item_cache::layout_item_cache(i_layout_item& aItem) :
        layout_item_cache{ ref_ptr<i_layout_item>{ ref_ptr<i_layout_item>{}, &aItem } } 
    {
    }

    layout_item_cache::layout_item_cache(i_ref_ptr<i_layout_item> const& aItem) :
        iSubject{ aItem },  
        iSubjectDestroyed{ *iSubject },
        iVisible{ static_cast<uint32_t>(-1), {} },
        iSizePolicy{ static_cast<uint32_t>(-1), { size_constraint::Minimum } }, 
        iWeight{ static_cast<uint32_t>(-1), {} },
        iIdealSize{ static_cast<uint32_t>(-1), {} },
        iMinimumSize{ static_cast<uint32_t>(-1), {} },
        iMaximumSize{ static_cast<uint32_t>(-1), {} },
        iFixedSize{ static_cast<uint32_t>(-1), {} },
        iTransformation{ static_cast<uint32_t>(-1), mat33::identity() },
        iCombinedTransformation{ static_cast<uint32_t>(-1), mat33::identity() }
    {
        set_alive();
    }

    layout_item_cache::layout_item_cache(const layout_item_cache& aOther) :
        iSubject{ aOther.iSubject }, 
        iSubjectDestroyed{ *iSubject },
        iVisible{ static_cast<uint32_t>(-1), {} },
        iSizePolicy{ static_cast<uint32_t>(-1), { size_constraint::Minimum } },
        iWeight{ static_cast<uint32_t>(-1), {} },
        iIdealSize{ static_cast<uint32_t>(-1), {} },
        iMinimumSize{ static_cast<uint32_t>(-1), {} },
        iMaximumSize{ static_cast<uint32_t>(-1), {} },
        iFixedSize{ static_cast<uint32_t>(-1), {} },
        iTransformation{ static_cast<uint32_t>(-1), mat33::identity() },
        iCombinedTransformation{ static_cast<uint32_t>(-1), mat33::identity() }
    {
        set_alive();
    }

    layout_item_cache::~layout_item_cache()
    {
    }

    i_layout_item& layout_item_cache::subject() const
    {
        if (!subject_destroyed())
            return *iSubject;
        throw layout_item_cache_subject_destroyed();
    }

    bool layout_item_cache::subject_destroyed() const
    {
        return iSubjectDestroyed;
    }
  
    layout_item_disposition& layout_item_cache::cached_disposition() const
    {
        return iCachedDisposition;
    }

    i_anchor& layout_item_cache::anchor_to(i_anchorable& aRhs, const i_string& aLhsAnchor, anchor_constraint_function aLhsFunction, const i_string& aRhsAnchor, anchor_constraint_function aRhsFunction)
    {
        return subject().anchor_to(aRhs, aLhsAnchor, aLhsFunction, aRhsAnchor, aRhsFunction);
    }

    const layout_item_cache::anchor_map_type& layout_item_cache::anchors() const
    {
        return subject().anchors();
    }

    layout_item_cache::anchor_map_type& layout_item_cache::anchors()
    {
        return subject().anchors();
    }

    const i_string& layout_item_cache::id() const
    {
        return subject().id();
    }

    void layout_item_cache::set_id(const i_string& aId)
    {
        subject().set_id(aId);
    }

    bool layout_item_cache::is_cache() const
    {
        return true;
    }

    bool layout_item_cache::is_layout() const
    {
        return subject().is_layout();
    }

    const i_layout& layout_item_cache::as_layout() const
    {
        return subject().as_layout();
    }

    i_layout& layout_item_cache::as_layout()
    {
        return subject().as_layout();
    }

    bool layout_item_cache::is_spacer() const
    {
        return subject().is_spacer();
    }

    const i_spacer& layout_item_cache::as_spacer() const
    {
        return subject().as_spacer();
    }

    i_spacer& layout_item_cache::as_spacer()
    {
        return subject().as_spacer();
    }

    bool layout_item_cache::is_widget() const
    {
        return subject().is_widget();
    }

    const i_widget& layout_item_cache::as_widget() const
    {
        return subject().as_widget();
    }

    i_widget& layout_item_cache::as_widget()
    {
        return subject().as_widget();
    }

    bool layout_item_cache::has_parent_layout_item() const
    {
        return subject().has_parent_layout_item();
    }

    const i_layout_item& layout_item_cache::parent_layout_item() const
    {
        return subject().parent_layout_item();
    }

    i_layout_item& layout_item_cache::parent_layout_item()
    {
        return subject().parent_layout_item();
    }

    bool layout_item_cache::has_parent_layout() const
    {
        return subject().has_parent_layout();
    }

    const i_layout& layout_item_cache::parent_layout() const
    {
        if (has_parent_layout())
            return subject().parent_layout();
        throw no_parent_layout();
    }

    i_layout& layout_item_cache::parent_layout()
    {
        return const_cast<i_layout&>(to_const(*this).parent_layout());
    }

    void layout_item_cache::set_parent_layout(i_layout* aParentLayout)
    {
        subject().set_parent_layout(aParentLayout);
    }

    bool layout_item_cache::has_parent_widget() const
    {
        return subject().has_parent_widget();
    }

    const i_widget& layout_item_cache::parent_widget() const
    {
        if (has_parent_widget())
            return subject().parent_widget();
        throw no_parent_widget();
    }

    i_widget& layout_item_cache::parent_widget()
    {
        return const_cast<i_widget&>(to_const(*this).parent_widget());
    }

    void layout_item_cache::set_parent_widget(i_widget* aParentWidget)
    {
        subject().set_parent_widget(aParentWidget);
    }

    bool layout_item_cache::has_layout_manager() const
    {
        return subject().has_layout_manager();
    }

    const i_widget& layout_item_cache::layout_manager() const
    {
        if (has_layout_manager())
            return subject().layout_manager();
        throw no_layout_manager();
    }

    i_widget& layout_item_cache::layout_manager()
    {
        return const_cast<i_widget&>(to_const(*this).layout_manager());
    }

    void layout_item_cache::update_layout(bool aDeferLayout, bool aAncestors)
    {
        subject().update_layout(aDeferLayout, aAncestors);
    }

    void layout_item_cache::layout_as(const point& aPosition, const size& aSize)
    {
        point adjustedPosition = aPosition;
        size adjustedSize = aSize.min(maximum_size(aSize));
        if (adjustedSize != aSize)
        {
            adjustedPosition += point{
            (parent_layout().alignment() & alignment::Center) == alignment::Center ?
                (aSize - adjustedSize).cx / 2.0 :
                (parent_layout().alignment() & alignment::Right) == alignment::Right ?
                    (aSize - adjustedSize).cx :
                    0.0,
            (parent_layout().alignment() & alignment::VCenter) == alignment::VCenter ?
                (aSize - adjustedSize).cy / 2.0 :
                (parent_layout().alignment() & alignment::Bottom) == alignment::Bottom ?
                    (aSize - adjustedSize).cy :
                    0.0 }.floor();
        }

        subject().layout_as(adjustedPosition, adjustedSize);
    }

    void layout_item_cache::invalidate_combined_transformation()
    {
        subject().invalidate_combined_transformation();
        iCombinedTransformation.second = subject().transformation(true);
    }

    void layout_item_cache::fix_weightings(bool aRecalculate)
    {
        subject().fix_weightings(aRecalculate);
    }

    bool layout_item_cache::device_metrics_available() const
    {
        return parent_layout().device_metrics_available();
    }

    const i_device_metrics& layout_item_cache::device_metrics() const
    {
        return parent_layout().device_metrics();
    }

    point layout_item_cache::origin() const
    {
        return subject().origin();
    }

    void layout_item_cache::reset_origin() const
    {
        subject().reset_origin();
    }

    point layout_item_cache::position() const
    {
        return subject().position();
    }

    void layout_item_cache::set_position(const point& aPosition)
    {
        subject().set_position(aPosition);
    }

    size layout_item_cache::extents() const
    {
        return subject().extents();
    }

    void layout_item_cache::set_extents(const size& aExtents)
    {
        subject().set_extents(aExtents);
    }

    bool layout_item_cache::has_size_policy() const noexcept
    {
        return subject().has_size_policy();
    }

    size_policy layout_item_cache::size_policy() const
    {
#ifdef NEOGFX_DEBUG
        if (&subject() == debug::layoutItem)
            service<debug::logger>() << neolib::logger::severity::Debug << "layout_item_cache::size_policy()" << endl;
#endif // NEOGFX_DEBUG
        auto& cachedSizePolicy = iSizePolicy.second;
        if (iSizePolicy.first != global_layout_id())
        {
            cachedSizePolicy = subject().size_policy();
            iSizePolicy.first = global_layout_id();
        }
#ifdef NEOGFX_DEBUG
        if (&subject() == debug::layoutItem)
            service<debug::logger>() << neolib::logger::severity::Debug << "layout_item_cache::size_policy() -> " << cachedSizePolicy << endl;
#endif // NEOGFX_DEBUG
        return cachedSizePolicy;
    }

    void layout_item_cache::set_size_policy(const optional_size_policy& aSizePolicy, bool aUpdateLayout)
    {
        subject().set_size_policy(aSizePolicy, aUpdateLayout);
    }

    bool layout_item_cache::has_weight() const noexcept
    {
        return subject().has_weight();
    }

    size layout_item_cache::weight() const
    {
#ifdef NEOGFX_DEBUG
        if (&subject() == debug::layoutItem)
            service<debug::logger>() << neolib::logger::severity::Debug << "layout_item_cache::weight()" << endl;
#endif // NEOGFX_DEBUG
        auto& cachedWeight = iWeight.second;
        if (iWeight.first != global_layout_id())
        {
            cachedWeight = subject().weight();
            iWeight.first = global_layout_id();
        }
#ifdef NEOGFX_DEBUG
        if (&subject() == debug::layoutItem)
            service<debug::logger>() << neolib::logger::severity::Debug << "layout_item_cache::weight() -> " << cachedWeight << endl;
#endif // NEOGFX_DEBUG
        return cachedWeight;
    }

    void layout_item_cache::set_weight(optional_size const& aWeight, bool aUpdateLayout)
    {
        subject().set_weight(aWeight, aUpdateLayout);
    }
    
    bool layout_item_cache::has_ideal_size() const noexcept
    {
        return subject().has_ideal_size();
    }

    bool layout_item_cache::is_ideal_size_constrained() const noexcept
    {
        return subject().is_ideal_size_constrained();
    }

    size layout_item_cache::ideal_size(optional_size const& aAvailableSpace) const
    {
#ifdef NEOGFX_DEBUG
        if (&subject() == debug::layoutItem)
            service<debug::logger>() << neolib::logger::severity::Debug << "layout_item_cache::ideal_size(" << aAvailableSpace << ")" << endl;
#endif // NEOGFX_DEBUG
        if (!visible())
            return size{};
        scoped_units su{ subject(), units::Pixels };
        auto& cachedIdealSize = iIdealSize.second.second;
        if (iIdealSize.first != global_layout_id() || iIdealSize.second.first != aAvailableSpace || is_minimum_size_constrained())
        {
#ifdef NEOGFX_DEBUG
            if (&subject() == debug::layoutItem)
                service<debug::logger>() << neolib::logger::severity::Debug << "layout_item_cache::ideal_size(" << aAvailableSpace << ") (cache invalid)" << endl;
#endif // NEOGFX_DEBUG
            cachedIdealSize = subject().ideal_size(aAvailableSpace);
            auto const ourSizePolicy = effective_size_policy();
            if (ourSizePolicy.maintain_aspect_ratio())
            {
                auto const& aspectRatio = ourSizePolicy.aspect_ratio();
                if (aspectRatio.cx < aspectRatio.cy)
                {
                    if (cachedIdealSize.cx < cachedIdealSize.cy)
                        cachedIdealSize = size{ cachedIdealSize.cx, cachedIdealSize.cx * (aspectRatio.cy / aspectRatio.cx) };
                    else
                        cachedIdealSize = size{ cachedIdealSize.cy * (aspectRatio.cx / aspectRatio.cy), cachedIdealSize.cy };
                }
                else
                {
                    if (cachedIdealSize.cx < cachedIdealSize.cy)
                        cachedIdealSize = size{ cachedIdealSize.cy * (aspectRatio.cx / aspectRatio.cy), cachedIdealSize.cy };
                    else
                        cachedIdealSize = size{ cachedIdealSize.cx, cachedIdealSize.cx * (aspectRatio.cy / aspectRatio.cx) };
                }
            }
            cachedIdealSize = subject().apply_fixed_size(cachedIdealSize);
            iIdealSize.first = global_layout_id();
            iIdealSize.second.first = aAvailableSpace;
        }
        auto const result = transformation() * cachedIdealSize;
#ifdef NEOGFX_DEBUG
        if (&subject() == debug::layoutItem)
            service<debug::logger>() << neolib::logger::severity::Debug << "layout_item_cache::ideal_size(" << aAvailableSpace << ") -> " << cachedIdealSize << " -> " << result << endl;
#endif // NEOGFX_DEBUG
        return to_units(subject(), su.saved_units(), result);
    }

    void layout_item_cache::set_ideal_size(optional_size const& aIdealSize, bool aUpdateLayout)
    {
        subject().set_ideal_size(aIdealSize, aUpdateLayout);
        if (aIdealSize != std::nullopt)
            iIdealSize.second.second = *aIdealSize;
    }

    bool layout_item_cache::has_minimum_size() const noexcept
    {
        return subject().has_minimum_size();
    }

    bool layout_item_cache::is_minimum_size_constrained() const noexcept
    {
        return subject().is_minimum_size_constrained();
    }

    size layout_item_cache::minimum_size(optional_size const& aAvailableSpace) const
    {
#ifdef NEOGFX_DEBUG
        if (&subject() == debug::layoutItem)
            service<debug::logger>() << neolib::logger::severity::Debug << "layout_item_cache::minimum_size(" << aAvailableSpace << ")" << endl;
#endif // NEOGFX_DEBUG
        if (!visible())
            return size{};
        scoped_units su{ subject(), units::Pixels };
        auto& cachedMinSize = iMinimumSize.second.second;
        if (iMinimumSize.first != global_layout_id() || iMinimumSize.second.first != aAvailableSpace || is_minimum_size_constrained())
        {
#ifdef NEOGFX_DEBUG
            if (&subject() == debug::layoutItem)
                service<debug::logger>() << neolib::logger::severity::Debug << "layout_item_cache::minimum_size(" << aAvailableSpace << ") (cache invalid)" << endl;
#endif // NEOGFX_DEBUG
            cachedMinSize = subject().minimum_size(aAvailableSpace);
            auto const ourSizePolicy = effective_size_policy();
            if (ourSizePolicy.maintain_aspect_ratio())
            {
                auto const& aspectRatio = ourSizePolicy.aspect_ratio();
                if (aspectRatio.cx < aspectRatio.cy)
                {
                    if (cachedMinSize.cx < cachedMinSize.cy)
                        cachedMinSize = size{ cachedMinSize.cx, cachedMinSize.cx * (aspectRatio.cy / aspectRatio.cx) };
                    else
                        cachedMinSize = size{ cachedMinSize.cy * (aspectRatio.cx / aspectRatio.cy), cachedMinSize.cy };
                }
                else
                {
                    if (cachedMinSize.cx < cachedMinSize.cy)
                        cachedMinSize = size{ cachedMinSize.cy * (aspectRatio.cx / aspectRatio.cy), cachedMinSize.cy };
                    else
                        cachedMinSize = size{ cachedMinSize.cx, cachedMinSize.cx * (aspectRatio.cy / aspectRatio.cx) };
                }
            }
            cachedMinSize = subject().apply_fixed_size(cachedMinSize);
            iMinimumSize.first = global_layout_id();
            iMinimumSize.second.first = aAvailableSpace;
        }
        auto const result = transformation() * cachedMinSize;
#ifdef NEOGFX_DEBUG
        if (&subject() == debug::layoutItem)
            service<debug::logger>() << neolib::logger::severity::Debug << "layout_item_cache::minimum_size(" << aAvailableSpace << ") -> " << cachedMinSize << " -> " << result << endl;
#endif // NEOGFX_DEBUG
        return to_units(subject(), su.saved_units(), result);
    }

    void layout_item_cache::set_minimum_size(optional_size const& aMinimumSize, bool aUpdateLayout)
    {
        subject().set_minimum_size(aMinimumSize, aUpdateLayout);
        if (aMinimumSize != std::nullopt)
            iMinimumSize.second.second = *aMinimumSize;
    }

    bool layout_item_cache::has_maximum_size() const noexcept
    {
        return subject().has_maximum_size();
    }

    bool layout_item_cache::is_maximum_size_constrained() const noexcept
    {
        return subject().is_maximum_size_constrained();
    }

    size layout_item_cache::maximum_size(optional_size const& aAvailableSpace) const
    {
#ifdef NEOGFX_DEBUG
        if (&subject() == debug::layoutItem)
            service<debug::logger>() << neolib::logger::severity::Debug << "layout_item_cache::maximum_size(" << aAvailableSpace << ")" << endl;
#endif // NEOGFX_DEBUG
        if (!visible())
            return size::max_size();
        scoped_units su{ subject(), units::Pixels };
        auto& cachedMaxSize = iMaximumSize.second.second;
        if (iMaximumSize.first != global_layout_id() || iMaximumSize.second.first != aAvailableSpace || is_maximum_size_constrained())
        {
#ifdef NEOGFX_DEBUG
            if (&subject() == debug::layoutItem)
                service<debug::logger>() << neolib::logger::severity::Debug << "layout_item_cache::maximum_size(" << aAvailableSpace << ") (cache invalid)" << endl;
#endif // NEOGFX_DEBUG
            cachedMaxSize = subject().apply_fixed_size(subject().maximum_size(aAvailableSpace));
            iMaximumSize.first = global_layout_id();
            iMaximumSize.second.first = aAvailableSpace;
        }
        auto const result = transformation() * cachedMaxSize;
#ifdef NEOGFX_DEBUG
        if (&subject() == debug::layoutItem)
            service<debug::logger>() << neolib::logger::severity::Debug << "layout_item_cache::maximum_size(" << aAvailableSpace << ") -> " << cachedMaxSize << " ->  " << result << endl;
#endif // NEOGFX_DEBUG
        return to_units(subject(), su.saved_units(), result);
    }

    void layout_item_cache::set_maximum_size(optional_size const& aMaximumSize, bool aUpdateLayout)
    {
        subject().set_maximum_size(aMaximumSize, aUpdateLayout);
        if (aMaximumSize != std::nullopt)
            iMaximumSize.second.second = *aMaximumSize;
    }

    bool layout_item_cache::has_fixed_size() const noexcept
    {
        return subject().has_fixed_size();
    }

    size layout_item_cache::fixed_size(optional_size const& aAvailableSpace) const
    {
#ifdef NEOGFX_DEBUG
        if (&subject() == debug::layoutItem)
            service<debug::logger>() << neolib::logger::severity::Debug << "layout_item_cache::fixed_size(" << aAvailableSpace << ")" << endl;
#endif // NEOGFX_DEBUG
        scoped_units su{ subject(), units::Pixels };
        auto& cachedFixedSize = iFixedSize.second.second;
        if (iFixedSize.first != global_layout_id() || iFixedSize.second.first != aAvailableSpace)
        {
#ifdef NEOGFX_DEBUG
            if (&subject() == debug::layoutItem)
                service<debug::logger>() << neolib::logger::severity::Debug << "layout_item_cache::fixed_size(" << aAvailableSpace << ") (cache invalid)" << endl;
#endif // NEOGFX_DEBUG
            cachedFixedSize = subject().fixed_size(aAvailableSpace);
            iFixedSize.first = global_layout_id();
            iFixedSize.second.first = aAvailableSpace;
        }
        auto const result = transformation() * cachedFixedSize;
#ifdef NEOGFX_DEBUG
        if (&subject() == debug::layoutItem)
            service<debug::logger>() << neolib::logger::severity::Debug << "layout_item_cache::fixed_size(" << aAvailableSpace << ") -> " << cachedFixedSize << " -> " << result << endl;
#endif // NEOGFX_DEBUG
        return to_units(subject(), su.saved_units(), result);
    }

    void layout_item_cache::set_fixed_size(optional_size const& aFixedSize, bool aUpdateLayout)
    {
        subject().set_fixed_size(aFixedSize, aUpdateLayout);
        if (aFixedSize != std::nullopt)
            iFixedSize.second.second = *aFixedSize;
    }

    bool layout_item_cache::has_transformation() const noexcept
    {
        return subject().has_transformation();
    }

    mat33 const& layout_item_cache::transformation(bool aCombineAncestorTransformations) const
    {
        auto& attribute = !aCombineAncestorTransformations ? iTransformation : iCombinedTransformation;
#ifdef NEOGFX_DEBUG
        if (&subject() == debug::layoutItem)
            service<debug::logger>() << neolib::logger::severity::Debug << "layout_item_cache::transformation(" << aCombineAncestorTransformations << ")" << endl;
#endif // NEOGFX_DEBUG
        auto& cachedTransformation = attribute.second;
        if (attribute.first != global_layout_id())
        {
#ifdef NEOGFX_DEBUG
            if (&subject() == debug::layoutItem)
                service<debug::logger>() << neolib::logger::severity::Debug << "layout_item_cache::transformation(" << aCombineAncestorTransformations << ") (cache invalid)" << endl;
#endif // NEOGFX_DEBUG
            cachedTransformation = subject().transformation(aCombineAncestorTransformations);
            attribute.first = global_layout_id();
        }
#ifdef NEOGFX_DEBUG
        if (&subject() == debug::layoutItem)
            service<debug::logger>() << neolib::logger::severity::Debug << "layout_item_cache::transformation(" << aCombineAncestorTransformations << ") -> " << cachedTransformation << endl;
#endif // transformation
        return cachedTransformation;
    }

    void layout_item_cache::set_transformation(optional_mat33 const& aTransformation, bool aUpdateLayout)
    {
        subject().set_transformation(aTransformation, aUpdateLayout);
        if (aTransformation != std::nullopt)
            iTransformation.second = *aTransformation;
        iCombinedTransformation.second = subject().transformation(true);
    }

    bool layout_item_cache::has_margin() const noexcept
    {
        return subject().has_margin();
    }

    margin layout_item_cache::margin() const
    {
        return subject().margin();
    }

    void layout_item_cache::set_margin(optional_margin const& aMargin, bool aUpdateLayout)
    {
        subject().set_margin(aMargin, aUpdateLayout);
    }

    bool layout_item_cache::has_border() const noexcept
    {
        return subject().has_border();
    }

    padding layout_item_cache::border() const
    {
        return subject().border();
    }

    void layout_item_cache::set_border(optional_border const& aBorder, bool aUpdateLayout)
    {
        subject().set_border(aBorder, aUpdateLayout);
    }

    bool layout_item_cache::has_padding() const noexcept
    {
        return subject().has_padding();
    }

    padding layout_item_cache::padding() const
    {
        return subject().padding();
    }

    void layout_item_cache::set_padding(optional_padding const& aPadding, bool aUpdateLayout)
    {
        subject().set_padding(aPadding, aUpdateLayout);
    }

    point layout_item_cache::unconstrained_origin() const
    {
        return subject().unconstrained_origin();
    }

    point layout_item_cache::unconstrained_position() const
    {
        return subject().unconstrained_position();
    }

    void layout_item_cache::layout_item_enabled(i_layout_item& aItem)
    {
        subject().layout_item_enabled(aItem);
    }

    void layout_item_cache::layout_item_disabled(i_layout_item& aItem)
    {
        subject().layout_item_disabled(aItem);
    }

    bool layout_item_cache::visible() const
    {
        auto& cachedVisible = iVisible.second;
        if (iVisible.first != global_layout_id())
        {
            cachedVisible =
                subject().visible() ||
                subject().effective_size_policy().ignore_visibility() ||
                (subject().has_parent_layout() && subject().parent_layout().ignore_child_visibility());
            iVisible.first = global_layout_id();
        }
        return cachedVisible;
    }

    bool layout_item_cache::operator==(const layout_item_cache& aOther) const
    {
        return iSubject == aOther.iSubject;
    }
}
