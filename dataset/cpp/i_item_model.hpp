// i_item_model.hpp
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
#include <boost/format.hpp>
#include <neolib/core/vector.hpp>
#include <neolib/core/pair.hpp>
#include <neolib/core/optional.hpp>
#include <neolib/core/variant.hpp>
#include <neolib/core/generic_iterator.hpp>
#include <neogfx/core/i_property.hpp>
#include <neogfx/gui/widget/item_index.hpp>

namespace neogfx
{
    class i_item_model;

    struct item_model_index : item_index<item_model_index> { using item_index::item_index; };
    typedef std::optional<item_model_index> optional_item_model_index;

    enum class item_data_type : uint32_t
    {
        Unknown,
        Bool,
        Int32,
        UInt32,
        Int64,
        UInt64,
        Float,
        Double,
        String,
        Pointer,
        CustomType,
        ChoiceBool,
        ChoiceInt32,
        ChoiceUInt32,
        ChoiceInt64,
        ChoiceUInt16,
        ChoiceFloat,
        ChoiceDouble,
        ChoiceString,
        ChoicePointer,
        ChoiceCustomType
    };

    enum class item_cell_data_category : uint32_t
    {
        Invalid,
        Value,
        Pointer,
        CustomType,
        ChooseValue,
        ChoosePointer,
        ChooseCustomType
    };
}

begin_declare_enum(neogfx::item_data_type)
declare_enum_string(neogfx::item_data_type, Unknown)
declare_enum_string(neogfx::item_data_type, Bool)
declare_enum_string(neogfx::item_data_type, Int32)
declare_enum_string(neogfx::item_data_type, UInt32)
declare_enum_string(neogfx::item_data_type, Int64)
declare_enum_string(neogfx::item_data_type, UInt64)
declare_enum_string(neogfx::item_data_type, Float)
declare_enum_string(neogfx::item_data_type, Double)
declare_enum_string(neogfx::item_data_type, String)
declare_enum_string(neogfx::item_data_type, Pointer)
declare_enum_string(neogfx::item_data_type, CustomType)
declare_enum_string(neogfx::item_data_type, ChoiceBool)
declare_enum_string(neogfx::item_data_type, ChoiceInt32)
declare_enum_string(neogfx::item_data_type, ChoiceUInt32)
declare_enum_string(neogfx::item_data_type, ChoiceInt64)
declare_enum_string(neogfx::item_data_type, ChoiceUInt16)
declare_enum_string(neogfx::item_data_type, ChoiceFloat)
declare_enum_string(neogfx::item_data_type, ChoiceDouble)
declare_enum_string(neogfx::item_data_type, ChoiceString)
declare_enum_string(neogfx::item_data_type, ChoicePointer)
declare_enum_string(neogfx::item_data_type, ChoiceCustomType)
end_declare_enum(neogfx::item_data_type)

begin_declare_enum(neogfx::item_cell_data_category)
declare_enum_string(neogfx::item_cell_data_category, Invalid)
declare_enum_string(neogfx::item_cell_data_category, Value)
declare_enum_string(neogfx::item_cell_data_category, Pointer)
declare_enum_string(neogfx::item_cell_data_category, CustomType)
declare_enum_string(neogfx::item_cell_data_category, ChooseValue)
declare_enum_string(neogfx::item_cell_data_category, ChoosePointer)
declare_enum_string(neogfx::item_cell_data_category, ChooseCustomType)
end_declare_enum(neogfx::item_cell_data_category)

namespace neogfx
{
    template <typename T>
    struct item_cell_choice_type
    {
        typedef T value_type;
        typedef neolib::pair<value_type, string> option;
        typedef neolib::vector<option> type;
    };

    typedef neolib::variant<
        bool,
        int32_t,
        uint32_t,
        int64_t,
        uint64_t,
        float,
        double,
        string,
        void*,
        custom_type,
        item_cell_choice_type<bool>::type::value_type const*,
        item_cell_choice_type<int32_t>::type::value_type const*,
        item_cell_choice_type<uint32_t>::type::value_type const*,
        item_cell_choice_type<int64_t>::type::value_type const*,
        item_cell_choice_type<uint64_t>::type::value_type const*,
        item_cell_choice_type<float>::type::value_type const*,
        item_cell_choice_type<double>::type::value_type const*,
        item_cell_choice_type<string>::type::value_type const*,
        item_cell_choice_type<void*>::type::value_type const*,
        item_cell_choice_type<custom_type>::type::value_type const*> item_cell_data_variant;

    template <typename T> struct classify_item_call_data { static constexpr item_cell_data_category category = item_cell_data_category::Invalid; };
    template <> struct classify_item_call_data<bool> { static constexpr item_cell_data_category category = item_cell_data_category::Value; };
    template <> struct classify_item_call_data<int32_t> { static constexpr item_cell_data_category category = item_cell_data_category::Value; };
    template <> struct classify_item_call_data<uint32_t> { static constexpr item_cell_data_category category = item_cell_data_category::Value; };
    template <> struct classify_item_call_data<int64_t> { static constexpr item_cell_data_category category = item_cell_data_category::Value; };
    template <> struct classify_item_call_data<uint64_t> { static constexpr item_cell_data_category category = item_cell_data_category::Value; };
    template <> struct classify_item_call_data<float> { static constexpr item_cell_data_category category = item_cell_data_category::Value; };
    template <> struct classify_item_call_data<double> { static constexpr item_cell_data_category category = item_cell_data_category::Value; };
    template <> struct classify_item_call_data<string> { static constexpr item_cell_data_category category = item_cell_data_category::Value; };
    template <> struct classify_item_call_data<void*> { static constexpr item_cell_data_category category = item_cell_data_category::Pointer; };
    template <> struct classify_item_call_data<custom_type> { static constexpr item_cell_data_category category = item_cell_data_category::CustomType; };
    template <> struct classify_item_call_data<item_cell_choice_type<bool>::type::value_type const*> { static constexpr item_cell_data_category category = item_cell_data_category::ChooseValue; };
    template <> struct classify_item_call_data<item_cell_choice_type<int32_t>::type::value_type const*> { static constexpr item_cell_data_category category = item_cell_data_category::ChooseValue; };
    template <> struct classify_item_call_data<item_cell_choice_type<uint32_t>::type::value_type const*> { static constexpr item_cell_data_category category = item_cell_data_category::ChooseValue; };
    template <> struct classify_item_call_data<item_cell_choice_type<int64_t>::type::value_type const*> { static constexpr item_cell_data_category category = item_cell_data_category::ChooseValue; };
    template <> struct classify_item_call_data<item_cell_choice_type<uint64_t>::type::value_type const*> { static constexpr item_cell_data_category category = item_cell_data_category::ChooseValue; };
    template <> struct classify_item_call_data<item_cell_choice_type<float>::type::value_type const*> { static constexpr item_cell_data_category category = item_cell_data_category::ChooseValue; };
    template <> struct classify_item_call_data<item_cell_choice_type<double>::type::value_type const*> { static constexpr item_cell_data_category category = item_cell_data_category::ChooseValue; };
    template <> struct classify_item_call_data<item_cell_choice_type<string>::type::value_type const*> { static constexpr item_cell_data_category category = item_cell_data_category::ChooseValue; };
    template <> struct classify_item_call_data<item_cell_choice_type<void*>::type::value_type const*> { static constexpr item_cell_data_category category = item_cell_data_category::ChoosePointer; };
    template <> struct classify_item_call_data<item_cell_choice_type<custom_type>::type::value_type const*> { static constexpr item_cell_data_category category = item_cell_data_category::ChooseCustomType; };

    class item_cell_data : public item_cell_data_variant
    {
    public:
        item_cell_data() 
        {
        }
        template <typename T, typename std::enable_if_t<is_alternative_v<T>, int> = 0>
        item_cell_data(T const& aValue) :
            item_cell_data_variant{ aValue }
        {
        }
        template <typename T, typename std::enable_if_t<is_alternative_v<T>, int> = 0>
        item_cell_data(T&& aValue) : 
            item_cell_data_variant{ std::forward<T>(aValue) }
        {
        }
        item_cell_data(item_cell_data const& aOther) :
            item_cell_data_variant{ static_cast<item_cell_data_variant const&>(aOther) } 
        {
        }
        item_cell_data(item_cell_data&& aOther) : 
            item_cell_data_variant{ static_cast<item_cell_data_variant&&>(std::move(aOther)) } 
        {
        }
    public:
        item_cell_data(char const* aString) : 
            item_cell_data_variant{ string{ aString } } 
        {
        }
        item_cell_data(std::string const& aString) :
            item_cell_data_variant{ string{ aString } }
        {
        }
        item_cell_data(std::string&& aString) :
            item_cell_data_variant{ string{ std::move(aString) } }
        {
        }
    public:
        item_cell_data& operator=(item_cell_data const& aOther)
        {
            item_cell_data_variant::operator=(static_cast<item_cell_data_variant const&>(aOther));
            return *this;
        }
        item_cell_data& operator=(item_cell_data&& aOther)
        {
            item_cell_data_variant::operator=(static_cast<item_cell_data_variant&&>(std::move(aOther)));
            return *this;
        }
        template <typename T, typename std::enable_if_t<is_alternative_v<T>, int> = 0>
        item_cell_data& operator=(T const& aArgument)
        {
            item_cell_data_variant::operator=(aArgument);
            return *this;
        }
        template <typename T, typename std::enable_if_t<is_alternative_v<T>, int> = 0>
        item_cell_data& operator=(T&& aArgument)
        {
            item_cell_data_variant::operator=(std::forward<T>(aArgument));
            return *this;
        }
    public:
        item_cell_data& operator=(char const* aString)
        {
            item_cell_data_variant::operator=(string{ aString });
            return *this;
        }
        item_cell_data& operator=(std::string const& aString)
        {
            item_cell_data_variant::operator=(string{ aString });
            return *this;
        }
        item_cell_data& operator=(std::string&& aString)
        {
            item_cell_data_variant::operator=(string{ aString });
            return *this;
        }
    public:
        std::string to_string() const
        {
            return std::visit([](auto&& arg) -> std::string
            {
                typedef typename std::remove_cv<typename std::remove_reference<decltype(arg)>::type>::type type;
                if constexpr(!std::is_same_v<type, std::monostate> && classify_item_call_data<type>::category == item_cell_data_category::Value)
                    return (boost::basic_format<char>{"%1%"} % arg).str();
                else
                    return "";
            }, *this);
        }
    };

    struct item_cell_info
    {
        item_data_type dataType;
        item_cell_data dataMin;
        item_cell_data dataMax;
        item_cell_data dataStep;
    };

    typedef std::optional<item_cell_info> optional_item_cell_info;

    class i_item_model : public i_reference_counted, public i_property_owner
    {
    public:
        declare_event(column_info_changed, item_model_index::column_type)
        declare_event(item_added, item_model_index const&)
        declare_event(item_changed, item_model_index const&)
        declare_event(item_removing, item_model_index const&)
        declare_event(item_removed, item_model_index const&)
        declare_event(cleared)
    public:
        typedef i_item_model abstract_type;
        typedef neolib::generic_iterator iterator;
        typedef neolib::generic_iterator const_iterator;
    public:
        struct wrong_model_type : std::logic_error { wrong_model_type() : std::logic_error("neogfx::i_item_model::wrong_model_type") {} };
        struct bad_column_index : std::logic_error { bad_column_index() : std::logic_error("neogfx::i_item_model::bad_column_index") {} };
        struct item_not_found : std::logic_error { item_not_found() : std::logic_error("neogfx::i_item_model::item_not_found") {} };
    public:
        virtual ~i_item_model() = default;
    public:
        virtual bool is_tree() const = 0;
        virtual uint32_t rows() const = 0;
        virtual uint32_t columns() const = 0;
        virtual uint32_t columns(item_model_index const& aIndex) const = 0;
        virtual std::string const& column_name(item_model_index::column_type aColumnIndex) const = 0;
        virtual void set_column_name(item_model_index::column_type aColumnIndex, std::string const& aName) = 0;
        virtual item_data_type column_data_type(item_model_index::column_type aColumnIndex) const = 0;
        virtual void set_column_data_type(item_model_index::column_type aColumnIndex, item_data_type aType) = 0;
        virtual item_cell_data const& column_min_value(item_model_index::column_type aColumnIndex) const = 0;
        virtual void set_column_min_value(item_model_index::column_type aColumnIndex, item_cell_data const& aValue) = 0;
        virtual item_cell_data const& column_max_value(item_model_index::column_type aColumnIndex) const = 0;
        virtual void set_column_max_value(item_model_index::column_type aColumnIndex, item_cell_data const& aValue) = 0;
        virtual item_cell_data const& column_step_value(item_model_index::column_type aColumnIndex) const = 0;
        virtual void set_column_step_value(item_model_index::column_type aColumnIndex, item_cell_data const& aValue) = 0;
    public:
        virtual iterator index_to_iterator(item_model_index const& aIndex) = 0;
        virtual const_iterator index_to_iterator(item_model_index const& aIndex) const = 0;
        virtual item_model_index iterator_to_index(const_iterator aPosition) const = 0;
        virtual iterator begin() = 0;
        virtual const_iterator begin() const = 0;
        virtual iterator end() = 0;
        virtual const_iterator end() const = 0;
        virtual iterator sbegin() = 0;
        virtual const_iterator sbegin() const = 0;
        virtual iterator send() = 0;
        virtual const_iterator send() const = 0;
        virtual bool has_children(const_iterator aParent) const = 0;
        virtual bool has_children(item_model_index const& aParentIndex) const = 0;
        virtual bool has_parent(const_iterator aChild) const = 0;
        virtual bool has_parent(item_model_index const& aChildIndex) const = 0;
        virtual iterator parent(const_iterator aChild) = 0;
        virtual const_iterator parent(const_iterator aChild) const = 0;
        virtual item_model_index parent(item_model_index const& aChildIndex) const = 0;
        virtual iterator sbegin(const_iterator aParent) = 0;
        virtual const_iterator sbegin(const_iterator aParent) const = 0;
        virtual iterator send(const_iterator aParent) = 0;
        virtual const_iterator send(const_iterator aParent) const = 0;
    public:
        virtual bool empty() const = 0;
        virtual void reserve(uint32_t aItemCount) = 0;
        virtual uint32_t capacity() const = 0;
        virtual iterator insert_item(const_iterator aPosition, item_cell_data const& aCellData) = 0;
        virtual iterator insert_item(item_model_index const& aIndex, item_cell_data const& aCellData) = 0;
        virtual iterator append_item(const_iterator aParent, item_cell_data const& aCellData) = 0;
        virtual iterator append_item(item_model_index const& aIndex, item_cell_data const& aCellData) = 0;
        virtual void clear() = 0;
        virtual iterator erase(const_iterator aPosition) = 0;
        virtual iterator erase(item_model_index const& aIndex) = 0;
        virtual void insert_cell_data(const_iterator aPosition, item_model_index::column_type aColumnIndex, item_cell_data const& aCellData) = 0;
        virtual void insert_cell_data(item_model_index const& aIndex, item_cell_data const& aCellData) = 0;
        virtual void update_cell_data(const_iterator aPosition, item_model_index::column_type aColumnIndex, item_cell_data const& aCellData) = 0;
        virtual void update_cell_data(item_model_index const& aIndex, item_cell_data const& aCellData) = 0;
    public:
        virtual const item_cell_info& cell_info(item_model_index const& aIndex) const = 0;
        virtual item_cell_data const& cell_data(item_model_index const& aIndex) const = 0;
    };
}