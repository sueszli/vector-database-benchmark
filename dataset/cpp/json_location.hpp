﻿// Copyright 2013-2023 Daniel Parker
// Distributed under the Boost license, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// See https://github.com/danielaparker/jsoncons for latest version

#ifndef JSONCONS_JSONPATH_JSON_LOCATION_HPP
#define JSONCONS_JSONPATH_JSON_LOCATION_HPP

#include <string>
#include <vector>
#include <functional>
#include <algorithm> // std::reverse
#include <jsoncons/config/jsoncons_config.hpp>
#include <jsoncons/detail/write_number.hpp>
#include <jsoncons/json_type.hpp>
#include <jsoncons/extension_traits.hpp>

namespace jsoncons { 
namespace jsonpath {

    enum class path_element_kind { name, index };

    template <class CharT,class Allocator>
    class basic_path_element 
    {
    public:
        using char_type = CharT;
        using allocator_type = Allocator;
        using char_allocator_type = typename std::allocator_traits<allocator_type>:: template rebind_alloc<CharT>;
        using string_type = std::basic_string<char_type,std::char_traits<char_type>,char_allocator_type>;
    private:
        path_element_kind element_kind_;
        string_type name_;
        std::size_t index_;

    public:
        basic_path_element(const string_type& name)
            : element_kind_(path_element_kind::name), name_(name), index_(0)
        {
        }

        basic_path_element(string_type&& name)
            : element_kind_(path_element_kind::name), name_(std::move(name)), index_(0)
        {
        }

        basic_path_element(std::size_t index)
            : element_kind_(path_element_kind::index), index_(index)
        {
        }

        basic_path_element(const basic_path_element& other) = default;

        basic_path_element& operator=(const basic_path_element& other) = default;

        path_element_kind element_kind() const
        {
            return element_kind_;
        }

        const string_type& name() const
        {
            return name_;
        }

        std::size_t index() const 
        {
            return index_;
        }

        std::size_t node_hash() const
        {
            std::size_t h = element_kind_ == path_element_kind::index ? std::hash<std::size_t>{}(index_) : std::hash<string_type>{}(name_);

            return h;
        }

        int compare(const basic_path_element& other) const
        {
            int diff = 0;
            if (element_kind_ != other.element_kind_)
            {
                diff = static_cast<int>(element_kind_) - static_cast<int>(other.element_kind_);
            }
            else
            {
                switch (element_kind_)
                {
                    case path_element_kind::name:
                        diff = name_.compare(other.name_);
                        break;
                    case path_element_kind::index:
                        diff = index_ < other.index_ ? -1 : index_ > other.index_ ? 1 : 0;
                        break;
                    default:
                        break;
                }
            }
            return diff;
        }
    };

    template <class CharT, class Allocator = std::allocator<char>>
    class basic_json_location
    {
    public:
        using char_type = CharT;
        using allocator_type = Allocator;
        using char_allocator_type = typename std::allocator_traits<allocator_type>:: template rebind_alloc<char_type>;
        using string_type = std::basic_string<char_type,std::char_traits<char_type>,char_allocator_type>;
        using string_view_type = jsoncons::basic_string_view<char_type, std::char_traits<char_type>>;
        using path_element_type = basic_path_element<CharT,Allocator>;
        using path_element_allocator_type = typename std::allocator_traits<allocator_type>:: template rebind_alloc<path_element_type>;
    private:
        allocator_type alloc_;
        std::vector<path_element_type> elements_;
    public:
        using iterator = typename std::vector<path_element_type>::iterator;
        using const_iterator = typename std::vector<path_element_type>::const_iterator;

        basic_json_location(const allocator_type& alloc=Allocator())
            : alloc_(alloc), elements_(alloc)
        {
        }

        iterator begin()
        {
            return elements_.begin();
        }

        iterator end()
        {
            return elements_.end();
        }

        const_iterator begin() const
        {
            return elements_.begin();
        }

        const_iterator end() const
        {
            return elements_.end();
        }

        std::size_t size() const
        {
            return elements_.size();
        }

        const basic_path_element<char_type,allocator_type>& operator[](std::size_t index) const
        {
            return elements_[index];
        }

        int compare(const basic_json_location& other) const
        {
            if (this == &other)
            {
               return 0;
            }

            auto it1 = elements_.begin();
            auto it2 = other.elements_.begin();
            while (it1 != elements_.end() && it2 != other.elements_.end())
            {
                int diff = it1->compare(*it2);
                if (diff != 0)
                {
                    return diff;
                }
                ++it1;
                ++it2;
            }
            return (elements_.size() < other.elements_.size()) ? -1 : (elements_.size() == other.elements_.size()) ? 0 : 1;
        }

        std::size_t hash() const
        {

            auto it = elements_.begin();
            std::size_t hash = (*it).hash();
            ++it;

            while (it != elements_.end())
            {
                hash += 17*(*it)->node_hash();
                ++it;
            }

            return hash;
        }

        // Modifiers

        void clear()
        {
            elements_.clear();
        }

        basic_json_location& operator/=(const string_view_type& s)
        {
            elements_.emplace_back(string_type(s.data(), s.size(), alloc_));
            return *this;
        }

        template <class IntegerType>
        typename std::enable_if<extension_traits::is_integer<IntegerType>::value, basic_json_location&>::type
            operator/=(IntegerType val)
        {
            elements_.emplace_back(static_cast<std::size_t>(val));

            return *this;
        }

        friend bool operator==(const basic_json_location& lhs, const basic_json_location& rhs) 
        {
            return lhs.compare(rhs) == 0;
        }

        friend bool operator!=(const basic_json_location& lhs, const basic_json_location& rhs)
        {
            return !(lhs == rhs);
        }

        friend bool operator<(const basic_json_location& lhs, const basic_json_location& rhs) 
        {
            return lhs.compare(rhs) < 0;
        }
    };

    using json_location = basic_json_location<char>;
    using wjson_location = basic_json_location<wchar_t>;
    using path_element = basic_path_element<char,std::allocator<char>>;
    using wpath_element = basic_path_element<wchar_t,std::allocator<char>>;

} // namespace jsonpath
} // namespace jsoncons

#endif
