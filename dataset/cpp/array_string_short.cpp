/*************************************************************************
 *
 * Copyright 2016 Realm Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 **************************************************************************/

#include <cstdlib>
#include <algorithm>
#include <cstring>

#ifdef REALM_DEBUG
#include <cstdio>
#include <iostream>
#include <iomanip>
#endif

#include <realm/utilities.hpp>
#include <realm/array_string_short.hpp>
#include <realm/impl/destroy_guard.hpp>
#include <realm/column_integer.hpp>

using namespace realm;


namespace {

// Round up to nearest possible block length: 0, 1, 2, 4, 8, 16, 32, 64, 128, 256. We include 1 to store empty
// strings in as little space as possible, because 0 can only store nulls.
size_t round_up(size_t size)
{
    REALM_ASSERT(size <= 256);

    if (size <= 2)
        return size;

    size--;
    size |= size >> 1;
    size |= size >> 2;
    size |= size >> 4;
    ++size;
    return size;
}

} // anonymous namespace

bool ArrayStringShort::is_null(size_t ndx) const
{
    REALM_ASSERT_3(ndx, <, m_size);
    StringData sd = get(ndx);
    return sd.is_null();
}

void ArrayStringShort::set_null(size_t ndx)
{
    REALM_ASSERT_3(ndx, <, m_size);
    StringData sd = realm::null();
    set(ndx, sd);
}

void ArrayStringShort::set(size_t ndx, StringData value)
{
    REALM_ASSERT_3(ndx, <, m_size);
    REALM_ASSERT_3(value.size(), <, max_width); // otherwise we have to use another column type

    // if m_width == 0 and m_nullable == true, then entire array contains only null entries
    // if m_width == 0 and m_nullable == false, then entire array contains only "" entries
    if ((m_nullable ? value.is_null() : value.size() == 0) && m_width == 0) {
        return; // existing element in array already equals the value we want to set it to
    }

    // Make room for the new value plus a zero-termination
    if (m_width <= value.size()) {
        // Calc min column width
        size_t new_width = ::round_up(value.size() + 1);
        const size_t old_width = m_width;
        alloc(m_size, new_width); // Throws

        char* base = m_data;
        char* new_end = base + m_size * new_width;

        // Expand the old values in reverse order
        if (old_width > 0) {
            const char* old_end = base + m_size * old_width;
            while (new_end != base) {
                *--new_end = char(*--old_end + (new_width - old_width));
                {
                    // extend 0-padding
                    char* new_begin = new_end - (new_width - old_width);
                    std::fill(new_begin, new_end, 0);
                    new_end = new_begin;
                }
                {
                    // copy string payload
                    const char* old_begin = old_end - (old_width - 1);
                    if (static_cast<size_t>(old_end - old_begin) < old_width) // non-null string
                        new_end = std::copy_backward(old_begin, old_end, new_end);
                    old_end = old_begin;
                }
            }
        }
        else {
            // old_width == 0. Expand to new width.
            while (new_end != base) {
                REALM_ASSERT_3(new_width, <=, max_width);
                *--new_end = static_cast<char>(new_width);
                {
                    char* new_begin = new_end - (new_width - 1);
                    std::fill(new_begin, new_end, 0); // Fill with zero bytes
                    new_end = new_begin;
                }
            }
        }
    }
    else if (is_read_only()) {
        if (get(ndx) == value)
            return;
        copy_on_write();
    }

    REALM_ASSERT_3(0, <, m_width);

    // Set the value
    char* begin = m_data + (ndx * m_width);
    char* end = begin + (m_width - 1);
    begin = realm::safe_copy_n(value.data(), value.size(), begin);
    std::fill(begin, end, 0); // Pad with zero bytes
    static_assert(max_width <= max_width, "Padding size must fit in 7-bits");

    if (value.is_null()) {
        REALM_ASSERT_3(m_width, <=, 128);
        *end = static_cast<char>(m_width);
    }
    else {
        int pad_size = int(end - begin);
        *end = char(pad_size);
    }
}


void ArrayStringShort::insert(size_t ndx, StringData value)
{
    REALM_ASSERT_3(ndx, <=, m_size);
    REALM_ASSERT(value.size() < max_width); // otherwise we have to use another column type

    // FIXME: this performs up to 2 memcpy() operations. This could be improved
    // by making the allocator make a gap for the new value for us, but it's a
    // bit complex.

    // Allocate room for the new value
    const auto old_size = m_size;
    alloc(m_size + 1, m_width); // Throws

    // Make gap for new value
    memmove(m_data + m_width * (ndx + 1), m_data + m_width * ndx, m_width * (old_size - ndx));

    // Set new value
    set(ndx, value);
    return;
}

void ArrayStringShort::erase(size_t ndx)
{
    REALM_ASSERT_3(ndx, <, m_size);

    // Check if we need to copy before modifying
    copy_on_write(); // Throws

    // move data backwards after deletion
    if (ndx < m_size - 1) {
        char* new_begin = m_data + ndx * m_width;
        char* old_begin = new_begin + m_width;
        char* old_end = m_data + m_size * m_width;
        realm::safe_copy_n(old_begin, old_end - old_begin, new_begin);
    }

    --m_size;

    // Update size in header
    set_header_size(m_size);
}

size_t ArrayStringShort::calc_byte_len(size_t num_items, size_t width) const
{
    return header_size + (num_items * width);
}

size_t ArrayStringShort::calc_item_count(size_t bytes, size_t width) const noexcept
{
    if (width == 0)
        return size_t(-1); // zero-width gives infinite space

    size_t bytes_without_header = bytes - header_size;
    return bytes_without_header / width;
}

size_t ArrayStringShort::count(StringData value, size_t begin, size_t end) const noexcept
{
    size_t num_matches = 0;

    size_t begin_2 = begin;
    for (;;) {
        size_t ndx = find_first(value, begin_2, end);
        if (ndx == not_found)
            break;
        ++num_matches;
        begin_2 = ndx + 1;
    }

    return num_matches;
}

size_t ArrayStringShort::find_first(StringData value, size_t begin, size_t end) const noexcept
{
    if (end == size_t(-1))
        end = m_size;
    REALM_ASSERT(begin <= m_size && end <= m_size && begin <= end);

    if (m_width == 0) {
        if (m_nullable)
            // m_width == 0 implies that all elements in the array are NULL
            return value.is_null() && begin < m_size ? begin : npos;
        else
            return value.size() == 0 && begin < m_size ? begin : npos;
    }

    const size_t value_size = value.size();
    // A string can never be wider than the column width
    if (m_width <= value_size)
        return size_t(-1);

    if (m_nullable ? value.is_null() : value_size == 0) {
        for (size_t i = begin; i != end; ++i) {
            if (m_nullable ? is_null(i) : get(i).size() == 0)
                return i;
        }
    }
    else if (value_size == 0) {
        const char* data = m_data + (m_width - 1);
        for (size_t i = begin; i != end; ++i) {
            size_t data_i_size = (m_width - 1) - data[i * m_width];
            // left-hand-side tests if array element is NULL
            if (REALM_UNLIKELY(data_i_size == 0))
                return i;
        }
    }
    else {
        for (size_t i = begin; i != end; ++i) {
            const char* data = m_data + (i * m_width);
            if (memcmp(data, value.data(), value_size) == 0) {
                size_t data_size = (m_width - 1) - data[m_width - 1];
                if (data_size == value_size) {
                    return i;
                }
            }
        }
    }

    return not_found;
}

void ArrayStringShort::find_all(IntegerColumn& result, StringData value, size_t add_offset, size_t begin, size_t end)
{
    size_t begin_2 = begin;
    for (;;) {
        size_t ndx = find_first(value, begin_2, end);
        if (ndx == not_found)
            break;
        result.add(add_offset + ndx); // Throws
        begin_2 = ndx + 1;
    }
}

bool ArrayStringShort::compare_string(const ArrayStringShort& c) const noexcept
{
    if (c.size() != size())
        return false;

    for (size_t i = 0; i < size(); ++i) {
        if (get(i) != c.get(i))
            return false;
    }

    return true;
}

#ifdef REALM_DEBUG // LCOV_EXCL_START ignore debug functions

void ArrayStringShort::string_stats() const
{
    size_t total = 0;
    size_t longest = 0;

    for (size_t i = 0; i < m_size; ++i) {
        StringData str = get(i);
        size_t str_size = str.size() + 1;
        total += str_size;
        if (str_size > longest)
            longest = str_size;
    }

    size_t array_size = m_size * m_width;
    size_t zeroes = array_size - total;
    size_t zavg = zeroes / (m_size ? m_size : 1); // avoid possible div by zero

    std::cout << "Size: " << m_size << "\n";
    std::cout << "Width: " << m_width << "\n";
    std::cout << "Total: " << array_size << "\n";
    // std::cout << "Capacity: " << m_capacity << "\n\n";
    std::cout << "Bytes string: " << total << "\n";
    std::cout << "     longest: " << longest << "\n";
    std::cout << "Bytes zeroes: " << zeroes << "\n";
    std::cout << "         avg: " << zavg << "\n";
}

#endif // LCOV_EXCL_STOP ignore debug functions
