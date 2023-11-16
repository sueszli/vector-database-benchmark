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

#include <realm/array_binary.hpp>
#include <realm/mixed.hpp>

using namespace realm;

ArrayBinary::ArrayBinary(Allocator& a)
    : m_alloc(a)
{
    m_arr = new (&m_storage.m_small_blobs) ArraySmallBlobs(a);
}

void ArrayBinary::create()
{
    static_cast<ArraySmallBlobs*>(m_arr)->create();
}

void ArrayBinary::init_from_mem(MemRef mem) noexcept
{
    char* header = mem.get_addr();

    ArrayParent* parent = m_arr->get_parent();
    size_t ndx_in_parent = m_arr->get_ndx_in_parent();

    m_is_big = Array::get_context_flag_from_header(header);
    if (!m_is_big) {
        auto arr = new (&m_storage.m_small_blobs) ArraySmallBlobs(m_alloc);
        arr->init_from_mem(mem);
    }
    else {
        auto arr = new (&m_storage.m_big_blobs) ArrayBigBlobs(m_alloc, true);
        arr->init_from_mem(mem);
    }

    m_arr->set_parent(parent, ndx_in_parent);
}


void ArrayBinary::init_from_parent()
{
    ref_type ref = m_arr->get_ref_from_parent();
    init_from_ref(ref);
}

size_t ArrayBinary::size() const
{
    if (!m_is_big) {
        return static_cast<ArraySmallBlobs*>(m_arr)->size();
    }
    else {
        return static_cast<ArrayBigBlobs*>(m_arr)->size();
    }
}

void ArrayBinary::add(BinaryData value)
{
    bool is_big = upgrade_leaf(value.size());
    if (!is_big) {
        static_cast<ArraySmallBlobs*>(m_arr)->add(value);
    }
    else {
        static_cast<ArrayBigBlobs*>(m_arr)->add(value);
    }
}

void ArrayBinary::set(size_t ndx, BinaryData value)
{
    bool is_big = upgrade_leaf(value.size());
    if (!is_big) {
        static_cast<ArraySmallBlobs*>(m_arr)->set(ndx, value);
    }
    else {
        static_cast<ArrayBigBlobs*>(m_arr)->set(ndx, value);
    }
}

void ArrayBinary::insert(size_t ndx, BinaryData value)
{
    bool is_big = upgrade_leaf(value.size());
    if (!is_big) {
        static_cast<ArraySmallBlobs*>(m_arr)->insert(ndx, value);
    }
    else {
        static_cast<ArrayBigBlobs*>(m_arr)->insert(ndx, value);
    }
}

BinaryData ArrayBinary::get(size_t ndx) const
{
    if (!m_is_big) {
        return static_cast<ArraySmallBlobs*>(m_arr)->get(ndx);
    }
    else {
        return static_cast<ArrayBigBlobs*>(m_arr)->get(ndx);
    }
}

BinaryData ArrayBinary::get_at(size_t ndx, size_t& pos) const
{
    if (!m_is_big) {
        pos = 0;
        return static_cast<ArraySmallBlobs*>(m_arr)->get(ndx);
    }
    else {
        return static_cast<ArrayBigBlobs*>(m_arr)->get_at(ndx, pos);
    }
}

Mixed ArrayBinary::get_any(size_t ndx) const
{
    return Mixed(get(ndx));
}

bool ArrayBinary::is_null(size_t ndx) const
{
    if (!m_is_big) {
        return static_cast<ArraySmallBlobs*>(m_arr)->is_null(ndx);
    }
    else {
        return static_cast<ArrayBigBlobs*>(m_arr)->is_null(ndx);
    }
}

void ArrayBinary::erase(size_t ndx)
{
    if (!m_is_big) {
        return static_cast<ArraySmallBlobs*>(m_arr)->erase(ndx);
    }
    else {
        return static_cast<ArrayBigBlobs*>(m_arr)->erase(ndx);
    }
}

void ArrayBinary::move(ArrayBinary& dst, size_t ndx)
{
    size_t sz = size();
    for (size_t i = ndx; i < sz; i++) {
        dst.add(get(i));
    }

    if (!m_is_big) {
        return static_cast<ArraySmallBlobs*>(m_arr)->truncate(ndx);
    }
    else {
        return static_cast<ArrayBigBlobs*>(m_arr)->truncate(ndx);
    }
}

void ArrayBinary::clear()
{
    if (!m_is_big) {
        return static_cast<ArraySmallBlobs*>(m_arr)->clear();
    }
    else {
        return static_cast<ArrayBigBlobs*>(m_arr)->clear();
    }
}

size_t ArrayBinary::find_first(BinaryData value, size_t begin, size_t end) const noexcept
{
    if (!m_is_big) {
        return static_cast<ArraySmallBlobs*>(m_arr)->find_first(value, false, begin, end);
    }
    else {
        return static_cast<ArrayBigBlobs*>(m_arr)->find_first(value, false, begin, end);
    }
}


bool ArrayBinary::upgrade_leaf(size_t value_size)
{
    if (m_is_big)
        return true;

    if (value_size <= small_blob_max_size)
        return false;

    // Upgrade root leaf from small to big blobs
    auto small_blobs = static_cast<ArraySmallBlobs*>(m_arr);
    ArrayBigBlobs big_blobs(m_alloc, true);
    big_blobs.create(); // Throws

    size_t n = small_blobs->size();
    for (size_t i = 0; i < n; i++) {
        big_blobs.add(small_blobs->get(i)); // Throws
    }
    auto parent = small_blobs->get_parent();
    auto ndx_in_parent = small_blobs->get_ndx_in_parent();
    small_blobs->destroy();

    auto arr = new (&m_storage.m_big_blobs) ArrayBigBlobs(m_alloc, true);
    arr->init_from_mem(big_blobs.get_mem());
    arr->set_parent(parent, ndx_in_parent);
    arr->update_parent(); // Throws

    m_is_big = true;
    return true;
}

void ArrayBinary::verify() const
{
#ifdef REALM_DEBUG
    if (!m_is_big) {
        static_cast<ArraySmallBlobs*>(m_arr)->verify();
    }
    else {
        static_cast<ArrayBigBlobs*>(m_arr)->verify();
    }
#endif
}
