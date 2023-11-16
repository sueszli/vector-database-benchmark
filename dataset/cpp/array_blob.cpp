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

#include <algorithm>

#include <realm/array.hpp>
#include <realm/array_blob.hpp>

using namespace realm;

BinaryData ArrayBlob::get_at(size_t& pos) const noexcept
{
    size_t offset = pos;
    if (get_context_flag()) {
        size_t ndx = 0;
        size_t current_size = Array::get_size_from_header(m_alloc.translate(Array::get_as_ref(ndx)));

        // Find the blob to start from
        while (offset >= current_size) {
            ndx++;
            if (ndx >= size()) {
                pos = 0;
                return {"", 0};
            }
            offset -= current_size;
            current_size = Array::get_size_from_header(m_alloc.translate(Array::get_as_ref(ndx)));
        }

        ArrayBlob blob(m_alloc);
        blob.init_from_ref(Array::get_as_ref(ndx));
        ndx++;
        size_t sz = current_size - offset;

        // Check if we are at last blob
        pos = (ndx >= size()) ? 0 : pos + sz;

        return {blob.get(offset), sz};
    }
    else {
        // All data is in this array
        pos = 0;
        if (offset < size()) {
            return {get(offset), size() - offset};
        }
        else {
            return {"", 0};
        }
    }
}

ref_type Array::blob_replace(size_t begin, size_t end, const char* data, size_t data_size, bool add_zero_term)
{
    // The context flag indicates if the array contains references to blobs
    // holding the actual data.
    REALM_ASSERT(get_context_flag());
    size_t sz = blob_size();
    REALM_ASSERT_3(begin, <=, end);
    REALM_ASSERT_3(end, <=, sz);
    REALM_ASSERT(data_size == 0 || data);

    REALM_ASSERT((begin == 0 || begin == sz) && end == sz); // Only append or total replace supported
    if (begin == sz && end == sz) {
        // Append
        // We might have room for more data in the last node
        ArrayBlob lastNode(m_alloc);
        lastNode.init_from_ref(get_as_ref(size() - 1));
        lastNode.set_parent(this, size() - 1);

        size_t space_left = ArrayBlob::max_binary_size - lastNode.size();
        size_t size_to_copy = std::min(space_left, data_size);
        lastNode.add(data, size_to_copy);
        data_size -= space_left;
        data += space_left;

        while (data_size) {
            // Create new nodes as required
            size_to_copy = std::min(size_t(ArrayBlob::max_binary_size), data_size);
            ArrayBlob new_blob(m_alloc);
            new_blob.create(); // Throws

            // Copy data
            ref_type ref = new_blob.add(data, size_to_copy);
            // Add new node in hosting node
            add(ref);

            data_size -= size_to_copy;
            data += size_to_copy;
        }
        return get_ref();
    }
    else if (begin == 0 && end == sz) {
        // Replace all. Start from scratch
        destroy_deep();
        ArrayBlob new_blob(m_alloc);
        new_blob.create();                                   // Throws
        return new_blob.add(data, data_size, add_zero_term); // Throws
    }

    REALM_UNREACHABLE();
    return 0;
}

ref_type ArrayBlob::replace(size_t begin, size_t end, const char* data, size_t data_size, bool add_zero_term)
{
    size_t sz = blob_size();
    REALM_ASSERT_3(begin, <=, end);
    REALM_ASSERT_3(end, <=, sz);
    REALM_ASSERT(data_size == 0 || data);

    REALM_ASSERT(!get_context_flag());
    size_t remove_size = end - begin;
    size_t add_size = add_zero_term ? data_size + 1 : data_size;
    size_t old_size = m_size;
    size_t new_size = m_size - remove_size + add_size;

    // If size of BinaryData is below 'max_binary_size', the data is stored directly
    // in a single ArrayBlob. If more space is needed, the root blob will just contain
    // references to child blobs holding the actual data. Context flag will indicate
    // if blob is split.
    if (new_size > max_binary_size) {
        Array new_root(m_alloc);
        // Create new array with context flag set
        new_root.create(type_HasRefs, true); // Throws

        // Add current node to the new root
        new_root.add(get_ref());
        return new_root.blob_replace(begin, end, data, data_size, add_zero_term);
    }

    if (remove_size == add_size && is_read_only() && (data_size == 0 || memcmp(m_data + begin, data, data_size) == 0))
        return get_ref();

    // Reallocate if needed - also updates header
    alloc(new_size, 1); // Throws

    char* modify_begin = m_data + begin;

    // Resize previous space to fit new data
    // (not needed if we append to end)
    if (begin != old_size) {
        const char* old_begin = m_data + end;
        const char* old_end = m_data + old_size;
        if (remove_size < add_size) { // expand gap
            char* new_end = m_data + new_size;
            std::copy_backward(old_begin, old_end, new_end);
        }
        else if (add_size < remove_size) { // shrink gap
            char* new_begin = modify_begin + add_size;
            realm::safe_copy_n(old_begin, old_end - old_begin, new_begin);
        }
    }

    // Insert the data
    modify_begin = realm::safe_copy_n(data, data_size, modify_begin);
    if (add_zero_term)
        *modify_begin = 0;

    return get_ref();
}

void ArrayBlob::verify() const
{
#ifdef REALM_DEBUG
    if (get_context_flag()) {
        REALM_ASSERT(has_refs());
        for (size_t i = 0; i < size(); ++i) {
            ref_type blob_ref = Array::get_as_ref(i);
            REALM_ASSERT(blob_ref != 0);
            ArrayBlob blob(m_alloc);
            blob.init_from_ref(blob_ref);
            blob.verify();
        }
    }
    else {
        REALM_ASSERT(!has_refs());
    }
#endif
}
