// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "file_area_freelist.h"
#include <cassert>

namespace vespalib::alloc {

FileAreaFreeList::FileAreaFreeList()
    : _free_areas(),
      _free_sizes(),
      _fences()
{
}

FileAreaFreeList::~FileAreaFreeList()
{
    assert(_fences.empty());
}

void
FileAreaFreeList::remove_from_size_set(uint64_t offset, size_t size)
{
    auto itr = _free_sizes.find(size);
    assert(itr != _free_sizes.end());
    auto &offsets = itr->second;
    auto erased_count = offsets.erase(offset);
    assert(erased_count != 0u);
    if (offsets.empty()) {
        _free_sizes.erase(itr);
    }
}

std::pair<uint64_t, size_t>
FileAreaFreeList::prepare_reuse_area(size_t size)
{
    auto itr = _free_sizes.lower_bound(size);
    if (itr == _free_sizes.end()) {
        return std::make_pair(bad_offset, 0); // No free areas of sufficient size
    }
    auto old_size = itr->first;
    assert(old_size >= size);
    auto &offsets = itr->second;
    assert(!offsets.empty());
    auto oitr = offsets.begin();
    auto offset = *oitr;
    offsets.erase(oitr);
    if (offsets.empty()) {
        _free_sizes.erase(itr);
    }
    // Note: Caller must update _free_areas
    return std::make_pair(offset, old_size);
}

uint64_t
FileAreaFreeList::alloc(size_t size)
{
    auto reuse_candidate = prepare_reuse_area(size);
    auto offset = reuse_candidate.first;
    if (offset == bad_offset) {
        return bad_offset; // No free areas of sufficient size
    }
    auto fa_itr = _free_areas.find(offset);
    assert(fa_itr != _free_areas.end());
    fa_itr = _free_areas.erase(fa_itr);
    auto old_size = reuse_candidate.second;
    if (old_size > size) {
        // Old area beyond what we reuse should still be a free area.
        auto ins_res = _free_sizes[old_size - size].insert(offset + size);
        assert(ins_res.second);
        _free_areas.emplace_hint(fa_itr, offset + size, old_size - size);
    }
    return offset;
}

void
FileAreaFreeList::free(uint64_t offset, size_t size)
{
    auto itr = _free_areas.lower_bound(offset);
    if (itr != _free_areas.end() && itr->first <= offset + size) {
        assert(itr->first == offset + size);
        if (!_fences.contains(offset + size)) {
            // Merge with next free area
            remove_from_size_set(itr->first, itr->second);
            size += itr->second;
            itr = _free_areas.erase(itr);
        }
    }
    bool adjusted_prev_area = false;
    if (itr != _free_areas.begin()) {
        --itr;
        if (itr->first + itr->second >= offset) {
            assert(itr->first + itr->second == offset);
            if (!_fences.contains(offset)) {
                // Merge with previous free area
                remove_from_size_set(itr->first, itr->second);
                offset = itr->first;
                size += itr->second;
                itr->second = size;
                adjusted_prev_area = true;
            } else {
                ++itr;
            }
        } else {
            ++itr;
        }
    }
    if (!adjusted_prev_area) {
        _free_areas.emplace_hint(itr, offset, size);
    }
    auto ins_res = _free_sizes[size].insert(offset);
    assert(ins_res.second);
}

void
FileAreaFreeList::add_premmapped_area(uint64_t offset, size_t size)
{
    auto itr = _free_areas.lower_bound(offset);
    if (itr != _free_areas.end()) {
        assert(itr->first >= offset + size);
    }
    auto ins_res = _free_sizes[size].insert(offset);
    assert(ins_res.second);
    _free_areas.emplace_hint(itr, offset, size);
    auto fences_ins_res = _fences.insert(offset);
    assert(fences_ins_res.second);
}

void
FileAreaFreeList::remove_premmapped_area(uint64_t offset, size_t size)
{
    auto itr = _free_areas.lower_bound(offset);
    assert(itr != _free_areas.end());
    assert(itr->first == offset);
    assert(itr->second == size);
    auto sizes_itr = _free_sizes.lower_bound(size);
    assert(sizes_itr != _free_sizes.end());
    assert(sizes_itr->first == size);
    assert(sizes_itr->second.contains(offset));
    assert(_fences.contains(offset));
    _free_areas.erase(itr);
    sizes_itr->second.erase(offset);
    if (sizes_itr->second.empty()) {
        _free_sizes.erase(sizes_itr);
    }
    _fences.erase(offset);
}

}
