/*************************************************************************
 *
 * Copyright 2019 Realm Inc.
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

#include <realm/chunked_binary.hpp>
#include <realm/util/hex_dump.hpp>

using namespace realm;

BinaryIterator ChunkedBinaryData::iterator() const noexcept
{
    return m_begin;
}

size_t ChunkedBinaryData::size() const noexcept
{
    BinaryIterator copy = m_begin;
    size_t result = 0;
    BinaryData chunk;
    do {
        chunk = copy.get_next();
        result += chunk.size();
    } while (chunk.size() > 0);
    return result;
}

bool ChunkedBinaryData::is_null() const
{
    BinaryIterator copy = m_begin;
    BinaryData chunk = copy.get_next();
    return chunk.is_null();
}

char ChunkedBinaryData::operator[](size_t index) const
{
    BinaryIterator copy = m_begin;
    size_t i = index;
    BinaryData chunk;
    do {
        chunk = copy.get_next();
        if (chunk.size() > i) {
            return chunk[i];
        }
        i -= chunk.size();
    } while (chunk.size() != 0);

    throw RuntimeError(ErrorCodes::RangeError, "Offset is out of range");
}

std::string ChunkedBinaryData::hex_dump(const char* separator, int min_digits) const
{
    BinaryIterator copy = m_begin;
    BinaryData chunk;
    std::string dump; // FIXME: Reserve memory
    while (!(chunk = copy.get_next()).is_null()) {
        dump += util::hex_dump(chunk.data(), chunk.size(), separator, min_digits);
    }
    return dump;
}

void ChunkedBinaryData::write_to(util::ResettableExpandableBufferOutputStream& out) const
{
    BinaryIterator copy = m_begin;
    BinaryData chunk;
    while (!(chunk = copy.get_next()).is_null()) {
        out.write(chunk.data(), chunk.size());
    }
}

void ChunkedBinaryData::copy_to(util::AppendBuffer<char>& dest) const
{
    size_t sz = size();
    dest.resize(sz);
    util::Span out(dest);

    BinaryIterator copy = m_begin;
    BinaryData chunk;
    while (!(chunk = copy.get_next()).is_null()) {
        std::copy(chunk.data(), chunk.data() + chunk.size(), out.data());
        out = out.sub_span(chunk.size());
    }
}

// get_first_chunk() is used in situations
// where it is known that there is exactly one
// chunk. This is the case if the ChunkedBinaryData
// has been constructed from BinaryData.
BinaryData ChunkedBinaryData::get_first_chunk() const
{
    return m_begin.get_only();
}
