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

#include "verified_integer.hpp"

using namespace realm;
using namespace realm::test_util;


void VerifiedInteger::verify_neighbours(size_t ndx)
{
    if (v.size() > ndx)
        REALM_ASSERT(v[ndx] == u.get(ndx));

    if (ndx > 0)
        REALM_ASSERT(v[ndx - 1] == u.get(ndx - 1));

    if (v.size() > ndx + 1)
        REALM_ASSERT(v[ndx + 1] == u.get(ndx + 1));
}

void VerifiedInteger::add(int64_t value)
{
    v.push_back(value);
    u.add(value);
    REALM_ASSERT(v.size() == u.size());
    verify_neighbours(v.size());
    REALM_ASSERT(occasional_verify());
}

void VerifiedInteger::insert(size_t ndx, int64_t value)
{
    v.insert(v.begin() + ndx, value);
    u.insert(ndx, value);
    REALM_ASSERT(v.size() == u.size());
    verify_neighbours(ndx);
    REALM_ASSERT(occasional_verify());
}

int64_t VerifiedInteger::get(size_t ndx)
{
    REALM_ASSERT(v[ndx] == u.get(ndx));
    return v[ndx];
}

int64_t VerifiedInteger::sum(size_t start, size_t end)
{
    int64_t running_sum = 0;

    if (start == end)
        return 0;

    if (end == size_t(-1))
        end = v.size();

    for (size_t t = start; t < end; ++t)
        running_sum += v[t];
#ifdef LEGACY_TEST
    REALM_ASSERT(running_sum == u.sum(start, end));
#endif
    return running_sum;
}

int64_t VerifiedInteger::maximum(size_t start, size_t end)
{
    if (end == size_t(-1))
        end = v.size();

    if (end == start)
        return 0;

    int64_t max = v[start];

    for (size_t t = start + 1; t < end; ++t)
        if (v[t] > max)
            max = v[t];

#ifdef LEGACY_TEST
    REALM_ASSERT(max == u.maximum(start, end));
#endif
    return max;
}

int64_t VerifiedInteger::minimum(size_t start, size_t end)
{
    if (end == size_t(-1))
        end = v.size();

    if (end == start)
        return 0;

    int64_t min = v[start];

    for (size_t t = start + 1; t < end; ++t)
        if (v[t] < min)
            min = v[t];

#ifdef LEGACY_TEST
    REALM_ASSERT(min == u.minimum(start, end));
#endif
    return min;
}

void VerifiedInteger::set(size_t ndx, int64_t value)
{
    v[ndx] = value;
    u.set(ndx, value);
    verify_neighbours(ndx);
    REALM_ASSERT(occasional_verify());
}

void VerifiedInteger::erase(size_t ndx)
{
    v.erase(v.begin() + ndx);
    u.erase(ndx);
    REALM_ASSERT(v.size() == u.size());
    verify_neighbours(ndx);
    REALM_ASSERT(occasional_verify());
}

void VerifiedInteger::clear()
{
    v.clear();
    u.clear();
    REALM_ASSERT(v.size() == u.size());
    REALM_ASSERT(occasional_verify());
}

size_t VerifiedInteger::find_first(int64_t value)
{
    std::vector<int64_t>::iterator it = std::find(v.begin(), v.end(), value);
    size_t ndx = std::distance(v.begin(), it);
    size_t index2 = u.find_first(value);
    REALM_ASSERT(ndx == index2 || (it == v.end() && index2 == size_t(-1)));
    static_cast<void>(index2);
    return ndx;
}

size_t VerifiedInteger::size()
{
    REALM_ASSERT(v.size() == u.size());
    return v.size();
}

bool VerifiedInteger::verify()
{
    REALM_ASSERT(u.size() == v.size());
    if (u.size() != v.size())
        return false;

    for (size_t t = 0; t < v.size(); ++t) {
        REALM_ASSERT(v[t] == u.get(t));
        if (v[t] != u.get(t))
            return false;
    }
    return true;
}

// makes it run amortized the same time complexity as original, even though the row count grows
bool VerifiedInteger::occasional_verify()
{
    if (m_random.draw_int_max(v.size() / 10) == 0)
        return verify();
    return true;
}

VerifiedInteger::~VerifiedInteger()
{
    u.destroy();
}
