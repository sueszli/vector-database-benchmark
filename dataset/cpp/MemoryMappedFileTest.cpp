/*
 * Copyright 2014-2023 Real Logic Limited.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cstdlib>
#include <gtest/gtest.h>
#include <string>

#include "util/MemoryMappedFile.h"
#include "TestUtils.h"

using namespace aeron::util;
using namespace aeron::test;

#ifdef _MSC_VER
#define unlink _unlink
#endif

TEST(mmfileTest, failToOpen)
{
    ASSERT_ANY_THROW({
        auto m = MemoryMappedFile::mapExisting("this file does no exist");
    });
}

TEST(mmfileTest, createCheck)
{
    MemoryMappedFile::ptr_t m;

    const std::size_t size = 10000;
    const std::string name(makeTempFileName());

    ASSERT_NO_THROW({
        m = MemoryMappedFile::createNew(name.c_str(), 0, size, true);
    });

    ASSERT_EQ(m->getMemorySize(), size);
    ASSERT_NE(m->getMemoryPtr(), nullptr);

    for (std::size_t n = 0; n < size; n++)
    {
        ASSERT_EQ(m->getMemoryPtr()[n], 0);
    }

    ::unlink(name.c_str());
}

TEST(mmfileTest, writeReadCheck)
{
    MemoryMappedFile::ptr_t m;

    const std::size_t size = 10000;
    std::string name = makeTempFileName();

    ASSERT_NO_THROW({
        m = MemoryMappedFile::createNew(name.c_str(), 0, size, true);
    });

    for (std::size_t n = 0; n < size; n++)
    {
        m->getMemoryPtr()[n] = static_cast<std::uint8_t>(n & 0xff);
    }

    m.reset();

    ASSERT_NO_THROW({
        m = MemoryMappedFile::mapExisting(name.c_str());
    });

    ASSERT_EQ(m->getMemorySize(), size);
    ASSERT_NE(m->getMemoryPtr(), nullptr);

    for (size_t n = 0; n < size; n++)
    {
        ASSERT_EQ(m->getMemoryPtr()[n], static_cast<std::uint8_t>(n & 0xff));
    }

    ::unlink(name.c_str());
}
