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

#include "testsettings.hpp"
#ifdef TEST_ARRAY_BINARY

#include <realm/array_blobs_small.hpp>

#include "test.hpp"

using namespace realm;


// Test independence and thread-safety
// -----------------------------------
//
// All tests must be thread safe and independent of each other. This
// is required because it allows for both shuffling of the execution
// order and for parallelized testing.
//
// In particular, avoid using std::rand() since it is not guaranteed
// to be thread safe. Instead use the API offered in
// `test/util/random.hpp`.
//
// All files created in tests must use the TEST_PATH macro (or one of
// its friends) to obtain a suitable file system path. See
// `test/util/test_path.hpp`.
//
//
// Debugging and the ONLY() macro
// ------------------------------
//
// A simple way of disabling all tests except one called `Foo`, is to
// replace TEST(Foo) with ONLY(Foo) and then recompile and rerun the
// test suite. Note that you can also use filtering by setting the
// environment varible `UNITTEST_FILTER`. See `README.md` for more on
// this.
//
// Another way to debug a particular test, is to copy that test into
// `experiments/testcase.cpp` and then run `sh build.sh
// check-testcase` (or one of its friends) from the command line.


TEST(ArraySmallBlobs_Basic)
{
    ArraySmallBlobs c(Allocator::get_default());
    c.create();

    // TEST(ArraySmallBlobs_MultiEmpty)

    c.add(BinaryData());
    c.add(BinaryData());
    c.add(BinaryData());
    c.add(BinaryData());
    c.add(BinaryData());
    c.add(BinaryData());

    CHECK_EQUAL(6, c.size());

    CHECK_EQUAL(0, c.get(0).size());
    CHECK_EQUAL(0, c.get(1).size());
    CHECK_EQUAL(0, c.get(2).size());
    CHECK_EQUAL(0, c.get(3).size());
    CHECK_EQUAL(0, c.get(4).size());
    CHECK_EQUAL(0, c.get(5).size());


    // TEST(ArraySmallBlobs_Set)

    c.set(0, BinaryData("hey", 4));

    CHECK_EQUAL(6, c.size());

    CHECK_EQUAL("hey", c.get(0).data());
    CHECK_EQUAL(4, c.get(0).size());
    CHECK_EQUAL(0, c.get(1).size());
    CHECK_EQUAL(0, c.get(2).size());
    CHECK_EQUAL(0, c.get(3).size());
    CHECK_EQUAL(0, c.get(4).size());
    CHECK_EQUAL(0, c.get(5).size());


    // TEST(ArraySmallBlobs_Add)

    c.clear();
    CHECK_EQUAL(0, c.size());

    c.add(BinaryData("abc", 4));
    CHECK_EQUAL("abc", c.get(0).data()); // single
    CHECK_EQUAL(4, c.get(0).size());
    CHECK_EQUAL(1, c.size());

    c.add(BinaryData("defg", 5)); // non-empty
    CHECK_EQUAL("abc", c.get(0).data());
    CHECK_EQUAL("defg", c.get(1).data());
    CHECK_EQUAL(4, c.get(0).size());
    CHECK_EQUAL(5, c.get(1).size());
    CHECK_EQUAL(2, c.size());


    // TEST(ArraySmallBlobs_Set2)

    // {shrink, grow} x {first, middle, last, single}
    c.clear();

    c.add(BinaryData("abc", 4));
    c.set(0, BinaryData("de", 3)); // shrink single
    CHECK_EQUAL("de", c.get(0).data());
    CHECK_EQUAL(1, c.size());

    c.set(0, BinaryData("abcd", 5)); // grow single
    CHECK_EQUAL("abcd", c.get(0).data());
    CHECK_EQUAL(1, c.size());

    c.add(BinaryData("efg", 4));
    CHECK_EQUAL("abcd", c.get(0).data());
    CHECK_EQUAL("efg", c.get(1).data());
    CHECK_EQUAL(2, c.size());

    c.set(1, BinaryData("hi", 3)); // shrink last
    CHECK_EQUAL("abcd", c.get(0).data());
    CHECK_EQUAL("hi", c.get(1).data());
    CHECK_EQUAL(2, c.size());

    c.set(1, BinaryData("jklmno", 7)); // grow last
    CHECK_EQUAL("abcd", c.get(0).data());
    CHECK_EQUAL("jklmno", c.get(1).data());
    CHECK_EQUAL(2, c.size());

    c.add(BinaryData("pq", 3));
    c.set(1, BinaryData("efghijkl", 9)); // grow middle
    CHECK_EQUAL("abcd", c.get(0).data());
    CHECK_EQUAL("efghijkl", c.get(1).data());
    CHECK_EQUAL("pq", c.get(2).data());
    CHECK_EQUAL(3, c.size());

    c.set(1, BinaryData("x", 2)); // shrink middle
    CHECK_EQUAL("abcd", c.get(0).data());
    CHECK_EQUAL("x", c.get(1).data());
    CHECK_EQUAL("pq", c.get(2).data());
    CHECK_EQUAL(3, c.size());

    c.set(0, BinaryData("qwertyuio", 10)); // grow first
    CHECK_EQUAL("qwertyuio", c.get(0).data());
    CHECK_EQUAL("x", c.get(1).data());
    CHECK_EQUAL("pq", c.get(2).data());
    CHECK_EQUAL(3, c.size());

    c.set(0, BinaryData("mno", 4)); // shrink first
    CHECK_EQUAL("mno", c.get(0).data());
    CHECK_EQUAL("x", c.get(1).data());
    CHECK_EQUAL("pq", c.get(2).data());
    CHECK_EQUAL(3, c.size());


    // TEST(ArraySmallBlobs_Insert)

    c.clear();

    c.insert(0, BinaryData("abc", 4)); // single
    CHECK_EQUAL("abc", c.get(0).data());
    CHECK_EQUAL(1, c.size());

    c.insert(1, BinaryData("d", 2)); // end
    CHECK_EQUAL("abc", c.get(0).data());
    CHECK_EQUAL("d", c.get(1).data());
    CHECK_EQUAL(2, c.size());

    c.insert(2, BinaryData("ef", 3)); // end
    CHECK_EQUAL("abc", c.get(0).data());
    CHECK_EQUAL("d", c.get(1).data());
    CHECK_EQUAL("ef", c.get(2).data());
    CHECK_EQUAL(3, c.size());

    c.insert(1, BinaryData("ghij", 5)); // middle
    CHECK_EQUAL("abc", c.get(0).data());
    CHECK_EQUAL("ghij", c.get(1).data());
    CHECK_EQUAL("d", c.get(2).data());
    CHECK_EQUAL("ef", c.get(3).data());
    CHECK_EQUAL(4, c.size());

    c.insert(0, BinaryData("klmno", 6)); // first
    CHECK_EQUAL("klmno", c.get(0).data());
    CHECK_EQUAL("abc", c.get(1).data());
    CHECK_EQUAL("ghij", c.get(2).data());
    CHECK_EQUAL("d", c.get(3).data());
    CHECK_EQUAL("ef", c.get(4).data());
    CHECK_EQUAL(5, c.size());


    // TEST(ArraySmallBlobs_Erase)

    c.clear();

    c.add(BinaryData("a", 2));
    c.add(BinaryData("bc", 3));
    c.add(BinaryData("def", 4));
    c.add(BinaryData("ghij", 5));
    c.add(BinaryData("klmno", 6));

    c.erase(0); // first
    CHECK_EQUAL("bc", c.get(0).data());
    CHECK_EQUAL("def", c.get(1).data());
    CHECK_EQUAL("ghij", c.get(2).data());
    CHECK_EQUAL("klmno", c.get(3).data());
    CHECK_EQUAL(4, c.size());

    c.erase(3); // last
    CHECK_EQUAL("bc", c.get(0).data());
    CHECK_EQUAL("def", c.get(1).data());
    CHECK_EQUAL("ghij", c.get(2).data());
    CHECK_EQUAL(3, c.size());

    c.erase(1); // middle
    CHECK_EQUAL("bc", c.get(0).data());
    CHECK_EQUAL("ghij", c.get(1).data());
    CHECK_EQUAL(2, c.size());

    c.erase(0); // single
    CHECK_EQUAL("ghij", c.get(0).data());
    CHECK_EQUAL(1, c.size());

    c.erase(0); // all
    CHECK_EQUAL(0, c.size());
    CHECK(c.is_empty());

    c.destroy();
}

TEST(ArraySmallBlobs_Nulls)
{
    ArraySmallBlobs c(Allocator::get_default());
    c.create();

    c.add(BinaryData());
    c.add(BinaryData("", 0));
    c.add(BinaryData("foo"));

    CHECK(c.get(0).is_null());
    CHECK(!c.get(1).is_null());
    CHECK(!c.get(1).is_null());

    // Contains
    //      Null
    CHECK(c.get(0).contains(c.get(0)));
    CHECK(!c.get(0).contains(c.get(1)));
    CHECK(!c.get(0).contains(c.get(2)));

    //      Empty string
    CHECK(c.get(1).contains(c.get(0)));
    CHECK(c.get(1).contains(c.get(1)));
    CHECK(!c.get(1).contains(c.get(2)));

    //      "foo"
    CHECK(c.get(2).contains(c.get(0)));
    CHECK(c.get(2).contains(c.get(1)));
    CHECK(c.get(2).contains(c.get(2)));


    // Begins with
    //      Null
    CHECK(c.get(0).begins_with(c.get(0)));
    CHECK(!c.get(0).begins_with(c.get(1)));
    CHECK(!c.get(0).begins_with(c.get(2)));

    //      Empty string
    CHECK(c.get(1).begins_with(c.get(0)));
    CHECK(c.get(1).begins_with(c.get(1)));
    CHECK(!c.get(1).begins_with(c.get(2)));

    //      "foo"
    CHECK(c.get(2).begins_with(c.get(0)));
    CHECK(c.get(2).begins_with(c.get(1)));
    CHECK(c.get(2).begins_with(c.get(2)));

    // Ends with
    //      Null
    CHECK(c.get(0).ends_with(c.get(0)));
    CHECK(!c.get(0).ends_with(c.get(1)));
    CHECK(!c.get(0).ends_with(c.get(2)));

    //      Empty string
    CHECK(c.get(1).ends_with(c.get(0)));
    CHECK(c.get(1).ends_with(c.get(1)));
    CHECK(!c.get(1).ends_with(c.get(2)));

    //      "foo"
    CHECK(c.get(2).ends_with(c.get(0)));
    CHECK(c.get(2).ends_with(c.get(1)));
    CHECK(c.get(2).ends_with(c.get(2)));

    c.destroy();
}


#endif // TEST_ARRAY_BINARY
