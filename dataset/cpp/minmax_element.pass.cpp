//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter>
//   requires LessThanComparable<Iter::value_type>
//   pair<Iter, Iter>
//   minmax_element(Iter first, Iter last);

#include <algorithm>
#include <random>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

std::mt19937 randomness;

template <class Iter>
void
test(Iter first, Iter last)
{
    std::pair<Iter, Iter> p = std::minmax_element(first, last);
    if (first != last)
    {
        for (Iter j = first; j != last; ++j)
        {
            assert(!(*j < *p.first));
            assert(!(*p.second < *j));
        }
    }
    else
    {
        assert(p.first == last);
        assert(p.second == last);
    }
}

template <class Iter>
void
test(int N)
{
    int* a = new int[N];
    for (int i = 0; i < N; ++i)
        a[i] = i;
    std::shuffle(a, a+N, randomness);
    test(Iter(a), Iter(a+N));
    delete [] a;
}

template <class Iter>
void
test()
{
    test<Iter>(0);
    test<Iter>(1);
    test<Iter>(2);
    test<Iter>(3);
    test<Iter>(10);
    test<Iter>(1000);
    {
    const int N = 100;
    int* a = new int[N];
    for (int i = 0; i < N; ++i)
        a[i] = 5;
    std::shuffle(a, a+N, randomness);
    std::pair<Iter, Iter> p = std::minmax_element(Iter(a), Iter(a+N));
    assert(base(p.first) == a);
    assert(base(p.second) == a+N-1);
    delete [] a;
    }
}

#if TEST_STD_VER >= 14
constexpr int il[] = { 2, 4, 6, 8, 7, 5, 3, 1 };
#endif

void constexpr_test()
{
#if TEST_STD_VER >= 14
    constexpr auto p = std::minmax_element(il, il+8);
    static_assert ( *(p.first)  == 1, "" );
    static_assert ( *(p.second) == 8, "" );
#endif
}

int main(int, char**)
{
    test<forward_iterator<const int*> >();
    test<bidirectional_iterator<const int*> >();
    test<random_access_iterator<const int*> >();
    test<const int*>();

   constexpr_test();

  return 0;
}
