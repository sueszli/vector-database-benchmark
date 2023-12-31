//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: !libcpp-has-debug-mode, c++03, c++11, c++14

// test container debugging

#include <map>
#include <set>
#include <utility>
#include <cassert>
#include "check_assertion.h"
#include "container_debug_tests.h"
#include "test_macros.h"

using namespace IteratorDebugChecks;

template <class Container, ContainerType CT>
struct AssociativeContainerChecks : BasicContainerChecks<Container, CT> {
  using Base = BasicContainerChecks<Container, CT>;
  using value_type = typename Container::value_type;
  using iterator = typename Container::iterator;
  using const_iterator = typename Container::const_iterator;
  using traits = std::iterator_traits<iterator>;
  using category = typename traits::iterator_category;

  using Base::makeContainer;
public:
  static void run() {
    Base::run();
  }

private:
  // FIXME Add tests here
};

int main(int, char**)
{
  using SetAlloc = test_allocator<int>;
  using MapAlloc = test_allocator<std::pair<const int, int>>;
  // FIXME: Add debug mode to these containers
  if ((false)) {
    AssociativeContainerChecks<
        std::set<int, std::less<int>, SetAlloc>, CT_Set>::run();
    AssociativeContainerChecks<
        std::multiset<int, std::less<int>, SetAlloc>, CT_MultiSet>::run();
    AssociativeContainerChecks<
        std::map<int, int, std::less<int>, MapAlloc>, CT_Map>::run();
    AssociativeContainerChecks<
        std::multimap<int, int, std::less<int>, MapAlloc>, CT_MultiMap>::run();
  }

  return 0;
}
