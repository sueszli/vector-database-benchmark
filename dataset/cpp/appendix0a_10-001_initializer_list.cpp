///////////////////////////////////////////////////////////////////////////////
//  Copyright Christopher Kormanyos 2019.
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// appendix0a_11-001_type_inference.cpp

#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <numeric>

// The type of n is int.
auto n = 3;

// The type of u is std::uint8_t.
auto u = std::uint8_t(3U);

// The type of lst is std::initializer_list<int>.
auto lst =
{
  1, 2, 3
};

int main()
{
  std::cout << "n: " << n << std::endl;

  std::cout << "u: " << std::uint32_t(u) << std::endl;

  std::cout << "sum: " << std::accumulate(lst.begin(), lst.end(), 0) << std::endl;
}
