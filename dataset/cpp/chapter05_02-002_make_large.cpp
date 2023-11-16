///////////////////////////////////////////////////////////////////////////////
//  Copyright Christopher Kormanyos 2019- 2020.
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// chapter05_02-001_make_large.cpp

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>

template<typename ularge_type,
         typename ushort_type>
ularge_type make_large(const ushort_type& lo,
                       const ushort_type& hi)
{
  constexpr int ushort_digits
    = std::numeric_limits<ushort_type>::digits;

  constexpr int ularge_digits
    = std::numeric_limits<ularge_type>::digits;

  // Ensure proper width of the large type.
  static_assert(ularge_digits == (2 * ushort_digits), "error: ularge_type size mismatch");

  // Shift the high part to the left.
  const auto uh = static_cast<ularge_type>(static_cast<ularge_type>(hi) << ushort_digits);

  // Return the composite result.
  return static_cast<ularge_type>(uh | lo);
}

int main()
{
  std::uint8_t  lo8  = UINT8_C(0x34);
  std::uint8_t  hi8  = UINT8_C(0x12);
  std::uint16_t u16
    = make_large<std::uint16_t>(lo8, hi8);

  std::uint16_t lo16 = UINT16_C(0x5678);
  std::uint16_t hi16 = UINT16_C(0x1234);
  std::uint32_t u32
    = make_large<std::uint32_t>(lo16, hi16);

  std::uint32_t lo32 = UINT32_C(0x9ABC'DEF0);
  std::uint32_t hi32 = UINT32_C(0x1234'5678);
  std::uint64_t u64
    = make_large<std::uint64_t>(lo32, hi32);

  std::cout << std::hex << std::showbase << std::uppercase << u16 << std::endl;
  std::cout << std::hex << std::showbase << std::uppercase << u32 << std::endl;
  std::cout << std::hex << std::showbase << std::uppercase << u64 << std::endl;
}
