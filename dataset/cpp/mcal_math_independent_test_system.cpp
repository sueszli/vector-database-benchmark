///////////////////////////////////////////////////////////////////////////////
//  Copyright Christopher Kormanyos 2019 - 2022.
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)
//

//#define MCAL_MATH_USE_INDEPENDENT_TEST_SYSTEM_MATHLINK

#include <mcal_math_independent_test_system.h>
#if defined(MCAL_MATH_USE_INDEPENDENT_TEST_SYSTEM_MATHLINK)
#include <math/test/math_test_independent_test_system_mathlink.h>
#else
#include <math/test/math_test_independent_test_system_boost.h>
#endif

#if defined(MCAL_MATH_USE_INDEPENDENT_TEST_SYSTEM_MATHLINK)
namespace
{
  // Use the default mathlink 12.1 kernel location on Win*.
  static const char independent_test_system_mathlink_location[] =
    "\"C:\\Program Files\\Wolfram Research\\Mathematica\\12.1\\MathKernel.exe\"";
}
#endif

WIDE_INTEGER_NAMESPACE::math::test::independent_test_system_base& mcal::math::independent_test_system0() noexcept
{
  #if defined(MCAL_MATH_USE_INDEPENDENT_TEST_SYSTEM_MATHLINK)
  using test_system_type = WIDE_INTEGER_NAMESPACE::math::test::independent_test_system_mathlink<independent_test_system_mathlink_location>;
  #else
  using test_system_type = WIDE_INTEGER_NAMESPACE::math::test::independent_test_system_boost;
  #endif

  static test_system_type ts0;

  return ts0;
}
