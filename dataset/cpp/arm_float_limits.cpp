///////////////////////////////////////////////////////////////////////////////
//  Copyright Christopher Kormanyos 2007 - 2020.
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <limits>

#include <impl/stl_local_constexpr.h>

namespace std
{
  namespace xfloat_impl
  {
    // Use some GCC internal stuff here.
    STL_LOCAL_CONSTEXPR float       avr_nan_flt  = static_cast<float>(__builtin_nan(""));
    STL_LOCAL_CONSTEXPR float       avr_inf_flt  = static_cast<float>(__builtin_inf());
    STL_LOCAL_CONSTEXPR double      avr_nan_dbl  = __builtin_nan("");
    STL_LOCAL_CONSTEXPR double      avr_inf_dbl  = __builtin_inf();
    STL_LOCAL_CONSTEXPR long double avr_nan_ldbl = static_cast<long double>(__builtin_nan(""));
    STL_LOCAL_CONSTEXPR long double avr_inf_ldbl = static_cast<long double>(__builtin_inf());
  }

  float numeric_limits_details::my_value_that_needs_to_be_provided_flt_quiet_NaN()
  {
    return std::xfloat_impl::avr_nan_flt;
  }

  float numeric_limits_details::my_value_that_needs_to_be_provided_flt_signaling_NaN()
  {
    return 0.0F;
  }

  float numeric_limits_details::my_value_that_needs_to_be_provided_flt_infinity()
  {
    return std::xfloat_impl::avr_inf_flt;
  }

  double numeric_limits_details::my_value_that_needs_to_be_provided_dbl_quiet_NaN()
  {
    return std::xfloat_impl::avr_nan_dbl;
  }

  double numeric_limits_details::my_value_that_needs_to_be_provided_dbl_signaling_NaN()
  {
    return 0.0;
  }

  double numeric_limits_details::my_value_that_needs_to_be_provided_dbl_infinity()
  {
    return std::xfloat_impl::avr_inf_dbl;
  }

  long double numeric_limits_details::my_value_that_needs_to_be_provided_ldbl_quiet_NaN()
  {
    return std::xfloat_impl::avr_nan_ldbl;
  }

  long double numeric_limits_details::my_value_that_needs_to_be_provided_ldbl_signaling_NaN()
  {
    return 0.0L;
  }

  long double numeric_limits_details::my_value_that_needs_to_be_provided_ldbl_infinity()
  {
    return std::xfloat_impl::avr_inf_ldbl;
  }
}
