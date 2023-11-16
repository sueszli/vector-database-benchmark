///////////////////////////////////////////////////////////////////////////////
//  Copyright Christopher Kormanyos 2021 - 2022.
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <app/benchmark/app_benchmark.h>
#include <app/benchmark/app_benchmark_detail.h>

#if(APP_BENCHMARK_TYPE == APP_BENCHMARK_TYPE_BOOST_MULTIPRECISION_CBRT)

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif

//#define APP_BENCHMARK_TYPE_BOOST_BOOST_MULTIPRECISION_CBRT_USE_BIN_FLOAT

#if !defined(BOOST_MP_STANDALONE)
#define BOOST_MP_STANDALONE
#endif

#if !defined(BOOST_MATH_STANDALONE)
#define BOOST_MATH_STANDALONE
#endif

#if !defined(BOOST_NO_RTTI)
#define BOOST_NO_RTTI
#endif

#if !defined(BOOST_DISABLE_THREADS)
#define BOOST_DISABLE_THREADS
#endif

#if !defined(BOOST_NO_EXCEPTIONS)
#define BOOST_NO_EXCEPTIONS
#endif

#if !defined(BOOST_NO_CXX11_THREAD_LOCAL)
#define BOOST_NO_CXX11_THREAD_LOCAL
#endif

#include <boost/config.hpp>
#include <boost/math/special_functions/cbrt.hpp>
#if defined(APP_BENCHMARK_TYPE_BOOST_BOOST_MULTIPRECISION_CBRT_USE_BIN_FLOAT)
#include <boost/multiprecision/cpp_bin_float.hpp>
#else
#include <boost/multiprecision/cpp_dec_float.hpp>
#endif

auto app::benchmark::run_boost_multiprecision_cbrt() -> bool
{
  using big_float_backend_type =
  #if defined(APP_BENCHMARK_TYPE_BOOST_BOOST_MULTIPRECISION_CBRT_USE_BIN_FLOAT)
    boost::multiprecision::cpp_bin_float<101, boost::multiprecision::backends::digit_base_10, void, std::int32_t>;
  #else
    boost::multiprecision::cpp_dec_float<101, std::int32_t, void>;
  #endif

  using big_float_type =
    boost::multiprecision::number<big_float_backend_type, boost::multiprecision::et_off>;

  // Compute a square root.
  static const big_float_type
    big_float_arg
    {
      big_float_type(UINT32_C(123456)) / 100U
    };

  using std::cbrt;
  using boost::math::cbrt;

  const big_float_type big_float_result = cbrt(big_float_arg);

  // N[(123456/100)^(1/3), 111]
  // 10.7276369432283170454869317373527647801772956394047834686224956433128028534945259441672192774907629718402457465
  static const big_float_type
    control
    {
      "10.7276369432283170454869317373527647801772956394047834686224956433128028534945259441672192774907629718402457465"
    };

  // Compare the calculated result with the known control value.
  const bool app_benchmark_result_is_ok = detail::is_close_fraction(big_float_result, control);

  return app_benchmark_result_is_ok;
}

#if defined(APP_BENCHMARK_STANDALONE_MAIN)
int main()
{
  // g++ -Wall -O3 -march=native -I./ref_app/src/mcal/host -I./ref_app/src -DAPP_BENCHMARK_TYPE=APP_BENCHMARK_TYPE_BOOST_MULTIPRECISION_CBRT -DAPP_BENCHMARK_STANDALONE_MAIN ./ref_app/src/app/benchmark/app_benchmark_boost_math_cbrt_tgamma.cpp -o ./ref_app/bin/app_benchmark_boost_multiprecision_cbrt.exe

  bool result_is_ok = true;

  for(unsigned i = 0U; i < 64U; ++i)
  {
    result_is_ok &= app::benchmark::run_boost_multiprecision_cbrt();
  }

  return (result_is_ok ? 0 : -1);
}

#endif

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#endif // APP_BENCHMARK_TYPE_BOOST_MULTIPRECISION_CBRT
