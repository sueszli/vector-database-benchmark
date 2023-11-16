#include <catch2/catch.hpp>

#include <SI/detail/cross_unit_operations.h>
#include <SI/detail/detail.h>
#include <SI/detail/unit.h>
#include <SI/detail/unit_cast.h>

TEMPLATE_TEST_CASE("eps_equals comparison of 0 and 0 returns true",
                   "[detail][eps_equals]", long double, double, float) {
  constexpr TestType zero{0};

  STATIC_REQUIRE(SI::detail::eps_equals(zero, zero));
}

TEMPLATE_TEST_CASE("eps_equals comparison of 0 and epsilon returns false",
                   "[detail][eps_equals]", long double, double, float) {
  constexpr TestType zero{0};
  constexpr TestType half_eps{std::numeric_limits<TestType>::epsilon()};

  STATIC_REQUIRE(!SI::detail::eps_equals(zero, half_eps));
  STATIC_REQUIRE(!SI::detail::eps_equals(half_eps, zero));
}

TEMPLATE_TEST_CASE("eps_equals comparison of 0 and -epsilon returns false",
                   "[detail][eps_equals]", long double, double, float) {
  constexpr TestType zero{0};
  constexpr TestType half_eps{-std::numeric_limits<TestType>::epsilon()};

  STATIC_REQUIRE(!SI::detail::eps_equals(zero, half_eps));
  STATIC_REQUIRE(!SI::detail::eps_equals(half_eps, zero));
}

TEMPLATE_TEST_CASE("eps_equals comparison of 0 and epsilon/2 returns true",
                   "[detail][eps_equals]", long double, double, float) {
  constexpr TestType zero{0};
  constexpr TestType half_eps{std::numeric_limits<TestType>::epsilon() / 2};

  STATIC_REQUIRE(SI::detail::eps_equals(zero, half_eps));
  STATIC_REQUIRE(SI::detail::eps_equals(half_eps, zero));
}

TEMPLATE_TEST_CASE("eps_equals comparison of 0 and -epsilon/2 returns true",
                   "[detail][eps_equals]", long double, double, float) {
  constexpr TestType zero{0};
  constexpr TestType half_eps{-std::numeric_limits<TestType>::epsilon() / 2};

  STATIC_REQUIRE(SI::detail::eps_equals(zero, half_eps));
  STATIC_REQUIRE(SI::detail::eps_equals(half_eps, zero));
}

TEMPLATE_TEST_CASE(
    "eps_equals comparison of non_zero and -epsilon/2 returns true",
    "[detail][eps_equals]", long double, double) {
  constexpr TestType non_zero{1234.345};
  constexpr TestType half_eps =
      non_zero + -std::numeric_limits<TestType>::epsilon() / 2;

  STATIC_REQUIRE(SI::detail::eps_equals(non_zero, half_eps));
  STATIC_REQUIRE(SI::detail::eps_equals(half_eps, non_zero));
}

// not covered by the template test case above because of the different literal
// construction using 'f'-suffix
TEST_CASE(
    "eps_equals comparison of non_zero and -epsilon/2 returns true for floats",
    "[detail][eps_equals]") {
  constexpr float non_zero = 1234.345f;
  constexpr float half_eps =
      non_zero + -std::numeric_limits<float>::epsilon() / 2;

  STATIC_REQUIRE(SI::detail::eps_equals(non_zero, half_eps));
  STATIC_REQUIRE(SI::detail::eps_equals(half_eps, non_zero));
}

TEMPLATE_TEST_CASE(
    "eps_equals comparison of non_zero and epsilon/2 returns true",
    "[detail][eps_equals]", long double, double) {
  constexpr TestType non_zero{1234.345};
  constexpr TestType half_eps{non_zero +
                              std::numeric_limits<TestType>::epsilon() / 2};

  STATIC_REQUIRE(SI::detail::eps_equals(non_zero, half_eps));
  STATIC_REQUIRE(SI::detail::eps_equals(half_eps, non_zero));
}

TEST_CASE(
    "eps_equals comparison of non_zero and epsilon/2 returns true for floats",
    "[detail][eps_equals]") {
  constexpr float non_zero{1234.345f};
  constexpr float half_eps{non_zero +
                           std::numeric_limits<float>::epsilon() / 2};

  STATIC_REQUIRE(SI::detail::eps_equals(non_zero, half_eps));
  STATIC_REQUIRE(SI::detail::eps_equals(half_eps, non_zero));
}

TEMPLATE_TEST_CASE(
    "eps_equals comparison of non_zero and quiet_NaN returns false",
    "[detail][eps_equals][runtime]", long double, double, float) {
  constexpr TestType zero{0};
  constexpr TestType nan{std::numeric_limits<TestType>::quiet_NaN()};

  REQUIRE(!SI::detail::eps_equals(zero, nan));
  REQUIRE(!SI::detail::eps_equals(nan, zero));
}

TEMPLATE_TEST_CASE(
    "eps_equals comparison of quiet_NaN and quiet_NaN returns false",
    "[detail][eps_equals][runtime]", long double, double, float) {

  constexpr TestType nan{std::numeric_limits<TestType>::quiet_NaN()};

  REQUIRE(!SI::detail::eps_equals(nan, nan));
}

TEMPLATE_TEST_CASE(
    "eps_equals comparison of signaling_NaN and quiet_NaN returns false",
    "[detail][eps_equals][runtime]", long double, double, float) {

  constexpr TestType signaling_nan{
      std::numeric_limits<TestType>::signaling_NaN()};
  constexpr TestType quiet_nan{std::numeric_limits<TestType>::quiet_NaN()};

  REQUIRE(!SI::detail::eps_equals(signaling_nan, quiet_nan));
}

TEMPLATE_TEST_CASE(
    "eps_equals comparison of signaling_NaN and signaling_NaN returns false",
    "[detail][eps_equals][runtime]", long double, double, float) {

  constexpr TestType signaling_nan{
      std::numeric_limits<TestType>::signaling_NaN()};
  constexpr TestType quiet_nan{std::numeric_limits<TestType>::signaling_NaN()};

  REQUIRE(!SI::detail::eps_equals(signaling_nan, quiet_nan));
}

template <typename _type, typename _ratio>
using resulting_t = SI::detail::unit_t<'Y', std::ratio<1>, _type, _ratio>;

TEST_CASE("GIVEN a value of an integral type AND a value of floating point "
          "type WHEN divided THEN result is of left hand type") {
  constexpr SI::detail::unit_t<'X', std::ratio<1>, long double, std::ratio<1>>
      v1{1};
  constexpr SI::detail::unit_t<'X', std::ratio<1>, int64_t, std::ratio<1>> v2{
      1};

  constexpr auto result = SI::detail::cross_unit_multiply<resulting_t>(v1, v2);

  STATIC_REQUIRE(std::is_same<long double,
                              typename decltype(result)::internal_type>::value);
}
