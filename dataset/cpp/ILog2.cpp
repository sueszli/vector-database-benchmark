#include <FEXCore/Utils/MathUtils.h>
#include <catch2/catch.hpp>

TEST_CASE("ILog2") {
  auto i = GENERATE(range(0, 64));
  REQUIRE(FEXCore::ilog2(1ull << i) == i);
}

TEST_CASE("DividePow2") {
  auto j = GENERATE(range(0, 64));
  auto i = GENERATE(range(0, 64));
  REQUIRE(FEXCore::DividePow2(1ull << j, 1ull << i) == ((1ull << j) / (1ull << i)));
}
