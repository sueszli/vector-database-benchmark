#define NO_SCALAR_REFERENCES_USED_IN_PATTERNS 1

#include "matchit.h"
#include <iostream>

constexpr bool isLarge(double value)
{
  using namespace matchit;
  return match(value)(
      // clang-format off
        pattern | app(_ * _, _ > 1000) = true,
        pattern | _                    = false
      // clang-format on
  );
}

// app with projection returning scalar types is supported by constexpr match.
static_assert(isLarge(100));

int32_t main()
{
  std::cout << isLarge(10) << std::endl;
  return 0;
}
