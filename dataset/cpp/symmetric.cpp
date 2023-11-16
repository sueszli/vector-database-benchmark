#include "matchit.h"
#include <iostream>

template <typename Range>
constexpr bool recursiveSymmetric(Range const &range)
{
  using namespace matchit;
  Id<int32_t> i;
  Id<SubrangeT<Range const>> subrange;
  return match(range)(
      // clang-format off
        pattern | ds(i, subrange.at(ooo), i) = [&] { return recursiveSymmetric(*subrange); },
        pattern | ds(_, ooo, _)              = false,
        pattern | _                          = true
      // clang-format on
  );
}

constexpr bool symmetricArray(std::array<int32_t, 5> const &arr)
{
  using namespace matchit;
  Id<int32_t> i, j;
  return match(arr)(
      // clang-format off
         pattern | ds(i, j, _, j, i) = true,
         pattern | _                 = false
      // clang-format on
  );
}

static_assert(symmetricArray(std::array<int32_t, 5>{5, 0, 3, 7, 10}) == false);
static_assert(symmetricArray(std::array<int32_t, 5>{5, 0, 3, 0, 5}) == true);
static_assert(symmetricArray(std::array<int32_t, 5>{5, 1, 3, 0, 5}) == false);

int32_t main()
{
  std::cout << recursiveSymmetric(std::array<int32_t, 5>{5, 0, 3, 7, 10})
            << std::endl;
  std::cout << recursiveSymmetric(std::array<int32_t, 5>{5, 0, 3, 0, 5})
            << std::endl;
  std::cout << recursiveSymmetric(std::array<int32_t, 4>{5, 0, 0, 5})
            << std::endl;
  std::cout << recursiveSymmetric(std::array<int32_t, 4>{5, 0, 0, 4})
            << std::endl;
  std::cout << recursiveSymmetric(std::array<int32_t, 4>{5, 1, 0, 5})
            << std::endl;
  return 0;
}
