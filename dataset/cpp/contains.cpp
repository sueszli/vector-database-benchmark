#include "matchit.h"
#include <iostream>
#include <map>

template <typename Map, typename Key>
constexpr bool contains(Map const &map, Key const &key)
{
  using namespace matchit;
  return match(map.find(key))(
      // clang-format off
        pattern | map.end() = false,
        pattern | _         = true
      // clang-format on
  );
}

int32_t main()
{
  std::cout << contains(std::map<int32_t, int32_t>{{1, 2}}, 1) << std::endl;
  return 0;
}
