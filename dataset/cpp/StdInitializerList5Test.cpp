// Source: https://twitter.com/olafurw/status/1214927552990601217?s=20
#include <initializer_list>

int f(std::initializer_list<int> l) {}

int main()
{
  return f({1, 2, 3});
}
