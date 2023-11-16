// RUN: %clangxx %cxxflags %include_flags %ld_flags %s -Xclang -load -Xclang %lib_pass -o %t
// RUN: %t > %t.out

#include <easy/jit.h>

#include <functional>
#include <cstdio>

using namespace std::placeholders;

template<class T>
struct Point {
  T x;
  T y;

  Point(T _x, T _y) {
    x = _x;
    y = _y;
  }
};

template<class T>
long whopie (Point<T> a, T b) {
  return a.x * b + a.y * b;
}

template<class T>
void test_swap() {
  T val = 1;
  auto whop = easy::jit(whopie<T>, _2, _1);

  for(int v = 4; v != 8; ++v) {
    long z = whop(val, Point<T>(v*2, v*3));
    printf("  swap(%d) %ld\n", v, z);

    if(z != whopie<T>(Point<T>(v*2, v*3), val))
      exit(-1);
  }
}

template<class T>
void test_specialzie_a() {
  T val = 1;
  auto whop = easy::jit(whopie<T>, _1, val);

  for(int v = 4; v != 8; ++v) {
    long z = whop(Point<T>(v*2, v*3));
    printf("  spec_a(%d) %ld\n", v, z);

    if(z != whopie<T>(Point<T>(v*2, v*3), val))
      exit(-1);
  }
}

template<class T>
void test_specialzie_b() {
  T val = 1;
  auto whop = easy::jit(whopie<T>, Point<T>(val*2, val*3), _1);

  for(int v = 4; v != 8; ++v) {
    long z = whop(v);
    printf("  spec_b(%d) %ld\n", v, z);

    if(z != whopie<T>(Point<T>(val*2, val*3), v))
      exit(-1);
  }
}

template<class T>
void test_specialzie_ab() {
  T val = 1;
  auto whop = easy::jit(whopie<T>, Point<T>(val*2, val*3), val);

  for(int v = 4; v != 8; ++v) {
    long z = whop();
    printf("  spec_ab() %ld\n", z);

    if(z != whopie<T>(Point<T>(val*2, val*3), val))
      exit(-1);
  }
}

template<class T>
void test() {
  printf("== %s ==\n", typeid(T).name());
  test_swap<T>();
  test_specialzie_a<T>();
  test_specialzie_b<T>();
  test_specialzie_ab<T>();
}

int main() {

  test<short>();
  test<int>();
  test<long>();
  test<float>();
  test<double>();
  test<long double>();

  return 0;
}
