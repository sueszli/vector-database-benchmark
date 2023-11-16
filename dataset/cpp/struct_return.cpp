// RUN: %clangxx %cxxflags %include_flags %ld_flags %s -Xclang -load -Xclang %lib_pass -o %t
// RUN: %t > %t.out
// RUN: %FileCheck %s < %t.out

#include <easy/jit.h>

#include <functional>
#include <cstdio>

using namespace std::placeholders;

template<class T>
struct Point {
  T x;
  T y;
};

template<class T>
Point<T> build (T a, T b) {
  return Point<T>{(T)(a-7), (T)(2*b)};
}

template<class T>
void test() {
  T val = 1;
  easy::FunctionWrapper<Point<T>(T)> build_val = easy::jit(build<T>, _1, val);

  for(int v = 4; v != 8; ++v) {
    Point<T> xy = build_val((T)v);
    printf("point = %d %d \n", (int)xy.x, (int)xy.y);
  }
}

int main() {

  // CHECK: point = -3 2
  // CHECK: point = -2 2
  // CHECK: point = -1 2
  // CHECK: point = 0 2
  test<short>();

  // CHECK: point = -3 2
  // CHECK: point = -2 2
  // CHECK: point = -1 2
  // CHECK: point = 0 2
  test<int>();
  
  // CHECK: point = -3 2
  // CHECK: point = -2 2
  // CHECK: point = -1 2
  // CHECK: point = 0 2
  test<long>();

  // CHECK: point = -3 2
  // CHECK: point = -2 2
  // CHECK: point = -1 2
  // CHECK: point = 0 2
  test<float>();

  // CHECK: point = -3 2
  // CHECK: point = -2 2
  // CHECK: point = -1 2
  // CHECK: point = 0 2
  test<double>();

  // CHECK: point = -3 2
  // CHECK: point = -2 2
  // CHECK: point = -1 2
  // CHECK: point = 0 2
  test<long double>();

  return 0;
}
