// RUN: %clangxx %cxxflags %include_flags %ld_flags %s -Xclang -load -Xclang %lib_pass -o %t
// RUN: %t > %t.out
// RUN: %FileCheck %s < %t.out

#include <easy/jit.h>

#include <functional>
#include <cstdio>

using namespace std::placeholders;

int add (long a, long b) {
  return a+b;
}

int main() {
  easy::FunctionWrapper<int(long)> inc = easy::jit(add, _1, -1);

  // CHECK: inc(4) is 3
  // CHECK: inc(5) is 4
  // CHECK: inc(6) is 5
  // CHECK: inc(7) is 6
  for(int v = 4; v != 8; ++v)
    printf("inc(%d) is %d\n", v, inc(v));

  return 0;
}
