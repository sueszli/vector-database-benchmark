#include <xtd/xtd.tunit>

using namespace xtd::tunit;

namespace unit_tests {
  class test_class_(test) {
  public:
    void test_method_(test_case_succeed) {
      std::vector a = {1, 2, 3, 4};
      collection_assume::are_not_equivalent({1, 2, 3, 5}, a);
    }
    
    void test_method_(test_case_aborted) {
      std::vector a = {1, 2, 3, 4};
      collection_assume::are_not_equivalent({4, 3, 2, 1}, a);
    }
  };
}

auto main()->int {
  return console_unit_test().run();
}

// This code produces the following output:
//
// Start 2 tests from 1 test case
// Run tests:
//   SUCCEED test.test_case_succeed (0 ms total)
//   ABORTED test.test_case_aborted (0 ms total)
//     Test aborted
//     Expected: not equivalent < 4, 3, 2, 1 >
//     But was:  < 1, 2, 3, 4 >
//     Stack Trace: in |---OMITTED---|/collection_assume_are_not_equivalent.cpp:13
//
// Test results:
//   SUCCEED 1 test.
//   ABORTED 1 test.
// End 2 tests from 1 test case ran. (0 ms total)
