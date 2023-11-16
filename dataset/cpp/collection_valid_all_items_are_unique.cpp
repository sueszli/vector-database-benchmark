#include <xtd/xtd.tunit>

using namespace xtd::tunit;

namespace unit_tests {
  class test_class_(test) {
  public:
    void test_method_(test_case_succeed) {
      std::vector a = {1, 2, 3, 4};
      collection_valid::all_items_are_unique(a);
    }
    
    void test_method_(test_case_failed) {
      std::vector a = {1, 2, 3, 4, 1};
      collection_valid::all_items_are_unique(a);
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
//   FAILED  test.test_case_failed (0 ms total)
//     Expected: all items are unique
//     But was:  < 1, 2, 3, 4, 1 >
//     Stack Trace: in |---OMITTED---|/collection_valid_all_items_are_unique.cpp:13
//
// Test results:
//   SUCCEED 1 test.
//   FAILED  1 test.
// End 2 tests from 1 test case ran. (0 ms total)
