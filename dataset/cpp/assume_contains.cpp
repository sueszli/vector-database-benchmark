#include <xtd/xtd.tunit>
#include <vector>

using namespace std;
using namespace xtd::tunit;

namespace unit_tests {
  class test_class_(test) {
  public:
    void test_method_(test_case_collection_succeed) {
      assume::contains(2, {0, 1, 2, 3});
    }
    
    void test_method_(test_case_collection_aborted) {
      assume::contains(4, {0, 1, 2, 3});
    }
    
    void test_method_(test_case_string_succeed) {
      assume::contains('i', "string");
    }
    
    void test_method_(test_case_string_aborted) {
      assume::contains('a', "string");
    }
  };
}

auto main()->int {
  return console_unit_test().run();
}

// This code produces the following output:
//
// Start 4 tests from 1 test case
// Run tests:
//   SUCCEED test.test_case_collection_succeed (0 ms total)
//   ABORTED test.test_case_collection_aborted (0 ms total)
//     Expected: collection containing 4
//     But was:  < 0, 1, 2, 3 >
//     Stack Trace: in |---OMITTED---|/assume_contains.cpp:14
//   SUCCEED test.test_case_string_succeed (0 ms total)
//   ABORTED test.test_case_string_aborted (0 ms total)
//     Expected: collection containing a
//     But was:  < 's', 't', 'r', 'i', 'n', 'g' >
//     Stack Trace: in |---OMITTED---|/assume_contains.cpp:22
//
// Test results:
//   SUCCEED 2 tests.
//   ABORTED 2 tests.
// End 4 tests from 1 test case ran. (0 ms total)
