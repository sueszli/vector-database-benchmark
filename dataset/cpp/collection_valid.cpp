#include <xtd/xtd.tunit>

using namespace std;
using namespace xtd::tunit;

namespace unit_tests {
  // The class test must be declared with test_class_ helper.
  class test_class_(test) {
  public:
    void test_method_(test_case1) {
      collection_valid::is_empty(vector<int> {});
    }
    
    void test_method_(test_case2) {
      collection_valid::is_empty(vector<int> {1, 2, 3});
    }
    
    void test_method_(test_case3) {
      collection_valid::is_ordered(vector<int> {1, 3, 2});
    }
  };
}

auto main()->int {
  return console_unit_test().run();
}

// This code can produce the following output:
//
// Start 3 tests from 1 test case
// Run tests:
//   SUCCEED test.test_case1 (0 ms total)
//   FAILED  test.test_case2 (0 ms total)
//     Expected: <empty>
//     But was:  < 1, 2, 3 >
//     Stack Trace: in |---OMITTED---|/collection_valid.cpp:14
//   FAILED  test.test_case3 (0 ms total)
//     Expected: < 1, 2, 3 >
//     But was: < 1, 3, 2 >
//     Stack Trace: in |---OMITTED---|/collection_valid.cpp:18
//
// Test results:
//   SUCCEED 1 test.
//   FAILED  2 tests.
// End 3 tests from 1 test case ran. (0 ms total)
