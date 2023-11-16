#include <xtd/xtd.tunit>
#include <stdexcept>

using namespace std;
using namespace xtd::io;
using namespace xtd::tunit;

namespace unit_tests {
  // The class test must be declared with test_class_ helper.
  class test_class_(test) {
  public:
    void test_method_(test_case1) {
      auto s = "A string value";
      string_assert::are_equal_ignoring_case("A STRING VALUE", s);
    }
    
    void test_method_(test_case2) {
      auto s = "A string value";
      string_assert::contains("item", s);
    }
    
    void test_method_(test_case3) {
      auto s = "A string value";
      string_assert::matches("item$", s);
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
//     Expected: string containing "item"
//     But was:  "A string value"
//     Stack Trace: in |---OMITTED---|/string_assert.cpp:17
//   FAILED  test.test_case3 (0 ms total)
//     Expected: string matching "item$"
//     But was:  "A string value"
//     Stack Trace: in |---OMITTED---|/string_assert.cpp:22
//
// Test results:
//   SUCCEED 1 test.
//   FAILED  2 tests.
// End 3 tests from 1 test case ran. (0 ms total)
