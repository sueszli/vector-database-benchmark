#include <xtd/xtd.tunit>
#include <string>

using namespace std;
using namespace std::string_literals;
using namespace xtd::tunit;

namespace unit_tests {
  class test_class_(test) {
  public:
    void test_method_(test_case_succeed) {
      auto s = "value"s;
      string_assert::ends_with("lue", s);
    }
    
    void test_method_(test_case_failed) {
      auto s = L"value";
      string_assert::ends_with(L"val", s);
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
//     Expected: string ending with "val"
//     But was:  "value"
//     Stack Trace: in |---OMITTED---|/string_assert_ends_with.cpp:16
//
// Test results:
//   SUCCEED 1 test.
//   FAILED  1 test.
// End 2 tests from 1 test case ran. (0 ms total)
