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
      string_assume::starts_with("val", s);
    }
    
    void test_method_(test_case_aborted) {
      auto s = L"value";
      string_assume::starts_with(L"lue", s);
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
//     Expected: string starting with "lue"
//     But was:  "value"
//     Stack Trace: in |---OMITTED---|/string_assume_starts_with.cpp:16
//
// Test results:
//   SUCCEED 1 test.
//   ABORTED 1 test.
// End 2 tests from 1 test case ran. (0 ms total)
