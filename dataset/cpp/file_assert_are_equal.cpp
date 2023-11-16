#include <xtd/xtd.tunit>

using namespace std;
using namespace xtd::tunit;

namespace unit_tests {
  class test_class_(test) {
  public:
    void test_method_(test_case_succeed) {
      auto is = istringstream {"xtd::tunit::file_assert::are_equal example."};
      file_assert::are_equal(istringstream {"xtd::tunit::file_assert::are_equal example."}, is);
    }
    
    void test_method_(test_case_failed) {
      auto is = istringstream {"xtd::tunit::file_assert::are_equal example."};
      file_assert::are_equal(istringstream {"xtd::tunit::file_assert::are_equel example."}, is);
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
//     Expected: istream at offset 32 value e
//     But was:  a
//     Stack Trace: in |---OMITTED---|/file_assert_are_equal.cpp:14
//
// Test results:
//   SUCCEED 1 test.
//   FAILED  1 test.
// End 2 tests from 1 test case ran. (0 ms total)
