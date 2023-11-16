#include <xtd/xtd.tunit>
#include <stdexcept>
#include <string>

using namespace std;
using namespace xtd::tunit;

namespace unit_tests {
  class test_class_(test) {
  public:
    void test_method_(test_case_succeed) {
      invalid_argument except("invalid argument");
      valid::is_not_instance_of<bad_cast>(except);
    }
    
    void test_method_(test_case_failed) {
      invalid_argument except("invalid argument");
      valid::is_not_instance_of<logic_error>(except);
    }
  };
}

int main() {
  return console_unit_test().run();
}

// This code produces the following output:
//
// Start 2 tests from 1 test case
// Run tests:
//   SUCCEED test.test_case_succeed (0 ms total)
//   FAILED  test.test_case_failed (0 ms total)
//     Expected: not instance of <std::logic_error>
//     But was:  <std::invalid_argument>
//     Stack Trace: in |---OMITTED---|/valid_is_not_instance_of.cpp:16
//
// Test results:
//   SUCCEED 1 test.
//   FAILED  1 test.
// End 2 tests from 1 test case ran. (0 ms total)
