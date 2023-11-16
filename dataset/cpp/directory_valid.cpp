#include <xtd/xtd.tunit>
#include <stdexcept>

using namespace xtd;
using namespace xtd::io;
using namespace xtd::tunit;

namespace unit_tests {
  // The class test must be declared with test_class_ helper.
  class test_class_(test) {
  public:
    void test_method_(test_case1) {
      directory_valid::exists(path::get_temp_path());
    }
    
    void test_method_(test_case2) {
      directory_valid::exists(path::combine("unknown root folder", "unknown sub folder", "unknown xtd example folder"));
    }
    
    void test_method_(test_case3) {
      directory_valid::are_equal(directory_info(environment::get_folder_path(environment::special_folder::my_music)), directory_info(environment::get_folder_path(environment::special_folder::my_videos)));
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
//     Expected: directory exists
//     But was:  "unknown root folder/unknown sub folder/unknown xtd example folder"
//     Stack Trace: in |---OMITTED---|/directory_valid.cpp:16
//   FAILED  test.test_case3 (0 ms total)
//     Expected: "|---OMITTED---|/Music"
//     But was:  "|---OMITTED---|/Movies"
//     Stack Trace: in |---OMITTED---|/directory_valid.cpp:20
//
// Test results:
//   SUCCEED 1 test.
//   FAILED  2 tests.
// End 3 tests from 1 test case ran. (0 ms total)
