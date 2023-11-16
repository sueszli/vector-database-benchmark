/* RUN: %{execute}%s

   Exercise buffer size-like member functions
*/

#include <CL/sycl.hpp>

#include <iostream>

#include <catch2/catch_test_macros.hpp>

using namespace cl::sycl;

TEST_CASE("size-like member functions", "[buffer]") {
  std::vector<int> v(10);
  size_t bufferSize = std::distance(v.begin(), v.end());
  std::shared_ptr<int> ptr { new int[bufferSize],
                             std::default_delete<int[]>() };
  std::copy(v.begin(), v.end(), ptr.get());
  buffer<int, 1> buf(ptr, range<1>(bufferSize));

  if (buf.get_count() != v.size())
  {
    std::cout << "v.size() = " << v.size() << std::endl;
    std::cout << "buf.get_count() = " << buf.get_count() << std::endl;
    assert(false);
  }
  cl::sycl::range<1> result { v.size() };
  if (buf.get_range() != result)
  {
    std::cout << "range { v.size } = " << result.get(0) << std::endl;
    std::cout << "buf.get_range() = " << buf.get_range().get(0) << std::endl;
    assert(false);
  }
}
