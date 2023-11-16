/* RUN: %{execute}%s

   A simple typical FPGA-like kernel
*/
#include <CL/sycl.hpp>
#include <iostream>
#include <numeric>

#include <catch2/catch_test_macros.hpp>

using namespace cl::sycl;

constexpr size_t N = 300;
using Type = int;


TEST_CASE("single task vector increment", "[example]") {
  buffer<Type> input { N };
  buffer<Type> output { N };

  {
    auto a_input = input.get_access<access::mode::write>();
    // Initialize buffer with increasing numbers starting at 0
    std::iota(a_input.begin(), a_input.end(), 0);
  }

  // Create a queue to launch the kernel
  queue q;

  // Launch a kernel to do the summation
  q.submit([&] (handler &cgh) {
      // Get access to the data
      auto a_input = input.get_access<access::mode::read>(cgh);
      auto x_output = output.get_access<access::mode::write>(cgh);

      // A typical FPGA-style pipelined kernel
      cgh.single_task<class add>([=] {
          for (int i = 0 ; i < N; ++i)
            x_output[i] = a_input[i] + 42;
        });
    });

  // Verify the result
  auto a_input = input.get_access<access::mode::read>();
  auto a_output = output.get_access<access::mode::read>();
  for (int i = 0 ; i < input.get_count(); ++i)
    REQUIRE(a_output[i] == a_input[i] + 42);
}
