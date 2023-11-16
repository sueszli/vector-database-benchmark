/* RUN: %{execute}%s
 */

#define TRISYCL_NO_BARRIER

#include <CL/sycl.hpp>

#include <catch2/catch_test_macros.hpp>

using namespace cl::sycl;

template <int Dimensions, class kernel_name>
void generic_par_for_wg(range<Dimensions> k_range,
                        range<Dimensions> workgroup_size) {
  queue my_queue;

  // the product of all Dimensions e.g. 10*10*10 for {10,10,10}
  auto linr_size = k_range.size(), linwg_size = workgroup_size.size();

  // these will simply have the group, local and global linear ids assigned to
  // them
  auto group_lin = buffer<int>(linr_size / linwg_size);
  auto loc_lin = buffer<int>(linr_size);
  auto gl_lin = buffer<int>(linr_size);

  // calculation to attain linr_size accross indices using values stored in
  // gl_lin and indexed into by linear_id calls from the second kernel
  auto loc_largest = buffer<int>(linr_size);
  auto gl_calc = buffer<int>(linr_size);

  my_queue.submit([&](handler &cgh) {
    auto group_lin_acc = group_lin.get_access<access::mode::write>(cgh);
    auto loc_lin_acc = loc_lin.get_access<access::mode::write>(cgh);
    auto gl_lin_acc = gl_lin.get_access<access::mode::read_write>(cgh);
    auto loc_largest_acc =
        loc_largest.get_access<access::mode::read_write>(cgh);
    auto gl_calc_acc = gl_calc.get_access<access::mode::read_write>(cgh);

    cgh.parallel_for_work_group<kernel_name>(
        nd_range<Dimensions>(k_range, workgroup_size),
        [=](group<Dimensions> group) {
          group_lin_acc[group.get_linear_id()] = group.get_linear_id();

          group.parallel_for_work_item([=](h_item<Dimensions> tile) {
            loc_lin_acc[tile.get_global_linear_id()] =
                tile.get_local_linear_id();
            gl_lin_acc[tile.get_global_linear_id()] =
                tile.get_global_linear_id();

            auto global =
                group.get_linear_id() * linwg_size + tile.get_local_linear_id();
            gl_calc_acc[global] = loc_largest_acc[global] = global;
          });

          group.parallel_for_work_item([=](h_item<Dimensions> tile) {
            auto global =
                group.get_linear_id() * linwg_size + tile.get_local_linear_id();

            loc_largest_acc[global] +=
                gl_calc_acc[((group.get_linear_id() + 1) * linwg_size) - 1 -
                            tile.get_local_linear_id()];
          });
        });
  });

  auto loc_lin_out = loc_lin.get_access<access::mode::read>();
  auto group_lin_out = group_lin.get_access<access::mode::read>();
  auto gl_lin_out = gl_lin.get_access<access::mode::read>();

  auto gl_calc_out = gl_calc.get_access<access::mode::read>();
  auto loc_largest_out = loc_largest.get_access<access::mode::read>();

  for (int i = 0; i < linr_size / linwg_size; ++i) {
    REQUIRE(group_lin_out[i] == i); // group id
  }

  int largest_local = 0;
  for (int i = 0; i < linr_size; ++i) {
    if (i % linwg_size == 0)
      largest_local = gl_calc_out[i] + gl_calc_out[(i + linwg_size - 1)];
    REQUIRE(gl_lin_out[i] == i);                            // w1 global id
    REQUIRE(loc_lin_out[i] == loc_lin_out[i] % linwg_size); // local id
    REQUIRE(loc_largest_out[i] == largest_local);
  }

  /* We must wait for for the queue to finish as none of buffer's destruction
     is blocking.
   */
  my_queue.wait();
}

TEST_CASE("new hierarchical", "[parallel_for]") {
  generic_par_for_wg<1, class par_1d>({10}, {2});
  generic_par_for_wg<2, class par_2d_square>({10, 10}, {2, 2});
  generic_par_for_wg<2, class par_2d_square>({12, 12}, {4, 4});
  generic_par_for_wg<2, class par_2d_rect>({12, 6}, {4, 2});
  generic_par_for_wg<3, class par_3d_square>({10, 10, 10}, {2, 2, 2});
  generic_par_for_wg<3, class par_3d_rect>({12, 8, 16}, {3, 2, 4});
  generic_par_for_wg<1, class par_1d>({1000}, {20});
  generic_par_for_wg<2, class par_2d_square>({100, 100}, {20, 20});
  generic_par_for_wg<2, class par_2d_rect>({120, 120}, {10, 20});
  generic_par_for_wg<2, class par_2d_rect>({120, 60}, {40, 10});
  generic_par_for_wg<3, class par_3d_square>({100, 100, 100}, {20, 20, 20});
  generic_par_for_wg<3, class par_3d_rect>({150, 200, 150}, {15, 20, 10});
  generic_par_for_wg<3, class par_3d_rect>({150, 200, 100}, {15, 20, 4});
}
