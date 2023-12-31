/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Codeplay's ComputeCpp SDK
 *
 *  vector-addition-tiled.cpp
 *
 *  Description:
 *    Samples of tiled vector addition kernels
 *
 **************************************************************************/

#include <iostream>

#include <CL/sycl.hpp>
#include <SYCL/codeplay.hpp>

using namespace cl;

/* Loads float data from a and b into tile pointers. In this sample, the
 * tile pointers point to local memory. */
void loadTiles(const float* a, const float* b, float* tile1, float* tile2,
               size_t id, size_t tile_i) {
  tile1[tile_i] = a[id];
  tile2[tile_i] = b[id];
}

/* Sums the values from local memory. */
void vecAdd(float* tile1, float* tile2, size_t tile_i) {
  tile1[tile_i] += tile2[tile_i];
}

/* In this sample, loads from local to store back to global memory. */
void storeTile(float* c, float* tile1, size_t id, size_t tile_i) {
  c[id] = tile1[tile_i];
}

class TiledVecAdd;
class TiledVecAddDMA;

/* First computes sum via normal tiled load, then by DMA. */
int main(int argc, char* argv[]) {
  constexpr const size_t N = 128000;  // this is the total vector size
  constexpr const size_t T = 32;      // this is the tile size
  static constexpr auto read = sycl::access::mode::read;
  static constexpr auto write = sycl::access::mode::write;
  static constexpr auto rw = sycl::access::mode::read_write;
  static constexpr auto dwrite = sycl::access::mode::discard_write;
  using local_acc = sycl::accessor<float, 1, rw, sycl::access::target::local>;
  const sycl::range<1> VecSize{N};
  const sycl::range<1> TileSize{T};

  sycl::queue myQueue;
  auto context = myQueue.get_context();
  namespace prop = sycl::codeplay::property;

  sycl::buffer<float> bufA{VecSize};
  sycl::buffer<float> bufB{VecSize};
  sycl::buffer<float> bufC{VecSize};
  sycl::property_list props{sycl::property::buffer::context_bound(context),
                            prop::buffer::use_onchip_memory(prop::prefer)};
  sycl::buffer<float, 1> onChipTile{VecSize, props};

  {
    auto h_a = bufA.get_access<dwrite>();
    auto h_b = bufB.get_access<dwrite>();
    for (auto i = 0u; i < N; i++) {
      h_a[i] = sin(i) * sin(i);
      h_b[i] = cos(i) * cos(i);
    }
  }

  {
    auto cg = [&](sycl::handler& h) {
      auto a = bufA.get_access<read>(h);
      auto b = bufB.get_access<read>(h);
      auto c = bufC.get_access<dwrite>(h);
      local_acc tile1(TileSize, h);
      local_acc tile2(TileSize, h);

      h.parallel_for<TiledVecAdd>(
          sycl::nd_range<1>(VecSize, TileSize), [=](sycl::nd_item<1> i) {
            loadTiles(&a[0], &b[0], &tile1[0], &tile2[0], i.get_global_id(0),
                      i.get_local_id(0));
            i.barrier();
            vecAdd(&tile1[0], &tile2[0], i.get_local_id(0));
            i.barrier();
            storeTile(&c[0], &tile1[0], i.get_global_id(0), i.get_local_id(0));
          });
    };
    myQueue.submit(cg);
  }

  {
    auto h_c = bufC.get_access<read>();
    float sum = 0.0f;
    for (auto i = 0u; i < N; i++) {
      sum += h_c[i];
    }
    std::cout << "total result: " << sum << std::endl;
  }

  {
    auto cg = [&](sycl::handler& h) {
      auto a = bufA.get_access<read>(h);
      auto b = bufB.get_access<read>(h);
      auto c = bufC.get_access<write>(h);
      local_acc tile1(TileSize, h);
      local_acc tile2(TileSize, h);

      h.parallel_for<TiledVecAddDMA>(
          sycl::nd_range<1>(VecSize, TileSize), [=](sycl::nd_item<1> i) {
            auto event1 = i.async_work_group_copy(
                tile1.get_pointer(), a.get_pointer() + i.get_global_id(0),
                TileSize[0]);
            auto event2 = i.async_work_group_copy(
                tile2.get_pointer(), b.get_pointer() + i.get_global_id(0),
                TileSize[0]);
            i.wait_for(event1, event2);
            i.barrier();
            vecAdd(&tile1[0], &tile2[0], i.get_local_id(0));
            i.barrier();
            auto event3 =
                i.async_work_group_copy(c.get_pointer() + i.get_global_id(0),
                                        tile1.get_pointer(), TileSize[0]);
            i.wait_for(event3);
            i.barrier();
          });
    };
    myQueue.submit(cg);
  }

  {
    auto h_c = bufC.get_access<read>();
    float sum = 0.0f;
    for (auto i = 0u; i < N; i++) {
      sum += h_c[i];
    }
    std::cout << "total result: " << sum << std::endl;
  }

  return 0;
}
