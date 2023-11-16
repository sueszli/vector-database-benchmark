///////////////////////////////////////////////////////////////////////////////
//  Copyright Christopher Kormanyos 2020.
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <mcal_memory_sram.h>

namespace
{
  constexpr mcal_sram_uintptr_t mcal_sram_physical_pool_size_value = UINT32_C(0x00020000);

  uintptr_t mcal_memory_sram_physical_pool_01()
  {
    static uint8_t mcal_memory_sram_physical_pool_object[mcal_sram_physical_pool_size_value];

    return reinterpret_cast<uintptr_t>(&mcal_memory_sram_physical_pool_object[0U]);
  }

  uintptr_t mcal_memory_sram_physical_pool_02()
  {
    static uint8_t mcal_memory_sram_physical_pool_object[mcal_sram_physical_pool_size_value];

    return reinterpret_cast<uintptr_t>(&mcal_memory_sram_physical_pool_object[0U]);
  }

  typedef uintptr_t(*mcal_memory_sram_physical_pool_function_type)(void);

  const mcal_memory_sram_physical_pool_function_type mcal_memory_sram_physical_pool_functions[2U] =
  {
   mcal_memory_sram_physical_pool_01,
   mcal_memory_sram_physical_pool_02
  };
}

extern "C"
uint8_t mcal_memory_sram_read_byte(const mcal_sram_uintptr_t src_addr)
{
  const mcal_sram_uintptr_t pool_index = src_addr / mcal_sram_physical_pool_size_value;

  const mcal_memory_sram_physical_pool_function_type local_pool_function = mcal_memory_sram_physical_pool_functions[pool_index];

  const uintptr_t physical_src_addr =   local_pool_function()
                                      + (src_addr % mcal_sram_physical_pool_size_value);

  const uint8_t dest = *((const uint8_t*) physical_src_addr);

  return dest;
}

extern "C"
uint16_t mcal_memory_sram_read_word(const mcal_sram_uintptr_t src_addr)
{
  uint16_t dest = 0U;

  *(((uint8_t*) &dest) + 0U) = mcal_memory_sram_read_byte(src_addr + 0U);
  *(((uint8_t*) &dest) + 1U) = mcal_memory_sram_read_byte(src_addr + 1U);

  return dest;
}

extern "C"
uint32_t mcal_memory_sram_read_dword(const mcal_sram_uintptr_t src_addr)
{
  uint32_t dest = 0U;

  *(((uint8_t*) &dest) + 0U) = mcal_memory_sram_read_byte(src_addr + 0U);
  *(((uint8_t*) &dest) + 1U) = mcal_memory_sram_read_byte(src_addr + 1U);
  *(((uint8_t*) &dest) + 2U) = mcal_memory_sram_read_byte(src_addr + 2U);
  *(((uint8_t*) &dest) + 3U) = mcal_memory_sram_read_byte(src_addr + 3U);

  return dest;
}

extern "C"
uint64_t mcal_memory_sram_read_qword(const mcal_sram_uintptr_t src_addr)
{
  uint64_t dest = 0U;

  *(((uint8_t*) &dest) + 0U) = mcal_memory_sram_read_byte(src_addr + 0U);
  *(((uint8_t*) &dest) + 1U) = mcal_memory_sram_read_byte(src_addr + 1U);
  *(((uint8_t*) &dest) + 2U) = mcal_memory_sram_read_byte(src_addr + 2U);
  *(((uint8_t*) &dest) + 3U) = mcal_memory_sram_read_byte(src_addr + 3U);
  *(((uint8_t*) &dest) + 4U) = mcal_memory_sram_read_byte(src_addr + 4U);
  *(((uint8_t*) &dest) + 5U) = mcal_memory_sram_read_byte(src_addr + 5U);
  *(((uint8_t*) &dest) + 6U) = mcal_memory_sram_read_byte(src_addr + 6U);
  *(((uint8_t*) &dest) + 7U) = mcal_memory_sram_read_byte(src_addr + 7U);

  return dest;
}

extern "C"
void mcal_memory_sram_write_byte(const uint8_t src_value, const mcal_sram_uintptr_t dest_addr)
{
  const mcal_sram_uintptr_t pool_index = dest_addr / mcal_sram_physical_pool_size_value;

  const mcal_memory_sram_physical_pool_function_type local_pool_function = mcal_memory_sram_physical_pool_functions[pool_index];

  const uintptr_t physical_dest_addr =   local_pool_function()
                                      + (dest_addr % mcal_sram_physical_pool_size_value);

  *((uint8_t*) physical_dest_addr) = src_value;
}

extern "C"
void mcal_memory_sram_write_word(const uint16_t src_value, const mcal_sram_uintptr_t dest_addr)
{
  const uint16_t local_src_value = src_value;

  mcal_memory_sram_write_byte(*(((uint8_t*) &local_src_value) + 0U), dest_addr + 0U);
  mcal_memory_sram_write_byte(*(((uint8_t*) &local_src_value) + 1U), dest_addr + 1U);
}

extern "C"
void mcal_memory_sram_write_dword(const uint32_t src_value, const mcal_sram_uintptr_t dest_addr)
{
  const uint32_t local_src_value = src_value;

  mcal_memory_sram_write_byte(*(((uint8_t*) &local_src_value) + 0U), dest_addr + 0U);
  mcal_memory_sram_write_byte(*(((uint8_t*) &local_src_value) + 1U), dest_addr + 1U);
  mcal_memory_sram_write_byte(*(((uint8_t*) &local_src_value) + 2U), dest_addr + 2U);
  mcal_memory_sram_write_byte(*(((uint8_t*) &local_src_value) + 3U), dest_addr + 3U);
}

extern "C"
void mcal_memory_sram_write_qword(const uint64_t src_value, const mcal_sram_uintptr_t dest_addr)
{
  const uint64_t local_src_value = src_value;

  mcal_memory_sram_write_byte(*(((uint8_t*) &local_src_value) + 0U), dest_addr + 0U);
  mcal_memory_sram_write_byte(*(((uint8_t*) &local_src_value) + 1U), dest_addr + 1U);
  mcal_memory_sram_write_byte(*(((uint8_t*) &local_src_value) + 2U), dest_addr + 2U);
  mcal_memory_sram_write_byte(*(((uint8_t*) &local_src_value) + 3U), dest_addr + 3U);
  mcal_memory_sram_write_byte(*(((uint8_t*) &local_src_value) + 4U), dest_addr + 4U);
  mcal_memory_sram_write_byte(*(((uint8_t*) &local_src_value) + 5U), dest_addr + 5U);
  mcal_memory_sram_write_byte(*(((uint8_t*) &local_src_value) + 6U), dest_addr + 6U);
  mcal_memory_sram_write_byte(*(((uint8_t*) &local_src_value) + 7U), dest_addr + 7U);
}
