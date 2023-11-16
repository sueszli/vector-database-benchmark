///////////////////////////////////////////////////////////////////////////////
//  Copyright Christopher Kormanyos 2020.
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <mcal_memory_sram.h>
#include <mcal_memory_sram_microchip_23lc1024.h>
#include <mcal_spi.h>

namespace
{
  constexpr mcal_sram_uintptr_t mcal_sram_chip_byte_size_value = UINT32_C(0x00020000);

  mcal::memory::sram::memory_sram_microchip_23lc1024& mcal_memory_sram_device()
  {
    static mcal::memory::sram::memory_sram_microchip_23lc1024 device(mcal::spi::spi_channels());

    return device;
  }
}

extern "C"
uint8_t mcal_memory_sram_read_byte(const mcal_sram_uintptr_t src_addr)
{
  mcal_memory_sram_device().select_channel(static_cast<std::size_t>(src_addr / mcal_sram_chip_byte_size_value));

  const mcal_sram_uintptr_t physical_src_addr = mcal_sram_uintptr_t(src_addr % mcal_sram_chip_byte_size_value);

  uint8_t byte_to_read = 0U;

  const bool read_is_ok = mcal_memory_sram_device().read_byte(physical_src_addr, byte_to_read);

  static_cast<void>(read_is_ok);

  return byte_to_read;
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
  mcal_memory_sram_device().select_channel(static_cast<std::size_t>(src_addr / mcal_sram_chip_byte_size_value));

  const mcal_sram_uintptr_t physical_src_addr = mcal_sram_uintptr_t(src_addr % mcal_sram_chip_byte_size_value);

  uint32_t dword_to_read = 0U;

  const bool read_is_ok = mcal_memory_sram_device().read_dword(physical_src_addr, dword_to_read);

  static_cast<void>(read_is_ok);

  return dword_to_read;
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
  mcal_memory_sram_device().select_channel(static_cast<std::size_t>(dest_addr / mcal_sram_chip_byte_size_value));

  const mcal_sram_uintptr_t physical_dest_addr = mcal_sram_uintptr_t(dest_addr % mcal_sram_chip_byte_size_value);

  const std::uint8_t byte_to_write = src_value;

  const bool write_is_ok = mcal_memory_sram_device().write_byte(physical_dest_addr, byte_to_write);

  static_cast<void>(write_is_ok);
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
  mcal_memory_sram_device().select_channel(static_cast<std::size_t>(dest_addr / mcal_sram_chip_byte_size_value));

  const mcal_sram_uintptr_t physical_dest_addr = mcal_sram_uintptr_t(dest_addr % mcal_sram_chip_byte_size_value);

  const std::uint32_t dword_to_write = src_value;

  const bool write_is_ok = mcal_memory_sram_device().write_dword(physical_dest_addr, dword_to_write);

  static_cast<void>(write_is_ok);
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
