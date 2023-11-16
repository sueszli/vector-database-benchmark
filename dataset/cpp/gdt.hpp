
#pragma once
#ifndef X86_GDT_HPP
#define X86_GDT_HPP

#include <cstddef>
#include <cstdint>
#include <cassert>
#include "cpu.hpp"

namespace x86
{
struct gdt_desc
{
  uint16_t  size;
  uintptr_t offset;
} __attribute__((packed));

struct gdt_entry
{
  uint32_t limit_lo  : 16;
  uint32_t base_lo   : 24;
  uint32_t access    : 8;
  uint32_t limit_hi  : 4;
  uint32_t flags     : 4;
  uint32_t base_hi   : 8;
} __attribute__((packed));

struct GDT
{
  static const int MAX_ENTRIES = 6;

  static void reload_gdt(GDT& base) noexcept;

#if defined(ARCH_x86)
  static inline void set_fs(uint16_t entry) noexcept {
    asm volatile("movw %%ax, %%fs" : : "a"(entry * 0x8));
  }
  static inline void set_gs(uint16_t entry) noexcept {
    asm volatile("movw %%ax, %%gs" : : "a"(entry * 0x8));
  }
#else
  static_assert(false, "Error: missing x86 arch");
#endif

  GDT() {
    desc.size   = 0;
    desc.offset = (uintptr_t) &entry[0];
  }

  void initialize() noexcept;

  gdt_entry& create() {
    assert(count < MAX_ENTRIES);
    auto& newent = entry[count++];
    desc.size = count * sizeof(gdt_entry) - 1;
    return newent;
  }

  // create new data entry and return index
  int create_data(void* base, int) noexcept;

private:
  gdt_entry& create_data(uint32_t base, int size) noexcept;

  uint16_t  count = 0;
  gdt_desc  desc;
  gdt_entry entry[MAX_ENTRIES];
};
}

#endif
