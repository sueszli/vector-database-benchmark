
#pragma once
#ifndef HW_MSI_HPP
#define HW_MSI_HPP

#include <cstdint>
#include <cassert>
#include <vector>

namespace hw {

  class PCI_Device;

  struct msix_entry
  {
    uint32_t vector_ctl;
    uint32_t msg_data;
    uint32_t msg_upper_addr;
    uint32_t msg_addr;
  };
  struct msix_pba
  {
    // qword pending bits 0-63 etc.
    uint64_t pending[0];
  };

  struct msix_t
  {
    msix_t(PCI_Device&, uint32_t capoff);

    // initialize msi-x tables for device
    void mask_entry(size_t);
    void unmask_entry(size_t);
    void zero_entry(size_t);
    // enable one (cpu, vector) entry for this device
    uint16_t setup_vector(uint8_t cpu, uint8_t vector);
    // redirect MSI-X vector to another CPU
    void redirect_vector(uint16_t idx, uint8_t cpu, uint8_t vector);

    size_t vectors() const noexcept {
      return used_vectors.size();
    }

    void reset_pba_bit(size_t vec);

  private:
    PCI_Device& dev;
    uintptr_t table_addr = 0;
    uintptr_t pba_addr   = 0;
    std::vector<bool> used_vectors;

    inline auto* get_entry(size_t idx)
    {
      assert(idx < vectors());
      return ((msix_entry*) table_addr) + idx;
    }
    inline uintptr_t get_entry(size_t idx, size_t offs);
    uintptr_t get_pba(size_t offs);

    // get physical address of BAR
    uintptr_t get_bar_paddr(size_t offset);
  };

}

#endif
