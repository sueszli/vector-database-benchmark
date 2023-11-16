
#include <cassert>
#include <cstdint>
#include <kprint>
#include <util/crc32.hpp>
#include <kernel/elf.hpp>
#include <os.hpp>

//#define ENABLE_CRC_RO

// Global constructors
static int gconstr_value = 0;
__attribute__((constructor, used))
static void self_test_gconstr() {
  gconstr_value = 1;
}

#ifdef ENABLE_CRC_RO
// NOTE: crc_ro MUST NOT be initialized to zero
static uint32_t crc_ro = CRC32_BEGIN();

static uint32_t generate_ro_crc() noexcept
{
  extern char _TEXT_START_;
  extern char _RODATA_END_;
  return crc32_fast(&_TEXT_START_, &_RODATA_END_ - &_TEXT_START_);
}
#endif

extern "C"
void __init_sanity_checks() noexcept
{
#ifdef ENABLE_CRC_RO
  // generate checksum for read-only portions of kernel
  crc_ro = generate_ro_crc();
#endif
}

extern "C"
void kernel_sanity_checks()
{
#ifdef ENABLE_CRC_RO
  // verify checksum of read-only portions of kernel
  uint32_t new_ro = generate_ro_crc();

  if (crc_ro != new_ro) {
    kprintf("CRC mismatch %#x vs %#x\n", crc_ro, new_ro);
    os::panic("Sanity checks: CRC of kernel read-only area failed");
  }
#endif

  // verify that Elf symbols were not overwritten
  bool symbols_verified = Elf::verify_symbols();
  if (!symbols_verified)
    os::panic("Sanity checks: Consistency of Elf symbols and string areas");

  // global constructor self-test
  if (gconstr_value != 1) {
    kprintf("Sanity checks: Global constructors not working (or modified during run-time)!\n");
    os::panic("Sanity checks: Global constructors verification failed");
  }

}
