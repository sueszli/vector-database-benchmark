#include "asmcode.h"
#include "debug.h"
#include "mmu.h"
#include "mem.h"

//TODO: Read breakpoints, alignment checks

#if defined(NO_TRANSLATION)
void flush_translations() {}
#endif

uint32_t FASTCALL read_word(uint32_t addr)
{
    uintptr_t entry = *(uintptr_t*)(addr_cache + ((addr >> 10) << 1));

    //If the sum doesn't contain the address directly
    if(unlikely(entry & AC_FLAGS))
    {
        if(entry & AC_INVALID) //Invalid entry
        {
            addr_cache_miss(addr, false, data_abort);
            return read_word(addr);
        }
        else //Physical address
        {
            entry &= ~AC_FLAGS;
            entry += addr;
            return mmio_read_word(entry);
        }
    }

    entry += addr;

    return *(uint32_t*)entry;
}

uint32_t FASTCALL read_byte(uint32_t addr)
{
    uintptr_t entry = *(uintptr_t*)(addr_cache + ((addr >> 10) << 1));

    //If the sum doesn't contain the address directly
    if(unlikely(entry & AC_FLAGS))
    {
        if(entry & AC_INVALID) //Invalid entry
        {
            addr_cache_miss(addr, false, data_abort);
            return read_byte(addr);
        }
        else //Physical address
        {
            entry &= ~AC_FLAGS;
            entry += addr;
            return mmio_read_byte(entry);
        }
    }

    entry += addr;

    return *(uint8_t*)entry;
}

uint32_t FASTCALL read_half(uint32_t addr)
{
    addr &= ~1;
    uintptr_t entry = *(uintptr_t*)(addr_cache + ((addr >> 10) << 1));

    //If the sum doesn't contain the address directly
    if(unlikely(entry & AC_FLAGS))
    {
        if(entry & AC_INVALID) //Invalid entry
        {
            addr_cache_miss(addr, false, data_abort);
            return read_half(addr);
        }
        else //Physical address
        {
            entry &= ~AC_FLAGS;
            entry += addr;
            return mmio_read_half(entry);
        }
    }

    entry += addr;

    return *(uint16_t*)entry;
}

void FASTCALL write_byte(uint32_t addr, uint32_t value)
{
    uintptr_t entry = *(uintptr_t*)(addr_cache + ((addr >> 10) << 1) + 1);

    //If the sum doesn't contain the address directly
    if(unlikely(entry & AC_FLAGS))
    {
        if(entry & AC_INVALID) //Invalid entry
        {
            addr_cache_miss(addr, true, data_abort);
            return write_byte(addr, value);
        }
        else //Physical address
        {
            entry &= ~AC_FLAGS;
            entry += addr;
            return mmio_write_byte(entry, value);
        }
    }

    entry += addr;

    if(RAM_FLAGS(entry & ~3) & DO_WRITE_ACTION)
        write_action((void*) entry);
    *(uint8_t*)entry = value;
}

void FASTCALL write_half(uint32_t addr, uint32_t value)
{
    addr &= ~1;
    uintptr_t entry = *(uintptr_t*)(addr_cache + ((addr >> 10) << 1) + 1);

    //If the sum doesn't contain the address directly
    if(unlikely(entry & AC_FLAGS))
    {
        if(entry & AC_INVALID) //Invalid entry
        {
            addr_cache_miss(addr, true, data_abort);
            return write_half(addr, value);
        }
        else //Physical address
        {
            entry &= ~AC_FLAGS;
            entry += addr;
            return mmio_write_half(entry, value);
        }
    }

    entry += addr;

    if(RAM_FLAGS(entry & ~3) & DO_WRITE_ACTION)
        write_action((void*) entry);
    *(uint16_t*)entry = value;
}

void FASTCALL write_word(uint32_t addr, uint32_t value)
{
    uintptr_t entry = *(uintptr_t*)(addr_cache + ((addr >> 10) << 1) + 1);

    //If the sum doesn't contain the address directly
    if(unlikely(entry & AC_FLAGS))
    {
        if(entry & AC_INVALID) //Invalid entry
        {
            addr_cache_miss(addr, true, data_abort);
            return write_word(addr, value);
        }
        else //Physical address
        {
            entry &= ~AC_FLAGS;
            entry += addr;
            return mmio_write_word(entry, value);
        }
    }
    entry += addr;

    if(RAM_FLAGS(entry & ~3) & DO_WRITE_ACTION)
        write_action((void*) entry);
    *(uint32_t*)entry = value;
}
