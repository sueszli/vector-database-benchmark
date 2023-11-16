#include <ACPI.h>

#include <APIC.h>
#include <CString.h>
#include <IOPorts.h>
#include <List.h>
#include <Logging.h>
#include <MM/KMalloc.h>
#include <PCI.h>
#include <Paging.h>
#include <Panic.h>
#include <Timer.h>

#include <Debug.h>

namespace ACPI {
uint8_t processors[256];
int processorCount = 1;

const char* signature = "RSD PTR ";

List<apic_iso_t*>* isos;

acpi_xsdp_t* desc;
acpi_rsdt_t* rsdtHeader;
acpi_xsdt_t* xsdtHeader;
acpi_fadt_t* fadt;
pci_mcfg_table_t* mcfg = nullptr;

char oem[7];

void* FindSDT(const char* signature, int index) {
    int entries = 0;

    if (desc->revision == 2) {
        entries = (rsdtHeader->header.length - sizeof(acpi_header_t)) / sizeof(uint64_t); // ACPI 2.0
    } else {
        entries = (rsdtHeader->header.length - sizeof(acpi_header_t)) / sizeof(uint32_t); // ACPI 1.0
    }

    auto getEntry = [](unsigned index) -> uintptr_t { // This will handle differences in ACPI revisions
        if (desc->revision == 2) {
            return xsdtHeader->tables[index];
        } else {
            return rsdtHeader->tables[index];
        }
    };

    int _index = 0;

    if (memcmp("DSDT", signature, 4) == 0)
        return (void*)Memory::GetIOMapping(fadt->dsdt);

    for (int i = 0; i < entries; i++) {
        acpi_header_t* h = (acpi_header_t*)Memory::GetIOMapping(getEntry(i));
        if (memcmp(h->signature, signature, 4) == 0 && _index++ == index)
            return h;
    }

    // No SDT found
    return NULL;
}

int ReadMADT() {
    void* madt = FindSDT("APIC", 0);

    if (!madt) {
        Log::Error("Could Not Find MADT");
        return 1;
    }

    acpi_madt_t* madtHeader = (acpi_madt_t*)madt;
    uintptr_t madtEnd = reinterpret_cast<uintptr_t>(madt) + madtHeader->header.length;
    uintptr_t madtEntry = reinterpret_cast<uintptr_t>(madt) + sizeof(acpi_madt_t);

    while (madtEntry < madtEnd) {
        acpi_madt_entry_t* entry = (acpi_madt_entry_t*)madtEntry;

        switch (entry->type) {
        case 0: {
            apic_local_t* localAPIC = (apic_local_t*)entry;

            if (((apic_local_t*)entry)->flags & 0x3) {
                if (localAPIC->apicID == 0)
                    break; // Found the BSP

                processors[processorCount++] = localAPIC->apicID;

                if (debugLevelACPI >= DebugLevelVerbose)
                    Log::Info("[ACPI] Found Processor, APIC ID: %d", localAPIC->apicID);
            }
        } break;
        case 1: {
            apic_io_t* ioAPIC = (apic_io_t*)entry;

            if (debugLevelACPI >= DebugLevelVerbose)
                Log::Info("[ACPI] Found I/O APIC, Address: %x", ioAPIC->address);

            if (!ioAPIC->gSIB)
                APIC::IO::SetBase(ioAPIC->address);
        } break;
        case 2: {
            apic_iso_t* interruptSourceOverride = (apic_iso_t*)entry;
            isos->add_back(interruptSourceOverride);
        } break;
        case 4:
            if (debugLevelACPI >= DebugLevelVerbose) {
                apic_nmi_t* nonMaskableInterrupt = (apic_nmi_t*)entry;
                Log::Info("[ACPI] Found NMI, LINT #%d", nonMaskableInterrupt->lINT);
            }
            break;
        case 5:
            // apic_local_address_override_t* addressOverride = (apic_local_address_override_t*)entry;
            break;
        default:
            Log::Error("Invalid MADT Entry, Type: %d", entry->type);
            break;
        }

        madtEntry += entry->length;
    }

    // Log::Info("[ACPI] System Contains %d Processors", processorCount);

    return 0;
}

void Init() {
    if (desc) {
        goto success; // Already found
    }

    for (int i = 0; i <= 0x7BFF; i += 16) { // Search first KB for RSDP, the RSDP is aligned on a 16 byte boundary
        if (memcmp((void*)Memory::GetIOMapping(i), signature, 8) == 0) {
            desc = ((acpi_xsdp_t*)Memory::GetIOMapping(i));

            goto success;
        }
    }

    for (int i = 0x80000; i <= 0x9FFFF; i += 16) { // Search further for RSDP
        if (memcmp((void*)Memory::GetIOMapping(i), signature, 8) == 0) {
            desc = ((acpi_xsdp_t*)Memory::GetIOMapping(i));

            goto success;
        }
    }

    for (int i = 0xE0000; i <= 0xFFFFF; i += 16) { // Search further for RSDP
        if (memcmp((void*)Memory::GetIOMapping(i), signature, 8) == 0) {
            desc = ((acpi_xsdp_t*)Memory::GetIOMapping(i));

            goto success;
        }
    }

    {
        const char* panicReasons[]{"System not ACPI Complaiant."};
        KernelPanic(panicReasons, 1);
        // return;
    }

success:

    isos = new List<apic_iso_t*>();

    if (desc->revision == 2) {
        rsdtHeader = ((acpi_rsdt_t*)Memory::GetIOMapping(desc->xsdt));
        xsdtHeader = ((acpi_xsdt_t*)Memory::GetIOMapping(desc->xsdt));
    } else {
        rsdtHeader = ((acpi_rsdt_t*)Memory::GetIOMapping(desc->rsdt));
    }

    memcpy(oem, rsdtHeader->header.oem, 6);
    oem[6] = 0; // Zero OEM String

    if (debugLevelACPI >= DebugLevelNormal) {
        Log::Info("[ACPI] Revision: %d", desc->revision);
    }

    fadt = reinterpret_cast<acpi_fadt_t*>(FindSDT("FACP", 0));

    asm("cli");

    ReadMADT();
    mcfg = reinterpret_cast<pci_mcfg_table_t*>(FindSDT("MCFG", 0)); // Attempt to find MCFG table for PCI

    asm("sti");
}

void SetRSDP(acpi_xsdp_t* p) { desc = reinterpret_cast<acpi_xsdp_t*>(p); }

} // namespace ACPI
