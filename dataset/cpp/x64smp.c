
// x64smp.c
// Symmetric multiprocessing

// Initialization
// Probing if smp is supported.
// + Via MP.
// + Via ACPI.
// #todo
// This is a work in progress.
// >> we need to disable PIC, and init the first AP.
// The routines are found in another place. 
// see: apic/ioapic routines.
// 2022: Created by Fred Nora.
// see:
// https://en.wikipedia.org/wiki/Symmetric_multiprocessing

#include <kernel.h>

/*
  The AP initialization. AP startup.

  + After you've gathered the information, you'll need to 
  + disable the PIC and 
  + prepare for I/O APIC. 
  + You also need to setup BSP's local APIC. 
  + Then, startup the APs using SIPIs. 
*/


// see: mp.h
struct rsdp_d *rsdp;
struct xsdp_d *xsdp;

struct rsdt_d *rsdt;
struct xsdt_d *xsdt;


// see: mp.h
struct mp_floating_pointer_structure_d *MPTable;
// see: mp.h
struct mp_configuration_table_d *MPConfigurationTable;
// see: mp.h
struct smp_info_d smp_info;

// ------------------------------------
static int __x64_probe_smp_via_acpi(void);
static int __x64_probe_smp_via_mptable(void);
// ------------------------------------

static int __x64_probe_smp_via_acpi(void)
{
// After you've gathered the information, 
// you'll need to disable the PIC and prepare for I/O APIC. 
// You also need to setup BSP's local APIC. 
// Then, startup the APs using SIPIs.

// You should be able to find a MADT table 
// in the RSDT table or in the XSDT table.

// #todo
// Probe the acpi table.
// + 'RSDP signature'


// 0x040E - The base address.
// Get a short value.
    unsigned short *bda = (unsigned short*) BDA_BASE;
    unsigned long ebda_address=0;
    register int i=0;
    unsigned char *p;
// Signature elements.
    unsigned char c1=0;
    unsigned char c2=0;
    unsigned char c3=0;
    unsigned char c4=0;
    unsigned char c5=0;
    unsigned char c6=0;
    unsigned char c7=0;
    unsigned char c8=0;


//
// Probe ebda address at bda base.
//

    printf("EBDA short Address: %x\n", bda[0] ); 
    ebda_address = (unsigned long) ( bda[0] << 4 );
    ebda_address = (unsigned long) ( ebda_address & 0xFFFFFFFF);
    printf("EBDA Address: %x\n", ebda_address ); 

// base
// between 0xF0000 and 0xFFFFF.
// #todo: filter
    p = ebda_address;

// The signature was found?
    static int Found = FALSE;

// Probe for the signature. // "RSD PTR "
    int max = (int) (0xFFFFF - ebda_address);
    for (i=0; i<max; i++){
        c1 = p[i+0];
        c2 = p[i+1];
        c3 = p[i+2];
        c4 = p[i+3];
        c5 = p[i+4];
        c6 = p[i+5];
        c7 = p[i+6];
        c8 = p[i+7];

        // "RSD PTR "
        if ( c1 == 'R' && 
             c2 == 'S' && 
             c3 == 'D' && 
             c4 == ' ' &&
             c5 == 'P' &&
             c6 == 'T' &&
             c7 == 'R' &&
             c8 == ' '  )
        {
            printf (":: Found [RSD PTR ] at index %d. :)\n",i);
            Found=TRUE;
            break;
        }
    };

// Signature not found.
    if (Found != TRUE){
        printf("__x64_probe_smp_via_acpi: [RSD PTR ] wasn't found!\n");
        goto fail;
    }

// Get address
    unsigned long __rsdp_Pointer = 
        (unsigned long) (ebda_address + i);

// Save the address
    smp_info.RSD_PTR_address = (unsigned long) __rsdp_Pointer;

// --------------------------------------------------------------
// To find the RSDT you need first to locate and check the RSDP, 
// then use the RsdtPointer for ACPI Version < 2.0 an XsdtPointer for any other case. 

// The root table.
    rsdp = (struct rsdp_d *) __rsdp_Pointer;  // 1.0
    xsdp = (struct xsdp_d *) __rsdp_Pointer;  // 2.0


// #debug ok
    printf("RSDP signature: \n");
    printf ("%c %c %c %c\n",
        rsdp->Signature[0],
        rsdp->Signature[1],
        rsdp->Signature[2],
        rsdp->Signature[3] );
    printf ("%c %c %c %c\n",
        rsdp->Signature[4],
        rsdp->Signature[5],
        rsdp->Signature[6],
        rsdp->Signature[7] );
    // #debug
    // #breakpoint
    //while(1){ asm("hlt"); }

    unsigned long __rsdt_address=0;
    unsigned long __xsdt_address=0;

    // Use xsdp
    if (rsdp->Revision == 0x02){
        printf("ACPI version 2.0\n");
        __rsdt_address = 
            (unsigned long) (rsdp->RsdtAddress & 0xFFFFFFFF);
        __xsdt_address = 
            (unsigned long) (xsdp->XsdtAddress & 0xFFFFFFFF);
        
        // #breakpoint
        //panic("x64smp.c: Revision 2.0 #todo\n");
        printf("x64smp.c: Revision 2.0 #todo\n");
        goto do_lapic;

    // Use rsdp
    }else if (rsdp->Revision == 0){
        printf("ACPI version 1.0\n");
        __rsdt_address = 
            (unsigned long) (rsdp->RsdtAddress & 0xFFFFFFFF);
        __xsdt_address = 0;
    }else{
        panic("x64smp.c: acpi revision\n");
    };

// Mapping the rsdt address.
// Used in both revisions.
    unsigned long address_pa = 0;
    unsigned long address_va = 0;
    int mapStatus = -1;


// ++
// ----------------------------------
// Revision 1.0.

// ---------------------------------------
// RSDT (Root System Description Table)
// This table contains pointers to all the other System Description Tables.
// Validating the RSDT:
// You only need to sum all the bytes in the table and compare the result to 0 (mod 0x100). 
// https://wiki.osdev.org/RSDT

    if (rsdp->Revision == 0x02){
        //panic("x64smp.c: #todo 2.0\n");
        printf("x64smp.c: #todo 2.0\n");
        goto do_lapic;
    }else if (rsdp->Revision == 0){
        // Print the address we have.
        //printf("rsdt address: %x \n", __rsdt_address);
        //while(1){ asm("hlt"); }

        // Mapping the rsdt address.
        address_pa = __rsdt_address;
        address_va = RSDT_VA;
        // OUT: 0=ok | -1=fail
        // see: pages.c
        mapStatus = (int) mm_map_2mb_region(address_pa,address_va);
        if (mapStatus != 0){
            panic("mapStatus\n");
        }
        // Now we have a valid pointer.
        rsdt = (struct rsdt_d *) RSDT_VA;
        // #debug
        printf("RSDT signature: \n");
        printf ("%c %c %c %c\n",
            rsdt->Signature[0],
            rsdt->Signature[1],
            rsdt->Signature[2],
            rsdt->Signature[3] );
        
        // #todo
        // Mesmo que a tabela rsdt nao seja satifatoria,
        // o melhor a fazer eh ignorar e procurar por "FACP"
        // e tocar o abrco.
        
    }else{
        panic("x64smp.c: Revision\n");
    };

// ----------------------------------
// --

// ---------------------------------------
// Multiple APIC Description Table (MADT)
// You should be able to find a MADT table in the RSDT table or in the XSDT table. 
// The table has a list of local-APICs, 
// https://wiki.osdev.org/MADT
// ...

//-------------------
// Initialize lapic
do_lapic:

    // #debug
    // #breakpoint

    //printf("#breakpoint\n");
    //while(1){ asm("hlt"); }

    // 0xFEE00000
    lapic_initializing(LAPIC_DEFAULT_ADDRESS);

    if (LAPIC.initialized == TRUE){
        printf("__x64_probe_smp_via_acpi: lapic initialization ok\n");
        return TRUE;
    }else if (LAPIC.initialized != TRUE){
        printf("__x64_probe_smp_via_acpi: lapic initialization fail\n");
        return FALSE;
    };

fail:
    return FALSE;
}


// x64_probe_smp:
// MP Floating Point Structure:
// To use these tables, the MP Floating Point Structure 
// must first be found. As the name suggests, 
// it is a Floating Point Structure and must be searched for.
// can't be found in this area, 
// then the area between 0xF0000 and 0xFFFFF should be searched. 
// To find the table, the following areas must be searched in:
// :: a) In the first kilobyte of Extended BIOS Data Area (EBDA), or
// :: b) Within the last kilobyte of system base memory 
// (e.g., 639K-640K for systems with 640KB of base memory 
// or 511K-512K for systems with 512 KB of base memory) 
// if the EBDA segment is undefined, or
// :: c) In the BIOS ROM address space between 0xF0000 and 0xFFFFF.
// See:
// https://wiki.osdev.org/Symmetric_Multiprocessing
// OUT:
// TRUE = OK.
// FALSE = FAIL.

// It works on qemu and qemu/kvm.
// It doesn't work on Virtualbox. (Table not found).
static int __x64_probe_smp_via_mptable(void)
{
// Probe using the MP Floating Point structure.
// Called by x64_initialize_smp().
// + Find the processor entries using the MP Floating point table.
// + Initialize local apic.

// 0x040E - The base address.
// Get a short value.
    unsigned short *bda = (unsigned short*) BDA_BASE;
    unsigned long ebda_address=0;
    register int i=0;
    unsigned char *p;
// Signature elements.
    unsigned char c1=0;
    unsigned char c2=0;
    unsigned char c3=0;
    unsigned char c4=0;

    printf("__x64_probe_smp:\n");

// #todo
// We can use a structure and put all these variable together,
    g_smp_initialized = FALSE;
    smp_info.initialized = FALSE;

// At this point we gotta have a lot of information
// in the structure 'processor'.
    if ((void*) processor == NULL){
        panic("__x64_probe_smp_via_mptable: processor\n");
    }
// Is APIC supported?
    if (processor->hasAPIC != TRUE){
        panic("__x64_probe_smp_via_mptable: No APIC\n");
    }

//
// Probe ebda address at bda base.
//

    printf("EBDA short Address: %x\n", bda[0] ); 
    ebda_address = (unsigned long) ( bda[0] << 4 );
    ebda_address = (unsigned long) ( ebda_address & 0xFFFFFFFF);
    printf("EBDA Address: %x\n", ebda_address ); 

    // #debug
    // refresh_screen();

//
// Probe 0x5F504D5F signature. 
// "_MP_".
//


// base
// between 0xF0000 and 0xFFFFF.
// #todo: filter
    p = ebda_address;

// The signature was found?
    static int mp_found = FALSE;

// Probe for the signature."_MP_"
// This signature is the first element of the table.
// MP Floating Pointer Structure
    int max = (int) (0xFFFFF - ebda_address);
    for (i=0; i<max; i++){
        c1 = p[i+0];
        c2 = p[i+1];
        c3 = p[i+2];
        c4 = p[i+3];
        if ( c1 == '_' && c2 == 'M' && c3 == 'P' && c4 == '_' )
        {
            printf (":: Found _MP_ at index %d. :)\n",i);
            mp_found=TRUE;
            break;
        }
    };

// Signature not found.
    if (mp_found != TRUE){
        printf("__x64_probe_smp_via_mptable: MP table wasn't found!\n");
        goto fail;
    }

//mp_table_found:

// ==============================================
// MPTable
// MP Floating Point Structure
// base + offset.
// This is the base of the structure.
// See:
// https://wiki.osdev.org/User:Shikhin/Tutorial_SMP
// hal/mp.h

    unsigned long table_address = 
        (unsigned long) (ebda_address + i);
    MPTable = 
        (struct mp_floating_pointer_structure_d *) table_address;

// Saving
    smp_info.mp_floating_point = 
        (struct mp_floating_pointer_structure_d *) MPTable;
// ---------------------------------------------
// Print table info

// Signature
// "_MP_" 
// + OK on qemu.
// + OK on kvm.
// + FAIL on Virtualbox. #todo: try APIC.
    printf("Signature: %c %c %c %c\n",
        MPTable->signature[0],
        MPTable->signature[1],
        MPTable->signature[2],
        MPTable->signature[3] );

    //#debug
    //refresh_screen();
    //while(1){}

// ------------------------------------------
// Getting the address of the MP Configuration Table. 
// Pointed by th MP Floating point structure.
// 32bit address.
    unsigned long configurationtable_address = 
        (unsigned long) (MPTable->configuration_table & 0xFFFFFFFF);

// Pointer for the configuration table.
    printf("Configuration table address: {%x}\n",
        configurationtable_address);
// Lenght: n*16 bytes
// The length of the floating point structure table, 
// in 16 byte units. 
// This field *should* contain 0x01, meaning 16-bytes.
    printf("Lenght: {%d}\n", MPTable->length);
// Revision: 1.x
// The version number of the MP Specification. 
// A value of 1 indicates 1.1, 4 indicates 1.4, and so on.
    printf("Revision: {%d}\n", MPTable->mp_specification_revision);
// Checksum
// The checksum of the Floating Point Structure. 
    printf("Checksum: {%d}\n", MPTable->checksum);
// Default configuration flag.
// If this is not zero then configuration_table should be 
// ignored and a default configuration should be loaded instead.
    printf("Default configuration flag: {%d}\n",
        MPTable->default_configuration );

    if ( MPTable->default_configuration != 0 ){
        printf("todo: The configuration table should be ignored\n");
    }

// Features
// Few feature bytes.
// If bit 7 is set 
// then the IMCR is present and PIC mode is being used, 
// otherwise virtual wire mode is. 
// All other bits are reserved.
    printf("Features: {%d}\n", MPTable->features);
// Se o bit 7 está ligado.
    if ( MPTable->features & (1 << 7) ){
         printf("The IMCR is present and PIC mode is being used.\n");
    }
// Se o bit 7 está desligado.
    if ( (MPTable->features & (1 << 7)) == 0 ){
         printf("Using the virtual wire mode.\n");
    }

    //#debug
    //refresh_screen();
    //while(1){}

// ==============================================
// MPConfigurationTable structure.
// Pointed by th MP Floating point structure.

// Struture pointer.
// MP Configuration table.
    MPConfigurationTable = 
        (struct mp_configuration_table_d *) configurationtable_address;

    if ((void*) MPConfigurationTable == NULL){
        printf("__x64_probe_smp: Invalid Configuration table address\n");
        goto fail;
    }
// Saving
    smp_info.mp_configuration_table = 
        (struct mp_configuration_table_d *) MPConfigurationTable;


// Signature
// "PCMP"
    printf("Signature: %c %c %c %c\n",
        MPConfigurationTable->signature[0],
        MPConfigurationTable->signature[1],
        MPConfigurationTable->signature[2],
        MPConfigurationTable->signature[3] );


// Intel strings: 
// "OEM00000" "PROD00000000"

// OEM ID STRING
    char oemid_string[8+1];
    for (i=0; i<8; i++){
        oemid_string[i] = MPConfigurationTable->oem_id[i];
    };
    oemid_string[8]=0;  // finish
    printf("OEM ID STRING: {%s}\n",oemid_string);

// PRODUCT ID STRING
    char productid_string[12+1];
    for (i=0; i<12; i++){
        productid_string[i] = MPConfigurationTable->product_id[i];
    };
    productid_string[12]=0;  // finish
    printf("PRODUCT ID STRING: {%s}\n",productid_string);

// Lapic address
    printf("lapic address: {%x}\n",
        MPConfigurationTable->lapic_address );

// Is this the standard lapic address?
    if ( MPConfigurationTable->lapic_address != LAPIC_BASE )
    {
        printf("fail: Not standard lapic address\n");
        printf("Found={%x} Standard{%x}\n",
            MPConfigurationTable->lapic_address,
            LAPIC_BASE );
    }

//
// Entries
//

// Probing the entries right below the MPConfigurationTable.

// -------------------------------
// Max number of entries.

    register int EntryCount = 
        (int) MPConfigurationTable->entry_count;

    printf("\n");
    printf("------------------\n");
    printf("Entry count: {%d}\n",
        MPConfigurationTable->entry_count);

    //#debug
    //refresh_screen();
    //while(1){};


// #bugbug
// Talvez essas entradas estão erradas.
// Talvez não haja entrada alguma nesse endereço aqui.


// =======================================================
// Entries
// ACPI processor, Local APIC.

// Logo abaixo da configuration table.
// começa uma sequência de entradas de tamanhos diferentes.
// Se a entrada for para descrever um processador, então 
// a entrada tem 20 bytes, caso contrario tem 8 bytes.
// see: mp.h

// The address of the first entry.
    unsigned long entry_base = 
    (unsigned long) ( configurationtable_address + 
                      sizeof(struct mp_configuration_table_d) );

/*
------------------------------------------------------
entry info:
Description | Type | Length | Comments

Processor   |    0 |     20 | One entry per processor.

Bus         |    1 |      8 | One entry per bus.

I/O APIC    |    2 |      8 | One entry per I/O APIC. :)

I/O 
Interrupt 
Assignment  |    3 |      8 | One entry per bus interrupt source.

Local 
Interrupt 
Assignment  |    4 |      8 | One entry per system interrupt source.
------------------------------------------------------
*/

// #remember:
// One type of entry is IOAPIC.

// Estrutura para entradas que descrevem um processador.
// Processor = type 0.

    struct entry_processor_d *e;

// This routine gets the number of processors.
// #todo:
// We can create a method for that routine.
// Register this number into the global data.
    unsigned int NumberOfProcessors=0;
    g_processor_count = NumberOfProcessors;

// #test
// #bugbug
// EntryCount has the max number of entries.
    if (EntryCount > 32)
    {
        printf("#bugbug: EntryCount > 32\n");
        EntryCount = 32;
    }

// Clean the list
    for (i=0; i<32; i++){
        smp_info.processors[i] = 0;
    };
    smp_info.number_of_processors = 0;

// loop:
// Check all the n entries indicated in the table above.
    for (i=0; i<EntryCount; i++)
    {
        // Tracing
        //printf("\n");
        //printf(":::: Entry %d:\n",i);
        
        // #bugbug
        // Actually, here we need to get not only the processor
        // entries. But also the i/o apic entry, to know
        // what is the address for the i/o apic registers.

        // Looking for processors.
        e = (struct entry_processor_d *) entry_base;

        // ---------------------------
        // It is a processor entry.
        // Size = 20.
        if (e->type == 0){

            printf("\n");
            printf("------------------\n");
            printf (">>>>> PROCESSOR found! in entry %d\n",i);

            smp_info.processors[NumberOfProcessors] = (unsigned long) e;
            NumberOfProcessors += 1;


            // apic id.
            printf("local_apic_id %d\n", e->local_apic_id);
            // apic version
            printf("local_apic_version %d\n", e->local_apic_version);
            // Flags:
            // If bit 0 is clear then the processor must be ignored.
            // If bit 1 is set then the processor is the bootstrap processor.
            // Ignore the processor.
            if( (e->flags & (1<<0)) == 0 ){
                printf("Processor must be ignored\n");
            }
            // BSP processor.
            if( e->flags & (1<<1) ){
                printf("__x64_probe_smp: The processor is a BSP\n");
            }
            printf ("stepping: %d\n", (e->signature & 0x00F));
            printf ("   model: %d\n",((e->signature & 0x0F0) >> 4) );
            printf ("  family: %d\n",((e->signature & 0xF00) >> 8) );

            entry_base = (unsigned long) (entry_base + 20);

        // ---------------------------
        // Not a processor entry.
        // Size = 8.
        } else if (e->type != 0){
            //printf ("Device type %d in entry %d\n", e->type, i );

            // #todo
            // OK, at this moment, we know that this entry is not
            // a processor entry.
            // Let's save the address for the i/o apic 
            // if we find an entry for i/o apic.
            
            // #test
            // This is the type for i/o apic entries.
            if (e->type == 2){
                printf("\n");
                printf("------------------\n");
                printf (">>>>> IOAPIC found! in entry %d\n",i);
                // #debug:
                // Checking if we found the i/o apic entry.
                // panic("#debug: ioapic\n");
            }
            
            //...
            
            entry_base = (unsigned long) (entry_base + 8);
        }
    };

done:

    printf("\n");
    printf("------------------\n");

// Global number of processors.
    g_processor_count = 
        (unsigned int) NumberOfProcessors;

// smp number of processors.
    smp_info.number_of_processors = 
        (unsigned int) NumberOfProcessors;

// #debug
    printf("Processor count: {%d}\n",
        smp_info.number_of_processors );

// smp done.
    smp_info.initialized = TRUE;
    printf("__x64_probe_smp_via_mptable: done\n");

    // #debug
    //refresh_screen();
    //while(1){
    //};

    // g_smp_initialized = TRUE;
    return TRUE;
fail:
    g_smp_initialized = FALSE;
    smp_info.initialized = FALSE;
    return FALSE;
}


// Probe for smp support and initialize lapic.
// see:
// https://wiki.osdev.org/SMP
int x64_initialize_smp(void)
{
// Called I_kmain() in kmain.c
// Probing if smp is supported.
// + Via MP.
// + Via ACPI.

    int smp_status = FALSE;  // fail

    smp_info.initialized = FALSE;
    smp_info.probe_via = 0;

//
// The SMP support.
//

    //PROGRESS("x64_initialize_smp:\n");

    // #debug
    //printf("\n");
    //printf("---- SMP START ----\n");
    printf("x64_initialize_smp:\n");


// ----------------------
// ACPI

    // #test #todo
    // Using the ACPI tables.
    smp_info.probe_via = SMP_VIA_ACPI;
    smp_status = (int) __x64_probe_smp_via_acpi();
    if (smp_status != TRUE){
        printf("x64_initialize_smp: [x64_probe_smp_via_acpi] fail\n");
    }

    // #debug
    //#breakpoint
    //while (1){ asm("hlt"); };

// ----------------------
// MP Table
// Initialize smp support via MP Floating point structure. (kinda).
// Testando a inicializaçao do lapic.
// Tentando ler o id e a versao.
// It works on qemu and qemu/kvm.
// It doesn't work on Virtualbox. (Table not found).
// See: x64.c

    smp_info.probe_via = SMP_VIA_MP_TABLE;
    smp_status = (int) __x64_probe_smp_via_mptable();

    if (smp_status == TRUE)
    {
        printf("x64_initialize_smp: [x64_probe_smp] ok\n");
        // Initialize LAPIC based on the address we found before.
        if ((void*) MPConfigurationTable != NULL)
        {
            if (MPConfigurationTable->lapic_address != 0)
            {
                // see: apic.c
                lapic_initializing( MPConfigurationTable->lapic_address );
                if (LAPIC.initialized == TRUE){
                    printf("??: lapic initialization ok\n");
                }else if (LAPIC.initialized != TRUE){
                    printf("??: lapic initialization fail\n");
                };
            }
        }
    }

    // #debug
    // #breakpoint
    // while (1){ asm("hlt"); };

    //printf("---- SMP END ----\n");
    //printf("\n");
    
    return (int) smp_status;
}



/*
// #todo
void x64_show_smp_into(void);
void x64_show_smp_into(void)
{
// Processor entry.
    struct entry_processor_d *p_entry;
    int i=0;

    if (smp_info.initialized != TRUE)
        return;
    // #debug
    if (smp_info.number_of_processors > 16)
        return;

    for (i=0; i<smp_info.number_of_processors; i++)
    {
        p_entry = (struct entry_processor_d *) smp_info.processors[i];
        if ( (void*) p_entry != NULL )
        {
            //#todo
            // Print info.
        }
    };
}
*/

