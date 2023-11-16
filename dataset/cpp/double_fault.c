/* SPDX-License-Identifier: BSD-2-Clause */

#include <tilck/common/basic_defs.h>
#include <tilck/kernel/hal.h>
#include <tilck/kernel/process_int.h>
#include <tilck/kernel/paging.h>
#include <tilck/kernel/debug_utils.h>

#include "double_fault.h"
#include "gdt_int.h"
#include "idt_int.h"

extern volatile bool __in_double_fault;

void double_fault_handler_asm(void);
static int double_fault_tss_num;

static const struct tss_entry df_tss_data = {
   .esp0 = ((ulong)kernel_initial_stack + PAGE_SIZE - 4),
   .ss0 = X86_KERNEL_DATA_SEL,
   .cr3 = 0 /* updated later */,
   .eip = (ulong)&double_fault_handler_asm,
   .eflags = 0x2,
   .cs = X86_KERNEL_CODE_SEL,
   .es = X86_KERNEL_DATA_SEL,
   .ss = X86_KERNEL_DATA_SEL,
   .ds = X86_KERNEL_DATA_SEL,
   .fs = X86_KERNEL_DATA_SEL,
   .gs = X86_KERNEL_DATA_SEL,
   .esp = ((ulong)kernel_initial_stack + PAGE_SIZE - 4),
};

static inline void double_fault_tss_update_cr3(void)
{
   tss_array[TSS_DOUBLE_FAULT].cr3 = read_cr3();
}

void register_double_fault_tss_entry(void)
{
   struct gdt_entry e;
   memcpy(&tss_array[TSS_DOUBLE_FAULT], &df_tss_data, sizeof(struct tss_entry));
   double_fault_tss_update_cr3();

   gdt_set_entry(&e,
                 (ulong)&tss_array[TSS_DOUBLE_FAULT],
                 sizeof(tss_array[TSS_DOUBLE_FAULT]),
                 GDT_DESC_TYPE_TSS | GDT_ACCESS_PL0,
                 GDT_GRAN_BYTE | GDT_32BIT);

   double_fault_tss_num = gdt_add_entry(&e);

   if (double_fault_tss_num < 0)
      panic("Unable to add a GDT entry for the double fault TSS");

   /* Install the task gate for the double fault */
   idt_set_entry(FAULT_DOUBLE_FAULT,
                 NULL, /* offset is not used for task gates: must be 0 */
                 X86_SELECTOR(double_fault_tss_num, TABLE_GDT, 0),
                 IDT_FLAG_PRESENT | IDT_FLAG_TASK_GATE | IDT_FLAG_DPL0);
}

void double_fault_handler(void)
{
   __in_double_fault = true;
   panic("[Double fault] Kernel stack: %p", get_curr_task()->kernel_stack);
}

/* ------------------------------------------------- */

void on_first_pdir_update(void)
{
   double_fault_tss_update_cr3();
}
