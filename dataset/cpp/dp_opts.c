/* SPDX-License-Identifier: BSD-2-Clause */

#include <tilck_gen_headers/config_debug.h>
#include <tilck_gen_headers/config_boot.h>
#include <tilck_gen_headers/config_kmalloc.h>
#include <tilck_gen_headers/config_userlim.h>
#include <tilck_gen_headers/config_mm.h>
#include <tilck_gen_headers/mod_kb8042.h>
#include <tilck_gen_headers/mod_console.h>
#include <tilck_gen_headers/mod_fb.h>
#include <tilck_gen_headers/mod_serial.h>
#include <tilck_gen_headers/mod_debugpanel.h>
#include <tilck_gen_headers/mod_tracing.h>
#include <tilck_gen_headers/mod_sysfs.h>
#include <tilck_gen_headers/mod_acpi.h>
#include <tilck_gen_headers/mod_pci.h>
#include <tilck_gen_headers/mod_sb16.h>

#include <tilck/common/basic_defs.h>
#include <tilck/common/string_util.h>

#include <tilck/kernel/term.h>
#include <tilck/kernel/cmdline.h>
#include <tilck/kernel/datetime.h>
#include <tilck/kernel/sched.h>

#include <tilck/mods/fb_console.h>

#include "termutil.h"

#define DUMP_LABEL(name)     dp_write(row++, col, "%-16s", name)
#define DUMP_INT(name, val)  dp_write(row++, col, "  %-16s: %d", name, val)
#define DUMP_STR_OPT(opt)    dp_write(row++, col, "  %-30s: %s", #opt, opt)
#define DUMP_INT_OPT(opt)    dp_write(row++, col, "  %-30s: %d", #opt, opt)
#define DUMP_BOOL_OPT(opt)                               \
   dp_write(row++,                                       \
            col,                                         \
            "  %-30s: %s%u" RESET_ATTRS,                 \
            #opt,                                        \
            opt ? E_COLOR_GREEN : DP_ESC_COLOR,          \
            opt)

static void dp_show_opts(void)
{
   int row = dp_screen_start_row + 1;
   int col = dp_start_col + 3;
   int rows_left;
   int rows_right;
   int max_rows;

   DUMP_LABEL("Main");
   DUMP_STR_OPT(BUILDTYPE_STR);
   DUMP_INT_OPT(DEBUG_CHECKS);
   DUMP_INT_OPT(TIMER_HZ);
   DUMP_INT_OPT(KERNEL_STACK_PAGES);
   DUMP_INT_OPT(USER_STACK_PAGES);

   DUMP_LABEL("Kernel modules");
   DUMP_BOOL_OPT(MOD_acpi);
   DUMP_BOOL_OPT(MOD_pci);
   DUMP_BOOL_OPT(MOD_kb8042);
   DUMP_BOOL_OPT(MOD_console);
   DUMP_BOOL_OPT(MOD_fb);
   DUMP_BOOL_OPT(MOD_serial);
   DUMP_BOOL_OPT(MOD_debugpanel);
   DUMP_BOOL_OPT(MOD_tracing);
   DUMP_BOOL_OPT(MOD_sysfs);
   DUMP_BOOL_OPT(MOD_sb16);

   DUMP_LABEL("Modules config");
   DUMP_INT_OPT(FBCON_BIGFONT_THR);
   DUMP_BOOL_OPT(FB_CONSOLE_BANNER);
   DUMP_BOOL_OPT(FB_CONSOLE_CURSOR_BLINK);
   DUMP_BOOL_OPT(FB_CONSOLE_USE_ALT_FONTS);
   DUMP_BOOL_OPT(KERNEL_SHOW_LOGO);
   DUMP_BOOL_OPT(PCI_VENDORS_LIST);

   DUMP_LABEL("Enabled by default");
   DUMP_BOOL_OPT(SERIAL_CON_IN_VIDEO_MODE);
   DUMP_BOOL_OPT(KRN_TRACK_NESTED_INTERR);
   DUMP_BOOL_OPT(PANIC_SHOW_STACKTRACE);
   DUMP_BOOL_OPT(DEBUG_CHECKS);
   DUMP_BOOL_OPT(KERNEL_SELFTESTS);
   DUMP_BOOL_OPT(KERNEL_STACK_ISOLATION);
   DUMP_BOOL_OPT(KERNEL_SYMBOLS);
   DUMP_BOOL_OPT(KRN_PRINTK_ON_CURR_TTY);
   DUMP_BOOL_OPT(BOOT_INTERACTIVE);
   DUMP_BOOL_OPT(KERNEL_64BIT_OFFT);
   DUMP_BOOL_OPT(KRN_CLOCK_DRIFT_COMP);

   DUMP_LABEL("Disabled by default");
   DUMP_BOOL_OPT(KRN_NO_SYS_WARN);
   DUMP_BOOL_OPT(KRN_PAGE_FAULT_PRINTK);
   DUMP_BOOL_OPT(KERNEL_UBSAN);
   DUMP_BOOL_OPT(TERM_BIG_SCROLL_BUF);
   DUMP_BOOL_OPT(KRN_RESCHED_ENABLE_PREEMPT);
   DUMP_BOOL_OPT(KERNEL_BIG_IO_BUF);
   DUMP_BOOL_OPT(PS2_DO_SELFTEST);
   DUMP_BOOL_OPT(PS2_VERBOSE_DEBUG_LOG);
   DUMP_BOOL_OPT(KERNEL_GCOV);
   DUMP_BOOL_OPT(FORK_NO_COW);
   DUMP_BOOL_OPT(MMAP_NO_COW);
   DUMP_BOOL_OPT(PANIC_SHOW_REGS);
   DUMP_BOOL_OPT(KMALLOC_HEAVY_STATS);
   DUMP_BOOL_OPT(KMALLOC_FREE_MEM_POISONING);
   DUMP_BOOL_OPT(KMALLOC_SUPPORT_DEBUG_LOG);
   DUMP_BOOL_OPT(KMALLOC_SUPPORT_LEAK_DETECTOR);
   DUMP_BOOL_OPT(BOOTLOADER_POISON_MEMORY);
   DUMP_BOOL_OPT(FB_CONSOLE_FAILSAFE_OPT);

   DUMP_LABEL("Other");
   DUMP_INT_OPT(PREFERRED_GFX_MODE_W);
   DUMP_INT_OPT(PREFERRED_GFX_MODE_H);

   rows_left = row - dp_screen_start_row - 1;
   row = dp_screen_start_row+1;
   col = dp_start_col + 48;

   DUMP_LABEL("Main");
   DUMP_INT("HYPERVISOR", in_hypervisor());

   DUMP_LABEL("Console");
   {
      struct term_params info;
      process_term_read_info(&info);
      DUMP_INT("TERM_ROWS", info.rows);
      DUMP_INT("TERM_COLS", info.cols);
   }

   DUMP_INT("USE_FRAMEBUFFER", use_framebuffer());

   if (MOD_fb && use_framebuffer()) {
      struct fb_console_info fbi;
      fb_console_get_info(&fbi);

      DUMP_INT("FB_OPT_FUNCS", fb_is_using_opt_funcs());
      DUMP_INT("FB_RES_X", fbi.res_x);
      DUMP_INT("FB_RES_Y", fbi.res_y);
      DUMP_INT("FB_BPP", fbi.bpp);
      DUMP_INT("FB_FONT_W", fbi.font_w);
      DUMP_INT("FB_FONT_H", fbi.font_h);
   }

   DUMP_INT("TTY_COUNT", kopt_ttys);

   DUMP_LABEL("System clock");
   {
      struct clock_resync_stats stats;
      clock_get_resync_stats(&stats);

      DUMP_INT("CLK_IN_RS", clock_in_resync());
      DUMP_INT("CLK_IN_FULL_RS", clock_in_full_resync());
      DUMP_INT("CLK_FULL_RS_CNT", stats.full_resync_count);
      DUMP_INT("CLK_FULL_RS_FAIL", stats.full_resync_fail_count);
      DUMP_INT("CLK_FULL_RS_SUCC", stats.full_resync_success_count);
      DUMP_INT("CLK_FULL_RS_ADG1", stats.full_resync_abs_drift_gt_1);
      DUMP_INT("CLK_RESYNC_CNT", stats.multi_second_resync_count);
   }

   rows_right = row - dp_screen_start_row - 1;
   max_rows = MAX(rows_left, rows_right);

   /* left rectangle */
   dp_draw_rect("Build-time", E_COLOR_GREEN,
                dp_screen_start_row, dp_start_col + 1, max_rows+2, 45);

   /* right rectangle */
   dp_draw_rect("Run-time", E_COLOR_GREEN,
                dp_screen_start_row, col - 2, max_rows+2, 29);
}

static struct dp_screen dp_opts_screen =
{
   .index = 0,
   .label = "Options",
   .draw_func = dp_show_opts,
   .on_keypress_func = NULL,
};

__attribute__((constructor))
static void dp_opts_init(void)
{
   dp_register_screen(&dp_opts_screen);
}
