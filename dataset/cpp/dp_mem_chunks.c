/* SPDX-License-Identifier: BSD-2-Clause */

#include <tilck_gen_headers/config_kmalloc.h>

#include <tilck/common/basic_defs.h>
#include <tilck/common/string_util.h>
#include <tilck/common/utils.h>

#include <tilck/kernel/sched.h>
#include <tilck/kernel/kmalloc.h>
#include <tilck/kernel/kmalloc_debug.h>
#include <tilck/kernel/sort.h>

#include "termutil.h"
#include "dp_int.h"

#define CHUNKS_ARR_BUF_SIZE                              (16 * KB)

struct chunk_info {

   size_t size;
   size_t count;
   u64 max_waste;
   u32 max_waste_p;
};

static struct debug_kmalloc_stats stats;
static u64 lf_allocs;
static u64 lf_waste;
static size_t chunks_count;
static struct chunk_info *chunks_arr;
static size_t chunks_max_count;
static char chunks_order_by;

static long dp_chunks_cmpf_size(const void *a, const void *b)
{
   const struct chunk_info *x = a;
   const struct chunk_info *y = b;
   return (long)y->size - (long)x->size;
}

static long dp_chunks_cmpf_count(const void *a, const void *b)
{
   const struct chunk_info *x = a;
   const struct chunk_info *y = b;
   return (long)y->count - (long)x->count;
}

static long dp_chunks_cmpf_waste(const void *a, const void *b)
{
   const struct chunk_info *x = a;
   const struct chunk_info *y = b;
   return (long)y->max_waste - (long)x->max_waste;
}

static long dp_chunks_cmpf_waste_p(const void *a, const void *b)
{
   const struct chunk_info *x = a;
   const struct chunk_info *y = b;
   return (long)y->max_waste_p - (long)x->max_waste_p;
}

static void dp_chunks_enter(void)
{
   struct debug_kmalloc_chunks_ctx ctx;
   size_t s, c;

   if (!KMALLOC_HEAVY_STATS)
      return;

   if (!chunks_arr) {

      size_t chunks_buf_sz = CHUNKS_ARR_BUF_SIZE;

      for (int i = 0; i < 4; i++) {

         if ((chunks_arr = kzmalloc(chunks_buf_sz)))
            break;

         chunks_buf_sz /= 2;
      }

      if (!chunks_arr)
         panic("Unable to alloc memory for chunks_arr");

      chunks_max_count = chunks_buf_sz / sizeof(struct chunk_info);
   }

   debug_kmalloc_get_stats(&stats);
   lf_allocs = 0;
   lf_waste = 0;
   chunks_count = 0;
   chunks_order_by = 's';

   disable_preemption();
   {
      debug_kmalloc_chunks_stats_start_read(&ctx);
      while (debug_kmalloc_chunks_stats_next(&ctx, &s, &c)) {

         if (chunks_count == chunks_max_count)
            break;

         const u64 waste = (u64)(
            UNSAFE_MAX(SMALL_HEAP_MBS, roundup_next_power_of_2(s)) - s
         ) * c;

         chunks_arr[chunks_count++] = (struct chunk_info) {
            .size = s,
            .count = c,
            .max_waste = waste,
            .max_waste_p = (u32)(waste * 1000 / (waste + (u64)s * (u64)c)),
         };

         lf_allocs += (u64)s * c;
         lf_waste += waste;
      }
   }
   enable_preemption();
}

static void dp_chunks_exit(void)
{
   if (!KMALLOC_HEAVY_STATS)
      return;
}

static int dp_chunks_keypress(struct key_event ke)
{
   const char c = ke.print_char;

   switch (c) {

      case 's':
         insertion_sort_generic(chunks_arr,
                                sizeof(chunks_arr[0]),
                                (u32)chunks_count,
                                dp_chunks_cmpf_size);
         ui_need_update = true;
         chunks_order_by = c;
         return kb_handler_ok_and_continue;

      case 'c':
         insertion_sort_generic(chunks_arr,
                                sizeof(chunks_arr[0]),
                                (u32)chunks_count,
                                dp_chunks_cmpf_count);
         ui_need_update = true;
         chunks_order_by = c;
         return kb_handler_ok_and_continue;

      case 'w':
         insertion_sort_generic(chunks_arr,
                                sizeof(chunks_arr[0]),
                                (u32)chunks_count,
                                dp_chunks_cmpf_waste);
         ui_need_update = true;
         chunks_order_by = c;
         return kb_handler_ok_and_continue;

      case 't':
         insertion_sort_generic(chunks_arr,
                                sizeof(chunks_arr[0]),
                                (u32)chunks_count,
                                dp_chunks_cmpf_waste_p);
         ui_need_update = true;
         chunks_order_by = c;
         return kb_handler_ok_and_continue;

      default:
         return kb_handler_nak;
   }
}


static void dp_show_chunks(void)
{
   int row = dp_screen_start_row;
   const u64 lf_tot = lf_allocs + lf_waste;

   if (!KMALLOC_HEAVY_STATS) {
      dp_writeln("Not available: recompile with KMALLOC_HEAVY_STATS=1");
      return;
   }

   dp_writeln("Chunk sizes count:         %5u sizes", chunks_count);
   dp_writeln("Lifetime data allocated:   %5llu %s [actual: %llu %s]",
              lf_allocs < 32*MB ? lf_allocs/KB : lf_allocs/MB,
              lf_allocs < 32*MB ? "KB" : "MB",
              lf_tot < 32*MB ? lf_tot/KB : lf_tot/MB,
              lf_tot < 32*MB ? "KB" : "MB");
   dp_writeln("Lifetime max data waste:   %5llu %s (%llu.%llu%%)",
              lf_waste < 32*MB ? lf_waste/KB : lf_waste/MB,
              lf_waste < 32*MB ? "KB" : "MB",
              lf_waste * 100 / lf_tot,
              (lf_waste * 1000 / lf_tot) % 10);

   dp_writeln(
      "Order by: "
      E_COLOR_BR_WHITE "s" RESET_ATTRS "ize, "
      E_COLOR_BR_WHITE "c" RESET_ATTRS "ount, "
      E_COLOR_BR_WHITE "w" RESET_ATTRS "aste, "
      "was" E_COLOR_BR_WHITE "t" RESET_ATTRS "e (%%)"
   );

   dp_writeln("");

   dp_writeln(
                 "%s" "   Size   "        RESET_ATTRS
      TERM_VLINE "%s" "  Count  "         RESET_ATTRS
      TERM_VLINE "%s" " Max waste "       RESET_ATTRS
      TERM_VLINE "%s" " Max waste (%%)"   RESET_ATTRS,
      chunks_order_by == 's' ? E_COLOR_BR_WHITE REVERSE_VIDEO : "",
      chunks_order_by == 'c' ? E_COLOR_BR_WHITE REVERSE_VIDEO : "",
      chunks_order_by == 'w' ? E_COLOR_BR_WHITE REVERSE_VIDEO : "",
      chunks_order_by == 't' ? E_COLOR_BR_WHITE REVERSE_VIDEO : ""
   );

   dp_writeln(
      GFX_ON
      "qqqqqqqqqqnqqqqqqqqqnqqqqqqqqqqqnqqqqqqqqqqqqqqqqqq"
      GFX_OFF
   );

   for (size_t i = 0; i < chunks_count; i++) {

      const u64 waste = chunks_arr[i].max_waste;

      dp_writeln("%9u "
                 TERM_VLINE " %7u "
                 TERM_VLINE " %6llu %s "
                 TERM_VLINE " %6u.%u%%",
                 chunks_arr[i].size,
                 chunks_arr[i].count,
                 waste < KB ? waste : waste / KB,
                 waste < KB ? "B " : "KB",
                 chunks_arr[i].max_waste_p / 10,
                 chunks_arr[i].max_waste_p % 10);
   }

   dp_writeln("");
}

static struct dp_screen dp_chunks_screen =
{
   .index = 5,
   .label = "MemChunks",
   .draw_func = dp_show_chunks,
   .on_dp_enter = dp_chunks_enter,
   .on_dp_exit = dp_chunks_exit,
   .on_keypress_func = dp_chunks_keypress,
};

__attribute__((constructor))
static void dp_chunks_init(void)
{
   dp_register_screen(&dp_chunks_screen);
}
