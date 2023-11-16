/* SPDX-License-Identifier: BSD-2-Clause */

#include <tilck_gen_headers/mod_tracing.h>

#include <tilck/common/basic_defs.h>
#include <tilck/common/printk.h>

#include <tilck/kernel/datetime.h>
#include <tilck/kernel/sched.h>
#include <tilck/kernel/kmalloc.h>
#include <tilck/kernel/errno.h>
#include <tilck/kernel/fs/vfs.h>

#include <tilck/mods/tracing.h>

#include "termutil.h"
#include "dp_tracing_int.h"

#if MOD_tracing

static char *line_buf;

/* Shared data with dp_tracing_sys.c */
char *rend_bufs[6];
int used_rend_bufs;
/* -- */

void init_dp_tracing(void)
{
   for (int i = 0; i < 6; i++) {

      if (!(rend_bufs[i] = kmalloc(REND_BUF_SZ)))
         panic("[dp] Unable to allocate rend_buf[%d]", i);
   }

   if (!(line_buf = kmalloc(TRACED_SYSCALLS_STR_LEN)))
      panic("[dp] Unable to allocate line_buf");
}

static void
tracing_ui_show_help(void)
{
   dp_write_raw("\r\n\r\n");
   dp_write_raw(E_COLOR_YELLOW "Tracing mode help" RESET_ATTRS "\r\n");

   dp_write_raw(
      E_COLOR_YELLOW "  "
      E_COLOR_YELLOW "o" RESET_ATTRS "     : Toggle always enter + exit\r\n"
      RESET_ATTRS
   );

   dp_write_raw(
      E_COLOR_YELLOW "  "
      E_COLOR_YELLOW "b" RESET_ATTRS "     : Toggle dump big buffers\r\n"
      RESET_ATTRS
   );

   dp_write_raw(
      E_COLOR_YELLOW "  "
      E_COLOR_YELLOW "e" RESET_ATTRS "     : Edit syscalls wildcard expr "
      E_COLOR_RED "[1]" RESET_ATTRS "\r\n"
      RESET_ATTRS
   );

   dp_write_raw(
      E_COLOR_YELLOW "  "
      E_COLOR_YELLOW "k" RESET_ATTRS "     : Set trace_printk() level\r\n"
      RESET_ATTRS
   );

   dp_write_raw(
      E_COLOR_YELLOW "  "
      E_COLOR_YELLOW "l" RESET_ATTRS "     : List traced syscalls\r\n"
      RESET_ATTRS
   );

   dp_write_raw(
      E_COLOR_YELLOW "  "
      E_COLOR_YELLOW "p" RESET_ATTRS "     : Dump user tasks list\r\n"
      RESET_ATTRS
   );

   dp_write_raw(
      E_COLOR_YELLOW "  "
      E_COLOR_YELLOW "P" RESET_ATTRS "     : Dump full task list\r\n"
      RESET_ATTRS
   );

   dp_write_raw(
      E_COLOR_YELLOW "  "
      E_COLOR_YELLOW "t" RESET_ATTRS "     : Edit list of traced PIDs\r\n"
      RESET_ATTRS
   );

   dp_write_raw(
      E_COLOR_YELLOW "  "
      E_COLOR_YELLOW "q" RESET_ATTRS "     : Back to the debug panel\r\n"
      RESET_ATTRS
   );

   dp_write_raw(
      E_COLOR_YELLOW "  "
      "ENTER" RESET_ATTRS " : Start / stop tracing\r\n"
   );

   dp_write_raw("\r\n" E_COLOR_RED "[1]" RESET_ATTRS " ");
   dp_write_raw("In the wildcard expr the " E_COLOR_BR_WHITE "*" RESET_ATTRS
                " character is allowed only once, at the end.\r\n");

   dp_write_raw("The " E_COLOR_BR_WHITE "!" RESET_ATTRS " character can be "
                "used, at the beginning of each sub-expr, to negate it.\r\n");

   dp_write_raw("Single sub-expressions are separated by comma or space. "
                "The " E_COLOR_BR_WHITE "?" RESET_ATTRS " character is\r\n");

   dp_write_raw("supported and has the usual meaning "
                "(matches 1 single char, any).\r\n");

   dp_write_raw(
      E_COLOR_BR_WHITE "Example: " RESET_ATTRS
      "read*,write*,!readlink* \r\n"
   );
}

static void
tracing_ui_msg(void)
{
   dp_write_raw(
      E_COLOR_YELLOW
      "Tilck syscall tracing (h: help)\r\n"
      RESET_ATTRS
   );

   dp_write_raw(

      TERM_VLINE " Always ENTER+EXIT: %s "
      TERM_VLINE " Big bufs: %s  "
      TERM_VLINE " #Sys traced: " E_COLOR_BR_BLUE "%d" RESET_ATTRS " "
      TERM_VLINE " #Tasks traced: " E_COLOR_BR_BLUE "%d" RESET_ATTRS " "
      TERM_VLINE "\r\n"
      TERM_VLINE " Printk lvl: " E_COLOR_BR_BLUE "%d" RESET_ATTRS
      "\r\n",

      tracing_is_force_exp_block_enabled()
         ? E_COLOR_GREEN "ON" RESET_ATTRS
         : E_COLOR_RED "OFF" RESET_ATTRS,

      tracing_are_dump_big_bufs_on()
         ? E_COLOR_GREEN "ON" RESET_ATTRS
         : E_COLOR_RED "OFF" RESET_ATTRS,

      get_traced_syscalls_count(),
      get_traced_tasks_count(),
      tracing_get_printk_lvl()
   );

   get_traced_syscalls_str(line_buf, TRACED_SYSCALLS_STR_LEN);

   dp_write_raw(
      TERM_VLINE
      " Trace expr: " E_COLOR_YELLOW "%s" RESET_ATTRS,
      line_buf
   );

   dp_write_raw("\r\n");
   dp_write_raw(E_COLOR_YELLOW "> " RESET_ATTRS);
}

static void
dp_dump_tracing_event(struct trace_event *e)
{
   dp_write_raw(
      "%05u.%03u [%05d] ",
      (u32)(e->sys_time / TS_SCALE),
      (u32)((e->sys_time % TS_SCALE) / (TS_SCALE / 1000)),
      e->tid
   );

   switch (e->type) {

      case te_sys_enter:
      case te_sys_exit:
         dp_handle_syscall_event(e);
         break;

      case te_printk:
         dp_write_raw(
            E_COLOR_YELLOW "LOG" RESET_ATTRS "[%02d]: %s\r\n",
            e->p_ev.level, e->p_ev.buf
         );
         break;

      case te_signal_delivered:
         dp_write_raw(
            E_COLOR_YELLOW "GOT SIGNAL: " RESET_ATTRS "%s[%d]\r\n",
            get_signal_name(e->sig_ev.signum),
            e->sig_ev.signum
         );
         break;

      case te_killed:
         dp_write_raw(
            E_COLOR_BR_RED "KILLED BY SIGNAL: " RESET_ATTRS "%s[%d]\r\n",
            get_signal_name(e->sig_ev.signum),
            e->sig_ev.signum
         );
         break;

      default:
         dp_write_raw(
            E_COLOR_BR_RED "<unknown event %d>\r\n" RESET_ATTRS,
            e->type
         );
   }
}

static bool
dp_tracing_screen_main_loop(void)
{
   struct trace_event e;
   int rc;
   char c;

   while (true) {

      /* Check the input for Ctrl+C */
      rc = vfs_read(dp_input_handle, &c, 1);

      if (rc < 0 && rc != -EAGAIN)
         return false; /* exit because of an error */

      if (rc == 1) {

         switch (c) {

            case 'q':
               return false; /* clean exit */

            case DP_KEY_ENTER:
               return true; /* stop dumping the trace buffer */
         }
      }

      if (read_trace_event(&e, TIMER_HZ / 10))
         dp_dump_tracing_event(&e);
   }

   NOT_REACHED();
}

static void
dp_edit_trace_syscall_str(void)
{
   get_traced_syscalls_str(line_buf, TRACED_SYSCALLS_STR_LEN);
   dp_move_left(2);
   dp_write_raw(E_COLOR_YELLOW "expr> " RESET_ATTRS);
   dp_set_input_blocking(true);
   dp_read_line(line_buf, TRACED_SYSCALLS_STR_LEN);
   dp_set_input_blocking(false);

   if (set_traced_syscalls(line_buf) < 0)
      dp_write_raw(E_COLOR_RED "Invalid input\r\n" RESET_ATTRS);
}

static void
dp_edit_trace_printk_level(void)
{
   line_buf[0] = 0;
   dp_move_left(2);
   dp_write_raw(E_COLOR_YELLOW "Level [0, 100]: " RESET_ATTRS);
   dp_set_input_blocking(true);
   dp_read_line(line_buf, TRACED_SYSCALLS_STR_LEN);
   dp_set_input_blocking(false);

   int err = 0;
   long val = tilck_strtol(line_buf, NULL, 10, &err);

   if (err || val < 0 || val > 100) {
      dp_write_raw("\r\n");
      dp_write_raw(E_COLOR_RED "Invalid input\r\n" RESET_ATTRS);
      return;
   }

   tracing_set_printk_lvl((int) val);
}

struct traced_list_cb_ctx {

   char *buf;
   size_t buf_sz;
   size_t written;
};

static int
dp_tracing_get_traced_list_str_cb(void *obj, void *arg)
{
   struct task *ti = obj;
   struct traced_list_cb_ctx *ctx = arg;
   char tidstr[16];
   size_t s;

   if (!ti->traced)
      return 0;

   s = (size_t)snprintk(tidstr, sizeof(tidstr), "%d,", ti->tid);

   if (ctx->written + s + 1 < ctx->buf_sz) {
      strcpy(ctx->buf + ctx->written, tidstr);
      ctx->written += s;
   }

   /* Disable tracing */
   ti->traced = false;
   return 0;
}

static int
dp_set_task_as_traced(const char *tidstr, int *traced_cnt)
{
   long tid = tilck_strtol(tidstr, NULL, 10, NULL);

   if (tid <= 0)
      return -1;

   struct task *ti;
   disable_preemption();
   {
      ti = get_task((int)tid);

      if (ti) {
         ti->traced = true;
         (*traced_cnt)++;
      }
   }
   enable_preemption();
   return 0;
}

static int
dp_set_traced_tids_str(const char *str, int *traced_cnt)
{
   const char *s = str;
   char *p, buf[32];
   int rc;

   for (p = buf; *s; s++) {

      if (p == buf + sizeof(buf))
         return -ENAMETOOLONG;

      if (*s == ',' || *s == ' ') {
         *p = 0;
         p = buf;

         if ((rc = dp_set_task_as_traced(buf, traced_cnt)))
            return rc;

         continue;
      }

      *p++ = *s;
   }

   if (p > buf) {

      *p = 0;

      if ((rc = dp_set_task_as_traced(buf, traced_cnt)))
         return rc;
   }

   return 0;
}

static void
dp_edit_traced_list(void)
{
   int traced_cnt = 0;
   line_buf[0] = 0;

   struct traced_list_cb_ctx ctx = {
      .buf = line_buf,
      .buf_sz = TRACED_SYSCALLS_STR_LEN,
      .written = 0,
   };

   disable_preemption();
   {
      iterate_over_tasks(dp_tracing_get_traced_list_str_cb, &ctx);
   }
   enable_preemption();

   dp_move_left(2);
   dp_write_raw(E_COLOR_YELLOW "PIDs> " RESET_ATTRS);
   dp_set_input_blocking(true);
   dp_read_line(line_buf, TRACED_SYSCALLS_STR_LEN);
   dp_set_input_blocking(false);

   dp_write_raw("\r\n");

   if (dp_set_traced_tids_str(line_buf, &traced_cnt) < 0) {
      dp_write_raw("Invalid input\r\n");
   } else {
      dp_write_raw("Tracing %d tasks\r\n", traced_cnt);
   }
}

static void
dp_list_traced_syscalls(void)
{
   dp_write_raw("\r\n\r\n");
   dp_write_raw(E_COLOR_YELLOW "Traced syscalls list" RESET_ATTRS);
   dp_write_raw("\r\n");

   for (u32 i = 0; i < MAX_SYSCALLS; i++) {

      if (!tracing_is_enabled_on_sys(i))
         continue;

      dp_write_raw("%s ", 4 + tracing_get_syscall_name(i));
   }

   dp_write_raw("\r\n");
}

static int
dp_tracing_dump_remaining_events(void)
{
   char c;
   int rem;
   struct key_event ke;
   struct trace_event e;

   if (!(rem = tracing_get_in_buffer_events_count()))
      return 0; /* no remaining events in the buffer */

   dp_write_raw("Discard remaining %d events in the buf? [Y/n] ", rem);

   do {

      if (dp_read_ke_from_tty(&ke) < 0)
         return -1; /* unexpected I/O error */

      c = ke.print_char;

   } while (c != 'y' && c != 'n' && c != '\r');

   if (c == '\r')
      c = 'y';

   dp_write_raw_int(&c, 1);

   while (true) {

      if (!read_trace_event_noblock(&e))
         break;

      if (c == 'n')
         dp_dump_tracing_event(&e);
   }

   dp_write_raw("\r\n");
   return 1;
}

enum kb_handler_action
dp_tracing_screen(void)
{
   char c;
   int rc;
   bool should_continue;

   dp_set_cursor_enabled(true);
   dp_clear();
   dp_move_cursor(1, 1);
   tracing_ui_msg();

   while (true) {

      dp_set_input_blocking(true);
      {
         rc = vfs_read(dp_input_handle, &c, 1);
      }
      dp_set_input_blocking(false);

      if (rc <= 0)
         break; /* something gone wrong */

      if (c == 'q')
         break; /* clean exit */

      if (c == DP_KEY_ENTER) {

         dp_write_raw("\r\n");
         dp_write_raw(
            E_COLOR_GREEN "-- Tracing active --" RESET_ATTRS "\r\n\r\n"
         );

         tracing_set_enabled(true);
         {
            should_continue = dp_tracing_screen_main_loop();
         }
         tracing_set_enabled(false);

         if (!should_continue)
            break;

         dp_write_raw(
            E_COLOR_RED "-- Tracing stopped --" RESET_ATTRS "\r\n"
         );

         if ((rc = dp_tracing_dump_remaining_events()) < 0)
            break; /* unexpected I/O error */

         dp_write_raw("\r\n");
         tracing_ui_msg();
         continue;
      }

      switch (c) {

         case 'o':
            dp_write_raw("%c", c);
            tracing_set_force_exp_block(!tracing_is_force_exp_block_enabled());
            break;

         case 'b':
            dp_write_raw("%c", c);
            tracing_set_dump_big_bufs_opt(!tracing_are_dump_big_bufs_on());
            break;

         case 'h':
            dp_write_raw("%c", c);
            tracing_ui_show_help();
            break;

         case 'p':
            dp_write_raw("%c", c);
            dp_dump_task_list(false, true);
            break;

         case 'P':
            dp_write_raw("%c", c);
            dp_dump_task_list(true, true);
            break;

         case 'l':
            dp_write_raw("%c", c);
            dp_list_traced_syscalls();
            break;

         case 'e':
            dp_edit_trace_syscall_str();
            break;

         case 'k':
            dp_edit_trace_printk_level();
            break;

         case 't':
            dp_edit_traced_list();
            break;

         default:
            continue;
      }

      dp_write_raw("\r\n\r\n");
      tracing_ui_msg();
   }

   ui_need_update = true;
   dp_set_cursor_enabled(false);
   return kb_handler_ok_and_continue;
}

#endif // #if MOD_tracing
