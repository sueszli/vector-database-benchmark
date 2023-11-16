/*
 * Copyright (c) 2015 Balabit
 * Copyright (c) 2015 Balázs Scheidler
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * As an additional exemption you are allowed to compile & link against the
 * OpenSSL libraries as published by the OpenSSL project. See the file
 * COPYING for details.
 *
 */

#include "debugger/debugger.h"
#include "logpipe.h"

static Debugger *current_debugger;

static gboolean
_pipe_hook(LogPipe *s, LogMessage *msg, const LogPathOptions *path_options)
{
  if ((s->flags & PIF_CONFIG_RELATED) == 0)
    return TRUE;

  if (msg->flags & LF_STATE_TRACING)
    return debugger_perform_tracing(current_debugger, s, msg);
  else
    return debugger_stop_at_breakpoint(current_debugger, s, msg);
}

void
debugger_start(MainLoop *main_loop, GlobalConfig *cfg)
{
  /* we don't support threaded mode (yet), force it to non-threaded */
  cfg->threaded = FALSE;
  current_debugger = debugger_new(main_loop, cfg);
  pipe_single_step_hook = _pipe_hook;
  debugger_start_console(current_debugger);
}
