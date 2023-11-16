/*
  Copyright (c) 2007 Thomas Jahns <Thomas.Jahns@gmx.net>
  Copyright (c) 2007 Center for Bioinformatics, University of Hamburg

  Permission to use, copy, modify, and distribute this software for any
  purpose with or without fee is hereby granted, provided that the above
  copyright notice and this permission notice appear in all copies.

  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
*/

#include "core/cstr_array_api.h"
#include "core/error_api.h"
#include "core/option_api.h"
#include "core/str_api.h"
#include "core/tool.h"
#include "core/toolbox.h"
#include "core/versionfunc_api.h"
#include "match/sfx-run.h"
#include "tools/gt_packedindex.h"
#include "tools/gt_packedindex_mkctxmap.h"
#include "tools/gt_packedindex_trsuftab.h"
#include "tools/gt_packedindex_chk_integrity.h"
#include "tools/gt_packedindex_chk_search.h"

/* rely on suffixerator for on the fly index construction */
static int gt_packedindex_make(int argc, const char *argv[], GtError *err)
{
  return gt_parseargsandcallsuffixerator(false, argc, argv, err);
}

static void* gt_packedindex_arguments_new(void)
{
  GtToolbox *packedindex_toolbox = gt_toolbox_new();
  gt_toolbox_add(packedindex_toolbox, "mkindex", gt_packedindex_make);
  gt_toolbox_add(packedindex_toolbox, "mkctxmap", gt_packedindex_mkctxmap);
  gt_toolbox_add(packedindex_toolbox, "trsuftab", gt_packedindex_trsuftab);
  gt_toolbox_add(packedindex_toolbox, "chkintegrity",
              gt_packedindex_chk_integrity );
  gt_toolbox_add(packedindex_toolbox, "chksearch", gt_packedindex_chk_search);
  return packedindex_toolbox;
}

static void gt_packedindex_arguments_delete(void *tool_arguments)
{
  GtToolbox *index_toolbox = tool_arguments;
  if (!index_toolbox) return;
  gt_toolbox_delete(index_toolbox);
}

static GtOptionParser* gt_packedindex_option_parser_new(void *tool_arguments)
{
  GtToolbox *index_toolbox = tool_arguments;
  GtOptionParser *op;
  gt_assert(index_toolbox);
  op = gt_option_parser_new("[option ...] index_tool [argument ...]",
                         "Call apacked index subtool and pass argument(s) to "
                         "it.");
  gt_option_parser_set_comment_func(op, gt_toolbox_show, index_toolbox);
  gt_option_parser_set_min_args(op, 1);
  gt_option_parser_refer_to_manual(op);
  return op;
}

static int gt_packedindex_runner(int argc, const char **argv, int parsed_args,
                                 void *tool_arguments, GtError *err)
{
  GtToolbox *index_toolbox = tool_arguments;
  GtToolfunc toolfunc;
  GtTool *tool = NULL;
  char **nargv = NULL;
  int had_err = 0;

  gt_error_check(err);
  gt_assert(index_toolbox);

  /* determine tool */
  if (!gt_toolbox_has_tool(index_toolbox, argv[parsed_args])) {
    gt_error_set(err, "packedindex tool '%s' not found; option -help lists "
                   "possible tools", argv[parsed_args]);
    had_err = -1;
  }

  /* call sub-tool */
  if (!had_err) {
    if (!(toolfunc = gt_toolbox_get(index_toolbox, argv[parsed_args]))) {
      tool = gt_toolbox_get_tool(index_toolbox, argv[parsed_args]);
      gt_assert(tool);
    }
    nargv = gt_cstr_array_prefix_first(argv + parsed_args,
                                       gt_error_get_progname(err));
    gt_error_set_progname(err, nargv[0]);
    if (toolfunc)
      had_err = toolfunc(argc - parsed_args, (const char**) nargv, err);
    else
      had_err = gt_tool_run(tool, argc - parsed_args, (const char**) nargv,
                            err);
  }

  gt_cstr_array_delete(nargv);
  return had_err;
}

GtTool* gt_packedindex(void)
{
  GtTool *tool = gt_tool_new(gt_packedindex_arguments_new,
                             gt_packedindex_arguments_delete,
                             gt_packedindex_option_parser_new,
                             NULL,
                             gt_packedindex_runner);
  gt_tool_set_toolbox_new(tool,
                          (GtToolToolboxNew) gt_packedindex_arguments_new);
  return tool;
}
