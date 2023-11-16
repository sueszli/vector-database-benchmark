/*
  Copyright (c) 2014 Sascha Steinbiss <ss34@sanger.ac.uk>

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

#include "core/ma_api.h"
#include "core/unused_api.h"
#include "extended/check_boundaries_visitor_api.h"
#include "extended/gff3_in_stream.h"
#include "extended/visitor_stream.h"
#include "tools/gt_loccheck.h"

typedef struct {
} GFF3ValidatorArguments;

static GtOptionParser* gt_loccheck_option_parser_new(GT_UNUSED
                                                         void *tool_arguments)
{
  GtOptionParser *op;

  /* init */
  op = gt_option_parser_new("[GFF3_file ...]",
                            "Checks parent-child containment in GFF3 input.");

  return op;
}

static int gt_loccheck_runner(int argc, const char **argv, int parsed_args,
                              GT_UNUSED void *tool_arguments, GtError *err)
{
  GtNodeStream *gff3_in_stream = NULL, *checker_stream = NULL;
  int had_err = 0;
  gt_error_check(err);

  gff3_in_stream = gt_gff3_in_stream_new_unsorted(argc - parsed_args,
                                                  argv + parsed_args);

  checker_stream = gt_visitor_stream_new(gff3_in_stream,
                                         gt_check_boundaries_visitor_new());
  gt_assert(checker_stream);

  /* pull the features through the stream and free them afterwards */
  if (!had_err)
    had_err = gt_node_stream_pull(checker_stream, err);

  /* free */
  gt_node_stream_delete(gff3_in_stream);
  gt_node_stream_delete(checker_stream);

  return had_err;
}

GtTool* gt_loccheck(void)
{
  return gt_tool_new(NULL,
                     NULL,
                     gt_loccheck_option_parser_new,
                     NULL,
                     gt_loccheck_runner);
}
