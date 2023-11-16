/*
  Copyright (c) 2014 Dirk Willrodt <willrodt@zbh.uni-hamburg.de>
  Copyright (c) 2014 Center for Bioinformatics, University of Hamburg

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

#include "core/assert_api.h"
#include "core/types_api.h"

#include "extended/io_function_pointers.h"

int gt_io_error_fwrite(void *ptr, size_t size, size_t nmemb, FILE *stream,
                       GtError *err)
{
  int had_err = 0;
  if (nmemb != fwrite((const void*) ptr, size, nmemb, stream)) {
    gt_error_set(err,
                 "fwrite failed to write " GT_WU " elements of "
                 "size " GT_WU, (GtUword) nmemb, (GtUword) size);
    had_err = 1;
  }
  return had_err;
}

int gt_io_error_fread(void *ptr, size_t size, size_t nmemb, FILE *stream,
                       GtError *err)
{
  int had_err = 0;
  if (nmemb != fread(ptr, size, nmemb, stream)) {
    had_err = 1;
    if (feof(stream) != 0) {
      gt_error_set(err,
                   "fread reached eof trying to read " GT_WU " elements of "
                   "size " GT_WU, (GtUword) nmemb, (GtUword) size);
    }
    else {
      gt_error_set(err,
                   "fread failed to read " GT_WU " elements of "
                   "size " GT_WU, (GtUword) nmemb, (GtUword) size);
    }
  }
  return had_err;
}
