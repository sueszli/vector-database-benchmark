/**
 * Copyright (c) 2016 DeepCortex GmbH <legal@eventql.io>
 * Authors:
 *   - Paul Asmuth <paul@eventql.io>
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License ("the license") as
 * published by the Free Software Foundation, either version 3 of the License,
 * or any later version.
 *
 * In accordance with Section 7(e) of the license, the licensing of the Program
 * under the license does not imply a trademark license. Therefore any rights,
 * title and interest in our trademarks remain entirely with us.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the license for more details.
 *
 * You can be released from the requirements of the license by purchasing a
 * commercial license. Buying such a license is mandatory as soon as you develop
 * commercial activities involving this program without disclosing the source
 * code of your own applications
 */
#include <eventql/util/stringutil.h>
#include <eventql/util/csv/CSVOutputStream.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

CSVOutputStream::CSVOutputStream(
    std::unique_ptr<OutputStream> output_stream,
    String col_sep /* = ';' */,
    String row_sep /* = '\n' */) :
    output_(std::move(output_stream)),
    col_sep_(col_sep),
    row_sep_(row_sep) {}

void CSVOutputStream::appendRow(const Vector<String>& row) {
  Buffer buf;

  for (int i = 0; i < row.size(); ++i) {
    if (i > 0) {
      buf.append(col_sep_);
    }

    buf.append(row[i].data(), row[i].size());
  }

  buf.append(row_sep_);
  output_->write(buf);
}
