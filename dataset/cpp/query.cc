/**
 * Copyright (c) 2017 DeepCortex GmbH <legal@eventql.io>
 * Authors:
 *   - Laura Schlimmer <laura@eventql.io>
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
#include <unistd.h>
#include "eventql/eventql.h"
#include "eventql/util/io/fileutil.h"
#include "eventql/util/io/inputstream.h"
#include "../test_runner.h"
#include "process.h"

namespace eventql {
namespace test {

const std::string kSQLFileEnding = ".sql";
const std::string kResultFileEnding = ".result.txt";
const std::string kOutputFileEnding = ".actual.result.txt";

void executeTestQuery(
    TestContext* ctx,
    const std::string& query_path,
    const std::string& host,
    const std::string& port,
    const std::string& database) {
  auto sql_file_path = FileUtil::joinPaths(ctx->srcdir, query_path);
  auto output_file_path = sql_file_path
      .substr(0, sql_file_path.size() - kSQLFileEnding.size())
      + kOutputFileEnding;
  auto result_file_path = sql_file_path
      .substr(0, sql_file_path.size() - kSQLFileEnding.size())
      + kResultFileEnding;

  Process::runOrDie(
      FileUtil::joinPaths(ctx->bindir, "evql"),
      {
        "--file", sql_file_path,
        "--host", host,
        "--port", port,
        "--database", database,
        "--output_file", output_file_path
      },
      {},
      "evql-test-query<" + query_path + ">",
      ctx->log_fd);


  auto result_is = FileInputStream::openFile(result_file_path);
  auto output_is = FileInputStream::openFile(output_file_path);

  std::string result_line;
  std::string output_line;
  while (output_is->readLine(&output_line)) {
    if (!result_is->readLine(&result_line)) {
      RAISE(kRuntimeError, "too many rows returned");
    }

    StringUtil::chomp(&result_line);
    StringUtil::chomp(&output_line);

    if (result_line != output_line) {
      RAISE(
          kRuntimeError,
          StringUtil::format("expected $0 to be $1", output_line, result_line));
    }

    result_line.clear();
    output_line.clear();
  }

  if (result_is->readLine(&result_line)) {
    RAISE(kRuntimeError, "not enough rows returned");
  }
}

void executeTestQueryWithRetries(
    TestContext* ctx,
    const std::string& query_path,
    const std::string& host,
    const std::string& port,
    const std::string& database,
    size_t retry_count) {
  while (retry_count > 0) {
    --retry_count;

    try {
      executeTestQuery(ctx, query_path, host, port, database);
      return;
    } catch (const std::exception& e) {
      if (retry_count == 0) {
        throw std::runtime_error(e.what());
      } else {
        usleep(1000000);
      }
    }
  }
}

void executeQueryAndExpectSuccess(
    TestContext* ctx,
    const std::string& query_path,
    const std::string& host,
    const std::string& port,
    const std::string& database) {
  auto sql_file_path = FileUtil::joinPaths(ctx->srcdir, query_path);

  Process::runOrDie(
      FileUtil::joinPaths(ctx->bindir, "evql"),
      {
        "--file", sql_file_path,
        "--host", host,
        "--port", port,
        "--database", database
      },
      {},
      "evql<" + query_path + ">",
      ctx->log_fd);
}

} // namespace test
} // namespace eventql
