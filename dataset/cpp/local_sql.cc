/**
 * Copyright (c) 2016 DeepCortex GmbH <legal@eventql.io>
 * Authors:
 *   - Paul Asmuth <paul@eventql.io>
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
#include <iostream>
#include <iomanip>
#include <eventql/cli/benchmarks/local_sql.h>
#include <eventql/util/cli/flagparser.h>
#include <eventql/util/wallclock.h>
#include <eventql/sql/CSTableScanProvider.h>

namespace eventql {
namespace cli {

const String LocalSQLBenchmark::kName_ = "local-sql";
const String LocalSQLBenchmark::kDescription_ = "Benchmark the SQL engine";

LocalSQLBenchmark::LocalSQLBenchmark() :
    verbose_(false),
    num_requests_(-1),
    request_counter_(0) {}

Status LocalSQLBenchmark::execute(
    const std::vector<std::string>& argv,
    FileInputStream* stdin_is,
    OutputStream* stdout_os,
    OutputStream* stderr_os) {
  ::cli::FlagParser flags;

  flags.defineFlag(
      "query",
      ::cli::FlagParser::T_STRING,
      true,
      NULL,
      NULL,
      "str",
      "<str>");

  flags.defineFlag(
      "input_cstable",
      ::cli::FlagParser::T_STRING,
      false,
      NULL,
      NULL,
      "str",
      "<str>");

  flags.defineFlag(
      "verbose",
      ::cli::FlagParser::T_SWITCH,
      false,
      NULL,
      NULL,
      "flag",
      "<flag>");

  flags.defineFlag(
      "num",
      ::cli::FlagParser::T_INTEGER,
      false,
      "n",
      NULL,
      "total number of request (inifite if unset)",
      "<rate>");

  flags.parseArgv(argv);
  query_ = flags.getString("query");
  if (flags.isSet("verbose")) {
    verbose_ = true;
  }

  auto runtime = csql::Runtime::getDefaultRuntime();
  auto tables = mkRef(new csql::TableRepository());

  for (const auto& t : flags.getStrings("input_cstable")) {
    auto parts = StringUtil::split(t, ":");
    if (parts.size() != 2) {
      return ReturnCode::errorf("EARG", "invalid argument: ", t);
    }

    tables->addProvider(new csql::CSTableScanProvider(parts[0], parts[1]));
  }

  if (flags.isSet("num")) {
    num_requests_ = flags.getInt("num");
  }

  for (size_t i = 0; i < num_requests_; i++) {
    auto rc = runQuery(runtime.get(), tables.get());

    if (!rc.isSuccess()) {
      return rc;
    }
  }

  if (runtimes_us_.size() > 0) {
    std::sort(runtimes_us_.begin(), runtimes_us_.end());
    auto min = runtimes_us_[0];
    auto max = runtimes_us_[runtimes_us_.size() - 1];
    uint32_t total = 0;
    for (auto i : runtimes_us_) {
      total += i;
    }

    auto mean = total / double(runtimes_us_.size());

    uint32_t median;
    if (runtimes_us_.size() % 2 == 0) {
      auto left = runtimes_us_[(runtimes_us_.size() - 1) / 2];
      auto right = runtimes_us_[runtimes_us_.size() / 2];
      median = left + right / 2;
    } else {
      median = runtimes_us_[(runtimes_us_.size() - 1) / 2];
    }

    uint32_t std_dev_sum = 0;
    for (auto i : runtimes_us_) {
      std_dev_sum += pow((i - mean), 2);
    }

    auto std_dev = std_dev_sum / double(runtimes_us_.size());

    std::cout
        << std::endl
        << std::endl
        << std::left
        << std::setw(22)
        << "Query: "
        << query_
        << std::endl
        << std::setw(22)
        << "Complete iterations: "
        << num_requests_
        << std::endl
        << std::setw(22)
        << "Total runtime: "
        << total / 1000000.f << " s"
        << std::endl
        << std::endl
        << "Runtimes (ms)"
        << std::endl
        << std::right
        << std::setw(8)
        << "min"
        << std::setw(8)
        << "mean"
        << std::setw(8)
        << "[+/-sd]"
        << std::setw(8)
        << "median"
        << std::setw(8)
        << "max"
        << std::endl
        << std::setprecision(3) << std::fixed
        << std::setw(8)
        << min / 1000.0f
        << std::setw(8)
        << mean / 1000.0f
        << std::setw(8)
        << std_dev / 1000.0f
        << std::setw(8)
        << median / 1000.0f
        << std::setw(8)
        << max / 1000.0f
        << std::endl
        << std::endl
        << "Percentage of the runtimes"
        << std::endl
        << std::setw(5)
        << "50%"
        << std::setw(8)
        << runtimes_us_[(runtimes_us_.size() - 1) * 0.5] / 1000.f
        << " ms"
        << std::endl
        << std::setw(5)
        << "66%"
        << std::setw(8)
        << runtimes_us_[(runtimes_us_.size() - 1) * 0.66] / 1000.f
        << " ms"
        << std::endl
        << std::setw(5)
        << "75%"
        << std::setw(8)
        << runtimes_us_[(runtimes_us_.size() - 1) * 0.75] / 1000.f
        << " ms"
        << std::endl
        << std::setw(5)
        << "80%"
        << std::setw(8)
        << runtimes_us_[(runtimes_us_.size() - 1) * 0.8] / 1000.f
        << " ms"
        << std::endl
        << std::setw(5)
        << "90%"
        << std::setw(8)
        << runtimes_us_[(runtimes_us_.size() - 1) * 0.9] / 1000.f
        << " ms"
        << std::endl
        << std::setw(5)
        << "95%"
        << std::setw(8)
        << runtimes_us_[(runtimes_us_.size() - 1) * 0.95] / 1000.f
        << " ms"
        << std::endl
        << std::setw(5)
        << "98%"
        << std::setw(8)
        << runtimes_us_[(runtimes_us_.size() - 1) * 0.98] / 1000.f
        << " ms"
        << std::endl
        << std::setw(5)
        << "99%"
        << std::setw(8)
        << runtimes_us_[(runtimes_us_.size() - 1) * 0.99] / 1000.f
        << " ms"
        << std::endl
        << std::setw(5)
        << "100%"
        << std::setw(8)
        << runtimes_us_[runtimes_us_.size() - 1] / 1000.f
        << " ms"
        << std::endl;
  }

  return Status::success();
}


ReturnCode LocalSQLBenchmark::runQuery(
    csql::Runtime* runtime,
    csql::TableProvider* tables) {
  auto txn = runtime->newTransaction();
  txn->setTableProvider(tables);

  auto t0 = WallClock::unixMicros();

  csql::ResultList result;
  try {
    auto qplan = runtime->buildQueryPlan(txn.get(), query_);
    qplan->execute(0, &result);
  } catch (const std::exception& e) {
    return ReturnCode::error("ERUNTIME", e.what());
  }

  auto t1 = WallClock::unixMicros();
  auto num_rows = result.getNumRows();
  auto runtime_us = t1 - t0;

  if (num_requests_) {
    runtimes_us_.emplace_back(runtime_us);
  }

  if (verbose_) {
    result.debugPrint();

    std::cout
        << "took " << runtime_us / 1000.0f << "ms" << "; "
        << num_rows << " rows returned" << std::endl;

  } else if (
      fmod(++request_counter_, num_requests_ / double(kDefaultNumPrints)) == 0) {
    std::cout
      << request_counter_ << "/"
      << num_requests_ << " requests"
      << std::endl;
  }

  return ReturnCode::success();
}

const String& LocalSQLBenchmark::getName() const {
  return kName_;
}

const String& LocalSQLBenchmark::getDescription() const {
  return kDescription_;
}

void LocalSQLBenchmark::printHelp(OutputStream* stdout_os) const {
  std::cout << StringUtil::format(
      "evqlbench-$0 - $1\n\n",
      kName_,
      kDescription_);

  std::cout <<
      "Usage: evqlbench local-sql [OPTIONS]\n\n"
      "   -q, --query <query>        Specify the SQL query to run\n\n"
      "   -n, --num   <num>           Maximum total number of request (default is infinite)\n";

}

} // namespace cli
} // namespace eventql

