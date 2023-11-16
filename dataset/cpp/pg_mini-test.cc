// Copyright (c) YugaByte, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.  You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied.  See the License for the specific language governing permissions and limitations
// under the License.
//

#include <atomic>
#include <optional>
#include <thread>

#include <boost/preprocessor/seq/for_each.hpp>

#include <gtest/gtest.h>

#include "yb/common/common_flags.h"
#include "yb/common/pgsql_error.h"

#include "yb/dockv/value_type.h"

#include "yb/integration-tests/mini_cluster.h"

#include "yb/master/catalog_entity_info.h"
#include "yb/master/catalog_manager_if.h"
#include "yb/master/mini_master.h"
#include "yb/master/sys_catalog.h"
#include "yb/master/sys_catalog_constants.h"
#include "yb/master/ts_manager.h"
#include "yb/rocksdb/db.h"

#include "yb/server/skewed_clock.h"

#include "yb/tserver/mini_tablet_server.h"
#include "yb/tserver/pg_client_service.h"
#include "yb/tserver/tablet_server.h"

#include "yb/tablet/tablet.h"
#include "yb/tablet/tablet_peer.h"
#include "yb/tablet/transaction_participant.h"

#include "yb/gutil/casts.h"

#include "yb/util/atomic.h"
#include "yb/util/backoff_waiter.h"
#include "yb/util/debug-util.h"
#include "yb/util/enums.h"
#include "yb/util/random_util.h"
#include "yb/util/range.h"
#include "yb/util/metrics.h"
#include "yb/util/scope_exit.h"
#include "yb/util/status_log.h"
#include "yb/util/test_macros.h"
#include "yb/util/test_thread_holder.h"
#include "yb/util/tsan_util.h"

#include "yb/yql/pggate/pggate_flags.h"

#include "yb/yql/pgwrapper/pg_mini_test_base.h"
#include "yb/yql/pgwrapper/pg_test_utils.h"

using std::string;

using namespace std::literals;

DECLARE_bool(TEST_force_master_leader_resolution);
DECLARE_bool(enable_automatic_tablet_splitting);
DECLARE_bool(enable_pg_savepoints);
DECLARE_bool(enable_tracing);
DECLARE_bool(flush_rocksdb_on_shutdown);
DECLARE_bool(enable_wait_queues);

DECLARE_double(TEST_respond_write_failed_probability);
DECLARE_double(TEST_transaction_ignore_applying_probability);

DECLARE_int32(TEST_txn_participant_inject_latency_on_apply_update_txn_ms);
DECLARE_int32(heartbeat_interval_ms);
DECLARE_int32(history_cutoff_propagation_interval_ms);
DECLARE_int32(timestamp_history_retention_interval_sec);
DECLARE_int32(timestamp_syscatalog_history_retention_interval_sec);
DECLARE_int32(tracing_level);
DECLARE_int32(tserver_heartbeat_metrics_interval_ms);
DECLARE_int32(txn_max_apply_batch_records);
DECLARE_int32(yb_num_shards_per_tserver);

DECLARE_int64(TEST_inject_random_delay_on_txn_status_response_ms);
DECLARE_int64(apply_intents_task_injected_delay_ms);
DECLARE_int64(db_block_size_bytes);
DECLARE_int64(db_filter_block_size_bytes);
DECLARE_int64(db_index_block_size_bytes);
DECLARE_int64(db_write_buffer_size);
DECLARE_int64(tablet_force_split_threshold_bytes);
DECLARE_int64(tablet_split_high_phase_shard_count_per_node);
DECLARE_int64(tablet_split_high_phase_size_threshold_bytes);
DECLARE_int64(tablet_split_low_phase_shard_count_per_node);
DECLARE_int64(tablet_split_low_phase_size_threshold_bytes);

DECLARE_uint64(max_clock_skew_usec);

DECLARE_string(time_source);

DECLARE_bool(rocksdb_disable_compactions);
DECLARE_uint64(pg_client_session_expiration_ms);
DECLARE_uint64(pg_client_heartbeat_interval_ms);

METRIC_DECLARE_entity(tablet);
METRIC_DECLARE_gauge_uint64(aborted_transactions_pending_cleanup);

namespace yb::pgwrapper {
namespace {

Result<int64_t> GetCatalogVersion(PGConn* conn) {
  if (FLAGS_ysql_enable_db_catalog_version_mode) {
    const auto db_oid = VERIFY_RESULT(conn->FetchRow<int32>(Format(
        "SELECT oid FROM pg_database WHERE datname = '$0'", PQdb(conn->get()))));
    return conn->FetchRow<PGUint64>(
        Format("SELECT current_version FROM pg_yb_catalog_version where db_oid = $0", db_oid));
  }
  return conn->FetchRow<PGUint64>("SELECT current_version FROM pg_yb_catalog_version");
}

Result<bool> IsCatalogVersionChangedDuringDdl(PGConn* conn, const std::string& ddl_query) {
  const auto initial_version = VERIFY_RESULT(GetCatalogVersion(conn));
  RETURN_NOT_OK(conn->Execute(ddl_query));
  return initial_version != VERIFY_RESULT(GetCatalogVersion(conn));
}

} // namespace

class PgMiniTest : public PgMiniTestBase {
 protected:
  // Have several threads doing updates and several threads doing large scans in parallel.
  // If deferrable is true, then the scans are in deferrable transactions, so no read restarts are
  // expected.
  // Otherwise, the scans are in transactions with snapshot isolation, but we still don't expect any
  // read restarts to be observed because they should be transparently handled on the postgres side.
  void TestReadRestart(bool deferrable = true);

  void TestForeignKey(IsolationLevel isolation);

  void TestBigInsert(bool restart);

  void CreateTableAndInitialize(std::string table_name, int num_tablets);

  void DestroyTable(std::string table_name);

  void StartReadWriteThreads(std::string table_name, TestThreadHolder *thread_holder);

  void TestConcurrentDeleteRowAndUpdateColumn(bool select_before_update);

  void CreateDBWithTablegroupAndTables(
      const std::string database_name, const std::string tablegroup_name, const int num_tables,
      const int keys, PGConn* conn);

  void VerifyFileSizeAfterCompaction(PGConn* conn, const int num_tables);

  void RunManyConcurrentReadersTest();

  void ValidateAbortedTxnMetric();

  int64_t GetBloomFilterCheckedMetric();
};

class PgMiniTestSingleNode : public PgMiniTest {
  size_t NumTabletServers() override {
    return 1;
  }
};

class PgMiniTestFailOnConflict : public PgMiniTest {
 protected:
  void SetUp() override {
    // This test depends on fail-on-conflict concurrency control to perform its validation.
    // TODO(wait-queues): https://github.com/yugabyte/yugabyte-db/issues/17871
    ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_wait_queues) = false;
    PgMiniTest::SetUp();
  }
};

class PgMiniPgClientServiceCleanupTest : public PgMiniTestSingleNode {
 public:
  void SetUp() override {
    ANNOTATE_UNPROTECTED_WRITE(FLAGS_pg_client_session_expiration_ms) = 5000;
    ANNOTATE_UNPROTECTED_WRITE(FLAGS_pg_client_heartbeat_interval_ms) = 2000;
    PgMiniTestBase::SetUp();
  }
};

TEST_F_EX(PgMiniTest, VerifyPgClientServiceCleanupQueue, PgMiniPgClientServiceCleanupTest) {
  constexpr size_t kTotalConnections = 30;
  std::vector<PGConn> connections;
  connections.reserve(kTotalConnections);
  for (size_t i = 0; i < kTotalConnections; ++i) {
    connections.push_back(ASSERT_RESULT(Connect()));
  }
  auto* client_service =
      cluster_->mini_tablet_server(0)->server()->TEST_GetPgClientService();
  ASSERT_EQ(connections.size(), client_service->TEST_SessionsCount());

  connections.erase(connections.begin() + connections.size() / 2, connections.end());
  ASSERT_OK(WaitFor([client_service, expected_count = connections.size()]() {
    return client_service->TEST_SessionsCount() == expected_count;
  }, 4 * FLAGS_pg_client_session_expiration_ms * 1ms, "client session cleanup", 1s));
}

// Try to change this to test follower reads.
TEST_F(PgMiniTest, FollowerReads) {
  auto conn = ASSERT_RESULT(Connect());
  ASSERT_OK(conn.Execute("CREATE TABLE t2 (key int PRIMARY KEY, word TEXT, phrase TEXT)"));
  ASSERT_OK(conn.Execute("INSERT INTO t2 (key, word, phrase) VALUES (1, 'old', 'old is gold')"));
  ASSERT_OK(conn.Execute("INSERT INTO t2 (key, word, phrase) VALUES (2, 'NEW', 'NEW is fine')"));

  ASSERT_OK(conn.Execute("CREATE TABLE t (key INT PRIMARY KEY, value TEXT)"));
  ASSERT_OK(conn.Execute("INSERT INTO t (key, value) VALUES (1, 'old')"));

  ASSERT_OK(conn.Execute("SET yb_debug_log_docdb_requests = true"));
  ASSERT_OK(conn.Execute("SET yb_read_from_followers = true"));

  // Try to set a value < 2 * max_clock_skew (500ms) should fail.
  ASSERT_NOK(conn.Execute(Format("SET yb_follower_read_staleness_ms = $0", 400)));
  ASSERT_NOK(conn.Execute(Format("SET yb_follower_read_staleness_ms = $0", 999)));
  // Setting a value > 2 * max_clock_skew should work.
  ASSERT_OK(conn.Execute(Format("SET yb_follower_read_staleness_ms = $0", 1001)));

  // Setting staleness to what we require for the test.
  // Sleep and then perform an update, such that follower reads should see the old value.
  // But current reads will see the new/updated value.
  constexpr int32_t kStalenessMs = 4000;
  SleepFor(MonoDelta::FromMilliseconds(kStalenessMs));
  ASSERT_OK(conn.Execute("UPDATE t SET value = 'NEW' WHERE key = 1"));
  auto kUpdateTime = MonoTime::Now();
  ASSERT_OK(conn.Execute(Format("SET yb_follower_read_staleness_ms = $0", kStalenessMs)));

  // Follower reads will not be enabled unless a transaction block is marked read-only.
  {
    ASSERT_OK(conn.Execute("BEGIN TRANSACTION"));
    auto value = ASSERT_RESULT(conn.FetchRow<std::string>("SELECT value FROM t WHERE key = 1"));
    ASSERT_EQ(value, "NEW");
    ASSERT_OK(conn.Execute("COMMIT"));
  }

  // Follower reads will be enabled for transaction block(s) marked read-only.
  {
    ASSERT_OK(conn.Execute("BEGIN TRANSACTION READ ONLY"));
    auto value = ASSERT_RESULT(conn.FetchRow<std::string>("SELECT value FROM t WHERE key = 1"));
    ASSERT_EQ(value, "old");
    ASSERT_OK(conn.Execute("COMMIT"));
  }

  // Follower reads will not be enabled unless the session or statement is marked read-only.
  auto value = ASSERT_RESULT(conn.FetchRow<std::string>("SELECT value FROM t WHERE key = 1"));
  ASSERT_EQ(value, "NEW");

  value = ASSERT_RESULT(conn.FetchRow<std::string>(
      "SELECT phrase FROM t, t2 WHERE t.value = t2.word"));
  ASSERT_EQ(value, "NEW is fine");

  // Follower reads can be enabled for a single statement with a pg hint.
  value = ASSERT_RESULT(conn.FetchRow<std::string>(
      "/*+ Set(transaction_read_only on) */ SELECT value FROM t WHERE key = 1"));
  ASSERT_EQ(value, "old");
  value = ASSERT_RESULT(conn.FetchRow<std::string>(
      "/*+ Set(transaction_read_only on) */ SELECT phrase FROM t, t2 WHERE t.value = t2.word"));
  ASSERT_EQ(value, "old is gold");

  // pg_hint only applies for the specific statement used.
  // Statements following it should not enable follower reads if it is not marked read-only.
  value = ASSERT_RESULT(conn.FetchRow<std::string>("SELECT value FROM t WHERE key = 1"));
  ASSERT_EQ(value, "NEW");

  // pg_hint should also apply for prepared statements, if the hint is provided
  // at PREPARE stage.
  {
    ASSERT_OK(
        conn.Execute("PREPARE hinted_select_stmt (int) AS "
                     "/*+ Set(transaction_read_only on) */ "
                     "SELECT value FROM t WHERE key = $1"));
    value = ASSERT_RESULT(conn.FetchRow<std::string>("EXECUTE hinted_select_stmt (1)"));
    ASSERT_EQ(value, "old");
    value = ASSERT_RESULT(conn.FetchRow<std::string>(
        "/*+ Set(transaction_read_only on) */ EXECUTE hinted_select_stmt (1)"));
    ASSERT_EQ(value, "old");
    value = ASSERT_RESULT(conn.FetchRow<std::string>(
        "/*+ Set(transaction_read_only off) */ EXECUTE hinted_select_stmt (1)"));
    ASSERT_EQ(value, "old");
  }
  // Adding a pg_hint at the EXECUTE stage has no effect.
  {
    ASSERT_OK(
        conn.Execute("PREPARE select_stmt (int) AS "
                     "SELECT value FROM t WHERE key = $1"));
    value = ASSERT_RESULT(conn.FetchRow<std::string>("EXECUTE select_stmt (1)"));
    ASSERT_EQ(value, "NEW");
    value = ASSERT_RESULT(conn.FetchRow<std::string>(
        "/*+ Set(transaction_read_only on) */ EXECUTE select_stmt (1)"));
    ASSERT_EQ(value, "NEW");
  }

  // pg_hint with func()
  // The hint may be provided when the function is defined, or when the function is
  // called.
  {
    ASSERT_OK(
        conn.Execute("CREATE FUNCTION func() RETURNS text AS"
                     " $$ SELECT value FROM t WHERE key = 1 $$ LANGUAGE SQL"));
    value = ASSERT_RESULT(conn.FetchRow<std::string>("SELECT func()"));
    ASSERT_EQ(value, "NEW");
    value = ASSERT_RESULT(conn.FetchRow<std::string>(
        "/*+ Set(transaction_read_only off) */ SELECT func()"));
    ASSERT_EQ(value, "NEW");
    value = ASSERT_RESULT(conn.FetchRow<std::string>(
        "/*+ Set(transaction_read_only on) */ SELECT func()"));
    ASSERT_EQ(value, "old");
    ASSERT_OK(conn.Execute("DROP FUNCTION func()"));
  }
  {
    ASSERT_OK(
        conn.Execute("CREATE FUNCTION hinted_func() RETURNS text AS"
                     " $$ "
                     "/*+ Set(transaction_read_only on) */ "
                     "SELECT value FROM t WHERE key = 1"
                     " $$ LANGUAGE SQL"));
    value = ASSERT_RESULT(conn.FetchRow<std::string>("SELECT hinted_func()"));
    ASSERT_EQ(value, "old");
    value = ASSERT_RESULT(conn.FetchRow<std::string>(
        "/*+ Set(transaction_read_only off) */ SELECT hinted_func()"));
    ASSERT_EQ(value, "old");
    value = ASSERT_RESULT(conn.FetchRow<std::string>(
        "/*+ Set(transaction_read_only on) */ SELECT hinted_func()"));
    ASSERT_EQ(value, "old");
    ASSERT_OK(conn.Execute("DROP FUNCTION hinted_func()"));
  }

  ASSERT_OK(conn.Execute("SET default_transaction_read_only = true"));
  // Follower reads will be enabled since the session is marked read-only.
  value = ASSERT_RESULT(conn.FetchRow<std::string>("SELECT value FROM t WHERE key = 1"));
  ASSERT_EQ(value, "old");

  // pg_hint can only mark a session from read-write to read-only.
  // Marking a statement in a read-only session as read-write is not allowed.
  // Writes operations fail.
  // Read operations will be performed as if they are read-write, so will not use follower
  // reads.
  {
    auto status = conn.Execute(
        "/*+ Set(transaction_read_only off) */ "
        "UPDATE t SET value = 'NEWER' WHERE key = 1");
    ASSERT_EQ(PgsqlError(status), YBPgErrorCode::YB_PG_READ_ONLY_SQL_TRANSACTION) << status;
    ASSERT_STR_CONTAINS(status.ToString(), "cannot execute UPDATE in a read-only transaction");

    value = ASSERT_RESULT(conn.FetchRow<std::string>(
        "/*+ Set(transaction_read_only off) */ SELECT value FROM t WHERE key = 1"));
    ASSERT_EQ(value, "old");
  }

  // After sufficient time has passed, even "follower reads" should see the newer value.
  SleepFor(kUpdateTime + MonoDelta::FromMilliseconds(kStalenessMs) - MonoTime::Now());

  ASSERT_OK(conn.Execute("SET default_transaction_read_only = false"));
  value = ASSERT_RESULT(conn.FetchRow<std::string>("SELECT value FROM t WHERE key = 1"));
  ASSERT_EQ(value, "NEW");

  ASSERT_OK(conn.Execute("SET default_transaction_read_only = true"));
  value = ASSERT_RESULT(conn.FetchRow<std::string>("SELECT value FROM t WHERE key = 1"));
  ASSERT_EQ(value, "NEW");
  value = ASSERT_RESULT(conn.FetchRow<std::string>(
      "/*+ Set(transaction_read_only on) */ SELECT phrase FROM t, t2 WHERE t.value = t2.word"));
  ASSERT_EQ(value, "NEW is fine");
}

TEST_F(PgMiniTest, MultiColFollowerReads) {
  auto conn = ASSERT_RESULT(Connect());
  ASSERT_OK(conn.Execute("CREATE TABLE t (k int PRIMARY KEY, c1 TEXT, c2 TEXT)"));
  ASSERT_OK(conn.Execute("SET yb_debug_log_docdb_requests = true"));
  ASSERT_OK(conn.Execute("SET yb_read_from_followers = true"));

  constexpr int32_t kSleepTimeMs = 1200 * kTimeMultiplier;

  ASSERT_OK(conn.Execute("INSERT INTO t (k, c1, c2) VALUES (1, 'old', 'old')"));
  auto kUpdateTime0 = MonoTime::Now();

  SleepFor(MonoDelta::FromMilliseconds(kSleepTimeMs));

  ASSERT_OK(conn.Execute("UPDATE t SET c1 = 'NEW' WHERE k = 1"));
  auto kUpdateTime1 = MonoTime::Now();

  SleepFor(MonoDelta::FromMilliseconds(kSleepTimeMs));

  ASSERT_OK(conn.Execute("UPDATE t SET c2 = 'NEW' WHERE k = 1"));
  auto kUpdateTime2 = MonoTime::Now();

  auto row = ASSERT_RESULT((conn.FetchRow<int32_t, std::string, std::string>(
      "/*+ Set(transaction_read_only off) */ SELECT * FROM t WHERE k = 1")));
  ASSERT_EQ(row, (decltype(row){1, "NEW", "NEW"}));

  const int32_t kOpDurationMs = 10;
  auto staleness_ms = (MonoTime::Now() - kUpdateTime0).ToMilliseconds() - kOpDurationMs;
  ASSERT_OK(conn.Execute(Format("SET yb_follower_read_staleness_ms = $0", staleness_ms)));
  row = ASSERT_RESULT((conn.FetchRow<int32_t, std::string, std::string>(
      "/*+ Set(transaction_read_only on) */ SELECT * FROM t WHERE k = 1")));
  ASSERT_EQ(row, (decltype(row){1, "old", "old"}));

  staleness_ms = (MonoTime::Now() - kUpdateTime1).ToMilliseconds() - kOpDurationMs;
  ASSERT_OK(conn.Execute(Format("SET yb_follower_read_staleness_ms = $0", staleness_ms)));
  row = ASSERT_RESULT((conn.FetchRow<int32_t, std::string, std::string>(
      "/*+ Set(transaction_read_only on) */ SELECT * FROM t WHERE k = 1")));
  ASSERT_EQ(row, (decltype(row){1, "NEW", "old"}));

  SleepFor(MonoDelta::FromMilliseconds(kSleepTimeMs));

  staleness_ms = (MonoTime::Now() - kUpdateTime2).ToMilliseconds();
  ASSERT_OK(conn.Execute(Format("SET yb_follower_read_staleness_ms = $0", staleness_ms)));
  row = ASSERT_RESULT((conn.FetchRow<int32_t, std::string, std::string>(
      "/*+ Set(transaction_read_only on) */ SELECT * FROM t WHERE k = 1")));
  ASSERT_EQ(row, (decltype(row){1, "NEW", "NEW"}));
}

TEST_F(PgMiniTest, Simple) {
  auto conn = ASSERT_RESULT(Connect());

  ASSERT_OK(conn.Execute("CREATE TABLE t (key INT PRIMARY KEY, value TEXT)"));
  ASSERT_OK(conn.Execute("INSERT INTO t (key, value) VALUES (1, 'hello')"));

  auto value = ASSERT_RESULT(conn.FetchRow<std::string>("SELECT value FROM t WHERE key = 1"));
  ASSERT_EQ(value, "hello");
}

TEST_F(PgMiniTest, Tracing) {
  class TraceLogSink : public google::LogSink {
   public:
    void send(
        google::LogSeverity severity, const char* full_filename, const char* base_filename,
        int line, const struct ::tm* tm_time, const char* message, size_t message_len) {
      if (strcmp(base_filename, "trace.cc") == 0) {
        last_logged_bytes_ = message_len;
      }
    }

    size_t last_logged_bytes() const { return last_logged_bytes_; }

   private:
    std::atomic<size_t> last_logged_bytes_{0};
  } trace_log_sink;
  google::AddLogSink(&trace_log_sink);
  size_t last_logged_trace_size;

  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_tracing) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_tracing_level) = 1;
  auto conn = ASSERT_RESULT(Connect());

  ASSERT_OK(conn.Execute("CREATE TABLE t (key INT PRIMARY KEY, value TEXT, value2 TEXT)"));
  LOG(INFO) << "Setting yb_enable_docdb_tracing";
  ASSERT_OK(conn.Execute("SET yb_enable_docdb_tracing = true"));

  LOG(INFO) << "Doing Insert";
  ASSERT_OK(conn.Execute("INSERT INTO t (key, value, value2) VALUES (1, 'hello', 'world')"));
  SleepFor(1s);
  last_logged_trace_size = trace_log_sink.last_logged_bytes();
  LOG(INFO) << "Logged " << last_logged_trace_size << " bytes";
  // 2601 is size of the current trace for insert.
  // being a little conservative for changes in ports/ip addr etc.
  ASSERT_GE(last_logged_trace_size, 2400);
  LOG(INFO) << "Done Insert";

  LOG(INFO) << "Doing Select";
  auto value = ASSERT_RESULT(conn.FetchRow<std::string>("SELECT value FROM t WHERE key = 1"));
  SleepFor(1s);
  last_logged_trace_size = trace_log_sink.last_logged_bytes();
  LOG(INFO) << "Logged " << last_logged_trace_size << " bytes";
  // 1884 is size of the current trace for select.
  // being a little conservative for changes in ports/ip addr etc.
  ASSERT_GE(last_logged_trace_size, 1600);
  ASSERT_EQ(value, "hello");
  LOG(INFO) << "Done Select";

  LOG(INFO) << "Doing block transaction";
  ASSERT_OK(conn.Execute("BEGIN TRANSACTION"));
  ASSERT_OK(conn.Execute("INSERT INTO t (key, value, value2) VALUES (2, 'good', 'morning')"));
  ASSERT_OK(conn.Execute("INSERT INTO t (key, value, value2) VALUES (3, 'good', 'morning')"));
  value = ASSERT_RESULT(conn.FetchRow<std::string>("SELECT value FROM t WHERE key = 1"));
  ASSERT_OK(conn.Execute("ABORT"));
  SleepFor(1s);
  last_logged_trace_size = trace_log_sink.last_logged_bytes();
  LOG(INFO) << "Logged " << last_logged_trace_size << " bytes";
  // 5446 is size of the current trace for the transaction block.
  // being a little conservative for changes in ports/ip addr etc.
  ASSERT_GE(last_logged_trace_size, 5200);
  LOG(INFO) << "Done block transaction";

  google::RemoveLogSink(&trace_log_sink);
  ValidateAbortedTxnMetric();
}

TEST_F(PgMiniTest, TracingSushant) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_tracing) = false;
  auto conn = ASSERT_RESULT(Connect());

  LOG(INFO) << "Setting yb_enable_docdb_tracing";
  ASSERT_OK(conn.Execute("SET yb_enable_docdb_tracing = true"));
  LOG(INFO) << "Doing Create";
  ASSERT_OK(conn.Execute("create table t (h int, r int, v int, primary key (h, r));"));
  LOG(INFO) << "Done Create";
  LOG(INFO) << "Doing Insert";
  ASSERT_OK(conn.Execute("insert into t  values (1,3,1), (1,4,1);"));
  LOG(INFO) << "Done Insert";
}

TEST_F(PgMiniTest, WriteRetry) {
  constexpr int kKeys = 100;
  auto conn = ASSERT_RESULT(Connect());

  ASSERT_OK(conn.Execute("CREATE TABLE t (key INT PRIMARY KEY)"));

  SetAtomicFlag(0.25, &FLAGS_TEST_respond_write_failed_probability);

  LOG(INFO) << "Insert " << kKeys << " keys";
  for (int key = 0; key != kKeys; ++key) {
    auto status = conn.ExecuteFormat("INSERT INTO t (key) VALUES ($0)", key);
    ASSERT_TRUE(status.ok() || PgsqlError(status) == YBPgErrorCode::YB_PG_UNIQUE_VIOLATION ||
                status.ToString().find("Duplicate request") != std::string::npos) << status;
  }

  SetAtomicFlag(0, &FLAGS_TEST_respond_write_failed_probability);

  auto result = ASSERT_RESULT(conn.FetchMatrix("SELECT * FROM t ORDER BY key", kKeys, 1));
  for (int key = 0; key != kKeys; ++key) {
    auto fetched_key = ASSERT_RESULT(GetValue<int32_t>(result.get(), key, 0));
    ASSERT_EQ(fetched_key, key);
  }

  LOG(INFO) << "Insert duplicate key";
  auto status = conn.Execute("INSERT INTO t (key) VALUES (1)");
  ASSERT_EQ(PgsqlError(status), YBPgErrorCode::YB_PG_UNIQUE_VIOLATION) << status;
  ASSERT_STR_CONTAINS(status.ToString(), "duplicate key value violates unique constraint");
}

TEST_F(PgMiniTest, With) {
  auto conn = ASSERT_RESULT(Connect());

  ASSERT_OK(conn.Execute("CREATE TABLE test (k int PRIMARY KEY, v int)"));

  ASSERT_OK(conn.Execute(
      "WITH test2 AS (UPDATE test SET v = 2 WHERE k = 1) "
      "UPDATE test SET v = 3 WHERE k = 1"));
}

void PgMiniTest::TestReadRestart(const bool deferrable) {
  constexpr CoarseDuration kWaitTime = 60s;
  constexpr int kKeys = 100;
  constexpr int kNumReadThreads = 8;
  constexpr int kNumUpdateThreads = 8;
  constexpr int kRequiredNumReads = 500;
  constexpr std::chrono::milliseconds kClockSkew = -100ms;
  std::atomic<int> num_read_restarts(0);
  std::atomic<int> num_read_successes(0);
  TestThreadHolder thread_holder;

  // Set up table
  auto setup_conn = ASSERT_RESULT(Connect());
  ASSERT_OK(setup_conn.Execute("CREATE TABLE t (key INT PRIMARY KEY, value INT)"));
  for (int key = 0; key != kKeys; ++key) {
    ASSERT_OK(setup_conn.Execute(Format("INSERT INTO t (key, value) VALUES ($0, 0)", key)));
  }

  // Introduce clock skew
  auto delta_changers = SkewClocks(cluster_.get(), kClockSkew);

  // Start read threads
  for (int i = 0; i < kNumReadThreads; ++i) {
    thread_holder.AddThreadFunctor([this, deferrable, &num_read_restarts, &num_read_successes,
                                    &stop = thread_holder.stop_flag()] {
      auto read_conn = ASSERT_RESULT(Connect());
      while (!stop.load(std::memory_order_acquire)) {
        if (deferrable) {
          ASSERT_OK(read_conn.Execute("BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE, READ ONLY, "
                                      "DEFERRABLE"));
        } else {
          ASSERT_OK(read_conn.Execute("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ"));
        }
        auto result = read_conn.FetchMatrix("SELECT * FROM t", kKeys, 2);
        if (!result.ok()) {
          ASSERT_TRUE(result.status().IsNetworkError()) << result.status();
          ASSERT_EQ(PgsqlError(result.status()), YBPgErrorCode::YB_PG_T_R_SERIALIZATION_FAILURE)
              << result.status();
          ASSERT_STR_CONTAINS(result.status().ToString(), "Restart read");
          ++num_read_restarts;
          ASSERT_OK(read_conn.Execute("ABORT"));
          break;
        } else {
          ASSERT_OK(read_conn.Execute("COMMIT"));
          ++num_read_successes;
        }
      }
    });
  }

  // Start update threads
  for (int i = 0; i < kNumUpdateThreads; ++i) {
    thread_holder.AddThreadFunctor([this, i, &stop = thread_holder.stop_flag()] {
      auto update_conn = ASSERT_RESULT(Connect());
      while (!stop.load(std::memory_order_acquire)) {
        for (int key = i; key < kKeys; key += kNumUpdateThreads) {
          ASSERT_OK(update_conn.Execute("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ"));
          ASSERT_OK(update_conn.Execute(
              Format("UPDATE t SET value = value + 1 WHERE key = $0", key)));
          ASSERT_OK(update_conn.Execute("COMMIT"));
        }
      }
    });
  }

  // Stop threads after a while
  thread_holder.WaitAndStop(kWaitTime);

  // Count successful reads
  int num_reads = (num_read_restarts.load(std::memory_order_acquire)
                   + num_read_successes.load(std::memory_order_acquire));
  LOG(INFO) << "Successful reads: " << num_read_successes.load(std::memory_order_acquire) << "/"
      << num_reads;
  ASSERT_EQ(num_read_restarts.load(std::memory_order_acquire), 0);
  ASSERT_GT(num_read_successes.load(std::memory_order_acquire), kRequiredNumReads);
  ValidateAbortedTxnMetric();
}

class PgMiniLargeClockSkewTest : public PgMiniTest {
 public:
  void SetUp() override {
    server::SkewedClock::Register();
    ANNOTATE_UNPROTECTED_WRITE(FLAGS_time_source) = server::SkewedClock::kName;
    SetAtomicFlag(250000ULL, &FLAGS_max_clock_skew_usec);
    PgMiniTestBase::SetUp();
  }
};

TEST_F_EX(PgMiniTest, YB_DISABLE_TEST_IN_SANITIZERS(ReadRestartSerializableDeferrable),
          PgMiniLargeClockSkewTest) {
  TestReadRestart(true /* deferrable */);
}

TEST_F_EX(PgMiniTest, YB_DISABLE_TEST_IN_SANITIZERS(ReadRestartSnapshot),
          PgMiniLargeClockSkewTest) {
  TestReadRestart(false /* deferrable */);
}

TEST_F_EX(PgMiniTest, SerializableReadOnly, PgMiniTestFailOnConflict) {
  PGConn read_conn = ASSERT_RESULT(Connect());
  PGConn setup_conn = ASSERT_RESULT(Connect());
  PGConn write_conn = ASSERT_RESULT(Connect());

  // Set up table
  ASSERT_OK(setup_conn.Execute("CREATE TABLE t (i INT)"));
  ASSERT_OK(setup_conn.Execute("INSERT INTO t (i) VALUES (0)"));

  // SERIALIZABLE, READ ONLY should use snapshot isolation
  ASSERT_OK(read_conn.Execute("BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE, READ ONLY"));
  ASSERT_OK(write_conn.Execute("BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE, READ WRITE"));
  ASSERT_OK(write_conn.Execute("UPDATE t SET i = i + 1"));
  ASSERT_OK(read_conn.Fetch("SELECT * FROM t"));
  ASSERT_OK(read_conn.Execute("COMMIT"));
  ASSERT_OK(write_conn.Execute("COMMIT"));

  // READ ONLY, SERIALIZABLE should use snapshot isolation
  ASSERT_OK(read_conn.Execute("BEGIN TRANSACTION READ ONLY, ISOLATION LEVEL SERIALIZABLE"));
  ASSERT_OK(write_conn.Execute("BEGIN TRANSACTION READ WRITE, ISOLATION LEVEL SERIALIZABLE"));
  ASSERT_OK(read_conn.Fetch("SELECT * FROM t"));
  ASSERT_OK(write_conn.Execute("UPDATE t SET i = i + 1"));
  ASSERT_OK(read_conn.Execute("COMMIT"));
  ASSERT_OK(write_conn.Execute("COMMIT"));

  // SHOW for READ ONLY should show serializable
  ASSERT_OK(read_conn.Execute("BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE, READ ONLY"));
  ASSERT_EQ(ASSERT_RESULT(read_conn.FetchRow<std::string>("SHOW transaction_isolation")),
            "serializable");
  ASSERT_OK(read_conn.Execute("COMMIT"));

  // SHOW for READ WRITE to READ ONLY should show serializable and read_only
  ASSERT_OK(write_conn.Execute("BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE, READ WRITE"));
  ASSERT_OK(write_conn.Execute("SET TRANSACTION READ ONLY"));
  ASSERT_EQ(ASSERT_RESULT(write_conn.FetchRow<std::string>("SHOW transaction_isolation")),
            "serializable");
  ASSERT_EQ(ASSERT_RESULT(write_conn.FetchRow<std::string>("SHOW transaction_read_only")), "on");
  ASSERT_OK(write_conn.Execute("COMMIT"));

  // SERIALIZABLE, READ ONLY to READ WRITE should not use snapshot isolation
  ASSERT_OK(read_conn.Execute("BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE, READ ONLY"));
  ASSERT_OK(write_conn.Execute("BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE, READ WRITE"));
  ASSERT_OK(read_conn.Execute("SET TRANSACTION READ WRITE"));
  ASSERT_OK(write_conn.Execute("UPDATE t SET i = i + 1"));
  // The result of the following statement is probabilistic.  If it does not fail now, then it
  // should fail during COMMIT.
  auto s = ResultToStatus(read_conn.Fetch("SELECT * FROM t"));
  if (s.ok()) {
    ASSERT_OK(read_conn.Execute("COMMIT"));
    Status status = write_conn.Execute("COMMIT");
    ASSERT_NOK(status);
    ASSERT_TRUE(status.IsNetworkError()) << status;
    ASSERT_EQ(PgsqlError(status), YBPgErrorCode::YB_PG_T_R_SERIALIZATION_FAILURE) << status;
  } else {
    ASSERT_TRUE(s.IsNetworkError()) << s;
    ASSERT_TRUE(IsSerializeAccessError(s)) << s;
    ASSERT_STR_CONTAINS(s.ToString(), "conflicts with higher priority transaction");
  }
}

void AssertAborted(const Status& status) {
  ASSERT_NOK(status);
  ASSERT_STR_CONTAINS(status.ToString(), "aborted");
}

TEST_F_EX(PgMiniTest, SelectModifySelect, PgMiniTestFailOnConflict) {
  {
    auto read_conn = ASSERT_RESULT(Connect());
    auto write_conn = ASSERT_RESULT(Connect());

    ASSERT_OK(read_conn.Execute("CREATE TABLE t (i INT)"));
    ASSERT_OK(read_conn.Execute("BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE"));
    ASSERT_RESULT(read_conn.FetchMatrix("SELECT * FROM t", 0, 1));
    ASSERT_OK(write_conn.Execute("INSERT INTO t VALUES (1)"));
    ASSERT_NO_FATALS(AssertAborted(ResultToStatus(read_conn.Fetch("SELECT * FROM t"))));
  }
  {
    auto read_conn = ASSERT_RESULT(Connect());
    auto write_conn = ASSERT_RESULT(Connect());

    ASSERT_OK(read_conn.Execute("CREATE TABLE t2 (i INT PRIMARY KEY)"));
    ASSERT_OK(read_conn.Execute("INSERT INTO t2 VALUES (1)"));

    ASSERT_OK(read_conn.Execute("BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE"));
    ASSERT_RESULT(read_conn.FetchMatrix("SELECT * FROM t2", 1, 1));
    ASSERT_OK(write_conn.Execute("DELETE FROM t2 WHERE i = 1"));
    ASSERT_NO_FATALS(AssertAborted(ResultToStatus(read_conn.Fetch("SELECT * FROM t2"))));
  }
}

class PgMiniSmallWriteBufferTest : public PgMiniTest {
 public:
  void SetUp() override {
    ANNOTATE_UNPROTECTED_WRITE(FLAGS_db_write_buffer_size) = 256_KB;
    PgMiniTest::SetUp();
  }
};

TEST_F(PgMiniTest, TruncateColocatedBigTable) {
  // Simulate truncating a big colocated table with multiple sst files flushed to disk.
  // To repro issue https://github.com/yugabyte/yugabyte-db/issues/15206
  // When using bloom filter, it might fail to find the table tombstone because it's stored in
  // a different sst file than the key is currently seeking.

  ANNOTATE_UNPROTECTED_WRITE(FLAGS_rocksdb_disable_compactions) = true;
  auto conn = ASSERT_RESULT(Connect());
  ASSERT_OK(conn.Execute("create tablegroup tg1"));
  ASSERT_OK(conn.Execute("create table t1(k int primary key) tablegroup tg1"));
  const auto& peers = ListTabletPeers(cluster_.get(), ListPeersFilter::kLeaders);
  tablet::TabletPeerPtr tablet_peer = nullptr;
  for (auto peer : peers) {
    if (peer->shared_tablet()->regular_db()) {
      tablet_peer = peer;
      break;
    }
  }
  ASSERT_NE(tablet_peer, nullptr);

  // Insert 2 rows, and flush.
  ASSERT_OK(conn.Execute("insert into t1 values (1)"));
  ASSERT_OK(conn.Execute("insert into t1 values (2)"));
  ASSERT_OK(tablet_peer->shared_tablet()->Flush(tablet::FlushMode::kSync));

  // Truncate the table, and flush. Tabletombstone should be in a seperate sst file.
  ASSERT_OK(conn.Execute("TRUNCATE t1"));
  SleepFor(1s);
  ASSERT_OK(tablet_peer->shared_tablet()->Flush(tablet::FlushMode::kSync));

  // Check if the row still visible.
  ASSERT_OK(conn.FetchMatrix("select k from t1 where k = 1", 0, 1));

  // Check if hit dup key error.
  ASSERT_OK(conn.Execute("insert into t1 values (1)"));
}

TEST_F_EX(PgMiniTest, BulkCopyWithRestart, PgMiniSmallWriteBufferTest) {
  const std::string kTableName = "key_value";
  auto conn = ASSERT_RESULT(Connect());
  ASSERT_OK(conn.ExecuteFormat(
      "CREATE TABLE $0 (key INTEGER NOT NULL PRIMARY KEY, value VARCHAR)",
      kTableName));

  TestThreadHolder thread_holder;
  constexpr int kTotalBatches = RegularBuildVsSanitizers(50, 5);
  constexpr int kBatchSize = 1000;
  constexpr int kValueSize = 128;

  std::atomic<int> key(0);

  thread_holder.AddThreadFunctor([this, &kTableName, &stop = thread_holder.stop_flag(), &key] {
    SetFlagOnExit set_flag(&stop);
    auto connection = ASSERT_RESULT(Connect());

    auto se = ScopeExit([&key] {
      LOG(INFO) << "Total keys: " << key;
    });

    while (!stop.load(std::memory_order_acquire) && key < kBatchSize * kTotalBatches) {
      ASSERT_OK(connection.CopyBegin(Format("COPY $0 FROM STDIN WITH BINARY", kTableName)));
      for (int j = 0; j != kBatchSize; ++j) {
        connection.CopyStartRow(2);
        connection.CopyPutInt32(++key);
        connection.CopyPutString(RandomHumanReadableString(kValueSize));
      }

      ASSERT_OK(connection.CopyEnd());
    }
  });

  thread_holder.AddThread(RestartsThread(cluster_.get(), 5s, &thread_holder.stop_flag()));

  thread_holder.WaitAndStop(120s); // Actually will stop when enough batches were copied

  ASSERT_EQ(key.load(std::memory_order_relaxed), kTotalBatches * kBatchSize);

  LOG(INFO) << "Restarting cluster";
  ASSERT_OK(RestartCluster());

  ASSERT_OK(WaitFor([this, &key, &kTableName] {
    auto intents_count = CountIntents(cluster_.get());
    LOG(INFO) << "Intents count: " << intents_count;

    if (intents_count <= 5000) {
      return true;
    }

    // We cleanup only transactions that were completely aborted/applied before last replication
    // happens.
    // So we could get into situation when intents of the last transactions are not cleaned.
    // To avoid such scenario in this test we write one more row to allow cleanup.
    // As the previous connection might have been dead (from the cluster restart), do the insert
    // from a new connection.
    auto new_conn = EXPECT_RESULT(Connect());
    EXPECT_OK(new_conn.ExecuteFormat("INSERT INTO $0 VALUES ($1, '$2')",
              kTableName, ++key, RandomHumanReadableString(kValueSize)));
    return false;
  }, 10s * kTimeMultiplier, "Intents cleanup", 200ms));
}

void PgMiniTest::TestForeignKey(IsolationLevel isolation_level) {
  const std::string kDataTable = "data";
  const std::string kReferenceTable = "reference";
  constexpr int kRows = 10;
  auto conn = ASSERT_RESULT(Connect());

  ASSERT_OK(conn.ExecuteFormat(
      "CREATE TABLE $0 (id int NOT NULL, name VARCHAR, PRIMARY KEY (id))",
      kReferenceTable));
  ASSERT_OK(conn.ExecuteFormat(
      "CREATE TABLE $0 (ref_id INTEGER, data_id INTEGER, name VARCHAR, "
          "PRIMARY KEY (ref_id, data_id))",
      kDataTable));
  ASSERT_OK(conn.ExecuteFormat(
      "ALTER TABLE $0 ADD CONSTRAINT fk FOREIGN KEY(ref_id) REFERENCES $1(id) "
          "ON DELETE CASCADE",
      kDataTable, kReferenceTable));

  ASSERT_OK(conn.ExecuteFormat(
      "INSERT INTO $0 VALUES ($1, 'reference_$1')", kReferenceTable, 1));

  for (int i = 1; i <= kRows; ++i) {
    ASSERT_OK(conn.StartTransaction(isolation_level));
    ASSERT_OK(conn.ExecuteFormat(
        "INSERT INTO $0 VALUES ($1, $2, 'data_$2')", kDataTable, 1, i));
    ASSERT_OK(conn.CommitTransaction());
  }

  ASSERT_OK(WaitFor([this] {
    return CountIntents(cluster_.get()) == 0;
  }, 15s, "Intents cleanup"));
}

TEST_F(PgMiniTest, ForeignKeySerializable) {
  TestForeignKey(IsolationLevel::SERIALIZABLE_ISOLATION);
}

TEST_F(PgMiniTest, ForeignKeySnapshot) {
  TestForeignKey(IsolationLevel::SNAPSHOT_ISOLATION);
}

TEST_F(PgMiniTest, ConcurrentSingleRowUpdate) {
  auto conn = ASSERT_RESULT(Connect());
  ASSERT_OK(conn.Execute("CREATE TABLE t(k INT PRIMARY KEY, counter INT)"));
  ASSERT_OK(conn.Execute("INSERT INTO t VALUES(1, 0)"));
  const size_t thread_count = 10;
  const size_t increment_per_thread = 5;
  {
    CountDownLatch latch(thread_count);
    TestThreadHolder thread_holder;
    for (size_t i = 0; i < thread_count; ++i) {
      thread_holder.AddThreadFunctor([this, &stop = thread_holder.stop_flag(), &latch] {
        auto thread_conn = ASSERT_RESULT(Connect());
        latch.CountDown();
        latch.Wait();
        for (size_t j = 0; j < increment_per_thread; ++j) {
          ASSERT_OK(thread_conn.Execute("UPDATE t SET counter = counter + 1 WHERE k = 1"));
        }
      });
    }
  }
  auto counter = ASSERT_RESULT(conn.FetchRow<int32_t>("SELECT counter FROM t WHERE k = 1"));
  ASSERT_EQ(thread_count * increment_per_thread, counter);
}

TEST_F(PgMiniTest, DropDBUpdateSysTablet) {
  const std::string kDatabaseName = "testdb";
  PGConn conn = ASSERT_RESULT(Connect());
  std::array<int, 4> num_tables;

  auto* catalog_manager = &ASSERT_RESULT(cluster_->GetLeaderMiniMaster())->catalog_manager();
  auto sys_tablet = ASSERT_RESULT(catalog_manager->GetTabletInfo(master::kSysCatalogTabletId));
  {
    auto tablet_lock = sys_tablet->LockForWrite();
    num_tables[0] = tablet_lock->pb.table_ids_size();
  }
  ASSERT_OK(conn.ExecuteFormat("CREATE DATABASE $0", kDatabaseName));
  {
    auto tablet_lock = sys_tablet->LockForWrite();
    num_tables[1] = tablet_lock->pb.table_ids_size();
  }
  ASSERT_OK(conn.ExecuteFormat("DROP DATABASE $0", kDatabaseName));
  {
    auto tablet_lock = sys_tablet->LockForWrite();
    num_tables[2] = tablet_lock->pb.table_ids_size();
  }
  // Make sure that the system catalog tablet table_ids is persisted.
  ASSERT_OK(RestartCluster());
  catalog_manager = &ASSERT_RESULT(cluster_->GetLeaderMiniMaster())->catalog_manager();
  sys_tablet = ASSERT_RESULT(catalog_manager->GetTabletInfo(master::kSysCatalogTabletId));
  {
    auto tablet_lock = sys_tablet->LockForWrite();
    num_tables[3] = tablet_lock->pb.table_ids_size();
  }
  ASSERT_LT(num_tables[0], num_tables[1]);
  ASSERT_EQ(num_tables[0], num_tables[2]);
  ASSERT_EQ(num_tables[0], num_tables[3]);
}

TEST_F(PgMiniTest, DropDBMarkDeleted) {
  const std::string kDatabaseName = "testdb";
  constexpr auto kSleepTime = 500ms;
  constexpr int kMaxNumSleeps = 20;
  auto *catalog_manager = &ASSERT_RESULT(cluster_->GetLeaderMiniMaster())->catalog_manager();
  PGConn conn = ASSERT_RESULT(Connect());

  ASSERT_FALSE(catalog_manager->AreTablesDeleting());
  ASSERT_OK(conn.ExecuteFormat("CREATE DATABASE $0", kDatabaseName));
  ASSERT_OK(conn.ExecuteFormat("DROP DATABASE $0", kDatabaseName));
  // System tables should be deleting then deleted.
  int num_sleeps = 0;
  while (catalog_manager->AreTablesDeleting() && (num_sleeps++ != kMaxNumSleeps)) {
    LOG(INFO) << "Tables are deleting...";
    std::this_thread::sleep_for(kSleepTime);
  }
  ASSERT_FALSE(catalog_manager->AreTablesDeleting()) << "Tables should have finished deleting";
  // Make sure that the table deletions are persisted.
  ASSERT_OK(RestartCluster());
  // Refresh stale local variable after RestartSync.
  catalog_manager = &ASSERT_RESULT(cluster_->GetLeaderMiniMaster())->catalog_manager();
  ASSERT_FALSE(catalog_manager->AreTablesDeleting());
}

TEST_F(PgMiniTest, DropDBWithTables) {
  const std::string kDatabaseName = "testdb";
  const std::string kTablePrefix = "testt";
  constexpr auto kSleepTime = 500ms;
  constexpr int kMaxNumSleeps = 20;
  int num_tables_before, num_tables_after;
  auto *catalog_manager = &ASSERT_RESULT(cluster_->GetLeaderMiniMaster())->catalog_manager();
  PGConn conn = ASSERT_RESULT(Connect());
  auto sys_tablet = ASSERT_RESULT(catalog_manager->GetTabletInfo(master::kSysCatalogTabletId));

  {
    auto tablet_lock = sys_tablet->LockForWrite();
    num_tables_before = tablet_lock->pb.table_ids_size();
  }
  ASSERT_OK(conn.ExecuteFormat("CREATE DATABASE $0", kDatabaseName));
  {
    PGConn conn_new = ASSERT_RESULT(ConnectToDB(kDatabaseName));
    for (int i = 0; i < 10; ++i) {
      ASSERT_OK(conn_new.ExecuteFormat("CREATE TABLE $0$1 (i int)", kTablePrefix, i));
    }
    ASSERT_OK(conn_new.ExecuteFormat("INSERT INTO $0$1 (i) VALUES (1), (2), (3)", kTablePrefix, 5));
  }
  ASSERT_OK(conn.ExecuteFormat("DROP DATABASE $0", kDatabaseName));
  // User and system tables should be deleting then deleted.
  int num_sleeps = 0;
  while (catalog_manager->AreTablesDeleting() && (num_sleeps++ != kMaxNumSleeps)) {
    LOG(INFO) << "Tables are deleting...";
    std::this_thread::sleep_for(kSleepTime);
  }
  ASSERT_FALSE(catalog_manager->AreTablesDeleting()) << "Tables should have finished deleting";
  // Make sure that the table deletions are persisted.
  ASSERT_OK(RestartCluster());
  catalog_manager = &ASSERT_RESULT(cluster_->GetLeaderMiniMaster())->catalog_manager();
  sys_tablet = ASSERT_RESULT(catalog_manager->GetTabletInfo(master::kSysCatalogTabletId));
  ASSERT_FALSE(catalog_manager->AreTablesDeleting());
  {
    auto tablet_lock = sys_tablet->LockForWrite();
    num_tables_after = tablet_lock->pb.table_ids_size();
  }
  ASSERT_EQ(num_tables_before, num_tables_after);
}

TEST_F(PgMiniTest, BigSelect) {
  auto conn = ASSERT_RESULT(Connect());

  ASSERT_OK(conn.Execute("CREATE TABLE t (key INT PRIMARY KEY, value TEXT)"));

  constexpr size_t kRows = 400;
  constexpr size_t kValueSize = RegularBuildVsSanitizers(256_KB, 4_KB);

  for (size_t i = 0; i != kRows; ++i) {
    ASSERT_OK(conn.ExecuteFormat(
        "INSERT INTO t VALUES ($0, '$1')", i, RandomHumanReadableString(kValueSize)));
  }

  auto start = MonoTime::Now();
  auto res = ASSERT_RESULT(conn.FetchRow<PGUint64>("SELECT COUNT(DISTINCT(value)) FROM t"));
  auto finish = MonoTime::Now();
  LOG(INFO) << "Time: " << finish - start;
  ASSERT_EQ(res, kRows);
}

TEST_F(PgMiniTest, MoveMaster) {
  ShutdownAllMasters(cluster_.get());
  cluster_->mini_master(0)->set_pass_master_addresses(false);
  ASSERT_OK(StartAllMasters(cluster_.get()));

  auto conn = ASSERT_RESULT(Connect());

  ASSERT_OK(WaitFor([&conn] {
    auto status = conn.Execute("CREATE TABLE t (key INT PRIMARY KEY)");
    WARN_NOT_OK(status, "Failed to create table");
    return status.ok();
  }, 15s, "Create table"));
}

TEST_F(PgMiniTest, DDLWithRestart) {
  SetAtomicFlag(1.0, &FLAGS_TEST_transaction_ignore_applying_probability);
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_TEST_force_master_leader_resolution) = true;

  auto conn = ASSERT_RESULT(Connect());

  ASSERT_OK(conn.StartTransaction(IsolationLevel::SERIALIZABLE_ISOLATION));
  ASSERT_OK(conn.Execute("CREATE TABLE t (a int PRIMARY KEY)"));
  ASSERT_OK(conn.CommitTransaction());

  ShutdownAllMasters(cluster_.get());

  LOG(INFO) << "Start masters";
  ASSERT_OK(StartAllMasters(cluster_.get()));

  auto res = ASSERT_RESULT(conn.FetchRow<PGUint64>("SELECT COUNT(*) FROM t"));
  ASSERT_EQ(res, 0);
}

TEST_F(PgMiniTest, CreateDatabase) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_flush_rocksdb_on_shutdown) = false;
  auto conn = ASSERT_RESULT(Connect());
  const std::string kDatabaseName = "testdb";
  ASSERT_OK(conn.ExecuteFormat("CREATE DATABASE $0", kDatabaseName));
  ASSERT_OK(RestartCluster());
}

void PgMiniTest::TestBigInsert(bool restart) {
  constexpr int64_t kNumRows = RegularBuildVsSanitizers(100000, 10000);
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_txn_max_apply_batch_records) = kNumRows / 10;

  auto conn = ASSERT_RESULT(Connect());
  ASSERT_OK(conn.Execute("CREATE TABLE t (a int PRIMARY KEY) SPLIT INTO 1 TABLETS"));
  ASSERT_OK(conn.Execute("INSERT INTO t VALUES (0)"));

  TestThreadHolder thread_holder;

  std::atomic<int> post_insert_reads{0};
  std::atomic<bool> restarted{false};
  thread_holder.AddThreadFunctor(
      [this, &stop = thread_holder.stop_flag(), &post_insert_reads, &restarted] {
    auto connection = ASSERT_RESULT(Connect());
    while (!stop.load(std::memory_order_acquire)) {
      auto res = connection.FetchRow<PGUint64>("SELECT SUM(a) FROM t");
      if (!res.ok()) {
        auto msg = res.status().message().ToBuffer();
        ASSERT_TRUE(msg.find("server closed the connection unexpectedly") != std::string::npos)
            << res.status();
        while (!restarted.load() && !stop.load()) {
          std::this_thread::sleep_for(10ms);
        }
        std::this_thread::sleep_for(1s);
        LOG(INFO) << "Establishing new connection";
        connection = ASSERT_RESULT(Connect());
        restarted = false;
        continue;
      }

      // We should see zero or full sum only.
      if (*res) {
        ASSERT_EQ(*res, kNumRows * (kNumRows + 1) / 2);
        ++post_insert_reads;
      }
    }
  });

  ASSERT_OK(conn.ExecuteFormat(
      "INSERT INTO t SELECT generate_series(1, $0)", kNumRows));

  if (restart) {
    LOG(INFO) << "Restart cluster";
    ASSERT_OK(RestartCluster());
    restarted = true;
  }

  ASSERT_OK(WaitFor([this, &post_insert_reads] {
    auto intents_count = CountIntents(cluster_.get());
    LOG(INFO) << "Intents count: " << intents_count;

    return intents_count == 0 && post_insert_reads.load(std::memory_order_acquire) > 0;
  }, 60s * kTimeMultiplier, "Intents cleanup", 200ms));

  thread_holder.Stop();

  FlushAndCompactTablets();

  auto peers = ListTabletPeers(cluster_.get(), ListPeersFilter::kAll);
  for (const auto& peer : peers) {
    auto db = peer->tablet()->regular_db();
    if (!db) {
      continue;
    }
    rocksdb::ReadOptions read_opts;
    read_opts.query_id = rocksdb::kDefaultQueryId;
    std::unique_ptr<rocksdb::Iterator> iter(db->NewIterator(read_opts));

    for (iter->SeekToFirst(); ASSERT_RESULT(iter->CheckedValid()); iter->Next()) {
      Slice key = iter->key();
      ASSERT_FALSE(key.TryConsumeByte(dockv::KeyEntryTypeAsChar::kTransactionApplyState))
          << "Key: " << iter->key().ToDebugString() << ", value: " << iter->value().ToDebugString();
    }
  }
}

TEST_F(PgMiniTest, BigInsert) {
  TestBigInsert(/* restart= */ false);
}

TEST_F(PgMiniTest, BigInsertWithRestart) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_apply_intents_task_injected_delay_ms) = 200;
  TestBigInsert(/* restart= */ true);
}

TEST_F(PgMiniTest, BigInsertWithDropTable) {
  constexpr int kNumRows = 10000;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_txn_max_apply_batch_records) = kNumRows / 10;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_apply_intents_task_injected_delay_ms) = 200;
  auto conn = ASSERT_RESULT(Connect());
  ASSERT_OK(conn.Execute("CREATE TABLE t(id int) SPLIT INTO 1 TABLETS"));
  ASSERT_OK(conn.ExecuteFormat(
      "INSERT INTO t SELECT generate_series(1, $0)", kNumRows));
  ASSERT_OK(conn.Execute("DROP TABLE t"));
}

void PgMiniTest::TestConcurrentDeleteRowAndUpdateColumn(bool select_before_update) {
  auto conn1 = ASSERT_RESULT(Connect());
  auto conn2 = ASSERT_RESULT(Connect());
  ASSERT_OK(conn1.Execute("CREATE TABLE t (i INT PRIMARY KEY, j INT)"));
  ASSERT_OK(conn1.Execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)"));
  ASSERT_OK(conn1.StartTransaction(IsolationLevel::SNAPSHOT_ISOLATION));
  if (select_before_update) {
    ASSERT_OK(conn1.Fetch("SELECT * FROM t"));
  }
  ASSERT_OK(conn2.Execute("DELETE FROM t WHERE i = 2"));
  auto status = conn1.Execute("UPDATE t SET j = 21 WHERE i = 2");
  if (select_before_update) {
    ASSERT_TRUE(IsSerializeAccessError(status)) << status;
    ASSERT_STR_CONTAINS(status.message().ToBuffer(), "Value write after transaction start");
    return;
  }
  ASSERT_OK(status);
  ASSERT_OK(conn1.CommitTransaction());
  const auto rows = ASSERT_RESULT((conn1.FetchRows<int32_t, int32_t>(
      "SELECT * FROM t ORDER BY i")));
  const decltype(rows) expected_rows = {{1, 10}, {3, 30}};
  ASSERT_EQ(rows, expected_rows);
}

TEST_F(PgMiniTest, ConcurrentDeleteRowAndUpdateColumn) {
  TestConcurrentDeleteRowAndUpdateColumn(/* select_before_update= */ false);
}

TEST_F(PgMiniTest, ConcurrentDeleteRowAndUpdateColumnWithSelect) {
  TestConcurrentDeleteRowAndUpdateColumn(/* select_before_update= */ true);
}

// The test checks catalog version is updated only in case of changes in sys catalog.
TEST_F(PgMiniTest, CatalogVersionUpdateIfNeeded) {
  auto conn = ASSERT_RESULT(Connect());
  const auto schema_ddl = "CREATE SCHEMA IF NOT EXISTS test";
  const auto first_create_schema = ASSERT_RESULT(
      IsCatalogVersionChangedDuringDdl(&conn, schema_ddl));
  ASSERT_TRUE(first_create_schema);
  const auto second_create_schema = ASSERT_RESULT(
      IsCatalogVersionChangedDuringDdl(&conn, schema_ddl));
  ASSERT_FALSE(second_create_schema);
  ASSERT_OK(conn.Execute("CREATE TABLE t (k INT PRIMARY KEY)"));
  const auto add_column_ddl = "ALTER TABLE t ADD COLUMN IF NOT EXISTS v INT";
  const auto first_add_column = ASSERT_RESULT(
      IsCatalogVersionChangedDuringDdl(&conn, add_column_ddl));
  ASSERT_TRUE(first_add_column);
  const auto second_add_column = ASSERT_RESULT(
      IsCatalogVersionChangedDuringDdl(&conn, add_column_ddl));
  ASSERT_FALSE(second_add_column);
}

// Test that we don't sequential restart read on the same table if intents were written
// after the first read. GH #6972.
TEST_F(PgMiniTest, NoRestartSecondRead) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_max_clock_skew_usec) = 1000000000LL * kTimeMultiplier;
  auto conn1 = ASSERT_RESULT(Connect());
  auto conn2 = ASSERT_RESULT(Connect());
  ASSERT_OK(conn1.Execute("CREATE TABLE t (a int PRIMARY KEY, b int) SPLIT INTO 1 TABLETS"));
  ASSERT_OK(conn1.Execute("INSERT INTO t VALUES (1, 1), (2, 1), (3, 1)"));
  auto start_time = MonoTime::Now();
  ASSERT_OK(conn1.StartTransaction(IsolationLevel::SNAPSHOT_ISOLATION));
  LOG(INFO) << "Select1";
  auto res = ASSERT_RESULT(conn1.FetchRow<int32_t>("SELECT b FROM t WHERE a = 1"));
  ASSERT_EQ(res, 1);
  LOG(INFO) << "Update";
  ASSERT_OK(conn2.StartTransaction(IsolationLevel::SNAPSHOT_ISOLATION));
  ASSERT_OK(conn2.Execute("UPDATE t SET b = 2 WHERE a = 2"));
  ASSERT_OK(conn2.CommitTransaction());
  auto update_time = MonoTime::Now();
  ASSERT_LE(update_time, start_time + FLAGS_max_clock_skew_usec * 1us);
  LOG(INFO) << "Select2";
  res = ASSERT_RESULT(conn1.FetchRow<int32_t>("SELECT b FROM t WHERE a = 2"));
  ASSERT_EQ(res, 1);
  ASSERT_OK(conn1.CommitTransaction());
}

// ------------------------------------------------------------------------------------------------
// Tablet Splitting Tests
// ------------------------------------------------------------------------------------------------

namespace {

YB_DEFINE_ENUM(KeyColumnType, (kHash)(kAsc)(kDesc));

class PgMiniTestAutoScanNextPartitions : public PgMiniTest {
 protected:
  void SetUp() override {
    ANNOTATE_UNPROTECTED_WRITE(FLAGS_TEST_index_read_multiple_partitions) = true;
    PgMiniTest::SetUp();
  }

  Status IndexScan(PGConn* conn, KeyColumnType table_key, KeyColumnType index_key) {
    RETURN_NOT_OK(conn->Execute("DROP TABLE IF EXISTS t"));
    RETURN_NOT_OK(conn->ExecuteFormat(
        "CREATE TABLE t (k INT, v1 INT, v2 INT, PRIMARY KEY (k $0)) $1",
        ToPostgresKeyType(table_key), TableSplitOptions(table_key)));
    RETURN_NOT_OK(conn->ExecuteFormat(
        "CREATE INDEX ON t(v1 $0, v2 $0)", ToPostgresKeyType(index_key)));

    constexpr int kNumRows = 100;
    RETURN_NOT_OK(conn->ExecuteFormat(
        "INSERT INTO t SELECT s, 1, s FROM generate_series(1, $0) AS s", kNumRows));

    // Secondary index read from the table
    // While performing secondary index read on ybctids, the pggate layer batches requests belonging
    // to the same tablet. However, if the tablet is split after batching, we need a mechanism to
    // execute the batched request across both the sub-tablets. We create a scenario to test this
    // phenomenon here.
    //
    // FLAGS_index_read_multiple_partitions is a test flag when set will create a scenario to check
    // if index scans of ybctids span across multiple tablets. Specifically in this example, we try
    // to scan the elements which contain value v1 = 1 and see if they match the expected number
    // of rows.
    constexpr auto kQuery = "SELECT k FROM t WHERE v1 = 1";
    RETURN_NOT_OK(conn->HasIndexScan(kQuery));
    return ResultToStatus(conn->FetchMatrix(kQuery, kNumRows, 1));
  }

  Status FKConstraint(PGConn* conn, KeyColumnType key_type) {
    RETURN_NOT_OK(conn->Execute("DROP TABLE IF EXISTS ref_t, t1, t2"));
    RETURN_NOT_OK(conn->ExecuteFormat("CREATE TABLE t1 (k INT, PRIMARY KEY(k $0)) $1",
                                      ToPostgresKeyType(key_type),
                                      TableSplitOptions(key_type)));
    RETURN_NOT_OK(conn->ExecuteFormat("CREATE TABLE t2 (k INT, PRIMARY KEY(k $0)) $1",
                                      ToPostgresKeyType(key_type),
                                      TableSplitOptions(key_type)));
    RETURN_NOT_OK(conn->Execute("CREATE TABLE ref_t (k INT,"
                                "                    fk_1 INT REFERENCES t1(k),"
                                "                    fk_2 INT REFERENCES t2(k))"));
    constexpr int kNumRows = 100;
    RETURN_NOT_OK(conn->ExecuteFormat(
        "INSERT INTO t1 SELECT s FROM generate_series(1, $0) AS s", kNumRows));
    RETURN_NOT_OK(conn->ExecuteFormat(
        "INSERT INTO t2 SELECT s FROM generate_series(1, $0) AS s", kNumRows));
    return conn->ExecuteFormat(
        "INSERT INTO ref_t SELECT s, s, s FROM generate_series(1, $0) AS s", kNumRows);
  }

 private:
  static std::string TableSplitOptions(KeyColumnType key_type) {
    switch(key_type) {
      case KeyColumnType::kHash:
        return "SPLIT INTO 10 TABLETS";
      case KeyColumnType::kAsc:
        return "SPLIT AT VALUES ((12), (25), (37), (50), (62), (75), (87))";
      case KeyColumnType::kDesc:
        return "SPLIT AT VALUES ((87), (75), (62), (50), (37), (25), (12))";
    }
    FATAL_INVALID_ENUM_VALUE(KeyColumnType, key_type);
  }

  static std::string ToPostgresKeyType(KeyColumnType key_type) {
    switch(key_type) {
      case KeyColumnType::kHash: return "";
      case KeyColumnType::kAsc: return "ASC";
      case KeyColumnType::kDesc: return "DESC";
    }
    FATAL_INVALID_ENUM_VALUE(KeyColumnType, key_type);
  }

};

} // namespace

// The test checks all rows are returned in case of index scan with dynamic table splitting for
// different table and index key column type combinations (hash, asc, desc)
TEST_F_EX(
    PgMiniTest, YB_DISABLE_TEST_IN_SANITIZERS(AutoScanNextPartitionsIndexScan),
    PgMiniTestAutoScanNextPartitions) {
  auto conn = ASSERT_RESULT(Connect());
  for (auto table_key : kKeyColumnTypeArray) {
    for (auto index_key : kKeyColumnTypeArray) {
      ASSERT_OK_PREPEND(IndexScan(&conn, table_key, index_key),
                        Format("Bad status in test with table_key=$0, index_key=$1",
                               ToString(table_key),
                               ToString(index_key)));
    }
  }
}

// The test checks foreign key constraint is not violated in case of referenced table dynamic
// splitting for different key column types (hash, asc, desc).
TEST_F_EX(
    PgMiniTest, YB_DISABLE_TEST_IN_SANITIZERS(AutoScanNextPartitionsFKConstraint),
    PgMiniTestAutoScanNextPartitions) {
  auto conn = ASSERT_RESULT(Connect());
  for (auto table_key : kKeyColumnTypeArray) {
    ASSERT_OK_PREPEND(FKConstraint(&conn, table_key),
                      Format("Bad status in test with table_key=$0", ToString(table_key)));
  }
}

class PgMiniTabletSplitTest : public PgMiniTest {
 public:
  void SetUp() override {
    ANNOTATE_UNPROTECTED_WRITE(FLAGS_yb_num_shards_per_tserver) = 1;
    ANNOTATE_UNPROTECTED_WRITE(FLAGS_tablet_split_low_phase_size_threshold_bytes) = 0;
    ANNOTATE_UNPROTECTED_WRITE(FLAGS_tablet_split_high_phase_size_threshold_bytes) = 0;
    ANNOTATE_UNPROTECTED_WRITE(FLAGS_tablet_split_low_phase_shard_count_per_node) = 0;
    ANNOTATE_UNPROTECTED_WRITE(FLAGS_tablet_split_high_phase_shard_count_per_node) = 0;
    ANNOTATE_UNPROTECTED_WRITE(FLAGS_tablet_force_split_threshold_bytes) = 30_KB;
    ANNOTATE_UNPROTECTED_WRITE(FLAGS_db_write_buffer_size) =
        FLAGS_tablet_force_split_threshold_bytes / 4;
    ANNOTATE_UNPROTECTED_WRITE(FLAGS_db_block_size_bytes) = 2_KB;
    ANNOTATE_UNPROTECTED_WRITE(FLAGS_db_filter_block_size_bytes) = 2_KB;
    ANNOTATE_UNPROTECTED_WRITE(FLAGS_db_index_block_size_bytes) = 2_KB;
    ANNOTATE_UNPROTECTED_WRITE(FLAGS_heartbeat_interval_ms) = 1000;
    ANNOTATE_UNPROTECTED_WRITE(FLAGS_tserver_heartbeat_metrics_interval_ms) = 1000;
    ANNOTATE_UNPROTECTED_WRITE(
        FLAGS_TEST_inject_delay_between_prepare_ybctid_execute_batch_ybctid_ms) = 4000;
    PgMiniTest::SetUp();
  }

  Status SetupConnection(PGConn* conn) const override {
    return conn->Execute("SET yb_fetch_row_limit = 32");
  }
};

void PgMiniTest::CreateTableAndInitialize(std::string table_name, int num_tablets) {
  auto conn = ASSERT_RESULT(Connect());

  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_automatic_tablet_splitting) = false;
  ASSERT_OK(conn.ExecuteFormat("CREATE TABLE $0 (h1 int, h2 int, r int, i int, "
                               "PRIMARY KEY ((h1, h2) HASH, r ASC)) "
                               "SPLIT INTO $1 TABLETS", table_name, num_tablets));

  ASSERT_OK(conn.ExecuteFormat("CREATE INDEX $0_idx "
                               "ON $1(i HASH, r ASC)", table_name, table_name));

  ASSERT_OK(conn.ExecuteFormat("INSERT INTO $0 SELECT i, i, i, 1 FROM "
                               "(SELECT generate_series(1, 500) i) t", table_name));
}

void PgMiniTest::DestroyTable(std::string table_name) {
  auto conn = ASSERT_RESULT(Connect());
  ASSERT_OK(conn.ExecuteFormat("DROP TABLE $0", table_name));
}

void PgMiniTest::StartReadWriteThreads(const std::string table_name,
    TestThreadHolder *thread_holder) {
  // Writer thread that does parallel writes into table
  thread_holder->AddThread([this, table_name] {
    auto conn = ASSERT_RESULT(Connect());
    for (int i = 501; i < 2000; i++) {
      ASSERT_OK(conn.ExecuteFormat("INSERT INTO $0 VALUES ($1, $2, $3, $4)",
                                   table_name, i, i, i, 1));
    }
  });

  // Index read from the table
  thread_holder->AddThread([this, &stop = thread_holder->stop_flag(), table_name] {
    auto conn = ASSERT_RESULT(Connect());
    do {
      auto result = ASSERT_RESULT(conn.FetchFormat("SELECT * FROM  $0 WHERE i = 1 order by r",
                                                   table_name));
      std::vector<int> sort_check;
      for(int x = 0; x < PQntuples(result.get()); x++) {
        auto value = ASSERT_RESULT(GetValue<int32_t>(result.get(), x, 2));
        sort_check.push_back(value);
      }
      ASSERT_TRUE(std::is_sorted(sort_check.begin(), sort_check.end()));
    }  while (!stop.load(std::memory_order_acquire));
  });
}

TEST_F_EX(
    PgMiniTest, YB_DISABLE_TEST_IN_SANITIZERS(TabletSplitSecondaryIndexYSQL),
    PgMiniTabletSplitTest) {

  std::string table_name = "update_pk_complex_two_hash_one_range_keys";
  CreateTableAndInitialize(table_name, 1);

  auto table_id = ASSERT_RESULT(GetTableIDFromTableName(table_name));
  auto start_num_tablets = ListTableActiveTabletLeadersPeers(cluster_.get(), table_id).size();
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_automatic_tablet_splitting) = true;

  // Insert elements into the table using a parallel thread
  TestThreadHolder thread_holder;

  /*
   * Writer thread writes into the table continously, while the index read thread does a secondary
   * index lookup. During the index lookup, we inject artificial delays, specified by the flag
   * FLAGS_TEST_tablet_split_injected_delay_ms. Tablets will split in between those delays into
   * two different partitions.
   *
   * The purpose of this test is to verify that when the secondary index read request is being
   * executed, the results from both the tablets are being represented. Without the fix from
   * the pggate layer, only one half of the results will be obtained. Hence we verify that after the
   * split the number of elements is > 500, which is the number of elements inserted before the
   * split.
   */
  StartReadWriteThreads(table_name, &thread_holder);

  thread_holder.WaitAndStop(200s);
  auto end_num_tablets = ListTableActiveTabletLeadersPeers(cluster_.get(), table_id).size();
  ASSERT_GT(end_num_tablets, start_num_tablets);
  DestroyTable(table_name);

  // Rerun the same test where table is created with 3 tablets.
  // When a table is created with three tablets, the lower and upper bounds are as follows;
  // tablet 1 -- empty to A
  // tablet 2 -- A to B
  // tablet 3 -- B to empty
  // However, in situations where tables are created with just one tablet lower_bound and
  // upper_bound for the tablet is empty to empty. Hence, to test both situations we run this test
  // with one tablet and three tablets respectively.
  CreateTableAndInitialize(table_name, 3);
  table_id = ASSERT_RESULT(GetTableIDFromTableName(table_name));
  start_num_tablets = ListTableActiveTabletLeadersPeers(cluster_.get(), table_id).size();
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_automatic_tablet_splitting) = true;

  StartReadWriteThreads(table_name, &thread_holder);
  thread_holder.WaitAndStop(200s);

  end_num_tablets = ListTableActiveTabletLeadersPeers(cluster_.get(), table_id).size();
  ASSERT_GT(end_num_tablets, start_num_tablets);
  DestroyTable(table_name);
}

void PgMiniTest::ValidateAbortedTxnMetric() {
  auto tablet_peers = cluster_->GetTabletPeers(0);
  for(size_t i = 0; i < tablet_peers.size(); ++i) {
    auto tablet = ASSERT_RESULT(tablet_peers[i]->shared_tablet_safe());
    const auto& metric_map = tablet->GetTabletMetricsEntity()->UnsafeMetricsMapForTests();
    std::reference_wrapper<const MetricPrototype> metric =
        METRIC_aborted_transactions_pending_cleanup;
    auto item = metric_map.find(&metric.get());
    if (item != metric_map.end()) {
      EXPECT_EQ(0, down_cast<const AtomicGauge<uint64>&>(*item->second).value());
    }
  }
}

void PgMiniTest::RunManyConcurrentReadersTest() {
  constexpr int kNumConcurrentRead = 8;
  constexpr int kMinNumNonEmptyReads = 10;
  const std::string kTableName = "savepoints";
  TestThreadHolder thread_holder;

  std::atomic<int32_t> next_write_start{0};
  std::atomic<int32_t> num_non_empty_reads{0};
  CountDownLatch reader_latch(0);
  CountDownLatch writer_latch(1);
  std::atomic<bool> writer_thread_is_stopped{false};
  CountDownLatch reader_threads_are_stopped(kNumConcurrentRead);

  {
    auto conn = ASSERT_RESULT(Connect());
    ASSERT_OK(conn.ExecuteFormat("CREATE TABLE $0 (a int)", kTableName));
  }

  thread_holder.AddThreadFunctor([
      &stop = thread_holder.stop_flag(), &next_write_start, &reader_latch, &writer_latch,
      &writer_thread_is_stopped, kTableName, this] {
    auto conn = ASSERT_RESULT(Connect());
    while (!stop.load(std::memory_order_acquire)) {
      auto write_start = (next_write_start += 5);
      ASSERT_OK(conn.StartTransaction(IsolationLevel::SNAPSHOT_ISOLATION));
      ASSERT_OK(conn.ExecuteFormat("INSERT INTO $0 VALUES ($1)", kTableName, write_start));
      ASSERT_OK(conn.Execute("SAVEPOINT one"));
      ASSERT_OK(conn.ExecuteFormat("INSERT INTO $0 VALUES ($1)", kTableName, write_start + 1));
      ASSERT_OK(conn.Execute("SAVEPOINT two"));
      ASSERT_OK(conn.ExecuteFormat("INSERT INTO $0 VALUES ($1)", kTableName, write_start + 2));
      ASSERT_OK(conn.Execute("ROLLBACK TO SAVEPOINT one"));
      ASSERT_OK(conn.ExecuteFormat("INSERT INTO $0 VALUES ($1)", kTableName, write_start + 3));
      ASSERT_OK(conn.Execute("ROLLBACK TO SAVEPOINT one"));
      ASSERT_OK(conn.ExecuteFormat("INSERT INTO $0 VALUES ($1)", kTableName, write_start + 4));

      // Start concurrent reader threads
      reader_latch.Reset(kNumConcurrentRead * 5);
      writer_latch.CountDown();

      // Commit while reader threads are running
      ASSERT_OK(conn.CommitTransaction());

      // Allow reader threads to complete and halt.
      ASSERT_TRUE(reader_latch.WaitFor(5s * kTimeMultiplier));
      writer_latch.Reset(1);
    }
    writer_thread_is_stopped = true;
  });

  for (int reader_idx = 0; reader_idx < kNumConcurrentRead; ++reader_idx) {
    thread_holder.AddThreadFunctor([
        &stop = thread_holder.stop_flag(), &next_write_start, &num_non_empty_reads,
        &reader_latch, &writer_latch, &reader_threads_are_stopped, kTableName, this] {
      auto conn = ASSERT_RESULT(Connect());
      while (!stop.load(std::memory_order_acquire)) {
        ASSERT_TRUE(writer_latch.WaitFor(10s * kTimeMultiplier));

        auto read_start = next_write_start.load();
        auto read_end = read_start + 4;
        auto fetch_query = strings::Substitute(
            "SELECT * FROM $0 WHERE a BETWEEN $1 AND $2 ORDER BY a ASC",
            kTableName, read_start, read_end);

        const auto values = ASSERT_RESULT(conn.FetchRows<int32_t>(fetch_query));
        const auto fetched_values = values.size();
        if (fetched_values != 0) {
          num_non_empty_reads++;
          if (fetched_values != 2) {
            LOG(INFO)
                << "Expected to fetch (" << read_start << ") and (" << read_end << "). "
                << "Instead, got the following results:";
            for (size_t i = 0; i < fetched_values; ++i) {
              LOG(INFO) << "Result " << i << " - " << values[i];
            }
          }
          EXPECT_EQ(values, (decltype(values){read_start, read_start + 4}));
        }
        reader_latch.CountDown(1);
      }
      reader_threads_are_stopped.CountDown(1);
    });
    ValidateAbortedTxnMetric();
  }

  std::this_thread::sleep_for(60s);
  thread_holder.stop_flag().store(true, std::memory_order_release);
  while (!writer_thread_is_stopped.load(std::memory_order_acquire) ||
          reader_threads_are_stopped.count() != 0) {
    reader_latch.Reset(0);
    writer_latch.Reset(0);
    std::this_thread::sleep_for(10ms * kTimeMultiplier);
  }
  thread_holder.Stop();
  EXPECT_GE(num_non_empty_reads, kMinNumNonEmptyReads);
}

TEST_F(PgMiniTest, BigInsertWithAbortedIntentsAndRestart) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_apply_intents_task_injected_delay_ms) = 200;

  constexpr int64_t kRowNumModToAbort = 7;
  constexpr int64_t kNumBatches = 10;
  constexpr int64_t kNumRows = RegularBuildVsSanitizers(10000, 1000);
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_txn_max_apply_batch_records) = kNumRows / kNumBatches;

  auto conn = ASSERT_RESULT(Connect());
  ASSERT_OK(conn.Execute("CREATE TABLE t (a int PRIMARY KEY) SPLIT INTO 1 TABLETS"));

  ASSERT_OK(conn.StartTransaction(IsolationLevel::SERIALIZABLE_ISOLATION));
  for (int32_t row_num = 0; row_num < kNumRows; ++row_num) {
    auto should_abort = row_num % kRowNumModToAbort == 0;
    if (should_abort) {
      ASSERT_OK(conn.Execute("SAVEPOINT A"));
    }
    ASSERT_OK(conn.ExecuteFormat("INSERT INTO t VALUES ($0)", row_num));
    if (should_abort) {
      ASSERT_OK(conn.Execute("ROLLBACK TO A"));
    }
  }

  ASSERT_OK(conn.CommitTransaction());

  LOG(INFO) << "Restart cluster";
  ASSERT_OK(RestartCluster());
  conn = ASSERT_RESULT(Connect());

  ASSERT_OK(WaitFor([this] {
    auto intents_count = CountIntents(cluster_.get());
    LOG(INFO) << "Intents count: " << intents_count;

    return intents_count == 0;
  }, 60s * kTimeMultiplier, "Intents cleanup", 200ms));

  for (int32_t row_num = 0; row_num < kNumRows; ++row_num) {
    auto should_abort = row_num % kRowNumModToAbort == 0;

    const auto values = ASSERT_RESULT(conn.FetchRows<int32_t>(Format(
        "SELECT * FROM t WHERE a = $0", row_num)));
    if (should_abort) {
      EXPECT_TRUE(values.empty()) << "Did not expect to find value for: " << row_num;
    } else {
      EXPECT_EQ(values.size(), 1);
      EXPECT_EQ(values[0], row_num);
    }
  }
  ValidateAbortedTxnMetric();
}

TEST_F(
    PgMiniTest,
    YB_DISABLE_TEST_IN_SANITIZERS(TestConcurrentReadersMaskAbortedIntentsWithApplyDelay)) {
  ASSERT_OK(cluster_->WaitForAllTabletServers());
  std::this_thread::sleep_for(10s);
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_apply_intents_task_injected_delay_ms) = 10000;
  RunManyConcurrentReadersTest();
}

TEST_F(
    PgMiniTest,
    YB_DISABLE_TEST_IN_SANITIZERS(TestConcurrentReadersMaskAbortedIntentsWithResponseDelay)) {
  ASSERT_OK(cluster_->WaitForAllTabletServers());
  std::this_thread::sleep_for(10s);
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_TEST_inject_random_delay_on_txn_status_response_ms) = 30;
  RunManyConcurrentReadersTest();
}

TEST_F(
    PgMiniTest,
    YB_DISABLE_TEST_IN_SANITIZERS(TestConcurrentReadersMaskAbortedIntentsWithUpdateDelay)) {
  ASSERT_OK(cluster_->WaitForAllTabletServers());
  std::this_thread::sleep_for(10s);
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_TEST_txn_participant_inject_latency_on_apply_update_txn_ms) = 30;
  RunManyConcurrentReadersTest();
}

// TODO(savepoint): This test would start failing until issue #9587 is fixed. It worked earlier but
// is expected to fail, as pointed out in https://phabricator.dev.yugabyte.com/D17177
// Change macro to YB_DISABLE_TEST_IN_TSAN if re-enabling.
TEST_F(PgMiniTest, YB_DISABLE_TEST(TestSerializableStrongReadLockNotAborted)) {
  auto conn = ASSERT_RESULT(Connect());
  ASSERT_OK(conn.Execute("CREATE TABLE t (a int PRIMARY KEY, b int) SPLIT INTO 1 TABLETS"));
  for (int i = 0; i < 100; ++i) {
    ASSERT_OK(conn.ExecuteFormat("INSERT INTO t VALUES ($0, $0)", i));
  }

  auto conn1 = ASSERT_RESULT(Connect());
  ASSERT_OK(conn1.StartTransaction(IsolationLevel::SERIALIZABLE_ISOLATION));
  ASSERT_OK(conn1.Execute("SAVEPOINT A"));
  auto res1 = ASSERT_RESULT(conn1.FetchFormat("SELECT b FROM t WHERE a = $0", 90));
  ASSERT_OK(conn1.Execute("ROLLBACK TO A"));

  auto conn2 = ASSERT_RESULT(Connect());
  ASSERT_OK(conn2.StartTransaction(IsolationLevel::SERIALIZABLE_ISOLATION));
  auto update_status = conn2.ExecuteFormat("UPDATE t SET b = $0 WHERE a = $1", 1000, 90);

  auto commit_status = conn1.CommitTransaction();

  EXPECT_TRUE(commit_status.ok() ^ update_status.ok())
      << "Expected exactly one of commit of first transaction or update of second transaction to "
      << "fail.\n"
      << "Commit status: " << commit_status << ".\n"
      << "Update status: " << update_status << ".\n";
  ValidateAbortedTxnMetric();
}

void PgMiniTest::VerifyFileSizeAfterCompaction(PGConn* conn, const int num_tables) {
  ASSERT_OK(cluster_->FlushTablets());
  uint64_t files_size = 0;
  for (const auto& peer : ListTabletPeers(cluster_.get(), ListPeersFilter::kAll)) {
    files_size += peer->tablet()->GetCurrentVersionSstFilesUncompressedSize();
  }

  ASSERT_OK(conn->ExecuteFormat("ALTER TABLE test$0 DROP COLUMN string;", num_tables - 1));
  ASSERT_OK(conn->ExecuteFormat("ALTER TABLE test$0 DROP COLUMN string;", 0));

  ASSERT_OK(cluster_->CompactTablets());

  uint64_t new_files_size = 0;
  for (const auto& peer : ListTabletPeers(cluster_.get(), ListPeersFilter::kAll)) {
    new_files_size += peer->tablet()->GetCurrentVersionSstFilesUncompressedSize();
  }

  LOG(INFO) << "Old files size: " << files_size << ", new files size: " << new_files_size;
  ASSERT_LE(new_files_size * 2, files_size);
  ASSERT_GE(new_files_size * 3, files_size);
}

TEST_F(PgMiniTest, ColocatedCompaction) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_timestamp_history_retention_interval_sec) = 0;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_history_cutoff_propagation_interval_ms) = 1;

  const std::string kDatabaseName = "testdb";
  const auto kNumTables = 3;
  constexpr int kKeys = 100;

  PGConn conn = ASSERT_RESULT(Connect());
  ASSERT_OK(conn.ExecuteFormat("CREATE DATABASE $0 with colocated=true", kDatabaseName));

  conn = ASSERT_RESULT(ConnectToDB(kDatabaseName));
  for (int i = 0; i < kNumTables; ++i) {
    ASSERT_OK(conn.ExecuteFormat(R"#(
        CREATE TABLE test$0 (
          key INTEGER NOT NULL PRIMARY KEY,
          value INTEGER,
          string VARCHAR
        )
      )#", i));
    for (int j = 0; j < kKeys; ++j) {
      ASSERT_OK(conn.ExecuteFormat(
          "INSERT INTO test$0(key, value, string) VALUES($1, -$1, '$2')", i, j,
          RandomHumanReadableString(128_KB)));
    }
  }
  VerifyFileSizeAfterCompaction(&conn, kNumTables);
}

void PgMiniTest::CreateDBWithTablegroupAndTables(
    const std::string database_name, const std::string tablegroup_name, const int num_tables,
    const int keys, PGConn* conn) {
  ASSERT_OK(conn->ExecuteFormat("CREATE DATABASE $0", database_name));
  *conn = ASSERT_RESULT(ConnectToDB(database_name));
  ASSERT_OK(conn->ExecuteFormat("CREATE TABLEGROUP $0", tablegroup_name));
  for (int i = 0; i < num_tables; ++i) {
    ASSERT_OK(conn->ExecuteFormat(R"#(
        CREATE TABLE test$0 (
          key INTEGER NOT NULL PRIMARY KEY,
          value INTEGER,
          string VARCHAR
        ) tablegroup $1
      )#", i, tablegroup_name));
    for (int j = 0; j < keys; ++j) {
      ASSERT_OK(conn->ExecuteFormat(
          "INSERT INTO test$0(key, value, string) VALUES($1, -$1, '$2')", i, j,
          RandomHumanReadableString(128_KB)));
    }
  }
}

TEST_F(PgMiniTest, TablegroupCompaction) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_timestamp_history_retention_interval_sec) = 0;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_history_cutoff_propagation_interval_ms) = 1;

  PGConn conn = ASSERT_RESULT(Connect());
  CreateDBWithTablegroupAndTables(
      "testdb" /* database_name */,
      "testtg" /* tablegroup_name */,
      3 /* num_tables */,
      100 /* keys */,
      &conn);
  VerifyFileSizeAfterCompaction(&conn, 3 /* num_tables */);
}

// Ensure that after restart, there is no data loss in compaction.
TEST_F(PgMiniTest, TablegroupCompactionWithRestart) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_timestamp_history_retention_interval_sec) = 0;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_history_cutoff_propagation_interval_ms) = 1;
  const auto num_tables = 3;
  constexpr int keys = 100;

  PGConn conn = ASSERT_RESULT(Connect());
  CreateDBWithTablegroupAndTables(
      "testdb" /* database_name */,
      "testtg" /* tablegroup_name */,
      num_tables,
      keys,
      &conn);
  ASSERT_OK(cluster_->FlushTablets());
  ASSERT_OK(cluster_->RestartSync());
  ASSERT_OK(cluster_->CompactTablets());
  conn = ASSERT_RESULT(ConnectToDB("testdb" /* database_name */));
  for (int i = 0; i < num_tables; ++i) {
    auto res =
        ASSERT_RESULT(conn.template FetchRow<PGUint64>(Format("SELECT COUNT(*) FROM test$0", i)));
    ASSERT_EQ(res, keys);
  }
}

TEST_F(PgMiniTest, CompactionAfterDBDrop) {
  const std::string kDatabaseName = "testdb";
  auto& catalog_manager = ASSERT_RESULT(cluster_->GetLeaderMiniMaster())->catalog_manager();
  auto sys_catalog_tablet = catalog_manager.sys_catalog()->tablet_peer()->tablet();

  ASSERT_OK(sys_catalog_tablet->Flush(tablet::FlushMode::kSync));
  ASSERT_OK(sys_catalog_tablet->ForceManualRocksDBCompact());
  uint64_t base_file_size = sys_catalog_tablet->GetCurrentVersionSstFilesUncompressedSize();;

  PGConn conn = ASSERT_RESULT(Connect());
  ASSERT_OK(conn.ExecuteFormat("CREATE DATABASE $0", kDatabaseName));
  ASSERT_OK(conn.ExecuteFormat("DROP DATABASE $0", kDatabaseName));
  ASSERT_OK(sys_catalog_tablet->Flush(tablet::FlushMode::kSync));

  // Make sure compaction works without error for the hybrid_time > history_cutoff case.
  ASSERT_OK(sys_catalog_tablet->ForceManualRocksDBCompact());

  ANNOTATE_UNPROTECTED_WRITE(FLAGS_timestamp_history_retention_interval_sec) = 0;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_timestamp_syscatalog_history_retention_interval_sec) = 0;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_history_cutoff_propagation_interval_ms) = 1;

  ASSERT_OK(sys_catalog_tablet->ForceManualRocksDBCompact());

  uint64_t new_file_size = sys_catalog_tablet->GetCurrentVersionSstFilesUncompressedSize();;
  LOG(INFO) << "Base file size: " << base_file_size << ", new file size: " << new_file_size;
  ASSERT_LE(new_file_size, base_file_size + 100_KB);
}

// The test checks that YSQL doesn't wait for sent RPC response in case of process termination.
TEST_F(PgMiniTest, NoWaitForRPCOnTermination) {
  auto conn = ASSERT_RESULT(Connect());
  ASSERT_OK(conn.Execute("CREATE TABLE t(k INT PRIMARY KEY) SPLIT INTO 1 TABLETS"));
  constexpr auto kRows = RegularBuildVsDebugVsSanitizers(1000000, 100000, 30000);
  ASSERT_OK(conn.ExecuteFormat(
      "INSERT INTO t SELECT s FROM generate_series(1, $0) AS s", kRows));
  constexpr auto kLongTimeQuery = "SELECT COUNT(*) FROM t";
  std::atomic<MonoTime> termination_start;
  MonoTime termination_end;
  {
    CountDownLatch latch(2);
    TestThreadHolder thread_holder;
    thread_holder.AddThreadFunctor([this, &latch, &termination_start, kLongTimeQuery] {
      auto thread_conn = ASSERT_RESULT(Connect());
      latch.CountDown();
      latch.Wait();
      const auto deadline = MonoTime::Now() + MonoDelta::FromSeconds(30);
      while (MonoTime::Now() < deadline) {
        const auto local_termination_start = MonoTime::Now();
        auto res = ASSERT_RESULT(thread_conn.FetchFormat(
          "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE query like '$0'",
          kLongTimeQuery));
        auto lines = PQntuples(res.get());
        if (lines) {
          termination_start.store(local_termination_start, std::memory_order_release);
          break;
        }
      }
    });
    latch.CountDown();
    latch.Wait();
    const auto res = conn.Fetch(kLongTimeQuery);
    ASSERT_NOK(res);
    ASSERT_STR_CONTAINS(res.status().ToString(),
                        "terminating connection due to administrator command");
    termination_end = MonoTime::Now();
  }
  const auto termination_duration =
      (termination_end - termination_start.load(std::memory_order_acquire)).ToMilliseconds();
  ASSERT_GT(termination_duration, 0);
  ASSERT_LT(termination_duration, RegularBuildVsDebugVsSanitizers(3000, 5000, 5000));
}

TEST_F_EX(
    PgMiniTest, CacheRefreshWithDroppedEntries, PgMiniTestSingleNode) {
  auto conn = ASSERT_RESULT(Connect());
  ASSERT_OK(conn.Execute("CREATE TABLE t (k INT PRIMARY KEY)"));
  constexpr size_t kNumViews = 30;
  for (size_t i = 0; i < kNumViews; ++i) {
    ASSERT_OK(conn.ExecuteFormat("CREATE VIEW v_$0 AS SELECT * FROM t", i));
  }
  // Trigger catalog version increment
  ASSERT_OK(conn.Execute("ALTER TABLE t ADD COLUMN v INT"));
  // New connection will load all the entries (tables and views) into catalog cache
  auto aux_conn = ASSERT_RESULT(Connect());
  for (size_t i = 0; i < kNumViews; ++i) {
    ASSERT_OK(conn.ExecuteFormat("DROP VIEW v_$0", i));
  }
  // Wait for update of catalog version in shared memory to trigger catalog refresh on next query
  SleepFor(MonoDelta::FromMilliseconds(2 * FLAGS_heartbeat_interval_ms));
  // Check that connection can handle query (i.e. the catalog cache was updated without an issue)
  ASSERT_OK(aux_conn.Fetch("SELECT 1"));
}

int64_t PgMiniTest::GetBloomFilterCheckedMetric() {
  auto peers = ListTabletPeers(cluster_.get(), ListPeersFilter::kAll);
  auto bloom_filter_checked = 0;
  for (auto &peer : peers) {
    const auto tablet = peer->shared_tablet();
    if (tablet) {
      bloom_filter_checked += tablet->regulardb_statistics()
        ->getTickerCount(rocksdb::BLOOM_FILTER_CHECKED);
    }
  }
  return bloom_filter_checked;
}

TEST_F(PgMiniTest, BloomFilterBackwardScanTest) {
  auto conn = ASSERT_RESULT(Connect());
  ASSERT_OK(conn.Execute("CREATE TABLE t (h int, r int, primary key(h, r))"));
  ASSERT_OK(conn.Execute("INSERT INTO t SELECT i / 10, i % 10"
                         "FROM generate_series(1, 500) i"));

  FlushAndCompactTablets();

  auto before_blooms_checked = GetBloomFilterCheckedMetric();

  ASSERT_OK(
      conn.Fetch("SELECT * FROM t WHERE h = 2 AND r > 2 ORDER BY r DESC;"));

  auto after_blooms_checked = GetBloomFilterCheckedMetric();
  ASSERT_EQ(after_blooms_checked, before_blooms_checked + 1);
}

} // namespace yb::pgwrapper
