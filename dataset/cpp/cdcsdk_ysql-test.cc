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

#include "yb/cdc/cdc_service.pb.h"
#include "yb/cdc/cdc_types.h"
#include "yb/common/entity_ids_types.h"
#include "yb/integration-tests/cdcsdk_test_base.h"
#include "yb/integration-tests/cdcsdk_ysql_test_base.h"

#include "yb/cdc/cdc_state_table.h"
#include "yb/master/tasks_tracker.h"
#include "yb/util/tostring.h"

namespace yb {

using client::YBTableName;

using pgwrapper::PGConn;
using pgwrapper::PGResultPtr;

using rpc::RpcController;

namespace cdc {

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestBaseFunctions)) {
  // setting up a cluster with 3 RF
  ASSERT_OK(SetUpWithParams(3, 1, false /* colocated */));

  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName));

  ASSERT_FALSE(table.is_cql_namespace());
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestLoadInsertionOnly)) {
  // set up an RF3 cluster
  ASSERT_OK(SetUpWithParams(3, 1, false));

  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName));

  ASSERT_OK(WriteRows(0, 10, &test_cluster_));
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(GetChangesWithRF1)) {
  TestGetChanges(1 /* replication factor */);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(GetChangesWithRF3)) {
  TestGetChanges(3 /* replication factor */);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(GetChanges_TablesWithNoPKPresentInDB)) {
  TestGetChanges(3 /* replication_factor */, true /* add_tables_without_primary_key */);
}

// Insert a single row.
// Expected records: (DDL, INSERT).
TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(SingleShardInsertWithAutoCommit)) {
  auto tablets = ASSERT_RESULT(SetUpCluster());
  ASSERT_EQ(tablets.size(), 1);
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStreamWithReplicationSlot());
  auto set_resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(set_resp.has_error());

  ASSERT_OK(WriteRows(1 /* start */, 2 /* end */, &test_cluster_));

  // The count array stores counts of DDL, INSERT, UPDATE, DELETE, READ, TRUNCATE in that order.
  const uint32_t expected_count[] = {1, 1, 0, 0, 0, 0};
  uint32_t count[] = {0, 0, 0, 0, 0, 0};

  ExpectedRecord expected_records[] = {{0, 0}, {1, 2}};

  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, 1));

  uint32_t seen_dml_records = 0;
  for (const auto& record : change_resp.cdc_sdk_proto_records()) {
    if (record.row_message().op() == RowMessage::BEGIN ||
        record.row_message().op() == RowMessage::COMMIT) {
      continue;
    }
    CheckRecord(record, expected_records[seen_dml_records], count);
    seen_dml_records++;
  }
  LOG(INFO) << "Got " << count[1] << " insert record";
  CheckCount(expected_count, count);
}

// Insert, update, delete rows.
// Expected records: (DDL, INSERT, UPDATE, INSERT, DELETE) in this order.
TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(SingleShardDMLWithAutoCommit)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_ysql_enable_packed_row) = false;

  auto tablets = ASSERT_RESULT(SetUpCluster());
  ASSERT_EQ(tablets.size(), 1);
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStreamWithReplicationSlot());
  auto set_resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(set_resp.has_error());

  ASSERT_OK(WriteRows(1 /* start */, 2 /* end */, &test_cluster_));
  ASSERT_OK(UpdateRows(1, 3, &test_cluster_));
  ASSERT_OK(WriteRows(2 /* start */, 3 /* end */, &test_cluster_));
  ASSERT_OK(DeleteRows(1, &test_cluster_));

  // The count array stores counts of DDL, INSERT, UPDATE, DELETE, READ, TRUNCATE in that order.
  const uint32_t expected_count[] = {1, 2, 1, 1, 0, 0};
  uint32_t count[] = {0, 0, 0, 0, 0, 0};

  ExpectedRecord expected_records[] = {{0, 0}, {1, 2}, {1, 3}, {2, 3}, {1, 3}};
  RowMessage::Op expected_record_types[] = {
      RowMessage::DDL, RowMessage::INSERT, RowMessage::UPDATE, RowMessage::INSERT,
      RowMessage::DELETE};

  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, 4));

  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  ASSERT_EQ(record_size, 13);

  uint32_t expected_record_count = 0;
  for (const auto& record : change_resp.cdc_sdk_proto_records()) {
    if (record.row_message().op() == RowMessage::BEGIN ||
        record.row_message().op() == RowMessage::COMMIT) {
      continue;
    }
    ASSERT_EQ(record.row_message().op(), expected_record_types[expected_record_count]);
    CheckRecord(record, expected_records[expected_record_count], count);
    expected_record_count++;
  }

  LOG(INFO) << "Got " << record_size << " records";
  CheckCount(expected_count, count);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCDCLagMetric)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_metrics_interval_ms) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ASSERT_OK(SetUpWithParams(1, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));

  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  const auto& tserver = test_cluster()->mini_tablet_server(0)->server();
  auto cdc_service = dynamic_cast<CDCServiceImpl*>(
      tserver->rpc_server()->TEST_service_pool("yb.cdc.CDCService")->TEST_get_service().get());

  ASSERT_OK(WaitFor(
      [&]() { return cdc_service->CDCEnabled(); }, MonoDelta::FromSeconds(30), "IsCDCEnabled"));
  ASSERT_OK(WaitFor(
      [&]() -> Result<bool> {
        auto metrics =
            std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
                {{}, stream_id, tablets[0].tablet_id()}, nullptr, CDCSDK));
        return metrics->cdcsdk_sent_lag_micros->value() == 0;
      },
      MonoDelta::FromSeconds(10) * kTimeMultiplier, "Wait for Lag == 0"));
  // Insert test rows, one at a time so they have different hybrid times.
  ASSERT_OK(WriteRowsHelper(0, 1, &test_cluster_, true));
  ASSERT_OK(WriteRowsHelper(1, 2, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  // Call get changes.
  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, 2));

  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  ASSERT_GT(record_size, 2);
  ASSERT_OK(WaitFor(
      [&]() -> Result<bool> {
        auto metrics =
            std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
                {{}, stream_id, tablets[0].tablet_id()}, nullptr, CDCSDK));
        return metrics->cdcsdk_sent_lag_micros->value() > 0;
      },
      MonoDelta::FromSeconds(10) * kTimeMultiplier, "Wait for Lag > 0"));

  GetChangesResponsePB change_resp_1 =
      ASSERT_RESULT(GetChangesFromCDC(stream_id, tablets, &change_resp.cdc_sdk_checkpoint()));
  ASSERT_OK(WaitFor(
      [&]() -> Result<bool> {
        auto metrics =
            std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
                {{}, stream_id, tablets[0].tablet_id()}, nullptr, CDCSDK));
        return metrics->cdcsdk_sent_lag_micros->value() == 0;
      },
      MonoDelta::FromSeconds(10) * kTimeMultiplier, "Wait for Lag == 0"));

  // Sleep to induce cdc lag.
  SleepFor(MonoDelta::FromSeconds(5));

  ASSERT_OK(WriteRowsHelper(3, 4, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  ASSERT_OK(WaitFor(
      [&]() -> Result<bool> {
        auto metrics =
            std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
                {{}, stream_id, tablets[0].tablet_id()}, nullptr, CDCSDK));
        return metrics->cdcsdk_sent_lag_micros->value() >= 5000000;
      },
      MonoDelta::FromSeconds(30) * kTimeMultiplier, "Wait for Lag to be around 5 seconds"));
}

// Begin transaction, perform some operations and abort transaction.
// Expected records: 1 (DDL).
TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(AbortAllWriteOperations)) {
  auto tablets = ASSERT_RESULT(SetUpCluster());
  ASSERT_EQ(tablets.size(), 1);
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStreamWithReplicationSlot());
  auto set_resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(set_resp.has_error());
  ASSERT_OK(WriteRowsHelper(1 /* start */, 4 /* end */, &test_cluster_, false));

  // The count array stores counts of DDL, INSERT, UPDATE, DELETE, READ, TRUNCATE in that order.
  const uint32_t expected_count[] = {1, 0, 0, 0, 0, 0};
  uint32_t count[] = {0, 0, 0, 0, 0, 0};

  ExpectedRecord expected_records[] = {{0, 0}};

  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, 0));

  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  for (uint32_t i = 0; i < record_size; ++i) {
    const CDCSDKProtoRecordPB record = change_resp.cdc_sdk_proto_records(i);
    CheckRecord(record, expected_records[i], count);
  }
  LOG(INFO) << "Got " << count[1] << " insert record and " << count[0] << " ddl record";
  CheckCount(expected_count, count);
}

// Insert one row, update the inserted row.
// Expected records: (DDL, INSERT, UPDATE).
TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(SingleShardUpdateWithAutoCommit)) {
  auto tablets = ASSERT_RESULT(SetUpCluster());
  ASSERT_EQ(tablets.size(), 1);
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStreamWithReplicationSlot());
  auto set_resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(set_resp.has_error());

  ASSERT_OK(WriteRows(1 /* start */, 2 /* end */, &test_cluster_));
  ASSERT_OK(UpdateRows(1 /* key */, 1 /* value */, &test_cluster_));

  // The count array stores counts of DDL, INSERT, UPDATE, DELETE, READ, TRUNCATE in that order.
  const uint32_t expected_count[] = {1, 1, 1, 0, 0, 0};
  const uint32_t expected_count_with_packed_row[] = {1, 2, 0, 0, 0, 0};
  uint32_t count[] = {0, 0, 0, 0, 0, 0};

  ExpectedRecord expected_records[] = {{0, 0}, {1, 2}, {1, 1}};

  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, 2));


  uint32_t seen_dml_records = 0;
  for (const auto& record : change_resp.cdc_sdk_proto_records()) {
    if (record.row_message().op() == RowMessage::BEGIN ||
        record.row_message().op() == RowMessage::COMMIT) {
      continue;
    }
    CheckRecord(record, expected_records[seen_dml_records], count);
    seen_dml_records++;
  }
  LOG(INFO) << "Got " << count[1] << " insert record and " << count[2] << " update record";
  if (FLAGS_ysql_enable_packed_row) {
    // For packed row if all the columns of a row is updated, it come as INSERT record.
    CheckCount(expected_count_with_packed_row, count);
  } else {
    CheckCount(expected_count, count);
  }
}

// Insert one row, update the inserted row.
// Expected records: (DDL, INSERT, UPDATE).
TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(SingleShardMultiColUpdateWithAutoCommit)) {
  uint32_t num_cols = 4;
  map<std::string, uint32_t> col_val_map;
  auto tablets = ASSERT_RESULT(SetUpClusterMultiColumnUsecase(num_cols));
  ASSERT_EQ(tablets.size(), 1);
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStreamWithReplicationSlot());
  auto set_resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(set_resp.has_error());

  ASSERT_OK(WriteRows(1 /* start */, 2 /* end */, &test_cluster_, num_cols));
  col_val_map.insert({"col2", 1});
  col_val_map.insert({"col3", 1});
  ASSERT_OK(UpdateRows(1 /* key */, col_val_map, &test_cluster_));

  // The count array stores counts of DDL, INSERT, UPDATE, DELETE, READ, TRUNCATE in that order.
  const uint32_t expected_count[] = {1, 1, 1, 0, 0, 0};
  uint32_t count[] = {0, 0, 0, 0, 0, 0};

  VaryingExpectedRecord expected_records[] = {
      {0, {{"col2", 0}, {"col3", 0}, {"col4", 0}}},
      {1, {{"col2", 2}, {"col3", 3}, {"col4", 4}}},
      {1, {{"col2", 1}, {"col3", 1}}}};

  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, 2));


  uint32_t seen_dml_records = 0;
  for (const auto& record : change_resp.cdc_sdk_proto_records()) {
    if (record.row_message().op() == RowMessage::BEGIN ||
        record.row_message().op() == RowMessage::COMMIT) {
      continue;
    }
    CheckRecord(record, expected_records[seen_dml_records], count, num_cols);
    seen_dml_records++;
  }
  LOG(INFO) << "Got " << count[1] << " insert record and " << count[2] << " update record";
  CheckCount(expected_count, count);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestSafeTimePersistedFromGetChangesRequest)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;

  auto tablets = ASSERT_RESULT(SetUpCluster());
  ASSERT_EQ(tablets.size(), 1);
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT, PG_FULL));
  auto set_resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(set_resp.has_error());

  ASSERT_OK(WriteRows(1, 2, &test_cluster_));

  int64 safe_hybrid_time = 12345678;

  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(
      &change_resp, stream_id, tablets, 1, /* is_explicit_checkpoint */false, nullptr, 0,
      safe_hybrid_time));

  auto record_count = change_resp.cdc_sdk_proto_records_size();
  ASSERT_EQ(record_count, 4);

  auto received_safe_time = ASSERT_RESULT(
      GetSafeHybridTimeFromCdcStateTable(stream_id, tablets[0].tablet_id(), test_client()));
  ASSERT_EQ(safe_hybrid_time, received_safe_time);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestSchemaEvolutionWithMultipleStreams)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_populate_end_markers_transactions) = false;
  auto tablets = ASSERT_RESULT(SetUpCluster());
  ASSERT_EQ(tablets.size(), 1);

  // Create 2 cdc streams.
  xrepl::StreamId stream_id_1 = ASSERT_RESULT(CreateDBStreamWithReplicationSlot());
  auto set_resp_1 = ASSERT_RESULT(SetCDCCheckpoint(stream_id_1, tablets));
  ASSERT_FALSE(set_resp_1.has_error());

  xrepl::StreamId stream_id_2 = ASSERT_RESULT(CreateDBStreamWithReplicationSlot());
  auto set_resp_2 = ASSERT_RESULT(SetCDCCheckpoint(stream_id_2, tablets));
  ASSERT_FALSE(set_resp_2.has_error());

  // The count array stores counts of DDL, INSERT, UPDATE, DELETE, READ, TRUNCATE in that order.
  const uint32_t expected_count[] = {3, 3, 3, 1, 0, 0};
  const uint32_t expected_packed_row_count[] = {3, 5, 1, 1, 0, 0};

  uint32_t count_1[] = {0, 0, 0, 0, 0, 0};
  uint32_t count_2[] = {0, 0, 0, 0, 0, 0};

  // Perform sql operations.
  ASSERT_OK(WriteRows(1, 2, &test_cluster_));
  ASSERT_OK(UpdateRows(1, 3, &test_cluster_));
  ASSERT_OK(WriteRows(2, 3, &test_cluster_));
  ASSERT_OK(DeleteRows(1, &test_cluster_));

  ExpectedRecord expected_records_1[] = {{0, 0}, {1, 2}, {1, 3}, {2, 3}, {1, 3}};
  RowMessage::Op expected_record_types_1[] = {
      RowMessage::DDL, RowMessage::INSERT, RowMessage::UPDATE, RowMessage::INSERT,
      RowMessage::DELETE};
  RowMessage::Op expected_packed_row_record_types_1[] = {
      RowMessage::DDL, RowMessage::INSERT, RowMessage::INSERT, RowMessage::INSERT,
      RowMessage::DELETE};

  // Catch up both streams.
  auto change_resp_1 = GetAllPendingChangesFromCdc(stream_id_1, tablets);
  size_t record_size_1 = change_resp_1.records.size();
  auto change_resp_2 = GetAllPendingChangesFromCdc(stream_id_2, tablets);
  size_t record_size_2 = change_resp_2.records.size();

  ASSERT_EQ(record_size_1, 5);
  ASSERT_EQ(record_size_2, 5);

  for (uint32_t i = 0; i < record_size_1; ++i) {
    const CDCSDKProtoRecordPB record = change_resp_1.records[i];
    if (FLAGS_ysql_enable_packed_row) {
      ASSERT_EQ(record.row_message().op(), expected_packed_row_record_types_1[i]);
    } else {
      ASSERT_EQ(record.row_message().op(), expected_record_types_1[i]);
    }
    CheckRecord(record, expected_records_1[i], count_1);
  }

  for (uint32_t i = 0; i < record_size_2; ++i) {
    const CDCSDKProtoRecordPB record = change_resp_2.records[i];
    if (FLAGS_ysql_enable_packed_row) {
      ASSERT_EQ(record.row_message().op(), expected_packed_row_record_types_1[i]);
    } else {
      ASSERT_EQ(record.row_message().op(), expected_record_types_1[i]);
    }
    CheckRecord(record, expected_records_1[i], count_2);
  }

  // Perform sql operations.
  auto conn = ASSERT_RESULT(test_cluster_.ConnectToDB(kNamespaceName));
  ASSERT_OK(conn.Execute("ALTER TABLE test_table ADD COLUMN value_2 INT"));
  ASSERT_OK(conn.Execute("UPDATE test_table SET value_2 = 10 WHERE key = 2"));
  ASSERT_OK(conn.Execute("INSERT INTO test_table VALUES (4, 5, 6)"));

  ExpectedRecordWithThreeColumns expected_records_2[] = {{0, 0, 0}, {2, 10, 0}, {4, 5, 6}};
  bool validate_three_columns_2[] = {false, false, true};
  RowMessage::Op expected_record_types_2[] = {
      RowMessage::DDL, RowMessage::UPDATE, RowMessage::INSERT};

  // Call GetChanges only on stream 1.
  auto previous_checkpoint_1 = change_resp_1.checkpoint;
  change_resp_1 = GetAllPendingChangesFromCdc(stream_id_1, tablets, &previous_checkpoint_1);
  record_size_1 = change_resp_1.records.size();
  ASSERT_EQ(record_size_1, 3);

  for (uint32_t i = 0; i < record_size_1; ++i) {
    const CDCSDKProtoRecordPB record = change_resp_1.records[i];
    ASSERT_EQ(record.row_message().op(), expected_record_types_2[i]);
    CheckRecordWithThreeColumns(
        record, expected_records_2[i], count_1, false, {}, validate_three_columns_2[i]);
  }

  uint32_t records_missed_by_stream_2 = 3;

  // Perform sql operations.
  ASSERT_OK(conn.Execute("ALTER TABLE test_table DROP COLUMN value_2"));
  ASSERT_OK(conn.Execute("UPDATE test_table SET value_1 = 1 WHERE key = 4"));

  ExpectedRecord expected_records_3[] = {{0, 0}, {4, 1}};
  RowMessage::Op expected_record_types_3[] = {RowMessage::DDL, RowMessage::UPDATE};
  RowMessage::Op expected_packed_row_record_types_3[] = {RowMessage::DDL, RowMessage::INSERT};

  // Call GetChanges on stream 1.
  previous_checkpoint_1 = change_resp_1.checkpoint;
  change_resp_1 = GetAllPendingChangesFromCdc(stream_id_1, tablets, &previous_checkpoint_1);
  record_size_1 = change_resp_1.records.size();
  ASSERT_EQ(record_size_1, 2);

  for (uint32_t i = 0; i < record_size_1; ++i) {
    const CDCSDKProtoRecordPB record = change_resp_1.records[i];
    if (FLAGS_ysql_enable_packed_row) {
      ASSERT_EQ(record.row_message().op(), expected_packed_row_record_types_3[i]);
    } else {
      ASSERT_EQ(record.row_message().op(), expected_record_types_3[i]);
    }
    CheckRecord(record, expected_records_3[i], count_1);
  }

  // Call GetChanges on stream 2. Except all records to be received in same order.
  auto previous_checkpoint_2 = change_resp_2.checkpoint;
  change_resp_2 = GetAllPendingChangesFromCdc(stream_id_2, tablets, &previous_checkpoint_2);
  record_size_2 = change_resp_2.records.size();
  ASSERT_EQ(record_size_2, 5);

  for (uint32_t i = 0; i < record_size_2; ++i) {
    const CDCSDKProtoRecordPB record = change_resp_2.records[i];

    if (i < records_missed_by_stream_2) {
      ASSERT_EQ(record.row_message().op(), expected_record_types_2[i]);
      CheckRecordWithThreeColumns(
          record, expected_records_2[i], count_2, false, {}, validate_three_columns_2[i]);
    } else {
      if (FLAGS_ysql_enable_packed_row) {
        ASSERT_EQ(
            record.row_message().op(),
            expected_packed_row_record_types_3[i - records_missed_by_stream_2]);
      } else {
        ASSERT_EQ(
            record.row_message().op(), expected_record_types_3[i - records_missed_by_stream_2]);
      }

      CheckRecord(record, expected_records_3[i - records_missed_by_stream_2], count_2);
    }
  }

  CheckCount(FLAGS_ysql_enable_packed_row ? expected_packed_row_count : expected_count, count_1);
  CheckCount(FLAGS_ysql_enable_packed_row ? expected_packed_row_count : expected_count, count_2);
}

// Insert 3 rows, update 2 of them.
// Expected records: (DDL, 3 INSERT, 2 UPDATE).
TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(SingleShardUpdateRows)) {
  auto tablets = ASSERT_RESULT(SetUpCluster());
  ASSERT_EQ(tablets.size(), 1);
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStreamWithReplicationSlot());
  auto set_resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(set_resp.has_error());

  ASSERT_OK(WriteRows(1 /* start */, 4 /* end */, &test_cluster_));
  ASSERT_OK(UpdateRows(1 /* key */, 1 /* value */, &test_cluster_));
  ASSERT_OK(UpdateRows(2 /* key */, 2 /* value */, &test_cluster_));

  // The count array stores counts of DDL, INSERT, UPDATE, DELETE, READ, TRUNCATE in that order.
  const uint32_t expected_count[] = {1, 3, 2, 0, 0, 0};
  const uint32_t expected_count_with_packed_row[] = {1, 5, 0, 0, 0, 0};
  uint32_t count[] = {0, 0, 0, 0, 0, 0};

  ExpectedRecord expected_records[] = {{0, 0}, {1, 2}, {2, 3}, {3, 4}, {1, 1}, {2, 2}};

  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, 5));

  uint32_t seen_dml_records = 0;
  for (const auto& record : change_resp.cdc_sdk_proto_records()) {
    if (record.row_message().op() == RowMessage::BEGIN ||
        record.row_message().op() == RowMessage::COMMIT) {
      continue;
    }
    CheckRecord(record, expected_records[seen_dml_records], count);
    seen_dml_records++;
  }
  LOG(INFO) << "Got " << count[1] << " insert record and " << count[2] << " update record";
  if (FLAGS_ysql_enable_packed_row) {
    // For packed row if all the columns of a row is updated, it come as INSERT record.
    CheckCount(expected_count_with_packed_row, count);
  } else {
    CheckCount(expected_count, count);
  }
}

// Insert 3 rows, update 2 of them.
// Expected records: (DDL, 3 INSERT, 2 UPDATE).
TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(SingleShardUpdateMultiColumn)) {
  uint32_t num_cols = 4;
  map<std::string, uint32_t> col_val_map;

  auto tablets = ASSERT_RESULT(SetUpClusterMultiColumnUsecase(num_cols));
  ASSERT_EQ(tablets.size(), 1);
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStreamWithReplicationSlot());
  auto set_resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(set_resp.has_error());

  ASSERT_OK(WriteRows(1 /* start */, 4 /* end */, &test_cluster_, num_cols));

  col_val_map.insert(pair<std::string, uint32_t>("col2", 9));
  col_val_map.insert(pair<std::string, uint32_t>("col3", 10));
  ASSERT_OK(UpdateRows(1 /* key */, col_val_map, &test_cluster_));
  ASSERT_OK(UpdateRows(2 /* key */, col_val_map, &test_cluster_));

  // The count array stores counts of DDL, INSERT, UPDATE, DELETE, READ, TRUNCATE in that order.
  const uint32_t expected_count[] = {1, 3, 2, 0, 0, 0};
  uint32_t count[] = {0, 0, 0, 0, 0, 0};

  VaryingExpectedRecord expected_records[] = {
      {0, {std::make_pair("col2", 0), std::make_pair("col3", 0), std::make_pair("col4", 0)}},
      {1, {std::make_pair("col2", 2), std::make_pair("col3", 3), std::make_pair("col4", 4)}},
      {2, {std::make_pair("col2", 3), std::make_pair("col3", 4), std::make_pair("col4", 5)}},
      {3, {std::make_pair("col2", 4), std::make_pair("col3", 5), std::make_pair("col4", 6)}},
      {1, {std::make_pair("col2", 9), std::make_pair("col3", 10)}},
      {2, {std::make_pair("col2", 9), std::make_pair("col3", 10)}}};

  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, 5));

  uint32_t seen_dml_records = 0;
  for (const auto& record : change_resp.cdc_sdk_proto_records()) {
    if (record.row_message().op() == RowMessage::BEGIN ||
        record.row_message().op() == RowMessage::COMMIT) {
      continue;
    }
    CheckRecord(record, expected_records[seen_dml_records++], count, num_cols);
  }
  LOG(INFO) << "Got " << count[1] << " insert record and " << count[2] << " update record";
  CheckCount(expected_count, count);
}

// Insert 3 rows, update 2 of them.
// Expected records: (DDL, 3 INSERT, 2 UPDATE).


// To test upadtes corresponding to a row packed into one CDC record. This verifies the generated
// CDC record in case of subsequent updates Expected records: (DDL, 1 INSERT, 2 UPDATE).
TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(MultiColumnUpdateFollowedByUpdate)) {
  uint32_t num_cols = 3;
  map<std::string, uint32_t> col_val_map1, col_val_map2;

  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_single_record_update) = true;
  ASSERT_OK(SetUpWithParams(3, 1, false));
  auto table = EXPECT_RESULT(CreateTable(
      &test_cluster_, kNamespaceName, kTableName, 1, true, false, 0, false, "", "public",
      num_cols));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, nullptr));
  ASSERT_EQ(tablets.size(), 1);
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStreamWithReplicationSlot());
  auto set_resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(set_resp.has_error());

  col_val_map1.insert({"col2", 9});
  col_val_map1.insert({"col3", 10});
  col_val_map2.insert({"col2", 10});
  col_val_map2.insert({"col3", 11});

  ASSERT_OK(UpdateRowsHelper(
      1 /* start */, 2 /* end */, &test_cluster_, true, 1, col_val_map1, col_val_map2, num_cols));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false,
      /* timeout_secs = */ 30, /* is_compaction = */ false));

  // The count array stores counts of DDL, INSERT, UPDATE, DELETE, READ, TRUNCATE, BEGIN, COMMIT in
  // that order.
  const uint32_t expected_count[] = {1, 1, 2, 0, 0, 0, 1, 1};
  const uint32_t expected_count_with_packed_row[] = {1, 3, 0, 0, 0, 0, 1, 1};
  uint32_t count[] = {0, 0, 0, 0, 0, 0, 0, 0};

  VaryingExpectedRecord expected_records[] = {
      {0, {{"col2", 0}, {"col3", 0}}},   {0, {{"col2", 0}, {"col3", 0}}},
      {1, {{"col2", 2}, {"col3", 3}}},   {1, {{"col2", 9}, {"col3", 10}}},
      {1, {{"col2", 10}, {"col3", 11}}}, {0, {{"col2", 0}, {"col3", 0}}}};

  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, 3));

  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  for (uint32_t i = 0; i < record_size; ++i) {
    const CDCSDKProtoRecordPB record = change_resp.cdc_sdk_proto_records(i);
    CheckRecord(record, expected_records[i], count, num_cols);
  }
  LOG(INFO) << "Got " << count[1] << " insert record and " << count[2] << " update record";
  if (FLAGS_ysql_enable_packed_row) {
    // For packed row if all the columns of a row is updated, it come as INSERT record.
    CheckCount(expected_count_with_packed_row, count);
  } else {
    CheckCount(expected_count, count);
  }
}

// To test upadtes corresponding to a row packed into one CDC record. This verifies the generated
// CDC record in case of subsequent update and delete operations on same row. Expected records:
// (DDL, 1 INSERT, 1 UPDATE, 1 DELETE).
TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(MultiColumnUpdateFollowedByDelete)) {
  uint32_t num_cols = 4;
  map<std::string, uint32_t> col_val_map;

  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_single_record_update) = true;
  ASSERT_OK(SetUpWithParams(3, 1, false));
  auto table = EXPECT_RESULT(CreateTable(
      &test_cluster_, kNamespaceName, kTableName, 1, true, false, 0, false, "", "public",
      num_cols));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, nullptr));
  ASSERT_EQ(tablets.size(), 1);
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStreamWithReplicationSlot());
  auto set_resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(set_resp.has_error());

  col_val_map.insert({"col2", 9});
  col_val_map.insert({"col3", 10});

  ASSERT_OK(UpdateDeleteRowsHelper(
      1 /* start */, 2 /* end */, &test_cluster_, true, 1, col_val_map, num_cols));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false,
      /* timeout_secs = */ 30, /* is_compaction = */ false));

  // The count array stores counts of DDL, INSERT, UPDATE, DELETE, READ, TRUNCATE, BEGIN, COMMIT in
  // that order.
  const uint32_t expected_count[] = {1, 1, 1, 1, 0, 0, 1, 1};
  uint32_t count[] = {0, 0, 0, 0, 0, 0, 0, 0};

  VaryingExpectedRecord expected_records[] = {
      {0, {{"col2", 0}, {"col3", 0}, {"col4", 0}}}, {0, {{"col2", 0}, {"col3", 0}, {"col4", 0}}},
      {1, {{"col2", 2}, {"col3", 3}, {"col4", 4}}}, {1, {{"col2", 9}, {"col3", 10}}},
      {1, {{"col2", 0}, {"col3", 0}, {"col4", 0}}}, {0, {{"col2", 0}, {"col3", 0}, {"col4", 0}}}};

  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, 3));

  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  for (uint32_t i = 0; i < record_size; ++i) {
    const CDCSDKProtoRecordPB record = change_resp.cdc_sdk_proto_records(i);
    CheckRecord(record, expected_records[i], count, num_cols);
  }
  LOG(INFO) << "Got " << count[1] << " insert record, " << count[2] << " update record, and "
            << count[3] << " delete record";
  CheckCount(expected_count, count);
}

// To test upadtes corresponding to a row packed into one CDC record. This verifies the generated
// CDC record in case of subsequent update and update operations on different columns of same row.
// Expected records: (DDL, 1 INSERT, 1 UPDATE, 1 UPDATE).
TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(MultiColumnUpdateFollowedByUpdateSameRow)) {
  uint32_t num_cols = 4;
  map<std::string, uint32_t> col_val_map1, col_val_map2;

  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_single_record_update) = true;
  ASSERT_OK(SetUpWithParams(3, 1, false));
  auto table = EXPECT_RESULT(CreateTable(
      &test_cluster_, kNamespaceName, kTableName, 1, true, false, 0, false, "", "public",
      num_cols));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, nullptr));
  ASSERT_EQ(tablets.size(), 1);
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStreamWithReplicationSlot());
  auto set_resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(set_resp.has_error());

  col_val_map1.insert({"col2", 9});
  col_val_map2.insert({"col3", 11});

  ASSERT_OK(UpdateRowsHelper(
      1 /* start */, 2 /* end */, &test_cluster_, true, 1, col_val_map1, col_val_map2, num_cols));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false,
      /* timeout_secs = */ 30, /* is_compaction = */ false));

  // The count array stores counts of DDL, INSERT, UPDATE, DELETE, READ, TRUNCATE, BEGIN, COMMIT in
  // that order.
  const uint32_t expected_count[] = {1, 1, 2, 0, 0, 0, 1, 1};
  uint32_t count[] = {0, 0, 0, 0, 0, 0, 0, 0};

  VaryingExpectedRecord expected_records[] = {
      {0, {{"col2", 0}, {"col3", 0}, {"col4", 0}}},
      {0, {{"col2", 0}, {"col3", 0}, {"col4", 0}}},
      {1, {{"col2", 2}, {"col3", 3}, {"col4", 4}}},
      {1, {{"col2", 9}}},
      {1, {{"col3", 11}}},
      {0, {{"col2", 0}, {"col3", 0}, {"col4", 0}}}};

  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, 3));

  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  for (uint32_t i = 0; i < record_size; ++i) {
    const CDCSDKProtoRecordPB record = change_resp.cdc_sdk_proto_records(i);
    CheckRecord(record, expected_records[i], count, num_cols);
  }
  LOG(INFO) << "Got " << count[1] << " insert record and " << count[2] << " update record";
  CheckCount(expected_count, count);
}

// Insert one row, delete inserted row.
// Expected records: (DDL, INSERT, DELETE).
TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(SingleShardDeleteWithAutoCommit)) {
  auto tablets = ASSERT_RESULT(SetUpCluster());
  ASSERT_EQ(tablets.size(), 1);
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStreamWithReplicationSlot());
  auto set_resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(set_resp.has_error());

  ASSERT_OK(WriteRows(1 /* start */, 2 /* end */, &test_cluster_));
  ASSERT_OK(DeleteRows(1 /* key */, &test_cluster_));

  // The count array stores counts of DDL, INSERT, UPDATE, DELETE, READ, TRUNCATE in that order.
  const uint32_t expected_count[] = {1, 1, 0, 1, 0, 0};
  uint32_t count[] = {0, 0, 0, 0, 0, 0};

  ExpectedRecord expected_records[] = {{0, 0}, {1, 2}, {1, 0}};

  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, 2));

  uint32_t seen_dml_records = 0;
  for (const auto& record : change_resp.cdc_sdk_proto_records()) {
    if (record.row_message().op() == RowMessage::BEGIN ||
        record.row_message().op() == RowMessage::COMMIT) {
      continue;
    }
    CheckRecord(record, expected_records[seen_dml_records], count);
    seen_dml_records++;
  }
  LOG(INFO) << "Got " << count[1] << " insert record and " << count[3] << " delete record";
  CheckCount(expected_count, count);
}

// Insert 4 rows.
// Expected records: (DDL, INSERT, INSERT, INSERT, INSERT).
TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(SingleShardInsert4Rows)) {
  auto tablets = ASSERT_RESULT(SetUpCluster());
  ASSERT_EQ(tablets.size(), 1);
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStreamWithReplicationSlot());
  auto set_resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(set_resp.has_error());

  ASSERT_OK(WriteRows(1 /* start */, 5 /* end */, &test_cluster_));

  // The count array stores counts of DDL, INSERT, UPDATE, DELETE, READ, TRUNCATE in that order.
  const uint32_t expected_count[] = {1, 4, 0, 0, 0, 0};
  uint32_t count[] = {0, 0, 0, 0, 0, 0};

  ExpectedRecord expected_records[] = {{0, 0}, {1, 2}, {2, 3}, {3, 4}, {4, 5}};

  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, 4));

  uint32_t seen_dml_records = 0;
  for (const auto& record : change_resp.cdc_sdk_proto_records()) {
    if (record.row_message().op() == RowMessage::BEGIN ||
        record.row_message().op() == RowMessage::COMMIT) {
      continue;
    }
    CheckRecord(record, expected_records[seen_dml_records], count);
    seen_dml_records++;
  }
  LOG(INFO) << "Got " << count[1] << " insert records";
  CheckCount(expected_count, count);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(DropDatabase)) {
  ASSERT_OK(SetUpWithParams(3, 1, false));
  ASSERT_RESULT(CreateDBStreamWithReplicationSlot());
  ASSERT_OK(DropDB(&test_cluster_));
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestNeedSchemaInfoFlag)) {
  ASSERT_OK(SetUpWithParams(1, 1, false));

  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName));

  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version = */ nullptr));

  std::string table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStreamWithReplicationSlot());

  auto set_resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(set_resp.has_error());
  // This will write one row with PK = 0.
  ASSERT_OK(WriteRows(0 /* start */, 1 /* end */, &test_cluster_));

  // This is the first call to GetChanges, we will get a DDL record.
  auto resp = ASSERT_RESULT(VerifyIfDDLRecordPresent(stream_id, tablets, false, true));

  // Write another row to the database with PK = 1.
  ASSERT_OK(WriteRows(1 /* start */, 2 /* end */, &test_cluster_));

  // We will not get any DDL record here since this is not the first call and the flag
  // need_schema_info is also unset.
  resp = ASSERT_RESULT(
      VerifyIfDDLRecordPresent(stream_id, tablets, false, false, &resp.cdc_sdk_checkpoint()));

  // Write another row to the database with PK = 2.
  ASSERT_OK(WriteRows(2 /* start */, 3 /* end */, &test_cluster_));

  // We will get a DDL record since we have enabled the need_schema_info flag.
  resp = ASSERT_RESULT(
      VerifyIfDDLRecordPresent(stream_id, tablets, true, false, &resp.cdc_sdk_checkpoint()));
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestEnableTruncateTable)) {
  ASSERT_OK(SetUpWithParams(1, 1, false));

  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName));

  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version = */ nullptr));

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStreamWithReplicationSlot());
  auto set_resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(set_resp.has_error());
  ASSERT_OK(WriteRows(0 /* start */, 1 /* end */, &test_cluster_));
  ASSERT_NOK(TruncateTable(&test_cluster_, {table_id}));

  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_truncate_cdcsdk_table) = true;
  ASSERT_OK(TruncateTable(&test_cluster_, {table_id}));
}

// Insert a single row, truncate table, insert another row.
TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestTruncateTable)) {
  ASSERT_OK(SetUpWithParams(1, 1, false));

  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName));

  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version = */ nullptr));

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStreamWithReplicationSlot());
  auto set_resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(set_resp.has_error());
  ASSERT_OK(WriteRows(0 /* start */, 1 /* end */, &test_cluster_));
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_truncate_cdcsdk_table) = true;
  ASSERT_OK(TruncateTable(&test_cluster_, {table_id}));
  ASSERT_OK(WriteRows(1 /* start */, 2 /* end */, &test_cluster_));

  // Calling Get Changes without enabling truncate flag.
  // Expected records: (DDL, INSERT, INSERT).
  GetChangesResponsePB resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&resp, stream_id, tablets, 2));

  // The count array stores counts of DDL, INSERT, UPDATE, DELETE, READ, TRUNCATE in that order.
  const uint32_t expected_count_truncate_disable[] = {1, 2, 0, 0, 0, 0};
  uint32_t count_truncate_disable[] = {0, 0, 0, 0, 0, 0};
  ExpectedRecord expected_records_truncate_disable[] = {{0, 0}, {0, 1}, {1, 2}};

  uint32_t expected_record_count = 0;
  for (const auto& record : resp.cdc_sdk_proto_records()) {
    if (record.row_message().op() == RowMessage::BEGIN ||
        record.row_message().op() == RowMessage::COMMIT) {
      continue;
    }
    CheckRecord(
        record, expected_records_truncate_disable[expected_record_count], count_truncate_disable);
    expected_record_count++;
  }
  CheckCount(expected_count_truncate_disable, count_truncate_disable);

  // Setting the flag true and calling Get Changes. This will enable streaming of truncate record.
  // Expected records: (DDL, INSERT, TRUNCATE, INSERT).
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_stream_truncate_record) = true;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&resp, stream_id, tablets, 2));

  // The count array stores counts of DDL, INSERT, UPDATE, DELETE, READ, TRUNCATE in that order.
  const uint32_t expected_count_truncate_enable[] = {1, 2, 0, 0, 0, 1};
  uint32_t count_truncate_enable[] = {0, 0, 0, 0, 0, 0};
  ExpectedRecord expected_records_truncate_enable[] = {{0, 0}, {0, 1}, {0, 0}, {1, 2}};
  expected_record_count = 0;
  for (const auto& record : resp.cdc_sdk_proto_records()) {
    if (record.row_message().op() == RowMessage::BEGIN ||
        record.row_message().op() == RowMessage::COMMIT) {
      continue;
    }
    CheckRecord(
        record, expected_records_truncate_enable[expected_record_count], count_truncate_enable);
    expected_record_count++;
  }
  CheckCount(expected_count_truncate_enable, count_truncate_enable);

  LOG(INFO) << "Got " << count_truncate_enable[0] << " ddl records, " << count_truncate_enable[1]
            << " insert records and " << count_truncate_enable[2] << " truncate records";
}
TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestGarbageCollectionFlag)) {
  TestIntentGarbageCollectionFlag(1, true, 10000);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestGarbageCollectionWithSmallInterval)) {
  TestIntentGarbageCollectionFlag(3, true, 5000);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestGarbageCollectionWithLargerInterval)) {
  TestIntentGarbageCollectionFlag(3, true, 10000);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestNoGarbageCollectionBeforeInterval)) {
  TestIntentGarbageCollectionFlag(3, false, 0);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestExtendingIntentRetentionTime)) {
  TestIntentGarbageCollectionFlag(3, true, 10000, true);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestSetCDCCheckpoint)) {
  TestSetCDCCheckpoint(1, false);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestDropTableBeforeCDCStreamDelete)) {
  // Setup cluster.
  ASSERT_OK(SetUpWithParams(1, 1, false));
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName));

  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version = */ nullptr));

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStreamWithReplicationSlot());
  DropTable(&test_cluster_, kTableName);

  // Drop table will trigger the background thread to start the stream metadata cleanup, here
  // test case wait for the metadata cleanup to finish by the background thread.
  ASSERT_OK(WaitFor(
      [&]() -> Result<bool> {
        while (true) {
          auto resp = GetDBStreamInfo(stream_id);
          if (resp.ok() && resp->has_error()) {
            return true;
          }
          continue;
        }
        return false;
      },
      MonoDelta::FromSeconds(60), "Waiting for stream metadata cleanup."));
  // Deleting the created DB Stream ID.
  ASSERT_EQ(DeleteCDCStream(stream_id), false);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestDropTableBeforeXClusterStreamDelete)) {
  // Setup cluster.
  ASSERT_OK(SetUpWithParams(1, 1, false));
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName));

  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version = */ nullptr));

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));

  RpcController rpc;
  CreateCDCStreamRequestPB create_req;
  CreateCDCStreamResponsePB create_resp;

  create_req.set_table_id(table_id);
  create_req.set_source_type(XCLUSTER);
  ASSERT_OK(cdc_proxy_->CreateCDCStream(create_req, &create_resp, &rpc));
  // Drop table on YSQL tables deletes associated xCluster streams.
  DropTable(&test_cluster_, kTableName);

  // Wait for bg thread to cleanup entries from cdc_state.
  CDCStateTable cdc_state_table(test_client());
  ASSERT_OK(WaitFor(
      [&]() -> Result<bool> {
        Status s;
        for (auto row_result : VERIFY_RESULT(cdc_state_table.GetTableRange(
                 CDCStateTableEntrySelector().IncludeCheckpoint(), &s))) {
          RETURN_NOT_OK(row_result);
          auto& row = *row_result;
          if (row.key.stream_id.ToString() == create_resp.stream_id()) {
            return false;
          }
        }
        RETURN_NOT_OK(s);
        return true;
      },
      MonoDelta::FromSeconds(60), "Waiting for stream metadata cleanup."));

  // This should fail now as the stream is deleted.
  ASSERT_EQ(
      DeleteCDCStream(ASSERT_RESULT(xrepl::StreamId::FromString(create_resp.stream_id()))), false);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCheckPointPersistencyNodeRestart)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_update_local_peer_min_index) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 1;
  ASSERT_OK(SetUpWithParams(3, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  std::string table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  // insert some records in transaction.
  ASSERT_OK(WriteRowsHelper(0 /* start */, 100 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  // call get changes.
  GetChangesResponsePB change_resp_1;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp_1, stream_id, tablets, 100));

  uint32_t record_size = change_resp_1.cdc_sdk_proto_records_size();
  LOG(INFO) << "Total records read by get change call: " << record_size;

  ASSERT_OK(WriteRowsHelper(100 /* start */, 200 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));
  // Greater than 100 check because  we got records for BEGIN, COMMIT also.
  ASSERT_GT(record_size, 100);

  // call get changes.
  GetChangesResponsePB change_resp_2;
  ASSERT_OK(WaitForGetChangesToFetchRecords(
      &change_resp_2, stream_id, tablets, 100, /* is_explicit_checkpoint */false,
      &change_resp_1.cdc_sdk_checkpoint()));

  record_size = change_resp_2.cdc_sdk_proto_records_size();
  ASSERT_GT(record_size, 100);
  LOG(INFO) << "Total records read by get change call: " << record_size;

  // Restart one of the node.
  SleepFor(MonoDelta::FromSeconds(1));
  test_cluster()->mini_tablet_server(1)->Shutdown();
  ASSERT_OK(test_cluster()->mini_tablet_server(1)->Start());

  // Check all the tserver checkpoint info it's should be valid.
  for (size_t i = 0; i < test_cluster()->num_tablet_servers(); ++i) {
    for (const auto& peer : test_cluster()->GetTabletPeers(i)) {
      if (peer->tablet_id() == tablets[0].tablet_id()) {
        // What ever checkpoint persisted in the RAFT logs should be same as what ever in memory
        // transaction participant tablet peer.
        ASSERT_EQ(
            peer->cdc_sdk_min_checkpoint_op_id(),
            peer->tablet()->transaction_participant()->GetRetainOpId());
      }
    }
  }
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCleanupSingleStreamSingleTserver)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 1;
  ASSERT_OK(SetUpWithParams(1, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());
  EnableCDCServiceInAllTserver(1);

  // Insert some records in transaction.
  ASSERT_OK(WriteRowsHelper(0 /* start */, 100 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));
  ASSERT_EQ(DeleteCDCStream(stream_id), true);
  VerifyStreamDeletedFromCdcState(test_client(), stream_id, tablets.Get(0).tablet_id());
  VerifyTransactionParticipant(tablets.Get(0).tablet_id(), OpId::Max());
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCleanupSingleStreamMultiTserver)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 1;
  ASSERT_OK(SetUpWithParams(3, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());
  EnableCDCServiceInAllTserver(3);

  // insert some records in transaction.
  ASSERT_OK(WriteRowsHelper(0 /* start */, 100 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));
  ASSERT_EQ(DeleteCDCStream(stream_id), true);
  VerifyStreamDeletedFromCdcState(test_client(), stream_id, tablets.Get(0).tablet_id());
  VerifyTransactionParticipant(tablets.Get(0).tablet_id(), OpId::Max());
}

TEST_F(
    CDCSDKYsqlTest,
    YB_DISABLE_TEST_IN_TSAN(TestCleanupMultiStreamDeleteSingleStreamSingleTserver)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 1;
  ASSERT_OK(SetUpWithParams(1, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  xrepl::StreamId stream_id_1 = ASSERT_RESULT(CreateDBStream(IMPLICIT));
  xrepl::StreamId stream_id_2 = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp_1 = ASSERT_RESULT(SetCDCCheckpoint(stream_id_1, tablets));
  ASSERT_FALSE(resp_1.has_error());
  auto resp_2 = ASSERT_RESULT(SetCDCCheckpoint(stream_id_2, tablets));
  ASSERT_FALSE(resp_2.has_error());
  EnableCDCServiceInAllTserver(1);

  // insert some records in transaction.
  ASSERT_OK(WriteRowsHelper(0 /* start */, 100 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));
  ASSERT_EQ(DeleteCDCStream(stream_id_1), true);
  VerifyStreamDeletedFromCdcState(test_client(), stream_id_1, tablets.Get(0).tablet_id());
  VerifyCdcStateMatches(test_client(), stream_id_2, tablets.Get(0).tablet_id(), 0, 0);
  VerifyTransactionParticipant(tablets.Get(0).tablet_id(), OpId(0, 0));
}

TEST_F(
    CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCleanupMultiStreamDeleteSingleStreamMultiTserver)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 1;
  ASSERT_OK(SetUpWithParams(3, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  xrepl::StreamId stream_id_1 = ASSERT_RESULT(CreateDBStream(IMPLICIT));
  xrepl::StreamId stream_id_2 = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp_1 = ASSERT_RESULT(SetCDCCheckpoint(stream_id_1, tablets));
  ASSERT_FALSE(resp_1.has_error());
  auto resp_2 = ASSERT_RESULT(SetCDCCheckpoint(stream_id_2, tablets));
  ASSERT_FALSE(resp_2.has_error());
  EnableCDCServiceInAllTserver(3);

  // Insert some records in transaction.
  ASSERT_OK(WriteRowsHelper(0 /* start */, 100 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));
  ASSERT_EQ(DeleteCDCStream(stream_id_1), true);
  VerifyStreamDeletedFromCdcState(test_client(), stream_id_1, tablets.Get(0).tablet_id());
  VerifyCdcStateMatches(test_client(), stream_id_2, tablets.Get(0).tablet_id(), 0, 0);
  VerifyTransactionParticipant(tablets.Get(0).tablet_id(), OpId(0, 0));
}

TEST_F(
    CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCleanupMultiStreamDeleteAllStreamsSingleTserver)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 1;
  ASSERT_OK(SetUpWithParams(1, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  xrepl::StreamId stream_id_1 = ASSERT_RESULT(CreateDBStream(IMPLICIT));
  xrepl::StreamId stream_id_2 = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp_1 = ASSERT_RESULT(SetCDCCheckpoint(stream_id_1, tablets));
  ASSERT_FALSE(resp_1.has_error());
  auto resp_2 = ASSERT_RESULT(SetCDCCheckpoint(stream_id_2, tablets));
  ASSERT_FALSE(resp_2.has_error());
  EnableCDCServiceInAllTserver(1);

  // insert some records in transaction.
  ASSERT_OK(WriteRowsHelper(0 /* start */, 100 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));
  ASSERT_EQ(DeleteCDCStream(stream_id_1), true);
  VerifyStreamDeletedFromCdcState(test_client(), stream_id_1, tablets.Get(0).tablet_id());
  VerifyTransactionParticipant(tablets.Get(0).tablet_id(), OpId(0, 0));
  ASSERT_EQ(DeleteCDCStream(stream_id_2), true);
  VerifyStreamDeletedFromCdcState(test_client(), stream_id_2, tablets.Get(0).tablet_id());
  VerifyTransactionParticipant(tablets.Get(0).tablet_id(), OpId::Max());
}

TEST_F(
    CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCleanupMultiStreamDeleteAllStreamsMultiTserver)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 1;
  ASSERT_OK(SetUpWithParams(3, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  xrepl::StreamId stream_id_1 = ASSERT_RESULT(CreateDBStream(IMPLICIT));
  xrepl::StreamId stream_id_2 = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp_1 = ASSERT_RESULT(SetCDCCheckpoint(stream_id_1, tablets));
  ASSERT_FALSE(resp_1.has_error());
  auto resp_2 = ASSERT_RESULT(SetCDCCheckpoint(stream_id_2, tablets));
  ASSERT_FALSE(resp_2.has_error());
  EnableCDCServiceInAllTserver(3);

  // insert some records in transaction.
  ASSERT_OK(WriteRowsHelper(0 /* start */, 100 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));
  ASSERT_EQ(DeleteCDCStream(stream_id_1), true);
  VerifyStreamDeletedFromCdcState(test_client(), stream_id_1, tablets.Get(0).tablet_id());
  VerifyTransactionParticipant(tablets.Get(0).tablet_id(), OpId(0, 0));
  ASSERT_EQ(DeleteCDCStream(stream_id_2), true);
  VerifyStreamDeletedFromCdcState(test_client(), stream_id_2, tablets.Get(0).tablet_id());
  VerifyTransactionParticipant(tablets.Get(0).tablet_id(), OpId::Max());
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestMultpleStreamOnSameTablet)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_intent_retention_ms) = 10000;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;
  ASSERT_OK(SetUpWithParams(1, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));

  vector<xrepl::StreamId> stream_id;
  // Create 2 streams
  for (uint32_t idx = 0; idx < 2; idx++) {
    stream_id.push_back(ASSERT_RESULT(CreateDBStream(IMPLICIT)));
    auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id[idx], tablets));
    ASSERT_FALSE(resp.has_error());
  }

  // Insert some records in transaction.
  vector<GetChangesResponsePB> change_resp_01(2);
  vector<GetChangesResponsePB> change_resp_02(2);
  ASSERT_OK(WriteRowsHelper(0 /* start */, 100 /* end */, &test_cluster_, true));
  for (uint32_t stream_idx = 0; stream_idx < 2; stream_idx++) {
    uint32_t record_size = 0;
    ASSERT_OK(WaitForGetChangesToFetchRecords(
        &change_resp_01[stream_idx], stream_id[stream_idx], tablets, 100));
    record_size = change_resp_01[stream_idx].cdc_sdk_proto_records_size();
    LOG(INFO) << "Total records read by get change call on stream_id_" << stream_idx
              << " total records: " << record_size;
  }

  // Keep inserting some records into the table and call GetChange on stream_id_02
  // to see the inserted record count.
  uint32_t idx = 0;
  const uint32_t loop_count = 10;
  GetChangesResponsePB change_resp_2_stream_id_02;
  while (idx < loop_count) {
    change_resp_02[1] = ASSERT_RESULT(UpdateCheckpoint(stream_id[1], tablets, &change_resp_01[1]));
    idx += 1;
    change_resp_01[0] = change_resp_02[1];
    SleepFor(MonoDelta::FromMilliseconds(200));
  }

  // Now call GetChanges for stream_01.
  SleepFor(MonoDelta::FromMilliseconds(FLAGS_cdc_intent_retention_ms));
  auto result = GetChangesFromCDC(stream_id[0], tablets, &change_resp_01[0].cdc_sdk_checkpoint());
  ASSERT_EQ(!result.ok(), true);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestMultpleActiveStreamOnSameTablet)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_update_local_peer_min_index) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;
  ASSERT_OK(SetUpWithParams(3, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));

  // Create 2 streams
  vector<xrepl::StreamId> stream_ids;
  for (uint32_t idx = 0; idx < 2; idx++) {
    auto stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));
    stream_ids.push_back(stream_id);
    auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
    ASSERT_FALSE(resp.has_error());
  }
  // GetChanges for the stream-1 and stream-2
  vector<GetChangesResponsePB> change_resp_01(2);
  vector<GetChangesResponsePB> change_resp_02(2);
  uint32_t start = 0;
  uint32_t end = 100;
  for (uint32_t insert_idx = 0; insert_idx < 3; insert_idx++) {
    ASSERT_OK(WriteRowsHelper(start /* start */, end /* end */, &test_cluster_, true));
    ASSERT_OK(test_client()->FlushTables(
        {table.table_id()}, /* add_indexes = */
        false,              /* timeout_secs = */
        30, /* is_compaction = */ false));
    for (uint32_t stream_idx = 0; stream_idx < 2; stream_idx++) {
      uint32_t record_size = 0;
      if (insert_idx == 0) {
        ASSERT_OK(WaitForGetChangesToFetchRecords(
            &change_resp_01[stream_idx], stream_ids[stream_idx], tablets, 100));

        record_size = change_resp_01[stream_idx].cdc_sdk_proto_records_size();
      } else {
        change_resp_02[stream_idx] = ASSERT_RESULT(
            UpdateCheckpoint(stream_ids[stream_idx], tablets, &change_resp_01[stream_idx]));
        change_resp_01[stream_idx] = change_resp_02[stream_idx];
        record_size = change_resp_02[stream_idx].cdc_sdk_proto_records_size();
      }
      ASSERT_GE(record_size, 100);
    }
    start = end;
    end = start + 100;
  }

  OpId min_checkpoint = OpId::Max();

  CDCStateTable cdc_state_table(test_client());
  Status s;
  for (auto row_result : ASSERT_RESULT(
           cdc_state_table.GetTableRange(CDCStateTableEntrySelector().IncludeCheckpoint(), &s))) {
    ASSERT_OK(row_result);
    auto& row = *row_result;

    LOG(INFO) << "Read cdc_state table with tablet_id: " << row.key.tablet_id
              << " stream_id: " << row.key.stream_id << " checkpoint is: " << *row.checkpoint;
    min_checkpoint = min(min_checkpoint, *row.checkpoint);
  }
  ASSERT_OK(s);

  ASSERT_OK(WaitFor(
      [&]() {
        // Read the tablet LEADER as well as FOLLOWER's transaction_participation
        // Check all the tserver checkpoint info it's should be valid.
        uint32_t i = 0;
        while (i < test_cluster()->num_tablet_servers()) {
          for (const auto& peer : test_cluster()->GetTabletPeers(i)) {
            if (peer->tablet_id() == tablets[0].tablet_id()) {
              if (peer->tablet()->transaction_participant()->GetRetainOpId() != min_checkpoint) {
                SleepFor(MonoDelta::FromMilliseconds(2));
              } else {
                i += 1;
                LOG(INFO) << "In tserver: " << i
                          << " tablet peer have transaction_participant op_id set as: "
                          << peer->tablet()->transaction_participant()->GetRetainOpId();
              }
              break;
            }
          }
        }
        return true;
      },
      MonoDelta::FromSeconds(60), "Waiting for all the tservers intent counts"));
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestActiveAndInActiveStreamOnSameTablet)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_intent_retention_ms) = 20000;
  uint32_t num_tservers = 3;
  ASSERT_OK(SetUpWithParams(num_tservers, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));

  vector<xrepl::StreamId> stream_id;
  // Create 2 streams
  for (uint32_t idx = 0; idx < 2; idx++) {
    stream_id.push_back(ASSERT_RESULT(CreateDBStream(IMPLICIT)));
    auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id[idx], tablets));
    ASSERT_FALSE(resp.has_error());
  }
  // Insert some records in transaction.
  ASSERT_OK(WriteRows(0 /* start */, 100 /* end */, &test_cluster_));

  vector<GetChangesResponsePB> change_resp(2);
  // Call GetChanges for the stream-1 and stream-2
  for (uint32_t idx = 0; idx < 2; idx++) {
    ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp[idx], stream_id[idx], tablets, 100));

    uint32_t record_size = change_resp[idx].cdc_sdk_proto_records_size();
    ASSERT_GE(record_size, 100);
    LOG(INFO) << "Total records read by GetChanges call on stream_id: " << record_size;
  }

  // Get the checkpoint details of the stream-2 and tablet-1 from the cdc_state table.
  auto checkpoints_stream_2 = ASSERT_RESULT(GetCDCCheckpoint(stream_id[1], tablets));

  // Keep stream-1 active.
  uint32_t idx = 0;
  const uint32_t total_count = 10;
  while (idx < total_count) {
    uint32_t record_size = 0;
    ASSERT_OK(WriteRows(100 + idx /* start */, 101 + idx /* end */, &test_cluster_));
    GetChangesResponsePB latest_change_resp;
    ASSERT_OK(WaitForGetChangesToFetchRecords(
        &latest_change_resp, stream_id[0], tablets, 1, /* is_explicit_checkpoint */false,
        &change_resp[0].cdc_sdk_checkpoint()));
    record_size = latest_change_resp.cdc_sdk_proto_records_size();
    change_resp[0] = latest_change_resp;
    ASSERT_GE(record_size, 1);
    idx += 1;
    // This check is to make sure that UpdatePeersAndMetrics thread gets the CPU slot to execute, so
    // that it updates minimum checkpoint and active time in tablet LEADER and FOLLOWERS so that GC
    // can be controlled.
    for (uint tserver_index = 0; tserver_index < num_tservers; tserver_index++) {
      for (const auto& peer : test_cluster()->GetTabletPeers(tserver_index)) {
        if (peer->tablet_id() == tablets[0].tablet_id()) {
          ASSERT_OK(WaitFor(
              [&]() -> Result<bool> {
                // Here checkpoints_stream_2[0].index is compared because on the same tablet 2
                // streams are created whereas on stream_2 there is no Getchanges call, so minimum
                // checkpoint that will be updated in tablet LEADER and FOLLOWERS will be the
                // checkpoint that is set for stream_id_2
                // + tablet_id during setCDCCheckpoint.
                if (checkpoints_stream_2[0].index == peer->cdc_sdk_min_checkpoint_op_id().index) {
                  return true;
                }
                SleepFor(MonoDelta::FromMilliseconds(100));
                return false;
              },
              MonoDelta::FromSeconds(60),
              "Failed to update checkpoint in tablet peer."));
        }
      }
    }
  }

  OpId overall_min_checkpoint = OpId::Max();
  OpId active_stream_checkpoint;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 100000;

  CDCStateTable cdc_state_table(test_client());
  Status s;
  for (auto row_result : ASSERT_RESULT(
           cdc_state_table.GetTableRange(CDCStateTableEntrySelector().IncludeCheckpoint(), &s))) {
    ASSERT_OK(row_result);
    auto& row = *row_result;

    GetChangesResponsePB latest_change_resp = ASSERT_RESULT(
        GetChangesFromCDC(stream_id[0], tablets, &change_resp[0].cdc_sdk_checkpoint()));
    if (row.key.tablet_id == tablets[0].tablet_id() &&
        stream_id[0] == row.key.stream_id) {
      LOG(INFO) << "Read cdc_state table with tablet_id: " << row.key.tablet_id
                << " stream_id: " << row.key.stream_id << " checkpoint is: " << *row.checkpoint;
      active_stream_checkpoint = *row.checkpoint;
    } else {
      overall_min_checkpoint = min(overall_min_checkpoint, *row.checkpoint);
    }
  }

  ASSERT_OK(WaitFor(
      [&]() -> Result<bool> {
        // Read the tablet LEADER as well as FOLLOWER's transaction_participation
        // Check all the tserver checkpoint info it's should be valid.
        uint32_t i = 0;
        while (i < test_cluster()->num_tablet_servers()) {
          for (const auto& peer : test_cluster()->GetTabletPeers(i)) {
            if (peer->tablet_id() == tablets[0].tablet_id()) {
              if (peer->tablet()->transaction_participant()->GetRetainOpId() !=
                      overall_min_checkpoint &&
                  peer->tablet()->transaction_participant()->GetRetainOpId() !=
                      active_stream_checkpoint) {
                SleepFor(MonoDelta::FromMilliseconds(2));
              } else {
                i += 1;
                LOG(INFO) << "In tserver: " << i
                          << " tablet peer have transaction_participant op_id set as: "
                          << peer->tablet()->transaction_participant()->GetRetainOpId();
              }
              break;
            }
          }
        }
        return true;
      },
      MonoDelta::FromSeconds(60), "Waiting for all the tservers intent counts"));
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCheckPointPersistencyAllNodesRestart)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_load_balancing) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_metrics_interval_ms) = 1;
  ASSERT_OK(SetUpWithParams(3, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  // Insert some records in transaction.
  ASSERT_OK(WriteRowsHelper(0 /* start */, 100 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  // Call get changes.
  GetChangesResponsePB change_resp_1;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp_1, stream_id, tablets, 100));
  uint32_t record_size = change_resp_1.cdc_sdk_proto_records_size();
  LOG(INFO) << "Total records read by GetChanges call: " << record_size;
  // Greater than 100 check because  we got records for BEGIN, COMMIT also.
  ASSERT_GT(record_size, 100);

  ASSERT_OK(WriteRowsHelper(100 /* start */, 200 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  // Call get changes.
  GetChangesResponsePB change_resp_2;
  ASSERT_OK(WaitForGetChangesToFetchRecords(
      &change_resp_2, stream_id, tablets, 100, /* is_explicit_checkpoint */false,
      &change_resp_1.cdc_sdk_checkpoint()));
  record_size = change_resp_2.cdc_sdk_proto_records_size();
  ASSERT_GT(record_size, 100);
  LOG(INFO) << "Total records read by second GetChanges call: " << record_size;

  auto checkpoints = ASSERT_RESULT(GetCDCCheckpoint(stream_id, tablets));
  LOG(INFO) << "Checkpoint after final GetChanges: " << checkpoints[0];

  // Restart all the nodes.
  SleepFor(MonoDelta::FromSeconds(1));
  for (size_t i = 0; i < test_cluster()->num_tablet_servers(); ++i) {
    test_cluster()->mini_tablet_server(i)->Shutdown();
    ASSERT_OK(test_cluster()->mini_tablet_server(i)->Start());
  }
  LOG(INFO) << "All nodes restarted";
  EnableCDCServiceInAllTserver(3);

  // Check the checkpoint info for all tservers - it should be valid.
  for (size_t i = 0; i < test_cluster()->num_tablet_servers(); ++i) {
    for (const auto& peer : test_cluster()->GetTabletPeers(i)) {
      if (peer->tablet_id() == tablets[0].tablet_id()) {
        ASSERT_OK(WaitFor(
            [&]() -> Result<bool> {
              // Checkpoint persisted in the RAFT logs should be same as in memory transaction
              // participant tablet peer.
              if (peer->cdc_sdk_min_checkpoint_op_id() !=
                      peer->tablet()->transaction_participant()->GetRetainOpId() ||
                  checkpoints[0] != peer->cdc_sdk_min_checkpoint_op_id()) {
                return false;
              }
              return true;
            },
            MonoDelta::FromSeconds(60),
            "Checkpoints are not as expected"));
      }
    }
  }
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestIntentCountPersistencyAllNodesRestart)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  // We want to force every GetChanges to update the cdc_state table.
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;
  ASSERT_OK(SetUpWithParams(1, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  // Insert some records in transaction.
  ASSERT_OK(WriteRowsHelper(0 /* start */, 100 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));
  GetChangesResponsePB change_resp_1;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp_1, stream_id, tablets, 100));
  LOG(INFO) << "Number of records after first transaction: " << change_resp_1.records().size();

  ASSERT_OK(WriteRowsHelper(100 /* start */, 200 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  ASSERT_OK(WriteRowsHelper(200 /* start */, 300 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));
  SleepFor(MonoDelta::FromSeconds(10));

  int64 initial_num_intents;
  PollForIntentCount(1, 0, IntentCountCompareOption::GreaterThan, &initial_num_intents);

  LOG(INFO) << "All nodes will be restarted";
  for (size_t i = 0; i < test_cluster()->num_tablet_servers(); ++i) {
    test_cluster()->mini_tablet_server(i)->Shutdown();
    ASSERT_OK(test_cluster()->mini_tablet_server(i)->Start());
  }
  LOG(INFO) << "All nodes restarted";
  SleepFor(MonoDelta::FromSeconds(60));

  int64 num_intents_after_restart;
  PollForIntentCount(
      initial_num_intents, 0, IntentCountCompareOption::EqualTo, &num_intents_after_restart);
  LOG(INFO) << "Number of intents after restart: " << num_intents_after_restart;
  ASSERT_EQ(num_intents_after_restart, initial_num_intents);

  GetChangesResponsePB change_resp_2;
  ASSERT_OK(WaitForGetChangesToFetchRecords(
      &change_resp_2, stream_id, tablets, 200, /* is_explicit_checkpoint */false,
      &change_resp_1.cdc_sdk_checkpoint()));

  uint32_t record_size = change_resp_2.cdc_sdk_proto_records_size();
  // We have run 2 transactions after the last call to "GetChangesFromCDC", thus we expect
  // atleast 200 records if we call "GetChangesFromCDC" now.
  LOG(INFO) << "Number of records after restart: " << record_size;
  ASSERT_GE(record_size, 200);

  // Now that there are no more transaction, and we have called "GetChangesFromCDC" already, there
  // must be no more records or intents remaining.
  GetChangesResponsePB change_resp_3 =
      ASSERT_RESULT(GetChangesFromCDC(stream_id, tablets, &change_resp_2.cdc_sdk_checkpoint()));
  uint32_t final_record_size = change_resp_3.cdc_sdk_proto_records_size();
  LOG(INFO) << "Number of recrods after no new transactions: " << final_record_size;
  ASSERT_EQ(final_record_size, 0);

  int64 final_num_intents;
  PollForIntentCount(0, 0, IntentCountCompareOption::EqualTo, &final_num_intents);
  ASSERT_EQ(0, final_num_intents);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestHighIntentCountPersistencyAllNodesRestart)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_log_segment_size_bytes) = 100;
  ASSERT_OK(SetUpWithParams(1, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  // Insert some records in transaction.
  ASSERT_OK(WriteRowsHelper(0 /* start */, 1 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  ASSERT_OK(WriteRowsHelper(1, 75, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  int64 initial_num_intents;
  PollForIntentCount(1, 0, IntentCountCompareOption::GreaterThan, &initial_num_intents);
  LOG(INFO) << "Number of intents before restart: " << initial_num_intents;

  LOG(INFO) << "All nodes will be restarted";
  for (size_t i = 0; i < test_cluster()->num_tablet_servers(); ++i) {
    test_cluster()->mini_tablet_server(i)->Shutdown();
    ASSERT_OK(test_cluster()->mini_tablet_server(i)->Start());
  }
  LOG(INFO) << "All nodes restarted";
  SleepFor(MonoDelta::FromSeconds(60));

  int64 num_intents_after_restart;
  PollForIntentCount(
      initial_num_intents, 0, IntentCountCompareOption::EqualTo, &num_intents_after_restart);
  LOG(INFO) << "Number of intents after restart: " << num_intents_after_restart;
  ASSERT_EQ(num_intents_after_restart, initial_num_intents);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestIntentCountPersistencyBootstrap)) {
  // Disable lb as we move tablets around
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_load_balancing) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_metrics_interval_ms) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 1;
  ASSERT_OK(SetUpWithParams(3, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  // Insert some records in transaction.
  ASSERT_OK(WriteRowsHelper(0 /* start */, 100 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));
  GetChangesResponsePB change_resp_1;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp_1, stream_id, tablets, 100));

  size_t first_leader_index = -1;
  size_t first_follower_index = -1;
  GetTabletLeaderAndAnyFollowerIndex(tablets, &first_leader_index, &first_follower_index);
  if (first_leader_index == 0) {
    // We want to avoid the scenario where the first TServer is the leader, since we want to shut
    // the leader TServer down and call GetChanges. GetChanges will be called on the cdc_proxy based
    // on the first TServer's address and we want to avoid the network issues.
    ASSERT_OK(ChangeLeaderOfTablet(first_follower_index, tablets[0].tablet_id()));
    std::swap(first_leader_index, first_follower_index);
  }

  ASSERT_OK(WriteRowsHelper(100 /* start */, 200 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  ASSERT_OK(ChangeLeaderOfTablet(first_follower_index, tablets[0].tablet_id()));
  // Shutdown tserver hosting tablet initial leader, now it is a follower.
  test_cluster()->mini_tablet_server(first_leader_index)->Shutdown();
  LOG(INFO) << "TServer hosting tablet leader shutdown";

  ASSERT_OK(WaitForGetChangesToFetchRecords(
      &change_resp_1, stream_id, tablets, 100, /* is_explicit_checkpoint */false,
      &change_resp_1.cdc_sdk_checkpoint()));

  // Restart the tserver hosting the initial leader.
  ASSERT_OK(test_cluster()->mini_tablet_server(first_leader_index)->Start());
  SleepFor(MonoDelta::FromSeconds(1));

  OpId last_seen_checkpoint_op_id = OpId::Invalid();
  int64 last_seen_num_intents = -1;
  for (uint32_t i = 0; i < test_cluster()->num_tablet_servers(); ++i) {
    auto tablet_peer_result =
        test_cluster()->GetTabletManager(i)->GetServingTablet(tablets[0].tablet_id());
    if (!tablet_peer_result.ok()) {
      continue;
    }
    auto tablet_peer = std::move(*tablet_peer_result);

    OpId checkpoint = (*tablet_peer).cdc_sdk_min_checkpoint_op_id();
    LOG(INFO) << "Checkpoint OpId : " << checkpoint << " ,  on tserver index: " << i;
    if (last_seen_checkpoint_op_id == OpId::Invalid()) {
      last_seen_checkpoint_op_id = checkpoint;
    } else {
      ASSERT_EQ(last_seen_checkpoint_op_id, checkpoint);
    }

    int64 num_intents;
    if (last_seen_num_intents == -1) {
      PollForIntentCount(0, i, IntentCountCompareOption::GreaterThan, &num_intents);
      last_seen_num_intents = num_intents;
    } else {
      PollForIntentCount(
          last_seen_num_intents, i, IntentCountCompareOption::GreaterThanOrEqualTo, &num_intents);
      ASSERT_EQ(last_seen_num_intents, num_intents);
    }
    LOG(INFO) << "Num of intents: " << num_intents << ", on tserver index" << i;
  }
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestEnum)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_update_local_peer_min_index) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 1;
  ASSERT_OK(SetUpWithParams(3, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(
      CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets, true, false, 0, true));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  std::string table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  int insert_count = 10;
  // Insert some records in transaction.
  ASSERT_OK(WriteEnumsRows(0, insert_count, &test_cluster_));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  // Call get changes.
  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, insert_count));
  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  ASSERT_GT(record_size, insert_count);

  int expected_key = 0;
  for (uint32_t i = 0; i < record_size; ++i) {
    if (change_resp.cdc_sdk_proto_records(i).row_message().op() == RowMessage::INSERT) {
      const CDCSDKProtoRecordPB record = change_resp.cdc_sdk_proto_records(i);
      ASSERT_EQ(expected_key, record.row_message().new_tuple(0).datum_int32());
      ASSERT_EQ(
          expected_key % 2 ? "FIXED" : "PERCENTAGE",
          record.row_message().new_tuple(1).datum_string());
      expected_key++;
    }
  }

  ASSERT_EQ(insert_count, expected_key);
}

// Tests that the enum cache is correctly re-populated on a cache miss.
TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestEnumOnRestart)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_update_local_peer_min_index) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 1;
  ASSERT_OK(SetUpWithParams(1, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(
      CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets, true, false, 0, true));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  std::string table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  int insert_count = 20;
  // Insert some records in transaction.
  ASSERT_OK(WriteEnumsRows(0, insert_count / 2, &test_cluster_));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  // Restart one of the node.
  SleepFor(MonoDelta::FromSeconds(1));
  test_cluster()->mini_tablet_server(0)->Shutdown();
  ASSERT_OK(test_cluster()->mini_tablet_server(0)->Start());

  // Insert some more records in transaction.
  ASSERT_OK(WriteEnumsRows(insert_count / 2, insert_count, &test_cluster_));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  size_t record_size = 0;
  GetAllPendingChangesResponse change_resp;
  // Call get changes.
  ASSERT_OK(WaitFor([&]() -> Result<bool> {
          change_resp = GetAllPendingChangesFromCdc(stream_id, tablets);
          record_size = change_resp.records.size();
          return static_cast<int> (record_size) > insert_count;
  }, MonoDelta::FromSeconds(2), "Wait for receiving all the records"));

  int expected_key = 0;
  for (uint32_t i = 0; i < record_size; ++i) {
    if (change_resp.records[i].row_message().op() == RowMessage::INSERT) {
      const CDCSDKProtoRecordPB record = change_resp.records[i];
      ASSERT_EQ(expected_key, record.row_message().new_tuple(0).datum_int32());
      ASSERT_EQ(
          expected_key % 2 ? "FIXED" : "PERCENTAGE",
          record.row_message().new_tuple(1).datum_string());
      expected_key++;
    }
  }

  ASSERT_EQ(insert_count, expected_key);
}

// Tests that the enum cache is correctly re-populated on stream creation.
TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestEnumMultipleStreams)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_update_local_peer_min_index) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 1;
  ASSERT_OK(SetUpWithParams(1, 1, false));

  const uint32_t num_tablets = 1;

  auto table = ASSERT_RESULT(
      CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets, true, false, 0, true));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version=*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  std::string table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));
  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  int insert_count = 10;

  auto table1 = ASSERT_RESULT(CreateTable(
      &test_cluster_, kNamespaceName, kTableName, num_tablets, true, false, 0, true, "1"));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets1;
  ASSERT_OK(test_client()->GetTablets(table1, 0, &tablets1, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets1.size(), num_tablets);

  xrepl::StreamId stream_id1 = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp1 = ASSERT_RESULT(SetCDCCheckpoint(stream_id1, tablets1));
  ASSERT_FALSE(resp1.has_error());

  // Insert some records in transaction.
  ASSERT_OK(WriteEnumsRows(0, insert_count, &test_cluster_, "1"));
  ASSERT_OK(test_client()->FlushTables(
      {table1.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  // Call get changes.
  GetChangesResponsePB change_resp1;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp1, stream_id1, tablets1, insert_count));
  uint32_t record_size1 = change_resp1.cdc_sdk_proto_records_size();
  ASSERT_GT(record_size1, insert_count);

  int expected_key = 0;
  for (uint32_t i = 0; i < record_size1; ++i) {
    if (change_resp1.cdc_sdk_proto_records(i).row_message().op() == RowMessage::INSERT) {
      const CDCSDKProtoRecordPB record = change_resp1.cdc_sdk_proto_records(i);
      ASSERT_EQ(expected_key, record.row_message().new_tuple(0).datum_int32());
      ASSERT_EQ(
          expected_key % 2 ? "FIXED1" : "PERCENTAGE1",
          record.row_message().new_tuple(1).datum_string());
      expected_key++;
    }
  }

  ASSERT_EQ(insert_count, expected_key);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCompositeType)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_update_local_peer_min_index) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 1;
  ASSERT_OK(SetUpWithParams(3, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateCompositeTable(&test_cluster_, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  std::string table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, "emp"));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  int insert_count = 10;
  // Insert some records in transaction.
  ASSERT_OK(WriteCompositeRows(0, insert_count, &test_cluster_));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  // Call get changes.
  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, insert_count));

  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  ASSERT_GT(record_size, insert_count);

  int expected_key = 0;
  for (uint32_t i = 0; i < record_size; ++i) {
    if (change_resp.cdc_sdk_proto_records(i).row_message().op() == RowMessage::INSERT) {
      const CDCSDKProtoRecordPB record = change_resp.cdc_sdk_proto_records(i);
      ASSERT_EQ(expected_key, record.row_message().new_tuple(0).datum_int32());
      ASSERT_EQ("(John,Doe)", record.row_message().new_tuple(1).datum_string());
      expected_key++;
    }
  }
  ASSERT_EQ(insert_count, expected_key);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCompositeTypeWithRestart)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_update_local_peer_min_index) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 1;
  ASSERT_OK(SetUpWithParams(3, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateCompositeTable(&test_cluster_, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  std::string table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, "emp"));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  int insert_count = 20;
  // Insert some records in transaction.
  ASSERT_OK(WriteCompositeRows(0, insert_count / 2, &test_cluster_));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  // Restart one of the node.
  SleepFor(MonoDelta::FromSeconds(1));
  test_cluster()->mini_tablet_server(0)->Shutdown();
  ASSERT_OK(test_cluster()->mini_tablet_server(0)->Start());

  // Insert some more records in transaction.
  ASSERT_OK(WriteCompositeRows(insert_count / 2, insert_count, &test_cluster_));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  // Call get changes.
  auto change_resp = GetAllPendingChangesFromCdc(stream_id, tablets);
  size_t record_size = change_resp.records.size();
  ASSERT_GT(record_size, insert_count);

  int expected_key = 0;
  for (uint32_t i = 0; i < record_size; ++i) {
    if (change_resp.records[i].row_message().op() == RowMessage::INSERT) {
      const CDCSDKProtoRecordPB record = change_resp.records[i];
      ASSERT_EQ(expected_key, record.row_message().new_tuple(0).datum_int32());
      ASSERT_EQ("(John,Doe)", record.row_message().new_tuple(1).datum_string());
      expected_key++;
    }
  }
  ASSERT_EQ(insert_count, expected_key);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestNestedCompositeType)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_update_local_peer_min_index) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 1;
  ASSERT_OK(SetUpWithParams(3, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateNestedCompositeTable(&test_cluster_, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  std::string table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, "emp_nested"));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  int insert_count = 10;
  // Insert some records in transaction.
  ASSERT_OK(WriteNestedCompositeRows(0, insert_count, &test_cluster_));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  // Call get changes.
  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, insert_count));

  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  ASSERT_GT(record_size, insert_count);

  int expected_key = 0;
  for (uint32_t i = 0; i < record_size; ++i) {
    if (change_resp.cdc_sdk_proto_records(i).row_message().op() == RowMessage::INSERT) {
      const CDCSDKProtoRecordPB record = change_resp.cdc_sdk_proto_records(i);
      ASSERT_EQ(expected_key, record.row_message().new_tuple(0).datum_int32());
      ASSERT_EQ("(\"(John,Middle)\",Doe)", record.row_message().new_tuple(1).datum_string());
      expected_key++;
    }
  }
  ASSERT_EQ(insert_count, expected_key);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestArrayCompositeType)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_update_local_peer_min_index) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 1;
  ASSERT_OK(SetUpWithParams(3, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateArrayCompositeTable(&test_cluster_, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  std::string table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, "emp_array"));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  int insert_count = 10;
  // Insert some records in transaction.
  ASSERT_OK(WriteArrayCompositeRows(0, insert_count, &test_cluster_));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  // Call get changes.
  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, insert_count));

  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  ASSERT_GT(record_size, insert_count);

  int expected_key = 0;
  for (uint32_t i = 0; i < record_size; ++i) {
    if (change_resp.cdc_sdk_proto_records(i).row_message().op() == RowMessage::INSERT) {
      const CDCSDKProtoRecordPB record = change_resp.cdc_sdk_proto_records(i);
      ASSERT_EQ(expected_key, record.row_message().new_tuple(0).datum_int32());
      ASSERT_EQ(
          "(\"{John,Middle,Doe}\",\"{123,456}\")",
          record.row_message().new_tuple(1).datum_string());
      expected_key++;
    }
  }
  ASSERT_EQ(insert_count, expected_key);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestRangeCompositeType)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_update_local_peer_min_index) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 1;
  ASSERT_OK(SetUpWithParams(3, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateRangeCompositeTable(&test_cluster_, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  std::string table_id =
      ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, "range_composite_table"));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  int insert_count = 10;
  // Insert some records in transaction.
  ASSERT_OK(WriteRangeCompositeRows(0, insert_count, &test_cluster_));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  // Call get changes.
  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, insert_count));

  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  ASSERT_GT(record_size, insert_count);

  int expected_key = 0;
  for (uint32_t i = 0; i < record_size; ++i) {
    if (change_resp.cdc_sdk_proto_records(i).row_message().op() == RowMessage::INSERT) {
      const CDCSDKProtoRecordPB record = change_resp.cdc_sdk_proto_records(i);
      ASSERT_EQ(expected_key, record.row_message().new_tuple(0).datum_int32());
      ASSERT_EQ(
          Format(
              "(\"[$0,$1]\",\"[$2,$3)\")", expected_key, expected_key + 10, expected_key + 11,
              expected_key + 21),
          record.row_message().new_tuple(1).datum_string());
      expected_key++;
    }
  }
  ASSERT_EQ(insert_count, expected_key);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestRangeArrayCompositeType)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_update_local_peer_min_index) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 1;
  ASSERT_OK(SetUpWithParams(3, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateRangeArrayCompositeTable(&test_cluster_, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  std::string table_id =
      ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, "range_array_composite_table"));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  int insert_count = 10;
  // Insert some records in transaction.
  ASSERT_OK(WriteRangeArrayCompositeRows(0, insert_count, &test_cluster_));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  // Call get changes.
  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, insert_count));
  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  ASSERT_GT(record_size, insert_count);

  int expected_key = 0;
  for (uint32_t i = 0; i < record_size; ++i) {
    if (change_resp.cdc_sdk_proto_records(i).row_message().op() == RowMessage::INSERT) {
      const CDCSDKProtoRecordPB record = change_resp.cdc_sdk_proto_records(i);
      ASSERT_EQ(expected_key, record.row_message().new_tuple(0).datum_int32());
      ASSERT_EQ(
          Format(
              "(\"{\"\"[$0,$1]\"\",\"\"[$2,$3]\"\"}\",\"{\"\"[$4,$5)\"\"}\")", expected_key,
              expected_key + 10, expected_key + 11, expected_key + 20, expected_key + 21,
              expected_key + 31),
          record.row_message().new_tuple(1).datum_string());
      expected_key++;
    }
  }
  ASSERT_EQ(insert_count, expected_key);
}

// Test GetChanges() can return records of a transaction with size was greater than
// 'consensus_max_batch_size_bytes'.
TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestTransactionWithLargeBatchSize)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_consensus_max_batch_size_bytes) = 1000;
  ASSERT_OK(SetUpWithParams(1, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version=*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  // Insert some records in transaction.
  ASSERT_OK(WriteRowsHelper(0 /* start */, 100 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));
  GetChangesResponsePB change_resp_1;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp_1, stream_id, tablets, 100));
  LOG(INFO) << "Number of records after first transaction: " << change_resp_1.records().size();

  // Insert some records in transaction.
  ASSERT_OK(WriteRowsHelper(100, 500, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  int64 initial_num_intents;
  PollForIntentCount(400, 0, IntentCountCompareOption::GreaterThan, &initial_num_intents);
  LOG(INFO) << "Number of intents: " << initial_num_intents;

  GetChangesResponsePB change_resp_2;
  ASSERT_OK(WaitForGetChangesToFetchRecords(
      &change_resp_2, stream_id, tablets, 400, /* is_explicit_checkpoint */false,
      &change_resp_1.cdc_sdk_checkpoint()));

  uint32_t record_size = change_resp_2.cdc_sdk_proto_records_size();
  // We have run 1 transactions after the last call to "GetChangesFromCDC", thus we expect
  // atleast 400 records if we call "GetChangesFromCDC" now.
  LOG(INFO) << "Number of records after second transaction: " << record_size;
  ASSERT_GE(record_size, 400);
  ASSERT_RESULT(GetChangesFromCDC(stream_id, tablets, &change_resp_2.cdc_sdk_checkpoint()));

  int64 final_num_intents;
  PollForIntentCount(0, 0, IntentCountCompareOption::EqualTo, &final_num_intents);
  ASSERT_EQ(0, final_num_intents);
  LOG(INFO) << "Final number of intents: " << final_num_intents;
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestIntentCountPersistencyAfterCompaction)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  // We want to force every GetChanges to update the cdc_state table.
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_aborted_intent_cleanup_ms) = 1000;  // 1 sec

  ASSERT_OK(SetUpWithParams(1, 1, false));
  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));

  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  // Insert some records in transaction.
  ASSERT_OK(WriteRowsHelper(0 /* start */, 100 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));
  GetChangesResponsePB change_resp_1;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp_1, stream_id, tablets, 100));
  LOG(INFO) << "Number of records after first transaction: " << change_resp_1.records().size();

  ASSERT_OK(WriteRowsHelper(100 /* start */, 200 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  ASSERT_OK(WriteRowsHelper(200 /* start */, 300 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));
  SleepFor(MonoDelta::FromSeconds(10));

  int64 initial_num_intents;
  PollForIntentCount(1, 0, IntentCountCompareOption::GreaterThan, &initial_num_intents);

  SleepFor(MonoDelta::FromSeconds(60));
  LOG(INFO) << "All nodes will be restarted";
  for (size_t i = 0; i < test_cluster()->num_tablet_servers(); ++i) {
    test_cluster()->mini_tablet_server(i)->Shutdown();
    ASSERT_OK(test_cluster()->mini_tablet_server(i)->Start());
  }
  LOG(INFO) << "All nodes restarted";

  std::this_thread::sleep_for(std::chrono::milliseconds(FLAGS_aborted_intent_cleanup_ms));
  ASSERT_OK(test_cluster_.mini_cluster_->CompactTablets());
  std::this_thread::sleep_for(std::chrono::seconds(10));

  int64 num_intents_after_compaction;
  PollForIntentCount(
      initial_num_intents, 0, IntentCountCompareOption::EqualTo, &num_intents_after_compaction);
  LOG(INFO) << "Number of intents after compaction: " << num_intents_after_compaction;
  ASSERT_EQ(num_intents_after_compaction, initial_num_intents);

  GetChangesResponsePB change_resp_2;
  ASSERT_OK(WaitForGetChangesToFetchRecords(
      &change_resp_2, stream_id, tablets, 200, /* is_explicit_checkpoint */false,
      &change_resp_1.cdc_sdk_checkpoint()));

  uint32_t record_size = change_resp_2.cdc_sdk_proto_records_size();

  // We have run 2 transactions after the last call to "GetChangesFromCDC", thus we expect
  // atleast 200 records if we call "GetChangesFromCDC" now.
  LOG(INFO) << "Number of records after compaction: " << record_size;
  ASSERT_GE(record_size, 200);

  // Now that there are no more transaction, and we have called "GetChangesFromCDC" already, there
  // must be no more records or intents remaining.
  GetChangesResponsePB change_resp_3 =
      ASSERT_RESULT(GetChangesFromCDC(stream_id, tablets, &change_resp_2.cdc_sdk_checkpoint()));
  uint32_t final_record_size = change_resp_3.cdc_sdk_proto_records_size();
  LOG(INFO) << "Number of recrods after no new transactions: " << final_record_size;
  ASSERT_EQ(final_record_size, 0);

  int64 final_num_intents;
  PollForIntentCount(0, 0, IntentCountCompareOption::EqualTo, &final_num_intents);
  ASSERT_EQ(0, final_num_intents);
}

// https://github.com/yugabyte/yugabyte-db/issues/19385
TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestLogGCForNewTablesAddedAfterCreateStream)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 100000;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_log_segment_size_bytes) = 100;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_log_min_seconds_to_retain) = 10;
  ASSERT_OK(SetUpWithParams(1, 1, false));
  const uint32_t num_tablets = 1;

  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(EXPLICIT));

  // Create the table AFTER the stream has been created
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;

  // Wait for a second for the table to be created and the tablets to be RUNNING
  // Only after this will the tablets of this table get entries in cdc_state table
  SleepFor(MonoDelta::FromSeconds(1));
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version=*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  // Insert some records.
  ASSERT_OK(WriteRows(0 /* start */, 100 /* end */, &test_cluster_));

  GetChangesResponsePB change_resp_1;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp_1, stream_id, tablets, 100));
  LOG(INFO) << "Number of records after first transaction: "
            << change_resp_1.cdc_sdk_proto_records_size();
  ASSERT_GE(change_resp_1.cdc_sdk_proto_records_size(), 100);

  ASSERT_OK(WriteRows(100 /* start */, 200 /* end */, &test_cluster_));

  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 100,
      /* is_compaction = */ false));

  SleepFor(MonoDelta::FromSeconds(FLAGS_log_min_seconds_to_retain));
  // Here testcase behave like a WAL cleaner thread.
  for (size_t i = 0; i < test_cluster()->num_tablet_servers(); ++i) {
    for (const auto& tablet_peer : test_cluster()->GetTabletPeers(i)) {
      if (tablet_peer->tablet_id() == tablets[0].tablet_id()) {
        // There are multiple factors that influence WAL retention.
        // The goal of this test is to verify whether "wal_retention_secs"
        // has been appropriately set or not (for tables added after stream creation).
        // Hence, we first ensure that other factors do not have an effect
        // on WAL retention through appropriate settings. For example, we set
        //   FLAGS_cdc_min_replicated_index_considered_stale_secs = 1
        //   FLAGS_log_min_seconds_to_retain = 10
        // Now, at this stage, "wal_retention_secs" will primarily influence WAL retention.
        // The test will pass if and only if "wal_retention_secs" has been approproately
        // set. If not, the WAL will be GCed, errors will result and the test will fail
        ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_min_replicated_index_considered_stale_secs) = 1;
        ASSERT_OK(tablet_peer->RunLogGC());
      }
    }
  }

  GetChangesResponsePB change_resp_2;
  ASSERT_OK(WaitForGetChangesToFetchRecords(
      &change_resp_2, stream_id, tablets, 100, /* is_explicit_checkpoint */true,
      &change_resp_1.cdc_sdk_checkpoint()));

  LOG(INFO) << "Number of records after second transaction: "
            << change_resp_2.cdc_sdk_proto_records_size();
  ASSERT_GE(change_resp_2.cdc_sdk_proto_records_size(), 100);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestLogGCedWithTabletBootStrap)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 100000;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_log_segment_size_bytes) = 100;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_log_min_seconds_to_retain) = 10;
  ASSERT_OK(SetUpWithParams(1, 1, false));
  const uint32_t num_tablets = 1;

  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;

  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version=*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));
  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  // Insert some records.
  ASSERT_OK(WriteRows(0 /* start */, 100 /* end */, &test_cluster_));

  GetChangesResponsePB change_resp_1;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp_1, stream_id, tablets, 100));
  LOG(INFO) << "Number of records after first transaction: "
            << change_resp_1.cdc_sdk_proto_records_size();
  ASSERT_GE(change_resp_1.cdc_sdk_proto_records_size(), 100);

  ASSERT_OK(WriteRows(100 /* start */, 200 /* end */, &test_cluster_));
  // SleepFor(MonoDelta::FromSeconds(FLAGS_cdc_min_replicated_index_considered_stale_secs * 2));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 100,
      /* is_compaction = */ false));

  // Restart of the tsever will make Tablet Bootstrap.
  test_cluster()->mini_tablet_server(0)->Shutdown();
  ASSERT_OK(test_cluster()->mini_tablet_server(0)->Start());

  SleepFor(MonoDelta::FromSeconds(FLAGS_log_min_seconds_to_retain));
  // Here testcase behave like a WAL cleaner thread.
  for (size_t i = 0; i < test_cluster()->num_tablet_servers(); ++i) {
    for (const auto& tablet_peer : test_cluster()->GetTabletPeers(i)) {
      if (tablet_peer->tablet_id() == tablets[0].tablet_id()) {
        // Here setting FLAGS_cdc_min_replicated_index_considered_stale_secs to 1, so that CDC
        // replication index will be set to max value, which will create a scenario to clean stale
        // WAL logs, even if CDCSDK no consumed those Logs.
        ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_min_replicated_index_considered_stale_secs) = 1;
        ASSERT_OK(tablet_peer->RunLogGC());
      }
    }
  }

  GetChangesResponsePB change_resp_2;
  ASSERT_OK(WaitForGetChangesToFetchRecords(
      &change_resp_2, stream_id, tablets, 100, /* is_explicit_checkpoint */false,
      &change_resp_1.cdc_sdk_checkpoint()));

  LOG(INFO) << "Number of records after second transaction: "
            << change_resp_2.cdc_sdk_proto_records_size();
  ASSERT_GE(change_resp_2.cdc_sdk_proto_records_size(), 100);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestXClusterLogGCedWithTabletBootStrap)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 100000;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_log_segment_size_bytes) = 100;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_log_min_seconds_to_retain) = 10;
  ASSERT_OK(SetUpWithParams(1, 1, false));
  const uint32_t num_tablets = 1;

  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;

  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version=*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  RpcController rpc;
  CreateCDCStreamRequestPB create_req;
  CreateCDCStreamResponsePB create_resp;
  create_req.set_table_id(table_id);
  create_req.set_source_type(XCLUSTER);
  ASSERT_OK(cdc_proxy_->CreateCDCStream(create_req, &create_resp, &rpc));

  // Insert some records.
  ASSERT_OK(WriteRows(0 /* start */, 100 /* end */, &test_cluster_));
  rpc.Reset();

  GetChangesRequestPB change_req;
  GetChangesResponsePB change_resp_1;
  change_req.set_stream_id(create_resp.stream_id());
  change_req.set_tablet_id(tablets[0].tablet_id());
  change_req.mutable_from_checkpoint()->mutable_op_id()->set_index(0);
  change_req.mutable_from_checkpoint()->mutable_op_id()->set_term(0);
  change_req.set_serve_as_proxy(true);
  rpc.set_timeout(MonoDelta::FromSeconds(kRpcTimeout));
  ASSERT_OK(cdc_proxy_->GetChanges(change_req, &change_resp_1, &rpc));
  ASSERT_FALSE(change_resp_1.has_error());

  ASSERT_OK(WriteRows(100 /* start */, 200 /* end */, &test_cluster_));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 100,
      /* is_compaction = */ false));

  // Restart of the tsever will make Tablet Bootstrap.
  test_cluster()->mini_tablet_server(0)->Shutdown();
  ASSERT_OK(test_cluster()->mini_tablet_server(0)->Start());

  SleepFor(MonoDelta::FromSeconds(FLAGS_log_min_seconds_to_retain));
  // Here testcase behave like a WAL cleaner thread.
  for (size_t i = 0; i < test_cluster()->num_tablet_servers(); ++i) {
    for (const auto& tablet_peer : test_cluster()->GetTabletPeers(i)) {
      if (tablet_peer->tablet_id() == tablets[0].tablet_id()) {
        // Here setting FLAGS_cdc_min_replicated_index_considered_stale_secs to 1, so that CDC
        // replication index will be set to max value, which will create a scenario to clean stale
        // WAL logs, even if CDCSDK no consumed those Logs.
        ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_min_replicated_index_considered_stale_secs) = 1;
        ASSERT_OK(tablet_peer->RunLogGC());
      }
    }
  }

  GetChangesResponsePB change_resp_2;
  rpc.Reset();
  change_req.set_stream_id(create_resp.stream_id());
  change_req.set_tablet_id(tablets[0].tablet_id());
  change_req.mutable_from_checkpoint()->mutable_op_id()->set_index(
      0);
  change_req.mutable_from_checkpoint()->mutable_op_id()->set_term(
      0);
  change_req.set_serve_as_proxy(true);
  rpc.set_timeout(MonoDelta::FromSeconds(kRpcTimeout));

  ASSERT_OK(cdc_proxy_->GetChanges(change_req, &change_resp_2, &rpc));
  ASSERT_FALSE(change_resp_2.has_error());
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestEnumWithMultipleTablets)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_update_local_peer_min_index) = false;

  const uint32_t num_tablets = 3;
  vector<TabletId> table_id(2);
  vector<const char*> listTablesName{"test_table_01", "test_table_02"};
  vector<std::string> tablePrefix{"_01", "_02"};
  const int total_stream_count = 2;
  vector<google::protobuf::RepeatedPtrField<master::TabletLocationsPB>> tablets(2);

  ASSERT_OK(SetUpWithParams(3, 1, false));

  // Here we are verifying Enum Cache for a tablespace that needs to be re-updated // if there is a
  // cache miss in any of the tsever. This can happen when enum cache entry is created for the
  // all the tservers as part of CreateCDCStream or GetChanges call and later stage client
  // created one more enum type on the same tablespace and a new table, then GetChanges call on
  // the newtable should not fail,(precondition:- create new stream in same namespace).
  for (int idx = 0; idx < total_stream_count; idx++) {
    auto table = ASSERT_RESULT(CreateTable(
        &test_cluster_, kNamespaceName, kTableName, num_tablets, true, false, 0, true,
        tablePrefix[idx]));
    ASSERT_OK(
        test_client()->GetTablets(table, 0, &tablets[idx], /* partition_list_version =*/nullptr));
    ASSERT_EQ(tablets[idx].size(), num_tablets);

    table_id[idx] = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, listTablesName[idx]));
    auto stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

    for (uint32_t jdx = 0; jdx < num_tablets; jdx++) {
      auto resp = ASSERT_RESULT(
          SetCDCCheckpoint(stream_id, tablets[idx], OpId::Min(), kuint64max, true, jdx));
    }

    ASSERT_OK(WriteEnumsRows(0, 100, &test_cluster_, tablePrefix[idx], kNamespaceName, kTableName));
    ASSERT_OK(test_client()->FlushTables(
        {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
        /* is_compaction = */ false));

    int total_count = 0;
    for (uint32_t kdx = 0; kdx < num_tablets; kdx++) {
      GetChangesResponsePB change_resp =
          ASSERT_RESULT(GetChangesFromCDC(stream_id, tablets[idx], nullptr, kdx));
      for (const auto& record : change_resp.cdc_sdk_proto_records()) {
        if (record.row_message().op() == RowMessage::INSERT) {
          total_count += 1;
        }
      }
    }
    LOG(INFO) << "Total GetChanges record counts: " << total_count;
    ASSERT_EQ(total_count, 100);
  }
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestSetCDCCheckpointWithHigherTserverThanTablet)) {
  // Create a cluster where the number of tservers are 5 (tserver-1, tserver-2, tserver-3,
  // tserver-4, tserver-5). Create table with tablet split 3(tablet-1, tablet-2, tablet-3).
  // Consider the tablet-1 LEADER is in tserver-3, tablet-2 LEADER in tserver-4 and tablet-3 LEADER
  // is in tserver-5. Consider cdc proxy connection is created with tserver-1. calling
  // setCDCCheckpoint from tserver-1 should PASS.
  // Since number of tablets is lesser than the number of tservers, there must be atleast 2 tservers
  // which do not host any of the tablet. But still, calling setCDCCheckpoint any of the
  // tserver, even the ones not hosting tablet, should PASS.
  ASSERT_OK(SetUpWithParams(5, 1, false));

  const uint32_t num_tablets = 3;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);
  std::string table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStreamWithReplicationSlot());

  for (uint32_t idx = 0; idx < num_tablets; idx++) {
    auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets, OpId::Min(), true, idx));
    ASSERT_FALSE(resp.has_error());
  }
}


TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestGetTabletListToPollForCDC)) {
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(SetUpWithParams(3, 1, false));
  const uint32_t num_tablets = 3;

  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);
  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));

  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));
  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  auto get_tablets_resp = ASSERT_RESULT(GetTabletListToPollForCDC(stream_id, table_id));

  ASSERT_EQ(get_tablets_resp.tablet_checkpoint_pairs().size(), num_tablets);

  // Assert that GetTabletListToPollForCDC also populates the snapshot_time.
  for (const auto& tablet_checkpoint_pair : get_tablets_resp.tablet_checkpoint_pairs()) {
    ASSERT_GE(tablet_checkpoint_pair.cdc_sdk_checkpoint().snapshot_time(), 0);
  }
}

// Here creating a single table inside a namespace and a CDC stream on top of the namespace.
// Deleting the table should clean every thing from master cache as well as the system
// catalog.
TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestStreamMetaDataCleanupAndDropTable)) {
  // Setup cluster.
  ASSERT_OK(SetUpWithParams(3, 1, false));

  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName));

  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version = */ nullptr));

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStreamWithReplicationSlot());
  ASSERT_OK(WriteRows(0 /* start */, 100 /* end */, &test_cluster_));

  DropTable(&test_cluster_, kTableName);
  ASSERT_OK(WaitFor(
      [&]() -> Result<bool> {
        while (true) {
          auto get_resp = GetDBStreamInfo(stream_id);
          // Wait until the background thread cleanup up the stream-id.
          if (get_resp.ok() && get_resp->has_error() && get_resp->table_info_size() == 0) {
            return true;
          }
        }
      },
      MonoDelta::FromSeconds(60), "Waiting for stream metadata cleanup."));
}

// Here we are creating multiple tables and a CDC stream on the same namespace.
// Deleting multiple tables from the namespace should only clean metadata related to
// deleted tables from master cache as well as system catalog.
TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestStreamMetaDataCleanupMultiTableDrop)) {
  // Setup cluster.
  ASSERT_OK(SetUpWithParams(3, 1, false));
  const vector<string> table_list_suffix = {"_1", "_2", "_3"};
  const int kNumTables = 3;
  vector<YBTableName> table(kNumTables);
  int idx = 0;
  vector<google::protobuf::RepeatedPtrField<master::TabletLocationsPB>> tablets(kNumTables);

  for (auto table_suffix : table_list_suffix) {
    table[idx] = ASSERT_RESULT(CreateTable(
        &test_cluster_, kNamespaceName, kTableName, 1, true, false, 0, true, table_suffix));
    ASSERT_OK(test_client()->GetTablets(
        table[idx], 0, &tablets[idx], /* partition_list_version = */ nullptr));
    TableId table_id =
        ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName + table_suffix));

    ASSERT_OK(WriteEnumsRows(
        0 /* start */, 100 /* end */, &test_cluster_, table_suffix, kNamespaceName, kTableName));
    idx += 1;
  }
  auto stream_id = ASSERT_RESULT(CreateDBStreamWithReplicationSlot());

  // Drop table1 and table2 from the namespace, check stream associated with namespace should not
  // be deleted, but metadata related to the dropped tables should be cleaned up from the master.
  for (int idx = 1; idx < kNumTables; idx++) {
    char drop_table[64] = {0};
    (void)snprintf(drop_table, sizeof(drop_table), "%s_%d", kTableName, idx);
    DropTable(&test_cluster_, drop_table);
  }

  ASSERT_OK(WaitFor(
      [&]() -> Result<bool> {
        while (true) {
          auto get_resp = GetDBStreamInfo(stream_id);
          // Wait until the background thread cleanup up the drop table metadata.
          if (get_resp.ok() && !get_resp->has_error() && get_resp->table_info_size() == 1) {
            return true;
          }
        }
      },
      MonoDelta::FromSeconds(60), "Waiting for stream metadata cleanup."));

  for (int idx = 0; idx < 2; idx++) {
    auto change_resp = GetChangesFromCDC(stream_id, tablets[idx], nullptr);
    // test_table_1 and test_table_2 GetChanges should retrun error where as test_table_3 should
    // succeed.
    if (idx == 0 || idx == 1) {
      ASSERT_FALSE(change_resp.ok());

    } else {
      uint32_t record_size = (*change_resp).cdc_sdk_proto_records_size();
      ASSERT_GT(record_size, 100);
    }
  }

  // Verify that cdc_state only has tablets from table3 left.
  std::unordered_set<TabletId> table_3_tablet_ids;
  for (const auto& tablet : tablets[2]) {
    table_3_tablet_ids.insert(tablet.tablet_id());
  }
  CDCStateTable cdc_state_table(test_client());
  ASSERT_OK(WaitFor(
      [&]() -> Result<bool> {
        Status s;
        std::unordered_set<TabletId> tablets_found;
        for (auto row_result : VERIFY_RESULT(cdc_state_table.GetTableRange(
                 CDCStateTableEntrySelector().IncludeCheckpoint(), &s))) {
          RETURN_NOT_OK(row_result);
          auto& row = *row_result;
          if (row.key.stream_id == stream_id && !table_3_tablet_ids.contains(row.key.tablet_id)) {
            // Still have a tablet left over from a dropped table.
            return false;
          }
          if (row.key.stream_id == stream_id) {
            tablets_found.insert(row.key.tablet_id);
          }
        }
        RETURN_NOT_OK(s);
        LOG(INFO) << "tablets found: " << AsString(tablets_found)
                  << ", expected tablets: " << AsString(table_3_tablet_ids);
        return table_3_tablet_ids == tablets_found;
      },
      MonoDelta::FromSeconds(60), "Waiting for stream metadata cleanup."));

  // Deleting the created stream.
  ASSERT_TRUE(DeleteCDCStream(stream_id));

  // GetChanges should retrun error, for all tables.
  for (int idx = 0; idx < 2; idx++) {
    auto change_resp = GetChangesFromCDC(stream_id, tablets[idx], nullptr);
    ASSERT_FALSE(change_resp.ok());
  }
}

// After delete stream, metadata related to stream should be deleted from the master cache as well
// as system catalog.
TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestStreamMetaCleanUpAndDeleteStream)) {
  // Setup cluster.
  ASSERT_OK(SetUpWithParams(3, 1, false));

  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName));

  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version = */ nullptr));

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStreamWithReplicationSlot());
  ASSERT_OK(WriteRows(0 /* start */, 100 /* end */, &test_cluster_));

  // Deleting the created DB Stream ID.
  ASSERT_TRUE(DeleteCDCStream(stream_id));

  ASSERT_OK(WaitFor(
      [&]() -> Result<bool> {
        while (true) {
          auto get_resp = GetDBStreamInfo(stream_id);
          // Wait until the background thread cleanup up the stream-id.
          if (get_resp.ok() && get_resp->has_error() && get_resp->table_info_size() == 0) {
            return true;
          }
        }
      },
      MonoDelta::FromSeconds(60), "Waiting for stream metadata cleanup."));
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestDeletedStreamRowRemovedEvenAfterGetChanges)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 60;

  ASSERT_OK(SetUpWithParams(1, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  // Insert some records in transaction.
  ASSERT_OK(WriteRowsHelper(0 /* start */, 100 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));
  GetChangesResponsePB change_resp_1;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp_1, stream_id, tablets, 100));

  ASSERT_OK(WriteRowsHelper(100 /* start */, 200 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  ASSERT_EQ(DeleteCDCStream(stream_id), true);
  VerifyStreamCheckpointInCdcState(
      test_client(), stream_id, tablets[0].tablet_id(), OpIdExpectedValue::MaxOpId);
  LOG(INFO) << "The stream's checkpoint has been marked as OpId::Max()";

  ASSERT_OK(WaitForGetChangesToFetchRecords(
      &change_resp_1, stream_id, tablets, 100, /* is_explicit_checkpoint */false,
      &change_resp_1.cdc_sdk_checkpoint()));

  VerifyStreamCheckpointInCdcState(
      test_client(), stream_id, tablets[0].tablet_id(), OpIdExpectedValue::ValidNonMaxOpId);
  LOG(INFO) << "Verified that GetChanges() overwrote checkpoint from OpId::Max().";

  // We shutdown the TServer so that the stream cache is cleared.
  test_cluster()->mini_tablet_server(0)->Shutdown();
  ASSERT_OK(test_cluster()->mini_tablet_server(0)->Start());

  // We verify that the row is deleted even after GetChanges() overwrote the OpId from Max.
  VerifyStreamDeletedFromCdcState(test_client(), stream_id, tablets[0].tablet_id());
}

// Here we are creating a table test_table_1 and a CDC stream ex:- stream-id-1.
// Now create another table test_table_2 and create another stream ex:- stream-id-2 on the same
// namespace. stream-id-1 and stream-id-2 are now associated with test_table_1. drop test_table_1,
// call GetDBStreamInfo on both stream-id, we should not get any information related to drop table.
TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestMultiStreamOnSameTableAndDropTable)) {
  // Prevent newly added tables to be added to existing active streams.
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdcsdk_table_processing_limit_per_run) = 0;
  // Setup cluster.
  ASSERT_OK(SetUpWithParams(3, 1, false));
  const vector<string> table_list_suffix = {"_1", "_2"};
  vector<YBTableName> table(2);
  vector<xrepl::StreamId> stream_ids;
  vector<google::protobuf::RepeatedPtrField<master::TabletLocationsPB>> tablets(2);

  for (int idx = 0; idx < 2; idx++) {
    table[idx] = ASSERT_RESULT(CreateTable(
        &test_cluster_, kNamespaceName, kTableName, 1, true, false, 0, true,
        table_list_suffix[idx]));
    ASSERT_OK(test_client()->GetTablets(
        table[idx], 0, &tablets[idx], /* partition_list_version = */ nullptr));
    TableId table_id = ASSERT_RESULT(
        GetTableId(&test_cluster_, kNamespaceName, kTableName + table_list_suffix[idx]));

    stream_ids.push_back(ASSERT_RESULT(CreateDBStreamWithReplicationSlot()));
    ASSERT_OK(WriteEnumsRows(
        0 /* start */, 100 /* end */, &test_cluster_, table_list_suffix[idx], kNamespaceName,
        kTableName));
  }

  // Drop table test_table_1 which is associated with both streams.
  for (int idx = 1; idx < 2; idx++) {
    char drop_table[64] = {0};
    (void)snprintf(drop_table, sizeof(drop_table), "%s_%d", kTableName, idx);
    DropTable(&test_cluster_, drop_table);
  }

  ASSERT_OK(WaitFor(
      [&]() -> Result<bool> {
        int idx = 1;
        while (idx <= 2) {
          auto get_resp = GetDBStreamInfo(stream_ids[idx - 1]);
          if (!get_resp.ok()) {
            return false;
          }
          // stream-1 is associated with a single table, so as part of table drop, stream-1 should
          // be cleaned and wait until the background thread is done with cleanup.
          if (idx == 1 && false == get_resp->has_error()) {
            continue;
          }
          // stream-2 is associated with both tables, so dropping one table, should not clean the
          // stream from cache as well as from system catalog, except the dropped table metadata.
          if (idx > 1 && get_resp->table_info_size() > 1) {
            continue;
          }
          idx += 1;
        }
        return true;
      },
      MonoDelta::FromSeconds(60), "Waiting for stream metadata cleanup."));
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestMultiStreamOnSameTableAndDeleteStream)) {
  // Setup cluster.
  ASSERT_OK(SetUpWithParams(3, 1, false));
  const vector<string> table_list_suffix = {"_1", "_2"};
  vector<YBTableName> table(2);
  vector<xrepl::StreamId> stream_ids;
  vector<google::protobuf::RepeatedPtrField<master::TabletLocationsPB>> tablets(2);

  for (int idx = 0; idx < 2; idx++) {
    table[idx] = ASSERT_RESULT(CreateTable(
        &test_cluster_, kNamespaceName, kTableName, 1, true, false, 0, true,
        table_list_suffix[idx]));
    ASSERT_OK(test_client()->GetTablets(
        table[idx], 0, &tablets[idx], /* partition_list_version = */ nullptr));
    TableId table_id = ASSERT_RESULT(
        GetTableId(&test_cluster_, kNamespaceName, kTableName + table_list_suffix[idx]));

    stream_ids.push_back(ASSERT_RESULT(CreateDBStreamWithReplicationSlot()));
    ASSERT_OK(WriteEnumsRows(
        0 /* start */, 100 /* end */, &test_cluster_, table_list_suffix[idx], kNamespaceName,
        kTableName));
  }

  // Deleting the stream-2 associated with both tables
  ASSERT_TRUE(DeleteCDCStream(stream_ids[1]));

  ASSERT_OK(WaitFor(
      [&]() -> Result<bool> {
        int idx = 1;
        while (idx <= 2) {
          auto get_resp = GetDBStreamInfo(stream_ids[idx - 1]);
          if (!get_resp.ok()) {
            return false;
          }

          // stream-2 is deleted, so its metadata from the master cache as well as from the system
          // catalog should be cleaned and wait until the background thread is done with the
          // cleanup.
          if (idx > 1 && (false == get_resp->has_error() || get_resp->table_info_size() != 0)) {
            continue;
          }
          idx += 1;
        }
        return true;
      },
      MonoDelta::FromSeconds(60), "Waiting for stream metadata cleanup."));
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCreateStreamAfterSetCheckpointMax)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  // We want to force every GetChanges to update the cdc_state table.
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;

  ASSERT_OK(SetUpWithParams(1, 1, false));
  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));

  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));
  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  // Insert some records in transaction.
  ASSERT_OK(WriteRowsHelper(0 /* start */, 100 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, 100));

  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  ASSERT_GE(record_size, 100);
  LOG(INFO) << "Total records read by GetChanges call on stream_id_1: " << record_size;

  // Forcefully update the checkpoint of the stream as MAX.
  auto max_commit_op_id = OpId::Max();
  CDCStateTable cdc_state_table(test_client());
  CDCStateTableEntry entry(tablets[0].tablet_id(), stream_id);
  entry.checkpoint = max_commit_op_id;
  ASSERT_OK(cdc_state_table.UpdateEntries({entry}));

  // Now Read the cdc_state table check checkpoint is updated to MAX.

  auto entry_opt = ASSERT_RESULT(
      cdc_state_table.TryFetchEntry(entry.key, CDCStateTableEntrySelector().IncludeCheckpoint()));
  ASSERT_TRUE(entry_opt.has_value()) << "Row not found in cdc_state table";
  ASSERT_EQ(*entry_opt->checkpoint, max_commit_op_id);

  VerifyCdcStateMatches(
      test_client(), stream_id, tablets[0].tablet_id(), max_commit_op_id.term,
      max_commit_op_id.index);

  xrepl::StreamId stream_id_2 = ASSERT_RESULT(CreateDBStream(IMPLICIT));
  resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id_2, tablets));
  ASSERT_FALSE(resp.has_error());
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCDCSDKCacheWithLeaderChange)) {
  // Disable lb as we move tablets around
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_load_balancing) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_intent_retention_ms) = 10000;
  // ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_intent_retention_ms) = 1000;
  const int num_tservers = 3;
  ASSERT_OK(SetUpWithParams(num_tservers, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  EnableCDCServiceInAllTserver(3);
  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));

  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));
  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  // Insert some records in transaction.
  ASSERT_OK(WriteRows(0 /* start */, 100 /* end */, &test_cluster_));

  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, 100));
  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  ASSERT_GE(record_size, 100);
  LOG(INFO) << "Total records read by GetChanges call on stream_id_1: " << record_size;

  int cache_hit_tservers =
      FindTserversWithCacheHit(stream_id, tablets[0].tablet_id(), num_tservers);
  ASSERT_GE(cache_hit_tservers, 1);

  // change LEADER of the tablet to tserver-2
  ASSERT_OK(ChangeLeaderOfTablet(1, tablets[0].tablet_id()));

  // check the condition of cache after LEADER step down.
  // we will see prev as well as current LEADER cache, search stream exist.
  cache_hit_tservers = FindTserversWithCacheHit(stream_id, tablets[0].tablet_id(), num_tservers);
  ASSERT_GE(cache_hit_tservers, 1);

  // Keep refreshing the stream from the new LEADER, till we cross the
  // FLAGS_cdc_intent_retention_ms.
  int idx = 0;
  while (idx < 10) {
    auto result =
        ASSERT_RESULT(GetChangesFromCDC(stream_id, tablets, &change_resp.cdc_sdk_checkpoint()));
    idx += 1;
    SleepFor(MonoDelta::FromMilliseconds(100));
  }

  // change LEADER of the tablet to tserver-1
  ASSERT_OK(ChangeLeaderOfTablet(0, tablets[0].tablet_id()));

  auto result = GetChangesFromCDC(stream_id, tablets, &change_resp.cdc_sdk_checkpoint());
  ASSERT_OK(result);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCDCSDKCacheWithLeaderReElect)) {
  // Disable lb as we move tablets around
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_load_balancing) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_metrics_interval_ms) = 1000;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;
  const int num_tservers = 3;
  ASSERT_OK(SetUpWithParams(num_tservers, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));

  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));

  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  // Insert some records in transaction.
  ASSERT_OK(WriteRowsHelper(0 /* start */, 100 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, 100));
  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  ASSERT_GE(record_size, 100);
  LOG(INFO) << "Total records read by GetChanges call on stream_id_1: " << record_size;
  SleepFor(MonoDelta::FromSeconds(1));
  size_t first_leader_index = 0;
  size_t first_follower_index = 0;
  GetTabletLeaderAndAnyFollowerIndex(tablets, &first_leader_index, &first_follower_index);

  GetChangesResponsePB change_resp_1 = ASSERT_RESULT(GetChangesFromCDC(stream_id, tablets));
  LOG(INFO) << "Number of records after first transaction: " << change_resp_1.records().size();

  ASSERT_OK(ChangeLeaderOfTablet(first_follower_index, tablets[0].tablet_id()));

  size_t second_leader_index = -1;
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets2;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets2, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  for (auto replica : tablets2[0].replicas()) {
    if (replica.role() == PeerRole::LEADER) {
      for (size_t i = 0; i < test_cluster()->num_tablet_servers(); i++) {
        if (test_cluster()->mini_tablet_server(i)->server()->permanent_uuid() ==
            replica.ts_info().permanent_uuid()) {
          second_leader_index = i;
          LOG(INFO) << "Found second leader index: " << i;
          break;
        }
      }
    }
  }

  // Insert some records in transaction after first leader stepdown.
  ASSERT_OK(WriteRowsHelper(100 /* start */, 200 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  // Call GetChanges so that the last active time is updated on the new leader.
  auto result = GetChangesFromCDC(stream_id, tablets2, &change_resp.cdc_sdk_checkpoint());

  SleepFor(MonoDelta::FromSeconds(2));
  CoarseTimePoint correct_expiry_time;
  for (auto const& peer : test_cluster()->GetTabletPeers(second_leader_index)) {
    if (peer->tablet_id() == tablets2[0].tablet_id()) {
      correct_expiry_time = peer->cdc_sdk_min_checkpoint_op_id_expiration();
      break;
    }
  }
  LOG(INFO) << "The correct expiry time after the final GetChanges call: "
            << correct_expiry_time.time_since_epoch().count();

  // we need to ensure the initial leader get's back leadership
  ASSERT_OK(ChangeLeaderOfTablet(first_follower_index, tablets[0].tablet_id()));
  LOG(INFO) << "Changed leadership back to the first leader TServer";

  // Call the test RPC to get last active time of the current leader (original), and it should
  // be lower than the previously recorded last_active_time.
  CompareExpirationTime(tablets2[0].tablet_id(), correct_expiry_time, first_leader_index);
  LOG(INFO) << "Succesfully compared expiry times";
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCDCSDKCacheWithLeaderRestart)) {
  // Disable lb as we move tablets around
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_load_balancing) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;
  const int num_tservers = 3;
  ASSERT_OK(SetUpWithParams(num_tservers, 1, false));

  // RF: 3, num of tservers: 4.
  for (int i = 0; i < 1; ++i) {
    ASSERT_OK(test_cluster()->AddTabletServer());
    ASSERT_OK(test_cluster()->WaitForAllTabletServers());
    LOG(INFO) << "Added new TServer to test cluster";
  }

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));

  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));
  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  size_t first_leader_index = 0;
  size_t first_follower_index = 0;
  GetTabletLeaderAndAnyFollowerIndex(tablets, &first_leader_index, &first_follower_index);

  // Insert some records in transaction.
  ASSERT_OK(WriteRowsHelper(0 /* start */, 100 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, 100));
  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  ASSERT_GE(record_size, 100);
  LOG(INFO) << "Total records read by GetChanges call on stream_id_1: " << record_size;
  SleepFor(MonoDelta::FromSeconds(10));

  ASSERT_OK(ChangeLeaderOfTablet(first_follower_index, tablets[0].tablet_id()));

  // Shutdown tserver hosting tablet leader.
  test_cluster()->mini_tablet_server(first_leader_index)->Shutdown();
  LOG(INFO) << "TServer hosting tablet leader shutdown";
  SleepFor(MonoDelta::FromSeconds(10));

  size_t second_leader_index = -1;
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets2;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets2, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  for (auto replica : tablets2[0].replicas()) {
    if (replica.role() == PeerRole::LEADER) {
      for (size_t i = 0; i < test_cluster()->num_tablet_servers(); i++) {
        if (i == first_leader_index) continue;
        if (test_cluster()->mini_tablet_server(i)->server()->permanent_uuid() ==
            replica.ts_info().permanent_uuid()) {
          second_leader_index = i;
          LOG(INFO) << "Found second leader index: " << i;
          break;
        }
      }
    }
    if (replica.role() == PeerRole::FOLLOWER) {
      for (size_t i = 0; i < test_cluster()->num_tablet_servers(); i++) {
        if (i == first_leader_index) continue;
        if (test_cluster()->mini_tablet_server(i)->server()->permanent_uuid() ==
            replica.ts_info().permanent_uuid()) {
          LOG(INFO) << "Found second follower index: " << i;
          break;
        }
      }
    }
  }

  // restart the initial leader tserver
  ASSERT_OK(test_cluster()->mini_tablet_server(first_leader_index)->Start());

  // Insert some records in transaction after leader shutdown.
  ASSERT_OK(WriteRowsHelper(100 /* start */, 200 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  // Call GetChanges so that the last active time is updated on the new leader.
  GetChangesResponsePB prev_change_resp = change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(
      &change_resp, stream_id, tablets2, 100, /* is_explicit_checkpoint */false,
      &prev_change_resp.cdc_sdk_checkpoint()));

  record_size = change_resp.cdc_sdk_proto_records_size();
  ASSERT_GE(record_size, 100);

  SleepFor(MonoDelta::FromSeconds(2));
  CoarseTimePoint correct_expiry_time;
  for (auto const& peer : test_cluster()->GetTabletPeers(second_leader_index)) {
    if (peer->tablet_id() == tablets2[0].tablet_id()) {
      correct_expiry_time = peer->cdc_sdk_min_checkpoint_op_id_expiration();
    }
  }
  LOG(INFO) << "CDKSDK checkpoint expiration time with LEADER tserver:" << second_leader_index
            << " : " << correct_expiry_time.time_since_epoch().count();

  // We need to ensure the initial leader get's back leadership.
  ASSERT_OK(ChangeLeaderOfTablet(first_leader_index, tablets[0].tablet_id()));

  ASSERT_OK(WriteRowsHelper(200 /* start */, 300 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  // Call GetChanges so that the last active time is updated on the new leader.
  prev_change_resp = change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(
      &change_resp, stream_id, tablets2, 100, /* is_explicit_checkpoint */false,
      &prev_change_resp.cdc_sdk_checkpoint()));
  record_size = change_resp.cdc_sdk_proto_records_size();
  ASSERT_GE(record_size, 100);

  // Call the test RPC to get last active time of the current leader (original), and it will
  // be lower than the previously recorded last_active_time.
  CompareExpirationTime(tablets2[0].tablet_id(), correct_expiry_time, first_leader_index);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCDCSDKActiveTimeCacheInSyncWithCDCStateTable)) {
  // Disable lb as we move tablets around
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_load_balancing) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_metrics_interval_ms) = 1000;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;
  const int num_tservers = 3;
  ASSERT_OK(SetUpWithParams(num_tservers, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));

  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));

  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  // Insert some records in transaction.
  ASSERT_OK(WriteRowsHelper(0 /* start */, 100 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, 100));
  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  ASSERT_GE(record_size, 100);
  LOG(INFO) << "Total records read by GetChanges call on stream_id_1: " << record_size;

  size_t first_leader_index = -1;
  size_t first_follower_index = -1;
  GetTabletLeaderAndAnyFollowerIndex(tablets, &first_leader_index, &first_follower_index);

  GetChangesResponsePB change_resp_1 =
      ASSERT_RESULT(GetChangesFromCDC(stream_id, tablets, &change_resp.cdc_sdk_checkpoint()));
  LOG(INFO) << "Number of records after first transaction: " << change_resp_1.records().size();

  const auto& first_leader_tserver =
      test_cluster()->mini_tablet_server(first_leader_index)->server();
  auto cdc_service = dynamic_cast<CDCServiceImpl*>(first_leader_tserver->rpc_server()
                                                       ->TEST_service_pool("yb.cdc.CDCService")
                                                       ->TEST_get_service()
                                                       .get());
  auto tablet_info = ASSERT_RESULT(
      cdc_service->TEST_GetTabletInfoFromCache({{}, stream_id, tablets[0].tablet_id()}));
  auto first_last_active_time = tablet_info.last_active_time;
  auto last_active_time_from_table = ASSERT_RESULT(
      GetLastActiveTimeFromCdcStateTable(stream_id, tablets[0].tablet_id(), test_client()));
  // Now check the active time in CDCSTate table, it should be greater than or equal to the
  // last_active_time from the cache.
  ASSERT_GE(last_active_time_from_table, first_last_active_time);
  LOG(INFO) << "The active time is equal in both the cache and cdc_state table";

  const size_t& second_leader_index = first_follower_index;
  ASSERT_OK(ChangeLeaderOfTablet(second_leader_index, tablets[0].tablet_id()));

  // Insert some records in transaction after first leader stepdown.
  ASSERT_OK(WriteRowsHelper(100 /* start */, 200 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  // Call GetChanges so that the last active time is updated on the new leader.
  auto result = GetChangesFromCDC(stream_id, tablets, &change_resp.cdc_sdk_checkpoint());

  const auto& second_leader_tserver =
      test_cluster()->mini_tablet_server(second_leader_index)->server();
  cdc_service = dynamic_cast<CDCServiceImpl*>(second_leader_tserver->rpc_server()
                                                  ->TEST_service_pool("yb.cdc.CDCService")
                                                  ->TEST_get_service()
                                                  .get());
  tablet_info = ASSERT_RESULT(
      cdc_service->TEST_GetTabletInfoFromCache({{}, stream_id, tablets[0].tablet_id()}));
  auto second_last_active_time = tablet_info.last_active_time;

  last_active_time_from_table = ASSERT_RESULT(
      GetLastActiveTimeFromCdcStateTable(stream_id, tablets[0].tablet_id(), test_client()));
  ASSERT_GE(last_active_time_from_table, second_last_active_time);
  LOG(INFO) << "The active time is equal in both the cache and cdc_state table";
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCDCSDKCacheWhenAFollowerIsUnavailable)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_metrics_interval_ms) = 500;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;
  const int num_tservers = 5;
  ASSERT_OK(SetUpWithParams(num_tservers, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));

  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));
  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  size_t first_leader_index = 0;
  size_t first_follower_index = 0;
  GetTabletLeaderAndAnyFollowerIndex(tablets, &first_leader_index, &first_follower_index);

  SleepFor(MonoDelta::FromSeconds(2));

  // Insert some records in transaction.
  ASSERT_OK(WriteRowsHelper(0 /* start */, 100 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, 100));
  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  ASSERT_GE(record_size, 100);
  LOG(INFO) << "Total records read by GetChanges call on stream_id_1: " << record_size;
  SleepFor(MonoDelta::FromSeconds(10));

  // Insert some records in transaction after leader shutdown.
  ASSERT_OK(WriteRowsHelper(100 /* start */, 200 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  auto result = GetChangesFromCDC(stream_id, tablets, &change_resp.cdc_sdk_checkpoint());

  CoarseTimePoint first_expiry_time;
  for (auto const& peer : test_cluster()->GetTabletPeers(first_leader_index)) {
    if (peer->tablet_id() == tablets[0].tablet_id()) {
      first_expiry_time = peer->cdc_sdk_min_checkpoint_op_id_expiration();
    }
  }
  LOG(INFO) << "The expiry time after the first GetChanges call: "
            << first_expiry_time.time_since_epoch().count();

  // Shutdown tserver having tablet FOLLOWER.
  test_cluster()->mini_tablet_server(first_follower_index)->Shutdown();
  LOG(INFO) << "TServer hosting tablet follower shutdown";
  // Call GetChanges so that the last active time is updated on the new leader.
  result = GetChangesFromCDC(stream_id, tablets, &change_resp.cdc_sdk_checkpoint());

  // Call the test RPC to get last active time of the current leader (original), and it must
  // be greater than or equal to the previously recorded last_active_time.
  CompareExpirationTime(tablets[0].tablet_id(), first_expiry_time, first_leader_index, true);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestColocation)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_update_local_peer_min_index) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 1;
  ASSERT_OK(SetUpWithParams(3, 1, false));

  ASSERT_OK(CreateColocatedObjects(&test_cluster_));
  auto table = ASSERT_RESULT(GetTable(&test_cluster_, kNamespaceName, "test1"));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), 1);

  std::string table_id = table.table_id();
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));
  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  int insert_count = 30;
  ASSERT_OK(PopulateColocatedData(&test_cluster_, insert_count));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  // Call get changes.
  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, insert_count*2));

  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  ASSERT_GT(record_size, insert_count);

  int expected_key1 = 0;
  int expected_key2 = 0;
  int ddl_count = 0;
  std::unordered_set<string> ddl_tables;
  for (uint32_t i = 0; i < record_size; ++i) {
    const auto record = change_resp.cdc_sdk_proto_records(i);
    if (record.row_message().op() == RowMessage::INSERT) {
      if (record.row_message().table() == "test1") {
        ASSERT_EQ(expected_key1, record.row_message().new_tuple(0).datum_int32());
        expected_key1++;
      } else if (record.row_message().table() == "test2") {
        ASSERT_EQ(std::to_string(expected_key2), record.row_message().new_tuple(0).datum_string());
        expected_key2++;
      }
    } else if (record.row_message().op() == RowMessage::DDL) {
      ddl_tables.insert(record.row_message().table());
      ddl_count++;
    }
  }

  ASSERT_TRUE(ddl_tables.contains("test1"));
  ASSERT_TRUE(ddl_tables.contains("test2"));

  ASSERT_EQ(insert_count, expected_key1);
  ASSERT_EQ(insert_count, expected_key2);
  ASSERT_EQ(ddl_count, 2);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestIntentsInColocation)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_update_local_peer_min_index) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 1;
  ASSERT_OK(SetUpWithParams(3, 1, false));

  ASSERT_OK(CreateColocatedObjects(&test_cluster_));
  auto table = ASSERT_RESULT(GetTable(&test_cluster_, kNamespaceName, "test1"));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), 1);

  std::string table_id = table.table_id();
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));
  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  int insert_count = 30;
  ASSERT_OK(PopulateColocatedData(&test_cluster_, insert_count, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  // Call get changes.
  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, insert_count*2));
  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  ASSERT_GT(record_size, insert_count);

  int expected_key1 = 0;
  int expected_key2 = 0;
  for (uint32_t i = 0; i < record_size; ++i) {
    const auto record = change_resp.cdc_sdk_proto_records(i);
    LOG(INFO) << "Record found: " << record.ShortDebugString();
    if (record.row_message().op() == RowMessage::INSERT) {
      if (record.row_message().table() == "test1") {
        ASSERT_EQ(expected_key1, record.row_message().new_tuple(0).datum_int32());
        expected_key1++;
      } else if (record.row_message().table() == "test2") {
        ASSERT_EQ(std::to_string(expected_key2), record.row_message().new_tuple(0).datum_string());
        expected_key2++;
      }
    }
  }

  ASSERT_EQ(insert_count, expected_key1);
  ASSERT_EQ(insert_count, expected_key2);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCDCSDKLagMetrics)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_metrics_interval_ms) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ASSERT_OK(SetUpWithParams(1, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  vector<xrepl::StreamId> stream_ids;
  for (int idx = 0; idx < 2; idx++) {
    stream_ids.push_back(ASSERT_RESULT(CreateDBStream(IMPLICIT)));
  }

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_ids[0], tablets));
  ASSERT_FALSE(resp.has_error());

  const auto& tserver = test_cluster()->mini_tablet_server(0)->server();
  auto cdc_service = dynamic_cast<CDCServiceImpl*>(
      tserver->rpc_server()->TEST_service_pool("yb.cdc.CDCService")->TEST_get_service().get());

  ASSERT_OK(WaitFor(
      [&]() { return cdc_service->CDCEnabled(); }, MonoDelta::FromSeconds(30), "IsCDCEnabled"));
  ASSERT_OK(WaitFor(
      [&]() -> Result<bool> {
        auto metrics =
            std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
                {{}, stream_ids[0], tablets[0].tablet_id()}, nullptr, CDCSDK));
        return metrics->cdcsdk_sent_lag_micros->value() == 0;
      },
      MonoDelta::FromSeconds(10) * kTimeMultiplier, "Wait for Lag == 0"));
  // Insert test rows, one at a time so they have different hybrid times.
  ASSERT_OK(WriteRowsHelper(0, 1, &test_cluster_, true));
  ASSERT_OK(WriteRowsHelper(1, 2, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  // Call get changes.
  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_ids[0], tablets, 2));

  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  ASSERT_GT(record_size, 2);
  ASSERT_OK(WaitFor(
      [&]() -> Result<bool> {
        auto metrics =
            std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
                {{}, stream_ids[0], tablets[0].tablet_id()}, nullptr, CDCSDK));
        return metrics->cdcsdk_sent_lag_micros->value() > 0;
      },
      MonoDelta::FromSeconds(10) * kTimeMultiplier, "Wait for Lag > 0"));

  // Now, delete the CDC stream and check the metrics information for the tablet_id and stream_id
  // combination should be deleted from the cdc metrics map.
  ASSERT_EQ(DeleteCDCStream(stream_ids[0]), true);
  VerifyStreamDeletedFromCdcState(test_client(), stream_ids[0], tablets.Get(0).tablet_id());
  ASSERT_OK(WaitFor(
      [&]() -> Result<bool> {
        auto metrics =
            std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
                {{}, stream_ids[0], tablets[0].tablet_id()},
                /* tablet_peer */ nullptr, CDCSDK, CreateCDCMetricsEntity::kFalse));
        return metrics == nullptr;
      },
      MonoDelta::FromSeconds(10) * kTimeMultiplier, "Wait for tablet metrics entry remove."));
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCDCSDKLastSentTimeMetric)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_metrics_interval_ms) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ASSERT_OK(SetUpWithParams(1, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  auto stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  const auto& tserver = test_cluster()->mini_tablet_server(0)->server();
  auto cdc_service = dynamic_cast<CDCServiceImpl*>(
      tserver->rpc_server()->TEST_service_pool("yb.cdc.CDCService")->TEST_get_service().get());

  ASSERT_OK(WriteRowsHelper(0, 1, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, 1));


  auto metrics =
      std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
          {{}, stream_id, tablets[0].tablet_id()},
          /* tablet_peer */ nullptr, CDCSDK, CreateCDCMetricsEntity::kFalse));
  uint64_t last_sent_time = metrics->cdcsdk_last_sent_physicaltime->value();

  ASSERT_OK(WriteRowsHelper(1, 2, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  GetChangesResponsePB new_change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(
      &new_change_resp, stream_id, tablets, 1, /* is_explicit_checkpoint */false,
      &change_resp.cdc_sdk_checkpoint()));

  auto metrics_ =
      std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
          {{}, stream_id, tablets[0].tablet_id()},
          /* tablet_peer */ nullptr, CDCSDK, CreateCDCMetricsEntity::kFalse));

  ASSERT_TRUE(
      last_sent_time < metrics_->cdcsdk_last_sent_physicaltime->value() &&
      last_sent_time * 2 > metrics_->cdcsdk_last_sent_physicaltime->value());
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCDCSDKExpiryMetric)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_metrics_interval_ms) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ASSERT_OK(SetUpWithParams(1, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  auto stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  const auto& tserver = test_cluster()->mini_tablet_server(0)->server();
  auto cdc_service = dynamic_cast<CDCServiceImpl*>(
      tserver->rpc_server()->TEST_service_pool("yb.cdc.CDCService")->TEST_get_service().get());
  ASSERT_OK(WriteRowsHelper(1, 100, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  // Call get changes.
  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, 99));

  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  ASSERT_GT(record_size, 100);
  auto metrics =
      std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
          {{}, stream_id, tablets[0].tablet_id()},
          /* tablet_peer */ nullptr, CDCSDK, CreateCDCMetricsEntity::kFalse));
  uint64_t current_stream_expiry_time = metrics->cdcsdk_expiry_time_ms->value();
  LOG(INFO) << "stream expiry time in milli seconds after GetChanges call: "
            << current_stream_expiry_time;
  ASSERT_OK(WaitFor(
      [&]() -> Result<bool> {
        auto metrics =
            std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
                {{}, stream_id, tablets[0].tablet_id()}, nullptr, CDCSDK));
        return current_stream_expiry_time > metrics->cdcsdk_expiry_time_ms->value();
      },
      MonoDelta::FromSeconds(10) * kTimeMultiplier, "Wait for stream expiry time update."));
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCDCSDKTrafficSentMetric)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_metrics_interval_ms) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ASSERT_OK(SetUpWithParams(1, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  auto stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  const auto& tserver = test_cluster()->mini_tablet_server(0)->server();
  auto cdc_service = dynamic_cast<CDCServiceImpl*>(
      tserver->rpc_server()->TEST_service_pool("yb.cdc.CDCService")->TEST_get_service().get());
  ASSERT_OK(WriteRowsHelper(1, 100, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  // Call get changes.
  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, 99));

  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  ASSERT_GT(record_size, 100);
  auto metrics =
      std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
          {{}, stream_id, tablets[0].tablet_id()},
          /* tablet_peer */ nullptr, CDCSDK, CreateCDCMetricsEntity::kFalse));
  int64_t current_traffic_sent_bytes = metrics->cdcsdk_traffic_sent->value();

  // Isnert few more records
  ASSERT_OK(WriteRowsHelper(101, 200, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  // Call get changes.
  GetChangesResponsePB new_change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(
      &new_change_resp, stream_id, tablets, 99, /* is_explicit_checkpoint */false,
      &change_resp.cdc_sdk_checkpoint()));

  record_size = new_change_resp.cdc_sdk_proto_records_size();
  ASSERT_GT(record_size, 100);

  LOG(INFO) << "Traffic sent in bytes after GetChanges call: " << current_traffic_sent_bytes;
  ASSERT_OK(WaitFor(
      [&]() -> Result<bool> {
        auto metrics =
            std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
                {{}, stream_id, tablets[0].tablet_id()}, nullptr, CDCSDK));
        return current_traffic_sent_bytes < metrics->cdcsdk_traffic_sent->value();
      },
      MonoDelta::FromSeconds(10) * kTimeMultiplier,
      "Wait for CDCSDK traffic sent attribute update."));
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCDCSDKChangeEventCountMetric)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_metrics_interval_ms) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;
  ASSERT_OK(SetUpWithParams(1, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  auto stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  const auto& tserver = test_cluster()->mini_tablet_server(0)->server();
  auto cdc_service = dynamic_cast<CDCServiceImpl*>(
      tserver->rpc_server()->TEST_service_pool("yb.cdc.CDCService")->TEST_get_service().get());
  ASSERT_OK(WriteRowsHelper(1, 100, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  // Call get changes.
  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, 99));

  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  ASSERT_GT(record_size, 100);
  auto metrics =
      std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
          {{}, stream_id, tablets[0].tablet_id()},
          /* tablet_peer */ nullptr, CDCSDK, CreateCDCMetricsEntity::kFalse));
  LOG(INFO) << "Total event counts after GetChanges call: "
            << metrics->cdcsdk_change_event_count->value();
  ASSERT_GT(metrics->cdcsdk_change_event_count->value(), 100);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCDCSDKMetricsTwoTablesSingleStream)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_metrics_interval_ms) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ASSERT_OK(SetUpWithParams(1, 1, false));

  const uint32_t num_tablets = 1;
  const uint32_t num_tables = 2;
  vector<string> table_suffix = {"_1", "_2"};

  vector<YBTableName> table(num_tables);
  vector<google::protobuf::RepeatedPtrField<master::TabletLocationsPB>> tablets(num_tables);
  vector<TableId> table_id(num_tables);

  for (uint32_t idx = 0; idx < num_tables; idx++) {
    table[idx] = ASSERT_RESULT(
        CreateTable(&test_cluster_, kNamespaceName, kTableName + table_suffix[idx], num_tablets));

    ASSERT_OK(test_client()->GetTablets(
        table[idx], 0, &tablets[idx],
        /* partition_list_version =*/nullptr));
    ASSERT_EQ(tablets[idx].size(), num_tablets);

    table_id[idx] =
        ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName + table_suffix[idx]));
  }

  auto stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  for (auto tablet : tablets) {
    auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablet));
    ASSERT_FALSE(resp.has_error());
  }

  const auto& tserver = test_cluster()->mini_tablet_server(0)->server();
  auto cdc_service = dynamic_cast<CDCServiceImpl*>(
      tserver->rpc_server()->TEST_service_pool("yb.cdc.CDCService")->TEST_get_service().get());

  int64_t current_traffic_sent_bytes = 0;
  vector<GetChangesResponsePB> change_resp(num_tables);
  vector<std::shared_ptr<cdc::CDCSDKTabletMetrics>> metrics(num_tables);
  uint32_t total_record_size = 0;
  int64_t total_traffic_sent = 0;
  uint64_t total_change_event_count = 0;


  for (uint32_t idx = 0; idx < num_tables; idx++) {
    ASSERT_OK(
        WriteRowsHelper(1, 50, &test_cluster_, true, 2, (kTableName + table_suffix[idx]).c_str()));
    ASSERT_OK(test_client()->FlushTables(
        {table[idx].table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
        /* is_compaction = */ false));

    ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp[idx], stream_id, tablets[idx], 49));

    total_record_size += change_resp[idx].cdc_sdk_proto_records_size();

    metrics[idx] =
        std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
            {{}, stream_id, tablets[idx][0].tablet_id()},
            /* tablet_peer */ nullptr, CDCSDK, CreateCDCMetricsEntity::kFalse));
    total_traffic_sent += metrics[idx]->cdcsdk_traffic_sent->value();
    total_change_event_count += metrics[idx]->cdcsdk_change_event_count->value();

    auto current_expiry_time = metrics[idx]->cdcsdk_expiry_time_ms->value();
    ASSERT_OK(WaitFor(
        [&]() -> Result<bool> {
          auto metrics =
              std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
                  {{}, stream_id, tablets[idx][0].tablet_id()}, nullptr, CDCSDK));
          return current_expiry_time > metrics->cdcsdk_expiry_time_ms->value();
        },
        MonoDelta::FromSeconds(10) * kTimeMultiplier, "Wait for stream expiry time update."));
  }

  ASSERT_GT(total_record_size, 100);
  ASSERT_GT(total_change_event_count, 100);
  ASSERT_TRUE(current_traffic_sent_bytes < total_traffic_sent);
}

TEST_F(
    CDCSDKYsqlTest,
    YB_DISABLE_TEST_IN_TSAN(TestCDCSDKMetricsTwoTablesTwoStreamsOnIndividualTables)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_metrics_interval_ms) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;
  ASSERT_OK(SetUpWithParams(1, 1, false));

  const uint32_t num_tablets = 1;
  const uint32_t num_tables = 2;
  string underscore = "_";

  vector<YBTableName> table(num_tables);
  vector<google::protobuf::RepeatedPtrField<master::TabletLocationsPB>> tablets(num_tables);
  vector<TableId> table_id(num_tables);
  vector<xrepl::StreamId> stream_ids;

  for (uint32_t idx = 0; idx < num_tables; idx++) {
    table[idx] = ASSERT_RESULT(CreateTable(
        &test_cluster_, kNamespaceName, kTableName + underscore + std::to_string(idx),
        num_tablets));
    ASSERT_OK(test_client()->GetTablets(
        table[idx], 0, &tablets[idx], /* partition_list_version =*/nullptr));
    ASSERT_EQ(tablets[idx].size(), num_tablets);

    table_id[idx] = ASSERT_RESULT(
        GetTableId(&test_cluster_, kNamespaceName, kTableName + underscore + std::to_string(idx)));

    auto stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));
    stream_ids.push_back(stream_id);
    auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets[idx]));
    ASSERT_FALSE(resp.has_error());
  }
  const auto& tserver = test_cluster()->mini_tablet_server(0)->server();
  auto cdc_service = dynamic_cast<CDCServiceImpl*>(
      tserver->rpc_server()->TEST_service_pool("yb.cdc.CDCService")->TEST_get_service().get());

  for (uint32_t idx = 0; idx < num_tables; idx++) {
    int64_t current_traffic_sent_bytes = 0;
    ASSERT_OK(WriteRowsHelper(
        1, 100, &test_cluster_, true, 2, (kTableName + underscore + std::to_string(idx)).c_str()));
    ASSERT_OK(test_client()->FlushTables(
        {table[idx].table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
        /* is_compaction = */ false));
    GetChangesResponsePB change_resp;
    ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_ids[idx], tablets[idx], 99));

    uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
    ASSERT_GT(record_size, 100);

    auto metrics =
        std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
            {{}, stream_ids[idx], tablets[idx][0].tablet_id()},
            /* tablet_peer */ nullptr, CDCSDK, CreateCDCMetricsEntity::kFalse));

    auto current_expiry_time = metrics->cdcsdk_expiry_time_ms->value();
    ASSERT_OK(WaitFor(
        [&]() -> Result<bool> {
          auto metrics =
              std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
                  {{}, stream_ids[idx], tablets[idx][0].tablet_id()}, nullptr, CDCSDK));
          return current_expiry_time > metrics->cdcsdk_expiry_time_ms->value();
        },
        MonoDelta::FromSeconds(10) * kTimeMultiplier, "Wait for stream expiry time update."));

    ASSERT_GT(metrics->cdcsdk_change_event_count->value(), 100);
    ASSERT_TRUE(current_traffic_sent_bytes < metrics->cdcsdk_traffic_sent->value());
  }
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCDCSDKMetricsTwoTablesTwoStreamsOnBothTables)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_metrics_interval_ms) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;
  ASSERT_OK(SetUpWithParams(1, 1, false));

  const uint32_t num_tablets = 1;
  const uint32_t num_tables = 2;
  const uint32_t num_streams = 2;
  string underscore = "_";

  vector<YBTableName> table(num_tables);
  vector<google::protobuf::RepeatedPtrField<master::TabletLocationsPB>> tablets(num_tables);
  vector<TableId> table_id(num_tables);
  vector<xrepl::StreamId> stream_ids;

  for (uint32_t idx = 0; idx < num_tables; idx++) {
    table[idx] = ASSERT_RESULT(CreateTable(
        &test_cluster_, kNamespaceName, kTableName + underscore + std::to_string(idx),
        num_tablets));
    ASSERT_OK(test_client()->GetTablets(
        table[idx], 0, &tablets[idx], /* partition_list_version =*/nullptr));
    ASSERT_EQ(tablets[idx].size(), num_tablets);

    table_id[idx] = ASSERT_RESULT(
        GetTableId(&test_cluster_, kNamespaceName, kTableName + underscore + std::to_string(idx)));
  }

  for (uint32_t idx = 0; idx < num_streams; idx++) {
    auto stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));
    stream_ids.push_back(stream_id);
    for (auto tablet : tablets) {
      auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablet));
      ASSERT_FALSE(resp.has_error());
    }
  }
  const auto& tserver = test_cluster()->mini_tablet_server(0)->server();
  auto cdc_service = dynamic_cast<CDCServiceImpl*>(
      tserver->rpc_server()->TEST_service_pool("yb.cdc.CDCService")->TEST_get_service().get());

  for (uint32_t idx = 0; idx < num_tables; idx++) {
    int64_t current_traffic_sent_bytes = 0;
    ASSERT_OK(WriteRowsHelper(
        1, 100, &test_cluster_, true, 2, (kTableName + underscore + std::to_string(idx)).c_str()));
    ASSERT_OK(test_client()->FlushTables(
        {table[idx].table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
        /* is_compaction = */ false));

    for (uint32_t stream_idx = 0; stream_idx < num_streams; stream_idx++) {
      GetChangesResponsePB change_resp;
      ASSERT_OK(
          WaitForGetChangesToFetchRecords(&change_resp, stream_ids[stream_idx], tablets[idx], 99));

      uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
      ASSERT_GT(record_size, 100);

      auto metrics =
          std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
              {{}, stream_ids[stream_idx], tablets[idx][0].tablet_id()},
              /* tablet_peer */ nullptr, CDCSDK, CreateCDCMetricsEntity::kFalse));
      auto current_expiry_time = metrics->cdcsdk_expiry_time_ms->value();
      ASSERT_OK(WaitFor(
          [&]() -> Result<bool> {
            auto metrics =
                std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
                    {{}, stream_ids[idx], tablets[idx][0].tablet_id()}, nullptr, CDCSDK));
            return current_expiry_time > metrics->cdcsdk_expiry_time_ms->value();
          },
          MonoDelta::FromSeconds(10) * kTimeMultiplier, "Wait for stream expiry time update."));
      ASSERT_GT(metrics->cdcsdk_change_event_count->value(), 100);
      ASSERT_TRUE(current_traffic_sent_bytes < metrics->cdcsdk_traffic_sent->value());
    }
  }
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCDCSDKMetricsWithAddStream)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_metrics_interval_ms) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;
  ASSERT_OK(SetUpWithParams(1, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  auto stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  const auto& tserver = test_cluster()->mini_tablet_server(0)->server();
  auto cdc_service = dynamic_cast<CDCServiceImpl*>(
      tserver->rpc_server()->TEST_service_pool("yb.cdc.CDCService")->TEST_get_service().get());

  int64_t current_traffic_sent_bytes = 0;

  ASSERT_OK(WriteRowsHelper(1, 100, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  auto change_resp = GetAllPendingChangesFromCdc(stream_id, tablets);
  size_t record_size = change_resp.records.size();
  ASSERT_GT(record_size, 100);

  auto metrics =
      std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
          {{}, stream_id, tablets[0].tablet_id()},
          /* tablet_peer */ nullptr, CDCSDK, CreateCDCMetricsEntity::kFalse));

  auto current_expiry_time = metrics->cdcsdk_expiry_time_ms->value();
  ASSERT_OK(WaitFor(
      [&]() -> Result<bool> {
        auto metrics =
            std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
                {{}, stream_id, tablets[0].tablet_id()}, nullptr, CDCSDK));
        return current_expiry_time > metrics->cdcsdk_expiry_time_ms->value();
      },
      MonoDelta::FromSeconds(10) * kTimeMultiplier, "Wait for stream expiry time update."));

  ASSERT_GT(metrics->cdcsdk_change_event_count->value(), 100);
  ASSERT_TRUE(current_traffic_sent_bytes < metrics->cdcsdk_traffic_sent->value());

  // Create a new stream
  auto
  new_stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto new_resp = ASSERT_RESULT(SetCDCCheckpoint(new_stream_id, tablets));
  ASSERT_FALSE(new_resp.has_error());

  current_traffic_sent_bytes = metrics->cdcsdk_traffic_sent->value();

  ASSERT_OK(WriteRowsHelper(101, 200, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  auto new_change_resp = GetAllPendingChangesFromCdc(new_stream_id, tablets);
  record_size = new_change_resp.records.size();
  ASSERT_GT(record_size, 100);

  auto new_metrics =
      std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
          {{}, new_stream_id, tablets[0].tablet_id()},
          /* tablet_peer */ nullptr, CDCSDK, CreateCDCMetricsEntity::kFalse));

  current_expiry_time = new_metrics->cdcsdk_expiry_time_ms->value();
  ASSERT_OK(WaitFor(
      [&]() -> Result<bool> {
        auto metrics =
            std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
                {{}, new_stream_id, tablets[0].tablet_id()}, nullptr, CDCSDK));
        return current_expiry_time > metrics->cdcsdk_expiry_time_ms->value();
      },
      MonoDelta::FromSeconds(10) * kTimeMultiplier, "Wait for stream expiry time update."));
  ASSERT_GT(new_metrics->cdcsdk_change_event_count->value(), 100);
  ASSERT_TRUE(current_traffic_sent_bytes < new_metrics->cdcsdk_traffic_sent->value());
}

TEST_F(
    CDCSDKYsqlTest,
    YB_DISABLE_TEST_IN_TSAN(TestCDCSDKAddColumnsWithImplictTransactionWithoutPackedRow)) {
  CDCSDKAddColumnsWithImplictTransaction(false);
}

TEST_F(
    CDCSDKYsqlTest,
    YB_DISABLE_TEST_IN_TSAN(TestCDCSDKAddColumnsWithImplictTransactionWithPackedRow)) {
  CDCSDKAddColumnsWithImplictTransaction(true);
}

TEST_F(
    CDCSDKYsqlTest,
    YB_DISABLE_TEST_IN_TSAN(TestCDCSDKAddColumnsWithExplictTransactionWithOutPackedRow)) {
  CDCSDKAddColumnsWithExplictTransaction(false);
}

TEST_F(
    CDCSDKYsqlTest,
    YB_DISABLE_TEST_IN_TSAN(TestCDCSDKAddColumnsWithExplictTransactionWithPackedRow)) {
  CDCSDKAddColumnsWithExplictTransaction(true);
}

TEST_F(
    CDCSDKYsqlTest,
    YB_DISABLE_TEST_IN_TSAN(TestCDCSDKDropColumnsWithRestartTServerWithOutPackedRow)) {
  CDCSDKDropColumnsWithRestartTServer(false);
}

TEST_F(
    CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCDCSDKDropColumnsWithRestartTServerWithPackedRow)) {
  CDCSDKDropColumnsWithRestartTServer(true);
}

TEST_F(
    CDCSDKYsqlTest,
    YB_DISABLE_TEST_IN_TSAN(TestCDCSDKDropColumnsWithImplictTransactionWithOutPackedRow)) {
  CDCSDKDropColumnsWithImplictTransaction(false);
}

TEST_F(
    CDCSDKYsqlTest,
    YB_DISABLE_TEST_IN_TSAN(TestCDCSDKDropColumnsWithImplictTransactionWithPackedRow)) {
  CDCSDKDropColumnsWithImplictTransaction(true);
}

TEST_F(
    CDCSDKYsqlTest,
    YB_DISABLE_TEST_IN_TSAN(TestCDCSDKDropColumnsWithExplictTransactionWithOutPackedRow)) {
  CDCSDKDropColumnsWithExplictTransaction(false);
}

TEST_F(
    CDCSDKYsqlTest,
    YB_DISABLE_TEST_IN_TSAN(TestCDCSDKDropColumnsWithExplictTransactionWithPackedRow)) {
  CDCSDKDropColumnsWithExplictTransaction(true);
}

TEST_F(
    CDCSDKYsqlTest,
    YB_DISABLE_TEST_IN_TSAN(TestCDCSDKRenameColumnsWithImplictTransactionWithOutPackedRow)) {
  CDCSDKRenameColumnsWithImplictTransaction(false);
}

TEST_F(
    CDCSDKYsqlTest,
    YB_DISABLE_TEST_IN_TSAN(TestCDCSDKRenameColumnsWithImplictTransactionWithPackedRow)) {
  CDCSDKRenameColumnsWithImplictTransaction(true);
}

TEST_F(
    CDCSDKYsqlTest,
    YB_DISABLE_TEST_IN_TSAN(TestCDCSDKRenameColumnsWithExplictTransactionWithOutPackedRow)) {
  CDCSDKRenameColumnsWithExplictTransaction(false);
}

TEST_F(
    CDCSDKYsqlTest,
    YB_DISABLE_TEST_IN_TSAN(TestCDCSDKRenameColumnsWithExplictTransactionWithPackedRow)) {
  CDCSDKRenameColumnsWithExplictTransaction(true);
}

TEST_F(
    CDCSDKYsqlTest,
    YB_DISABLE_TEST_IN_TSAN(TestCDCSDKMultipleAlterWithRestartTServerWithOutPackedRow)) {
  CDCSDKMultipleAlterWithRestartTServer(false);
}

TEST_F(
    CDCSDKYsqlTest,
    YB_DISABLE_TEST_IN_TSAN(TestCDCSDKMultipleAlterWithRestartTServerWithPackedRow)) {
  CDCSDKMultipleAlterWithRestartTServer(true);
}

TEST_F(
    CDCSDKYsqlTest,
    YB_DISABLE_TEST_IN_TSAN(TestCDCSDKMultipleAlterWithTabletLeaderSwitchWithOutPackedRow)) {
  CDCSDKMultipleAlterWithTabletLeaderSwitch(false);
}
TEST_F(
    CDCSDKYsqlTest,
    YB_DISABLE_TEST_IN_TSAN(TestCDCSDKMultipleAlterWithTabletLeaderSwitchWithPackedRow)) {
  CDCSDKMultipleAlterWithTabletLeaderSwitch(true);
}

TEST_F(
    CDCSDKYsqlTest,
    YB_DISABLE_TEST_IN_TSAN(TestCDCSDKAlterWithSysCatalogCompactionWithOutPackedRow)) {
  CDCSDKAlterWithSysCatalogCompaction(false);
}

TEST_F(
    CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCDCSDKAlterWithSysCatalogCompactionWithPackedRow)) {
  CDCSDKAlterWithSysCatalogCompaction(true);
}

TEST_F(
    CDCSDKYsqlTest,
    YB_DISABLE_TEST_IN_TSAN(
        TestCDCSDKIntentsBatchReadWithAlterAndTabletLeaderSwitchWithOutPackedRow)) {
  CDCSDKIntentsBatchReadWithAlterAndTabletLeaderSwitch(false);
}

TEST_F(
    CDCSDKYsqlTest,
    YB_DISABLE_TEST_IN_TSAN(
        TestCDCSDKIntentsBatchReadWithAlterAndTabletLeaderSwitchWithPackedRow)) {
  CDCSDKIntentsBatchReadWithAlterAndTabletLeaderSwitch(true);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestAddTableToNamespaceWithActiveStream)) {
  ASSERT_OK(SetUpWithParams(1, 1, false));

  const uint32_t num_tablets = 3;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  std::vector<TableId> expected_table_ids;
  expected_table_ids.reserve(2);
  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  expected_table_ids.push_back(table_id);
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  std::unordered_set<TabletId> expected_tablet_ids;
  for (const auto& tablet : tablets) {
    expected_tablet_ids.insert(tablet.tablet_id());
  }
  ASSERT_EQ(expected_tablet_ids.size(), num_tablets);
  CheckTabletsInCDCStateTable(expected_tablet_ids, test_client());

  auto table_2 =
      ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, "test_table_1", num_tablets));
  TableId table_2_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, "test_table_1"));
  expected_table_ids.push_back(table_2_id);
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets_2;
  ASSERT_OK(
      test_client()->GetTablets(table_2, 0, &tablets_2, /* partition_list_version =*/nullptr));
  for (const auto& tablet : tablets_2) {
    expected_tablet_ids.insert(tablet.tablet_id());
  }
  ASSERT_EQ(expected_tablet_ids.size(), num_tablets * 2);

  CheckTabletsInCDCStateTable(expected_tablet_ids, test_client());

  ASSERT_EQ(ASSERT_RESULT(GetCDCStreamTableIds(stream_id)), expected_table_ids);

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets_2));
  ASSERT_FALSE(resp.has_error());
  ASSERT_RESULT(GetChangesFromCDC(stream_id, tablets_2));
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestAdd100TableToNamespaceWithActiveStream)) {
  ASSERT_OK(SetUpWithParams(1, 1, true));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  const int num_new_tables = 100;
  auto conn = ASSERT_RESULT(test_cluster_.ConnectToDB(kNamespaceName));
  for (int i = 1; i <= num_new_tables; i++) {
    std::string table_name = "test_table_" + std::to_string(i);
    ASSERT_OK(conn.ExecuteFormat("CREATE TABLE $0(key int PRIMARY KEY, value_1 int);", table_name));
  }

  ASSERT_OK(WaitFor(
      [&]() -> Result<bool> {
        auto result = GetCDCStreamTableIds(stream_id);
        if (!result.ok()) return false;

        return (result.get().size() == num_new_tables + 1);
      },
      MonoDelta::FromSeconds(180),
      "Could not find all the added table's in the stream's metadata"));
}

TEST_F(
    CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestAddTableToNamespaceWithActiveStreamMasterRestart)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_catalog_manager_bg_task_wait_ms) = 60 * 1000;
  ASSERT_OK(SetUpWithParams(1, 1, false));

  const uint32_t num_tablets = 3;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  std::vector<TableId> expected_table_ids;
  expected_table_ids.reserve(2);
  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  expected_table_ids.push_back(table_id);
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  std::unordered_set<TabletId> expected_tablet_ids;
  for (const auto& tablet : tablets) {
    expected_tablet_ids.insert(tablet.tablet_id());
  }
  ASSERT_EQ(expected_tablet_ids.size(), num_tablets);
  CheckTabletsInCDCStateTable(expected_tablet_ids, test_client());
  LOG(INFO) << "Verified tablets of first table exist in cdc_state table";

  auto table_2 =
      ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, "test_table_1", num_tablets));
  TableId table_2_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, "test_table_1"));
  expected_table_ids.push_back(table_2_id);
  LOG(INFO) << "Created second table";

  test_cluster_.mini_cluster_->mini_master()->Shutdown();
  ASSERT_OK(test_cluster_.mini_cluster_->StartMasters());
  LOG(INFO) << "Restarted Master";

  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets_2;
  ASSERT_OK(
      test_client()->GetTablets(table_2, 0, &tablets_2, /* partition_list_version =*/nullptr));
  for (const auto& tablet : tablets_2) {
    expected_tablet_ids.insert(tablet.tablet_id());
  }
  ASSERT_EQ(expected_tablet_ids.size(), num_tablets * 2);

  CheckTabletsInCDCStateTable(expected_tablet_ids, test_client());
  LOG(INFO) << "Verified the number of tablets in the cdc_state table";

  test_cluster_.mini_cluster_->mini_master()->Shutdown();
  ASSERT_OK(test_cluster_.mini_cluster_->StartMasters());
  LOG(INFO) << "Restarted Master";

  ASSERT_EQ(ASSERT_RESULT(GetCDCStreamTableIds(stream_id)), expected_table_ids);

  auto get_tablets_resp = ASSERT_RESULT(GetTabletListToPollForCDC(stream_id, table_2_id));
  for (const auto& tablet_checkpoint_pair : get_tablets_resp.tablet_checkpoint_pairs()) {
    const auto& tablet_id = tablet_checkpoint_pair.tablet_locations().tablet_id();
    ASSERT_TRUE(expected_tablet_ids.contains(tablet_id));
  }
  ASSERT_EQ(get_tablets_resp.tablet_checkpoint_pairs_size(), 3);

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets_2));
  ASSERT_FALSE(resp.has_error());
  ASSERT_RESULT(GetChangesFromCDC(stream_id, tablets_2));
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestAddColocatedTableToNamespaceWithActiveStream)) {
  ASSERT_OK(SetUpWithParams(1, 1, false));

  const uint32_t num_tablets = 1;

  ASSERT_OK(CreateColocatedObjects(&test_cluster_));
  auto table = ASSERT_RESULT(GetTable(&test_cluster_, kNamespaceName, "test1"));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  std::vector<TableId> expected_table_ids;
  expected_table_ids.reserve(3);
  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, "test1"));
  TableId table_id_2 = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, "test2"));
  expected_table_ids.push_back(table_id);
  expected_table_ids.push_back(table_id_2);

  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  std::unordered_set<TabletId> expected_tablet_ids;
  for (const auto& tablet : tablets) {
    expected_tablet_ids.insert(tablet.tablet_id());
  }
  ASSERT_EQ(expected_tablet_ids.size(), num_tablets);
  CheckTabletsInCDCStateTable(expected_tablet_ids, test_client());

  ASSERT_OK(AddColocatedTable(&test_cluster_, "test3"));
  auto table_3 = ASSERT_RESULT(GetTable(&test_cluster_, kNamespaceName, "test3"));
  TableId table_id_3 = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, "test3"));
  expected_table_ids.push_back(table_id_3);
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets_3;
  ASSERT_OK(
      test_client()->GetTablets(table_3, 0, &tablets_3, /* partition_list_version =*/nullptr));
  for (const auto& tablet : tablets_3) {
    expected_tablet_ids.insert(tablet.tablet_id());
  }
  // Since we added a new table to an existing table group, no new tablet details is expected.
  ASSERT_EQ(expected_tablet_ids.size(), num_tablets);
  CheckTabletsInCDCStateTable(expected_tablet_ids, test_client());

  // Wait for a background task cycle to complete.
  std::sort(expected_table_ids.begin(), expected_table_ids.end());
  auto result = WaitFor(
      [&]() -> Result<bool> {
        auto actual_table_ids = VERIFY_RESULT(GetCDCStreamTableIds(stream_id));
        std::sort(actual_table_ids.begin(), actual_table_ids.end());
        return actual_table_ids == expected_table_ids;
      },
      MonoDelta::FromSeconds(10) * kTimeMultiplier,
      "Waiting for background task to update cdc streams.");
  EXPECT_OK(result);
  // Extra ASSERT here to get nicely formatted debug information in case of failure.
  if (!result.ok()) {
    EXPECT_THAT(ASSERT_RESULT(GetCDCStreamTableIds(stream_id)),
        testing::UnorderedElementsAreArray(expected_table_ids));
  }
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestAddTableToNamespaceWithMultipleActiveStreams)) {
  ASSERT_OK(SetUpWithParams(1, 1, false));

  const uint32_t num_tablets = 3;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  std::vector<TableId> expected_table_ids;
  expected_table_ids.reserve(2);
  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  expected_table_ids.push_back(table_id);
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  std::unordered_set<TabletId> expected_tablet_ids;
  for (const auto& tablet : tablets) {
    expected_tablet_ids.insert(tablet.tablet_id());
  }
  ASSERT_EQ(expected_tablet_ids.size(), num_tablets);

  auto table_1 =
      ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, "test_table_1", num_tablets));
  TableId table_1_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, "test_table_1"));
  expected_table_ids.push_back(table_1_id);
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets_1;
  ASSERT_OK(
      test_client()->GetTablets(table_1, 0, &tablets_1, /* partition_list_version =*/nullptr));
  for (const auto& tablet : tablets_1) {
    expected_tablet_ids.insert(tablet.tablet_id());
  }
  ASSERT_EQ(expected_tablet_ids.size(), num_tablets * 2);

  xrepl::StreamId stream_id_1 = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto table_2 =
      ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, "test_table_2", num_tablets));
  TableId table_2_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, "test_table_2"));
  expected_table_ids.push_back(table_2_id);
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets_2;
  ASSERT_OK(
      test_client()->GetTablets(table_2, 0, &tablets_2, /* partition_list_version =*/nullptr));
  for (const auto& tablet : tablets_2) {
    expected_tablet_ids.insert(tablet.tablet_id());
  }
  ASSERT_EQ(expected_tablet_ids.size(), num_tablets * 3);

  // Check that 'cdc_state' table has all the expected tables for both streams.
  CheckTabletsInCDCStateTable(expected_tablet_ids, test_client(), stream_id);
  CheckTabletsInCDCStateTable(expected_tablet_ids, test_client(), stream_id_1);

  // Check that both the streams metadata has all the 3 table ids.
  ASSERT_THAT(
      ASSERT_RESULT(GetCDCStreamTableIds(stream_id)),
      testing::UnorderedElementsAreArray(expected_table_ids));
  ASSERT_THAT(
      ASSERT_RESULT(GetCDCStreamTableIds(stream_id_1)),
      testing::UnorderedElementsAreArray(expected_table_ids));
}

TEST_F(
    CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestAddTableWithMultipleActiveStreamsMasterRestart)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_catalog_manager_bg_task_wait_ms) = 60 * 1000;
  ASSERT_OK(SetUpWithParams(1, 1, false));

  const uint32_t num_tablets = 3;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  std::vector<TableId> expected_table_ids;
  expected_table_ids.reserve(2);
  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  expected_table_ids.push_back(table_id);
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  std::unordered_set<TabletId> expected_tablet_ids;
  for (const auto& tablet : tablets) {
    expected_tablet_ids.insert(tablet.tablet_id());
  }
  ASSERT_EQ(expected_tablet_ids.size(), num_tablets);

  auto table_1 =
      ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, "test_table_1", num_tablets));
  TableId table_1_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, "test_table_1"));
  expected_table_ids.push_back(table_1_id);
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets_1;
  ASSERT_OK(
      test_client()->GetTablets(table_1, 0, &tablets_1, /* partition_list_version =*/nullptr));
  for (const auto& tablet : tablets_1) {
    expected_tablet_ids.insert(tablet.tablet_id());
  }
  ASSERT_EQ(expected_tablet_ids.size(), num_tablets * 2);

  xrepl::StreamId stream_id_1 = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto table_2 =
      ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, "test_table_2", num_tablets));
  TableId table_2_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, "test_table_2"));
  expected_table_ids.push_back(table_2_id);

  xrepl::StreamId stream_id_2 = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  test_cluster_.mini_cluster_->mini_master()->Shutdown();
  ASSERT_OK(test_cluster_.mini_cluster_->StartMasters());
  LOG(INFO) << "Restarted Master";

  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets_2;
  ASSERT_OK(
      test_client()->GetTablets(table_2, 0, &tablets_2, /* partition_list_version =*/nullptr));
  for (const auto& tablet : tablets_2) {
    expected_tablet_ids.insert(tablet.tablet_id());
  }
  ASSERT_EQ(expected_tablet_ids.size(), num_tablets * 3);

  // Check that 'cdc_state' table has all the expected tables for both streams.
  CheckTabletsInCDCStateTable(expected_tablet_ids, test_client(), stream_id);
  CheckTabletsInCDCStateTable(expected_tablet_ids, test_client(), stream_id_1);
  CheckTabletsInCDCStateTable(expected_tablet_ids, test_client(), stream_id_2);

  // Check that both the streams metadata has all the 3 table ids.
  ASSERT_THAT(
      ASSERT_RESULT(GetCDCStreamTableIds(stream_id)),
      testing::UnorderedElementsAreArray(expected_table_ids));
  ASSERT_THAT(
      ASSERT_RESULT(GetCDCStreamTableIds(stream_id_1)),
      testing::UnorderedElementsAreArray(expected_table_ids));
  ASSERT_THAT(
      ASSERT_RESULT(GetCDCStreamTableIds(stream_id_2)),
      testing::UnorderedElementsAreArray(expected_table_ids));
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestAddMultipleTableToNamespaceWithActiveStream)) {
  // We set the limit of newly added tables per iteration to 1.
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdcsdk_table_processing_limit_per_run) = 1;
  ASSERT_OK(SetUpWithParams(1, 1, false));

  std::unordered_set<TableId> expected_table_ids;
  std::unordered_set<TabletId> expected_tablet_ids;
  const uint32_t num_tablets = 2;
  const uint32_t num_new_tables = 3;
  expected_table_ids.reserve(num_new_tables + 1);

  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  for (const auto& tablet : tablets) {
    expected_tablet_ids.insert(tablet.tablet_id());
  }
  expected_table_ids.insert(table_id);
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  // We add another table without a primary key. And we do not include the table_id in
  // 'expected_table_ids' nor do we add the tablets to 'expected_tablet_ids', since this table will
  // not be added to the stream.
  ASSERT_OK(CreateTableWithoutPK(&test_cluster_));

  // Add 3 more tables after the stream is created.
  for (uint32_t i = 1; i <= num_new_tables; ++i) {
    std::string table_name = "test_table_" + std::to_string(i);
    auto table =
        ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, table_name, num_tablets));
    TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, table_name));
    expected_table_ids.insert(table_id);

    google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
    ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));

    for (const auto& tablet : tablets) {
      expected_tablet_ids.insert(tablet.tablet_id());
    }
  }

  CheckTabletsInCDCStateTable(expected_tablet_ids, test_client());

  ASSERT_OK(WaitFor(
      [&]() -> Result<bool> {
        std::unordered_set<TableId> stream_table_ids_set;
        for (const auto& stream_id : VERIFY_RESULT(GetCDCStreamTableIds(stream_id))) {
          stream_table_ids_set.insert(stream_id);
        }

        return stream_table_ids_set == expected_table_ids;
      },
      MonoDelta::FromSeconds(60),
      "Tables associated to the stream are not the same as expected"));
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestStreamActiveOnEmptyNamespace)) {
  ASSERT_OK(SetUpWithParams(1, 1, false));

  // Create a stream on the empty namespace: test_namespace (kNamespaceName).
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  NamespaceId ns_id;
  std::vector<TableId> stream_table_ids;
  std::unordered_map<std::string, std::string> options;
  StreamModeTransactional transactional(false);
  ASSERT_OK(
      test_client()->GetCDCStream(stream_id, &ns_id, &stream_table_ids, &options, &transactional));

  const std::string& stream_state = options.at(kStreamState);
  ASSERT_EQ(
      stream_state, master::SysCDCStreamEntryPB::State_Name(master::SysCDCStreamEntryPB::ACTIVE));

  // Now add a new table to the same namespace.
  const uint32_t num_tablets = 3;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);
  std::unordered_set<TabletId> expected_tablet_ids;
  for (const auto& tablet : tablets) {
    expected_tablet_ids.insert(tablet.tablet_id());
  }

  std::vector<TableId> expected_table_ids;
  expected_table_ids.reserve(1);
  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  expected_table_ids.push_back(table_id);

  // Check that 'cdc_state' table to see if the tablets of the newly added table are also in
  // the'cdc_state' table.
  CheckTabletsInCDCStateTable(expected_tablet_ids, test_client(), stream_id);

  // Check that the stream's metadata has the newly added table_id.
  auto resp = ASSERT_RESULT(GetDBStreamInfo(stream_id));
  ASSERT_EQ(resp.table_info(0).table_id(), table_id);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestStreamActiveOnNamespaceNoPKTable)) {
  ASSERT_OK(SetUpWithParams(1, 1, false));

  // Create a table without a PK.
  ASSERT_OK(CreateTableWithoutPK(&test_cluster_));

  // Create a stream on the namespace: test_namespace (kNamespaceName).
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  NamespaceId ns_id;
  std::vector<TableId> stream_table_ids;
  std::unordered_map<std::string, std::string> options;
  StreamModeTransactional transactional(false);
  ASSERT_OK(
      test_client()->GetCDCStream(stream_id, &ns_id, &stream_table_ids, &options, &transactional));

  const std::string& stream_state = options.at(kStreamState);
  ASSERT_EQ(
      stream_state, master::SysCDCStreamEntryPB::State_Name(master::SysCDCStreamEntryPB::ACTIVE));

  const uint32_t num_tablets = 3;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);
  std::unordered_set<TabletId> expected_tablet_ids;
  for (const auto& tablet : tablets) {
    expected_tablet_ids.insert(tablet.tablet_id());
  }

  std::vector<TableId> expected_table_ids;
  expected_table_ids.reserve(1);
  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  expected_table_ids.push_back(table_id);

  // Check that 'cdc_state' table to see if the tablets of the newly added table are also in
  // the'cdc_state' table.
  CheckTabletsInCDCStateTable(expected_tablet_ids, test_client(), stream_id);

  // Check that the stream's metadata has the newly added table_id.
  auto resp = ASSERT_RESULT(GetDBStreamInfo(stream_id));
  ASSERT_EQ(resp.table_info(0).table_id(), table_id);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestIntentGCedWithTabletBootStrap)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_load_balancing) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;

  ASSERT_OK(SetUpWithParams(3, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));

  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));
  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  EnableCDCServiceInAllTserver(3);
  // Insert some records.
  ASSERT_OK(WriteRows(0 /* start */, 100 /* end */, &test_cluster_));

  // Forcefully change the tablet state from RUNNING to BOOTSTRAPPING and check metadata should not
  // set to MAX.
  for (size_t i = 0; i < test_cluster()->num_tablet_servers(); i++) {
    for (const auto& tablet_peer : test_cluster()->GetTabletPeers(i)) {
      if (tablet_peer->tablet_id() == tablets[0].tablet_id()) {
        ASSERT_OK(tablet_peer->UpdateState(
            tablet::RaftGroupStatePB::RUNNING, tablet::RaftGroupStatePB::BOOTSTRAPPING,
            "Incorrect state to start TabletPeer, "));
      }
    }
  }
  SleepFor(MonoDelta::FromSeconds(10));
  for (size_t i = 0; i < test_cluster()->num_tablet_servers(); i++) {
    for (const auto& tablet_peer : test_cluster()->GetTabletPeers(i)) {
      if (tablet_peer->tablet_id() == tablets[0].tablet_id()) {
        ASSERT_NE(tablet_peer->cdc_sdk_min_checkpoint_op_id(), OpId::Max());
        ASSERT_OK(tablet_peer->UpdateState(
            tablet::RaftGroupStatePB::BOOTSTRAPPING, tablet::RaftGroupStatePB::RUNNING,
            "Incorrect state to start TabletPeer, "));
      }
    }
  }
  LOG(INFO) << "All nodes will be restarted";
  for (size_t i = 0; i < test_cluster()->num_tablet_servers(); ++i) {
    test_cluster()->mini_tablet_server(i)->Shutdown();
    ASSERT_OK(test_cluster()->mini_tablet_server(i)->Start());
  }

  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, 100));

  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  ASSERT_GE(record_size, 100);
  LOG(INFO) << "Total records read by GetChanges call on stream_id_1: " << record_size;
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestBackwardCompatibillitySupportActiveTime)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  // We want to force every GetChanges to update the cdc_state table.
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;

  ASSERT_OK(SetUpWithParams(1, 1, false));
  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));

  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));
  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  GetChangesResponsePB change_resp;
  change_resp = ASSERT_RESULT(GetChangesFromCDC(stream_id, tablets));

  // Here we are creating a scenario where active_time is not set in the cdc_state table because of
  // older server version, if we upgrade the server where active_time is part of cdc_state table,
  // GetChanges call should successful not intents GCed error.
  CDCStateTable cdc_state_table(test_client());
  auto entry_opt = ASSERT_RESULT(cdc_state_table.TryFetchEntry(
      {tablets[0].tablet_id(), stream_id}, CDCStateTableEntrySelector().IncludeAll()));
  ASSERT_TRUE(entry_opt.has_value());
  entry_opt->active_time = std::nullopt;

  ASSERT_OK(cdc_state_table.DeleteEntries({entry_opt->key}));
  ASSERT_OK(cdc_state_table.InsertEntries({*entry_opt}));

  // Now Read the cdc_state table check active_time is set to null.
  entry_opt = ASSERT_RESULT(cdc_state_table.TryFetchEntry(
      {tablets[0].tablet_id(), stream_id}, CDCStateTableEntrySelector().IncludeAll()));
  ASSERT_TRUE(entry_opt.has_value());
  ASSERT_TRUE(!entry_opt->active_time.has_value());

  SleepFor(MonoDelta::FromSeconds(10));

  // Insert some records in transaction.
  ASSERT_OK(WriteRowsHelper(0 /* start */, 100 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, 100));

  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  ASSERT_GE(record_size, 100);
  LOG(INFO) << "Total records read by GetChanges call on stream_id_1: " << record_size;
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestBackwardCompatibillitySupportSafeTime)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_metrics_interval_ms) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  // We want to force every GetChanges to update the cdc_state table.
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;

  const uint32_t num_tservers = 3;
  ASSERT_OK(SetUpWithParams(num_tservers, 1, false));
  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));

  xrepl::StreamId stream_id =
      ASSERT_RESULT(CreateDBStream(CDCCheckpointType::IMPLICIT, CDCRecordType::PG_FULL));
  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  GetChangesResponsePB change_resp;
  change_resp = ASSERT_RESULT(GetChangesFromCDC(stream_id, tablets));

  // Here we are creating a scenario where active_time is not set in the cdc_state table because of
  // older server version, if we upgrade the server where active_time is part of cdc_state table,
  // GetChanges call should successful not intents GCed error.
  CDCStateTable cdc_state_table(test_client());

  // Insert some records in transaction.
  ASSERT_OK(WriteRowsHelper(0 /* start */, 100 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, 100));
  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  ASSERT_GE(record_size, 100);
  LOG(INFO) << "Total records read by GetChanges call on stream_id_1: " << record_size;

  // Call GetChanges again so that the checkpoint is updated.
  change_resp =
      ASSERT_RESULT(GetChangesFromCDC(stream_id, tablets, &change_resp.cdc_sdk_checkpoint()));

  auto entry_opt = ASSERT_RESULT(cdc_state_table.TryFetchEntry(
      {tablets[0].tablet_id(), stream_id}, CDCStateTableEntrySelector().IncludeAll()));
  ASSERT_TRUE(entry_opt.has_value());
  entry_opt->cdc_sdk_safe_time = std::nullopt;

  ASSERT_OK(cdc_state_table.DeleteEntries({entry_opt->key}));
  ASSERT_OK(cdc_state_table.InsertEntries({*entry_opt}));

  // Now Read the cdc_state table check cdc_sdk_safe_time is set to null.
  entry_opt = ASSERT_RESULT(cdc_state_table.TryFetchEntry(
      {tablets[0].tablet_id(), stream_id}, CDCStateTableEntrySelector().IncludeAll()));
  ASSERT_TRUE(entry_opt.has_value());
  ASSERT_TRUE(!entry_opt->cdc_sdk_safe_time.has_value());

  // We confirm if 'UpdatePeersAndMetrics' thread has updated the checkpoint in tablet tablet peer.
  for (uint tserver_index = 0; tserver_index < num_tservers; tserver_index++) {
    for (const auto& peer : test_cluster()->GetTabletPeers(tserver_index)) {
      if (peer->tablet_id() == tablets[0].tablet_id()) {
        ASSERT_OK(WaitFor(
            [&]() -> Result<bool> {
              return change_resp.cdc_sdk_checkpoint().index() ==
                     peer->cdc_sdk_min_checkpoint_op_id().index;
            },
            MonoDelta::FromSeconds(60),
            "Failed to update checkpoint in tablet peer."));
      }
    }
  }
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestDDLRecordValidationWithColocation)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_update_local_peer_min_index) = false;
  ASSERT_OK(SetUpWithParams(3, 1, false));

  ASSERT_OK(CreateColocatedObjects(&test_cluster_));
  auto table = ASSERT_RESULT(GetTable(&test_cluster_, kNamespaceName, "test1"));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), 1);

  std::string table_id = table.table_id();
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));
  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  int insert_count = 30;
  ASSERT_OK(PopulateColocatedData(&test_cluster_, insert_count, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  // Call get changes.
  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, insert_count*2));

  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  ASSERT_GT(record_size, insert_count);

  std::unordered_map<std::string, std::string> excepected_schema_name{
      {"test1", "public"}, {"test2", "public"}};
  std::unordered_map<std::string, std::vector<std::string>> excepected_column_name{
      {"test1", {"id1"}}, {"test2", {"id2"}}};

  for (uint32_t i = 0; i < record_size; ++i) {
    const auto record = change_resp.cdc_sdk_proto_records(i);
    LOG(INFO) << "Record found: " << record.ShortDebugString();
    if (record.row_message().op() == RowMessage::DDL) {
      if (excepected_schema_name.find(record.row_message().table()) ==
          excepected_schema_name.end()) {
        LOG(INFO) << "Tablename got in the record is wrong: " << record.row_message().table();
        FAIL();
      }
      ASSERT_EQ(
          excepected_schema_name[record.row_message().table()],
          record.row_message().pgschema_name());
      for (auto& ech_column_info : record.row_message().schema().column_info()) {
        if (excepected_column_name.find(record.row_message().table()) ==
            excepected_column_name.end()) {
          LOG(INFO) << "Tablename got in the record is wrong: " << record.row_message().table();
          FAIL();
        }
        auto& excepted_column_list = excepected_column_name[record.row_message().table()];
        if (std::find(
                excepted_column_list.begin(), excepted_column_list.end(), ech_column_info.name()) ==
            excepted_column_list.end()) {
          LOG(INFO) << "Colname found in the record:" << ech_column_info.name()
                    << " for the table: " << record.row_message().table()
                    << " doesn't match the expected result.";
          FAIL();
        }
      }
    }
  }
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestBeginCommitRecordValidationWithColocation)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_update_local_peer_min_index) = false;
  ASSERT_OK(SetUpWithParams(3, 1, false));

  ASSERT_OK(CreateColocatedObjects(&test_cluster_));
  auto table = ASSERT_RESULT(GetTable(&test_cluster_, kNamespaceName, "test1"));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), 1);

  std::string table_id = table.table_id();
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));
  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  int insert_count = 30;
  ASSERT_OK(PopulateColocatedData(&test_cluster_, insert_count, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  // Call get changes.
  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, insert_count*2));

  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  ASSERT_GT(record_size, insert_count);

  int expected_begin_records = 2;
  int expected_commit_records = 2;
  int actual_begin_records = 0;
  int actual_commit_records = 0;
  std::vector<std::string> excepted_table_list{"test1", "test2"};
  for (uint32_t i = 0; i < record_size; ++i) {
    const auto record = change_resp.cdc_sdk_proto_records(i);
    LOG(INFO) << "Record found: " << record.ShortDebugString();
    if (std::find(
            excepted_table_list.begin(), excepted_table_list.end(), record.row_message().table()) ==
        excepted_table_list.end()) {
      LOG(INFO) << "Tablename got in the record is wrong: " << record.row_message().table();
      FAIL();
    }

    if (record.row_message().op() == RowMessage::BEGIN) {
      actual_begin_records += 1;
    } else if (record.row_message().op() == RowMessage::COMMIT) {
      actual_commit_records += 1;
    }
  }
  ASSERT_EQ(actual_begin_records, expected_begin_records);
  ASSERT_EQ(actual_commit_records, expected_commit_records);
}

TEST_F(
    CDCSDKYsqlTest,
    YB_DISABLE_TEST_IN_TSAN(TestCDCSDKChangeEventCountMetricUnchangedOnEmptyBatches)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_metrics_interval_ms) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;
  ASSERT_OK(SetUpWithParams(1, 1, false));

  const uint32_t num_tablets = 1;
  const uint32_t num_get_changes_before_commit = 3;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  auto stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  const auto& tserver = test_cluster()->mini_tablet_server(0)->server();
  auto cdc_service = dynamic_cast<CDCServiceImpl*>(
      tserver->rpc_server()->TEST_service_pool("yb.cdc.CDCService")->TEST_get_service().get());

  // Initiate a transaction with 'BEGIN' statement.
  auto conn = ASSERT_RESULT(test_cluster_.ConnectToDB(kNamespaceName));
  ASSERT_OK(conn.Execute("BEGIN"));

  // Insert 100 rows as part of the initiated transaction.
  for (uint32_t i = 0; i < 100; ++i) {
    uint32_t value = i;
    std::stringstream statement_buff;
    statement_buff << "INSERT INTO $0 VALUES (";
    for (uint32_t iter = 0; iter < 2; ++value, ++iter) {
      statement_buff << value << ",";
    }

    std::string statement(statement_buff.str());
    statement.at(statement.size() - 1) = ')';
    ASSERT_OK(conn.ExecuteFormat(statement, kTableName));
  }

  GetChangesResponsePB change_resp = ASSERT_RESULT(GetChangesFromCDC(stream_id, tablets));
  auto metrics =
      std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
          {{}, stream_id, tablets[0].tablet_id()},
          /* tablet_peer */ nullptr, CDCSDK, CreateCDCMetricsEntity::kFalse));
  // The 'cdcsdk_change_event_count' will be 1 due to the DDL record on the first GetChanges call.
  ASSERT_EQ(metrics->cdcsdk_change_event_count->value(), 1);

  // Call 'GetChanges' 3 times, and ensure that the 'cdcsdk_change_event_count' metric dosen't
  // increase since there are no records.
  for (uint32_t i = 0; i < num_get_changes_before_commit; ++i) {
    change_resp =
        ASSERT_RESULT(GetChangesFromCDC(stream_id, tablets, &change_resp.cdc_sdk_checkpoint()));

    metrics = std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
        {{}, stream_id, tablets[0].tablet_id()},
        /* tablet_peer */ nullptr, CDCSDK, CreateCDCMetricsEntity::kFalse));

    ASSERT_EQ(metrics->cdcsdk_change_event_count->value(), 1);
  }

  // Commit the trasaction.
  ASSERT_OK(conn.Execute("COMMIT"));

  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  // Call get changes after the transaction is committed.
  ASSERT_OK(WaitForGetChangesToFetchRecords(
      &change_resp, stream_id, tablets, 100, /* is_explicit_checkpoint */false,
      &change_resp.cdc_sdk_checkpoint()));
  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  ASSERT_GT(record_size, 100);
  metrics = std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
      {{}, stream_id, tablets[0].tablet_id()},
      /* tablet_peer */ nullptr, CDCSDK, CreateCDCMetricsEntity::kFalse));
  LOG(INFO) << "Total event counts after GetChanges call: "
            << metrics->cdcsdk_change_event_count->value();
  ASSERT_GT(metrics->cdcsdk_change_event_count->value(), 100);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCDCSDKLagMetricUnchangedOnEmptyBatches)) {
  ASSERT_OK(SetUpWithParams(1, 1, false));

  const uint32_t num_tablets = 1;
  const uint32_t num_get_changes_before_commit = 3;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  auto stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  const auto& tserver = test_cluster()->mini_tablet_server(0)->server();
  auto cdc_service = dynamic_cast<CDCServiceImpl*>(
      tserver->rpc_server()->TEST_service_pool("yb.cdc.CDCService")->TEST_get_service().get());

  // Initiate a transaction with 'BEGIN' statement.
  auto conn = ASSERT_RESULT(test_cluster_.ConnectToDB(kNamespaceName));
  ASSERT_OK(conn.Execute("BEGIN"));

  // Insert 100 rows as part of the initiated transaction.
  for (uint32_t i = 0; i < 100; ++i) {
    uint32_t value = i;
    std::stringstream statement_buff;
    statement_buff << "INSERT INTO $0 VALUES (";
    for (uint32_t iter = 0; iter < 2; ++value, ++iter) {
      statement_buff << value << ",";
    }

    std::string statement(statement_buff.str());
    statement.at(statement.size() - 1) = ')';
    ASSERT_OK(conn.ExecuteFormat(statement, kTableName));
  }

  GetChangesResponsePB change_resp = ASSERT_RESULT(GetChangesFromCDC(stream_id, tablets));

  // First GetChanges call would give a single DDL record. We need to see lag in subsequent calls
  change_resp =
      ASSERT_RESULT(GetChangesFromCDC(stream_id, tablets, &change_resp.cdc_sdk_checkpoint()));
  auto metrics =
      std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
          {{}, stream_id, tablets[0].tablet_id()},
          /* tablet_peer */ nullptr, CDCSDK, CreateCDCMetricsEntity::kFalse));

  auto current_lag = metrics->cdcsdk_sent_lag_micros->value();
  ASSERT_EQ(current_lag, 0);

  // Call 'GetChanges' 3 times, and ensure that the 'cdcsdk_sent_lag_micros' metric dosen't increase
  for (uint32_t i = 0; i < num_get_changes_before_commit; ++i) {
    change_resp =
        ASSERT_RESULT(GetChangesFromCDC(stream_id, tablets, &change_resp.cdc_sdk_checkpoint()));

    metrics = std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
        {{}, stream_id, tablets[0].tablet_id()},
        /* tablet_peer */ nullptr, CDCSDK, CreateCDCMetricsEntity::kFalse));

    ASSERT_EQ(metrics->cdcsdk_sent_lag_micros->value(), current_lag);
  }

  // Commit the trasaction.
  ASSERT_OK(conn.Execute("COMMIT"));

  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));

  // Call get changes after the transaction is committed.
  ASSERT_OK(WaitForGetChangesToFetchRecords(
      &change_resp, stream_id, tablets, 100, /* is_explicit_checkpoint */false,
      &change_resp.cdc_sdk_checkpoint()));

  uint32_t record_size = change_resp.cdc_sdk_proto_records_size();
  ASSERT_GT(record_size, 100);
  metrics = std::static_pointer_cast<cdc::CDCSDKTabletMetrics>(cdc_service->GetCDCTabletMetrics(
      {{}, stream_id, tablets[0].tablet_id()},
      /* tablet_peer */ nullptr, CDCSDK, CreateCDCMetricsEntity::kFalse));
  ASSERT_GE(metrics->cdcsdk_sent_lag_micros->value(), current_lag);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestExpiredStreamWithCompaction)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;
  ASSERT_OK(SetUpWithParams(3, 1, false));
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, nullptr));
  ASSERT_EQ(tablets.size(), 1);
  xrepl::StreamId stream_id =
      ASSERT_RESULT(CreateDBStream(CDCCheckpointType::IMPLICIT, CDCRecordType::PG_FULL));
  auto set_resp =
      ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets, OpId::Min(), kuint64max, false, 0, true));
  ASSERT_FALSE(set_resp.has_error());

  ANNOTATE_UNPROTECTED_WRITE(FLAGS_timestamp_history_retention_interval_sec) = 0;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_rocksdb_level0_file_num_compaction_trigger) = 0;
  // Testing compaction without compaction file filtering for TTL expiration.
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_tablet_enable_ttl_file_filter) = false;

  ASSERT_OK(WriteRows(1 /* start */, 2 /* end */, &test_cluster_));
  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, 1));

  ASSERT_OK(UpdateRows(1 /* key */, 3 /* value */, &test_cluster_));
  ASSERT_OK(UpdateRows(1 /* key */, 4 /* value */, &test_cluster_));

  LOG(INFO) << "Sleeping to expire files according to TTL (history retention prevents deletion)";
  SleepFor(MonoDelta::FromSeconds(5));
  auto peers = ListTabletPeers(test_cluster(), ListPeersFilter::kLeaders);

  auto checkpoints = ASSERT_RESULT(GetCDCCheckpoint(stream_id, tablets));
  OpId op_id = {change_resp.cdc_sdk_checkpoint().term(), change_resp.cdc_sdk_checkpoint().index()};
  auto set_resp2 =
      ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets, op_id, change_resp.safe_hybrid_time()));
  ASSERT_FALSE(set_resp2.has_error());

  auto count_before_compaction = CountEntriesInDocDB(peers, table.table_id());
  ASSERT_OK(test_cluster_.mini_cluster_->CompactTablets());
  auto count_after_compaction = CountEntriesInDocDB(peers, table.table_id());

  ASSERT_EQ(count_before_compaction, count_after_compaction);

  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_intent_retention_ms) = 100;
  ASSERT_OK(WaitFor(
      [&]() {
        auto result = GetChangesFromCDC(stream_id, tablets);
        if (!result.ok()) {
          return true;
        }
        return false;
      },
      MonoDelta::FromSeconds(60),
      "Stream is not expired."));
  ASSERT_OK(test_cluster_.mini_cluster_->CompactTablets());
  auto count_compaction_after_expired = CountEntriesInDocDB(peers, table.table_id());
  ASSERT_LE(count_compaction_after_expired, count_after_compaction);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestStreamWithAllTablesHaveNonPrimaryKey)) {
  ASSERT_OK(SetUpWithParams(3, 1, false));

  // Adding tables without primary keys, they should not disturb any CDC related processes.
  std::vector<std::string> tables_wo_pk{"table_wo_pk_1", "table_wo_pk_2", "table_wo_pk_3"};
  std::vector<YBTableName> table_list(3);
  uint32_t idx = 0;
  for (const auto& table_name : tables_wo_pk) {
    table_list[idx] = ASSERT_RESULT(
        CreateTable(&test_cluster_, kNamespaceName, table_name, 1 /* num_tablets */, false));
    idx += 1;
  }

  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(
      table_list[0], 0, &tablets,
      /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), 1);

  std::string table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, tables_wo_pk[0]));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStreamWithReplicationSlot());

  // Set checkpoint should throw an error, for the tablet that is not part of the stream, because
  // it's non-primary key table.
  ASSERT_NOK(SetCDCCheckpoint(stream_id, tablets));

  ASSERT_OK(WriteRowsHelper(
      0 /* start */, 1 /* end */, &test_cluster_, true, 2, tables_wo_pk[0].c_str()));

  // Get changes should throw an error, for the tablet that is not part of the stream, because
  // it's non-primary key table.
  auto change_resp = GetChangesFromCDC(stream_id, tablets);
  ASSERT_FALSE(change_resp.ok());
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCommitTimeOfTransactionRecords)) {
  ASSERT_OK(SetUpWithParams(1, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  // Insert some records in transaction.
  ASSERT_OK(WriteRowsHelper(0 /* start */, 100 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));
  GetChangesResponsePB change_resp_1;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp_1, stream_id, tablets, 100));
  LOG(INFO) << "Number of records after first transaction: " << change_resp_1.records().size();

  uint64_t begin_record_commit_time = 0;
  for (auto const& record : change_resp_1.cdc_sdk_proto_records()) {
    if (record.row_message().op() == RowMessage::BEGIN) {
      begin_record_commit_time = record.row_message().commit_time();
    } else if (
        record.row_message().op() == RowMessage::INSERT ||
        record.row_message().op() == RowMessage::COMMIT) {
      ASSERT_NE(begin_record_commit_time, 0);
      ASSERT_EQ(record.row_message().commit_time(), begin_record_commit_time);
    }
  }
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCommitTimeIncreasesForTransactions)) {
  ASSERT_OK(SetUpWithParams(1, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  // Insert some records in transaction.
  ASSERT_OK(WriteRowsHelper(0 /* start */, 100 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));
  GetChangesResponsePB change_resp_1;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp_1, stream_id, tablets, 100));
  LOG(INFO) << "Number of records after first transaction: " << change_resp_1.records().size();

  uint64_t commit_time_first_txn = 0;
  for (auto const& record : change_resp_1.cdc_sdk_proto_records()) {
    if (record.row_message().op() == RowMessage::BEGIN) {
      commit_time_first_txn = record.row_message().commit_time();
      break;
    }
  }

  // Insert some records in transaction.
  ASSERT_OK(WriteRowsHelper(100 /* start */, 200 /* end */, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));
  ASSERT_OK(WaitForGetChangesToFetchRecords(
      &change_resp_1, stream_id, tablets, 100, /* is_explicit_checkpoint */false,
      &change_resp_1.cdc_sdk_checkpoint()));

  LOG(INFO) << "Number of records after second transaction: " << change_resp_1.records().size();

  uint64_t commit_time_second_txn = 0;
  for (auto const& record : change_resp_1.cdc_sdk_proto_records()) {
    if (record.row_message().op() == RowMessage::BEGIN) {
      commit_time_second_txn = record.row_message().commit_time();
      break;
    }
  }

  ASSERT_GE(commit_time_second_txn, commit_time_first_txn);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCommitTimeOrderAcrossMultiTableTransactions)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_populate_safepoint_record) = true;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_max_stream_intent_records) = 100;
  ASSERT_OK(SetUpWithParams(1, 1, false, true /* cdc_populate_safepoint_record */));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  constexpr static const char* const second_table_name = "test_table_1";
  auto second_table =
      ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, second_table_name, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets_second_table;
  ASSERT_OK(test_client()->GetTablets(
      second_table, 0, &tablets_second_table, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets_second_table.size(), num_tablets);

  TableId first_table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  TableId second_table_id =
      ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, second_table_name));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());
  resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets_second_table));
  ASSERT_FALSE(resp.has_error());

  // Insert some records in two separate transaction, affecting two tables. The promary key of each
  // row will be sorted in order of insert.
  ASSERT_OK(WriteRowsToTwoTables(0, 2, &test_cluster_, true, kTableName, second_table_name));
  ASSERT_OK(WriteRowsToTwoTables(2, 4, &test_cluster_, true, kTableName, second_table_name));

  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));
  ASSERT_OK(test_client()->FlushTables(
      {second_table.table_id()}, /* add_indexes = */ false, /* timeout_secs = */ 30,
      /* is_compaction = */ false));
  LOG(INFO) << "inserted two transactions";

  std::vector<CDCSDKProtoRecordPB> combined_records;
  combined_records.reserve(500);
  GetChangesResponsePB change_resp;
  bool first_iter = true;
  // Collect all cdcsdk records from first table into a single vector: 'combined_records'
  while (true) {
    if (first_iter) {
      change_resp = ASSERT_RESULT(GetChangesFromCDC(stream_id, tablets));
      first_iter = false;
    } else {
      change_resp =
          ASSERT_RESULT(GetChangesFromCDC(stream_id, tablets, &change_resp.cdc_sdk_checkpoint()));
    }

    if (change_resp.cdc_sdk_proto_records_size() == 1) {
      break;
    }

    bool seen_safepoint_record = false;
    for (const auto& cdc_sdk_record : change_resp.cdc_sdk_proto_records()) {
      if (cdc_sdk_record.row_message().op() == RowMessage::SAFEPOINT) {
        seen_safepoint_record = true;
      } else if (cdc_sdk_record.row_message().op() != RowMessage::DDL) {
        combined_records.push_back(cdc_sdk_record);
      }
    }
    ASSERT_TRUE(seen_safepoint_record);
  }
  LOG(INFO) << "Got all records from the first table";

  // Collect all cdcsdk records from the second table into a single vector: 'combined_records'
  first_iter = true;
  while (true) {
    if (first_iter) {
      change_resp = ASSERT_RESULT(GetChangesFromCDC(stream_id, tablets_second_table));
      first_iter = false;
    } else {
      change_resp = ASSERT_RESULT(
          GetChangesFromCDC(stream_id, tablets_second_table, &change_resp.cdc_sdk_checkpoint()));
    }

    if (change_resp.cdc_sdk_proto_records_size() == 1) {
      break;
    }

    bool seen_safepoint_record = false;
    for (const auto& cdc_sdk_record : change_resp.cdc_sdk_proto_records()) {
      if (cdc_sdk_record.row_message().op() == RowMessage::SAFEPOINT) {
        seen_safepoint_record = true;
      } else if (cdc_sdk_record.row_message().op() != RowMessage::DDL) {
        combined_records.push_back(cdc_sdk_record);
      }
    }
    ASSERT_TRUE(seen_safepoint_record);
  }
  LOG(INFO) << "Got all records from the second table";

  // Sort the combined records based on the commit and record times.
  std::sort(
      combined_records.begin(), combined_records.end(),
      [](const CDCSDKProtoRecordPB& left, const CDCSDKProtoRecordPB& right) {
        if (left.row_message().commit_time() != right.row_message().commit_time()) {
          return left.row_message().commit_time() < right.row_message().commit_time();
        } else if (
            left.row_message().op() == RowMessage::BEGIN &&
            right.row_message().op() != RowMessage::BEGIN) {
          return true;
        } else if (
            left.row_message().op() == RowMessage::COMMIT &&
            right.row_message().op() != RowMessage::COMMIT) {
          return false;
        } else if (
            right.row_message().op() == RowMessage::BEGIN &&
            left.row_message().op() != RowMessage::BEGIN) {
          return false;
        } else if (
            right.row_message().op() == RowMessage::COMMIT &&
            left.row_message().op() != RowMessage::COMMIT) {
          return true;
        } else if (left.row_message().has_record_time() && right.row_message().has_record_time()) {
          return left.row_message().record_time() < right.row_message().record_time();
        }

        return false;
      });

  // Filter out only insert records from the combined list into two separate lists based on source
  // table.
  std::vector<int32> table1_seen_record_pks, table2_seen_record_pks;
  for (auto iter = combined_records.begin(); iter != combined_records.end(); ++iter) {
    if (iter->row_message().op() != RowMessage::BEGIN &&
        iter->row_message().op() != RowMessage::COMMIT) {
      if (iter->row_message().table() == kTableName) {
        table1_seen_record_pks.push_back(iter->row_message().new_tuple(0).datum_int32());
      } else if (iter->row_message().table() == second_table_name) {
        table2_seen_record_pks.push_back(iter->row_message().new_tuple(0).datum_int32());
      }
    }
  }

  // Assert that the records are sorted in primary key (i.e order of insertion is maintained) after
  // combining all the records and sorting based on commit and record times.
  ASSERT_TRUE(std::is_sorted(table1_seen_record_pks.begin(), table1_seen_record_pks.end()));
  ASSERT_TRUE(std::is_sorted(table2_seen_record_pks.begin(), table2_seen_record_pks.end()));
  ASSERT_EQ(table1_seen_record_pks.size(), table2_seen_record_pks.size());
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCheckPointWithNoCDCStream)) {
  ASSERT_OK(SetUpWithParams(3, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  std::string table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));

  // Assert the cdc_sdk_min_checkpoint_op_id is -1.-1.
  for (size_t i = 0; i < test_cluster()->num_tablet_servers(); ++i) {
    for (const auto& peer : test_cluster()->GetTabletPeers(i)) {
      if (peer->tablet_id() == tablets[0].tablet_id()) {
        // What ever checkpoint persisted in the RAFT logs should be same as what ever in memory
        // transaction participant tablet peer.
        ASSERT_EQ(peer->cdc_sdk_min_checkpoint_op_id(), OpId::Invalid());
        ASSERT_EQ(
            peer->cdc_sdk_min_checkpoint_op_id(),
            peer->tablet()->transaction_participant()->GetRetainOpId());
      }
    }
  }

  // Restart all nodes.
  SleepFor(MonoDelta::FromSeconds(1));
  test_cluster()->mini_tablet_server(1)->Shutdown();
  ASSERT_OK(test_cluster()->mini_tablet_server(1)->Start());
  ASSERT_OK(test_cluster()->mini_tablet_server(1)->WaitStarted());

  // Re-Assert the cdc_sdk_min_checkpoint_op_id is -1.-1, even after restart
  for (size_t i = 0; i < test_cluster()->num_tablet_servers(); ++i) {
    for (const auto& peer : test_cluster()->GetTabletPeers(i)) {
      if (peer->tablet_id() == tablets[0].tablet_id()) {
        // What ever checkpoint persisted in the RAFT logs should be same as what ever in memory
        // transaction participant tablet peer.
        ASSERT_EQ(peer->cdc_sdk_min_checkpoint_op_id(), OpId::Invalid());
        ASSERT_EQ(
            peer->cdc_sdk_min_checkpoint_op_id(),
            peer->tablet()->transaction_participant()->GetRetainOpId());
      }
    }
  }
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestIsUnderCDCSDKReplicationField)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_metrics_interval_ms) = 1;
  ASSERT_OK(SetUpWithParams(3, 1, false));

  const uint32_t num_tablets = 1;
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, num_tablets));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), num_tablets);

  TableId table_id = ASSERT_RESULT(GetTableId(&test_cluster_, kNamespaceName, kTableName));
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));

  EnableCDCServiceInAllTserver(3);
  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  auto check_is_under_cdc_sdk_replication = [&](bool expected_value) {
    for (size_t i = 0; i < test_cluster()->num_tablet_servers(); ++i) {
      for (const auto& peer : test_cluster()->GetTabletPeers(i)) {
        if (peer->tablet_id() == tablets[0].tablet_id()) {
          // Check value of 'is_under_cdc_sdk_replication' in all tablet peers.
          ASSERT_EQ(peer->is_under_cdc_sdk_replication(), expected_value);
        }
      }
    }
  };

  // Assert that 'is_under_cdc_sdk_replication' remains true even after restart.
  check_is_under_cdc_sdk_replication(true);

  // Restart all the nodes.
  SleepFor(MonoDelta::FromSeconds(1));
  for (size_t i = 0; i < test_cluster()->num_tablet_servers(); ++i) {
    test_cluster()->mini_tablet_server(i)->Shutdown();
    ASSERT_OK(test_cluster()->mini_tablet_server(i)->Start());
  }
  LOG(INFO) << "All nodes restarted";
  EnableCDCServiceInAllTserver(3);

  check_is_under_cdc_sdk_replication(true);

  ASSERT_EQ(DeleteCDCStream(stream_id), true);
  VerifyStreamDeletedFromCdcState(test_client(), stream_id, tablets.Get(0).tablet_id());
  VerifyTransactionParticipant(tablets.Get(0).tablet_id(), OpId::Max());

  // Assert that after deleting the stream, 'is_under_cdc_sdk_replication' will be set to 'false'.
  check_is_under_cdc_sdk_replication(false);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestColocationWithDropColumns)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_update_local_peer_min_index) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 1;
  ASSERT_OK(SetUpWithParams(3, 1, false));

  // ASSERT_OK(CreateColocatedObjects(&test_cluster_));
  auto conn = ASSERT_RESULT(test_cluster_.ConnectToDB(kNamespaceName));
  ASSERT_OK(conn.ExecuteFormat("CREATE TABLEGROUP tg1"));
  ASSERT_OK(conn.ExecuteFormat(
      "CREATE TABLE test1(id1 int primary key, value_2 int, value_3 int) TABLEGROUP tg1;"));
  ASSERT_OK(
      conn.ExecuteFormat("CREATE TABLE test2(id2 int primary key, value_2 int, value_3 int, "
                         "value_4 int) TABLEGROUP tg1;"));

  auto table = ASSERT_RESULT(GetTable(&test_cluster_, kNamespaceName, "test1"));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), 1);

  std::string table_id = table.table_id();
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));
  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  int insert_count = 30;
  for (int i = 0; i < insert_count; ++i) {
    ASSERT_OK(conn.ExecuteFormat("INSERT INTO test1 VALUES ($0, $1, $2)", i, i + 1, i + 2));
    ASSERT_OK(
        conn.ExecuteFormat("INSERT INTO test2 VALUES ($0, $1, $2, $3)", i, i + 1, i + 2, i + 3));
  }

  ASSERT_OK(DropColumn(&test_cluster_, kNamespaceName, "test1", kValue2ColumnName));
  ASSERT_OK(DropColumn(&test_cluster_, kNamespaceName, "test2", kValue3ColumnName));
  SleepFor(MonoDelta::FromSeconds(10));

  // Call get changes.
  auto change_resp = GetAllPendingChangesFromCdc(stream_id, tablets);
  size_t record_size = change_resp.records.size();
  ASSERT_GT(record_size, insert_count);

  int expected_key1 = 0;
  int expected_key2 = 0;
  int ddl_count = 0;
  std::unordered_set<string> ddl_tables;
  for (const auto& record : change_resp.records) {
    if (record.row_message().op() == RowMessage::INSERT) {
      if (record.row_message().table() == "test1") {
        ASSERT_EQ(record.row_message().new_tuple_size(), 3);
        ASSERT_EQ(expected_key1, record.row_message().new_tuple(0).datum_int32());
        expected_key1++;
      } else if (record.row_message().table() == "test2") {
        ASSERT_EQ(record.row_message().new_tuple_size(), 4);
        ASSERT_EQ(expected_key2, record.row_message().new_tuple(0).datum_int32());
        expected_key2++;
      }
    } else if (record.row_message().op() == RowMessage::DDL) {
      ddl_tables.insert(record.row_message().table());
      ddl_count++;
    }
  }
  ASSERT_EQ(insert_count, expected_key1);
  ASSERT_EQ(insert_count, expected_key2);
  ASSERT_EQ(ddl_count, 4);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestColocationWithAddColumns)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_update_local_peer_min_index) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 1;
  ASSERT_OK(SetUpWithParams(3, 1, false));

  auto conn = ASSERT_RESULT(test_cluster_.ConnectToDB(kNamespaceName));
  ASSERT_OK(conn.ExecuteFormat("CREATE TABLEGROUP tg1"));
  ASSERT_OK(conn.ExecuteFormat(
      "CREATE TABLE test1(id1 int primary key, value_1 int, value_2 int) TABLEGROUP tg1;"));
  ASSERT_OK(
      conn.ExecuteFormat("CREATE TABLE test2(id2 int primary key, value_1 int, value_2 int, "
                         "value_3 int) TABLEGROUP tg1;"));

  auto table = ASSERT_RESULT(GetTable(&test_cluster_, kNamespaceName, "test1"));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), 1);

  std::string table_id = table.table_id();
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));
  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  int insert_count = 30;
  for (int i = 0; i < insert_count; ++i) {
      LOG(INFO) << "Inserting entry " << i;
      ASSERT_OK(conn.ExecuteFormat("INSERT INTO test1 VALUES ($0, $1, $2)", i, i + 1, i + 2));
      ASSERT_OK(
          conn.ExecuteFormat("INSERT INTO test2 VALUES ($0, $1, $2, $3)", i, i + 1, i + 2, i + 3));
  }

  ASSERT_OK(AddColumn(&test_cluster_, kNamespaceName, "test1", kValue3ColumnName));
  ASSERT_OK(AddColumn(&test_cluster_, kNamespaceName, "test2", kValue4ColumnName));
  SleepFor(MonoDelta::FromSeconds(30));

  ASSERT_OK(conn.Execute("BEGIN"));
  insert_count = 60;
  for (int i = 30; i < insert_count; ++i) {
      LOG(INFO) << "Inserting entry " << i;
      ASSERT_OK(
          conn.ExecuteFormat("INSERT INTO test1 VALUES ($0, $1, $2, $3)", i, i + 1, i + 2, i + 3));
      ASSERT_OK(conn.ExecuteFormat(
          "INSERT INTO test2 VALUES ($0, $1, $2, $3, $4)", i, i + 1, i + 2, i + 3, i + 4));
  }
  ASSERT_OK(conn.Execute("COMMIT"));

  // Call get changes.
  auto change_resp = GetAllPendingChangesFromCdc(stream_id, tablets);
  size_t record_size = change_resp.records.size();
  ASSERT_GT(record_size, insert_count);

  int expected_key1 = 0;
  int expected_key2 = 0;
  int ddl_count = 0;
  std::unordered_set<string> ddl_tables;
  for (uint32_t i = 0; i < record_size; ++i) {
      const auto record = change_resp.records[i];
      if (record.row_message().op() == RowMessage::INSERT) {
      if (record.row_message().table() == "test1") {
        ASSERT_EQ(expected_key1, record.row_message().new_tuple(0).datum_int32());
        if (expected_key1 >= 0 && expected_key1 < 30) {
          ASSERT_EQ(record.row_message().new_tuple_size(), 3);
        } else {
          ASSERT_EQ(record.row_message().new_tuple_size(), 4);
        }
        expected_key1++;
      } else if (record.row_message().table() == "test2") {
        ASSERT_EQ(expected_key2, record.row_message().new_tuple(0).datum_int32());
        if (expected_key2 >= 0 && expected_key2 < 30) {
          ASSERT_EQ(record.row_message().new_tuple_size(), 4);
        } else {
          ASSERT_EQ(record.row_message().new_tuple_size(), 5);
        }
        expected_key2++;
      }
      } else if (record.row_message().op() == RowMessage::DDL) {
      ddl_tables.insert(record.row_message().table());
      ddl_count++;
      }
  }
  ASSERT_EQ(insert_count, expected_key1);
  ASSERT_EQ(insert_count, expected_key2);
  ASSERT_EQ(ddl_count, 4);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestColocationWithAddAndDropColumns)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_update_local_peer_min_index) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 1;
  ASSERT_OK(SetUpWithParams(3, 1, true));

  auto conn = ASSERT_RESULT(test_cluster_.ConnectToDB(kNamespaceName));
  ASSERT_OK(
      conn.ExecuteFormat("CREATE TABLE test1(id1 int primary key, value_1 int, value_2 int);"));
  ASSERT_OK(
      conn.ExecuteFormat("CREATE TABLE test2(id2 int primary key, value_1 int, value_2 int, "
                         "value_3 int);"));

  auto table = ASSERT_RESULT(GetTable(&test_cluster_, kNamespaceName, "test1"));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), 1);

  std::string table_id = table.table_id();
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));
  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  int insert_count = 30;
  for (int i = 0; i < insert_count; ++i) {
    LOG(INFO) << "Inserting entry " << i;
    ASSERT_OK(conn.ExecuteFormat("INSERT INTO test1 VALUES ($0, $1, $2)", i, i + 1, i + 2));
    ASSERT_OK(
        conn.ExecuteFormat("INSERT INTO test2 VALUES ($0, $1, $2, $3)", i, i + 1, i + 2, i + 3));
  }

  ASSERT_OK(AddColumn(&test_cluster_, kNamespaceName, "test1", kValue3ColumnName));
  ASSERT_OK(DropColumn(&test_cluster_, kNamespaceName, "test2", kValue3ColumnName));
  ASSERT_OK(DropColumn(&test_cluster_, kNamespaceName, "test2", kValue2ColumnName));
  SleepFor(MonoDelta::FromSeconds(30));

  ASSERT_OK(conn.Execute("BEGIN"));
  insert_count = 60;
  for (int i = 30; i < insert_count; ++i) {
      LOG(INFO) << "Inserting entry " << i;
      ASSERT_OK(
          conn.ExecuteFormat("INSERT INTO test1 VALUES ($0, $1, $2, $3)", i, i + 1, i + 2, i + 3));
      ASSERT_OK(conn.ExecuteFormat("INSERT INTO test2 VALUES ($0, $1)", i, i + 1));
  }
  ASSERT_OK(conn.Execute("COMMIT"));

  // Call get changes.
  auto change_resp = GetAllPendingChangesFromCdc(stream_id, tablets);
  size_t record_size = change_resp.records.size();
  ASSERT_GT(record_size, insert_count);

  int expected_key1 = 0;
  int expected_key2 = 0;
  int ddl_count = 0;
  std::unordered_set<string> ddl_tables;
  for (uint32_t i = 0; i < record_size; ++i) {
      const auto record = change_resp.records[i];
      if (record.row_message().op() == RowMessage::INSERT) {
      if (record.row_message().table() == "test1") {
        ASSERT_EQ(expected_key1, record.row_message().new_tuple(0).datum_int32());
        if (expected_key1 >= 0 && expected_key1 < 30) {
          ASSERT_EQ(record.row_message().new_tuple_size(), 3);
        } else {
          ASSERT_EQ(record.row_message().new_tuple_size(), 4);
        }
        expected_key1++;
      } else if (record.row_message().table() == "test2") {
        ASSERT_EQ(expected_key2, record.row_message().new_tuple(0).datum_int32());
        if (expected_key2 >= 0 && expected_key2 < 30) {
          ASSERT_EQ(record.row_message().new_tuple_size(), 4);
        } else {
          ASSERT_EQ(record.row_message().new_tuple_size(), 2);
        }
        expected_key2++;
      }
      } else if (record.row_message().op() == RowMessage::DDL) {
      ddl_tables.insert(record.row_message().table());
      ddl_count++;
      }
  }
  ASSERT_EQ(insert_count, expected_key1);
  ASSERT_EQ(insert_count, expected_key2);
  ASSERT_EQ(ddl_count, 5);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestColocationWithMultipleAlterAndRestart)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_update_local_peer_min_index) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 1;
  ASSERT_OK(SetUpWithParams(3, 1, true));

  auto conn = ASSERT_RESULT(test_cluster_.ConnectToDB(kNamespaceName));
  ASSERT_OK(
      conn.ExecuteFormat("CREATE TABLE test1(id1 int primary key, value_1 int, value_2 int);"));
  ASSERT_OK(
      conn.ExecuteFormat("CREATE TABLE test2(id2 int primary key, value_1 int, value_2 int, "
                         "value_3 int);"));

  auto table = ASSERT_RESULT(GetTable(&test_cluster_, kNamespaceName, "test1"));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), 1);

  std::string table_id = table.table_id();
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));
  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  int insert_count = 30;
  for (int i = 0; i < insert_count; ++i) {
      LOG(INFO) << "Inserting entry " << i;
      ASSERT_OK(conn.ExecuteFormat("INSERT INTO test1 VALUES ($0, $1, $2)", i, i + 1, i + 2));
      ASSERT_OK(
          conn.ExecuteFormat("INSERT INTO test2 VALUES ($0, $1, $2, $3)", i, i + 1, i + 2, i + 3));
  }

  ASSERT_OK(AddColumn(&test_cluster_, kNamespaceName, "test1", kValue3ColumnName));
  ASSERT_OK(DropColumn(&test_cluster_, kNamespaceName, "test2", kValue3ColumnName));
  ASSERT_OK(DropColumn(&test_cluster_, kNamespaceName, "test2", kValue2ColumnName));

  // Call get changes.
  auto change_resp = GetAllPendingChangesFromCdc(stream_id, tablets);
  size_t record_size = change_resp.records.size();
  ASSERT_GT(record_size, insert_count);

  int expected_key1 = 0;
  int expected_key2 = 0;
  int ddl_count = 0;
  std::unordered_set<string> ddl_tables;
  for (uint32_t i = 0; i < record_size; ++i) {
      const auto record = change_resp.records[i];
      if (record.row_message().op() == RowMessage::INSERT) {
      if (record.row_message().table() == "test1") {
        ASSERT_EQ(expected_key1, record.row_message().new_tuple(0).datum_int32());
        ASSERT_EQ(record.row_message().new_tuple_size(), 3);
        expected_key1++;
      } else if (record.row_message().table() == "test2") {
        ASSERT_EQ(expected_key2, record.row_message().new_tuple(0).datum_int32());
        ASSERT_EQ(record.row_message().new_tuple_size(), 4);
        expected_key2++;
      }
      } else if (record.row_message().op() == RowMessage::DDL) {
      ddl_tables.insert(record.row_message().table());
      ddl_count++;
      }
  }
  ASSERT_EQ(insert_count, expected_key1);
  ASSERT_EQ(insert_count, expected_key2);
  ASSERT_EQ(ddl_count, 5);

  for (int idx = 0; idx < 3; idx++) {
    test_cluster()->mini_tablet_server(idx)->Shutdown();
    ASSERT_OK(test_cluster()->mini_tablet_server(idx)->Start());
    ASSERT_OK(test_cluster()->mini_tablet_server(idx)->WaitStarted());
  }
  LOG(INFO) << "All nodes restarted";

  conn = ASSERT_RESULT(test_cluster_.ConnectToDB(kNamespaceName));
  ASSERT_OK(conn.Execute("BEGIN"));
  insert_count = 60;
  for (int i = 30; i < insert_count; ++i) {
      LOG(INFO) << "Inserting entry " << i;
      ASSERT_OK(conn.ExecuteFormat("INSERT INTO test1 VALUES ($0, $1, $2)", i, i + 1, i + 2));
      ASSERT_OK(conn.ExecuteFormat("INSERT INTO test2 VALUES ($0, $1)", i, i + 1));
  }
  ASSERT_OK(conn.Execute("COMMIT"));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false,
      /* timeout_secs = */ 30, /* is_compaction = */ false));

  // Call get changes.
  change_resp = GetAllPendingChangesFromCdc(stream_id, tablets, &change_resp.checkpoint);
  record_size = change_resp.records.size();
  ASSERT_GT(record_size, insert_count/2);

  expected_key1 = 30;
  expected_key2 = 30;
  ddl_count = 0;
  for (uint32_t i = 0; i < record_size; ++i) {
      const auto record = change_resp.records[i];
      if (record.row_message().op() == RowMessage::INSERT) {
      if (record.row_message().table() == "test1") {
        ASSERT_EQ(expected_key1, record.row_message().new_tuple(0).datum_int32());
        ASSERT_EQ(record.row_message().new_tuple_size(), 4);
        expected_key1++;
      } else if (record.row_message().table() == "test2") {
        ASSERT_EQ(expected_key2, record.row_message().new_tuple(0).datum_int32());
        ASSERT_EQ(record.row_message().new_tuple_size(), 2);
        expected_key2++;
      }
    } else if (record.row_message().op() == RowMessage::DDL) {
      ddl_tables.insert(record.row_message().table());
      ddl_count++;
    }
  }
  ASSERT_EQ(insert_count, expected_key1);
  ASSERT_EQ(insert_count, expected_key2);
  ASSERT_EQ(ddl_count, 2);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestColocationWithMultipleAlterAndLeaderSwitch)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_update_local_peer_min_index) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_load_balancing) = false;
  ASSERT_OK(SetUpWithParams(3, 1, true));

  auto conn = ASSERT_RESULT(test_cluster_.ConnectToDB(kNamespaceName));
  ASSERT_OK(
      conn.ExecuteFormat("CREATE TABLE test1(id1 int primary key, value_1 int, value_2 int);"));
  ASSERT_OK(
      conn.ExecuteFormat("CREATE TABLE test2(id2 int primary key, value_1 int, value_2 int, "
                         "value_3 int);"));

  auto table = ASSERT_RESULT(GetTable(&test_cluster_, kNamespaceName, "test1"));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), 1);

  std::string table_id = table.table_id();
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));
  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  int insert_count = 30;
  for (int i = 0; i < insert_count; ++i) {
    LOG(INFO) << "Inserting entry " << i;
    ASSERT_OK(conn.ExecuteFormat("INSERT INTO test1 VALUES ($0, $1, $2)", i, i + 1, i + 2));
    ASSERT_OK(
        conn.ExecuteFormat("INSERT INTO test2 VALUES ($0, $1, $2, $3)", i, i + 1, i + 2, i + 3));
  }

  ASSERT_OK(AddColumn(&test_cluster_, kNamespaceName, "test1", kValue3ColumnName));
  ASSERT_OK(DropColumn(&test_cluster_, kNamespaceName, "test2", kValue3ColumnName));
  ASSERT_OK(DropColumn(&test_cluster_, kNamespaceName, "test2", kValue2ColumnName));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false,
      /* timeout_secs = */ 30, /* is_compaction = */ false));

  // Call get changes.
  auto change_resp = GetAllPendingChangesFromCdc(stream_id, tablets);
  size_t record_size = change_resp.records.size();
  ASSERT_GT(record_size, insert_count);

  int expected_key1 = 0;
  int expected_key2 = 0;
  int ddl_count = 0;
  std::unordered_set<string> ddl_tables;
  for (uint32_t i = 0; i < record_size; ++i) {
    const auto record = change_resp.records[i];
    if (record.row_message().op() == RowMessage::INSERT) {
      if (record.row_message().table() == "test1") {
        ASSERT_EQ(expected_key1, record.row_message().new_tuple(0).datum_int32());
        ASSERT_EQ(record.row_message().new_tuple_size(), 3);
        expected_key1++;
      } else if (record.row_message().table() == "test2") {
        ASSERT_EQ(expected_key2, record.row_message().new_tuple(0).datum_int32());
        ASSERT_EQ(record.row_message().new_tuple_size(), 4);
        expected_key2++;
      }
    } else if (record.row_message().op() == RowMessage::DDL) {
      ddl_tables.insert(record.row_message().table());
      ddl_count++;
    }
  }
  ASSERT_EQ(insert_count, expected_key1);
  ASSERT_EQ(insert_count, expected_key2);
  ASSERT_EQ(ddl_count, 5);

  size_t first_leader_index = -1;
  size_t first_follower_index = -1;
  GetTabletLeaderAndAnyFollowerIndex(tablets, &first_leader_index, &first_follower_index);
  if (first_leader_index == 0) {
    // We want to avoid the scenario where the first TServer is the leader, since we want to shut
    // the leader TServer down and call GetChanges. GetChanges will be called on the cdc_proxy
    // based on the first TServer's address and we want to avoid the network issues.
    ASSERT_OK(ChangeLeaderOfTablet(first_follower_index, tablets[0].tablet_id()));
  }
  ASSERT_OK(ChangeLeaderOfTablet(first_follower_index, tablets[0].tablet_id()));

  conn = ASSERT_RESULT(test_cluster_.ConnectToDB(kNamespaceName));
  ASSERT_OK(conn.Execute("BEGIN"));
  insert_count = 60;
  for (int i = 30; i < insert_count; ++i) {
    LOG(INFO) << "Inserting entry " << i;
    ASSERT_OK(conn.ExecuteFormat("INSERT INTO test1 VALUES ($0, $1, $2)", i, i + 1, i + 2));
    ASSERT_OK(conn.ExecuteFormat("INSERT INTO test2 VALUES ($0, $1)", i, i + 1));
  }
  ASSERT_OK(conn.Execute("COMMIT"));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false,
      /* timeout_secs = */ 30, /* is_compaction = */ false));

  // Call get changes.
  change_resp = GetAllPendingChangesFromCdc(stream_id, tablets, &change_resp.checkpoint);
  record_size = change_resp.records.size();
  ASSERT_GT(record_size, insert_count / 2);

  expected_key1 = 30;
  expected_key2 = 30;
  ddl_count = 0;
  for (uint32_t i = 0; i < record_size; ++i) {
    const auto record = change_resp.records[i];
    if (record.row_message().op() == RowMessage::INSERT) {
      if (record.row_message().table() == "test1") {
        ASSERT_EQ(expected_key1, record.row_message().new_tuple(0).datum_int32());
        ASSERT_EQ(record.row_message().new_tuple_size(), 4);
        expected_key1++;
      } else if (record.row_message().table() == "test2") {
        ASSERT_EQ(expected_key2, record.row_message().new_tuple(0).datum_int32());
        ASSERT_EQ(record.row_message().new_tuple_size(), 2);
        expected_key2++;
      }
    } else if (record.row_message().op() == RowMessage::DDL) {
      ddl_tables.insert(record.row_message().table());
      ddl_count++;
    }
  }
  ASSERT_EQ(insert_count, expected_key1);
  ASSERT_EQ(insert_count, expected_key2);
  ASSERT_EQ(ddl_count, 2);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestColocationWithRepeatedRequestFromOpId)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_update_local_peer_min_index) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_load_balancing) = false;
  ASSERT_OK(SetUpWithParams(3, 1, true));

  auto conn = ASSERT_RESULT(test_cluster_.ConnectToDB(kNamespaceName));
  ASSERT_OK(
      conn.ExecuteFormat("CREATE TABLE test1(id1 int primary key, value_1 int, value_2 int);"));
  ASSERT_OK(
      conn.ExecuteFormat("CREATE TABLE test2(id2 int primary key, value_1 int, value_2 int, "
                         "value_3 int);"));

  auto table = ASSERT_RESULT(GetTable(&test_cluster_, kNamespaceName, "test1"));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), 1);

  std::string table_id = table.table_id();
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));
  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  int insert_count = 30;
  for (int i = 0; i < insert_count; ++i) {
    LOG(INFO) << "Inserting entry " << i;
    ASSERT_OK(conn.ExecuteFormat("INSERT INTO test1 VALUES ($0, $1, $2)", i, i + 1, i + 2));
    ASSERT_OK(
        conn.ExecuteFormat("INSERT INTO test2 VALUES ($0, $1, $2, $3)", i, i + 1, i + 2, i + 3));
  }

  ASSERT_OK(AddColumn(&test_cluster_, kNamespaceName, "test1", kValue3ColumnName));
  ASSERT_OK(DropColumn(&test_cluster_, kNamespaceName, "test2", kValue3ColumnName));
  ASSERT_OK(DropColumn(&test_cluster_, kNamespaceName, "test2", kValue2ColumnName));

  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false,
      /* timeout_secs = */ 30, /* is_compaction = */ false));

  // Call get changes.
  auto change_resp = GetAllPendingChangesFromCdc(stream_id, tablets);
  size_t record_size = change_resp.records.size();
  ASSERT_GT(record_size, insert_count);

  int expected_key1 = 0;
  int expected_key2 = 0;
  int ddl_count = 0;
  std::unordered_set<string> ddl_tables;
  for (uint32_t i = 0; i < record_size; ++i) {
    const auto record = change_resp.records[i];
    if (record.row_message().op() == RowMessage::INSERT) {
      if (record.row_message().table() == "test1") {
        ASSERT_EQ(expected_key1, record.row_message().new_tuple(0).datum_int32());
        ASSERT_EQ(record.row_message().new_tuple_size(), 3);
        expected_key1++;
      } else if (record.row_message().table() == "test2") {
        ASSERT_EQ(expected_key2, record.row_message().new_tuple(0).datum_int32());
        ASSERT_EQ(record.row_message().new_tuple_size(), 4);
        expected_key2++;
      }
    } else if (record.row_message().op() == RowMessage::DDL) {
      ddl_tables.insert(record.row_message().table());
      ddl_count++;
    }
  }
  ASSERT_EQ(insert_count, expected_key1);
  ASSERT_EQ(insert_count, expected_key2);
  ASSERT_EQ(ddl_count, 5);

  conn = ASSERT_RESULT(test_cluster_.ConnectToDB(kNamespaceName));
  ASSERT_OK(conn.Execute("BEGIN"));
  insert_count = 60;
  for (int i = 30; i < insert_count; ++i) {
    LOG(INFO) << "Inserting entry " << i;
    ASSERT_OK(conn.ExecuteFormat("INSERT INTO test1 VALUES ($0, $1, $2)", i, i + 1, i + 2));
    ASSERT_OK(conn.ExecuteFormat("INSERT INTO test2 VALUES ($0, $1)", i, i + 1));
  }
  ASSERT_OK(conn.Execute("COMMIT"));
  ASSERT_OK(test_client()->FlushTables(
      {table.table_id()}, /* add_indexes = */ false,
      /* timeout_secs = */ 30, /* is_compaction = */ false));

  // Call get changes.
  const auto repeat_checkpoint = change_resp.checkpoint;
  change_resp = GetAllPendingChangesFromCdc(stream_id, tablets, &change_resp.checkpoint);
  record_size = change_resp.records.size();
  ASSERT_GT(record_size, insert_count / 2);

  // Call get changes again with the same from_op_id.
  change_resp = GetAllPendingChangesFromCdc(stream_id, tablets, &repeat_checkpoint);

  expected_key1 = 30;
  expected_key2 = 30;
  ddl_count = 0;
  for (uint32_t i = 0; i < record_size; ++i) {
    const auto record = change_resp.records[i];
    if (record.row_message().op() == RowMessage::INSERT) {
      if (record.row_message().table() == "test1") {
        ASSERT_EQ(expected_key1, record.row_message().new_tuple(0).datum_int32());
        ASSERT_EQ(record.row_message().new_tuple_size(), 4);
        expected_key1++;
      } else if (record.row_message().table() == "test2") {
        ASSERT_EQ(expected_key2, record.row_message().new_tuple(0).datum_int32());
        ASSERT_EQ(record.row_message().new_tuple_size(), 2);
        expected_key2++;
      }
    } else if (record.row_message().op() == RowMessage::DDL) {
      ddl_tables.insert(record.row_message().table());
      ddl_count++;
    }
  }
  ASSERT_EQ(insert_count, expected_key1);
  ASSERT_EQ(insert_count, expected_key2);
  ASSERT_EQ(ddl_count, 2);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestExplicitCheckpointGetChangesRequest)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;
  auto tablets = ASSERT_RESULT(SetUpCluster());
  ASSERT_EQ(tablets.size(), 1);
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStreamWithReplicationSlot());
  auto set_resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(set_resp.has_error());

  ASSERT_OK(WriteRowsHelper(1 /* start */, 101 /* end */, &test_cluster_, true));

  // Not setting explicit checkpoint here.
  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, 100));

  // Since stream is in EXPLICIT mode, the checkpoint won't be stored in cdc_state table.
  auto checkpoint = ASSERT_RESULT(
      GetStreamCheckpointInCdcState(test_client(), stream_id, tablets[0].tablet_id()));
  ASSERT_EQ(checkpoint, OpId());

  // This time call 'GetChanges' with an explicit checkpoint.
  ASSERT_RESULT(GetChangesFromCDCWithExplictCheckpoint(
      stream_id, tablets, &change_resp.cdc_sdk_checkpoint()));

  // The checkpoint stored in the cdc_state table will be updated.
  checkpoint = ASSERT_RESULT(
      GetStreamCheckpointInCdcState(test_client(), stream_id, tablets[0].tablet_id()));
  ASSERT_EQ(
      checkpoint,
      OpId(change_resp.cdc_sdk_checkpoint().term(), change_resp.cdc_sdk_checkpoint().index()));
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestTransactionWithZeroIntents)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_ysql_num_shards_per_tserver) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_update_local_peer_min_index) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 1;
  ASSERT_OK(SetUpWithParams(1, 1, false));

  auto conn = ASSERT_RESULT(test_cluster_.ConnectToDB(kNamespaceName));
  ASSERT_OK(conn.ExecuteFormat("CREATE TABLE test1(id1 int primary key, value_1 int);"));
  ASSERT_OK(
      conn.ExecuteFormat("CREATE TABLE test2(id2 int primary key, id_fk int, FOREIGN KEY (id_fk) "
                         "REFERENCES test1 (id1));"));

  // Create two tables with parent key - foreign key relation.
  auto parent_table = ASSERT_RESULT(GetTable(&test_cluster_, kNamespaceName, "test1"));
  auto fk_table = ASSERT_RESULT(GetTable(&test_cluster_, kNamespaceName, "test2"));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> fk_tablets;
  ASSERT_OK(
      test_client()->GetTablets(fk_table, 0, &fk_tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(fk_tablets.size(), 1);

  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> parent_tablets;
  ASSERT_OK(test_client()->GetTablets(
      parent_table, 0, &parent_tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(parent_tablets.size(), 1);

  std::string fk_table_id = fk_table.table_id();
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));
  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, fk_tablets));
  ASSERT_FALSE(resp.has_error());
  resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, parent_tablets));
  ASSERT_FALSE(resp.has_error());

  const int insert_count = 30;
  ASSERT_OK(conn.Execute("BEGIN"));
  for (int i = 0; i < insert_count; ++i) {
    ASSERT_OK(conn.ExecuteFormat("INSERT INTO test1 VALUES ($0, $1)", i, i + 1));
  }
  ASSERT_OK(conn.Execute("COMMIT"));

  // This transaction on the foreign key table, will induce another transaction on the parent table
  // to have 0 intents.
  ASSERT_OK(conn.Execute("BEGIN"));
  for (int i = 0; i < insert_count; ++i) {
    ASSERT_OK(conn.ExecuteFormat("INSERT INTO test2 VALUES ($0, $1)", i + 1, i));
  }
  ASSERT_OK(conn.Execute("COMMIT"));

  ASSERT_OK(test_client()->FlushTables(
      {parent_table.table_id()}, /* add_indexes = */ false,
      /* timeout_secs = */ 30, /* is_compaction = */ false));
  ASSERT_OK(test_client()->FlushTables(
      {fk_table.table_id()}, /* add_indexes = */ false,
      /* timeout_secs = */ 30, /* is_compaction = */ false));

  // Assert get changes works without error on both the tables.
  GetChangesResponsePB change_resp = ASSERT_RESULT(GetChangesFromCDC(stream_id, fk_tablets));

  change_resp = ASSERT_RESULT(GetChangesFromCDC(stream_id, parent_tablets));
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestGetCheckpointForColocatedTable)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_snapshot_batch_size) = 100;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_update_local_peer_min_index) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;
  ASSERT_OK(SetUpWithParams(3, 1, true /* colocated */));

  auto conn = ASSERT_RESULT(test_cluster_.ConnectToDB(kNamespaceName));
  ASSERT_OK(
      conn.ExecuteFormat("CREATE TABLE test1(id1 int primary key, value_2 int, value_3 int);"));
  ASSERT_OK(
      conn.ExecuteFormat("CREATE TABLE test2(id2 int primary key, value_2 int, value_3 int, "
                         "value_4 int);"));

  auto table = ASSERT_RESULT(GetTable(&test_cluster_, kNamespaceName, "test1"));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), 1);

  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));
  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  const int64_t snapshot_recrods_per_table = 500;
  for (int i = 0; i < snapshot_recrods_per_table; ++i) {
    ASSERT_OK(conn.ExecuteFormat("INSERT INTO test1 VALUES ($0, $1, $2)", i, i + 1, i + 2));
    ASSERT_OK(
        conn.ExecuteFormat("INSERT INTO test2 VALUES ($0, $1, $2, $3)", i, i + 1, i + 2, i + 3));
  }

  auto verify_snapshot_checkpoint = [&](const GetChangesResponsePB& initial_change_resp,
                                        const TableId& req_table_id) {
    bool first_call = true;
    GetChangesResponsePB change_resp;
    GetChangesResponsePB next_change_resp;
    uint64 expected_snapshot_time;

    while (true) {
      if (first_call) {
        next_change_resp =
            ASSERT_RESULT(UpdateCheckpoint(stream_id, tablets, &initial_change_resp, req_table_id));
      } else {
        next_change_resp =
            ASSERT_RESULT(UpdateCheckpoint(stream_id, tablets, &change_resp, req_table_id));
      }

      auto resp =
          ASSERT_RESULT(GetCDCSnapshotCheckpoint(stream_id, tablets[0].tablet_id(), req_table_id));
      ASSERT_GE(resp.snapshot_time(), 0);

      if (first_call) {
        ASSERT_EQ(
            resp.checkpoint().op_id().term(), initial_change_resp.cdc_sdk_checkpoint().term());
        ASSERT_EQ(
            resp.checkpoint().op_id().index(), initial_change_resp.cdc_sdk_checkpoint().index());
        ASSERT_EQ(resp.snapshot_key(), "");
        expected_snapshot_time = resp.snapshot_time();
        first_call = false;
      } else {
        ASSERT_EQ(resp.checkpoint().op_id().term(), change_resp.cdc_sdk_checkpoint().term());
        ASSERT_EQ(resp.checkpoint().op_id().index(), change_resp.cdc_sdk_checkpoint().index());
        ASSERT_EQ(resp.snapshot_key(), change_resp.cdc_sdk_checkpoint().key());
        ASSERT_EQ(resp.snapshot_time(), expected_snapshot_time);
      }

      change_resp = next_change_resp;

      if (change_resp.cdc_sdk_checkpoint().key().empty() &&
          change_resp.cdc_sdk_checkpoint().write_id() == 0 &&
          change_resp.cdc_sdk_checkpoint().snapshot_time() == 0) {
        break;
      }
    }
  };

  auto req_table_id = GetColocatedTableId("test1");
  ASSERT_NE(req_table_id, "");
  // Assert that we get all records from the second table: "test1".
  GetChangesResponsePB initial_change_resp =
      ASSERT_RESULT(GetChangesFromCDCSnapshot(stream_id, tablets));
  verify_snapshot_checkpoint(initial_change_resp, req_table_id);
  LOG(INFO) << "Verified snapshot records for table: test1";

  // Assert that we get all records from the second table: "test2".
  req_table_id = GetColocatedTableId("test2");
  ASSERT_NE(req_table_id, "");
  verify_snapshot_checkpoint(initial_change_resp, req_table_id);
  LOG(INFO) << "Verified snapshot records for table: test2";
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestGetCheckpointOnStreamedColocatedTable)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_update_local_peer_min_index) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;
  ASSERT_OK(SetUpWithParams(3, 1, true /* colocated */));

  auto conn = ASSERT_RESULT(test_cluster_.ConnectToDB(kNamespaceName));
  ASSERT_OK(
      conn.ExecuteFormat("CREATE TABLE test1(id1 int primary key, value_2 int, value_3 int);"));

  auto table = ASSERT_RESULT(GetTable(&test_cluster_, kNamespaceName, "test1"));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), 1);

  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));
  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  const int64_t snapshot_recrods_per_table = 100;
  for (int i = 0; i < snapshot_recrods_per_table; ++i) {
    ASSERT_OK(conn.ExecuteFormat("INSERT INTO test1 VALUES ($0, $1, $2)", i, i + 1, i + 2));
  }

  auto req_table_id = GetColocatedTableId("test1");
  ASSERT_NE(req_table_id, "");
  auto change_resp = ASSERT_RESULT(GetChangesFromCDCSnapshot(stream_id, tablets));
  while (true) {
    change_resp = ASSERT_RESULT(UpdateCheckpoint(stream_id, tablets, &change_resp, req_table_id));

    if (change_resp.cdc_sdk_checkpoint().key().empty() &&
        change_resp.cdc_sdk_checkpoint().write_id() == 0 &&
        change_resp.cdc_sdk_checkpoint().snapshot_time() == 0) {
      break;
    }
  }
  LOG(INFO) << "Streamed snapshot records for table: test1";

  for (int i = snapshot_recrods_per_table; i < 2 * snapshot_recrods_per_table; ++i) {
    ASSERT_OK(conn.ExecuteFormat("INSERT INTO test1 VALUES ($0, $1, $2)", i, i + 1, i + 2));
  }

  auto snapshot_done_resp = ASSERT_RESULT(UpdateSnapshotDone(stream_id, tablets, req_table_id));
  auto checkpoint_resp =
      ASSERT_RESULT(GetCDCSnapshotCheckpoint(stream_id, tablets[0].tablet_id(), req_table_id));
  ASSERT_TRUE(!checkpoint_resp.has_snapshot_key() || checkpoint_resp.snapshot_key().empty());

  auto stream_change_resp =
      ASSERT_RESULT(UpdateCheckpoint(stream_id, tablets, &change_resp, req_table_id));
  stream_change_resp =
      ASSERT_RESULT(UpdateCheckpoint(stream_id, tablets, &change_resp, req_table_id));
  checkpoint_resp =
      ASSERT_RESULT(GetCDCSnapshotCheckpoint(stream_id, tablets[0].tablet_id(), req_table_id));

  ASSERT_EQ(
      OpId::FromPB(checkpoint_resp.checkpoint().op_id()),
      OpId::FromPB(change_resp.cdc_sdk_checkpoint()));
  ASSERT_FALSE(checkpoint_resp.has_snapshot_key());
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestGetCheckpointOnAddedColocatedTable)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_update_local_peer_min_index) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;
  ASSERT_OK(SetUpWithParams(1, 1, true /* colocated */));

  auto conn = ASSERT_RESULT(test_cluster_.ConnectToDB(kNamespaceName));
  ASSERT_OK(
      conn.ExecuteFormat("CREATE TABLE test1(id1 int primary key, value_2 int, value_3 int);"));

  auto table = ASSERT_RESULT(GetTable(&test_cluster_, kNamespaceName, "test1"));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), 1);

  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(IMPLICIT));
  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  const int64_t snapshot_recrods_per_table = 100;
  for (int i = 0; i < snapshot_recrods_per_table; ++i) {
    ASSERT_OK(conn.ExecuteFormat("INSERT INTO test1 VALUES ($0, $1, $2)", i, i + 1, i + 2));
  }

  auto req_table_id = GetColocatedTableId("test1");
  ASSERT_NE(req_table_id, "");
  auto change_resp = ASSERT_RESULT(GetChangesFromCDCSnapshot(stream_id, tablets, req_table_id));
  while (true) {
    change_resp = ASSERT_RESULT(UpdateCheckpoint(stream_id, tablets, &change_resp, req_table_id));

    if (change_resp.cdc_sdk_checkpoint().key().empty() &&
        change_resp.cdc_sdk_checkpoint().write_id() == 0 &&
        change_resp.cdc_sdk_checkpoint().snapshot_time() == 0) {
      break;
    }
  }
  ASSERT_RESULT(UpdateSnapshotDone(stream_id, tablets, req_table_id));
  LOG(INFO) << "Streamed snapshot records for table: test1";

  for (int i = snapshot_recrods_per_table; i < 2 * snapshot_recrods_per_table; ++i) {
    ASSERT_OK(conn.ExecuteFormat("INSERT INTO test1 VALUES ($0, $1, $2)", i, i + 1, i + 2));
  }

  auto stream_change_resp_before_add_table =
      ASSERT_RESULT(UpdateCheckpoint(stream_id, tablets, &change_resp, req_table_id));
  stream_change_resp_before_add_table = ASSERT_RESULT(
      UpdateCheckpoint(stream_id, tablets, &stream_change_resp_before_add_table, req_table_id));

  auto streaming_checkpoint_resp =
      ASSERT_RESULT(GetCDCSnapshotCheckpoint(stream_id, tablets[0].tablet_id(), req_table_id));
  ASSERT_FALSE(streaming_checkpoint_resp.has_snapshot_key());

  // Create a new table and wait for the table to be added to the stream.
  ASSERT_OK(
      conn.ExecuteFormat("CREATE TABLE test2(id1 int primary key, value_2 int, value_3 int);"));
  auto added_table_id = GetColocatedTableId("test2");

  // Wait until the newly added table is added to the stream's metadata.
  ASSERT_OK(WaitFor(
      [&]() {
        auto result = GetCDCStreamTableIds(stream_id);
        if (!result.ok()) {
          return false;
        }
        const auto& table_ids = result.get();
        return std::find(table_ids.begin(), table_ids.end(), added_table_id) != table_ids.end();
      },
      MonoDelta::FromSeconds(180), "New table not added to stream"));

  auto added_table_checkpoint_resp =
      ASSERT_RESULT(GetCDCSnapshotCheckpoint(stream_id, tablets[0].tablet_id(), added_table_id));
  ASSERT_EQ(OpId::FromPB(added_table_checkpoint_resp.checkpoint().op_id()), OpId::Invalid());
  ASSERT_FALSE(added_table_checkpoint_resp.has_snapshot_key());

  for (int i = 0; i < snapshot_recrods_per_table; ++i) {
    ASSERT_OK(conn.ExecuteFormat("INSERT INTO test2 VALUES ($0, $1, $2)", i, i + 1, i + 2));
  }

  auto added_table_change_resp =
      ASSERT_RESULT(GetChangesFromCDCSnapshot(stream_id, tablets, added_table_id));
  added_table_checkpoint_resp =
      ASSERT_RESULT(GetCDCSnapshotCheckpoint(stream_id, tablets[0].tablet_id(), added_table_id));
  ASSERT_GT(
      OpId::FromPB(added_table_checkpoint_resp.checkpoint().op_id()),
      OpId::FromPB(streaming_checkpoint_resp.checkpoint().op_id()));

  int64_t seen_snapshot_records = 0;
  while (true) {
    added_table_change_resp = ASSERT_RESULT(
        UpdateCheckpoint(stream_id, tablets, &added_table_change_resp, added_table_id));

    for (const auto& record : added_table_change_resp.cdc_sdk_proto_records()) {
      if (record.row_message().op() == RowMessage::READ) {
        seen_snapshot_records += 1;
      }
    }

    if (added_table_change_resp.cdc_sdk_checkpoint().key().empty() &&
        added_table_change_resp.cdc_sdk_checkpoint().write_id() == 0 &&
        added_table_change_resp.cdc_sdk_checkpoint().snapshot_time() == 0) {
      break;
    }
  }
  ASSERT_EQ(seen_snapshot_records, snapshot_recrods_per_table);

  added_table_change_resp = ASSERT_RESULT(UpdateSnapshotDone(stream_id, tablets, added_table_id));
  added_table_checkpoint_resp =
      ASSERT_RESULT(GetCDCSnapshotCheckpoint(stream_id, tablets[0].tablet_id(), added_table_id));

  ASSERT_EQ(
      OpId::FromPB(added_table_checkpoint_resp.checkpoint().op_id()),
      OpId::FromPB(streaming_checkpoint_resp.checkpoint().op_id()));

  ASSERT_EQ(
      OpId(
          added_table_change_resp.cdc_sdk_checkpoint().term(),
          added_table_change_resp.cdc_sdk_checkpoint().index()),
      OpId::FromPB(streaming_checkpoint_resp.checkpoint().op_id()));
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestAddManyColocatedTablesOnNamesapceWithStream)) {
  ASSERT_OK(SetUpWithParams(3 /* replication_factor */, 2 /* num_masters */, true /* colocated */));

  auto conn = ASSERT_RESULT(test_cluster_.ConnectToDB(kNamespaceName));
  ASSERT_RESULT(CreateDBStream(IMPLICIT));

  for (int i = 1; i <= 400; i++) {
    std::string table_name = "test" + std::to_string(i);
    ASSERT_OK(conn.ExecuteFormat(
        "CREATE TABLE $0(id1 int primary key, value_2 int, value_3 int);", table_name));
    LOG(INFO) << "Done create table: " << table_name;
  }
}

TEST_F(
    CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestGetAndSetCheckpointWithDefaultNumTabletsForTable)) {
  ASSERT_OK(SetUpWithParams(3, 1, false));
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  // Create a table with default number of tablets.
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, nullptr));
  ASSERT_EQ(tablets.size(), 1);
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(CDCCheckpointType::IMPLICIT));
  auto set_resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(set_resp.has_error());

  ASSERT_OK(WriteRows(1 /* start */, 2 /* end */, &test_cluster_));
  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, 1));

  ASSERT_OK(UpdateRows(1 /* key */, 3 /* value */, &test_cluster_));
  ASSERT_OK(UpdateRows(1 /* key */, 4 /* value */, &test_cluster_));

  auto checkpoints = ASSERT_RESULT(GetCDCCheckpoint(stream_id, tablets));
  OpId op_id = {change_resp.cdc_sdk_checkpoint().term(), change_resp.cdc_sdk_checkpoint().index()};
  auto set_resp2 =
      ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets, op_id, change_resp.safe_hybrid_time()));
  ASSERT_FALSE(set_resp2.has_error());

  GetChangesResponsePB change_resp2;
  ASSERT_OK(WaitForGetChangesToFetchRecords(
      &change_resp2, stream_id, tablets, 2, /* is_explicit_checkpoint */false,
      &change_resp.cdc_sdk_checkpoint()));

  checkpoints = ASSERT_RESULT(GetCDCCheckpoint(stream_id, tablets));
  OpId op_id2 = {
      change_resp.cdc_sdk_checkpoint().term(), change_resp2.cdc_sdk_checkpoint().index()};
  auto set_resp3 =
      ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets, op_id2, change_resp2.safe_hybrid_time()));
  ASSERT_FALSE(set_resp2.has_error());
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestBeginAndCommitRecordsForSingleShardTxns)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_populate_end_markers_transactions) = true;
  ASSERT_OK(SetUpWithParams(3, 1, false));
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  // Create a table with default number of tablets.
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, nullptr));
  ASSERT_EQ(tablets.size(), 1);
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(CDCCheckpointType::IMPLICIT));
  auto set_resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(set_resp.has_error());

  const int total_rows_inserted = 100;
  for (int i = 1; i <= total_rows_inserted; ++i) {
    ASSERT_OK(WriteRows(i, i + 1, &test_cluster_));
  }

  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecords(&change_resp, stream_id, tablets, total_rows_inserted));

  int seen_begin_records = 0, seen_commit_records = 0, seen_insert_records = 0;
  bool expected_begin_record = true, expected_insert_record = false, expected_commit_record = false;
  // Confirm that we place a "BEGIN" record before, and "COMMIT" record after every "INSERT" record.
  for (const auto& record : change_resp.cdc_sdk_proto_records()) {
    if (record.row_message().op() == RowMessage::INSERT) {
      ++seen_insert_records;
      ASSERT_TRUE(expected_insert_record);
      expected_insert_record = false;
      expected_commit_record = true;
    } else if (record.row_message().op() == RowMessage::BEGIN) {
      ++seen_begin_records;
      ASSERT_TRUE(expected_begin_record);
      expected_begin_record = false;
      expected_insert_record = true;
    } else if (record.row_message().op() == RowMessage::COMMIT) {
      ++seen_commit_records;
      ASSERT_TRUE(expected_commit_record);
      expected_commit_record = false;
      expected_begin_record = true;
    }
  }

  ASSERT_EQ(seen_insert_records, total_rows_inserted);
  ASSERT_EQ(seen_begin_records, total_rows_inserted);
  ASSERT_EQ(seen_commit_records, total_rows_inserted);
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestFromOpIdInGetChangesResponse)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_max_stream_intent_records) = 40;

  ASSERT_OK(SetUpWithParams(3, 1, false));
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, nullptr));
  ASSERT_EQ(tablets.size(), 1);
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStreamWithReplicationSlot());
  auto set_resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets, OpId::Min()));
  ASSERT_FALSE(set_resp.has_error());
  ASSERT_OK(WriteRowsHelper(0, 30, &test_cluster_, true));

  auto conn = ASSERT_RESULT(test_cluster_.ConnectToDB(kNamespaceName));
  ASSERT_OK(conn.Execute("ALTER TABLE test_table ADD value_2 int;"));
  ASSERT_OK(conn.Execute("ALTER TABLE test_table DROP value_2;"));

  ASSERT_OK(WriteRowsHelper(30, 80, &test_cluster_, true));

  ASSERT_OK(test_client()->FlushTables({table.table_id()}, false, 1000, false));

  const int expected_count[] = {3, 80, 0, 0, 0, 0, 2, 2};
  int count[] = {0, 0, 0, 0, 0, 0, 0, 0};

  auto get_changes_resp = ASSERT_RESULT(GetChangesFromCDC(stream_id, tablets));
  for (int i = 0; i < get_changes_resp.cdc_sdk_proto_records_size(); i++) {
    auto record = get_changes_resp.cdc_sdk_proto_records(i);
    UpdateRecordCount(record, count);
    ASSERT_EQ(record.from_op_id().term(), 0);
    ASSERT_EQ(record.from_op_id().index(), 0);
  }

  const auto prev_checkpoint = get_changes_resp.cdc_sdk_checkpoint();
  get_changes_resp = ASSERT_RESULT(GetChangesFromCDC(stream_id, tablets, &prev_checkpoint));
  for (int i = 0; i < get_changes_resp.cdc_sdk_proto_records_size(); i++) {
    auto record = get_changes_resp.cdc_sdk_proto_records(i);
    UpdateRecordCount(record, count);
    ASSERT_EQ(record.from_op_id().term(), prev_checkpoint.term());
    ASSERT_EQ(record.from_op_id().index(), prev_checkpoint.index());
  }

  auto pending_changes_resp = GetAllPendingChangesFromCdc(
      stream_id, tablets, &get_changes_resp.cdc_sdk_checkpoint(), 0,
      get_changes_resp.safe_hybrid_time(), get_changes_resp.wal_segment_index());
  for (const auto& record : pending_changes_resp.records) {
    UpdateRecordCount(record, count);
  }

  for (int i = 0; i < 8; i++) {
    ASSERT_EQ(expected_count[i], count[i]);
  }
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestLargeTxnWithExplicitStream)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_max_stream_intent_records) = 40;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_ysql_enable_packed_row) = true;

  ASSERT_OK(SetUpWithParams(3, 1, false));
  auto table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, nullptr));
  ASSERT_EQ(tablets.size(), 1);
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStreamWithReplicationSlot());
  auto set_resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets, OpId::Min()));
  ASSERT_FALSE(set_resp.has_error());

  // Number of rows is intentionally kept equal to the value of cdc_max_stream_intent_records.
  const int row_count = 40;
  ASSERT_OK(WriteRowsHelper(0, row_count, &test_cluster_, true));
  ASSERT_OK(test_client()->FlushTables({table.table_id()}, false, 1000, false));

  const int expected_count[] = {1, 40, 0, 0, 0, 0, 1, 1};
  int count[] = {0, 0, 0, 0, 0, 0, 0, 0};

  GetChangesResponsePB get_changes_resp;

  ASSERT_OK(WaitFor(
      [&]() -> Result<bool> {
        auto get_changes_resp_result = GetChangesFromCDC(stream_id, tablets);
        if (get_changes_resp_result.ok()) {
          get_changes_resp = (*get_changes_resp_result);
          for (const auto& record : get_changes_resp.cdc_sdk_proto_records()) {
            UpdateRecordCount(record, count);
          }
        }
        return count[1] == row_count;
      },
      MonoDelta::FromSeconds(5), "Wait for getchanges to fetch records"));

  const auto& prev_checkpoint = get_changes_resp.cdc_sdk_checkpoint();
  set_resp = ASSERT_RESULT(
      SetCDCCheckpoint(stream_id, tablets, OpId{prev_checkpoint.term(), prev_checkpoint.index()}));
  ASSERT_FALSE(set_resp.has_error());

  auto get_changes_resp_2 = ASSERT_RESULT(GetChangesFromCDC(stream_id, tablets, &prev_checkpoint));
  for (const auto& record : get_changes_resp_2.cdc_sdk_proto_records()) {
    UpdateRecordCount(record, count);
  }

  for (int i = 0; i < 8; i++) {
    ASSERT_EQ(expected_count[i], count[i]);
  }
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestGetCheckpointOnSnapshotBootstrapExplicit)) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_update_local_peer_min_index) = false;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_update_min_cdc_indices_interval_secs) = 1;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_cdc_state_checkpoint_update_interval_ms) = 0;
  ASSERT_OK(SetUpWithParams(1, 1, false /* colocated */));

  auto conn = ASSERT_RESULT(test_cluster_.ConnectToDB(kNamespaceName));
  ASSERT_OK(
      conn.ExecuteFormat("CREATE TABLE test1(id1 int primary key, value_2 int, value_3 int);"));

  auto table = ASSERT_RESULT(GetTable(&test_cluster_, kNamespaceName, "test1"));
  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, /* partition_list_version =*/nullptr));
  ASSERT_EQ(tablets.size(), 1);

  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream(EXPLICIT));
  auto resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets));
  ASSERT_FALSE(resp.has_error());

  const int64_t snapshot_recrods_per_table = 100;
  for (int i = 0; i < snapshot_recrods_per_table; ++i) {
    ASSERT_OK(conn.ExecuteFormat("INSERT INTO test1 VALUES ($0, $1, $2)", i, i + 1, i + 2));
  }

  auto change_resp = ASSERT_RESULT(GetChangesFromCDCSnapshot(stream_id, tablets));

  auto checkpoint_resp = ASSERT_RESULT(GetCDCSnapshotCheckpoint(stream_id, tablets[0].tablet_id()));

  ASSERT_EQ(change_resp.cdc_sdk_checkpoint().term(), checkpoint_resp.checkpoint().op_id().term());
  ASSERT_EQ(change_resp.cdc_sdk_checkpoint().index(), checkpoint_resp.checkpoint().op_id().index());
}

TEST_F(CDCSDKYsqlTest, TestAlterOperationTableRewrite) {
  ASSERT_OK(SetUpWithParams(3, 1, false));
  constexpr auto kColumnName = "c1";
  auto conn = ASSERT_RESULT(test_cluster_.ConnectToDB(kNamespaceName));
  ASSERT_OK(conn.ExecuteFormat(
      "CREATE TABLE $0(id1 INT PRIMARY KEY, $1 varchar(10))", kTableName, kColumnName));
  ASSERT_RESULT(CreateDBStreamWithReplicationSlot());

  // Verify alter primary key, column type operations are disallowed on the table.
  auto res = conn.ExecuteFormat("ALTER TABLE $0 DROP CONSTRAINT $0_pkey", kTableName);
  ASSERT_NOK(res);
  ASSERT_STR_CONTAINS(res.ToString(),
      "cannot change the primary key of a table that is a part of CDC or XCluster replication.");
  res = conn.ExecuteFormat("ALTER TABLE $0 ALTER $1 TYPE varchar(1)", kTableName, kColumnName);
  ASSERT_NOK(res);
  ASSERT_STR_CONTAINS(res.ToString(),
      "cannot change a column type of a table that is a part of CDC or XCluster replication.");
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestUnrelatedTableDropUponTserverRestart)) {
  FLAGS_catalog_manager_bg_task_wait_ms = 50000;
  ASSERT_OK(SetUpWithParams(3, 1, false));
  auto old_table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, "old_table"));

  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(old_table, 0, &tablets, nullptr));
  ASSERT_EQ(tablets.size(), 1);
  LOG(INFO) << "Tablet ID at index 0 is " << tablets.Get(0).tablet_id();
  xrepl::StreamId stream_id = ASSERT_RESULT(CreateDBStream());

  auto set_resp = ASSERT_RESULT(SetCDCCheckpoint(stream_id, tablets, OpId::Min()));
  ASSERT_FALSE(set_resp.has_error());

  GetChangesResponsePB change_resp_1 = ASSERT_RESULT(GetChangesFromCDC(stream_id, tablets));

  // Call GetChanges again.
  GetChangesResponsePB change_resp_2 =
    ASSERT_RESULT(GetChangesFromCDC(stream_id, tablets, &change_resp_1.cdc_sdk_checkpoint()));

  // Create new table.
  auto new_table = ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, "new_table"));

  // Wait till this table gets added to the stream.
  ASSERT_OK(WaitFor(
      [&]() -> Result<bool> {
        while (true) {
          auto resp = GetDBStreamInfo(stream_id);
          if (resp.ok()) {
            for (auto table_info : resp->table_info()) {
              if (table_info.has_table_id() && table_info.table_id() == new_table.table_id()) {
                return true;
              }
            }
          }
          continue;
        }
        return false;
      },
      MonoDelta::FromSeconds(70), "Waiting for table to be added."));

  // Restart tserver hosting old tablet
  TabletId old_tablet = tablets.Get(0).tablet_id();
  uint32_t tserver_idx = -1;
  for (uint32_t idx = 0; idx < 3; ++idx) {
    auto tablet_peer_ptr =
      test_cluster_.mini_cluster_->GetTabletManager(idx)->LookupTablet(old_tablet);
    if (tablet_peer_ptr != nullptr && !tablet_peer_ptr->IsNotLeader()) {
      LOG(INFO) << "Tserver at index " << idx << " hosts the leader for tablet " << old_tablet;
      tserver_idx = idx;
      break;
    }
  }

  ASSERT_OK(test_cluster_.mini_cluster_->mini_tablet_server(tserver_idx)->Restart());

  // Drop newly created table.
  auto conn = ASSERT_RESULT(test_cluster_.ConnectToDB(kNamespaceName));
  DropTable(&test_cluster_, "new_table");

  // Call GetChanges on the old table.
  LOG(INFO) << "Calling last GetChanges";
  GetChangesResponsePB change_resp_3 = ASSERT_RESULT(
      GetChangesFromCDCWithoutRetry(stream_id, tablets, &change_resp_2.cdc_sdk_checkpoint()));
}

TEST_F(CDCSDKYsqlTest, YB_DISABLE_TEST_IN_TSAN(TestCDCStreamCreationDisabledDuringUpgrade)) {
  ASSERT_OK(SetUpWithParams(/*replication_factor=*/3, /*num_masters=*/1, /*colocated=*/false));
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_ysql_yb_enable_replication_commands) = false;

  CreateCDCStreamRequestPB req;
  CreateCDCStreamResponsePB resp;
  rpc::RpcController rpc;
  rpc.set_timeout(MonoDelta::FromMilliseconds(FLAGS_cdc_write_rpc_timeout_ms));
  InitCreateStreamRequest(&req);
  ASSERT_OK(cdc_proxy_->CreateCDCStream(req, &resp, &rpc));

  ASSERT_TRUE(resp.has_error());
  ASSERT_NE(
      resp.error().status().message().find("Creating a CDC stream is disallowed during an upgrade"),
      std::string::npos)
      << resp.error().status().message();
}

TEST_F(CDCSDKYsqlTest, TestReplicationSlotDropWithActiveInvalid) {
  // Set the active window to a smaller value for faster test execution.
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_ysql_cdc_active_replication_slot_window_ms) =
      10000 * yb::kTimeMultiplier;

  ASSERT_OK(SetUpWithParams(3, 1, false));
  auto table =
      ASSERT_RESULT(CreateTable(&test_cluster_, kNamespaceName, kTableName, /*num_tablets=*/3));

  google::protobuf::RepeatedPtrField<master::TabletLocationsPB> tablets;
  ASSERT_OK(test_client()->GetTablets(table, 0, &tablets, nullptr));
  ASSERT_EQ(tablets.size(), 3);
  xrepl::StreamId stream_id =
      ASSERT_RESULT(CreateDBStreamWithReplicationSlot("repl_slot_drop_with_active_invalid"));

  // set checkpoint for each tablet.
  for (int tablet_idx = 0; tablet_idx < tablets.size(); tablet_idx++) {
    auto set_resp = ASSERT_RESULT(SetCDCCheckpoint(
        stream_id, tablets, OpId::Min(), /*cdc_sdk_safe_time=*/0, /*initial_checkpoint=*/true,
        tablet_idx));
    ASSERT_FALSE(set_resp.has_error());
  }

  auto repl_conn = ASSERT_RESULT(test_cluster_.ConnectToDBWithReplication(kNamespaceName));

  ASSERT_OK(WriteRows(1 /* start */, 11 /* end */, &test_cluster_));

  GetChangesResponsePB change_resp;
  ASSERT_OK(WaitForGetChangesToFetchRecordsAcrossTablets(
      stream_id, tablets, /*expected_count=*/10, /* is_explicit_checkpoint */true));

  // Dropping the replication slot which was consumed within the last
  // ysql_cdc_active_replication_slot_window_ms duration fails.
  auto drop_status = repl_conn.Execute("DROP_REPLICATION_SLOT repl_slot_drop_with_active_invalid");
  ASSERT_NOK(drop_status);
  ASSERT_NE(
      drop_status.message().AsStringView().find(
          "replication slot \"repl_slot_drop_with_active_invalid\" is active"),
      std::string::npos)
      << drop_status.message();

  SleepFor(MonoDelta::FromMilliseconds(FLAGS_ysql_cdc_active_replication_slot_window_ms));

  // Dropping the replication slot should succeed now that we haven't consumed it in the last
  // ysql_cdc_active_replication_slot_window_ms duration.
  ASSERT_OK(repl_conn.Execute("DROP_REPLICATION_SLOT repl_slot_drop_with_active_invalid"));
}

// Test for https://github.com/yugabyte/yugabyte-db/issues/19894.
TEST_F(CDCSDKYsqlTest, TestPgReplicationSlotsView) {
  ASSERT_OK(
      SetUpWithParams(3 /* replication_factor */, 1 /* num_masters */, false /* colocated */));

  // Disable VLOG for this test so that the log statements present in
  // CdcStateTableIterator::operator*() - cdc_state_table.cc do not execute.
  // It is important to not execute them so that we don't check the status of the
  // Result<CDCStateTableEntry> inside it and be able to reproduce the crash that happens afterwards
  // due to us not checking the status correctly while iterating the cdc state table entry in
  // ListReplicationSlots - pg_client_service.cc.
  google::SetVLOGLevel("cdc*", 0);

  auto conn = ASSERT_RESULT(test_cluster_.ConnectToDB(kNamespaceName));

  ASSERT_OK(conn.Execute(
      "create table t1 (id int primary key, name text, l_name varchar, hours float);"));

  ASSERT_OK(conn.Execute("create publication pub for all tables;"));

  ASSERT_OK(conn.FetchFormat(
      "SELECT * FROM pg_create_logical_replication_slot('$0', 'yboutput', false)",
      "test_replication_slot_create_with_view"));

  ASSERT_OK(conn.Fetch("SELECT * FROM pg_replication_slots"));
}

TEST_F(CDCSDKYsqlTest, TestCreateCDCStreamReplicationSlotLimit) {
  auto tablets = ASSERT_RESULT(SetUpCluster());
  ASSERT_EQ(tablets.size(), 1);

  // Overwrite the max_replication_slots value to a small value for ease of testing.
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_max_replication_slots) = 2;

  int limit = 2;
  for (int i = 1; i <= limit; i++) {
    ASSERT_RESULT(CreateDBStreamWithReplicationSlot(Format("repl_slot_limit_test_$0", i)));
  }

  // Creation should fail since we have reached the limit.
  auto creation_at_limit =
      CreateDBStreamWithReplicationSlot(Format("repl_slot_limit_test_$0", limit + 1));
  ASSERT_NOK(creation_at_limit);
  ASSERT_NE(
      creation_at_limit.status().ToUserMessage().find(
          "all replication slots are in use"),
      std::string::npos)
      << creation_at_limit.status().ToUserMessage();

  // Increase limit by 1 and creation should succeed.
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_max_replication_slots) = 3;
  ASSERT_RESULT(CreateDBStreamWithReplicationSlot(Format("repl_slot_limit_test_$0", limit + 1)));
}

}  // namespace cdc
}  // namespace yb
