// Author: jinliang

#pragma once

#include <petuum_ps/oplog/abstract_oplog.hpp>

#include <petuum_ps_common/util/striped_lock.hpp>

namespace petuum {
class DenseOpLog : public AbstractOpLog {
public:
  DenseOpLog(int32_t capacity, const AbstractRow *sample_row,
             size_t dense_row_oplog_capacity,
             int32_t row_oplog_type);
  ~DenseOpLog();

  void RegisterThread() { }
  void DeregisterThread() { }
  void FlushOpLog() { }

  // exclusive access
  int32_t Inc(int32_t row_id, int32_t column_id, const void *delta);
  int32_t BatchInc(int32_t row_id, const int32_t *column_ids,
    const void *deltas, int32_t num_updates);

  int32_t DenseBatchInc(int32_t row_id, const void *updates,
                     int32_t index_st, int32_t num_updates);

  // Guaranteed exclusive accesses to the same row id.
  bool FindOpLog(int32_t row_id, OpLogAccessor *oplog_accessor);
  // return true if a new row oplog is created
  bool FindInsertOpLog(int32_t row_id, OpLogAccessor *oplog_accessor);
  // oplog_accessor aquires the lock on the row whether or not the
  // row oplog exists.
  bool FindAndLock(int32_t row_id, OpLogAccessor *oplog_accessor);

  // Not mutual exclusive but is less expensive than FIndOpLog above as it does
  // not use any lock.
  AbstractRowOpLog *FindOpLog(int32_t row_id);
  AbstractRowOpLog *FindInsertOpLog(int32_t row_id);

  // Mutual exclusive accesses
  bool GetEraseOpLog(int32_t row_id, AbstractRowOpLog **row_oplog_ptr);
  bool GetEraseOpLogIf(int32_t row_id, GetOpLogTestFunc test,
                       void *test_args, AbstractRowOpLog **row_oplog_ptr);

  bool GetInvalidateOpLogMeta(int32_t row_id, RowOpLogMeta *row_oplog_meta);

  AbstractAppendOnlyBuffer *GetAppendOnlyBuffer(int32_t comm_channel_idx);
  void PutBackBuffer(int32_t comm_channel_idx, AbstractAppendOnlyBuffer* buff);

private:

  AbstractRowOpLog *FindRowOpLog(int32_t row_id);
  AbstractRowOpLog *CreateAndInsertRowOpLog(int32_t row_id);

  int32_t GetVecIndex(int32_t row_id);

  const size_t update_size_;
  StripedLock<int32_t> locks_;
  std::vector<AbstractRowOpLog*> oplog_vec_;
  const AbstractRow *sample_row_;
  const size_t dense_row_oplog_capacity_;
  CreateRowOpLog::CreateRowOpLogFunc CreateRowOpLog_;
  const size_t capacity_;
};

}   // namespace petuum
