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
#include <eventql/sql/statements/select/hash_join.h>
#include <murmurhash/murmur3.h>

namespace csql {

static uint32_t computeHash(const char* data, size_t size) {
  uint32_t hash;
  MurmurHash3_x86_32(data, size, 42, &hash);
  return hash;
}

HashJoin::HashJoin(
    Transaction* txn,
    JoinType join_type,
    const Vector<JoinNode::InputColumnRef>& input_map,
    Vector<ValueExpression> select_expressions,
    Option<ValueExpression> join_cond_expr,
    Option<ValueExpression> where_expr,
    std::vector<std::pair<ValueExpression, ValueExpression>> conjunction_exprs,
    ScopedPtr<TableExpression> base_tbl,
    ScopedPtr<TableExpression> joined_tbl) :
    txn_(txn),
    join_type_(join_type),
    input_map_(input_map),
    select_exprs_(std::move(select_expressions)),
    join_cond_expr_(std::move(join_cond_expr)),
    where_expr_(std::move(where_expr)),
    conjunction_exprs_(std::move(conjunction_exprs)),
    base_tbl_(std::move(base_tbl)),
    joined_tbl_(std::move(joined_tbl)) {}

ReturnCode HashJoin::execute() {
  /* read joined table into hash map */
  {
    auto rc = readJoinedTable();
    if (!rc.isSuccess()) {
      return rc;
    }
  }

  /* start base table execution */
  {
    auto rc = base_tbl_->execute();
    if (!rc.isSuccess()) {
      return rc;
    }
  }

  for (size_t i = 0; i < base_tbl_->getColumnCount(); ++i) {
    base_tbl_cols_.emplace_back(base_tbl_->getColumnType(i));
  }

  return ReturnCode::success();
}

ReturnCode HashJoin::nextBatch(
    SVector* out,
    size_t* nrecords) {
  for (;;) {
    *nrecords = 0;

    /* fetch base table input columns */
    for (auto& c : base_tbl_cols_) {
      c.clear();
    }

    size_t base_nrecords = 0;
    {
      auto rc = base_tbl_->nextBatch(base_tbl_cols_.data(), &base_nrecords);
      if (!rc.isSuccess()) {
        RAISE(kRuntimeError, rc.getMessage());
      }
    }

    if (base_nrecords == 0) {
      return ReturnCode::success();
    }

    /* compute join */
    auto rc = computeInnerJoin(base_nrecords, out, nrecords);
    if (!rc.isSuccess()) {
      return rc;
    }

    if (*nrecords > 0) {
      return ReturnCode::success();
    }
  }
}

size_t HashJoin::getColumnCount() const {
  return select_exprs_.size();
}

SType HashJoin::getColumnType(size_t idx) const {
  assert(idx < select_exprs_.size());
  return select_exprs_[idx].getReturnType();
}

ReturnCode HashJoin::computeInnerJoin(
    size_t base_records,
    SVector* output,
    size_t* output_records) {
  for (size_t i = 0; i < select_exprs_.size(); ++i) {
    auto etype = select_exprs_[i].getReturnType();
    if (sql_sizeof_static(etype)) {
      output[i].increaseCapacity(sql_sizeof_static(etype) * base_records);
    }
  }

  Vector<SValue> row;
  for (size_t i = 0; i < input_map_.size(); ++i) {
    row.emplace_back(input_map_[i].type);
  }

  Vector<void*> base_col_cursor;
  for (const auto& c : base_tbl_cols_) {
    base_col_cursor.emplace_back((void*) c.getData());
  }

  Vector<SType> hash_key_tpl_types;
  for (size_t i = conjunction_exprs_.size(); i-- > 0; ) {
    hash_key_tpl_types.emplace_back(conjunction_exprs_[i].first.getReturnType());
  }

  auto nrecs = 0;
  for (size_t n = 0; n < base_records; ++n) {
    for (size_t i = 0; i < input_map_.size(); ++i) {
      const auto& m = input_map_[i];
      if (m.table_idx == 0) {
        row[i].copyFrom(base_col_cursor[m.column_idx]);
      }
    }

    for (size_t i = 0; i < conjunction_exprs_.size(); ++i) {
      VM::evaluateBoxed(
          txn_,
          conjunction_exprs_[i].first.program(),
          conjunction_exprs_[i].first.program()->method_call,
          &vm_stack_,
          nullptr,
          row.size(),
          row.data());
    }

    auto hash_key_tpl_len = sql_sizeof_tuple(
        vm_stack_.top,
        hash_key_tpl_types.data(),
        hash_key_tpl_types.size());

    auto hash_key = computeHash(vm_stack_.top, hash_key_tpl_len);
    vm::popStack(&vm_stack_, hash_key_tpl_len);

    auto joined_data_range = joined_tbl_data_.equal_range(hash_key);
    auto joined_row = joined_data_range.first;
    for (; joined_row != joined_data_range.second; ++joined_row) {
      for (size_t i = 0; i < input_map_.size(); ++i) {
        const auto& m = input_map_[i];
        if (m.table_idx == 1) {
          row[i] = joined_row->second[i];
        }
      }

      if (computeOutputRow(row, output)) {
        ++nrecs;
      }
    }

    for (size_t i = 0; i < base_col_cursor.size(); ++i) {
      if (base_col_cursor[i]) {
        base_tbl_cols_[i].next(&base_col_cursor[i]);
      }
    }
  }

  *output_records = nrecs;
  return ReturnCode::success();
}

bool HashJoin::computeOutputRow(
    const std::vector<SValue>& input,
    SVector* output) {
  if (!join_cond_expr_.isEmpty()) {
    VM::evaluateBoxed(
        txn_,
        join_cond_expr_.get().program(),
        join_cond_expr_.get().program()->method_call,
        &vm_stack_,
        nullptr,
        input.size(),
        input.data());

    if (!popBool(&vm_stack_)) {
      return false;
    }
  }

  if (!where_expr_.isEmpty()) {
    VM::evaluateBoxed(
        txn_,
        where_expr_.get().program(),
        where_expr_.get().program()->method_call,
        &vm_stack_,
        nullptr,
        input.size(),
        input.data());

    if (!popBool(&vm_stack_)) {
      return false;
    }
  }

  for (int i = 0; i < select_exprs_.size(); ++i) {
    SValue val(select_exprs_[i].getReturnType());
    VM::evaluateBoxed(
        txn_,
        select_exprs_[i].program(),
        select_exprs_[i].program()->method_call,
        &vm_stack_,
        nullptr,
        input.size(),
        input.data());

    popVector(&vm_stack_, output + i);
  }

  return true;
}

ReturnCode HashJoin::readJoinedTable() {
  auto joined_rc = joined_tbl_->execute();
  if (!joined_rc.isSuccess()) {
    return joined_rc;
  }

  Vector<SVector> input_cols;
  for (size_t i = 0; i < joined_tbl_->getColumnCount(); ++i) {
    input_cols.emplace_back(joined_tbl_->getColumnType(i));
  }

  size_t cnt = 0;
  for (;;) {
    for (auto& c : input_cols) {
      c.clear();
    }

    size_t nrecords = 0;
    {
      auto rc = joined_tbl_->nextBatch(input_cols.data(), &nrecords);
      if (!rc.isSuccess()) {
        RAISE(kRuntimeError, rc.getMessage());
      }
    }

    if (nrecords == 0) {
      break;
    }

    {
      auto rc = txn_->triggerHeartbeat();
      if (!rc.isSuccess()) {
        RAISE(kRuntimeError, rc.getMessage());
      }
    }

    Vector<void*> input_col_cursor;
    for (const auto& c : input_cols) {
      input_col_cursor.emplace_back((void*) c.getData());
    }

    Vector<SValue> row;
    for (size_t i = 0; i < input_map_.size(); ++i) {
      row.emplace_back(input_map_[i].type);
    }

    Vector<SType> hash_key_tpl_types;
    for (size_t i = conjunction_exprs_.size(); i-- > 0; ) {
      hash_key_tpl_types.emplace_back(conjunction_exprs_[i].first.getReturnType());
    }

    for (size_t n = 0; n < nrecords; n++) {
      for (size_t i = 0; i < input_map_.size(); ++i) {
        const auto& m = input_map_[i];
        if (m.table_idx != 1) {
          continue;
        }

        row[i].copyFrom(input_col_cursor[m.column_idx]);
      }

      for (size_t i = 0; i < conjunction_exprs_.size(); ++i) {
        VM::evaluateBoxed(
            txn_,
            conjunction_exprs_[i].second.program(),
            conjunction_exprs_[i].second.program()->method_call,
            &vm_stack_,
            nullptr,
            row.size(),
            row.data());
      }

      auto hash_key_tpl_len = sql_sizeof_tuple(
          vm_stack_.top,
          hash_key_tpl_types.data(),
          hash_key_tpl_types.size());

      auto hash_key = computeHash(vm_stack_.top, hash_key_tpl_len);
      vm::popStack(&vm_stack_, hash_key_tpl_len);

      joined_tbl_data_.emplace(hash_key, row);

      for (size_t i = 0; i < input_col_cursor.size(); ++i) {
        if (input_col_cursor[i]) {
          input_cols[i].next(&input_col_cursor[i]);
        }
      }
    }
  }

  return ReturnCode::success();
}

} // namespace csql

