/* Copyright (c) 2022 StoneAtom, Inc. All rights reserved.
   Use is subject to license terms

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; version 2 of the License.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1335 USA
*/

#include "tianmu_table_index.h"

#include "rocksdb/status.h"

#include "common/common_definitions.h"
#include "core/transaction.h"
#include "handler/ha_tianmu.h"
#include "index/rdb_meta_manager.h"

namespace Tianmu {
namespace index {

// Here, In future, tianmu maybe has uniquekey, secondary key, fulltext, etc. Therefore, a type of index should be
// added. But, now. we only have only one index: PK.
TianmuTableIndex::TianmuTableIndex(const std::string &name, TABLE *table) {
  std::string fullname;
  // normalize the table name.
  NormalizeName(name, fullname);

  // does the table exists now.
  KVStore *store = (reinterpret_cast<core::Engine *>(tianmu_hton->data))->getStore();
  rocksdb_tbl_ = store->FindTable(fullname);
  TIANMU_LOG(LogCtl_Level::WARN, "normalize tablename %s, table_full_name %s!", name.c_str(), fullname.c_str());

  keyid_ = table->s->primary_key;
  rocksdb_key_ = rocksdb_tbl_->GetRdbTableKeys().at(store->pk_index(table, rocksdb_tbl_));
  // compatible version that primary key make up of one part
  if (table->key_info[keyid_].actual_key_parts == 1)
    index_of_columns_.push_back(table->key_info[keyid_].key_part[0].field->field_index);
  else
    rocksdb_key_->get_key_cols(index_of_columns_);
}

bool TianmuTableIndex::FindIndexTable(const std::string &name) {
  std::string str;
  if (!NormalizeName(name, str)) {
    throw common::Exception("Normalization wrong of table  " + name);
  }

  KVStore *store = (reinterpret_cast<core::Engine *>(tianmu_hton->data))->getStore();
  if (store->FindTable(str)) {
    return true;
  }

  return false;
}

common::ErrorCode TianmuTableIndex::CreateIndexTable(const std::string &name, TABLE *table) {
  std::string str;
  if (!NormalizeName(name, str)) {
    throw common::Exception("Normalization wrong of table  " + name);
  }

  // ref from mariadb: https://mariadb.com/kb/en/uniqueness-within-a-columnstore-table
  // ColumnStore like many other analytical database engines does not support unique constraints.
  // This helps with performance and scaling out to much larger volumes than innodb supports.
  // It is assumed that your data preparation / ETL phase will ensure correct data being fed into columnstore.
  if (table->s->keys > 1) {
    TIANMU_LOG(LogCtl_Level::WARN, "Table :%s have other keys except primary key, only use primary key!", name.data());
  }

  //  Create table/key descriptions and put them into the data dictionary
  std::shared_ptr<RdbTable> tbl = std::make_shared<RdbTable>(str);
  tbl->GetRdbTableKeys().resize(table->s->keys);

  KVStore *store = (reinterpret_cast<core::Engine *>(tianmu_hton->data))->getStore();
  if (store->create_keys_and_cf(table, tbl) != common::ErrorCode::SUCCESS) {
    return common::ErrorCode::FAILED;
  }

  return store->KVWriteTableMeta(tbl);
}

common::ErrorCode TianmuTableIndex::DropIndexTable(const std::string &name) {
  std::string str;
  if (!NormalizeName(name, str)) {
    throw common::Exception("Exception: table name  " + name);
  }

  KVStore *store = (reinterpret_cast<core::Engine *>(tianmu_hton->data))->getStore();
  //  Find the table in the hash
  return store->KVDelTableMeta(str);
}

common::ErrorCode TianmuTableIndex::RefreshIndexTable(const std::string &name) {
  std::string fullname;

  if (!NormalizeName(name, fullname)) {
    return common::ErrorCode::FAILED;
  }

  KVStore *store = (reinterpret_cast<core::Engine *>(tianmu_hton->data))->getStore();
  rocksdb_tbl_ = store->FindTable(fullname);
  if (rocksdb_tbl_ == nullptr) {
    TIANMU_LOG(LogCtl_Level::WARN, "table %s init ddl error", fullname.c_str());
    return common::ErrorCode::FAILED;
  }

  rocksdb_key_ = rocksdb_tbl_->GetRdbTableKeys().at(keyid_);
  return common::ErrorCode::SUCCESS;
}

common::ErrorCode TianmuTableIndex::RenameIndexTable(const std::string &from, const std::string &to) {
  std::string sname, dname;

  if (!NormalizeName(from, sname)) {
    return common::ErrorCode::FAILED;
  }

  if (!NormalizeName(to, dname)) {
    return common::ErrorCode::FAILED;
  }

  KVStore *store = (reinterpret_cast<core::Engine *>(tianmu_hton->data))->getStore();
  return store->KVRenameTableMeta(sname, dname);
}

void TianmuTableIndex::TruncateIndexTable() {
  rocksdb::WriteOptions wopts;
  rocksdb::ReadOptions ropts;
  ropts.total_order_seek = true;
  uchar key_buf[INDEX_NUMBER_SIZE];
  KVStore *store = (reinterpret_cast<core::Engine *>(tianmu_hton->data))->getStore();

  for (auto &rocksdb_key_ : rocksdb_tbl_->GetRdbTableKeys()) {
    be_store_index(key_buf, rocksdb_key_->get_gl_index_id().index_id);
    auto cf = rocksdb_key_->get_cf();
    std::unique_ptr<rocksdb::Iterator> it(store->GetScanIter(ropts, cf));
    it->Seek({(const char *)key_buf, INDEX_NUMBER_SIZE});

    while (it->Valid()) {
      auto key = it->key();
      if (!rocksdb_key_->covers_key(key)) {
        break;
      }

      if (!store->KVDeleteKey(wopts, cf, key)) {
        throw common::Exception("Rdb delete key fail!");
      }
      it->Next();
    }
  }
}

common::ErrorCode TianmuTableIndex::CheckUniqueness(core::Transaction *tx, const rocksdb::Slice &pk_slice) {
  std::string retrieved_value;

  rocksdb::Status s = tx->KVTrans().Get(rocksdb_key_->get_cf(), pk_slice, &retrieved_value);

  if (s.IsBusy() && !s.IsDeadlock()) {
    tx->KVTrans().Releasesnapshot();
    tx->KVTrans().Acquiresnapshot();
    s = tx->KVTrans().Get(rocksdb_key_->get_cf(), pk_slice, &retrieved_value);
  }

  if (s.ok()) {  // means that there's  another key existed.
    return common::ErrorCode::DUPP_KEY;
  } else if (s.IsNotFound()) {  // not exist a key.
    return common::ErrorCode::NOT_FOUND_KEY;
  } else  // failed.
    return common::ErrorCode::FAILED;
}

common::ErrorCode TianmuTableIndex::InsertIndex(core::Transaction *tx, std::vector<std::string> &fields, uint64_t row) {
  StringWriter value, key;

  rocksdb_key_->pack_key(key, fields, value);

  // 1)inserts into a PK; 2) insert into without a PK. the return value of `CheckUniqueness` is as following:
  // 1: common::ErrorCode::DUPP_KEY; 2: common::ErrorCode::NOT_FOUND_KEY; 3:common::ErrorCode::FAILED
  common::ErrorCode err_code = CheckUniqueness(tx, {(const char *)key.ptr(), key.length()});
  if (idx_type_ == IndexType::INDEX_TYPE_PRIMARY && err_code == common::ErrorCode::DUPP_KEY) {  // PK
    return err_code;
  } else if (idx_type_ == IndexType::INDEX_TYPE_SECONDARY) {  // do not need uniqueness constrain
  }

  value.write_uint64(row);
  const auto cf = rocksdb_key_->get_cf();
  const auto s =
      tx->KVTrans().Put(cf, {(const char *)key.ptr(), key.length()}, {(const char *)value.ptr(), value.length()});
  if (!s.ok()) {
    TIANMU_LOG(LogCtl_Level::ERROR, "RocksDB: insert key fail!");
    err_code = common::ErrorCode::FAILED;
  } else if (s.ok())
    err_code = common::ErrorCode::SUCCESS;

  return err_code;
}

common::ErrorCode TianmuTableIndex::UpdateIndex(core::Transaction *tx, std::string &nkey, std::string &okey,
                                                uint64_t row) {
  StringWriter new_value, old_value, new_packke, old_packkey;
  std::vector<std::string> old_fields, new_fields;

  old_fields.emplace_back(okey);
  new_fields.emplace_back(nkey);

  rocksdb_key_->pack_key(new_packke, new_fields, new_value);
  rocksdb_key_->pack_key(old_packkey, old_fields, old_value);

  common::ErrorCode err_code = CheckUniqueness(tx, {(const char *)new_packke.ptr(), new_packke.length()});

  if (err_code == common::ErrorCode::DUPP_KEY) {
  } else if (err_code == common::ErrorCode::NOT_FOUND_KEY || err_code == common::ErrorCode::SUCCESS) {
    // gets the old index. then update it.
    common::ErrorCode code = CheckUniqueness(tx, {(const char *)old_packkey.ptr(), old_packkey.length()});
    if (code != common::ErrorCode::SUCCESS) {
      const auto cf = rocksdb_key_->get_cf();
      // delete old index then insert a new one.
      rocksdb::Status ret_del = tx->KVTrans().Delete(cf, {(const char *)old_packkey.ptr(), old_packkey.length()});
      err_code = InsertIndex(tx, new_fields, row);
    }
  }

  return err_code;
}

common::ErrorCode TianmuTableIndex::DeleteIndex(core::Transaction *tx, std::vector<std::string> &fields,
                                                uint64_t row [[maybe_unused]]) {
  StringWriter value, packkey;

  rocksdb_key_->pack_key(packkey, fields, value);
  common::ErrorCode rc = CheckUniqueness(tx, {(const char *)packkey.ptr(), packkey.length()});
  if (rc == common::ErrorCode::DUPP_KEY) {
    const auto cf = rocksdb_key_->get_cf();
    const auto rockdbStatus = tx->KVTrans().Delete(cf, {(const char *)packkey.ptr(), packkey.length()});
    if (!rockdbStatus.ok()) {
      TIANMU_LOG(LogCtl_Level::ERROR, "RockDb: delete key fail!");
      return common::ErrorCode::FAILED;
    }
  }
  return common::ErrorCode::SUCCESS;
}

common::ErrorCode TianmuTableIndex::GetRowByKey(core::Transaction *tx, std::vector<std::string> &fields,
                                                uint64_t &row) {
  std::string value;
  StringWriter packkey, info;
  rocksdb_key_->pack_key(packkey, fields, info);
  rocksdb::Status s =
      tx->KVTrans().Get(rocksdb_key_->get_cf(), {(const char *)packkey.ptr(), packkey.length()}, &value);

  if (!s.IsNotFound() && !s.ok()) {
    return common::ErrorCode::FAILED;
  }

  if (s.IsNotFound())
    return common::ErrorCode::NOT_FOUND_KEY;

  StringReader reader({value.data(), value.length()});
  // ver compatible
  if (rocksdb_key_->GetIndexVersion() > static_cast<uint16_t>(IndexInfoType::INDEX_INFO_VERSION_INITIAL)) {
    uint16_t packlen;
    reader.read_uint16(&packlen);
    reader.read(packlen);
  }
  reader.read_uint64(&row);
  return common::ErrorCode::SUCCESS;
}

void KeyIterator::ScanToKey(std::shared_ptr<TianmuTableIndex> tab, std::vector<std::string> &fields,
                            common::Operator op) {
  if (!tab || !txn_) {
    return;
  }
  StringWriter packkey, info;
  valid = true;
  rocksdb_key_ = tab->rocksdb_key_;
  rocksdb_key_->pack_key(packkey, fields, info);
  rocksdb::Slice key_slice((const char *)packkey.ptr(), packkey.length());

  iter_ = std::shared_ptr<rocksdb::Iterator>(txn_->GetIterator(rocksdb_key_->get_cf(), true));
  switch (op) {
    case common::Operator::O_EQ:  //==
      iter_->Seek(key_slice);
      if (!iter_->Valid() || !rocksdb_key_->value_matches_prefix(iter_->key(), key_slice))
        valid = false;
      break;
    case common::Operator::O_MORE_EQ:  //'>='
      iter_->Seek(key_slice);
      if (!iter_->Valid() || !rocksdb_key_->covers_key(iter_->key()))
        valid = false;
      break;
    case common::Operator::O_MORE:  //'>'
      iter_->Seek(key_slice);
      if (!iter_->Valid() || rocksdb_key_->value_matches_prefix(iter_->key(), key_slice)) {
        if (rocksdb_key_->IsInReverseOrder()) {
          iter_->Prev();
        } else {
          iter_->Next();
        }
      }
      if (!iter_->Valid() || !rocksdb_key_->covers_key(iter_->key()))
        valid = false;
      break;
    default:
      TIANMU_LOG(LogCtl_Level::ERROR, "key not support this op:%d", op);
      valid = false;
      break;
  }
}

void KeyIterator::ScanToEdge(std::shared_ptr<TianmuTableIndex> tab, bool forward) {
  if (!tab || !txn_) {
    return;
  }
  valid = true;
  rocksdb_key_ = tab->rocksdb_key_;
  iter_ = std::shared_ptr<rocksdb::Iterator>(txn_->GetIterator(rocksdb_key_->get_cf(), true));
  std::string key = rocksdb_key_->get_boundary_key(forward);

  rocksdb::Slice key_slice((const char *)key.data(), key.length());

  iter_->Seek(key_slice);
  if (forward) {
    if (!iter_->Valid() || !rocksdb_key_->value_matches_prefix(iter_->key(), key_slice))
      valid = false;
  } else {
    if (!iter_)
      valid = false;
    else {
      iter_->Prev();
      if (!iter_->Valid() || !rocksdb_key_->covers_key(iter_->key()))
        valid = false;
    }
  }
}

KeyIterator &KeyIterator::operator++() {
  if (!iter_ || !iter_->Valid() || !rocksdb_key_) {
    valid = false;
    return *this;
  }

  iter_->Next();
  if (!iter_->Valid() || !rocksdb_key_->covers_key(iter_->key())) {
    valid = false;
    return *this;
  }

  return *this;
}

KeyIterator &KeyIterator::operator--() {
  if (!iter_ || !iter_->Valid() || !rocksdb_key_) {
    valid = false;
    return *this;
  }

  iter_->Prev();
  if (!iter_->Valid() || !rocksdb_key_->covers_key(iter_->key())) {
    valid = false;
    return *this;
  }

  return *this;
}

common::ErrorCode KeyIterator::GetCurKV(std::vector<std::string> &keys, uint64_t &row) {
  StringReader key({iter_->key().data(), iter_->key().size()});
  StringReader value({iter_->value().data(), iter_->value().size()});

  common::ErrorCode ret = rocksdb_key_->unpack_key(key, value, keys);
  value.read_uint64(&row);
  return ret;
}

}  // namespace index
}  // namespace Tianmu
