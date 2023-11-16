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
#include <eventql/util/stdtypes.h>
#include <eventql/util/io/fileutil.h>
#include <eventql/util/io/mmappedfile.h>
#include <eventql/util/io/inputstream.h>
#include <eventql/util/io/outputstream.h>
#include <eventql/util/io/mmappedfile.h>
#include <eventql/util/io/BufferedOutputStream.h>
#include <eventql/util/logging.h>
#include <eventql/db/tablet_index.h>
#include <eventql/io/cstable/cstable_writer.h>

#include "eventql/eventql.h"

namespace eventql {

void LSMTableIndex::write(
    const OrderedMap<SHA1Hash, uint64_t>& map,
    const String& filename) {
  auto os = BufferedOutputStream::fromStream(
      FileOutputStream::openFile(filename));

  os->appendUInt8(0x1);
  os->appendUInt64(map.size());
  for (const auto& p : map) {
    os->write((const char*) p.first.data(),  p.first.size());
    os->appendUInt64(p.second);
  }
}

LSMTableIndex::LSMTableIndex() :
    size_(0),
    data_(nullptr) {}

LSMTableIndex::LSMTableIndex(const String& filename) : LSMTableIndex() {
  load(filename);
}

LSMTableIndex::~LSMTableIndex() {
  if (data_) {
    free(data_);
  }
}

bool LSMTableIndex::load(const String& filename) {
  std::unique_lock<std::mutex> lk(load_mutex_);
  if (data_) {
    return false;
  }

  auto is = FileInputStream::openFile(filename);
  is->readUInt8();
  size_ = is->readUInt64();

  auto data_size = size_ * kSlotSize;
  data_ = malloc(data_size);
  if (!data_) {
    RAISEF(kMallocError, "can't load LSMTableIndex $0 into memory", filename);
  }

  is->readNextBytes(data_, data_size);
  return true;
}

size_t LSMTableIndex::size() const {
  return size_ * kSlotSize;
}

void LSMTableIndex::list(HashMap<SHA1Hash, uint64_t>* map) {
  for (size_t i = 0; i < size_; ++i) {
    SHA1Hash id(getID(i), SHA1Hash::kSize);
    auto ver = *getVersion(i);
    auto& slot = (*map)[id];
    if (ver > slot) {
      slot = ver;
    }
  }
}

void LSMTableIndex::lookup(HashMap<SHA1Hash, uint64_t>* map) {
  if (size_ == 0) {
    return;
  }

  for (auto& p : *map) {
    uint64_t begin = 0;
    uint64_t end = size_ - 1;

    while (begin <= end) {
      uint64_t cur = begin + ((end - begin) / uint64_t(2));
      auto cur_id = getID(cur);
      auto cmp = SHA1::compare(cur_id, p.first.data());

      if (cmp == 0) {
        auto cur_version = *getVersion(cur);

        if (cur_version > p.second) {
          p.second = cur_version;
        }

        break;
      }

      if (cmp < 0) {
        if (cur == end) {
          break;
        }

        begin = cur + 1;
      } else {
        if (cur == 0) {
          break;
        }

        end = cur - 1;
      }
    }
  }
}


} // namespace eventql
