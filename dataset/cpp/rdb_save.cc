// Copyright 2022, DragonflyDB authors.  All rights reserved.
// See LICENSE for licensing terms.
//

#include "server/rdb_save.h"

#include <absl/cleanup/cleanup.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <lz4frame.h>
#include <zstd.h>

#include <jsoncons/json.hpp>
#include <queue>

extern "C" {
#include "redis/intset.h"
#include "redis/listpack.h"
#include "redis/rdb.h"
#include "redis/stream.h"
#include "redis/util.h"
#include "redis/ziplist.h"
#include "redis/zmalloc.h"
#include "redis/zset.h"
}

#include "base/flags.h"
#include "base/logging.h"
#include "core/json_object.h"
#include "core/sorted_map.h"
#include "core/string_map.h"
#include "core/string_set.h"
#include "server/engine_shard_set.h"
#include "server/error.h"
#include "server/main_service.h"
#include "server/rdb_extensions.h"
#include "server/search/doc_index.h"
#include "server/serializer_commons.h"
#include "server/snapshot.h"
#include "util/fibers/simple_channel.h"

ABSL_FLAG(int, compression_mode, 3,
          "set 0 for no compression,"
          "set 1 for single entry lzf compression,"
          "set 2 for multi entry zstd compression on df snapshot and single entry on rdb snapshot,"
          "set 3 for multi entry lz4 compression on df snapshot and single entry on rdb snapshot");
ABSL_FLAG(int, compression_level, 2, "The compression level to use on zstd/lz4 compression");

namespace dfly {

using namespace std;
using base::IoBuf;
using io::Bytes;

namespace {

/* Encodes the "value" argument as integer when it fits in the supported ranges
 * for encoded types. If the function successfully encodes the integer, the
 * representation is stored in the buffer pointer to by "enc" and the string
 * length is returned. Otherwise 0 is returned. */
unsigned EncodeInteger(long long value, uint8_t* enc) {
  if (value >= -(1 << 7) && value <= (1 << 7) - 1) {
    enc[0] = (RDB_ENCVAL << 6) | RDB_ENC_INT8;
    enc[1] = value & 0xFF;
    return 2;
  }

  if (value >= -(1 << 15) && value <= (1 << 15) - 1) {
    enc[0] = (RDB_ENCVAL << 6) | RDB_ENC_INT16;
    enc[1] = value & 0xFF;
    enc[2] = (value >> 8) & 0xFF;
    return 3;
  }

  constexpr long long k31 = (1LL << 31);
  if (value >= -k31 && value <= k31 - 1) {
    enc[0] = (RDB_ENCVAL << 6) | RDB_ENC_INT32;
    enc[1] = value & 0xFF;
    enc[2] = (value >> 8) & 0xFF;
    enc[3] = (value >> 16) & 0xFF;
    enc[4] = (value >> 24) & 0xFF;
    return 5;
  }

  return 0;
}

/* String objects in the form "2391" "-100" without any space and with a
 * range of values that can fit in an 8, 16 or 32 bit signed value can be
 * encoded as integers to save space */
unsigned TryIntegerEncoding(string_view input, uint8_t* dest) {
  long long value;

  /* Check if it's possible to encode this value as a number */
  if (!absl::SimpleAtoi(input, &value))
    return 0;
  absl::AlphaNum alpha(value);

  /* If the number converted back into a string is not identical
   * then it's not possible to encode the string as integer */
  if (alpha.size() != input.size() || alpha.Piece() != input)
    return 0;

  return EncodeInteger(value, dest);
}

constexpr size_t kBufLen = 64_KB;
constexpr size_t kAmask = 4_KB - 1;

}  // namespace

uint8_t RdbObjectType(unsigned type, unsigned compact_enc) {
  switch (type) {
    case OBJ_STRING:
      return RDB_TYPE_STRING;
    case OBJ_LIST:
      if (compact_enc == OBJ_ENCODING_QUICKLIST)
        return RDB_TYPE_LIST_QUICKLIST;
      break;
    case OBJ_SET:
      if (compact_enc == kEncodingIntSet)
        return RDB_TYPE_SET_INTSET;
      else if (compact_enc == kEncodingStrMap || compact_enc == kEncodingStrMap2)
        return RDB_TYPE_SET;
      break;
    case OBJ_ZSET:
      if (compact_enc == OBJ_ENCODING_LISTPACK)
        return RDB_TYPE_ZSET_ZIPLIST;  // we save using the old ziplist encoding.
      else if (compact_enc == OBJ_ENCODING_SKIPLIST)
        return RDB_TYPE_ZSET_2;
      break;
    case OBJ_HASH:
      if (compact_enc == kEncodingListPack)
        return RDB_TYPE_HASH_ZIPLIST;
      else if (compact_enc == kEncodingStrMap2)
        return RDB_TYPE_HASH;
      break;
    case OBJ_STREAM:
      return RDB_TYPE_STREAM_LISTPACKS;
    case OBJ_MODULE:
      return RDB_TYPE_MODULE_2;
    case OBJ_JSON:
      return RDB_TYPE_JSON_OLD;
  }
  LOG(FATAL) << "Unknown encoding " << compact_enc << " for type " << type;
  return 0; /* avoid warning */
}

class CompressorImpl {
 public:
  CompressorImpl() {
    compression_level_ = absl::GetFlag(FLAGS_compression_level);
  }
  virtual ~CompressorImpl() {
    VLOG(1) << "compressed size: " << compressed_size_total_;
    VLOG(1) << "uncompressed size: " << uncompressed_size_total_;
  }
  virtual io::Result<io::Bytes> Compress(io::Bytes data) = 0;

 protected:
  int compression_level_ = 1;
  size_t compressed_size_total_ = 0;
  size_t uncompressed_size_total_ = 0;
  base::PODArray<uint8_t> compr_buf_;
};

class ZstdCompressor : public CompressorImpl {
 public:
  ZstdCompressor() {
    cctx_ = ZSTD_createCCtx();
  }
  ~ZstdCompressor() {
    ZSTD_freeCCtx(cctx_);
  }

  io::Result<io::Bytes> Compress(io::Bytes data);

 private:
  ZSTD_CCtx* cctx_;
  base::PODArray<uint8_t> compr_buf_;
};

io::Result<io::Bytes> ZstdCompressor::Compress(io::Bytes data) {
  size_t buf_size = ZSTD_compressBound(data.size());
  if (compr_buf_.capacity() < buf_size) {
    compr_buf_.reserve(buf_size);
  }
  size_t compressed_size = ZSTD_compressCCtx(cctx_, compr_buf_.data(), compr_buf_.capacity(),
                                             data.data(), data.size(), compression_level_);

  if (ZSTD_isError(compressed_size)) {
    return make_unexpected(error_code{int(compressed_size), generic_category()});
  }
  compressed_size_total_ += compressed_size;
  uncompressed_size_total_ += data.size();
  return io::Bytes(compr_buf_.data(), compressed_size);
}

class Lz4Compressor : public CompressorImpl {
 public:
  Lz4Compressor() {
    lz4_pref_.compressionLevel = compression_level_;
  }

  ~Lz4Compressor() {
  }

  // compress a string of data
  io::Result<io::Bytes> Compress(io::Bytes data);

 private:
  LZ4F_preferences_t lz4_pref_ = LZ4F_INIT_PREFERENCES;
};

io::Result<io::Bytes> Lz4Compressor::Compress(io::Bytes data) {
  lz4_pref_.frameInfo.contentSize = data.size();
  size_t buf_size = LZ4F_compressFrameBound(data.size(), &lz4_pref_);
  if (compr_buf_.capacity() < buf_size) {
    compr_buf_.reserve(buf_size);
  }

  size_t frame_size = LZ4F_compressFrame(compr_buf_.data(), compr_buf_.capacity(), data.data(),
                                         data.size(), &lz4_pref_);
  if (LZ4F_isError(frame_size)) {
    return make_unexpected(error_code{int(frame_size), generic_category()});
  }
  compressed_size_total_ += frame_size;
  uncompressed_size_total_ += data.size();
  return io::Bytes(compr_buf_.data(), frame_size);
}

RdbSerializer::RdbSerializer(CompressionMode compression_mode)
    : mem_buf_{4_KB}, tmp_buf_(nullptr), compression_mode_(compression_mode) {
}

RdbSerializer::~RdbSerializer() {
  VLOG(2) << "compression mode: " << uint32_t(compression_mode_);
  if (compression_stats_) {
    VLOG(2) << "compression not effective: " << compression_stats_->compression_no_effective;
    VLOG(2) << "small string none compression applied: " << compression_stats_->small_str_count;
    VLOG(2) << "compression failed: " << compression_stats_->compression_failed;
    VLOG(2) << "compressed blobs:" << compression_stats_->compressed_blobs;
  }
}

std::error_code RdbSerializer::SaveValue(const PrimeValue& pv) {
  std::error_code ec;
  if (pv.ObjType() == OBJ_STRING) {
    auto opt_int = pv.TryGetInt();
    if (opt_int) {
      ec = SaveLongLongAsString(*opt_int);
    } else {
      ec = SaveString(pv.GetSlice(&tmp_str_));
    }
  } else {
    ec = SaveObject(pv);
  }
  return ec;
}

error_code RdbSerializer::SelectDb(uint32_t dbid) {
  if (dbid == last_entry_db_index_) {
    return error_code{};
  }
  last_entry_db_index_ = dbid;
  uint8_t buf[16];
  buf[0] = RDB_OPCODE_SELECTDB;
  unsigned enclen = WritePackedUInt(dbid, io::MutableBytes{buf}.subspan(1));
  return WriteRaw(Bytes{buf, enclen + 1});
}

// Called by snapshot
io::Result<uint8_t> RdbSerializer::SaveEntry(const PrimeKey& pk, const PrimeValue& pv,
                                             uint64_t expire_ms, DbIndex dbid) {
  DVLOG(3) << "Selecting " << dbid << " previous: " << last_entry_db_index_;
  SelectDb(dbid);

  /* Save the expire time */
  if (expire_ms > 0) {
    uint8_t buf[16] = {RDB_OPCODE_EXPIRETIME_MS};
    absl::little_endian::Store64(buf + 1, expire_ms);
    if (auto ec = WriteRaw(Bytes{buf, 9}); ec)
      return make_unexpected(ec);
  }

  /* Save the key poperties */
  uint32_t df_mask_flags = pk.IsSticky() ? DF_MASK_FLAG_STICKY : 0;
  if (df_mask_flags != 0) {
    uint8_t buf[8] = {RDB_OPCODE_DF_MASK};
    absl::little_endian::Store32(buf + 1, df_mask_flags);
    if (auto ec = WriteRaw(Bytes{buf, 5}); ec)
      return make_unexpected(ec);
  }

  string_view key = pk.GetSlice(&tmp_str_);
  unsigned obj_type = pv.ObjType();
  unsigned encoding = pv.Encoding();
  uint8_t rdb_type = RdbObjectType(obj_type, encoding);

  DVLOG(3) << ((void*)this) << ": Saving key/val start " << key << " in dbid=" << dbid;

  if (auto ec = WriteOpcode(rdb_type); ec)
    return make_unexpected(ec);

  if (auto ec = SaveString(key); ec)
    return make_unexpected(ec);

  if (auto ec = SaveValue(pv); ec) {
    LOG(ERROR) << "Problems saving value for key " << key << " in dbid=" << dbid;
    return make_unexpected(ec);
  }

  return rdb_type;
}

error_code RdbSerializer::SaveObject(const PrimeValue& pv) {
  unsigned obj_type = pv.ObjType();
  CHECK_NE(obj_type, OBJ_STRING);

  if (obj_type == OBJ_LIST) {
    return SaveListObject(pv.AsRObj());
  }

  if (obj_type == OBJ_SET) {
    return SaveSetObject(pv);
  }

  if (obj_type == OBJ_HASH) {
    return SaveHSetObject(pv);
  }

  if (obj_type == OBJ_ZSET) {
    return SaveZSetObject(pv);
  }

  if (obj_type == OBJ_STREAM) {
    return SaveStreamObject(pv.AsRObj());
  }

  if (obj_type == OBJ_JSON) {
    return SaveJsonObject(pv);
  }

  LOG(ERROR) << "Not implemented " << obj_type;
  return make_error_code(errc::function_not_supported);
}

error_code RdbSerializer::SaveListObject(const robj* obj) {
  /* Save a list value */
  DCHECK_EQ(OBJ_ENCODING_QUICKLIST, obj->encoding);
  const quicklist* ql = reinterpret_cast<const quicklist*>(obj->ptr);
  quicklistNode* node = ql->head;
  DVLOG(2) << "Saving list of length " << ql->len;

  RETURN_ON_ERR(SaveLen(ql->len));

  while (node) {
    DVLOG(3) << "QL node (encoding/container/sz): " << node->encoding << "/" << node->container
             << "/" << node->sz;

    if (QL_NODE_IS_PLAIN(node)) {
      if (quicklistNodeIsCompressed(node)) {
        void* data;
        size_t compress_len = quicklistGetLzf(node, &data);

        RETURN_ON_ERR(SaveLzfBlob(Bytes{reinterpret_cast<uint8_t*>(data), compress_len}, node->sz));
      } else {
        RETURN_ON_ERR(SaveString(node->entry, node->sz));
      }
    } else {
      // listpack
      uint8_t* lp = node->entry;
      uint8_t* decompressed = NULL;

      if (quicklistNodeIsCompressed(node)) {
        void* data;
        size_t compress_len = quicklistGetLzf(node, &data);
        decompressed = (uint8_t*)zmalloc(node->sz);

        if (lzf_decompress(data, compress_len, decompressed, node->sz) == 0) {
          /* Someone requested decompress, but we can't decompress.  Not good. */
          zfree(decompressed);
          return make_error_code(errc::illegal_byte_sequence);
        }
        lp = decompressed;
      }

      auto cleanup = absl::MakeCleanup([=] {
        if (decompressed)
          zfree(decompressed);
      });
      RETURN_ON_ERR(SaveListPackAsZiplist(lp));
    }
    node = node->next;
  }
  return error_code{};
}

error_code RdbSerializer::SaveSetObject(const PrimeValue& obj) {
  if (obj.Encoding() == kEncodingStrMap) {
    dict* set = (dict*)obj.RObjPtr();

    RETURN_ON_ERR(SaveLen(dictSize(set)));

    dictIterator* di = dictGetIterator(set);
    dictEntry* de;
    auto cleanup = absl::MakeCleanup([di] { dictReleaseIterator(di); });

    while ((de = dictNext(di)) != NULL) {
      sds ele = (sds)de->key;

      RETURN_ON_ERR(SaveString(string_view{ele, sdslen(ele)}));
    }
  } else if (obj.Encoding() == kEncodingStrMap2) {
    StringSet* set = (StringSet*)obj.RObjPtr();

    RETURN_ON_ERR(SaveLen(set->SizeSlow()));

    for (sds ele : *set) {
      RETURN_ON_ERR(SaveString(string_view{ele, sdslen(ele)}));
    }
  } else {
    CHECK_EQ(obj.Encoding(), kEncodingIntSet);
    intset* is = (intset*)obj.RObjPtr();
    size_t len = intsetBlobLen(is);

    RETURN_ON_ERR(SaveString(string_view{(char*)is, len}));
  }

  return error_code{};
}

error_code RdbSerializer::SaveHSetObject(const PrimeValue& pv) {
  DCHECK_EQ(OBJ_HASH, pv.ObjType());

  if (pv.Encoding() == kEncodingStrMap2) {
    StringMap* string_map = (StringMap*)pv.RObjPtr();

    RETURN_ON_ERR(SaveLen(string_map->SizeSlow()));

    for (const auto& k_v : *string_map) {
      RETURN_ON_ERR(SaveString(string_view{k_v.first, sdslen(k_v.first)}));
      RETURN_ON_ERR(SaveString(string_view{k_v.second, sdslen(k_v.second)}));
    }
  } else {
    CHECK_EQ(kEncodingListPack, pv.Encoding());

    uint8_t* lp = (uint8_t*)pv.RObjPtr();
    RETURN_ON_ERR(SaveListPackAsZiplist(lp));
  }

  return error_code{};
}

error_code RdbSerializer::SaveZSetObject(const PrimeValue& pv) {
  DCHECK_EQ(OBJ_ZSET, pv.ObjType());
  const detail::RobjWrapper* robj_wrapper = pv.GetRobjWrapper();
  if (pv.Encoding() == OBJ_ENCODING_SKIPLIST) {
    detail::SortedMap* zs = (detail::SortedMap*)robj_wrapper->inner_obj();

    RETURN_ON_ERR(SaveLen(zs->Size()));
    std::error_code ec;

    /* We save the skiplist elements from the greatest to the smallest
     * (that's trivial since the elements are already ordered in the
     * skiplist): this improves the load process, since the next loaded
     * element will always be the smaller, so adding to the skiplist
     * will always immediately stop at the head, making the insertion
     * O(1) instead of O(log(N)). */
    zs->Iterate(0, zs->Size(), true, [&](sds ele, double score) {
      ec = SaveString(string_view{ele, sdslen(ele)});
      if (ec)
        return false;
      ec = SaveBinaryDouble(score);
      if (ec)
        return false;
      return true;
    });
  } else {
    CHECK_EQ(pv.Encoding(), unsigned(OBJ_ENCODING_LISTPACK)) << "Unknown zset encoding";
    uint8_t* lp = (uint8_t*)robj_wrapper->inner_obj();
    RETURN_ON_ERR(SaveListPackAsZiplist(lp));
  }

  return error_code{};
}

error_code RdbSerializer::SaveStreamObject(const robj* obj) {
  /* Store how many listpacks we have inside the radix tree. */
  stream* s = (stream*)obj->ptr;
  rax* rax = s->rax_tree;

  RETURN_ON_ERR(SaveLen(raxSize(rax)));

  /* Serialize all the listpacks inside the radix tree as they are,
   * when loading back, we'll use the first entry of each listpack
   * to insert it back into the radix tree. */
  raxIterator ri;
  raxStart(&ri, rax);
  raxSeek(&ri, "^", NULL, 0);
  while (raxNext(&ri)) {
    uint8_t* lp = (uint8_t*)ri.data;
    size_t lp_bytes = lpBytes(lp);
    error_code ec = SaveString((uint8_t*)ri.key, ri.key_len);
    if (ec) {
      raxStop(&ri);
      return ec;
    }

    ec = SaveString(lp, lp_bytes);
    if (ec) {
      raxStop(&ri);
      return ec;
    }
  }
  raxStop(&ri);

  /* Save the number of elements inside the stream. We cannot obtain
   * this easily later, since our macro nodes should be checked for
   * number of items: not a great CPU / space tradeoff. */

  RETURN_ON_ERR(SaveLen(s->length));

  /* Save the last entry ID. */
  RETURN_ON_ERR(SaveLen(s->last_id.ms));
  RETURN_ON_ERR(SaveLen(s->last_id.seq));

  /* The consumer groups and their clients are part of the stream
   * type, so serialize every consumer group. */

  /* Save the number of groups. */
  size_t num_cgroups = s->cgroups ? raxSize(s->cgroups) : 0;
  RETURN_ON_ERR(SaveLen(num_cgroups));

  if (num_cgroups) {
    /* Serialize each consumer group. */
    raxStart(&ri, s->cgroups);
    raxSeek(&ri, "^", NULL, 0);

    auto cleanup = absl::MakeCleanup([&] { raxStop(&ri); });

    while (raxNext(&ri)) {
      streamCG* cg = (streamCG*)ri.data;

      /* Save the group name. */
      RETURN_ON_ERR(SaveString((uint8_t*)ri.key, ri.key_len));

      /* Last ID. */
      RETURN_ON_ERR(SaveLen(s->last_id.ms));

      RETURN_ON_ERR(SaveLen(s->last_id.seq));

      /* Save the global PEL. */
      RETURN_ON_ERR(SaveStreamPEL(cg->pel, true));

      /* Save the consumers of this group. */

      RETURN_ON_ERR(SaveStreamConsumers(cg));
    }
  }

  return error_code{};
}

error_code RdbSerializer::SaveJsonObject(const PrimeValue& pv) {
  auto json_string = pv.GetJson()->to_string();
  return SaveString(json_string);
}

/* Save a long long value as either an encoded string or a string. */
error_code RdbSerializer::SaveLongLongAsString(int64_t value) {
  uint8_t buf[32];
  unsigned enclen = EncodeInteger(value, buf);
  if (enclen > 0) {
    return WriteRaw(Bytes{buf, enclen});
  }

  /* Encode as string */
  enclen = ll2string((char*)buf, 32, value);
  DCHECK_LT(enclen, 32u);

  RETURN_ON_ERR(SaveLen(enclen));
  return WriteRaw(Bytes{buf, enclen});
}

/* Saves a double for RDB 8 or greater, where IE754 binary64 format is assumed.
 * We just make sure the integer is always stored in little endian, otherwise
 * the value is copied verbatim from memory to disk.
 *
 * Return -1 on error, the size of the serialized value on success. */
error_code RdbSerializer::SaveBinaryDouble(double val) {
  static_assert(sizeof(val) == 8);
  const uint64_t* src = reinterpret_cast<const uint64_t*>(&val);
  uint8_t buf[8];
  absl::little_endian::Store64(buf, *src);

  return WriteRaw(Bytes{buf, sizeof(buf)});
}

error_code RdbSerializer::SaveListPackAsZiplist(uint8_t* lp) {
  uint8_t* lpfield = lpFirst(lp);
  int64_t entry_len;
  uint8_t* entry;
  uint8_t buf[32];
  uint8_t* zl = ziplistNew();

  while (lpfield) {
    entry = lpGet(lpfield, &entry_len, buf);
    zl = ziplistPush(zl, entry, entry_len, ZIPLIST_TAIL);
    lpfield = lpNext(lp, lpfield);
  }
  size_t ziplen = ziplistBlobLen(zl);
  error_code ec = SaveString(string_view{reinterpret_cast<char*>(zl), ziplen});
  zfree(zl);

  return ec;
}

error_code RdbSerializer::SaveStreamPEL(rax* pel, bool nacks) {
  /* Number of entries in the PEL. */

  RETURN_ON_ERR(SaveLen(raxSize(pel)));

  /* Save each entry. */
  raxIterator ri;
  raxStart(&ri, pel);
  raxSeek(&ri, "^", NULL, 0);
  auto cleanup = absl::MakeCleanup([&] { raxStop(&ri); });

  while (raxNext(&ri)) {
    /* We store IDs in raw form as 128 big big endian numbers, like
     * they are inside the radix tree key. */
    RETURN_ON_ERR(WriteRaw(Bytes{ri.key, sizeof(streamID)}));

    if (nacks) {
      streamNACK* nack = (streamNACK*)ri.data;
      uint8_t buf[8];
      absl::little_endian::Store64(buf, nack->delivery_time);
      RETURN_ON_ERR(WriteRaw(buf));
      RETURN_ON_ERR(SaveLen(nack->delivery_count));

      /* We don't save the consumer name: we'll save the pending IDs
       * for each consumer in the consumer PEL, and resolve the consumer
       * at loading time. */
    }
  }

  return error_code{};
}

error_code RdbSerializer::SaveStreamConsumers(streamCG* cg) {
  /* Number of consumers in this consumer group. */

  RETURN_ON_ERR(SaveLen(raxSize(cg->consumers)));

  /* Save each consumer. */
  raxIterator ri;
  raxStart(&ri, cg->consumers);
  raxSeek(&ri, "^", NULL, 0);
  auto cleanup = absl::MakeCleanup([&] { raxStop(&ri); });
  uint8_t buf[8];

  while (raxNext(&ri)) {
    streamConsumer* consumer = (streamConsumer*)ri.data;

    /* Consumer name. */
    RETURN_ON_ERR(SaveString(ri.key, ri.key_len));

    /* Last seen time. */
    absl::little_endian::Store64(buf, consumer->seen_time);
    RETURN_ON_ERR(WriteRaw(buf));

    /* Consumer PEL, without the ACKs (see last parameter of the function
     * passed with value of 0), at loading time we'll lookup the ID
     * in the consumer group global PEL and will put a reference in the
     * consumer local PEL. */

    RETURN_ON_ERR(SaveStreamPEL(consumer->pel, false));
  }

  return error_code{};
}

error_code RdbSerializer::SendJournalOffset(uint64_t journal_offset) {
  VLOG(2) << "SendJournalOffset";
  RETURN_ON_ERR(WriteOpcode(RDB_OPCODE_JOURNAL_OFFSET));
  uint8_t buf[sizeof(uint64_t)];
  absl::little_endian::Store64(buf, journal_offset);
  return WriteRaw(buf);
}

error_code RdbSerializer::SendFullSyncCut() {
  VLOG(2) << "SendFullSyncCut";
  RETURN_ON_ERR(WriteOpcode(RDB_OPCODE_FULLSYNC_END));

  // RDB_OPCODE_FULLSYNC_END followed by 8 bytes of 0.
  // The reason for this is that some opcodes require to have at least 8 bytes of data
  // in the read buffer when consuming the rdb data, and since RDB_OPCODE_FULLSYNC_END is one of
  // the last opcodes sent to replica, we respect this requirement by sending a blob of 8 bytes.
  uint8_t buf[8] = {0};
  return WriteRaw(buf);
}

size_t RdbSerializer::GetTotalBufferCapacity() const {
  return mem_buf_.Capacity();
}

error_code RdbSerializer::WriteRaw(const io::Bytes& buf) {
  mem_buf_.Reserve(mem_buf_.InputLen() + buf.size());
  IoBuf::Bytes dest = mem_buf_.AppendBuffer();
  memcpy(dest.data(), buf.data(), buf.size());
  mem_buf_.CommitWrite(buf.size());
  return error_code{};
}

error_code RdbSerializer::FlushToSink(io::Sink* s) {
  auto bytes = PrepareFlush();
  if (bytes.empty())
    return error_code{};

  DVLOG(2) << "FlushToSink " << bytes.size() << " bytes";

  // interrupt point.
  RETURN_ON_ERR(s->Write(bytes));
  mem_buf_.ConsumeInput(bytes.size());
  // After every flush we should write the DB index again because the blobs in the channel are
  // interleaved and multiple savers can correspond to a single writer (in case of single file rdb
  // snapshot)
  last_entry_db_index_ = kInvalidDbId;

  return error_code{};
}

size_t RdbSerializer::SerializedLen() const {
  return mem_buf_.InputLen();
}

io::Bytes RdbSerializer::PrepareFlush() {
  size_t sz = mem_buf_.InputLen();
  if (sz == 0)
    return mem_buf_.InputBuffer();

  if (compression_mode_ == CompressionMode::MULTY_ENTRY_ZSTD ||
      compression_mode_ == CompressionMode::MULTY_ENTRY_LZ4) {
    CompressBlob();
  }

  return mem_buf_.InputBuffer();
}

error_code RdbSerializer::WriteJournalEntry(std::string_view serialized_entry) {
  VLOG(2) << "WriteJournalEntry";
  RETURN_ON_ERR(WriteOpcode(RDB_OPCODE_JOURNAL_BLOB));
  RETURN_ON_ERR(SaveLen(1));
  RETURN_ON_ERR(SaveString(serialized_entry));
  return error_code{};
}

error_code RdbSerializer::SaveString(string_view val) {
  /* Try integer encoding */
  if (val.size() <= 11) {
    uint8_t buf[16];

    unsigned enclen = TryIntegerEncoding(val, buf);
    if (enclen > 0) {
      return WriteRaw(Bytes{buf, unsigned(enclen)});
    }
  }

  /* Try LZF compression - under 20 bytes it's unable to compress even
   * aaaaaaaaaaaaaaaaaa so skip it */
  size_t len = val.size();
  if ((compression_mode_ == CompressionMode::SINGLE_ENTRY) && (len > 20)) {
    size_t comprlen, outlen = len;
    tmp_buf_.resize(outlen + 1);

    // Due to stack constraints im fibers we can not allow large arrays on stack.
    // Therefore I am lazily allocating it on heap. It's not fixed in quicklist.
    if (!lzf_) {
      lzf_.reset(new LZF_HSLOT[1 << HLOG]);
    }

    /* We require at least 8 bytes compression for this to be worth it */
    comprlen = lzf_compress(val.data(), len, tmp_buf_.data(), outlen, lzf_.get());
    if (comprlen > 0 && comprlen < len - 8 && comprlen < size_t(len * 0.85)) {
      return SaveLzfBlob(Bytes{tmp_buf_.data(), comprlen}, len);
    }
  }

  /* Store verbatim */
  RETURN_ON_ERR(SaveLen(len));
  if (len > 0) {
    Bytes b{reinterpret_cast<const uint8_t*>(val.data()), val.size()};
    RETURN_ON_ERR(WriteRaw(b));
  }
  return error_code{};
}

error_code RdbSerializer::SaveLen(size_t len) {
  uint8_t buf[16];
  unsigned enclen = WritePackedUInt(len, buf);
  return WriteRaw(Bytes{buf, enclen});
}

error_code RdbSerializer::SaveLzfBlob(const io::Bytes& src, size_t uncompressed_len) {
  /* Data compressed! Let's save it on disk */
  uint8_t opcode = (RDB_ENCVAL << 6) | RDB_ENC_LZF;
  RETURN_ON_ERR(WriteOpcode(opcode));
  RETURN_ON_ERR(SaveLen(src.size()));
  RETURN_ON_ERR(SaveLen(uncompressed_len));
  RETURN_ON_ERR(WriteRaw(src));

  return error_code{};
}

AlignedBuffer::AlignedBuffer(size_t cap, ::io::Sink* upstream)
    : capacity_(cap), upstream_(upstream) {
  aligned_buf_ = (char*)mi_malloc_aligned(kBufLen, 4_KB);
}

AlignedBuffer::~AlignedBuffer() {
  mi_free(aligned_buf_);
}

io::Result<size_t> AlignedBuffer::WriteSome(const iovec* v, uint32_t len) {
  size_t total_len = 0;
  uint32_t vindx = 0;

  for (; vindx < len; ++vindx) {
    auto item = v[vindx];
    total_len += item.iov_len;

    while (buf_offs_ + item.iov_len > capacity_) {
      size_t to_write = capacity_ - buf_offs_;
      memcpy(aligned_buf_ + buf_offs_, item.iov_base, to_write);
      iovec ivec{.iov_base = aligned_buf_, .iov_len = capacity_};
      error_code ec = upstream_->Write(&ivec, 1);
      if (ec)
        return nonstd::make_unexpected(ec);

      item.iov_len -= to_write;
      item.iov_base = reinterpret_cast<char*>(item.iov_base) + to_write;
      buf_offs_ = 0;
    }

    DCHECK_GT(item.iov_len, 0u);
    memcpy(aligned_buf_ + buf_offs_, item.iov_base, item.iov_len);
    buf_offs_ += item.iov_len;
  }

  return total_len;
}

// Note that it may write more than AlignedBuffer has at this point since it rounds up the length
// to the nearest page boundary.
error_code AlignedBuffer::Flush() {
  size_t len = (buf_offs_ + kAmask) & (~kAmask);
  if (len == 0)
    return error_code{};

  iovec ivec{.iov_base = aligned_buf_, .iov_len = len};
  buf_offs_ = 0;

  return upstream_->Write(&ivec, 1);
}

class RdbSaver::Impl {
 private:
  // This is a helper struct to pop records from channel while enfocing returned records order.
  struct RecordsPopper {
    RecordsPopper(bool enforce_order, SliceSnapshot::RecordChannel* c)
        : enforce_order(enforce_order), channel(c) {
    }

    // Blocking function, pops from channel.
    // If enforce_order is enabled return the records by order.
    // returns nullopt if channel was closed.
    std::optional<SliceSnapshot::DbRecord> Pop();

    // Non blocking function, trys to pop from channel.
    // If enforce_order is enabled return the records by order.
    // returns nullopt if nothing in channel.
    std::optional<SliceSnapshot::DbRecord> TryPop();

   private:
    std::optional<SliceSnapshot::DbRecord> InternalPop(bool blocking);
    // Checks if next record is in queue, if so set record_holder and return true, otherwise
    // return false.
    bool TryPopFromQueue();

    struct Compare {
      bool operator()(const SliceSnapshot::DbRecord& a, const SliceSnapshot::DbRecord& b) {
        return a.id > b.id;
      }
    };
    // min heap holds the DbRecord that poped from channel OOO
    std::priority_queue<SliceSnapshot::DbRecord, std::vector<SliceSnapshot::DbRecord>, Compare>
        q_records;
    uint64_t next_record_id = 0;
    bool enforce_order;
    SliceSnapshot::RecordChannel* channel;
    SliceSnapshot::DbRecord record_holder;
  };

 public:
  // We pass K=sz to say how many producers are pushing data in order to maintain
  // correct closing semantics - channel is closing when K producers marked it as closed.
  Impl(bool align_writes, unsigned producers_len, CompressionMode compression_mode,
       SaveMode save_mode, io::Sink* sink);

  void StartSnapshotting(bool stream_journal, const Cancellation* cll, EngineShard* shard);
  void StartIncrementalSnapshotting(Context* cntx, EngineShard* shard, LSN start_lsn);

  void StopSnapshotting(EngineShard* shard);

  error_code ConsumeChannel(const Cancellation* cll);

  void FillFreqMap(RdbTypeFreqMap* dest) const;

  error_code SaveAuxFieldStrStr(string_view key, string_view val);

  void Cancel();

  size_t GetTotalBuffersSize() const;

  error_code Flush() {
    return aligned_buf_ ? aligned_buf_->Flush() : error_code{};
  }

  size_t Size() const {
    return shard_snapshots_.size();
  }

  RdbSerializer* serializer() {
    return &meta_serializer_;
  }

  io::Sink* sink() {
    return sink_;
  }

 private:
  unique_ptr<SliceSnapshot>& GetSnapshot(EngineShard* shard);

  io::Sink* sink_;
  vector<unique_ptr<SliceSnapshot>> shard_snapshots_;
  // used for serializing non-body components in the calling fiber.
  RdbSerializer meta_serializer_;
  SliceSnapshot::RecordChannel channel_;
  bool push_to_sink_with_order_ = false;
  std::optional<AlignedBuffer> aligned_buf_;

  // Single entry compression is compatible with redis rdb snapshot
  // Multi entry compression is available only on df snapshot, this will
  // make snapshot size smaller and opreation faster.
  CompressionMode compression_mode_;
};

// We pass K=sz to say how many producers are pushing data in order to maintain
// correct closing semantics - channel is closing when K producers marked it as closed.
RdbSaver::Impl::Impl(bool align_writes, unsigned producers_len, CompressionMode compression_mode,
                     SaveMode sm, io::Sink* sink)
    : sink_(sink),
      shard_snapshots_(producers_len),
      meta_serializer_(CompressionMode::NONE),  // Note: I think there is not need for compression
                                                // at all in meta serializer
      channel_{128, producers_len},
      compression_mode_(compression_mode) {
  if (align_writes) {
    aligned_buf_.emplace(kBufLen, sink);
    sink_ = &aligned_buf_.value();
  }
  if (sm == SaveMode::SINGLE_SHARD || sm == SaveMode::SINGLE_SHARD_WITH_SUMMARY) {
    push_to_sink_with_order_ = true;
  }

  DCHECK(producers_len > 0 || channel_.IsClosing());
}

error_code RdbSaver::Impl::SaveAuxFieldStrStr(string_view key, string_view val) {
  auto& ser = meta_serializer_;
  RETURN_ON_ERR(ser.WriteOpcode(RDB_OPCODE_AUX));
  RETURN_ON_ERR(ser.SaveString(key));
  RETURN_ON_ERR(ser.SaveString(val));

  return error_code{};
}

bool RdbSaver::Impl::RecordsPopper::TryPopFromQueue() {
  if (enforce_order && !q_records.empty() && q_records.top().id == next_record_id) {
    record_holder = std::move(const_cast<SliceSnapshot::DbRecord&>(q_records.top()));
    q_records.pop();
    ++next_record_id;
    return true;
  }
  return false;
}

std::optional<SliceSnapshot::DbRecord> RdbSaver::Impl::RecordsPopper::Pop() {
  return InternalPop(true);
}

std::optional<SliceSnapshot::DbRecord> RdbSaver::Impl::RecordsPopper::TryPop() {
  return InternalPop(false);
}

std::optional<SliceSnapshot::DbRecord> RdbSaver::Impl::RecordsPopper::InternalPop(bool blocking) {
  if (TryPopFromQueue()) {
    return std::move(record_holder);
  }

  auto pop_fn =
      blocking ? &SliceSnapshot::RecordChannel::Pop : &SliceSnapshot::RecordChannel::TryPop;

  while ((channel->*pop_fn)(record_holder)) {
    if (!enforce_order) {
      return std::move(record_holder);
    }
    if (record_holder.id == next_record_id) {
      ++next_record_id;
      return std::move(record_holder);
    }
    // record popped from channel is ooo, push to queue
    q_records.emplace(std::move(record_holder));
  }
  return std::nullopt;
}

error_code RdbSaver::Impl::ConsumeChannel(const Cancellation* cll) {
  error_code io_error;
  std::optional<SliceSnapshot::DbRecord> record;

  RecordsPopper records_popper(push_to_sink_with_order_, &channel_);

  // we can not exit on io-error since we spawn fibers that push data.
  // TODO: we may signal them to stop processing and exit asap in case of the error.
  while ((record = records_popper.Pop())) {
    if (io_error || cll->IsCancelled())
      continue;

    do {
      if (cll->IsCancelled())
        continue;

      DVLOG(2) << "Pulled " << record->id;
      io_error = sink_->Write(io::Buffer(record->value));
      if (io_error) {
        break;
      }
    } while ((record = records_popper.TryPop()));
  }  // while (records_popper.Pop())

  for (auto& ptr : shard_snapshots_) {
    ptr->Join();
  }

  DCHECK(!record.has_value() || !channel_.TryPop(*record));

  return io_error;
}

void RdbSaver::Impl::StartSnapshotting(bool stream_journal, const Cancellation* cll,
                                       EngineShard* shard) {
  auto& s = GetSnapshot(shard);
  s = std::make_unique<SliceSnapshot>(&shard->db_slice(), &channel_, compression_mode_);

  s->Start(stream_journal, cll);
}

void RdbSaver::Impl::StartIncrementalSnapshotting(Context* cntx, EngineShard* shard,
                                                  LSN start_lsn) {
  auto& s = GetSnapshot(shard);
  s = std::make_unique<SliceSnapshot>(&shard->db_slice(), &channel_, compression_mode_);

  s->StartIncremental(cntx, start_lsn);
}

void RdbSaver::Impl::StopSnapshotting(EngineShard* shard) {
  GetSnapshot(shard)->Stop();
}

void RdbSaver::Impl::Cancel() {
  auto* shard = EngineShard::tlocal();
  if (!shard)
    return;

  auto& snapshot = GetSnapshot(shard);
  if (snapshot)
    snapshot->Cancel();

  dfly::SliceSnapshot::DbRecord rec;
  while (channel_.Pop(rec)) {
  }

  snapshot->Join();
}

// This function is called from connection thread when info command is invoked.
// All accessed variableds must be thread safe, as they are fetched not from the rdb saver thread.
size_t RdbSaver::Impl::GetTotalBuffersSize() const {
  std::atomic<size_t> channel_bytes{0};
  std::atomic<size_t> serializer_bytes{0};

  auto cb = [this, &channel_bytes, &serializer_bytes](ShardId sid) {
    auto& snapshot = shard_snapshots_[sid];
    channel_bytes.fetch_add(snapshot->GetTotalChannelCapacity(), memory_order_relaxed);
    serializer_bytes.store(snapshot->GetTotalBufferCapacity(), memory_order_relaxed);
  };

  if (shard_snapshots_.size() == 1) {
    cb(0);
  } else {
    shard_set->RunBriefInParallel([&](EngineShard* es) { cb(es->shard_id()); });
  }

  VLOG(2) << "channel_bytes:" << channel_bytes.load(memory_order_relaxed)
          << " serializer_bytes: " << serializer_bytes.load(memory_order_relaxed);
  return channel_bytes.load(memory_order_relaxed) + serializer_bytes.load(memory_order_relaxed);
}

RdbSaver::GlobalData RdbSaver::GetGlobalData(const Service* service) {
  StringVec script_bodies, search_indices;

  {
    auto scripts = service->script_mgr()->GetAll();
    script_bodies.reserve(scripts.size());
    for (auto& [sha, data] : scripts)
      script_bodies.push_back(move(data.body));
  }

#ifndef __APPLE__
  {
    shard_set->Await(0, [&] {
      auto* indices = EngineShard::tlocal()->search_indices();
      for (auto index_name : indices->GetIndexNames()) {
        auto index_info = indices->GetIndex(index_name)->GetInfo();
        search_indices.emplace_back(
            absl::StrCat(index_name, " ", index_info.BuildRestoreCommand()));
      }
    });
  }
#endif

  return RdbSaver::GlobalData{std::move(script_bodies), std::move(search_indices)};
}

void RdbSaver::Impl::FillFreqMap(RdbTypeFreqMap* dest) const {
  for (auto& ptr : shard_snapshots_) {
    const RdbTypeFreqMap& src_map = ptr->freq_map();
    for (const auto& k_v : src_map)
      (*dest)[k_v.first] += k_v.second;
  }
}

unique_ptr<SliceSnapshot>& RdbSaver::Impl::GetSnapshot(EngineShard* shard) {
  // For single shard configuration, we maintain only one snapshot,
  // so we do not have to map it via shard_id.
  unsigned sid = shard_snapshots_.size() == 1 ? 0 : shard->shard_id();
  CHECK(sid < shard_snapshots_.size());
  return shard_snapshots_[sid];
}

RdbSaver::RdbSaver(::io::Sink* sink, SaveMode save_mode, bool align_writes) {
  CHECK_NOTNULL(sink);
  int compression_mode = absl::GetFlag(FLAGS_compression_mode);
  int producer_count = 0;
  switch (save_mode) {
    case SaveMode::SUMMARY:
      producer_count = 0;
      if (compression_mode >= 1) {
        compression_mode_ = CompressionMode::SINGLE_ENTRY;
      } else {
        compression_mode_ = CompressionMode::NONE;
      }
      break;
    case SaveMode::SINGLE_SHARD:
    case SaveMode::SINGLE_SHARD_WITH_SUMMARY:
      producer_count = 1;
      if (compression_mode == 3) {
        compression_mode_ = CompressionMode::MULTY_ENTRY_LZ4;
      } else if (compression_mode == 2) {
        compression_mode_ = CompressionMode::MULTY_ENTRY_ZSTD;
      } else if (compression_mode == 1) {
        compression_mode_ = CompressionMode::SINGLE_ENTRY;
      } else {
        compression_mode_ = CompressionMode::NONE;
      }
      break;
    case SaveMode::RDB:
      producer_count = shard_set->size();
      if (compression_mode >= 1) {
        compression_mode_ = CompressionMode::SINGLE_ENTRY;
      } else {
        compression_mode_ = CompressionMode::NONE;
      }
      break;
  }
  VLOG(1) << "Rdb save using compression mode:" << uint32_t(compression_mode_);
  impl_.reset(new Impl(align_writes, producer_count, compression_mode_, save_mode, sink));
  save_mode_ = save_mode;
}

RdbSaver::~RdbSaver() {
}

void RdbSaver::StartSnapshotInShard(bool stream_journal, const Cancellation* cll,
                                    EngineShard* shard) {
  impl_->StartSnapshotting(stream_journal, cll, shard);
}

void RdbSaver::StartIncrementalSnapshotInShard(Context* cntx, EngineShard* shard, LSN start_lsn) {
  impl_->StartIncrementalSnapshotting(cntx, shard, start_lsn);
}

void RdbSaver::StopSnapshotInShard(EngineShard* shard) {
  impl_->StopSnapshotting(shard);
}

error_code RdbSaver::SaveHeader(const GlobalData& glob_state) {
  char magic[16];
  // We should use RDB_VERSION here from rdb.h when we ditch redis 6 support
  // For now we serialize to an older version.
  size_t sz = absl::SNPrintF(magic, sizeof(magic), "REDIS%04d", RDB_SER_VERSION);
  CHECK_EQ(9u, sz);

  RETURN_ON_ERR(impl_->serializer()->WriteRaw(Bytes{reinterpret_cast<uint8_t*>(magic), sz}));
  RETURN_ON_ERR(SaveAux(std::move(glob_state)));

  return error_code{};
}

error_code RdbSaver::SaveBody(Context* cntx, RdbTypeFreqMap* freq_map) {
  RETURN_ON_ERR(impl_->serializer()->FlushToSink(impl_->sink()));

  if (save_mode_ == SaveMode::SUMMARY) {
    impl_->serializer()->SendFullSyncCut();
  } else {
    VLOG(1) << "SaveBody , snapshots count: " << impl_->Size();
    error_code io_error = impl_->ConsumeChannel(cntx->GetCancellation());
    if (io_error) {
      LOG(ERROR) << "io error " << io_error;
      return io_error;
    }
    if (cntx->GetError()) {
      return cntx->GetError();
    }
  }

  RETURN_ON_ERR(SaveEpilog());

  if (freq_map) {
    freq_map->clear();
    impl_->FillFreqMap(freq_map);
  }

  return error_code{};
}

error_code RdbSaver::SaveAux(const GlobalData& glob_state) {
  static_assert(sizeof(void*) == 8, "");

  int aof_preamble = false;
  error_code ec;

  /* Add a few fields about the state when the RDB was created. */
  RETURN_ON_ERR(impl_->SaveAuxFieldStrStr("redis-ver", REDIS_VERSION));
  RETURN_ON_ERR(SaveAuxFieldStrInt("redis-bits", 64));

  RETURN_ON_ERR(SaveAuxFieldStrInt("ctime", time(NULL)));
  RETURN_ON_ERR(SaveAuxFieldStrInt("used-mem", used_mem_current.load(memory_order_relaxed)));
  RETURN_ON_ERR(SaveAuxFieldStrInt("aof-preamble", aof_preamble));

  // Save lua scripts only in rdb or summary file
  DCHECK(save_mode_ != SaveMode::SINGLE_SHARD || glob_state.lua_scripts.empty());
  for (const string& s : glob_state.lua_scripts)
    RETURN_ON_ERR(impl_->SaveAuxFieldStrStr("lua", s));

  if (save_mode_ == SaveMode::RDB) {
    if (!glob_state.search_indices.empty())
      LOG(WARNING) << "Dragonfly search index data is incompatible with the RDB format";
  } else {
    // Search index definitions are not tied to shards and are saved in the summary file
    DCHECK(save_mode_ != SaveMode::SINGLE_SHARD || glob_state.search_indices.empty());
    for (const string& s : glob_state.search_indices)
      RETURN_ON_ERR(impl_->SaveAuxFieldStrStr("search-index", s));
  }

  // TODO: "repl-stream-db", "repl-id", "repl-offset"
  return error_code{};
}

error_code RdbSaver::SaveEpilog() {
  uint8_t buf[8];
  uint64_t chksum;

  auto& ser = *impl_->serializer();

  /* EOF opcode */
  RETURN_ON_ERR(ser.WriteOpcode(RDB_OPCODE_EOF));

  /* CRC64 checksum. It will be zero if checksum computation is disabled, the
   * loading code skips the check in this case. */
  chksum = 0;

  absl::little_endian::Store64(buf, chksum);
  RETURN_ON_ERR(ser.WriteRaw(buf));

  RETURN_ON_ERR(ser.FlushToSink(impl_->sink()));

  return impl_->Flush();
}

error_code RdbSaver::SaveAuxFieldStrInt(string_view key, int64_t val) {
  char buf[LONG_STR_SIZE];
  int vlen = ll2string(buf, sizeof(buf), val);
  return impl_->SaveAuxFieldStrStr(key, string_view(buf, vlen));
}

void RdbSaver::Cancel() {
  impl_->Cancel();
}

size_t RdbSaver::GetTotalBuffersSize() const {
  return impl_->GetTotalBuffersSize();
}

void RdbSerializer::AllocateCompressorOnce() {
  if (compressor_impl_) {
    return;
  }
  if (compression_mode_ == CompressionMode::MULTY_ENTRY_ZSTD) {
    compressor_impl_.reset(new ZstdCompressor());
  } else if (compression_mode_ == CompressionMode::MULTY_ENTRY_LZ4) {
    compressor_impl_.reset(new Lz4Compressor());
  } else {
    CHECK(false) << "Compressor allocation should not be done";
  }
}

void RdbSerializer::CompressBlob() {
  if (!compression_stats_) {
    compression_stats_.emplace(CompressionStats{});
  }
  Bytes blob_to_compress = mem_buf_.InputBuffer();
  size_t blob_size = blob_to_compress.size();
  if (blob_size < kMinStrSizeToCompress) {
    ++compression_stats_->small_str_count;
    return;
  }

  AllocateCompressorOnce();
  // Compress the data
  auto ec = compressor_impl_->Compress(blob_to_compress);
  if (!ec) {
    ++compression_stats_->compression_failed;
    return;
  }
  Bytes compressed_blob = *ec;
  if (compressed_blob.length() > blob_size * kMinCompressionReductionPrecentage) {
    ++compression_stats_->compression_no_effective;
    return;
  }

  // Clear membuf and write the compressed blob to it
  mem_buf_.ConsumeInput(blob_size);
  mem_buf_.Reserve(compressed_blob.length() + 1 + 9);  // reserve space for blob + opcode + len

  // First write opcode for compressed string
  auto dest = mem_buf_.AppendBuffer();
  uint8_t opcode = compression_mode_ == CompressionMode::MULTY_ENTRY_ZSTD
                       ? RDB_OPCODE_COMPRESSED_ZSTD_BLOB_START
                       : RDB_OPCODE_COMPRESSED_LZ4_BLOB_START;
  dest[0] = opcode;
  mem_buf_.CommitWrite(1);

  // Write encoded compressed blob len
  dest = mem_buf_.AppendBuffer();
  unsigned enclen = WritePackedUInt(compressed_blob.length(), dest);
  mem_buf_.CommitWrite(enclen);

  // Write compressed blob
  dest = mem_buf_.AppendBuffer();
  memcpy(dest.data(), compressed_blob.data(), compressed_blob.length());
  mem_buf_.CommitWrite(compressed_blob.length());
  ++compression_stats_->compressed_blobs;
}

}  // namespace dfly
