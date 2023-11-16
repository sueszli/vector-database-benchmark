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

/*
 This is a part of TianmuAttr implementation concerned with the query execution
 mechanisms
*/

#include "common/assert.h"
#include "core/transaction.h"
#include "core/value_set.h"
#include "data/pack_guardian.h"
#include "data/pack_str.h"
#include "optimizer/compile/cq_term.h"
#include "util/hash64.h"
#include "util/tools.h"
#include "vc/const_column.h"
#include "vc/in_set_column.h"
#include "vc/single_column.h"
#include "vc/tianmu_attr.h"
#include "vc/tianmu_attr_typeinfo.h"

namespace Tianmu {
namespace core {

void TianmuAttr::EvaluatePack(MIUpdatingIterator &mit, int dim, Descriptor &d) {
  MEASURE_FET("TianmuAttr::EvaluatePack(...)");
  ASSERT(d.encoded, "Descriptor is not encoded!");
  if (d.op == common::Operator::O_FALSE) {
    mit.ResetCurrentPack();
    mit.NextPackrow();
  } else if (d.op == common::Operator::O_TRUE)
    mit.NextPackrow();
  else if (d.op == common::Operator::O_NOT_NULL)
    EvaluatePack_NotNull(mit, dim);
  else if (d.op == common::Operator::O_IS_NULL)
    EvaluatePack_IsNull(mit, dim);
  else if (d.val1.vc && !d.val1.vc->IsConst()) {
    if (GetPackType() == common::PackType::INT) {
      if (ATI::IsRealType(TypeName()))
        EvaluatePack_AttrAttrReal(mit, dim, d);
      else
        EvaluatePack_AttrAttr(mit, dim, d);
    } else
      DEBUG_ASSERT(0);  // case not implemented
  } else if (GetPackType() == common::PackType::INT &&
             (d.op == common::Operator::O_BETWEEN || d.op == common::Operator::O_NOT_BETWEEN)) {
    if (!ATI::IsRealType(TypeName()))
      EvaluatePack_BetweenInt(mit, dim, d);
    else
      EvaluatePack_BetweenReal(mit, dim, d);
  } else if (GetPackType() == common::PackType::STR &&
             (d.op == common::Operator::O_BETWEEN || d.op == common::Operator::O_NOT_BETWEEN)) {
    if (types::RequiresUTFConversions(d.GetCollation()))
      EvaluatePack_BetweenString_UTF(mit, dim, d);
    else
      EvaluatePack_BetweenString(mit, dim, d);
  } else if (d.op == common::Operator::O_LIKE || d.op == common::Operator::O_NOT_LIKE) {
    if (types::RequiresUTFConversions(d.GetCollation()))
      EvaluatePack_Like_UTF(mit, dim, d);
    else
      EvaluatePack_Like(mit, dim, d);
  } else if (GetPackType() == common::PackType::STR &&
             (d.op == common::Operator::O_IN || d.op == common::Operator::O_NOT_IN)) {
    if (types::RequiresUTFConversions(d.GetCollation()))
      EvaluatePack_InString_UTF(mit, dim, d);
    else
      EvaluatePack_InString(mit, dim, d);
  } else if (GetPackType() == common::PackType::INT &&
             (d.op == common::Operator::O_IN || d.op == common::Operator::O_NOT_IN))
    EvaluatePack_InNum(mit, dim, d);
  else if (d.op == common::Operator::O_EQ_ALL)
    EvaluatePack_IsNoDelete(mit, dim);
  else
    DEBUG_ASSERT(0);  // case not implemented!
}

// TODO: op (common::Operator::O_LIKE common::Operator::O_IN)
common::ErrorCode TianmuAttr::EvaluateOnIndex(MIUpdatingIterator &mit, int dim, Descriptor &d, int64_t limit) {
  common::ErrorCode rv = common::ErrorCode::FAILED;

  if (GetPackType() == common::PackType::INT &&
      (d.op == common::Operator::O_BETWEEN || d.op == common::Operator::O_NOT_BETWEEN)) {
    rv = EvaluateOnIndex_BetweenInt(mit, dim, d, limit);
  } else if (GetPackType() == common::PackType::STR &&
             (d.op == common::Operator::O_BETWEEN || d.op == common::Operator::O_NOT_BETWEEN)) {
    if (types::RequiresUTFConversions(d.GetCollation()))
      rv = EvaluateOnIndex_BetweenString_UTF(mit, dim, d, limit);
    else
      rv = EvaluateOnIndex_BetweenString(mit, dim, d, limit);
  }

  return rv;
}
common::ErrorCode TianmuAttr::EvaluateOnIndex_BetweenInt(MIUpdatingIterator &mit, int dim, Descriptor &d,
                                                         int64_t limit) {
  common::ErrorCode rv = common::ErrorCode::FAILED;
  core::Engine *eng = reinterpret_cast<core::Engine *>(tianmu_hton->data);
  assert(eng);

  auto indextab = eng->GetTableIndex(m_share->owner->Path());
  if (!indextab)
    return rv;

  int64_t pv1 = d.val1.vc->GetValueInt64(mit);
  int64_t pv2 = d.val2.vc->GetValueInt64(mit);
  auto filter = mit.GetMultiIndex()->GetFilter(dim);
  if (d.op != common::Operator::O_NOT_BETWEEN)
    filter->Reset();

  std::vector<uint> keycols = indextab->KeyCols();
  if (keycols.size() > 0 && keycols[0] == ColId()) {
    int64_t passed = 0;
    index::KeyIterator iter(&current_txn_->KVTrans());
    std::vector<std::string> fields;
    fields.emplace_back((const char *)&pv1, sizeof(int64_t));

    iter.ScanToKey(indextab, fields, common::Operator::O_MORE_EQ);
    while (iter.IsValid()) {
      uint64_t row = 0;
      std::vector<std::string> vkeys;
      rv = iter.GetCurKV(vkeys, row);
      if (common::ErrorCode::SUCCESS == rv) {
        int64_t part1 = *(reinterpret_cast<int64_t *>(vkeys[0].data()));
        bool res = part1 > pv2;
        if (d.op == common::Operator::O_NOT_BETWEEN) {
          // If operator is common::Operator::O_NOT_BETWEEN, only set 0 to the bit by row,
          // limit not use because don't have graceful approach to process the
          // bit(1)
          if (!res)
            filter->Reset(row);
          else
            break;

        } else {
          if (res || (limit != -1 && ++passed > limit))
            break;
          filter->Set(row);
        }

        ++iter;
      } else {
        TIANMU_LOG(LogCtl_Level::ERROR, "GetCurKV valid! col:[%u]=%I64d, Path:%s", ColId(), pv1,
                   m_share->owner->Path().data());
        break;
      }
    }
  }
  // Clear packs not in range
  if (rv == common::ErrorCode::SUCCESS && d.op != common::Operator::O_NOT_BETWEEN) {
    while (mit.IsValid()) {
      if (!filter->GetBlockChangeStatus(mit.GetCurPackrow(dim)))
        mit.ResetCurrentPack();
      mit.NextPackrow();
    }
    mit.Commit();
  }
  return rv;
}

common::ErrorCode TianmuAttr::EvaluateOnIndex_BetweenString(MIUpdatingIterator &mit, int dim, Descriptor &d,
                                                            int64_t limit) {
  common::ErrorCode rv = common::ErrorCode::FAILED;
  core::Engine *eng = reinterpret_cast<core::Engine *>(tianmu_hton->data);
  assert(eng);

  auto indextab = eng->GetTableIndex(m_share->owner->Path());
  if (!indextab)
    return rv;

  types::BString pv1, pv2;
  d.val1.vc->GetValueString(pv1, mit);
  d.val2.vc->GetValueString(pv2, mit);
  auto filter = mit.GetMultiIndex()->GetFilter(dim);
  if (d.op != common::Operator::O_NOT_BETWEEN)
    filter->Reset();

  std::vector<uint> keycols = indextab->KeyCols();
  if (keycols.size() > 0 && keycols[0] == ColId()) {
    int64_t passed = 0;
    index::KeyIterator iter(&current_txn_->KVTrans());
    std::vector<std::string> fields;
    fields.emplace_back(pv1.GetDataBytesPointer(), pv1.size());

    iter.ScanToKey(indextab, fields, (d.sharp ? common::Operator::O_MORE : common::Operator::O_MORE_EQ));
    while (iter.IsValid()) {
      uint64_t row = 0;
      std::vector<std::string> vkeys;
      rv = iter.GetCurKV(vkeys, row);
      if (common::ErrorCode::SUCCESS == rv) {
        types::BString pv(vkeys[0].data(), vkeys[0].length());
        bool res = (d.sharp && ((pv1.IsNull() || pv > pv1) && (pv2.IsNull() || pv < pv2))) ||
                   (!d.sharp && ((pv1.IsNull() || pv >= pv1) && (pv2.IsNull() || pv <= pv2)));

        if (d.op == common::Operator::O_NOT_BETWEEN) {
          // If operator is common::Operator::O_NOT_BETWEEN, only set 0 to the bit by row,
          // limit not use because don't have graceful approach to process the
          // bit(1)
          if (res)
            filter->Reset(row);
          else
            break;

        } else {
          if (!res || (limit != -1 && ++passed > limit))
            break;
          filter->Set(row);
        }

        ++iter;
      } else {
        TIANMU_LOG(LogCtl_Level::ERROR, "GetCurKV valid! col:[%u]=%s, Path:%s", ColId(), pv1.ToString().data(),
                   m_share->owner->Path().data());
        break;
      }
    }
  }
  // Clear packs not in range
  if (rv == common::ErrorCode::SUCCESS && d.op != common::Operator::O_NOT_BETWEEN) {
    while (mit.IsValid()) {
      if (!filter->GetBlockChangeStatus(mit.GetCurPackrow(dim)))
        mit.ResetCurrentPack();
      mit.NextPackrow();
    }
    mit.Commit();
  }
  return rv;
}

common::ErrorCode TianmuAttr::EvaluateOnIndex_BetweenString_UTF(MIUpdatingIterator &mit, int dim, Descriptor &d,
                                                                int64_t limit) {
  common::ErrorCode rv = common::ErrorCode::FAILED;
  core::Engine *eng = reinterpret_cast<core::Engine *>(tianmu_hton->data);
  assert(eng);

  auto indextab = eng->GetTableIndex(m_share->owner->Path());
  if (!indextab)
    return rv;

  types::BString pv1, pv2;
  d.val1.vc->GetValueString(pv1, mit);
  d.val2.vc->GetValueString(pv2, mit);
  auto filter = mit.GetMultiIndex()->GetFilter(dim);
  if (d.op != common::Operator::O_NOT_BETWEEN)
    filter->Reset();

  std::vector<uint> keycols = indextab->KeyCols();
  if (keycols.size() > 0 && keycols[0] == ColId()) {
    int64_t passed = 0;
    index::KeyIterator iter(&current_txn_->KVTrans());
    std::vector<std::string> fields;
    fields.emplace_back(pv1.GetDataBytesPointer(), pv1.size());
    iter.ScanToKey(indextab, fields, (d.sharp ? common::Operator::O_MORE : common::Operator::O_MORE_EQ));
    DTCollation coll = d.GetCollation();
    while (iter.IsValid()) {
      uint64_t row = 0;
      std::vector<std::string> vkeys;
      rv = iter.GetCurKV(vkeys, row);
      if (common::ErrorCode::SUCCESS == rv) {
        types::BString pv(vkeys[0].data(), vkeys[0].length());
        bool res = (d.sharp && ((pv1.IsNull() || CollationStrCmp(coll, pv, pv1) > 0) &&
                                (pv2.IsNull() || CollationStrCmp(coll, pv, pv2) < 0))) ||
                   (!d.sharp && ((pv1.IsNull() || CollationStrCmp(coll, pv, pv1) >= 0) &&
                                 (pv2.IsNull() || CollationStrCmp(coll, pv, pv2) <= 0)));

        if (d.op == common::Operator::O_NOT_BETWEEN) {
          // If operator is common::Operator::O_NOT_BETWEEN, only set 0 to the bit by row,
          // limit not use because don't have graceful approach to process the
          // bit(1)
          if (res)
            filter->Reset(row);
          else
            break;

        } else {
          if (!res || (limit != -1 && ++passed > limit))
            break;
          filter->Set(row);
        }

        ++iter;
      } else {
        TIANMU_LOG(LogCtl_Level::ERROR, "GetCurKV valid! col:[%u]=%s, Path:%s", ColId(), pv1.ToString().data(),
                   m_share->owner->Path().data());
        break;
      }
    }
  }
  // Clear packs not in range
  if (rv == common::ErrorCode::SUCCESS && d.op != common::Operator::O_NOT_BETWEEN) {
    while (mit.IsValid()) {
      if (!filter->GetBlockChangeStatus(mit.GetCurPackrow(dim)))
        mit.ResetCurrentPack();
      mit.NextPackrow();
    }
    mit.Commit();
  }
  return rv;
}

void TianmuAttr::EvaluatePack_IsNoDelete(MIUpdatingIterator &mit, int dim) {
  MEASURE_FET("TianmuAttr::EvaluatePack_IsNoDelete(...)");
  // Keep consistent with the method of other functions to obtain the pack index
  auto pack = mit.GetCurPackrow(dim);
  if (pack == -1) {
    mit.ResetCurrentPack();
    mit.NextPackrow();
    return;
  }
  auto const &dpn(get_dpn(pack));
  if (dpn.numOfDeleted > 0) {
    if (dpn.numOfDeleted == dpn.numOfRecords) {
      mit.ResetCurrentPack();
      mit.NextPackrow();
      return;
    }
    FunctionExecutor fe([this, pack]() { LockPackForUse(pack); }, [this, pack]() { UnlockPackFromUse(pack); });
    do {
      if (mit[dim] == common::NULL_VALUE_64 || get_pack(pack)->IsDeleted(mit.GetCurInpack(dim))) {
        mit.ResetCurrent();
      }
      ++mit;
    } while (mit.IsValid() && !mit.PackrowStarted());
  } else {
    mit.NextPackrow();
  }
}

void TianmuAttr::EvaluatePack_IsNull(MIUpdatingIterator &mit, int dim) {
  MEASURE_FET("TianmuAttr::EvaluatePack_IsNull(...)");
  int pack = mit.GetCurPackrow(dim);
  if (pack == -1) {
    EvaluatePack_IsNoDelete(mit, dim);
    return;
  }
  auto const &dpn(get_dpn(pack));
  if (!dpn.Trivial() && dpn.numOfNulls != 0) {  // nontrivial pack exists
    do {
      if ((mit[dim] != common::NULL_VALUE_64 && !get_pack(pack)->IsNull(mit.GetCurInpack(dim))) ||
          get_pack(pack)->IsDeleted(mit.GetCurInpack(dim))) {
        mit.ResetCurrent();
      }
      ++mit;
    } while (mit.IsValid() && !mit.PackrowStarted());
  } else {  // pack is trivial - uniform or null only
    if (GetPackOntologicalStatus(pack) != PackOntologicalStatus::NULLS_ONLY) {
      if (mit.NullsPossibleInPack(dim)) {
        do {
          if (mit[dim] != common::NULL_VALUE_64 || get_pack(pack)->IsDeleted(mit.GetCurInpack(dim)))
            mit.ResetCurrent();
          ++mit;
        } while (mit.IsValid() && !mit.PackrowStarted());
      } else
        mit.ResetCurrentPack();
    }
    mit.NextPackrow();
  }
}

void TianmuAttr::EvaluatePack_NotNull(MIUpdatingIterator &mit, int dim) {
  MEASURE_FET("TianmuAttr::EvaluatePack_NotNull(...)");
  int pack = mit.GetCurPackrow(dim);
  if (pack == -1) {  // nulls only
    mit.ResetCurrentPack();
    mit.NextPackrow();
    return;
  }
  auto const &dpn(get_dpn(pack));
  if (!dpn.Trivial() && dpn.numOfNulls != 0) {
    do {
      if (mit[dim] == common::NULL_VALUE_64 || get_pack(pack)->IsNull(mit.GetCurInpack(dim)))
        mit.ResetCurrent();
      ++mit;
    } while (mit.IsValid() && !mit.PackrowStarted());
  } else {  // pack is trivial - uniform or null only
    if (GetPackOntologicalStatus(pack) == PackOntologicalStatus::NULLS_ONLY)
      mit.ResetCurrentPack();
    else if (mit.NullsPossibleInPack(dim)) {
      do {
        if (mit[dim] == common::NULL_VALUE_64)
          mit.ResetCurrent();
        ++mit;
      } while (mit.IsValid() && !mit.PackrowStarted());
    }
    mit.NextPackrow();
  }
}

bool IsSpecialChar(char c, Descriptor &d) { return c == '%' || c == '_' || c == d.like_esc; }

void TianmuAttr::EvaluatePack_Like(MIUpdatingIterator &mit, int dim, Descriptor &d) {
  MEASURE_FET("TianmuAttr::EvaluatePack_Like(...)");
  int pack = mit.GetCurPackrow(dim);
  if (pack == -1) {
    mit.ResetCurrentPack();
    mit.NextPackrow();
    return;
  }
  auto p = get_packS(pack);
  if (p == nullptr) {  // => nulls only
    mit.ResetCurrentPack();
    mit.NextPackrow();
    return;
  }
  types::BString pattern;
  d.val1.vc->GetValueString(pattern, mit);
  size_t min_len = 0;  // the number of fixed characters
  for (uint i = 0; i < pattern.len_; i++) {
    if (pattern[i] != '%')
      min_len++;
    if (pattern[i] == d.like_esc) {  // disable optimization, escape character
                                     // may do a lot of mess
      min_len = 0;
      break;
    }
  }
  std::unordered_set<uint16_t> possible_ids;
  bool use_trie = false;
  bool pure_prefix = false;
  if (!pattern.IsNullOrEmpty() && !IsSpecialChar(pattern[0], d) && p->IsTrie()) {
    auto first_wildcard = pattern.begin();
    std::size_t prefixlen = 0;
    while (first_wildcard != pattern.end() && !IsSpecialChar(*first_wildcard, d)) {
      first_wildcard++;
      prefixlen++;
    }
    use_trie = p->LikePrefix(pattern, prefixlen, possible_ids);
    if (possible_ids.empty()) {
      mit.ResetCurrentPack();
      mit.NextPackrow();
      return;
    }
    if (first_wildcard == pattern.end() || (*first_wildcard == '%' && (++first_wildcard) == pattern.end()))
      pure_prefix = true;
  }
  do {
    int inpack = mit.GetCurInpack(dim);
    if (mit[dim] == common::NULL_VALUE_64 || p->IsNull(inpack)) {
      mit.ResetCurrent();
    } else if (use_trie && p->IsNotMatched(inpack, possible_ids)) {
      mit.ResetCurrent();
    } else if (pure_prefix) {
      // The query is something like 'Pattern%' or 'Pattern', so
      // if p->IsNotMatched() == false, then this is a match and
      // there is no need to check it here.
    } else {
      types::BString v(p->GetValueBinary(inpack));
      auto len = v.size();
      bool res;
      if (len < min_len)
        res = false;
      else {
        v.MakePersistent();
        res = v.Like(pattern, d.like_esc);
      }
      if (d.op == common::Operator::O_NOT_LIKE)
        res = !res;
      if (!res)
        mit.ResetCurrent();
    }
    ++mit;
  } while (mit.IsValid() && !mit.PackrowStarted());
}

void TianmuAttr::EvaluatePack_Like_UTF(MIUpdatingIterator &mit, int dim, Descriptor &d) {
  MEASURE_FET("TianmuAttr::EvaluatePack_Like_UTF(...)");
  int pack = mit.GetCurPackrow(dim);
  if (pack == -1) {
    mit.ResetCurrentPack();
    mit.NextPackrow();
    return;
  }
  auto p = get_packS(pack);
  if (p == nullptr) {  // => nulls only
    mit.ResetCurrentPack();
    mit.NextPackrow();
    return;
  }
  types::BString pattern;
  d.val1.vc->GetValueString(pattern, mit);
  size_t min_len = 0;  // the number of fixed characters
  /*
    When calculating the length of the matching string,
    need to exclude wildcard characters and characters in ESCAPE
  */
  for (uint i = 0; i < pattern.len_; i++)
    if (pattern[i] != '%' && pattern[i] != '\\' && pattern[i] != d.like_esc)
      min_len++;
  std::unordered_set<uint16_t> possible_ids;
  bool use_trie = false;
  bool pure_prefix = false;
  if (!pattern.IsNullOrEmpty() && !IsSpecialChar(pattern[0], d) && p->IsTrie()) {
    auto first_wildcard = pattern.begin();
    std::size_t prefixlen = 0;
    while (first_wildcard != pattern.end() && !IsSpecialChar(*first_wildcard, d)) {
      first_wildcard++;
      prefixlen++;
    }
    use_trie = p->LikePrefix(pattern, prefixlen, possible_ids);
    if (possible_ids.empty()) {
      mit.ResetCurrentPack();
      mit.NextPackrow();
      return;
    }
    if (first_wildcard == pattern.end() || (*first_wildcard == '%' && (++first_wildcard) == pattern.end()))
      pure_prefix = true;
  }
  do {
    int inpack = mit.GetCurInpack(dim);
    if (mit[dim] == common::NULL_VALUE_64 || p->IsNull(inpack)) {
      mit.ResetCurrent();
    } else if (use_trie && p->IsNotMatched(inpack, possible_ids)) {
      mit.ResetCurrent();
    } else if (pure_prefix) {
      // The query is something like 'Pattern%' or 'Pattern', so
      // if p->IsNotMatched() == false, then this is a match and
      // there is no need to check it here.
    } else {
      types::BString v(p->GetValueBinary(inpack));
      auto len = v.size();
      bool res;
      if (len < min_len)
        res = false;
      else {
        v.MakePersistent();
        int x = common::wildcmp(d.GetCollation(), v.val_, v.val_ + v.len_, pattern.val_, pattern.val_ + pattern.len_,
                                d.like_esc, '_', '%');
        res = (x == 0 ? true : false);
      }
      if (d.op == common::Operator::O_NOT_LIKE)
        res = !res;
      if (!res)
        mit.ResetCurrent();
    }
    ++mit;
  } while (mit.IsValid() && !mit.PackrowStarted());
}

void TianmuAttr::EvaluatePack_InString(MIUpdatingIterator &mit, int dim, Descriptor &d) {
  MEASURE_FET("TianmuAttr::EvaluatePack_InString(...)");
  int pack = mit.GetCurPackrow(dim);
  if (pack == -1) {
    mit.ResetCurrentPack();
    mit.NextPackrow();
    return;
  }
  auto p = get_packS(pack);
  if (p == nullptr) {  // => nulls only
    mit.ResetCurrentPack();
    mit.NextPackrow();
    return;
  }
  DEBUG_ASSERT(dynamic_cast<vcolumn::MultiValColumn *>(d.val1.vc) != nullptr);
  vcolumn::MultiValColumn *multival_column = static_cast<vcolumn::MultiValColumn *>(d.val1.vc);
  bool encoded_set = multival_column->IsSetEncoded(TypeName(), ct.GetScale());
  do {
    int inpack = mit.GetCurInpack(dim);
    if (mit[dim] == common::NULL_VALUE_64 || p->IsNull(inpack))
      mit.ResetCurrent();
    else {
      common::Tribool res;
      types::BString s;
      s.PersistentCopy(p->GetValueBinary(inpack));
      if (encoded_set)  // fast path for numerics vs. encoded constant set
        res = multival_column->ContainsString(mit, s);
      else
        res = multival_column->Contains(mit, s);
      if (d.op == common::Operator::O_NOT_IN)
        res = !res;
      if (res != true)
        mit.ResetCurrent();
    }
    if (current_txn_->Killed())
      throw common::KilledException();
    ++mit;
  } while (mit.IsValid() && !mit.PackrowStarted());
}

void TianmuAttr::EvaluatePack_InString_UTF(MIUpdatingIterator &mit, int dim, Descriptor &d) {
  MEASURE_FET("TianmuAttr::EvaluatePack_InString_UTF(...)");
  int pack = mit.GetCurPackrow(dim);
  if (pack == -1) {
    mit.ResetCurrentPack();
    mit.NextPackrow();
    return;
  }

  auto p = get_packS(pack);
  if (p == nullptr) {  // => nulls only
    mit.ResetCurrentPack();
    mit.NextPackrow();
    return;
  }

  DEBUG_ASSERT(dynamic_cast<vcolumn::MultiValColumn *>(d.val1.vc) != nullptr);
  vcolumn::MultiValColumn *multival_column = static_cast<vcolumn::MultiValColumn *>(d.val1.vc);
  DTCollation coll = d.GetCollation();
  int arraysize = d.val1.cond_value.size();
  do {
    int inpack = mit.GetCurInpack(dim);
    if (mit[dim] == common::NULL_VALUE_64 || p->IsNull(inpack))
      mit.ResetCurrent();
    else {
      common::Tribool res = false;
      types::BString vt(p->GetValueBinary(inpack));  //, true
      if (arraysize > 0 && arraysize < 10) {
        for (auto &it : d.val1.cond_value) {
          if (coll.collation->coll->strnncoll(coll.collation, (const uchar *)it.val_, it.len_, (const uchar *)vt.val_,
                                              vt.len_, 0) == 0) {
            res = true;
            break;
          }
        }
      } else {
        res = multival_column->Contains(mit, vt);
      }

      if (d.op == common::Operator::O_NOT_IN)
        res = !res;
      if (res != true)
        mit.ResetCurrent();
    }
    if (current_txn_->Killed())
      throw common::KilledException();
    ++mit;
  } while (mit.IsValid() && !mit.PackrowStarted());
}

void TianmuAttr::EvaluatePack_InNum(MIUpdatingIterator &mit, int dim, Descriptor &d) {
  MEASURE_FET("TianmuAttr::EvaluatePack_InNum(...)");
  int pack = mit.GetCurPackrow(dim);
  if (pack == -1) {
    mit.ResetCurrentPack();
    mit.NextPackrow();
    return;
  }

  // added trivial case due to OR tree
  if (get_dpn(pack).NullOnly()) {
    mit.ResetCurrentPack();
    mit.NextPackrow();
    return;
  }

  auto &dpn = get_dpn(pack);
  auto p = get_packN(pack);
  int64_t local_min = dpn.min_i;
  int64_t local_max = dpn.max_i;

  DEBUG_ASSERT(dynamic_cast<vcolumn::MultiValColumn *>(d.val1.vc) != nullptr);
  vcolumn::MultiValColumn *multival_column = static_cast<vcolumn::MultiValColumn *>(d.val1.vc);
  bool lookup_to_num = ATI::IsStringType(TypeName());
  bool encoded_set = (lookup_to_num ? multival_column->IsSetEncoded(common::ColumnType::NUM, 0)
                                    : multival_column->IsSetEncoded(TypeName(), ct.GetScale()));
  common::Tribool res;
  std::unique_ptr<types::TianmuDataType> value(ValuePrototype(lookup_to_num).Clone());
  bool not_in = (d.op == common::Operator::O_NOT_IN);
  int arraysize = 0;
  if (d.val1.cond_numvalue != nullptr)
    arraysize = d.val1.cond_numvalue->capacity();
  if (local_min == local_max) {
    if (GetPackOntologicalStatus(pack) == PackOntologicalStatus::NULLS_ONLY) {
      mit.ResetCurrentPack();
      mit.NextPackrow();
    } else {
      // only nulls and single value
      // pack does not exist
      do {
        if (IsNull(mit[dim]))
          mit.ResetCurrent();
        else {
          // find the first non-null and set the rest basing on it.
          // const TianmuValueObject& val = GetValue(mit[dim], lookup_to_num);
          // note: res may be UNKNOWN for NOT IN (...null...)
          res = multival_column->Contains(mit, GetValueData(mit[dim], *value, lookup_to_num));
          if (not_in)
            res = !res;
          if (res == true) {
            if (dpn.numOfNulls != 0)
              EvaluatePack_NotNull(mit, dim);
            else
              mit.NextPackrow();
          } else {
            mit.ResetCurrentPack();
            mit.NextPackrow();
          }
          break;
        }
        ++mit;
      } while (mit.IsValid() && !mit.PackrowStarted());
    }
  } else {
    do {
      if (mit[dim] == common::NULL_VALUE_64 || p->IsNull(mit.GetCurInpack(dim)))
        mit.ResetCurrent();
      else {
        if (arraysize > 0 && arraysize < 100) {
          res = false;
          int64_t val = GetNotNullValueInt64(mit[dim]);
          res = d.val1.cond_numvalue->Find(val);

        } else {
          // note: res may be UNKNOWN for NOT IN (...null...)
          if (encoded_set)  // fast path for numerics vs. encoded constant set
            res = multival_column->Contains64(mit, GetNotNullValueInt64(mit[dim]));
          else
            res = multival_column->Contains(mit, GetValueData(mit[dim], *value, lookup_to_num));
        }
        if (not_in)
          res = !res;
        if (res != true)
          mit.ResetCurrent();
      }
      ++mit;
    } while (mit.IsValid() && !mit.PackrowStarted());
    if (current_txn_->Killed())
      throw common::KilledException();
  }
}

void TianmuAttr::EvaluatePack_BetweenString(MIUpdatingIterator &mit, int dim, Descriptor &d) {
  MEASURE_FET("TianmuAttr::EvaluatePack_BetweenString(...)");
  int pack = mit.GetCurPackrow(dim);
  if (pack == -1) {
    mit.ResetCurrentPack();
    mit.NextPackrow();
    return;
  }
  // added trivial case due to OR tree
  if (get_dpn(pack).NullOnly()) {
    mit.ResetCurrentPack();
    mit.NextPackrow();
    return;
  }

  auto p = get_packS(pack);
  if (p == nullptr) {  // => nulls only
    mit.ResetCurrentPack();
    mit.NextPackrow();
    return;
  }
  types::BString v1, v2;
  d.val1.vc->GetValueString(v1, mit);
  d.val2.vc->GetValueString(v2, mit);
  bool use_trie = false;
  uint16_t trie_id;
  if (v1 == v2 && p->IsTrie()) {
    use_trie = p->Lookup(v1, trie_id);
    if (!use_trie) {
      mit.ResetCurrentPack();
      mit.NextPackrow();
      return;
    }
  }
  do {
    int inpack = mit.GetCurInpack(dim);  // row number inside the pack
    if (mit[dim] == common::NULL_VALUE_64 || p->IsNull(inpack)) {
      mit.ResetCurrent();
    } else if (use_trie) {
      if (p->IsNotMatched(inpack, trie_id))
        mit.ResetCurrent();
    } else {
      types::BString v(p->GetValueBinary(inpack));  // change to materialized in case
                                                    // of problems, but the pack should
                                                    // be locked and unchanged here
      // IsNull() below means +/-inf
      bool res = (d.sharp && ((v1.IsNull() || v > v1) && (v2.IsNull() || v < v2))) ||
                 (!d.sharp && ((v1.IsNull() || v >= v1) && (v2.IsNull() || v <= v2)));
      if (d.op == common::Operator::O_NOT_BETWEEN)
        res = !res;
      if (!res)
        mit.ResetCurrent();
    }
    ++mit;
  } while (mit.IsValid() && !mit.PackrowStarted());
}

void TianmuAttr::EvaluatePack_BetweenString_UTF(MIUpdatingIterator &mit, int dim, Descriptor &d) {
  MEASURE_FET("TianmuAttr::EvaluatePack_BetweenString_UTF(...)");
  int pack = mit.GetCurPackrow(dim);
  if (pack == -1) {
    mit.ResetCurrentPack();
    mit.NextPackrow();
    return;
  }
  // added trivial case due to OR tree
  if (get_dpn(pack).NullOnly()) {
    mit.ResetCurrentPack();
    mit.NextPackrow();
    return;
  }

  auto p = get_packS(pack);
  if (p == nullptr) {  // => nulls only
    mit.ResetCurrentPack();
    mit.NextPackrow();
    return;
  }
  types::BString v1, v2;
  DTCollation coll = d.GetCollation();
  d.val1.vc->GetValueString(v1, mit);
  d.val2.vc->GetValueString(v2, mit);
  bool use_trie = false;
  uint16_t trie_id;
  if (v1 == v2 && p->IsTrie()) {
    use_trie = p->Lookup(v1, trie_id);
    if (!use_trie) {
      mit.ResetCurrentPack();
      mit.NextPackrow();
      return;
    }
  }

  auto check_item_digital = ([&](CQTerm &val) {
    return ((nullptr != val.item) &&
            (val.item->type() == Item::Type::INT_ITEM || (val.item->type() == Item::Type::DECIMAL_ITEM) ||
             (val.item->type() == Item::Type::REAL_ITEM) ||
             (val.item->type() == Item::Type::FUNC_ITEM &&
              down_cast<Item_func *>(val.item)->functype() == Item_func::Functype::NEG_FUNC)));
  });

  const bool v1_is_digital = check_item_digital(d.val1);
  const bool v2_is_digital = check_item_digital(d.val2);

  do {
    int inpack = mit.GetCurInpack(dim);  // row number inside the pack
    if (mit[dim] == common::NULL_VALUE_64 || p->IsNull(inpack)) {
      mit.ResetCurrent();
    } else if (use_trie) {
      if (p->IsNotMatched(inpack, trie_id))
        mit.ResetCurrent();
    } else {
      // change to materialized in case
      // of problems, but the pack should
      // be locked and unchanged here
      types::BString v(p->GetValueBinary(inpack));

      // IsNull() below means +/-inf
      bool res =
          (d.sharp &&
           ((v1.IsNull() || ((v1_is_digital ? CollationRealCmp(coll, v, v1) : CollationStrCmp(coll, v, v1)) > 0)) &&
            (v2.IsNull() || ((v2_is_digital ? CollationRealCmp(coll, v, v2) : CollationStrCmp(coll, v, v2)) < 0)))) ||
          (!d.sharp &&
           ((v1.IsNull() || ((v1_is_digital ? CollationRealCmp(coll, v, v1) : CollationStrCmp(coll, v, v1)) >= 0)) &&
            (v2.IsNull() || ((v2_is_digital ? CollationRealCmp(coll, v, v2) : CollationStrCmp(coll, v, v2)) <= 0))));

      if (d.op == common::Operator::O_NOT_BETWEEN)
        res = !res;
      if (!res)
        mit.ResetCurrent();
    }
    ++mit;
  } while (mit.IsValid() && !mit.PackrowStarted());
}

void TianmuAttr::EvaluatePack_BetweenInt(MIUpdatingIterator &mit, int dim, Descriptor &d) {
  MEASURE_FET("TianmuAttr::EvaluatePack_BetweenInt(...)");
  int pack = mit.GetCurPackrow(dim);
  if (pack == -1) {
    mit.ResetCurrentPack();
    mit.NextPackrow();
    return;
  }

  auto &dpn = get_dpn(pack);

  // added trivial case due to OR tree
  if (dpn.NullOnly()) {
    mit.ResetCurrentPack();
    mit.NextPackrow();
    return;
  }
  int64_t pv1 = d.val1.vc->GetValueInt64(mit);  // e.g.: between pv1 and pv2
  int64_t pv2 = d.val2.vc->GetValueInt64(mit);  // e.g.: between pv1 and pv2
  uint64_t upv1, upv2;                          // used uint64_t to store level 2 encode, ranges from [0, uint64_max]
  // special case(pv2 < local_min): table has both numerics and null values,for between pv1 and pv2, if pv1<local_min,
  // rough check = RS_ALL(no numeric data matchs), but table has null, rough check will be reset RS_SOME,
  // so function EvaluatePack_BetweenInt() still be called to filter nulls, but no data matchs. This is a tmp solution,
  // we may find a better way to deal with this case. Used with upv2.
  int64_t local_min = dpn.min_i;
  int64_t local_max = dpn.max_i;
  if (common::MINUS_INF_64 == pv1 || pv1 <= local_min)
    // we reserve common::MINUS_INF_64 case for "where a < 3" tranform to "[MINUS_INF_64, 2]".
    upv1 = 0;
  else
    upv1 = pv1 - local_min;

  if (common::PLUS_INF_64 == pv2 || pv2 >= local_max)
    // case: local_min < local_max <= pv2, we also reserve PLUS_INF_64 for special case like pv1
    upv2 = local_max - local_min;
  else
    // case: local_min <= pv2 < local_max or pv2 < local_min
    upv2 = pv2 - local_min;

  if (local_min != local_max) {
    auto p = get_packN(pack);
    auto filter = mit.GetMultiIndex()->GetFilter(dim);
    // MIIterator iteratoring costs too much
    // Loop without it when packs are nearly full
    if (tianmu_sysvar_filterevaluation_speedup && filter &&
        filter->NumOfOnes(pack) > static_cast<uint>(1 << (mit.GetPower() - 1))) {
      if (d.op == common::Operator::O_BETWEEN && !mit.NullsPossibleInPack(dim) && dpn.numOfNulls == 0) {
        // easy and fast case - no "if"s
        for (uint32_t n = 0; n < dpn.numOfRecords; n++) {
          uint64_t v = p->GetValInt(n);
          if ((pv2 < local_min) || v < upv1 || v > upv2)
            filter->Reset(pack, n);
        }
      } else {
        // more general case
        for (uint32_t n = 0; n < dpn.numOfRecords; n++) {
          if (unlikely(p->IsNull(n)))
            filter->Reset(pack, n);
          else {
            uint64_t v = p->GetValInt(n);
            bool res = (pv2 < local_min) ? false : (upv1 <= v && v <= upv2);
            if (d.op == common::Operator::O_NOT_BETWEEN)
              res = !res;
            if (!res)
              filter->Reset(pack, n);
          }
        }
      }
      mit.NextPackrow();
    } else {
      if (d.op == common::Operator::O_BETWEEN && !mit.NullsPossibleInPack(dim) && dpn.numOfNulls == 0) {
        // easy and fast case - no "if"s
        do {
          uint64_t v = p->GetValInt(mit.GetCurInpack(dim));
          if ((pv2 < local_min) || v < upv1 || v > upv2)
            mit.ResetCurrent();
          ++mit;
        } while (mit.IsValid() && !mit.PackrowStarted());
      } else {
        // more general case
        do {  // e.g.: table has null, DELETE FROM cs1 WHERE d1 NOT IN (-125); --> not between: (-125, -125)
          int inpack = mit.GetCurInpack(dim);
          if (mit[dim] == common::NULL_VALUE_64 || p->IsNull(inpack))
            mit.ResetCurrent();
          else {
            uint64_t v = p->GetValInt(inpack);
            bool res = ((pv2 < local_min) ? false : upv1 <= v && v <= upv2);
            if (d.op == common::Operator::O_NOT_BETWEEN)
              res = !res;
            if (!res)
              mit.ResetCurrent();
          }
          ++mit;
        } while (mit.IsValid() && !mit.PackrowStarted());
      }
    }
  } else {  // when local_min = local_max && has null in table, execute: select * from c where a not in (-125);
    // local_min==local_max, and in 2-level encoding both are 0, upv1 = upv2 = 0.
    if (((upv1 != 0 || upv2 != 0) && d.op == common::Operator::O_BETWEEN) ||
        (upv1 == 0 && upv2 == 0 && d.op == common::Operator::O_NOT_BETWEEN)) {
      mit.ResetCurrentPack();
      mit.NextPackrow();
    } else
      EvaluatePack_NotNull(mit, dim);
  }
}

void TianmuAttr::EvaluatePack_BetweenReal(MIUpdatingIterator &mit, int dim, Descriptor &d) {
  MEASURE_FET("TianmuAttr::EvaluatePack_BetweenReal(...)");
  int pack = mit.GetCurPackrow(dim);
  if (pack == -1) {
    mit.ResetCurrentPack();
    mit.NextPackrow();
    return;
  }
  auto dpn = get_dpn(pack);
  // added trivial case due to OR tree
  if (dpn.NullOnly()) {
    mit.ResetCurrentPack();
    mit.NextPackrow();
    return;
  }

  int64_t pv1 = d.val1.vc->GetValueInt64(mit);
  int64_t pv2 = d.val2.vc->GetValueInt64(mit);
  double dv1 = *((double *)&pv1);
  double dv2 = *((double *)&pv2);
  if (dpn.min_i != dpn.max_i) {
    auto p = get_packN(pack);
    auto filter = mit.GetMultiIndex()->GetFilter(dim);
    // MIIterator iteratoring costs too much
    // Loop without it when packs are nearly full
    if (tianmu_sysvar_filterevaluation_speedup && filter &&
        filter->NumOfOnes(pack) > static_cast<uint>(1 << (mit.GetPower() - 1))) {
      for (uint32_t n = 0; n < dpn.numOfRecords; n++) {
        if (p->IsNull(n))
          filter->Reset(pack, n);
        else {
          double v = p->GetValDouble(n);
          bool res = (dv1 <= v && v <= dv2);
          if (d.op == common::Operator::O_NOT_BETWEEN)
            res = !res;
          if (!res)
            filter->Reset(pack, n);
        }
      }
      mit.NextPackrow();
    } else {
      do {
        int inpack = mit.GetCurInpack(dim);
        if (mit[dim] == common::NULL_VALUE_64 || p->IsNull(inpack))
          mit.ResetCurrent();
        else {
          double v = p->GetValDouble(inpack);
          bool res = (dv1 <= v && v <= dv2);
          if (d.op == common::Operator::O_NOT_BETWEEN)
            res = !res;
          if (!res)
            mit.ResetCurrent();
        }
        ++mit;
      } while (mit.IsValid() && !mit.PackrowStarted());
    }
  } else {
    double uni_val = get_dpn(pack).min_d;
    if (((dv1 > uni_val || dv2 < uni_val) && d.op == common::Operator::O_BETWEEN) ||
        (dv1 <= uni_val && dv2 >= uni_val && d.op == common::Operator::O_NOT_BETWEEN)) {
      mit.ResetCurrentPack();
      mit.NextPackrow();
    } else
      EvaluatePack_NotNull(mit, dim);
  }
}

void TianmuAttr::EvaluatePack_AttrAttr(MIUpdatingIterator &mit, int dim, Descriptor &d) {
  MEASURE_FET("TianmuAttr::EvaluatePack_AttrAttr(...)");
  int pack = mit.GetCurPackrow(dim);
  if (pack == -1) {
    mit.ResetCurrentPack();
    mit.NextPackrow();
    return;
  }
  TianmuAttr *a2 = (TianmuAttr *)(((vcolumn::SingleColumn *)d.val1.vc)->GetPhysical());
  if (get_dpn(pack).numOfNulls == get_dpn(pack).numOfRecords ||
      a2->get_dpn(pack).numOfNulls == a2->get_dpn(pack).numOfRecords) {
    mit.ResetCurrentPack();  // nulls only
    mit.NextPackrow();
    return;
  }
  auto p1 = get_packN(pack);
  auto p2 = a2->get_packN(pack);
  int64_t min1 = get_dpn(pack).min_i;
  int64_t min2 = a2->get_dpn(pack).min_i;
  int64_t max1 = get_dpn(pack).max_i;
  int64_t max2 = a2->get_dpn(pack).max_i;
  bool pack1_uniform = (min1 == max1);
  bool pack2_uniform = (min2 == max2);
  int64_t val1_offset = min1 - min2;  // GetVal_1 + val_offset = GetVal_2
  do {
    int obj_in_pack = mit.GetCurInpack(dim);
    if (mit[dim] == common::NULL_VALUE_64 || (p1 && p1->IsNull(obj_in_pack)) ||
        (p2 && p2->IsNull(obj_in_pack)))  // p1, p2 may be null for uniform
      mit.ResetCurrent();
    else {
      int64_t v1, v2;
      bool res = false;
      v1 = (pack1_uniform ? 0 : p1->GetValInt(obj_in_pack)) + val1_offset;
      v2 = (pack2_uniform ? 0 : p2->GetValInt(obj_in_pack));
      switch (d.op) {
        case common::Operator::O_EQ:
          res = (v1 == v2);
          break;
        case common::Operator::O_NOT_EQ:
          res = (v1 != v2);
          break;
        case common::Operator::O_LESS:
          res = (v1 < v2);
          break;
        case common::Operator::O_LESS_EQ:
          res = (v1 <= v2);
          break;
        case common::Operator::O_MORE:
          res = (v1 > v2);
          break;
        case common::Operator::O_MORE_EQ:
          res = (v1 >= v2);
          break;
        default:
          DEBUG_ASSERT(0);
      }
      if (!res)
        mit.ResetCurrent();
    }
    ++mit;
  } while (mit.IsValid() && !mit.PackrowStarted());
}

void TianmuAttr::EvaluatePack_AttrAttrReal(MIUpdatingIterator &mit, int dim, Descriptor &d) {
  MEASURE_FET("TianmuAttr::EvaluatePack_AttrAttrReal(...)");
  int pack = mit.GetCurPackrow(dim);
  if (pack == -1) {
    mit.ResetCurrentPack();
    mit.NextPackrow();
    return;
  }
  TianmuAttr *a2 = (TianmuAttr *)(((vcolumn::SingleColumn *)d.val1.vc)->GetPhysical());
  if (get_dpn(pack).numOfNulls == get_dpn(pack).numOfRecords ||
      a2->get_dpn(pack).numOfNulls == a2->get_dpn(pack).numOfRecords) {
    mit.ResetCurrentPack();  // nulls only
    mit.NextPackrow();
    return;
  }
  auto p1 = get_packN(pack);
  auto p2 = a2->get_packN(pack);
  int64_t min1 = get_dpn(pack).min_i;
  int64_t min2 = a2->get_dpn(pack).min_i;
  int64_t max1 = get_dpn(pack).max_i;
  int64_t max2 = a2->get_dpn(pack).max_i;
  bool pack1_uniform = (min1 == max1);
  bool pack2_uniform = (min2 == max2);
  do {
    int obj_in_pack = mit.GetCurInpack(dim);
    if (mit[dim] == common::NULL_VALUE_64 || (p1 && p1->IsNull(obj_in_pack)) ||
        (p2 && p2->IsNull(obj_in_pack)))  // p1, p2 may be null for uniform
      mit.ResetCurrent();
    else {
      int64_t pv1, pv2;
      bool res = false;
      pv1 = (pack1_uniform ? min1 : p1->GetValInt(obj_in_pack));
      pv2 = (pack2_uniform ? min2 : p2->GetValInt(obj_in_pack));
      double v1 = *((double *)&pv1);
      double v2 = *((double *)&pv2);
      switch (d.op) {
        case common::Operator::O_EQ:
          res = (v1 == v2);
          break;
        case common::Operator::O_NOT_EQ:
          res = (v1 != v2);
          break;
        case common::Operator::O_LESS:
          res = (v1 < v2);
          break;
        case common::Operator::O_LESS_EQ:
          res = (v1 <= v2);
          break;
        case common::Operator::O_MORE:
          res = (v1 > v2);
          break;
        case common::Operator::O_MORE_EQ:
          res = (v1 >= v2);
          break;
        default:
          DEBUG_ASSERT(0);
      }
      if (!res)
        mit.ResetCurrent();
    }
    ++mit;
  } while (mit.IsValid() && !mit.PackrowStarted());
}

bool TianmuAttr::IsDistinct(Filter *f) {
  MEASURE_FET("TianmuAttr::IsDistinct(...)");
  if (ct.Lookup() && types::RequiresUTFConversions(GetCollation()))
    return false;

  if (PhysicalColumn::IsDistinct() == common::RoughSetValue::RS_ALL) {  // = is_unique_updated && is_unique
    if (f == nullptr)
      return (NumOfNulls() == 0);  // no nulls at all, and is_unique  => distinct

    LoadPackInfo();
    for (uint b = 0; b < SizeOfPack(); b++)
      if (!f->IsEmpty(b) && get_dpn(b).numOfNulls > 0)  // any null in nonempty pack?
        return false;

    return true;
  }

  return false;
}

uint64_t TianmuAttr::ApproxAnswerSize(Descriptor &d) {
  MEASURE_FET("TianmuAttr::ApproxAnswerSize(...)");
  ASSERT(d.encoded, "The descriptor is not encoded!");
  static MIIterator const mit(nullptr, pss);
  LoadPackInfo();

  if (d.op == common::Operator::O_NOT_NULL)
    return NumOfObj() - NumOfNulls();

  if (d.op == common::Operator::O_IS_NULL)
    return NumOfNulls();

  if (d.val1.vc && !d.val1.vc->IsConst()) {
    uint64_t no_distinct = ApproxDistinctVals(false, nullptr, nullptr, false);
    if (no_distinct == 0)
      no_distinct = 1;

    if (d.op == common::Operator::O_EQ)
      return NumOfObj() / no_distinct;

    if (d.op == common::Operator::O_NOT_EQ)
      return NumOfObj() - (NumOfObj() / no_distinct);

    return (NumOfObj() - NumOfNulls()) / 2;  // default
  }

  if (d.op == common::Operator::O_BETWEEN && d.val1.vc->IsConst() && d.val2.vc->IsConst() &&
      GetPackType() == common::PackType::INT) {
    double res = 0;
    int64_t val1 = d.val1.vc->GetValueInt64(mit);
    int64_t val2 = d.val2.vc->GetValueInt64(mit);
    if (!ATI::IsRealType(TypeName())) {
      int64_t span1, span2;  // numerical case: approximate number of rows in each pack
      if (val1 == val2) {
        res = (NumOfObj() - NumOfNulls()) / 2;  // return default; up func will make Prior other types
        return int64_t(res);
      }

      for (uint b = 0; b < SizeOfPack(); b++) {
        if (get_dpn(b).min_i > val2 || get_dpn(b).max_i < val1 || get_dpn(b).numOfNulls == get_dpn(b).numOfRecords)
          continue;  // pack irrelevant

        span1 = get_dpn(b).max_i - get_dpn(b).min_i + 1;
        if (span1 <= 0)  // out of int64_t range
          span1 = 1;

        if (val2 < get_dpn(b).max_i)  // calculate the size of intersection
          span2 = val2;
        else
          span2 = get_dpn(b).max_i;

        if (val1 > get_dpn(b).min_i)
          span2 -= val1;
        else
          span2 -= get_dpn(b).min_i;

        span2 += 1;
        res += (get_dpn(b).numOfRecords - get_dpn(b).numOfNulls) * double(span2) /
               span1;  // supposing uniform distribution of values
      }
    } else {                // double
      double span1, span2;  // numerical case: approximate number of rows in each pack
      double v_min = *(double *)&val1;
      double v_max = *(double *)&val2;

      for (uint b = 0; b < SizeOfPack(); b++) {
        double d_min = get_dpn(b).min_d;
        double d_max = get_dpn(b).max_d;
        if (d_min > v_max || d_max < v_min || get_dpn(b).numOfNulls == get_dpn(b).numOfRecords)
          continue;  // pack irrelevant

        span1 = d_max - d_min;
        span2 = std::min(v_max, d_max) - std::max(v_min, d_min);
        if (span1 == 0)
          res += get_dpn(b).numOfRecords - get_dpn(b).numOfNulls;
        else if (span2 == 0)  // just one value
          res += 1;
        else
          res += (get_dpn(b).numOfRecords - get_dpn(b).numOfNulls) *
                 (span2 / span1);  // supposing uniform distribution of values
      }
    }

    return int64_t(res);
  }

  return (NumOfObj() - NumOfNulls()) / 2;  // default
}

size_t TianmuAttr::MaxStringSize(Filter *f)  // maximal byte string length in column
{
  LoadPackInfo();
  size_t max_size = 1;
  if (Type().Lookup()) {
    int64_t cur_min = common::PLUS_INF_64;
    int64_t cur_max = common::MINUS_INF_64;
    for (uint b = 0; b < SizeOfPack(); b++) {
      if ((f && f->IsEmpty(b)) || GetPackOntologicalStatus(b) == PackOntologicalStatus::NULLS_ONLY)
        continue;

      auto &d = get_dpn(b);
      if (d.min_i < cur_min)
        cur_min = d.min_i;

      if (d.max_i > cur_max)
        cur_max = d.max_i;
    }

    if (cur_min != common::PLUS_INF_64)
      max_size = m_dict->MaxValueSize(int(cur_min), int(cur_max));
  } else {
    for (uint b = 0; b < SizeOfPack(); b++) {
      if (f && f->IsEmpty(b))
        continue;

      size_t cur_size = GetActualSize(b);
      if (max_size < cur_size)
        max_size = cur_size;

      if (max_size == Type().GetPrecision())
        break;
    }
  }

  return max_size;
}

bool TianmuAttr::TryToMerge(Descriptor &d1, Descriptor &d2) {
  MEASURE_FET("TianmuAttr::TryToMerge(...)");

  if ((d1.op != common::Operator::O_BETWEEN && d1.op != common::Operator::O_NOT_BETWEEN) ||
      (d2.op != common::Operator::O_BETWEEN && d2.op != common::Operator::O_NOT_BETWEEN))
    return false;

  if (GetPackType() == common::PackType::INT && d1.val1.vc && d1.val2.vc && d2.val1.vc && d2.val2.vc &&
      d1.val1.vc->IsConst() && d1.val2.vc->IsConst() && d2.val1.vc->IsConst() && d2.val2.vc->IsConst()) {
    static MIIterator const mit(nullptr, pss);
    int64_t d1min = d1.val1.vc->GetValueInt64(mit);
    int64_t d1max = d1.val2.vc->GetValueInt64(mit);
    int64_t d2min = d2.val1.vc->GetValueInt64(mit);
    int64_t d2max = d2.val2.vc->GetValueInt64(mit);

    if (!ATI::IsRealType(TypeName())) {
      if (d1.op == common::Operator::O_BETWEEN && d2.op == common::Operator::O_BETWEEN) {
        if (d2min > d1min) {
          std::swap(d1.val1, d2.val1);
          std::swap(d1min, d2min);
        }

        if (d2max < d1max) {
          std::swap(d1.val2, d2.val2);
          std::swap(d1max, d2max);
        }

        if (d1min > d1max)
          d1.op = common::Operator::O_FALSE;  // disjoint?
        return true;
      }

      if (d1.op == common::Operator::O_NOT_BETWEEN && d2.op == common::Operator::O_NOT_BETWEEN) {
        if (d1min < d2max && d2min < d1max) {
          if (d2min < d1min)
            std::swap(d1.val1, d2.val1);

          if (d2max > d1max)
            std::swap(d1.val2, d2.val2);

          return true;
        }
      }
    } else {  // double
      if (d1.sharp != d2.sharp)
        return false;

      double dv1min = *((double *)&d1min);
      double dv1max = *((double *)&d1max);
      double dv2min = *((double *)&d2min);
      double dv2max = *((double *)&d2max);
      if (d1.op == common::Operator::O_BETWEEN && d2.op == common::Operator::O_BETWEEN) {
        if (dv2min > dv1min) {
          std::swap(d1.val1, d2.val1);
          std::swap(dv2min, dv1min);
        }

        if (dv2max < dv1max) {
          std::swap(d1.val2, d2.val2);
          std::swap(dv2max, dv1max);
        }

        if (dv1min > dv1max)
          d1.op = common::Operator::O_FALSE;  // disjoint?

        return true;
      }

      if (d1.op == common::Operator::O_NOT_BETWEEN && d2.op == common::Operator::O_NOT_BETWEEN) {
        if (dv1min < dv2max && dv2min < dv1max) {
          if (dv2min < dv1min)
            std::swap(d1.val1, d2.val1);

          if (dv2max > dv1max)
            std::swap(d1.val2, d2.val2);

          return true;
        }
      }
    }
  } else if (GetPackType() == common::PackType::STR && d1.val1.vc && d1.val2.vc && d2.val1.vc && d2.val2.vc &&
             d1.val1.vc->IsConst() && d1.val2.vc->IsConst() && d2.val1.vc->IsConst() && d2.val2.vc->IsConst() &&
             d1.sharp == d2.sharp && d1.GetCollation().collation == d2.GetCollation().collation) {
    static MIIterator const mit(nullptr, pss);
    types::BString d1min, d1max, d2min, d2max;
    d1.val1.vc->GetValueString(d1min, mit);
    d1.val2.vc->GetValueString(d1max, mit);
    d2.val1.vc->GetValueString(d2min, mit);
    d2.val2.vc->GetValueString(d2max, mit);
    DTCollation my_coll = d1.GetCollation();

    if (d1.op == common::Operator::O_BETWEEN && d2.op == common::Operator::O_BETWEEN) {
      if (types::RequiresUTFConversions(my_coll)) {
        if (d1min.IsNull() || CollationStrCmp(my_coll, d2min, d1min, common::Operator::O_MORE)) {
          std::swap(d1.val1, d2.val1);
          std::swap(d1min, d2min);
        }

        if (d1max.IsNull() || (!d2max.IsNull() && CollationStrCmp(my_coll, d2max, d1max, common::Operator::O_LESS))) {
          std::swap(d1.val2, d2.val2);
          std::swap(d1max, d2max);
        }

        if (CollationStrCmp(my_coll, d1min, d1max, common::Operator::O_MORE))
          d1.op = common::Operator::O_FALSE;  // disjoint?
      } else {
        if (d1min.IsNull() || d2min > d1min) {  // IsNull() means infinity here
          std::swap(d1.val1, d2.val1);
          std::swap(d1min, d2min);
        }

        if (d1max.IsNull() || d2max < d1max) {
          std::swap(d1.val2, d2.val2);
          std::swap(d1max, d2max);
        }

        if (d1min > d1max)
          d1.op = common::Operator::O_FALSE;  // disjoint?
      }

      return true;
    }

    if (d1.op == common::Operator::O_NOT_BETWEEN && d2.op == common::Operator::O_NOT_BETWEEN) {
      if (d1min.IsNull() || d1max.IsNull() || d2min.IsNull() || d2max.IsNull())
        return false;  // should not appear in normal circumstances

      if (types::RequiresUTFConversions(my_coll)) {
        if (CollationStrCmp(my_coll, d1min, d2max, common::Operator::O_LESS) &&
            CollationStrCmp(my_coll, d2min, d1max, common::Operator::O_LESS)) {
          if (CollationStrCmp(my_coll, d2min, d1min, common::Operator::O_LESS))
            std::swap(d1.val1, d2.val1);

          if (CollationStrCmp(my_coll, d2max, d1max, common::Operator::O_MORE))
            std::swap(d1.val2, d2.val2);

          return true;
        }
      } else {
        if (d1min < d2max && d2min < d1max) {
          if (d2min < d1min)
            std::swap(d1.val1, d2.val1);

          if (d2max > d1max)
            std::swap(d1.val2, d2.val2);

          return true;
        }
      }
    }
  }
  return false;
}

}  // namespace core
}  // namespace Tianmu
