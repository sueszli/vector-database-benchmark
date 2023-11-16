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

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

#include "aggregator_advanced.h"
#include "types/tianmu_num.h"

namespace Tianmu {
namespace core {
void AggregatorStat64::PutAggregatedValue(unsigned char *buf, int64_t v, int64_t factor) {
  // efficient implementation from WIKI
  // http://en.wikipedia.org/wiki/Standard_deviation
  stats_updated = false;
  if (v != common::NULL_VALUE_64) {
    if (NumOfObj(buf) == 0) {
      NumOfObj(buf) += 1;
      A(buf) = double(v);  // m
      Q(buf) = 0;          // s
      factor--;
    }
    for (int i = 0; i < factor; i++) {
      NumOfObj(buf) += 1;
      double vd = double(v);
      double A_prev = A(buf);
      A(buf) = A_prev + (vd - A_prev) / (double)NumOfObj(buf);
      Q(buf) += (vd - A_prev) * (vd - A(buf));
    }
  }
}

void AggregatorStatD::PutAggregatedValue(unsigned char *buf, int64_t v, int64_t factor) {
  // efficient implementation from WIKI  http://en.wikipedia.org/wiki/Standard_deviation
  stats_updated = false;
  if (v != common::NULL_VALUE_64) {
    if (NumOfObj(buf) == 0) {
      NumOfObj(buf) += 1;
      A(buf) = *((double *)(&v));  // m
      Q(buf) = 0;                  // s
      factor--;
    }
    for (int i = 0; i < factor; i++) {
      NumOfObj(buf) += 1;
      double vd = *((double *)(&v));
      double A_prev = A(buf);
      A(buf) = A_prev + (vd - A_prev) / (double)NumOfObj(buf);
      Q(buf) += (vd - A_prev) * (vd - A(buf));
    }
  }
}

void AggregatorStat::Merge(unsigned char *buf, unsigned char *src_buf) {
  if (NumOfObj(src_buf) == 0)
    return;
  stats_updated = false;
  if (NumOfObj(buf) == 0)
    std::memcpy(buf, src_buf, BufferByteSize());
  else {
    int64_t n = NumOfObj(buf);
    int64_t m = NumOfObj(src_buf);
    // n*var(X) = Q(X)
    // var(X+Y) = (n*var(X) + m*var(Y)) / (n+m) + nm / (n+m)^2 * (avg(X) -
    // avg(Y))^2
    Q(buf) = Q(buf) + Q(src_buf) + n * m / double(n + m) * (A(buf) - A(src_buf)) * (A(buf) - A(src_buf));

    // avg(X+Y) = (avg(X)*n + avg(Y)*m) / (n+m)
    A(buf) = (A(buf) * n + A(src_buf) * m) / double(n + m);
    NumOfObj(buf) = n + m;
  }
}

void AggregatorStatD::PutAggregatedValue(unsigned char *buf, const types::BString &v, int64_t factor) {
  stats_updated = false;
  types::TianmuNum val(common::ColumnType::REAL);
  if (!v.IsEmpty() && types::TianmuNum::ParseReal(v, val, common::ColumnType::REAL) == common::ErrorCode::SUCCESS &&
      !val.IsNull()) {
    double d_val = double(val);
    PutAggregatedValue(buf, *((int64_t *)(&d_val)), factor);
  }
}

int64_t AggregatorVarPop64::GetValue64(unsigned char *buf) {
  if (NumOfObj(buf) < 1)
    return common::NULL_VALUE_64;
  double vd = VarPop(buf) / prec_factor / double(prec_factor);
  return *(int64_t *)(&vd);
}

int64_t AggregatorVarSamp64::GetValue64(unsigned char *buf) {
  if (NumOfObj(buf) < 2)
    return common::NULL_VALUE_64;
  double vd = VarSamp(buf) / prec_factor / double(prec_factor);
  return *(int64_t *)(&vd);
}

int64_t AggregatorStdPop64::GetValue64(unsigned char *buf) {
  if (NumOfObj(buf) < 1)
    return common::NULL_VALUE_64;
  double vd = sqrt(VarPop(buf)) / prec_factor;
  return *(int64_t *)(&vd);
}

int64_t AggregatorStdSamp64::GetValue64(unsigned char *buf) {
  if (NumOfObj(buf) < 2)
    return common::NULL_VALUE_64;
  double vd = sqrt(VarSamp(buf)) / prec_factor;
  return *(int64_t *)(&vd);
}

int64_t AggregatorVarPopD::GetValue64(unsigned char *buf) {
  if (NumOfObj(buf) < 1)
    return common::NULL_VALUE_64;
  double vd = VarPop(buf);
  return *(int64_t *)(&vd);
}

int64_t AggregatorVarSampD::GetValue64(unsigned char *buf) {
  if (NumOfObj(buf) < 2)
    return common::NULL_VALUE_64;
  double vd = VarSamp(buf);
  return *(int64_t *)(&vd);
}

int64_t AggregatorStdPopD::GetValue64(unsigned char *buf) {
  if (NumOfObj(buf) < 1)
    return common::NULL_VALUE_64;
  // double vd = Q(buf) / NumOfObj(buf);
  // vd = sqrt(vd);
  double vd = sqrt(VarPop(buf));
  return *(int64_t *)(&vd);
}

int64_t AggregatorStdSampD::GetValue64(unsigned char *buf) {
  if (NumOfObj(buf) < 2)
    return common::NULL_VALUE_64;
  // double vd = Q(buf) / (NumOfObj(buf) - 1);
  // vd = sqrt(vd);
  double vd = sqrt(VarSamp(buf));
  return *(int64_t *)(&vd);
}

void AggregatorBitAnd::PutAggregatedValue(unsigned char *buf, const types::BString &v, int64_t factor) {
  stats_updated = false;
  types::TianmuNum val(common::ColumnType::BIGINT);
  if (!v.IsEmpty() && types::TianmuNum::Parse(v, val, common::ColumnType::BIGINT) == common::ErrorCode::SUCCESS &&
      !val.IsNull()) {
    PutAggregatedValue(buf, int64_t(val), factor);
  }
}

void AggregatorBitOr::PutAggregatedValue(unsigned char *buf, const types::BString &v, int64_t factor) {
  stats_updated = false;
  types::TianmuNum val(common::ColumnType::BIGINT);
  if (!v.IsEmpty() && types::TianmuNum::Parse(v, val, common::ColumnType::BIGINT) == common::ErrorCode::SUCCESS &&
      !val.IsNull()) {
    PutAggregatedValue(buf, int64_t(val), factor);
  }
}

void AggregatorBitXor::PutAggregatedValue(unsigned char *buf, const types::BString &v, int64_t factor) {
  stats_updated = false;
  types::TianmuNum val(common::ColumnType::BIGINT);
  if (!v.IsEmpty() && types::TianmuNum::Parse(v, val, common::ColumnType::BIGINT) == common::ErrorCode::SUCCESS &&
      !val.IsNull()) {
    PutAggregatedValue(buf, int64_t(val), factor);
  }
}

void AggregatorGroupConcat::PutAggregatedValue(unsigned char *buf, const types::BString &v,
                                               [[maybe_unused]] int64_t factor) {
  stats_updated = false;

  auto it = lenmap_.find(buf);
  if (it == lenmap_.end()) {
    auto copylen = (v.len_ > gconcat_maxlen_) ? gconcat_maxlen_ : v.len_;
    std::memcpy(buf, v.val_, copylen);
    lenmap_.emplace(buf, copylen);
  } else {
    auto pos = it->second;

    if (pos == gconcat_maxlen_)
      return;

    if (pos < gconcat_maxlen_) {
      std::string src = si_.separator + v.ToString();  // combine the delimeter and value
      auto copylen = (pos + v.len_ + si_.separator.length()) >= gconcat_maxlen_ ? (gconcat_maxlen_ - pos)
                                                                                : (v.len_ + si_.separator.length());
      std::memcpy(buf + pos, src.c_str(), copylen);  // append the separator
      it->second = it->second + copylen;             // update the length of the buffer
    } else {
      TIANMU_LOG(LogCtl_Level::ERROR,
                 "Internal error for AggregatorGroupConcat: buffer length is "
                 "%d, which beyond threshold %d.",
                 pos, gconcat_maxlen_);
    }
  }
}

// puts the group concat value into group_concator.
void AggregatorGroupConcat::PutAggregatedValue([[maybe_unused]] unsigned char *buf, [[maybe_unused]] int64_t v,
                                               [[maybe_unused]] int64_t factor) {
  return;
}

// od the merge operations.
void AggregatorGroupConcat::Merge([[maybe_unused]] unsigned char *buf, [[maybe_unused]] unsigned char *src_buf) {
  return;
}

// the the values from group concator.
int64_t AggregatorGroupConcat::GetValue64([[maybe_unused]] unsigned char *buf) { return -1; }

types::BString AggregatorGroupConcat::GetValueT(unsigned char *buf) {
  auto it = lenmap_.find(buf);
  if (it == lenmap_.end()) {
    // cases that grouping value is nullptr
    return types::BString();
  }

  int len = (it->second < gconcat_maxlen_) ? it->second : gconcat_maxlen_;
  // TIANMU_LOG(LogCtl_Level::INFO, "GetValueT: buf %s, buf addr  %x, len %d", buf, buf,
  // len);
  if (len == 0) {
    types::BString res("", 0);
    return res;
  }

  if (si_.order == ORDER::ORDER_NOT_RELEVANT)  // NO order by logic
  {
    char *p = (char *)buf;
    types::BString res(p, len);
    return res;
  }

  // sorting the output
  std::string tmpstr(reinterpret_cast<char *>(buf), len);
  size_t pos = 0;
  size_t start_pos = 0;
  std::string token;
  std::vector<std::string> vstr;
  vstr.reserve(len);

  // parse and split the buf
  while (pos != std::string::npos && static_cast<int>(pos) < len) {
    pos = tmpstr.find(si_.separator, start_pos);
    token = tmpstr.substr(start_pos, pos - start_pos);
    vstr.emplace_back(token);
    start_pos = pos + si_.separator.length();
  }

  if (si_.order == ORDER::ORDER_DESC) {
    if (ATI::IsStringType(attrtype_))
      std::sort(vstr.begin(), vstr.end(), std::greater<std::string>());
    else  // numeric
      std::sort(vstr.begin(), vstr.end(), [](const auto &a, const auto &b) { return std::stold(a) > std::stold(b); });

  } else {
    if (ATI::IsStringType(attrtype_))
      std::sort(vstr.begin(), vstr.end());
    else  // numeric
      std::sort(vstr.begin(), vstr.end(), [](const auto &a, const auto &b) { return std::stold(a) < std::stold(b); });
  }

  std::ostringstream outbuf_stream;
  std::copy(vstr.begin(), vstr.end(), std::ostream_iterator<std::string>(outbuf_stream, si_.separator.c_str()));
  // TIANMU_LOG(LogCtl_Level::DEBUG, "buf %s, tmpbuf1 %s, pos %d, len %d \n", buf,
  // outbuf_stream.str().c_str(), pos, len);

  types::BString res(outbuf_stream.str().c_str(), len, true);
  return res;
}

}  // namespace core
}  // namespace Tianmu
