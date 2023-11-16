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
#include <eventql/util/stringutil.h>
#include <eventql/util/inspect.h>
#include <eventql/util/json/jsonoutputstream.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <cmath>

namespace json {

JSONOutputStream::JSONOutputStream(
    std::unique_ptr<OutputStream> output_stream) {
  output_.reset(output_stream.release());
}

void JSONOutputStream::write(const JSONObject& obj) {
  for (const auto& t : obj) {
    emplace_back(t.type, t.data);
  }
}

void JSONOutputStream::emplace_back(kTokenType token) {
  emplace_back(token, "");
}

void JSONOutputStream::emplace_back(const JSONToken& token) {
  emplace_back(token.type, token.data);
}

void JSONOutputStream::emplace_back(
    kTokenType token,
    const std::string& data) {

  switch (token) {
    case JSON_ARRAY_END:
      endArray();

      if (!stack_.empty()) {
        stack_.pop();
      }

      if (!stack_.empty()) {
        stack_.top().second++;
      }
      return;

    case JSON_OBJECT_END:
      endObject();

      if (!stack_.empty()) {
        stack_.pop();
      }

      if (!stack_.empty()) {
        stack_.top().second++;
      }
      return;

    default:
      break;

  }

  if (!stack_.empty() && stack_.top().second > 0) {
    switch (stack_.top().first) {
      case JSON_ARRAY_BEGIN:
        addComma();
        break;

      case JSON_OBJECT_BEGIN:
        (stack_.top().second % 2 == 0) ? addComma() : addColon();
        break;

      default:
        break;
    }
  }

  switch (token) {

    case JSON_ARRAY_BEGIN:
      beginArray();
      stack_.emplace(JSON_ARRAY_BEGIN, 0);
      break;

    case JSON_OBJECT_BEGIN:
      beginObject();
      stack_.emplace(JSON_OBJECT_BEGIN, 0);
      break;

    case JSON_STRING:
      addString(data);

      if (!stack_.empty()) {
        stack_.top().second++;
      }
      break;

    case JSON_NUMBER:
      addString(data);

      if (!stack_.empty()) {
        stack_.top().second++;
      }
      break;

    case JSON_TRUE:
      addTrue();

      if (!stack_.empty()) {
        stack_.top().second++;
      }
      break;

    case JSON_FALSE:
      addFalse();

      if (!stack_.empty()) {
        stack_.top().second++;
      }
      break;

    case JSON_NULL:
      addNull();

      if (!stack_.empty()) {
        stack_.top().second++;
      }
      break;

    default:
      break;
  }
}

void JSONOutputStream::beginObject() {
  output_->printf("{");
}

void JSONOutputStream::endObject() {
  output_->printf("}");
}

void JSONOutputStream::addObjectEntry(const std::string& key) {
  output_->printf("\"%s\": ", escapeString(key).c_str());
}

void JSONOutputStream::addComma() {
  output_->printf(",");
}

void JSONOutputStream::addColon() {
  output_->printf(":");
}

void JSONOutputStream::addString(const std::string& string) {
  output_->write("\"");
  output_->write(escapeString(string));
  output_->write("\"");
}

void JSONOutputStream::addInteger(int64_t value) {
  output_->write(StringUtil::toString(value));
}

void JSONOutputStream::addNull() {
  output_->write("null");
}

void JSONOutputStream::addBool(bool val) {
  if (val) {
    addTrue();
  } else {
    addFalse();
  }
}

void JSONOutputStream::addTrue() {
  output_->write("true");
}

void JSONOutputStream::addFalse() {
  output_->write("false");
}

void JSONOutputStream::addFloat(double value) {
  if (std::isnormal(value) || value == 0.0f) {
    output_->write(StringUtil::toString(value));
  } else {
    addNull();
  }
}

void JSONOutputStream::beginArray() {
  output_->printf("[");
}

void JSONOutputStream::endArray() {
  output_->printf("]");
}

/*
template <>
void JSONOutputStream::addValue(const std::string& value) {
  addString(value);
}

template <>
void JSONOutputStream::addValue(const int& value) {
  output_->write(StringUtil::toString(value));
}

template <>
void JSONOutputStream::addValue(const unsigned long& value) {
  output_->write(StringUtil::toString(value));
}

template <>
void JSONOutputStream::addValue(const unsigned long long& value) {
  output_->write(StringUtil::toString(value));
}

template <>
void JSONOutputStream::addValue(const double& value) {
  addFloat(value);
}

template <>
void JSONOutputStream::addValue(const bool& value) {
  value ? addTrue() : addFalse();
}

template <>
void JSONOutputStream::addValue(const std::nullptr_t& value) {
  addNull();
}
*/

std::string escapeString(const std::string& string) {
  std::string new_str;

  for (int i = 0; i < string.size(); ++i) {
    switch (string.at(i)) {
      case 0x00:
        new_str += "\\u0000";
        break;
      case 0x01:
        new_str += "\\u0001";
        break;
      case 0x02:
        new_str += "\\u0002";
        break;
      case 0x03:
        new_str += "\\u0003";
        break;
      case 0x04:
        new_str += "\\u0004";
        break;
      case 0x05:
        new_str += "\\u0005";
        break;
      case 0x06:
        new_str += "\\u0006";
        break;
      case 0x07:
        new_str += "\\u0007";
        break;
      case '\b':
        new_str += "\\b";
        break;
      case '\t':
        new_str += "\\t";
        break;
      case '\n':
        new_str += "\\n";
        break;
      case 0x0b:
        new_str += "\\u000b";
        break;
      case '\f':
        new_str += "\\f";
        break;
      case '\r':
        new_str += "\\r";
        break;
      case 0x0e:
        new_str += "\\u000e";
        break;
      case 0x0f:
        new_str += "\\u000f";
        break;
      case 0x10:
        new_str += "\\u0010";
        break;
      case 0x11:
        new_str += "\\u0011";
        break;
      case 0x12:
        new_str += "\\u0012";
        break;
      case 0x13:
        new_str += "\\u0013";
        break;
      case 0x14:
        new_str += "\\u0014";
        break;
      case 0x15:
        new_str += "\\u0015";
        break;
      case 0x16:
        new_str += "\\u0016";
        break;
      case 0x17:
        new_str += "\\u0017";
        break;
      case 0x18:
        new_str += "\\u0018";
        break;
      case 0x19:
        new_str += "\\u0019";
        break;
      case 0x1a:
        new_str += "\\u001a";
        break;
      case 0x1b:
        new_str += "\\u001b";
        break;
      case 0x1c:
        new_str += "\\u001c";
        break;
      case 0x1d:
        new_str += "\\u001d";
        break;
      case 0x1e:
        new_str += "\\u001e";
        break;
      case 0x1f:
        new_str += "\\u001f";
        break;
      case '"':
        new_str += "\\\"";
        break;
      case '\\':
        new_str += "\\\\";
        break;
      default:
        new_str += string.at(i);
    }
  }

  return new_str;
}


} // namespace json

