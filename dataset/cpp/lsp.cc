#include "lsp.h"

#include "lru_cache.h"
#include "platform.h"
#include "recorder.h"
#include "serializers/json.h"

#include <doctest/doctest.h>
#include <rapidjson/writer.h>
#include <loguru.hpp>

#include <stdio.h>
#include <iostream>

namespace {

constexpr const int kMaxUriCacheEntries = 5000;
struct UriCache {
  LruCache<AbsolutePath, std::string> cache;
  UriCache() : cache(kMaxUriCacheEntries) {}

  void RecordPath(const std::string& path) {
    optional<AbsolutePath> normalized = NormalizePath(path);
    if (normalized) {
      LOG_S(INFO) << "RecordPath: client=" << path
                  << ", normalized=" << *normalized;
      cache.Insert(*normalized, path);
    }
  }

  std::string GetPath(const AbsolutePath& path) {
    std::string resolved;
    if (cache.TryGet(path, &resolved))
      return resolved;
    LOG_S(INFO) << "No cached URI for " << path;

    // If we do not have the value in the cache, try to renormalize it.
    // Otherwise we will return paths with all lower-case letters which may
    // break vscode.
    optional<AbsolutePath> normalized = NormalizePath(
        path.path, true /*ensure_exists*/, false /*force_lower_on_windows*/);
    if (normalized)
      return normalized->path;

    return path;
  }

  static UriCache* instance() {
    static UriCache* instance = nullptr;
    if (!instance)
      instance = new UriCache();
    return instance;
  }
};

}  // namespace

MessageRegistry* MessageRegistry::instance_ = nullptr;

lsTextDocumentIdentifier
lsVersionedTextDocumentIdentifier::AsTextDocumentIdentifier() const {
  lsTextDocumentIdentifier result;
  result.uri = uri;
  return result;
}

// Reads a JsonRpc message. |read| returns the next input character.
optional<std::string> ReadJsonRpcContentFrom(
    std::function<optional<char>()> read) {
  // Read the header. The header itself, along with each field, is terminated by
  // the "\r\n" sequence.
  const char* kContentLengthStart = "Content-Length: ";
  const char* kContentTypeStart = "Content-Type: ";
  int content_length = -1;

  while (true) {
    int exit_seq = 0;
    std::string stringified_header_field;
    while (true) {
      optional<char> opt_c = read();
      if (!opt_c) {
        LOG_S(INFO) << "No more input when reading header";
        return nullopt;
      }
      char c = *opt_c;

      if (exit_seq == 0 && c == '\r') {
        ++exit_seq;
      } else if (exit_seq == 1 && c == '\n') {
        break;
      } else {
        stringified_header_field += c;
      }
    }

    if (stringified_header_field.size()) {
      if (StartsWith(stringified_header_field, kContentLengthStart)) {
        content_length =
          atoi(stringified_header_field.c_str() + strlen(kContentLengthStart));
      } else if (StartsWith(stringified_header_field, kContentTypeStart)) {
        // Content-Type field is ignored.
      } else {
        LOG_S(INFO) << "Unknown field in the header";
        return nullopt;
      }
    } else {
      break;
    }
  }

  if (content_length == -1) {
    LOG_S(INFO) << "Missing content length";
    return nullopt;
  }

  // Read content.
  std::string content;
  content.reserve(content_length);
  for (int i = 0; i < content_length; ++i) {
    optional<char> c = read();
    if (!c) {
      LOG_S(INFO) << "No more input when reading content body";
      return nullopt;
    }
    content += *c;
  }

  RecordInput(content);

  return content;
}

std::function<optional<char>()> MakeContentReader(std::string* content,
                                                  bool can_be_empty) {
  return [content, can_be_empty]() -> optional<char> {
    if (!can_be_empty)
      REQUIRE(!content->empty());
    if (content->empty())
      return nullopt;
    char c = (*content)[0];
    content->erase(content->begin());
    return c;
  };
}

TEST_SUITE("FindIncludeLine") {
  TEST_CASE("ReadContentFromSource") {
    auto parse_correct = [](std::string content) -> std::string {
      auto reader = MakeContentReader(&content, false /*can_be_empty*/);
      auto got = ReadJsonRpcContentFrom(reader);
      REQUIRE(got);
      return got.value();
    };

    auto parse_incorrect = [](std::string content) -> optional<std::string> {
      auto reader = MakeContentReader(&content, true /*can_be_empty*/);
      return ReadJsonRpcContentFrom(reader);
    };

    REQUIRE(parse_correct("Content-Length: 0\r\n\r\n") == "");
    REQUIRE(parse_correct("Content-Length: 1\r\n\r\na") == "a");
    REQUIRE(parse_correct("Content-Length: 4\r\n\r\nabcd") == "abcd");

    REQUIRE(parse_incorrect("ggg") == optional<std::string>());
    REQUIRE(parse_incorrect("Content-Length: 0\r\n") ==
            optional<std::string>());
    REQUIRE(parse_incorrect("Content-Length: 5\r\n\r\nab") ==
            optional<std::string>());
  }
}

optional<char> ReadCharFromStdinBlocking() {
  // We do not use std::cin because it does not read bytes once stuck in
  // cin.bad(). We can call cin.clear() but C++ iostream has other annoyance
  // like std::{cin,cout} is tied by default, which causes undesired cout flush
  // for cin operations.
  int c = getchar();
  if (c >= 0)
    return c;
  return nullopt;
}

optional<std::string> MessageRegistry::ReadMessageFromStdin(
    std::unique_ptr<InMessage>* message) {
  optional<std::string> content =
      ReadJsonRpcContentFrom(&ReadCharFromStdinBlocking);
  if (!content) {
    LOG_S(ERROR) << "Failed to read JsonRpc input; exiting";
    exit(1);
  }

  rapidjson::Document document;
  document.Parse(content->c_str(), content->length());
  assert(!document.HasParseError());

  JsonReader json_reader{&document};
  return Parse(json_reader, message);
}

optional<std::string> MessageRegistry::Parse(
    Reader& visitor,
    std::unique_ptr<InMessage>* message) {
  if (!visitor.HasMember("jsonrpc") ||
      std::string(visitor["jsonrpc"]->GetString()) != "2.0") {
    LOG_S(FATAL) << "Bad or missing jsonrpc version";
    exit(1);
  }

  std::string method;
  ReflectMember(visitor, "method", method);

  if (allocators.find(method) == allocators.end())
    return std::string("Unable to find registered handler for method '") +
           method + "'";

  Allocator& allocator = allocators[method];
  try {
    allocator(visitor, message);
    return nullopt;
  } catch (std::invalid_argument& e) {
    // *message is partially deserialized but some field (e.g. |id|) are likely
    // available.
    return std::string("Fail to parse '") + method + "' " +
           static_cast<JsonReader&>(visitor).GetPath() + ", expected " +
           e.what();
  }
}

MessageRegistry* MessageRegistry::instance() {
  if (!instance_)
    instance_ = new MessageRegistry();

  return instance_;
}

lsBaseOutMessage::~lsBaseOutMessage() = default;

void lsBaseOutMessage::Write(std::ostream& out) {
  rapidjson::StringBuffer output;
  rapidjson::Writer<rapidjson::StringBuffer> writer(output);
  JsonWriter json_writer{&writer};
  ReflectWriter(json_writer);

  out << "Content-Length: " << output.GetSize() << "\r\n\r\n"
      << output.GetString();
  out.flush();
}

void lsResponseError::Write(Writer& visitor) {
  auto& value = *this;
  int code2 = static_cast<int>(this->code);

  visitor.StartObject();
  REFLECT_MEMBER2("code", code2);
  REFLECT_MEMBER(message);
  visitor.EndObject();
}

lsDocumentUri lsDocumentUri::FromPath(const AbsolutePath& path) {
  lsDocumentUri result;
  result.SetPath(UriCache::instance()->GetPath(path));
  return result;
}

lsDocumentUri::lsDocumentUri() {}

bool lsDocumentUri::operator==(const lsDocumentUri& other) const {
  return raw_uri_ == other.raw_uri_;
}

void lsDocumentUri::SetPath(const AbsolutePath& path) {
  // file:///c%3A/Users/jacob/Desktop/superindex/indexer/full_tests
  raw_uri_ = path;

  size_t index = raw_uri_.find(":");
  if (index == 1) {  // widows drive letters must always be 1 char
    raw_uri_.replace(raw_uri_.begin() + index, raw_uri_.begin() + index + 1,
                     "%3A");
  }

  // subset of reserved characters from the URI standard
  // http://www.ecma-international.org/ecma-262/6.0/#sec-uri-syntax-and-semantics
  std::string t;
  t.reserve(8 + raw_uri_.size());
  // TODO: proper fix
#if defined(_WIN32)
  t += "file:///";
#else
  t += "file://";
#endif

  // clang-format off
  for (char c : raw_uri_)
    switch (c) {
    case ' ': t += "%20"; break;
    case '#': t += "%23"; break;
    case '$': t += "%24"; break;
    case '&': t += "%26"; break;
    case '(': t += "%28"; break;
    case ')': t += "%29"; break;
    case '+': t += "%2B"; break;
    case ',': t += "%2C"; break;
    case ';': t += "%3B"; break;
    case '?': t += "%3F"; break;
    case '@': t += "%40"; break;
    default: t += c; break;
    }
  // clang-format on
  raw_uri_ = std::move(t);
}

std::string lsDocumentUri::GetRawPath() const {
  if (raw_uri_.compare(0, 8, "file:///"))
    return raw_uri_;
  std::string ret;
#if defined(_WIN32)
  size_t i = 8;
#else
  size_t i = 7;
#endif
  auto from_hex = [](unsigned char c) {
    return c - '0' < 10 ? c - '0' : (c | 32) - 'a' + 10;
  };
  for (; i < raw_uri_.size(); i++) {
    if (i + 3 <= raw_uri_.size() && raw_uri_[i] == '%') {
      ret.push_back(from_hex(raw_uri_[i + 1]) * 16 + from_hex(raw_uri_[i + 2]));
      i += 2;
    } else
      ret.push_back(raw_uri_[i] == '\\' ? '/' : raw_uri_[i]);
  }
  return ret;
}

AbsolutePath lsDocumentUri::GetAbsolutePath() const {
  return *NormalizePath(GetRawPath(), false /*ensure_exists*/);
}

void Reflect(Writer& visitor, lsDocumentUri& value) {
  Reflect(visitor, value.raw_uri_);
}
void Reflect(Reader& visitor, lsDocumentUri& value) {
  Reflect(visitor, value.raw_uri_);
  // Only record the path when we deserialize a URI, since it most likely came
  // from the client.
  UriCache::instance()->RecordPath(value.GetRawPath());
}

lsPosition::lsPosition() {}
lsPosition::lsPosition(int line, int character)
    : line(line), character(character) {}

bool lsPosition::operator==(const lsPosition& other) const {
  return line == other.line && character == other.character;
}

bool lsPosition::operator<(const lsPosition& other) const {
  return line != other.line ? line < other.line : character < other.character;
}

std::string lsPosition::ToString() const {
  return std::to_string(line) + ":" + std::to_string(character);
}
const lsPosition lsPosition::kZeroPosition = lsPosition();

lsRange::lsRange() {}
lsRange::lsRange(lsPosition start, lsPosition end) : start(start), end(end) {}

bool lsRange::operator==(const lsRange& o) const {
  return start == o.start && end == o.end;
}

bool lsRange::operator<(const lsRange& o) const {
  return !(start == o.start) ? start < o.start : end < o.end;
}

lsLocation::lsLocation() {}
lsLocation::lsLocation(lsDocumentUri uri, lsRange range)
    : uri(uri), range(range) {}

bool lsLocation::operator==(const lsLocation& o) const {
  return uri == o.uri && range == o.range;
}

bool lsLocation::operator<(const lsLocation& o) const {
  return std::make_tuple(uri.raw_uri_, range) <
         std::make_tuple(o.uri.raw_uri_, o.range);
}

bool lsTextEdit::operator==(const lsTextEdit& that) {
  return range == that.range && newText == that.newText;
}

void Reflect(Writer& visitor, lsMarkedString& value) {
  // If there is a language, emit a `{language:string, value:string}` object. If
  // not, emit a string.
  if (value.language) {
    REFLECT_MEMBER_START();
    REFLECT_MEMBER(language);
    REFLECT_MEMBER(value);
    REFLECT_MEMBER_END();
  } else {
    Reflect(visitor, value.value);
  }
}

std::string Out_ShowLogMessage::method() {
  if (display_type == DisplayType::Log)
    return "window/logMessage";
  return "window/showMessage";
}
