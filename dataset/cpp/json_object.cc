// Copyright 2022, DragonflyDB authors.  All rights reserved.
// See LICENSE for licensing terms.
//

#include "core/json_object.h"
extern "C" {
#include "redis/object.h"
#include "redis/zmalloc.h"
}
#include <jsoncons/json.hpp>

#include "base/logging.h"

namespace dfly {

std::optional<JsonType> JsonFromString(std::string_view input) {
  using namespace jsoncons;

  std::error_code ec;
  auto JsonErrorHandler = [](json_errc ec, const ser_context&) {
    VLOG(1) << "Error while decode JSON: " << make_error_code(ec).message();
    return false;
  };

  json_decoder<JsonType> decoder;
  json_parser parser(basic_json_decode_options<char>{}, JsonErrorHandler);

  parser.update(input);
  parser.finish_parse(decoder, ec);

  if (decoder.is_valid()) {
    return decoder.get_result();
  }
  return {};
}

}  // namespace dfly
