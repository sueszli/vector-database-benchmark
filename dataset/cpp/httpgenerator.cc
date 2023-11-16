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
#include <eventql/util/exception.h>
#include <eventql/util/stringutil.h>
#include <eventql/util/http/httpgenerator.h>

namespace http {

void HTTPGenerator::generateHeaders(const HTTPRequest& req, OutputStream* os) {
  os->write(
      StringUtil::format(
          "$0 $1 $2\r\n",
          req.method(),
          req.uri(),
          req.version()));

  for (const auto& header : req.headers()) {
    os->write(StringUtil::format("$0: $1\r\n", header.first, header.second));
  }

  if (req.body().size() > 0 && !req.hasHeader("Content-Length")) {
    os->write(StringUtil::format("Content-Length: $0\r\n", req.body().size()));
  }

  os->write("\r\n");
}

void HTTPGenerator::generate(const HTTPRequest& req, OutputStream* os) {
  generateHeaders(req, os);

  const auto& body = req.body();
  if (body.size() > 0) {
    os->write(body);
  }
}

void HTTPGenerator::generateHeaders(const HTTPResponse& res, OutputStream* os) {
  if (res.version().length() < 4) {
    RAISEF(kRuntimeError, "invalid http version: $0", res.version());
  }

  os->write(
      StringUtil::format(
          "$0 $1 $2\r\n",
          res.version(),
          res.statusCode(),
          res.statusName()));

  for (const auto& header : res.headers()) {
    os->write(StringUtil::format("$0: $1\r\n", header.first, header.second));
  }

  os->write("\r\n");
}

void HTTPGenerator::generate(const HTTPResponse& res, OutputStream* os) {
  generateHeaders(res, os);

  const auto& body = res.body();
  if (body.size() > 0) {
    os->write(body);
  }
}

}

