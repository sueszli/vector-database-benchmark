/**
 * Copyright (c) 2016 DeepCortex GmbH <legal@eventql.io>
 * Authors:
 *   - Paul Asmuth <paul@eventql.io>
 *   - Laura Schlimmer <laura@eventql.io>
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
#include "eventql/eventql.h"
#include <eventql/util/stdtypes.h>
#include <eventql/util/exception.h>
#include <eventql/util/wallclock.h>
#include <eventql/util/test/unittest.h>
#include <eventql/config/process_config.h>

using namespace eventql;
UNIT_TEST(ProcessConfigTest);

TEST_CASE(ProcessConfigTest, TestProcessConfigBuilder, [] () {
  ProcessConfigBuilder builder;
  builder.setProperty("evql", "host", "localhost");
  builder.setProperty("evql", "port", "8080");

  auto config = builder.getConfig();
  {
    auto p = config->getString("evql", "host");
    EXPECT_FALSE(p.isEmpty());
    EXPECT_EQ(p.get(), "localhost");
  }

  {
    auto p = config->getInt("evql", "port");
    EXPECT_FALSE(p.isEmpty());
    EXPECT_EQ(p.get(), 8080);
  }
});

TEST_CASE(ProcessConfigTest, TestProcessConfigBuilderLoadFile, [] () {
  auto test_file_path = "eventql/config/testdata/.process_cfg";
  ProcessConfigBuilder builder;

  auto status = builder.loadFile(test_file_path);
  EXPECT_TRUE(status.isSuccess());

  builder.setProperty("test", "port", "9175");

  auto config = builder.getConfig();
  {
    auto p = config->getString("test", "host");
    EXPECT_FALSE(p.isEmpty());
    EXPECT_EQ(p.get(), "localhost");
  }
  {
    auto p = config->getInt("test", "port");
    EXPECT_FALSE(p.isEmpty());
    EXPECT_EQ(p.get(), 9175);
  }
  {
    auto p = config->getString("test", "authors");
    EXPECT_FALSE(p.isEmpty());
    EXPECT_EQ(p.get(), "eventQL Authors");
  }
  {
    auto p = config->getString("test2", "mail");
    EXPECT_FALSE(p.isEmpty());
    EXPECT_EQ(p.get(), "authors@test.com");
  }
});
