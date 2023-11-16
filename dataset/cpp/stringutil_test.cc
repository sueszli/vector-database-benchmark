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
#include <stdlib.h>
#include <stdio.h>
#include "eventql/util/stringutil.h"
#include "eventql/util/test/unittest.h"

#include "eventql/eventql.h"

UNIT_TEST(StringUtilTest);

TEST_CASE(StringUtilTest, TestToString, [] () {
  EXPECT_EQ(StringUtil::toString(123), "123");
  EXPECT_EQ(StringUtil::toString(1230000000), "1230000000");
  EXPECT_EQ(StringUtil::toString(24.5), "24.5");
  EXPECT_EQ(StringUtil::toString("abc"), "abc");
  EXPECT_EQ(StringUtil::toString(std::string("abc")), "abc");
});

TEST_CASE(StringUtilTest, TestStripTrailingSlashes, [] () {
  std::string s1 = "fnord/bar/";
  StringUtil::stripTrailingSlashes(&s1);
  EXPECT_EQ(s1, "fnord/bar");

  std::string s2 = "fnord/bar///";
  StringUtil::stripTrailingSlashes(&s2);
  EXPECT_EQ(s2, "fnord/bar");

  std::string s3 = "fnord/bar";
  StringUtil::stripTrailingSlashes(&s3);
  EXPECT_EQ(s3, "fnord/bar");

  std::string s4 = "/";
  StringUtil::stripTrailingSlashes(&s4);
  EXPECT_EQ(s4, "");
});

TEST_CASE(StringUtilTest, TestBeginsWith, [] () {
  EXPECT_TRUE(StringUtil::beginsWith("fnord", "fn"));
  EXPECT_TRUE(StringUtil::beginsWith("fnahrad", "fn"));
  EXPECT_FALSE(StringUtil::beginsWith("ford", "fn"));
  EXPECT_TRUE(StringUtil::beginsWith("fnord", "fnord"));
  EXPECT_FALSE(StringUtil::beginsWith("fnord", "fnordbar"));
});

TEST_CASE(StringUtilTest, TestEndsWith, [] () {
  EXPECT_TRUE(StringUtil::endsWith("fnord", "ord"));
  EXPECT_TRUE(StringUtil::endsWith("ford", "ord"));
  EXPECT_FALSE(StringUtil::endsWith("ford", "x"));
  EXPECT_FALSE(StringUtil::endsWith("ford", "fnord"));
  EXPECT_TRUE(StringUtil::endsWith("fnord", "fnord"));
  EXPECT_FALSE(StringUtil::endsWith("fnord", "fnordbar"));
});

TEST_CASE(StringUtilTest, TestHexPrint, [] () {
  auto data1 = "\x17\x23\x42\x01";
  EXPECT_EQ(StringUtil::hexPrint(data1, 4), "17 23 42 01");
  EXPECT_EQ(StringUtil::hexPrint(data1, 4, false), "17234201");
  EXPECT_EQ(StringUtil::hexPrint(data1, 4, true, true), "01 42 23 17");
  EXPECT_EQ(StringUtil::hexPrint(data1, 4, false, true), "01422317");
});

TEST_CASE(StringUtilTest, TestReplaceAll, [] () {
  std::string str =
      "cloud computing, or in simpler shorthand just >the cloud<...";

  StringUtil::replaceAll(&str, "cloud", "butt");
  EXPECT_EQ(str, "butt computing, or in simpler shorthand just >the butt<...");

  StringUtil::replaceAll(&str, "butt", "");
  EXPECT_EQ(str, " computing, or in simpler shorthand just >the <...");
});

TEST_CASE(StringUtilTest, TestFormat, [] () {
  auto str1 = StringUtil::format(
      "The quick $0 fox jumps over the $1 dog",
      "brown",
      "lazy");

  EXPECT_EQ(str1, "The quick brown fox jumps over the lazy dog");

  auto str2 = StringUtil::format(
      "1 $3 2 $2 3 $1 4 $0 5 $8 6 $9",
      "A",
      "B",
      "C",
      "D",
      "E",
      "F",
      "G",
      "H",
      "I",
      "K");

  EXPECT_EQ(str2, "1 D 2 C 3 B 4 A 5 I 6 K");

  auto str3 = StringUtil::format("$0 + $1 = $2", 2.5, 6.5, 9);
  EXPECT_EQ(str3, "2.5 + 6.5 = 9");

  auto str5 = StringUtil::format("$0, $1", 1.0, 0.0625);
  EXPECT_EQ(str5, "1.0, 0.0625");

  auto str4 = StringUtil::format("$1$1$1$1$1 $0", "Batman", "Na");
  EXPECT_EQ(str4, "NaNaNaNaNa Batman");
});

TEST_CASE(StringUtilTest, TestSplit, [] () {
  auto parts1 = StringUtil::split("one,two,three", ",");
  EXPECT_EQ(parts1.size(), 3);
  EXPECT_EQ(parts1[0], "one");
  EXPECT_EQ(parts1[1], "two");
  EXPECT_EQ(parts1[2], "three");

  auto parts2 = StringUtil::split("onexxtwoxxthree", "xx");
  EXPECT_EQ(parts2.size(), 3);
  EXPECT_EQ(parts2[0], "one");
  EXPECT_EQ(parts2[1], "two");
  EXPECT_EQ(parts2[2], "three");
});
