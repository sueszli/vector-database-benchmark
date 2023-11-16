// Copyright 2022, DragonflyDB authors.  All rights reserved.
// See LICENSE for licensing terms.
//

#include "server/acl/acl_family.h"

#include "absl/container/flat_hash_map.h"
#include "absl/flags/internal/flag.h"
#include "absl/strings/str_cat.h"
#include "base/gtest.h"
#include "base/logging.h"
#include "facade/facade_test.h"
#include "server/acl/acl_commands_def.h"
#include "server/command_registry.h"
#include "server/test_utils.h"

using namespace testing;

ABSL_DECLARE_FLAG(std::vector<std::string>, rename_command);

namespace dfly {

class AclFamilyTest : public BaseFamilyTest {
 protected:
};

class AclFamilyTestRename : public BaseFamilyTest {
  void SetUp() override {
    absl::SetFlag(&FLAGS_rename_command, {"ACL=ROCKS"});
    ResetService();
  }
};

TEST_F(AclFamilyTest, AclSetUser) {
  TestInitAclFam();
  auto resp = Run({"ACL", "SETUSER"});
  EXPECT_THAT(resp, ErrArg("ERR wrong number of arguments for 'acl setuser' command"));

  resp = Run({"ACL", "SETUSER", "kostas", "ONN"});
  EXPECT_THAT(resp, ErrArg("ERR Unrecognized parameter ONN"));

  resp = Run({"ACL", "SETUSER", "kostas", "+@nonsense"});
  EXPECT_THAT(resp, ErrArg("ERR Unrecognized parameter +@NONSENSE"));

  resp = Run({"ACL", "SETUSER", "vlad"});
  EXPECT_THAT(resp, "OK");
  resp = Run({"ACL", "LIST"});
  auto vec = resp.GetVec();
  EXPECT_THAT(vec, UnorderedElementsAre("user default on nopass +@ALL +ALL",
                                        "user vlad off nopass +@NONE"));

  resp = Run({"ACL", "SETUSER", "vlad", "+ACL"});
  EXPECT_THAT(resp, "OK");

  resp = Run({"ACL", "LIST"});
  vec = resp.GetVec();
  EXPECT_THAT(vec, UnorderedElementsAre("user default on nopass +@ALL +ALL",
                                        "user vlad off nopass +@NONE +ACL"));
}

TEST_F(AclFamilyTest, AclDelUser) {
  TestInitAclFam();
  auto resp = Run({"ACL", "DELUSER"});
  EXPECT_THAT(resp, ErrArg("ERR wrong number of arguments for 'acl deluser' command"));

  resp = Run({"ACL", "DELUSER", "default"});
  EXPECT_THAT(resp, ErrArg("ERR The'default' user cannot be removed"));

  resp = Run({"ACL", "DELUSER", "NOTEXISTS"});
  EXPECT_THAT(resp, ErrArg("ERR User NOTEXISTS does not exist"));

  resp = Run({"ACL", "SETUSER", "kostas", "ON"});
  EXPECT_THAT(resp, "OK");

  resp = Run({"ACL", "DELUSER", "KOSTAS", "NONSENSE"});
  EXPECT_THAT(resp, ErrArg("ERR wrong number of arguments for 'acl deluser' command"));

  resp = Run({"ACL", "DELUSER", "kostas"});
  EXPECT_THAT(resp, "OK");

  resp = Run({"ACL", "LIST"});
  EXPECT_THAT(resp.GetString(), "user default on nopass +@ALL +ALL");
}

TEST_F(AclFamilyTest, AclList) {
  TestInitAclFam();
  auto resp = Run({"ACL", "LIST", "NONSENSE"});
  EXPECT_THAT(resp, ErrArg("ERR wrong number of arguments for 'acl list' command"));

  resp = Run({"ACL", "SETUSER", "kostas", ">pass", "+@admin"});
  EXPECT_THAT(resp, "OK");

  resp = Run({"ACL", "SETUSER", "adi", ">pass", "+@fast"});
  EXPECT_THAT(resp, "OK");

  resp = Run({"ACL", "LIST"});
  auto vec = resp.GetVec();
  EXPECT_THAT(vec, UnorderedElementsAre("user default on nopass +@ALL +ALL",
                                        "user kostas off d74ff0ee8da3b98 +@ADMIN",
                                        "user adi off d74ff0ee8da3b98 +@FAST"));
}

TEST_F(AclFamilyTest, AclAuth) {
  TestInitAclFam();
  auto resp = Run({"AUTH", "default", R"("")"});
  EXPECT_THAT(resp, "OK");

  resp = Run({"ACL", "SETUSER", "shahar", ">mypass"});
  EXPECT_THAT(resp, "OK");

  resp = Run({"AUTH", "shahar", "wrongpass"});
  EXPECT_THAT(resp, ErrArg("WRONGPASS invalid username-password pair or user is disabled."));

  resp = Run({"AUTH", "shahar", "mypass"});
  EXPECT_THAT(resp, ErrArg("WRONGPASS invalid username-password pair or user is disabled."));

  // Activate the user
  resp = Run({"ACL", "SETUSER", "shahar", "ON", "+@fast"});
  EXPECT_THAT(resp, "OK");

  resp = Run({"AUTH", "shahar", "mypass"});
  EXPECT_THAT(resp, "OK");
}

TEST_F(AclFamilyTest, AclWhoAmI) {
  TestInitAclFam();
  auto resp = Run({"ACL", "WHOAMI", "WHO"});
  EXPECT_THAT(resp, ErrArg("ERR wrong number of arguments for 'acl whoami' command"));

  resp = Run({"ACL", "SETUSER", "kostas", "ON", ">pass", "+@SLOW"});
  EXPECT_THAT(resp, "OK");

  resp = Run({"AUTH", "kostas", "pass"});
  EXPECT_THAT(resp, "OK");

  resp = Run({"ACL", "WHOAMI"});
  EXPECT_THAT(resp, "User is kostas");
}

TEST_F(AclFamilyTest, TestAllCategories) {
  TestInitAclFam();
  for (auto& cat : acl::REVERSE_CATEGORY_INDEX_TABLE) {
    if (cat != "_RESERVED") {
      auto resp = Run({"ACL", "SETUSER", "kostas", absl::StrCat("+@", cat)});
      EXPECT_THAT(resp, "OK");

      resp = Run({"ACL", "LIST"});
      EXPECT_THAT(resp.GetVec(),
                  UnorderedElementsAre("user default on nopass +@ALL +ALL",
                                       absl::StrCat("user kostas off nopass ", "+@", cat)));

      resp = Run({"ACL", "SETUSER", "kostas", absl::StrCat("-@", cat)});
      EXPECT_THAT(resp, "OK");

      resp = Run({"ACL", "LIST"});
      EXPECT_THAT(resp.GetVec(),
                  UnorderedElementsAre("user default on nopass +@ALL +ALL",
                                       absl::StrCat("user kostas off nopass ", "+@NONE")));

      resp = Run({"ACL", "DELUSER", "kostas"});
      EXPECT_THAT(resp, "OK");
    }
  }

  for (auto& cat : acl::REVERSE_CATEGORY_INDEX_TABLE) {
    if (cat != "_RESERVED") {
      auto resp = Run({"ACL", "SETUSER", "kostas", absl::StrCat("+@", cat)});
      EXPECT_THAT(resp, "OK");
    }
  }
  // This won't work because of __RESERVED
  // TODO(fix this)
  //  auto resp = Run({"ACL", "LIST"});
  //  EXPECT_THAT(resp.GetVec(), UnorderedElementsAre("user default on nopass +@ALL",
  //  absl::StrCat("user kostas off nopass ", "+@ALL")));
  //

  // TODO(Bug here fix none/all)
  //  auto resp = Run({"ACL", "SETUSER", "kostas", "+@NONE"});
  //  EXPECT_THAT(resp, "OK");
  //
  //  resp = Run({"ACL", "LIST"});
  //  EXPECT_THAT(resp.GetVec(), UnorderedElementsAre("user default on nopass +@ALL", "user kostas
  //  off nopass +@NONE"));
}

TEST_F(AclFamilyTest, TestAllCommands) {
  TestInitAclFam();
  const auto& rev_indexer = acl::CommandsRevIndexer();
  for (const auto& family : rev_indexer) {
    for (const auto& command_name : family) {
      auto resp = Run({"ACL", "SETUSER", "kostas", absl::StrCat("+", command_name)});
      EXPECT_THAT(resp, "OK");

      resp = Run({"ACL", "LIST"});
      EXPECT_THAT(resp.GetVec(), UnorderedElementsAre("user default on nopass +@ALL +ALL",
                                                      absl::StrCat("user kostas off nopass +@NONE ",
                                                                   "+", command_name)));

      resp = Run({"ACL", "SETUSER", "kostas", absl::StrCat("-", command_name)});

      resp = Run({"ACL", "LIST"});
      EXPECT_THAT(resp.GetVec(),
                  UnorderedElementsAre("user default on nopass +@ALL +ALL",
                                       absl::StrCat("user kostas off nopass ", "+@NONE")));

      resp = Run({"ACL", "DELUSER", "kostas"});
      EXPECT_THAT(resp, "OK");
    }
  }
}

TEST_F(AclFamilyTest, TestUsers) {
  TestInitAclFam();
  auto resp = Run({"ACL", "SETUSER", "abhra", "ON"});
  EXPECT_THAT(resp, "OK");

  resp = Run({"ACL", "SETUSER", "ari"});
  EXPECT_THAT(resp, "OK");

  resp = Run({"ACL", "USERS"});
  EXPECT_THAT(resp.GetVec(), UnorderedElementsAre("default", "abhra", "ari"));
}

TEST_F(AclFamilyTest, TestCat) {
  TestInitAclFam();
  auto resp = Run({"ACL", "CAT", "nonsense"});
  EXPECT_THAT(resp, ErrArg("ERR Unkown category: NONSENSE"));

  resp = Run({"ACL", "CAT"});
  EXPECT_THAT(resp.GetVec().size(), 24);

  resp = Run({"ACL", "CAT", "STRING"});

  EXPECT_THAT(resp.GetVec(),
              UnorderedElementsAre("GETSET", "GETRANGE", "INCRBYFLOAT", "GETDEL", "DECRBY",
                                   "PREPEND", "SETEX", "MSET", "SET", "PSETEX", "SUBSTR", "DECR",
                                   "STRLEN", "INCR", "INCRBY", "MGET", "GET", "SETNX", "GETEX",
                                   "APPEND", "MSETNX", "SETRANGE"));
}

TEST_F(AclFamilyTest, TestGetUser) {
  TestInitAclFam();
  auto resp = Run({"ACL", "GETUSER", "kostas"});
  EXPECT_THAT(resp, ErrArg("ERR User: kostas does not exists!"));

  resp = Run({"ACL", "GETUSER", "default"});
  const auto& vec = resp.GetVec();
  EXPECT_THAT(vec[0], "flags");
  EXPECT_THAT(vec[1].GetVec(), UnorderedElementsAre("on", "nopass"));
  EXPECT_THAT(vec[2], "passwords");
  EXPECT_TRUE(vec[3].GetVec().empty());
  EXPECT_THAT(vec[4], "commands");
  EXPECT_THAT(vec[5], "+@ALL +ALL");

  resp = Run({"ACL", "SETUSER", "kostas", "+@STRING", "+HSET"});
  resp = Run({"ACL", "GETUSER", "kostas"});
  const auto& kvec = resp.GetVec();
  EXPECT_THAT(kvec[0], "flags");
  EXPECT_THAT(kvec[1].GetVec(), UnorderedElementsAre("off", "nopass"));
  EXPECT_THAT(kvec[2], "passwords");
  EXPECT_TRUE(kvec[3].GetVec().empty());
  EXPECT_THAT(kvec[4], "commands");
  EXPECT_THAT(kvec[5], "+@STRING +HSET");
}

TEST_F(AclFamilyTest, TestDryRun) {
  TestInitAclFam();
  auto resp = Run({"ACL", "DRYRUN"});
  EXPECT_THAT(resp, ErrArg("ERR wrong number of arguments for 'acl dryrun' command"));

  resp = Run({"ACL", "DRYRUN", "default"});
  EXPECT_THAT(resp, ErrArg("ERR wrong number of arguments for 'acl dryrun' command"));

  resp = Run({"ACL", "DRYRUN", "default", "get", "more"});
  EXPECT_THAT(resp, ErrArg("ERR wrong number of arguments for 'acl dryrun' command"));

  resp = Run({"ACL", "DRYRUN", "kostas", "more"});
  EXPECT_THAT(resp, ErrArg("ERR User: kostas does not exists!"));

  resp = Run({"ACL", "DRYRUN", "default", "nope"});
  EXPECT_THAT(resp, ErrArg("ERR Command: NOPE does not exists!"));

  resp = Run({"ACL", "DRYRUN", "default", "SET"});
  EXPECT_THAT(resp, "OK");

  resp = Run({"ACL", "SETUSER", "kostas", "+GET"});
  EXPECT_THAT(resp, "OK");

  resp = Run({"ACL", "DRYRUN", "kostas", "GET"});
  EXPECT_THAT(resp, "OK");

  resp = Run({"ACL", "DRYRUN", "kostas", "SET"});
  EXPECT_THAT(resp, ErrArg("ERR User: kostas is not allowed to execute command: SET"));
}

TEST_F(AclFamilyTest, AclGenPassTooManyArguments) {
  TestInitAclFam();

  auto resp = Run({"ACL", "GENPASS", "1", "2"});
  EXPECT_THAT(resp.GetString(),
              "ERR Unknown subcommand or wrong number of arguments for 'GENPASS'. Try ACL HELP.");
}

TEST_F(AclFamilyTest, AclGenPassOutOfRange) {
  std::string expectedError =
      "ERR ACL GENPASS argument must be the number of bits for the output password, a positive "
      "number up to 4096";

  auto resp = Run({"ACL", "GENPASS", "-1"});
  EXPECT_THAT(resp.GetString(), expectedError);

  resp = Run({"ACL", "GENPASS", "0"});
  EXPECT_THAT(resp.GetString(), expectedError);

  resp = Run({"ACL", "GENPASS", "4097"});
  EXPECT_THAT(resp.GetString(), expectedError);
}

TEST_F(AclFamilyTest, AclGenPass) {
  auto resp = Run({"ACL", "GENPASS"});
  auto actualPassword = resp.GetString();

  // should be 256 bits or 64 bytes in hex
  EXPECT_THAT(actualPassword.length(), 64);

  // 1 bit - 4 bits should all produce a single hex character
  for (int i = 1; i <= 4; i++) {
    resp = Run({"ACL", "GENPASS", std::to_string(i)});
    EXPECT_THAT(resp.GetString().length(), 1);
  }
  // 5 bits - 8 bits should all produce two hex characters
  for (int i = 5; i <= 8; i++) {
    resp = Run({"ACL", "GENPASS", std::to_string(i)});
    EXPECT_THAT(resp.GetString().length(), 2);
  }

  // and the pattern continues
  resp = Run({"ACL", "GENPASS", "9"});
  EXPECT_THAT(resp.GetString().length(), 3);
}

TEST_F(AclFamilyTestRename, AclRename) {
  auto resp = Run({"ACL", "SETUSER", "billy"});
  EXPECT_THAT(resp, ErrArg("ERR unknown command `ACL`"));

  resp = Run({"ROCKS", "SETUSER", "billy", "ON", ">mypass"});
  EXPECT_THAT(resp.GetString(), "OK");

  resp = Run({"ROCKS", "DELUSER", "billy"});
  EXPECT_THAT(resp.GetString(), "OK");
}

}  // namespace dfly
