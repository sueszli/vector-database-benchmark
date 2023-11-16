// Copyright 2022, DragonflyDB authors.  All rights reserved.
// See LICENSE for licensing terms.
//

#include "server/list_family.h"

#include <absl/strings/match.h>

#include "base/gtest.h"
#include "base/logging.h"
#include "facade/facade_test.h"
#include "server/blocking_controller.h"
#include "server/command_registry.h"
#include "server/conn_context.h"
#include "server/engine_shard_set.h"
#include "server/string_family.h"
#include "server/test_utils.h"
#include "server/transaction.h"

using namespace testing;
using namespace std;
using namespace util;
using absl::StrCat;

namespace dfly {

class ListFamilyTest : public BaseFamilyTest {
 protected:
  ListFamilyTest() {
    num_threads_ = 4;
  }

  static unsigned NumWatched() {
    atomic_uint32_t sum{0};
    shard_set->RunBriefInParallel([&](EngineShard* es) {
      auto* bc = es->blocking_controller();
      if (bc)
        sum.fetch_add(bc->NumWatched(0), memory_order_relaxed);
    });

    return sum.load();
  }

  static bool HasAwakened() {
    atomic_uint32_t sum{0};
    shard_set->RunBriefInParallel([&](EngineShard* es) {
      auto* bc = es->blocking_controller();
      if (bc)
        sum.fetch_add(bc->HasAwakedTransaction(), memory_order_relaxed);
    });

    return sum.load() > 0;
  }
};

const char kKey1[] = "x";
const char kKey2[] = "b";
const char kKey3[] = "c";

TEST_F(ListFamilyTest, Basic) {
  auto resp = Run({"lpush", kKey1, "1"});
  EXPECT_THAT(resp, IntArg(1));
  resp = Run({"lpush", kKey2, "2"});
  ASSERT_THAT(resp, IntArg(1));
  resp = Run({"llen", kKey1});
  ASSERT_THAT(resp, IntArg(1));
}

TEST_F(ListFamilyTest, Expire) {
  auto resp = Run({"lpush", kKey1, "1"});
  EXPECT_THAT(resp, IntArg(1));

  resp = Run({"expire", kKey1, "1"});
  EXPECT_THAT(resp, IntArg(1));

  AdvanceTime(1000);

  resp = Run({"lpush", kKey1, "1"});
  EXPECT_THAT(resp, IntArg(1));
}

TEST_F(ListFamilyTest, BLPopUnblocking) {
  auto resp = Run({"lpush", kKey1, "1"});
  EXPECT_THAT(resp, IntArg(1));
  resp = Run({"lpush", kKey2, "2"});
  ASSERT_THAT(resp, IntArg(1));

  resp = Run({"blpop", kKey1, kKey2});  // missing "0" delimiter.
  ASSERT_THAT(resp, ErrArg("timeout is not a float"));

  resp = Run({"blpop", kKey1, kKey2, "0"});
  ASSERT_EQ(2, GetDebugInfo().shards_count);
  ASSERT_THAT(resp, ArrLen(2));
  EXPECT_THAT(resp.GetVec(), ElementsAre(kKey1, "1"));

  resp = Run({"blpop", kKey1, kKey2, "0"});
  ASSERT_THAT(resp, ArrLen(2));
  EXPECT_THAT(resp.GetVec(), ElementsAre(kKey2, "2"));

  Run({"set", "z", "1"});

  resp = Run({"blpop", "z", "0"});
  ASSERT_THAT(resp, ErrArg("WRONGTYPE "));

  ASSERT_FALSE(IsLocked(0, "x"));
  ASSERT_FALSE(IsLocked(0, "y"));
  ASSERT_FALSE(IsLocked(0, "z"));
}

TEST_F(ListFamilyTest, BLPopBlocking) {
  RespExpr resp0, resp1;

  // Run the fiber at creation.
  auto fb0 = pp_->at(0)->LaunchFiber(Launch::dispatch, [&] {
    resp0 = Run({"blpop", "x", "0"});
    LOG(INFO) << "pop0";
  });

  ThisFiber::SleepFor(50us);
  auto fb1 = pp_->at(1)->LaunchFiber([&] {
    resp1 = Run({"blpop", "x", "0"});
    LOG(INFO) << "pop1";
  });
  ThisFiber::SleepFor(30us);

  RespExpr resp = pp_->at(1)->Await([&] { return Run("B1", {"lpush", "x", "2", "1"}); });
  ASSERT_THAT(resp, IntArg(2));

  fb0.Join();
  fb1.Join();

  // fb0 should start first and be the first transaction blocked. Therefore, it should pop '1'.
  // sometimes order is switched, need to think how to fix it.
  int64_t epoch0 = GetDebugInfo("IO0").clock;
  int64_t epoch1 = GetDebugInfo("IO1").clock;
  ASSERT_LT(epoch0, epoch1);
  ASSERT_THAT(resp0, ArrLen(2));
  EXPECT_THAT(resp0.GetVec(), ElementsAre("x", "1"));
  ASSERT_FALSE(IsLocked(0, "x"));
  ASSERT_EQ(0, NumWatched());
}

TEST_F(ListFamilyTest, BLPopMultiple) {
  RespExpr resp0, resp1;

  resp0 = Run({"blpop", kKey1, kKey2, "0.01"});  // timeout
  EXPECT_THAT(resp0, ArgType(RespExpr::NIL_ARRAY));
  ASSERT_EQ(2, GetDebugInfo().shards_count);

  ASSERT_FALSE(IsLocked(0, kKey1));
  ASSERT_FALSE(IsLocked(0, kKey2));

  auto fb1 = pp_->at(0)->LaunchFiber(Launch::dispatch, [&] {
    resp0 = Run({"blpop", kKey1, kKey2, "0"});
  });

  pp_->at(1)->Await([&] { Run({"lpush", kKey1, "1", "2", "3"}); });
  fb1.Join();

  ASSERT_THAT(resp0, ArrLen(2));
  EXPECT_THAT(resp0.GetVec(), ElementsAre(kKey1, "3"));
  ASSERT_FALSE(IsLocked(0, kKey1));
  ASSERT_FALSE(IsLocked(0, kKey2));
  ASSERT_EQ(0, NumWatched());
}

TEST_F(ListFamilyTest, BLPopTimeout) {
  RespExpr resp = Run({"blpop", kKey1, kKey2, kKey3, "0.01"});
  EXPECT_THAT(resp, ArgType(RespExpr::NIL_ARRAY));
  EXPECT_EQ(3, GetDebugInfo().shards_count);
  ASSERT_FALSE(service_->IsLocked(0, kKey1));

  // Under Multi
  resp = Run({"multi"});
  ASSERT_EQ(resp, "OK");

  Run({"blpop", kKey1, "0"});
  resp = Run({"exec"});

  EXPECT_THAT(resp, ArgType(RespExpr::NIL_ARRAY));
  ASSERT_FALSE(service_->IsLocked(0, kKey1));
  ASSERT_EQ(0, NumWatched());
}

TEST_F(ListFamilyTest, BLPopTimeout2) {
  Run({"BLPOP", "blist1", "blist2", "0.1"});

  Run({"RPUSH", "blist2", "d"});
  Run({"RPUSH", "blist2", "hello"});

  auto resp = Run({"BLPOP", "blist1", "blist2", "1"});
  ASSERT_THAT(resp, ArrLen(2));
  ASSERT_THAT(resp.GetVec(), ElementsAre("blist2", "d"));

  Run({"RPUSH", "blist1", "a"});
  Run({"DEL", "blist2"});
  Run({"RPUSH", "blist2", "d"});
  Run({"BLPOP", "blist1", "blist2", "1"});
  ASSERT_EQ(0, NumWatched());
}

TEST_F(ListFamilyTest, BLPopMultiPush) {
  Run({"exists", kKey1, kKey2, kKey3});
  ASSERT_EQ(3, GetDebugInfo().shards_count);
  RespExpr blpop_resp;
  auto pop_fb = pp_->at(0)->LaunchFiber(Launch::dispatch, [&] {
    blpop_resp = Run({"blpop", kKey1, kKey2, kKey3, "0"});
  });

  WaitUntilLocked(0, kKey1);

  auto p1_fb = pp_->at(1)->LaunchFiber([&] {
    for (unsigned i = 0; i < 100; ++i) {
      // a filler command to create scheduling queue.
      Run({"exists", kKey1, kKey2, kKey3});
    }
  });

  auto p2_fb = pp_->at(2)->LaunchFiber([&] {
    Run({"multi"});
    Run({"lpush", kKey3, "C"});
    Run({"exists", kKey2});
    Run({"lpush", kKey2, "B"});
    Run({"exists", kKey1});
    Run({"lpush", kKey1, "A"});
    Run({"exists", kKey1, kKey2, kKey3});
    auto resp = Run({"exec"});
    ASSERT_THAT(resp, ArrLen(6));
  });

  p1_fb.Join();
  p2_fb.Join();

  pop_fb.Join();

  // We can't determine what key was popped, so only check result presence.
  // It might not be first kKey3 "C" because of squashing and re-ordering.
  ASSERT_THAT(blpop_resp, ArrLen(2));
  ASSERT_THAT(Run({"exists", kKey1, kKey2, kKey3}), IntArg(2));
  ASSERT_EQ(0, NumWatched());
}

TEST_F(ListFamilyTest, WrongTypeDoesNotWake) {
  RespExpr blpop_resp;

  auto pop_fb = pp_->at(0)->LaunchFiber(Launch::dispatch, [&] {
    blpop_resp = Run({"blpop", kKey1, "0"});
  });

  WaitUntilLocked(0, kKey1);

  auto p1_fb = pp_->at(1)->LaunchFiber([&] {
    Run({"multi"});
    Run({"lpush", kKey1, "A"});
    Run({"set", kKey1, "foo"});

    auto resp = Run({"exec"});
    EXPECT_THAT(resp.GetVec(), ElementsAre(IntArg(1), "OK"));

    Run({"del", kKey1});
    Run({"lpush", kKey1, "B"});
  });

  p1_fb.Join();
  pop_fb.Join();
  ASSERT_THAT(blpop_resp, ArrLen(2));
  EXPECT_THAT(blpop_resp.GetVec(), ElementsAre(kKey1, "B"));
}

TEST_F(ListFamilyTest, BPopSameKeyTwice) {
  RespExpr blpop_resp;

  auto pop_fb = pp_->at(0)->LaunchFiber(Launch::dispatch, [&] {
    blpop_resp = Run({"blpop", kKey1, kKey2, kKey2, kKey1, "0"});
    EXPECT_EQ(0, NumWatched());
  });

  WaitUntilLocked(0, kKey1);

  pp_->at(1)->Await([&] { EXPECT_EQ(1, CheckedInt({"lpush", kKey1, "bar"})); });
  pop_fb.Join();

  ASSERT_THAT(blpop_resp, ArrLen(2));
  EXPECT_THAT(blpop_resp.GetVec(), ElementsAre(kKey1, "bar"));

  pop_fb = pp_->at(0)->LaunchFiber(Launch::dispatch, [&] {
    blpop_resp = Run({"blpop", kKey1, kKey2, kKey2, kKey1, "0"});
  });

  WaitUntilLocked(0, kKey1);

  pp_->at(1)->Await([&] { EXPECT_EQ(1, CheckedInt({"lpush", kKey2, "bar"})); });
  pop_fb.Join();

  ASSERT_THAT(blpop_resp, ArrLen(2));
  EXPECT_THAT(blpop_resp.GetVec(), ElementsAre(kKey2, "bar"));
}

TEST_F(ListFamilyTest, BPopTwoKeysSameShard) {
  Run({"exists", "x", "y"});
  ASSERT_EQ(1, GetDebugInfo().shards_count);
  RespExpr blpop_resp;

  auto pop_fb = pp_->at(0)->LaunchFiber(Launch::dispatch, [&] {
    blpop_resp = Run({"blpop", "x", "y", "0"});
    EXPECT_FALSE(IsLocked(0, "y"));
    ASSERT_EQ(0, NumWatched());
  });

  WaitUntilLocked(0, "x");

  pp_->at(1)->Await([&] { EXPECT_EQ(1, CheckedInt({"lpush", "x", "bar"})); });
  pop_fb.Join();

  ASSERT_THAT(blpop_resp, ArrLen(2));
  EXPECT_THAT(blpop_resp.GetVec(), ElementsAre("x", "bar"));
}

TEST_F(ListFamilyTest, BPopRename) {
  RespExpr blpop_resp;

  Run({"exists", kKey1, kKey2});
  ASSERT_EQ(2, GetDebugInfo().shards_count);

  auto pop_fb = pp_->at(0)->LaunchFiber(Launch::dispatch, [&] {
    blpop_resp = Run({"blpop", kKey1, "0"});
  });

  WaitUntilLocked(0, kKey1);

  pp_->at(1)->Await([&] {
    EXPECT_EQ(1, CheckedInt({"lpush", "a", "bar"}));
    Run({"rename", "a", kKey1});
  });
  pop_fb.Join();

  ASSERT_THAT(blpop_resp, ArrLen(2));
  EXPECT_THAT(blpop_resp.GetVec(), ElementsAre(kKey1, "bar"));
}

TEST_F(ListFamilyTest, BPopFlush) {
  RespExpr blpop_resp;
  auto pop_fb = pp_->at(0)->LaunchFiber(Launch::dispatch, [&] {
    blpop_resp = Run({"blpop", kKey1, "0"});
  });

  WaitUntilLocked(0, kKey1);

  pp_->at(1)->Await([&] {
    Run({"flushdb"});
    EXPECT_EQ(1, CheckedInt({"lpush", kKey1, "bar"}));
  });
  pop_fb.Join();
}

TEST_F(ListFamilyTest, LRem) {
  auto resp = Run({"rpush", kKey1, "a", "b", "a", "c"});
  ASSERT_THAT(resp, IntArg(4));
  resp = Run({"lrem", kKey1, "2", "a"});
  ASSERT_THAT(resp, IntArg(2));

  resp = Run({"lrange", kKey1, "0", "1"});
  ASSERT_THAT(resp, ArrLen(2));
  ASSERT_THAT(resp.GetVec(), ElementsAre("b", "c"));
}

TEST_F(ListFamilyTest, LTrim) {
  Run({"rpush", kKey1, "a", "b", "c", "d"});
  ASSERT_EQ(Run({"ltrim", kKey1, "-2", "-1"}), "OK");
  auto resp = Run({"lrange", kKey1, "0", "1"});
  ASSERT_THAT(resp, ArrLen(2));
  ASSERT_THAT(resp.GetVec(), ElementsAre("c", "d"));
  ASSERT_EQ(Run({"ltrim", kKey1, "0", "0"}), "OK");
  ASSERT_EQ(Run({"lrange", kKey1, "0", "1"}), "c");
}

TEST_F(ListFamilyTest, LRange) {
  auto resp = Run({"lrange", kKey1, "0", "5"});
  ASSERT_THAT(resp, ArrLen(0));
  Run({"rpush", kKey1, "0", "1", "2"});
  resp = Run({"lrange", kKey1, "-2", "-1"});

  ASSERT_THAT(resp, ArrLen(2));
  ASSERT_THAT(resp.GetVec(), ElementsAre("1", "2"));
}

TEST_F(ListFamilyTest, Lset) {
  Run({"rpush", kKey1, "0", "1", "2"});
  ASSERT_EQ(Run({"lset", kKey1, "0", "bar"}), "OK");
  ASSERT_EQ(Run({"lpop", kKey1}), "bar");
  ASSERT_EQ(Run({"lset", kKey1, "-1", "foo"}), "OK");
  ASSERT_EQ(Run({"rpop", kKey1}), "foo");
  Run({"rpush", kKey2, "a"});
  ASSERT_THAT(Run({"lset", kKey2, "1", "foo"}), ErrArg("index out of range"));
}

TEST_F(ListFamilyTest, LPos) {
  auto resp = Run({"rpush", kKey1, "1", "a", "b", "1", "1", "a", "1"});
  ASSERT_THAT(resp, IntArg(7));

  ASSERT_THAT(Run({"lpos", kKey1, "1"}), IntArg(0));

  ASSERT_THAT(Run({"lpos", kKey1, "f"}), ArgType(RespExpr::NIL));
  ASSERT_THAT(Run({"lpos", kKey1, "1", "COUNT", "-1"}), ArgType(RespExpr::ERROR));
  ASSERT_THAT(Run({"lpos", kKey1, "1", "MAXLEN", "-1"}), ArgType(RespExpr::ERROR));
  ASSERT_THAT(Run({"lpos", kKey1, "1", "RANK", "0"}), ArgType(RespExpr::ERROR));

  resp = Run({"lpos", kKey1, "a", "RANK", "-1", "COUNT", "2"});
  ASSERT_THAT(resp.GetVec(), ElementsAre(IntArg(5), IntArg(1)));

  resp = Run({"lpos", kKey1, "1", "COUNT", "0"});
  ASSERT_THAT(resp.GetVec(), ElementsAre(IntArg(0), IntArg(3), IntArg(4), IntArg(6)));

  resp = Run({"lpos", kKey1, "1", "COUNT", "0", "MAXLEN", "5"});
  ASSERT_THAT(resp.GetVec(), ElementsAre(IntArg(0), IntArg(3), IntArg(4)));
}

TEST_F(ListFamilyTest, RPopLPush) {
  // src and dest are diffrent keys
  auto resp = Run({"rpush", kKey1, "1", "a", "b", "1", "2", "3", "4"});
  ASSERT_THAT(resp, IntArg(7));

  resp = Run({"rpoplpush", kKey1, kKey2});
  ASSERT_THAT(resp, "4");

  resp = Run({"rpoplpush", kKey1, kKey2});
  ASSERT_THAT(resp, "3");

  resp = Run({"rpoplpush", kKey1, kKey2});
  ASSERT_THAT(resp, "2");

  resp = Run({"rpoplpush", kKey1, kKey2});
  ASSERT_THAT(resp, "1");

  resp = Run({"lrange", kKey1, "0", "-1"});
  ASSERT_THAT(resp, ArrLen(3));
  ASSERT_THAT(resp.GetVec(), ElementsAre("1", "a", "b"));

  resp = Run({"lrange", kKey2, "0", "-1"});
  ASSERT_THAT(resp, ArrLen(4));
  ASSERT_THAT(resp.GetVec(), ElementsAre("1", "2", "3", "4"));

  resp = Run({"rpoplpush", kKey1, kKey2});
  ASSERT_THAT(resp, "b");

  resp = Run({"rpoplpush", kKey1, kKey2});
  ASSERT_THAT(resp, "a");

  resp = Run({"rpoplpush", kKey1, kKey2});
  ASSERT_THAT(resp, "1");

  ASSERT_THAT(Run({"lrange", kKey1, "0", "-1"}), ArrLen(0));
  EXPECT_THAT(Run({"exists", kKey1}), IntArg(0));
  ASSERT_THAT(Run({"rpoplpush", kKey1, kKey2}), ArgType(RespExpr::NIL));

  resp = Run({"lrange", kKey2, "0", "-1"});
  ASSERT_THAT(resp, ArrLen(7));
  ASSERT_THAT(resp.GetVec(), ElementsAre("1", "a", "b", "1", "2", "3", "4"));

  // src and dest are the same key
  resp = Run({"rpush", kKey1, "1", "a", "b", "1", "2", "3", "4"});
  ASSERT_THAT(resp, IntArg(7));

  resp = Run({"rpoplpush", kKey1, kKey1});
  ASSERT_THAT(resp, "4");

  resp = Run({"rpoplpush", kKey1, kKey1});
  ASSERT_THAT(resp, "3");

  resp = Run({"rpoplpush", kKey1, kKey1});
  ASSERT_THAT(resp, "2");

  resp = Run({"rpoplpush", kKey1, kKey1});
  ASSERT_THAT(resp, "1");

  resp = Run({"lrange", kKey1, "0", "-1"});
  ASSERT_THAT(resp, ArrLen(7));
  ASSERT_THAT(resp.GetVec(), ElementsAre("1", "2", "3", "4", "1", "a", "b"));

  resp = Run({"rpoplpush", kKey1, kKey1});
  ASSERT_THAT(resp, "b");

  resp = Run({"rpoplpush", kKey1, kKey1});
  ASSERT_THAT(resp, "a");

  resp = Run({"rpoplpush", kKey1, kKey1});
  ASSERT_THAT(resp, "1");

  resp = Run({"lrange", kKey1, "0", "-1"});
  ASSERT_THAT(resp, ArrLen(7));
  ASSERT_THAT(resp.GetVec(), ElementsAre("1", "a", "b", "1", "2", "3", "4"));
}

TEST_F(ListFamilyTest, LMove) {
  // src and dest are different keys
  auto resp = Run({"rpush", kKey1, "1", "2", "3", "4", "5"});
  ASSERT_THAT(resp, IntArg(5));

  resp = Run({"lmove", kKey1, kKey2, "LEFT", "RIGHT"});
  ASSERT_THAT(resp, "1");

  resp = Run({"lmove", kKey1, kKey2, "LEFT", "LEFT"});
  ASSERT_THAT(resp, "2");

  resp = Run({"lrange", kKey2, "0", "-1"});
  ASSERT_THAT(resp, ArrLen(2));
  ASSERT_THAT(resp.GetVec(), ElementsAre("2", "1"));

  resp = Run({"lmove", kKey1, kKey2, "RIGHT", "LEFT"});
  ASSERT_THAT(resp, "5");

  resp = Run({"lrange", kKey2, "0", "-1"});
  ASSERT_THAT(resp, ArrLen(3));
  ASSERT_THAT(resp.GetVec(), ElementsAre("5", "2", "1"));

  resp = Run({"lmove", kKey1, kKey2, "RIGHT", "RIGHT"});
  ASSERT_THAT(resp, "4");

  resp = Run({"lrange", kKey1, "0", "-1"});
  ASSERT_EQ(resp, "3");

  resp = Run({"lrange", kKey2, "0", "-1"});
  ASSERT_THAT(resp, ArrLen(4));
  ASSERT_THAT(resp.GetVec(), ElementsAre("5", "2", "1", "4"));

  resp = Run({"lmove", kKey1, kKey2, "RIGHT", "RIGHT"});
  ASSERT_THAT(resp, "3");

  ASSERT_THAT(Run({"lrange", kKey1, "0", "-1"}), ArrLen(0));
  EXPECT_THAT(Run({"exists", kKey1}), IntArg(0));
  ASSERT_THAT(Run({"lmove", kKey1, kKey2, "LEFT", "RIGHT"}), ArgType(RespExpr::NIL));
  ASSERT_THAT(Run({"lmove", kKey1, kKey2, "RIGHT", "RIGHT"}), ArgType(RespExpr::NIL));

  resp = Run({"lrange", kKey2, "0", "-1"});
  ASSERT_THAT(resp, ArrLen(5));
  ASSERT_THAT(resp.GetVec(), ElementsAre("5", "2", "1", "4", "3"));

  // src and dest are the same key
  resp = Run({"rpush", kKey1, "1", "2", "3", "4", "5"});
  ASSERT_THAT(resp, IntArg(5));

  resp = Run({"lmove", kKey1, kKey1, "LEFT", "RIGHT"});
  ASSERT_THAT(resp, "1");

  resp = Run({"lmove", kKey1, kKey1, "LEFT", "LEFT"});
  ASSERT_THAT(resp, "2");

  resp = Run({"lmove", kKey1, kKey1, "RIGHT", "LEFT"});
  ASSERT_THAT(resp, "1");

  resp = Run({"lmove", kKey1, kKey1, "RIGHT", "RIGHT"});
  ASSERT_THAT(resp, "5");

  resp = Run({"lmove", kKey1, kKey1, "LEFT", "RIGHT"});
  ASSERT_THAT(resp, "1");

  resp = Run({"lrange", kKey1, "0", "-1"});
  ASSERT_THAT(resp, ArrLen(5));
  ASSERT_THAT(resp.GetVec(), ElementsAre("2", "3", "4", "5", "1"));

  resp = Run({"lmove", kKey1, kKey1, "LEFT", "RIGHT"});
  ASSERT_THAT(resp, "2");

  resp = Run({"lmove", kKey1, kKey1, "LEFT", "RIGHT"});
  ASSERT_THAT(resp, "3");

  resp = Run({"lmove", kKey1, kKey1, "RIGHT", "RIGHT"});
  ASSERT_THAT(resp, "3");

  resp = Run({"lmove", kKey1, kKey1, "LEFT", "RIGHT"});
  ASSERT_THAT(resp, "4");

  resp = Run({"lrange", kKey1, "0", "-1"});
  ASSERT_THAT(resp, ArrLen(5));
  ASSERT_THAT(resp.GetVec(), ElementsAre("5", "1", "2", "3", "4"));

  ASSERT_THAT(Run({"lmove", kKey1, kKey1, "LEFT", "R"}), ArgType(RespExpr::ERROR));
}

TEST_F(ListFamilyTest, TwoQueueBug451) {
  // The bug was that if 2 push operations where queued together in the tx queue,
  // and the first awoke pending blpop, then the PollExecution function would continue with the
  // second push before switching to blpop, which contradicts the spec.
  std::atomic_bool running{true};
  std::atomic_int it_cnt{0};

  auto pop_fiber = [&]() {
    auto id = "t-" + std::to_string(it_cnt.fetch_add(1));
    while (running.load()) {
      Run(id, {"blpop", "a", "0.1"});
    }
  };

  auto push_fiber = [&]() {
    auto id = "t-" + std::to_string(it_cnt.fetch_add(1));
    for (int i = 0; i < 300; i++) {
      Run(id, {"rpush", "a", "DATA"});
    }
    ThisFiber::SleepFor(50ms);
    running = false;
  };

  vector<Fiber> fbs;

  // more likely to reproduce the bug if we start pop_fiber first.
  for (int i = 0; i < 2; i++) {
    fbs.push_back(pp_->at(i)->LaunchFiber(pop_fiber));
  }

  for (int i = 0; i < 2; i++) {
    fbs.push_back(pp_->at(i)->LaunchFiber(push_fiber));
  }

  for (auto& f : fbs)
    f.Join();
  ASSERT_EQ(0, NumWatched());
}

TEST_F(ListFamilyTest, BRPopLPushSingleShard) {
  EXPECT_THAT(Run({"brpoplpush", "x", "y", "0.05"}), ArgType(RespExpr::NIL));
  ASSERT_EQ(0, NumWatched());

  EXPECT_THAT(Run({"lpush", "x", "val1"}), IntArg(1));
  EXPECT_EQ(Run({"brpoplpush", "x", "y", "0.01"}), "val1");
  ASSERT_EQ(1, GetDebugInfo().shards_count);

  EXPECT_THAT(Run({
                  "exists",
                  "x",
              }),
              IntArg(0));
  Run({"set", "x", "str"});
  EXPECT_THAT(Run({"brpoplpush", "y", "x", "0.01"}), ErrArg("wrong kind of value"));

  Run({"del", "x", "y"});
  Run({"multi"});
  Run({"brpoplpush", "y", "x", "0"});
  RespExpr resp = Run({"exec"});
  EXPECT_THAT(resp, ArgType(RespExpr::NIL));
  ASSERT_FALSE(IsLocked(0, "x"));
  ASSERT_FALSE(IsLocked(0, "y"));
  ASSERT_EQ(0, NumWatched());
}

TEST_F(ListFamilyTest, BRPopLPushSingleShardBlocking) {
  RespExpr resp;

  // Run the fiber at creation.
  auto fb0 = pp_->at(0)->LaunchFiber(Launch::dispatch, [&] {
    resp = Run({"brpoplpush", "x", "y", "0"});
  });
  ThisFiber::SleepFor(30us);
  pp_->at(1)->Await([&] { Run("B1", {"lpush", "y", "2"}); });

  pp_->at(1)->Await([&] { Run("B1", {"lpush", "x", "1"}); });
  fb0.Join();
  ASSERT_EQ(resp, "1");
  ASSERT_FALSE(IsLocked(0, "x"));
  ASSERT_FALSE(IsLocked(0, "y"));
  ASSERT_EQ(0, NumWatched());
}

TEST_F(ListFamilyTest, BRPopContended) {
  RespExpr resp;
  atomic_bool done{false};
  constexpr auto kNumFibers = 4;

  // Run the fiber at creation.
  Fiber fb[kNumFibers];
  for (int i = 0; i < kNumFibers; i++) {
    fb[i] = pp_->at(1)->LaunchFiber(Launch::dispatch, [&] {
      string id = StrCat("id", i);
      while (!done) {
        Run(id, {"brpop", "k0", "k1", "k2", "k3", "k4", "0.1"});
      };
    });
  }

  for (int i = 0; i < 500; i++) {
    string key = absl::StrCat("k", i % 3);
    Run({"lpush", key, "foo"});
  }

  done = true;
  for (int i = 0; i < kNumFibers; i++) {
    fb[i].Join();
  }
  ASSERT_EQ(0, NumWatched());
  ASSERT_FALSE(HasAwakened());
}

TEST_F(ListFamilyTest, BRPopLPushTwoShards) {
  RespExpr resp;
  EXPECT_THAT(Run({"brpoplpush", "x", "z", "0.05"}), ArgType(RespExpr::NIL));

  ASSERT_EQ(0, NumWatched());

  Run({"lpush", "x", "val"});
  EXPECT_EQ(Run({"brpoplpush", "x", "z", "0"}), "val");
  resp = Run({"lrange", "z", "0", "-1"});
  ASSERT_EQ(resp, "val");
  Run({"del", "z"});
  ASSERT_EQ(0, NumWatched());

  // Run the fiber at creation.
  auto fb0 = pp_->at(0)->LaunchFiber(Launch::dispatch, [&] {
    resp = Run({"brpoplpush", "x", "z", "0"});
  });

  ThisFiber::SleepFor(30us);
  RespExpr resp_push = pp_->at(1)->Await([&] { return Run("B1", {"lpush", "z", "val2"}); });
  ASSERT_THAT(resp_push, IntArg(1));

  resp_push = pp_->at(1)->Await([&] { return Run("B1", {"lpush", "x", "val1"}); });
  ASSERT_THAT(resp_push, IntArg(1));
  fb0.Join();

  // Result of brpoplpush above.
  ASSERT_EQ(resp, "val1");

  resp = Run({"lrange", "z", "0", "-1"});
  ASSERT_THAT(resp, ArrLen(2));
  ASSERT_THAT(resp.GetVec(), ElementsAre("val1", "val2"));
  ASSERT_FALSE(IsLocked(0, "x"));
  ASSERT_FALSE(IsLocked(0, "z"));
  ASSERT_EQ(0, NumWatched());
  ASSERT_FALSE(HasAwakened());

  // TODO: there is a bug here.
  // we do not wake the dest shard, when source is awaked which prevents
  // the atomicity and causes the first bug as well.
}

TEST_F(ListFamilyTest, BLMove) {
  EXPECT_THAT(Run({"blmove", "x", "y", "right", "right", "0.05"}), ArgType(RespExpr::NIL));
  ASSERT_EQ(0, NumWatched());

  EXPECT_THAT(Run({"lpush", "x", "val1"}), IntArg(1));
  EXPECT_THAT(Run({"lpush", "y", "val2"}), IntArg(1));

  EXPECT_EQ(Run({"blmove", "x", "y", "right", "left", "0.01"}), "val1");
  auto resp = Run({"lrange", "y", "0", "-1"});
  ASSERT_THAT(resp, ArrLen(2));
  ASSERT_THAT(resp.GetVec(), ElementsAre("val1", "val2"));
}

TEST_F(ListFamilyTest, LPushX) {
  // No push for 'lpushx' on nonexisting key.
  EXPECT_THAT(Run({"lpushx", kKey1, "val1"}), IntArg(0));
  EXPECT_THAT(Run({"llen", kKey1}), IntArg(0));

  EXPECT_THAT(Run({"lpush", kKey1, "val1"}), IntArg(1));
  EXPECT_THAT(Run({"lrange", kKey1, "0", "-1"}), "val1");

  EXPECT_THAT(Run({"lpushx", kKey1, "val2"}), IntArg(2));
  EXPECT_THAT(Run({"lrange", kKey1, "0", "-1"}).GetVec(), ElementsAre("val2", "val1"));
}

TEST_F(ListFamilyTest, RPushX) {
  // No push for 'rpushx' on nonexisting key.
  EXPECT_THAT(Run({"rpushx", kKey1, "val1"}), IntArg(0));
  EXPECT_THAT(Run({"llen", kKey1}), IntArg(0));

  EXPECT_THAT(Run({"rpush", kKey1, "val1"}), IntArg(1));
  EXPECT_THAT(Run({"lrange", kKey1, "0", "-1"}), "val1");

  EXPECT_THAT(Run({"rpushx", kKey1, "val2"}), IntArg(2));
  EXPECT_THAT(Run({"lrange", kKey1, "0", "-1"}).GetVec(), ElementsAre("val1", "val2"));
}

TEST_F(ListFamilyTest, LInsert) {
  // List not found.
  EXPECT_THAT(Run({"linsert", "notfound", "before", "foo", "bar"}), ErrArg("no such key"));

  // Key is not a list.
  Run({"set", "notalist", "x"});
  EXPECT_THAT(Run({"linsert", "notalist", "before", "foo", "bar"}),
              ErrArg("Operation against a key holding the wrong kind of value"));

  // Insert before.
  Run({"rpush", "mylist", "foo"});
  EXPECT_THAT(Run({"linsert", "mylist", "before", "foo", "bar"}), IntArg(2));
  auto resp = Run({"lrange", "mylist", "0", "1"});
  ASSERT_THAT(resp, ArrLen(2));
  ASSERT_THAT(resp.GetVec(), ElementsAre("bar", "foo"));

  // Insert after.
  EXPECT_THAT(Run({"linsert", "mylist", "after", "foo", "car"}), IntArg(3));
  resp = Run({"lrange", "mylist", "0", "2"});
  ASSERT_THAT(resp, ArrLen(3));
  ASSERT_THAT(resp.GetVec(), ElementsAre("bar", "foo", "car"));

  // Insert before, pivot not found.
  EXPECT_THAT(Run({"linsert", "mylist", "before", "notfound", "x"}), IntArg(-1));

  // Insert after, pivot not found.
  EXPECT_THAT(Run({"linsert", "mylist", "after", "notfound", "x"}), IntArg(-1));
}

TEST_F(ListFamilyTest, BLPopUnwakesInScript) {
  const string_view SCRIPT = R"(
    for i = 1, 1000 do
      redis.call('MGET', 'a', 'b', 'c', 'd')
      redis.call('LPUSH', 'l', tostring(i))
    end
  )";

  // Start blpop with without timeout
  auto f1 = pp_->at(1)->LaunchFiber(Launch::dispatch, [&]() {
    auto resp = Run("blpop", {"BLPOP", "l", "0"});
    // blpop should only be awakened after the script has completed, so the
    // last element added in the script should be returned.
    EXPECT_THAT(resp, ArgType(RespExpr::ARRAY));
    EXPECT_THAT(resp.GetVec(), ElementsAre("l", "1000"));
  });

  // Start long running script that intends to wake up blpop
  auto f2 = pp_->at(2)->LaunchFiber([&]() {
    Run("script", {"EVAL", SCRIPT, "5", "a", "b", "c", "d", "l"});
  });

  // Run blpop that times out
  auto resp = Run({"blpop", "g", "0.01"});
  EXPECT_THAT(resp, ArgType(RespExpr::NIL_ARRAY));

  f1.Join();
  f2.Join();
}

TEST_F(ListFamilyTest, OtherMultiWakesBLpop) {
  const string_view SCRIPT = R"(
    redis.call('LPUSH', 'l', 'bad')
    for i = 1, 1000 do
      redis.call('MGET', 'a', 'b', 'c', 'd')
    end
    redis.call('LPUSH', 'l', 'good')
  )";

  const string_view SCRIPT_SHORT = R"(
    redis.call('GET', KEYS[1])
  )";

  // Start BLPOP with infinite timeout
  auto f1 = pp_->at(1)->LaunchFiber(Launch::dispatch, [&]() {
    auto resp = Run("blpop", {"BLPOP", "l", "0"});
    // blpop should only be awakened after the script has completed, so the
    // last element added in the script should be returned.
    EXPECT_THAT(resp, ArgType(RespExpr::ARRAY));
    EXPECT_THAT(resp.GetVec(), ElementsAre("l", "good"));
  });

  // Start long running script that accesses the list, but should wake up blpop only after it
  // finished
  auto f2 = pp_->at(2)->LaunchFiber(Launch::dispatch, [&]() {
    Run("script", {"EVAL", SCRIPT, "5", "a", "b", "c", "d", "l"});
  });

  // Run quick multi transaction that concludes after one hop
  Run({"EVAL", SCRIPT_SHORT, "1", "y"});

  f1.Join();
  f2.Join();
}

}  // namespace dfly
