// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#include <vespa/vespalib/testkit/test_kit.h>
#include <vespa/vespalib/net/socket_spec.h>
#include <vespa/vespalib/net/tls/capability_env_config.h>
#include <vespa/vespalib/net/tls/statistics.h>
#include <vespa/vespalib/util/benchmark_timer.h>
#include <vespa/vespalib/util/latch.h>
#include <vespa/fnet/frt/supervisor.h>
#include <vespa/fnet/frt/target.h>
#include <vespa/fnet/frt/rpcrequest.h>
#include <vespa/fnet/frt/invoker.h>
#include <vespa/fnet/frt/request_access_filter.h>
#include <vespa/fnet/frt/require_capabilities.h>
#include <mutex>
#include <condition_variable>
#include <string_view>

using vespalib::SocketSpec;
using vespalib::BenchmarkTimer;
using vespalib::net::tls::CapabilityStatistics;
using namespace vespalib::net::tls;

constexpr double timeout = 60.0;
constexpr double short_timeout = 0.1;

//-------------------------------------------------------------

#include "my_crypto_engine.hpp"
vespalib::CryptoEngine::SP crypto;

//-------------------------------------------------------------

class RequestLatch : public FRT_IRequestWait {
private:
    vespalib::Latch<FRT_RPCRequest*> _latch;
public:
    RequestLatch() : _latch() {}
    ~RequestLatch() override { ASSERT_TRUE(!has_req()); }
    bool has_req() { return _latch.has_value(); }
    FRT_RPCRequest *read() { return _latch.read(); }
    void write(FRT_RPCRequest *req) { _latch.write(req); }
    void RequestDone(FRT_RPCRequest *req) override { write(req); }
};

//-------------------------------------------------------------

class MyReq {
private:
    FRT_RPCRequest *_req;
public:
    explicit MyReq(FRT_RPCRequest *req) : _req(req) {}
    explicit MyReq(const char *method_name)
        : _req(new FRT_RPCRequest())
    {
        _req->SetMethodName(method_name);
    }
    MyReq(uint32_t value, bool async, uint32_t error, uint8_t extra)
        : _req(new FRT_RPCRequest())
    {
        _req->SetMethodName("test");
        _req->GetParams()->AddInt32(value);
        _req->GetParams()->AddInt32(error);
        _req->GetParams()->AddInt8(extra);
        _req->GetParams()->AddInt8((async) ? 1 : 0);
    }
    ~MyReq() {
        if (_req != nullptr) {
            _req->internal_subref();
        }
    }
    MyReq(const MyReq &rhs) = delete;
    MyReq &operator=(const MyReq &rhs) = delete;
    FRT_RPCRequest &get() { return *_req; }
    FRT_RPCRequest *borrow() { return _req; }
    FRT_RPCRequest *steal() {
        auto ret = _req;
        _req = nullptr;
        return ret;
    }
    uint32_t get_int_ret() {
        ASSERT_TRUE(_req != nullptr);
        ASSERT_TRUE(_req->CheckReturnTypes("i"));
        return _req->GetReturn()->GetValue(0)._intval32;
    }
};

//-------------------------------------------------------------

class EchoTest : public FRT_Invokable
{
private:
    vespalib::Stash _echo_stash;
    FRT_Values      _echo_args;

public:
    EchoTest(const EchoTest &) = delete;
    EchoTest &operator=(const EchoTest &) = delete;
    EchoTest(FRT_Supervisor *supervisor);
    ~EchoTest() override;

    bool prepare_params(FRT_RPCRequest &req)
    {
        FNET_DataBuffer buf;

        _echo_args.EncodeCopy(&buf);
        req.GetParams()->DecodeCopy(&buf, buf.GetDataLen());
        return (req.GetParams()->Equals(&_echo_args) &&
                _echo_args.Equals(req.GetParams()));
    }

    void RPC_Echo(FRT_RPCRequest *req)
    {
        FNET_DataBuffer buf;

        req->GetParams()->EncodeCopy(&buf);
        req->GetReturn()->DecodeCopy(&buf, buf.GetDataLen());
        if (!req->GetReturn()->Equals(&_echo_args) ||
            !req->GetReturn()->Equals(req->GetParams()))
        {
            req->SetError(10000, "Streaming error");
        }
    }
};

EchoTest::~EchoTest() = default;

EchoTest::EchoTest(FRT_Supervisor *supervisor)
    : _echo_stash(),
      _echo_args(_echo_stash)
{
    FRT_ReflectionBuilder rb(supervisor);
    rb.DefineMethod("echo", "*", "*", FRT_METHOD(EchoTest::RPC_Echo), this);

    FRT_Values *args = &_echo_args;
    args->EnsureFree(16);

    args->AddInt8(8);
    uint8_t *pt_int8 = args->AddInt8Array(3);
    pt_int8[0] = 1;
    pt_int8[1] = 2;
    pt_int8[2] = 3;

    args->AddInt16(16);
    uint16_t *pt_int16 = args->AddInt16Array(3);
    pt_int16[0] = 2;
    pt_int16[1] = 4;
    pt_int16[2] = 6;

    args->AddInt32(32);
    uint32_t *pt_int32 = args->AddInt32Array(3);
    pt_int32[0] = 4;
    pt_int32[1] = 8;
    pt_int32[2] = 12;

    args->AddInt64(64);
    uint64_t *pt_int64 = args->AddInt64Array(3);
    pt_int64[0] = 8;
    pt_int64[1] = 16;
    pt_int64[2] = 24;

    args->AddFloat(32.5);
    float *pt_float = args->AddFloatArray(3);
    pt_float[0] = 0.25;
    pt_float[1] = 0.5;
    pt_float[2] = 0.75;

    args->AddDouble(64.5);
    double *pt_double = args->AddDoubleArray(3);
    pt_double[0] = 0.1;
    pt_double[1] = 0.2;
    pt_double[2] = 0.3;

    args->AddString("string");
    FRT_StringValue *pt_string = args->AddStringArray(3);
    args->SetString(&pt_string[0], "str1");
    args->SetString(&pt_string[1], "str2");
    args->SetString(&pt_string[2], "str3");

    args->AddData("data", 4);
    FRT_DataValue *pt_data = args->AddDataArray(3);
    args->SetData(&pt_data[0], "dat1", 4);
    args->SetData(&pt_data[1], "dat2", 4);
    args->SetData(&pt_data[2], "dat3", 4);
}
//-------------------------------------------------------------

struct MyAccessFilter : FRT_RequestAccessFilter {
    ~MyAccessFilter() override = default;

    constexpr static std::string_view WRONG_KEY   = "...mellon!";
    constexpr static std::string_view CORRECT_KEY = "let me in, I have cake";

    bool allow(FRT_RPCRequest& req) const noexcept override {
        const auto& req_param = req.GetParams()->GetValue(0)._string;
        const auto magic_key = std::string_view(req_param._str, req_param._len);
        return (magic_key == CORRECT_KEY);
    }
};

class TestRPC : public FRT_Invokable
{
private:
    uint32_t          _intValue;
    RequestLatch      _detached_req;
    std::atomic<bool> _restricted_method_was_invoked;

    TestRPC(const TestRPC &);
    TestRPC &operator=(const TestRPC &);

public:
    TestRPC(FRT_Supervisor *supervisor)
        : _intValue(0),
          _detached_req(),
          _restricted_method_was_invoked(false)
    {
        FRT_ReflectionBuilder rb(supervisor);

        rb.DefineMethod("inc", "i", "i",
                        FRT_METHOD(TestRPC::RPC_Inc), this);
        rb.DefineMethod("setValue", "i", "",
                        FRT_METHOD(TestRPC::RPC_SetValue), this);
        rb.DefineMethod("incValue", "", "",
                        FRT_METHOD(TestRPC::RPC_IncValue), this);
        rb.DefineMethod("getValue", "", "i",
                        FRT_METHOD(TestRPC::RPC_GetValue), this);
        rb.DefineMethod("test", "iibb", "i",
                        FRT_METHOD(TestRPC::RPC_Test), this);
        rb.DefineMethod("accessRestricted", "s", "",
                        FRT_METHOD(TestRPC::RPC_AccessRestricted), this);
        rb.RequestAccessFilter(std::make_unique<MyAccessFilter>());
        // The authz rules used for this test only grant the telemetry capability set
        rb.DefineMethod("capabilityRestricted", "", "",
                        FRT_METHOD(TestRPC::RPC_AccessRestricted), this);
        rb.RequestAccessFilter(FRT_RequireCapabilities::of(CapabilitySet::content_node()));
        rb.DefineMethod("capabilityAllowed", "", "",
                        FRT_METHOD(TestRPC::RPC_AccessRestricted), this);
        rb.RequestAccessFilter(FRT_RequireCapabilities::of(CapabilitySet::telemetry()));
        rb.DefineMethod("emptyCapabilitySet", "", "",
                        FRT_METHOD(TestRPC::RPC_AccessRestricted), this);
        rb.RequestAccessFilter(FRT_RequireCapabilities::of(CapabilitySet::make_empty()));
    }

    void RPC_Test(FRT_RPCRequest *req)
    {
        FRT_Values &param = *req->GetParams();
        uint32_t value = param[0]._intval32;
        uint32_t error = param[1]._intval32;
        uint8_t  extra = param[2]._intval8;
        uint8_t  async = param[3]._intval8;

        req->GetReturn()->AddInt32(value);
        if (extra != 0) {
            req->GetReturn()->AddInt32(value);
        }
        if (error != 0) {
            req->SetError(error);
        }
        if (async != 0) {
            _detached_req.write(req->Detach());
        }
    }

    void RPC_Inc(FRT_RPCRequest *req)
    {
        req->GetReturn()->AddInt32(req->GetParams()->GetValue(0)._intval32 + 1);
    }

    void RPC_SetValue(FRT_RPCRequest *req)
    {
        _intValue = req->GetParams()->GetValue(0)._intval32;
    }

    void RPC_IncValue(FRT_RPCRequest *req)
    {
        (void) req;
        _intValue++;
    }

    void RPC_GetValue(FRT_RPCRequest *req)
    {
        req->GetReturn()->AddInt32(_intValue);
    }

    void RPC_AccessRestricted([[maybe_unused]] FRT_RPCRequest *req)
    {
        // We'll only get here if the access filter lets us in
        _restricted_method_was_invoked.store(true);
    }

    bool restricted_method_was_invoked() const noexcept {
        return _restricted_method_was_invoked.load();
    }

    RequestLatch &detached_req() { return _detached_req; }
};

//-------------------------------------------------------------

class Fixture
{
private:
    fnet::frt::StandaloneFRT  _client;
    fnet::frt::StandaloneFRT  _server;
    vespalib::string   _peerSpec;
    FRT_Target        *_target;
    TestRPC            _testRPC;
    EchoTest           _echoTest;

public:
    FRT_Target &target() { return *_target; }
    FRT_Target *make_bad_target() { return _client.supervisor().GetTarget("bogus address"); }
    RequestLatch &detached_req() { return _testRPC.detached_req(); }
    EchoTest &echo() { return _echoTest; }
    const TestRPC& server_instance() const noexcept { return _testRPC; }

    Fixture()
        : _client(crypto),
          _server(crypto),
          _peerSpec(),
          _target(nullptr),
          _testRPC(&_server.supervisor()),
          _echoTest(&_server.supervisor())
    {
        ASSERT_TRUE(_server.supervisor().Listen("tcp/0"));
        _peerSpec = SocketSpec::from_host_port("localhost", _server.supervisor().GetListenPort()).spec();
        _target = _client.supervisor().GetTarget(_peerSpec.c_str());
        //---------------------------------------------------------------------
        MyReq req("frt.rpc.ping");
        target().InvokeSync(req.borrow(), timeout);
        ASSERT_TRUE(!req.get().IsError());
    }

    ~Fixture() {
        _target->internal_subref();
    }
};

//-------------------------------------------------------------

TEST_F("require that simple invocation works", Fixture()) {
    MyReq req("inc");
    req.get().GetParams()->AddInt32(502);
    f1.target().InvokeSync(req.borrow(), timeout);
    EXPECT_EQUAL(req.get_int_ret(), 503u);
}

TEST_F("require that void invocation works", Fixture()) {
    {
        MyReq req("setValue");
        req.get().GetParams()->AddInt32(40);
        f1.target().InvokeSync(req.borrow(), timeout);
        EXPECT_TRUE(req.get().CheckReturnTypes(""));
    }
    {
        MyReq req("incValue");
        f1.target().InvokeVoid(req.steal());
    }
    {
        MyReq req("incValue");
        f1.target().InvokeVoid(req.steal());
    }
    {
        MyReq req("getValue");
        f1.target().InvokeSync(req.borrow(), timeout);
        EXPECT_EQUAL(req.get_int_ret(), 42u);
    }
}

TEST_F("measure minimal invocation latency", Fixture()) {
    size_t cnt = 0;
    uint32_t val = 0;
    BenchmarkTimer timer(1.0);
    while (timer.has_budget()) {
        timer.before();
        {
            MyReq req("inc");
            req.get().GetParams()->AddInt32(val);
            f1.target().InvokeSync(req.borrow(), timeout);
            ASSERT_TRUE(!req.get().IsError());
            val = req.get_int_ret();
            ++cnt;
        }
        timer.after();
    }
    EXPECT_EQUAL(cnt, val);
    double t = timer.min_time();
    fprintf(stderr, "latency of invocation: %1.3f ms\n", t * 1000.0);
}

TEST_F("require that abort has no effect on a completed request", Fixture()) {
    MyReq req(42, false, FRTE_NO_ERROR, 0);
    f1.target().InvokeSync(req.borrow(), timeout);
    EXPECT_EQUAL(req.get_int_ret(), 42u);
    req.get().Abort();
    EXPECT_EQUAL(req.get_int_ret(), 42u);
}

TEST_F("require that a request can be responded to at a later time", Fixture()) {
    RequestLatch result;
    MyReq req(42, true, FRTE_NO_ERROR, 0);
    f1.target().InvokeAsync(req.steal(), timeout, &result);
    EXPECT_TRUE(!result.has_req());
    f1.detached_req().read()->Return();
    MyReq ret(result.read());
    EXPECT_EQUAL(ret.get_int_ret(), 42u);
}

TEST_F("require that a bad target gives connection error", Fixture()) {
    MyReq req("frt.rpc.ping");
    {
        FRT_Target *bad_target = f1.make_bad_target();
        bad_target->InvokeSync(req.borrow(), timeout);
        bad_target->internal_subref();
    }
    EXPECT_EQUAL(req.get().GetErrorCode(), FRTE_RPC_CONNECTION);
}

TEST_F("require that non-existing method gives appropriate error", Fixture()) {
    MyReq req("bogus");
    f1.target().InvokeSync(req.borrow(), timeout);
    EXPECT_EQUAL(req.get().GetErrorCode(), FRTE_RPC_NO_SUCH_METHOD);    
}

TEST_F("require that wrong parameter types give appropriate error", Fixture()) {
    MyReq req("setValue");
    req.get().GetParams()->AddString("40");
    f1.target().InvokeSync(req.borrow(), timeout);
    EXPECT_EQUAL(req.get().GetErrorCode(), FRTE_RPC_WRONG_PARAMS);
}

TEST_F("require that wrong return value types give appropriate error", Fixture()) {
    MyReq req(42, false, FRTE_NO_ERROR, 1);
    f1.target().InvokeSync(req.borrow(), timeout);
    EXPECT_EQUAL(req.get().GetErrorCode(), FRTE_RPC_WRONG_RETURN);
}

TEST_F("require that the method itself can signal failure", Fixture()) {
    MyReq req(42, false, 5000, 1);
    f1.target().InvokeSync(req.borrow(), timeout);
    EXPECT_EQUAL(req.get().GetErrorCode(), 5000u);
}

TEST_F("require that invocation can time out", Fixture()) {
    RequestLatch result;
    MyReq req(42, true, FRTE_NO_ERROR, 0);
    f1.target().InvokeAsync(req.steal(), short_timeout, &result);
    MyReq ret(result.read());
    f1.detached_req().read()->Return();
    EXPECT_EQUAL(ret.get().GetErrorCode(), FRTE_RPC_TIMEOUT);
}

TEST_F("require that invocation can be aborted", Fixture()) {
    RequestLatch result;
    MyReq req(42, true, FRTE_NO_ERROR, 0);
    FRT_RPCRequest *will_be_mine_again_soon = req.steal();
    f1.target().InvokeAsync(will_be_mine_again_soon, timeout, &result);
    will_be_mine_again_soon->Abort();
    MyReq ret(result.read());
    f1.detached_req().read()->Return();
    EXPECT_EQUAL(ret.get().GetErrorCode(), FRTE_RPC_ABORT);
}

TEST_F("require that parameters can be echoed as return values", Fixture()) {
    MyReq req("echo");
    ASSERT_TRUE(f1.echo().prepare_params(req.get()));
    f1.target().InvokeSync(req.borrow(), timeout);
    EXPECT_TRUE(!req.get().IsError());
    EXPECT_TRUE(req.get().GetReturn()->Equals(req.get().GetParams()));
    EXPECT_TRUE(req.get().GetParams()->Equals(req.get().GetReturn()));
}

TEST_F("request denied by access filter returns PERMISSION_DENIED and does not invoke server method", Fixture()) {
    MyReq req("accessRestricted");
    auto key = MyAccessFilter::WRONG_KEY;
    req.get().GetParams()->AddString(key.data(), key.size());
    f1.target().InvokeSync(req.borrow(), timeout);
    EXPECT_EQUAL(req.get().GetErrorCode(), FRTE_RPC_PERMISSION_DENIED);
    EXPECT_FALSE(f1.server_instance().restricted_method_was_invoked());
}

TEST_F("request allowed by access filter invokes server method as usual", Fixture()) {
    MyReq req("accessRestricted");
    auto key = MyAccessFilter::CORRECT_KEY;
    req.get().GetParams()->AddString(key.data(), key.size());
    f1.target().InvokeSync(req.borrow(), timeout);
    ASSERT_FALSE(req.get().IsError());
    EXPECT_TRUE(f1.server_instance().restricted_method_was_invoked());
}

TEST_F("capability checking filter is enforced under mTLS unless overridden by env var", Fixture()) {
    const auto cap_stats_before = CapabilityStatistics::get().snapshot();
    MyReq req("capabilityRestricted"); // Requires content node cap set; disallowed
    f1.target().InvokeSync(req.borrow(), timeout);
    auto cap_mode = capability_enforcement_mode_from_env();
    fprintf(stderr, "capability enforcement mode: %s\n", to_string(cap_mode));
    if (crypto->use_tls_when_client() && (cap_mode == CapabilityEnforcementMode::Enforce)) {
        // Default authz rule does not give required capabilities; must fail.
        EXPECT_EQUAL(req.get().GetErrorCode(), FRTE_RPC_PERMISSION_DENIED);
        EXPECT_FALSE(f1.server_instance().restricted_method_was_invoked());
        // Permission denied should bump capability check failure statistic
        const auto cap_stats = CapabilityStatistics::get().snapshot().subtract(cap_stats_before);
        EXPECT_EQUAL(cap_stats.rpc_capability_checks_failed, 1u);
    } else {
        // Either no mTLS configured (implicit full capability set) or capabilities not enforced.
        ASSERT_FALSE(req.get().IsError());
        EXPECT_TRUE(f1.server_instance().restricted_method_was_invoked());
    }
}

TEST_F("access is allowed by capability filter when peer is granted the required capability", Fixture()) {
    const auto cap_stats_before = CapabilityStatistics::get().snapshot();
    MyReq req("capabilityAllowed"); // Requires telemetry cap set; allowed
    f1.target().InvokeSync(req.borrow(), timeout);
    // Should always be allowed, regardless of mTLS mode or capability enforcement
    ASSERT_FALSE(req.get().IsError());
    EXPECT_TRUE(f1.server_instance().restricted_method_was_invoked());
    // Should _not_ bump capability check failure statistic
    const auto cap_stats = CapabilityStatistics::get().snapshot().subtract(cap_stats_before);
    EXPECT_EQUAL(cap_stats.rpc_capability_checks_failed, 0u);
}

TEST_F("access is allowed by capability filter when required capability set is empty", Fixture()) {
    MyReq req("emptyCapabilitySet");
    f1.target().InvokeSync(req.borrow(), timeout);
    ASSERT_FALSE(req.get().IsError());
    EXPECT_TRUE(f1.server_instance().restricted_method_was_invoked());
}

TEST_MAIN() {
    crypto = my_crypto_engine();
    TEST_RUN_ALL();
    crypto.reset();
}
