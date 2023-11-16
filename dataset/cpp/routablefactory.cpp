// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#include <vespa/document/repo/documenttyperepo.h>
#include <vespa/documentapi/messagebus/documentprotocol.h>
#include <vespa/documentapi/messagebus/routablefactories60.h>
#include <vespa/messagebus/testlib/receptor.h>
#include <vespa/messagebus/testlib/slobrok.h>
#include <vespa/messagebus/testlib/testserver.h>
#include <vespa/vespalib/testkit/testapp.h>

using document::DocumentTypeRepo;
using namespace documentapi;

///////////////////////////////////////////////////////////////////////////////
//
// Utilities
//
///////////////////////////////////////////////////////////////////////////////

class MyReply : public DocumentReply {
public:
    enum {
        TYPE = 777
    };

    MyReply() :
        DocumentReply(TYPE) {
        // empty
    }
};

class MyMessage : public DocumentMessage {
public:
    enum {
        TYPE = 666
    };

    MyMessage() {
        getTrace().setLevel(9);
    }

    DocumentReply::UP doCreateReply() const override {
        return DocumentReply::UP(new MyReply());
    }

    uint32_t getType() const override {
        return TYPE;
    }
};

class MyMessageFactory : public RoutableFactories60::DocumentMessageFactory {
protected:
    DocumentMessage::UP doDecode(document::ByteBuffer &buf) const override {
        (void)buf;
        return DocumentMessage::UP(new MyMessage());
    }

    bool doEncode(const DocumentMessage &msg, vespalib::GrowableByteBuffer &buf) const override {
        (void)msg;
        (void)buf;
        return true;
    }
public:
    ~MyMessageFactory() override;
};

MyMessageFactory::~MyMessageFactory() = default;

class MyReplyFactory : public RoutableFactories60::DocumentReplyFactory {
protected:
    DocumentReply::UP doDecode(document::ByteBuffer &buf) const override {
        (void)buf;
        return DocumentReply::UP(new MyReply());
    }

    bool doEncode(const DocumentReply &reply, vespalib::GrowableByteBuffer &buf) const override {
        (void)reply;
        (void)buf;
        return true;
    }
public:
    ~MyReplyFactory() override;
};

MyReplyFactory::~MyReplyFactory() = default;

///////////////////////////////////////////////////////////////////////////////
//
// Setup
//
///////////////////////////////////////////////////////////////////////////////

class TestData {
    const std::shared_ptr<const DocumentTypeRepo> _repo;

public:
    mbus::Slobrok                _slobrok;
    DocumentProtocol::SP         _srcProtocol;
    mbus::TestServer             _srcServer;
    mbus::SourceSession::UP      _srcSession;
    mbus::Receptor               _srcHandler;
    DocumentProtocol::SP         _dstProtocol;
    mbus::TestServer             _dstServer;
    mbus::DestinationSession::UP _dstSession;
    mbus::Receptor               _dstHandler;

public:
    TestData();
    ~TestData();
    bool start();
};

class Test : public vespalib::TestApp {
protected:
    void testFactory(TestData &data);

public:
    int Main() override;
};

TEST_APPHOOK(Test);

TestData::TestData() :
    _repo(std::make_shared<DocumentTypeRepo>()),
    _slobrok(),
    _srcProtocol(std::make_shared<DocumentProtocol>(_repo)),
    _srcServer(mbus::MessageBusParams().addProtocol(_srcProtocol),
               mbus::RPCNetworkParams(_slobrok.config())),
    _srcSession(),
    _srcHandler(),
    _dstProtocol(std::make_shared<DocumentProtocol>(_repo)),
    _dstServer(mbus::MessageBusParams().addProtocol(_dstProtocol),
               mbus::RPCNetworkParams(_slobrok.config()).setIdentity(mbus::Identity("dst"))),
    _dstSession(),
    _dstHandler()
{ }

TestData::~TestData() = default;

bool
TestData::start()
{
    _srcSession = _srcServer.mb.createSourceSession(mbus::SourceSessionParams().setReplyHandler(_srcHandler));
    if ( ! _srcSession) {
        return false;
    }
    _dstSession = _dstServer.mb.createDestinationSession(mbus::DestinationSessionParams().setName("session").setMessageHandler(_dstHandler));
    if ( ! _dstSession) {
        return false;
    }
    if (!_srcServer.waitSlobrok("dst/session", 1u)) {
        return false;
    }
    return true;
}

int
Test::Main()
{
    TEST_INIT("routablefactory_test");

    TestData data;
    ASSERT_TRUE(data.start());

    testFactory(data); TEST_FLUSH();

    TEST_DONE();
}

///////////////////////////////////////////////////////////////////////////////
//
// Tests
//
///////////////////////////////////////////////////////////////////////////////

const vespalib::duration TIMEOUT = 600s;

void
Test::testFactory(TestData &data)
{
    mbus::Route route = mbus::Route::parse("dst/session");

    // Source should fail to encode the message.
    EXPECT_TRUE(data._srcSession->send(mbus::Message::UP(new MyMessage()), route).isAccepted());
    mbus::Reply::UP reply = data._srcHandler.getReply(TIMEOUT);
    ASSERT_TRUE(reply);
    fprintf(stderr, "%s\n", reply->getTrace().toString().c_str());
    ASSERT_TRUE(reply->hasErrors());
    EXPECT_EQUAL((uint32_t)mbus::ErrorCode::ENCODE_ERROR, reply->getError(0).getCode());
    EXPECT_EQUAL("", reply->getError(0).getService());

    // Destination should fail to decode the message.
    data._srcProtocol->putRoutableFactory(MyMessage::TYPE, IRoutableFactory::SP(new MyMessageFactory()),
                                          vespalib::VersionSpecification());
    EXPECT_TRUE(data._srcSession->send(mbus::Message::UP(new MyMessage()), route).isAccepted());
    reply = data._srcHandler.getReply(TIMEOUT);
    ASSERT_TRUE(reply);
    fprintf(stderr, "%s\n", reply->getTrace().toString().c_str());
    EXPECT_TRUE(reply->hasErrors());
    EXPECT_EQUAL((uint32_t)mbus::ErrorCode::DECODE_ERROR, reply->getError(0).getCode());
    EXPECT_EQUAL("dst/session", reply->getError(0).getService());

    // Destination should fail to encode the reply->
    data._dstProtocol->putRoutableFactory(MyMessage::TYPE, IRoutableFactory::SP(new MyMessageFactory()),
                                          vespalib::VersionSpecification());
    EXPECT_TRUE(data._srcSession->send(mbus::Message::UP(new MyMessage()), route).isAccepted());
    mbus::Message::UP msg = data._dstHandler.getMessage(TIMEOUT);
    ASSERT_TRUE(msg);
    reply.reset(new MyReply());
    reply->swapState(*msg);
    data._dstSession->reply(std::move(reply));
    reply = data._srcHandler.getReply(TIMEOUT);
    ASSERT_TRUE(reply);
    fprintf(stderr, "%s\n", reply->getTrace().toString().c_str());
    EXPECT_TRUE(reply->hasErrors());
    EXPECT_EQUAL((uint32_t)mbus::ErrorCode::ENCODE_ERROR, reply->getError(0).getCode());
    EXPECT_EQUAL("dst/session", reply->getError(0).getService());

    // Source should fail to decode the reply.
    data._dstProtocol->putRoutableFactory(MyReply::TYPE, IRoutableFactory::SP(new MyReplyFactory()),
                                          vespalib::VersionSpecification());
    EXPECT_TRUE(data._srcSession->send(mbus::Message::UP(new MyMessage()), route).isAccepted());
    msg = data._dstHandler.getMessage(TIMEOUT);
    ASSERT_TRUE(msg);
    reply.reset(new MyReply());
    reply->swapState(*msg);
    data._dstSession->reply(std::move(reply));
    reply = data._srcHandler.getReply(TIMEOUT);
    ASSERT_TRUE(reply);
    fprintf(stderr, "%s\n", reply->getTrace().toString().c_str());
    EXPECT_TRUE(reply->hasErrors());
    EXPECT_EQUAL((uint32_t)mbus::ErrorCode::DECODE_ERROR, reply->getError(0).getCode());
    EXPECT_EQUAL("", reply->getError(0).getService());

    // All should succeed.
    data._srcProtocol->putRoutableFactory(MyReply::TYPE, IRoutableFactory::SP(new MyReplyFactory()),
                                          vespalib::VersionSpecification());
    EXPECT_TRUE(data._srcSession->send(mbus::Message::UP(new MyMessage()), route).isAccepted());
    msg = data._dstHandler.getMessage(TIMEOUT);
    ASSERT_TRUE(msg);
    reply.reset(new MyReply());
    reply->swapState(*msg);
    data._dstSession->reply(std::move(reply));
    reply = data._srcHandler.getReply(TIMEOUT);
    ASSERT_TRUE(reply);
    fprintf(stderr, "%s\n", reply->getTrace().toString().c_str());
    EXPECT_TRUE(!reply->hasErrors());
}
