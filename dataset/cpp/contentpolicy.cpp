// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "contentpolicy.h"
#include <vespa/document/base/documentid.h>
#include <vespa/document/update/documentupdate.h>
#include <vespa/document/fieldvalue/document.h>
#include <vespa/messagebus/emptyreply.h>
#include <vespa/messagebus/error.h>
#include <vespa/documentapi/documentapi.h>
#include <vespa/vdslib/state/clusterstate.h>
#include <vespa/vespalib/stllike/asciistream.h>
#include <vespa/vespalib/util/stringfmt.h>
#include <vespa/config-stor-distribution.h>
#include <vespa/config/helper/configfetcher.hpp>
#include <vespa/config/subscription/configuri.h>
#include <cassert>

#include <vespa/log/log.h>
LOG_SETUP(".contentpolicy");

using vespalib::make_string;

namespace documentapi {

ContentPolicy::ContentPolicy(const string& param)
    : ExternSlobrokPolicy(parse(param)),
      _bucketIdFactory()
{
    std::map<string, string> params(parse(param));

    if (params.find("cluster") != params.end()) {
        _clusterName = params.find("cluster")->second;
    } else {
        _error = "Required parameter clustername not set";
    }

    if (params.find("clusterconfigid") != params.end()) {
        _clusterConfigId = params.find("clusterconfigid")->second;
    }
}

namespace {
    class CallBack : public config::IFetcherCallback<storage::lib::Distribution::DistributionConfig>
    {
    public:
        explicit CallBack(ContentPolicy & policy) : _policy(policy) { }
        void configure(std::unique_ptr<storage::lib::Distribution::DistributionConfig> config) override {
            _policy.configure(std::move(config));
        }
    private:
        ContentPolicy & _policy;
    };
}
string ContentPolicy::init()
{
    string error = ExternSlobrokPolicy::init();
    if (!error.empty()) {
        return error;
    }

    if (_clusterConfigId.empty()) {
        _clusterConfigId = createConfigId(_clusterName);
    }

    using storage::lib::Distribution;
    config::ConfigUri uri(_clusterConfigId);
    if (!configSources().empty()) {
        _configFetcher = std::make_unique<config::ConfigFetcher>(config::ServerSpec(configSources()));
    } else {
        _configFetcher = std::make_unique<config::ConfigFetcher>(uri.getContext());
    }
    _callBack = std::make_unique<CallBack>(*this);
    _configFetcher->subscribe<vespa::config::content::StorDistributionConfig>(uri.getConfigId(), static_cast<CallBack *>(_callBack.get()));
    _configFetcher->start();
    return "";
}

ContentPolicy::~ContentPolicy() = default;

string
ContentPolicy::createConfigId(const string & clusterName)
{
    return clusterName;
}

string
ContentPolicy::createPattern(const string & clusterName, int distributor)
{
    vespalib::asciistream ost;

    ost << "storage/cluster." << clusterName << "/distributor/";

    if (distributor == -1) {
        ost << '*';
    } else {
        ost << distributor;
    }
    ost << "/default";
    return ost.str();
}

void
ContentPolicy::configure(std::unique_ptr<vespa::config::content::StorDistributionConfig> config)
{
    try {
        std::lock_guard guard(_rw_lock);
        _distribution = std::make_unique<storage::lib::Distribution>(*config);
    } catch (const std::exception& e) {
        LOG(warning, "Got exception when configuring distribution, config id was %s", _clusterConfigId.c_str());
        throw e;
    }
}

void
ContentPolicy::doSelect(mbus::RoutingContext &context)
{
    const mbus::Message &msg = context.getMessage();

    int distributor = -1;
    auto [cur_state, cur_distribution] = internal_state_snapshot();

    if (cur_state) {
        document::BucketId id;
        switch(msg.getType()) {
        case DocumentProtocol::MESSAGE_PUTDOCUMENT:
            id = _bucketIdFactory.getBucketId(static_cast<const PutDocumentMessage&>(msg).getDocument().getId());
            break;

        case DocumentProtocol::MESSAGE_GETDOCUMENT:
            id = _bucketIdFactory.getBucketId(static_cast<const GetDocumentMessage&>(msg).getDocumentId());
            break;

        case DocumentProtocol::MESSAGE_REMOVEDOCUMENT:
            id = _bucketIdFactory.getBucketId(static_cast<const RemoveDocumentMessage&>(msg).getDocumentId());
            break;

        case DocumentProtocol::MESSAGE_UPDATEDOCUMENT:
            id = _bucketIdFactory.getBucketId(static_cast<const UpdateDocumentMessage&>(msg).getDocumentUpdate().getId());
            break;

        case DocumentProtocol::MESSAGE_STATBUCKET:
            id = static_cast<const StatBucketMessage&>(msg).getBucketId();
            break;

        case DocumentProtocol::MESSAGE_GETBUCKETLIST:
            id = static_cast<const GetBucketListMessage&>(msg).getBucketId();
            break;

        case DocumentProtocol::MESSAGE_CREATEVISITOR:
            id = static_cast<const CreateVisitorMessage&>(msg).getBuckets()[0];
            break;

        case DocumentProtocol::MESSAGE_REMOVELOCATION:
            id = static_cast<const RemoveLocationMessage&>(msg).getBucketId();
            break;

        default:
            LOG(error, "Message type '%d' not supported.", msg.getType());
            return;
        }

        // _P_A_R_A_N_O_I_A_
        if (id.getRawId() == 0) {
            mbus::Reply::UP reply(new mbus::EmptyReply());
            reply->addError(mbus::Error(mbus::ErrorCode::APP_FATAL_ERROR,
                                    "No bucket id available in message."));
            context.setReply(std::move(reply));
            return;
        }

        // Pick a distributor using ideal state algorithm
        try {
            assert(cur_distribution);
            distributor = cur_distribution->getIdealDistributorNode(*cur_state, id);
        } catch (storage::lib::TooFewBucketBitsInUseException& e) {
            auto reply = std::make_unique<WrongDistributionReply>(cur_state->toString());
            reply->addError(mbus::Error(
                    DocumentProtocol::ERROR_WRONG_DISTRIBUTION,
                    "Too few distribution bits used for given cluster state"));
            context.setReply(std::move(reply));
            return;
        } catch (storage::lib::NoDistributorsAvailableException& e) {
            // No distributors available in current cluster state. Remove
            // cluster state we cannot use and send to random target
            reset_state();
            distributor = -1;
        }
    }

    mbus::Hop hop = getRecipient(context, distributor);

    if (distributor != -1 && !hop.hasDirectives()) {
        hop = getRecipient(context, -1);
    }

    if (hop.hasDirectives()) {
        mbus::Route route = context.getRoute();
        route.setHop(0, hop);
        context.addChild(route);
    } else {
        context.setError(
                mbus::ErrorCode::NO_ADDRESS_FOR_SERVICE,
                make_string("Could not resolve a distributor to send to in cluster %s", _clusterName.c_str()));
    }
}

mbus::Hop
ContentPolicy::getRecipient(mbus::RoutingContext& context, int distributor)
{
    slobrok::api::IMirrorAPI::SpecList entries = lookup(context, createPattern(_clusterName, distributor));

    if (!entries.empty()) {
        return mbus::Hop::parse(entries[random() % entries.size()].second + "/default");
    }

    return {};
}

void
ContentPolicy::merge(mbus::RoutingContext &context)
{
    mbus::RoutingNodeIterator it = context.getChildIterator();
    mbus::Reply::UP reply = it.removeReply();

    if (reply->getType() == DocumentProtocol::REPLY_WRONGDISTRIBUTION) {
        updateStateFromReply(dynamic_cast<WrongDistributionReply&>(*reply));
    } else if (reply->hasErrors()) {
        reset_state();
    }

    context.setReply(std::move(reply));
}

void
ContentPolicy::updateStateFromReply(WrongDistributionReply& wdr)
{
    auto newState = std::make_unique<storage::lib::ClusterState>(wdr.getSystemState());
    std::lock_guard guard(_rw_lock);
    if (!_state || newState->getVersion() >= _state->getVersion()) {
        if (_state) {
            wdr.getTrace().trace(1, make_string("System state changed from version %u to %u",
                                                _state->getVersion(), newState->getVersion()));
        } else {
            wdr.getTrace().trace(1, make_string("System state set to version %u", newState->getVersion()));
        }

        _state = std::move(newState);
    } else {
        wdr.getTrace().trace(1, make_string("System state cleared because system state returned had version %d, "
                                            "while old state had version %d. New states should not have a lower version than the old.",
                                            newState->getVersion(), _state->getVersion()));
        _state.reset();
    }
}

ContentPolicy::StateSnapshot
ContentPolicy::internal_state_snapshot()
{
    std::shared_lock guard(_rw_lock);
    return {_state, _distribution};
}

std::shared_ptr<const storage::lib::ClusterState>
ContentPolicy::getSystemState() const noexcept
{
    std::shared_lock guard(_rw_lock);
    return _state;
}

void
ContentPolicy::reset_state()
{
    // It's possible for the caller to race between checking and resetting the state,
    // but this should never lead to a worse outcome than sending to a random distributor
    // as if no state had been cached prior.
    std::lock_guard guard(_rw_lock);
    _state.reset();
}

} // documentapi
