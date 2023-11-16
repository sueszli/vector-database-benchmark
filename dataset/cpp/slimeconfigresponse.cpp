// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#include "slimeconfigresponse.h"
#include <vespa/config/common/misc.h>
#include <vespa/fnet/frt/values.h>
#include <vespa/vespalib/stllike/string.h>
#include <vespa/log/log.h>
LOG_SETUP(".config.frt.slimeconfigresponse");

using namespace vespalib;
using namespace vespalib::slime;
using namespace vespalib::slime::convenience;
using namespace config::protocol::v2;

namespace config {

SlimeConfigResponse::SlimeConfigResponse(FRT_RPCRequest * request)
    : FRTConfigResponse(request),
      _key(),
      _value(),
      _trace(),
      _filled(false)
{
}

SlimeConfigResponse::~SlimeConfigResponse() = default;

void
SlimeConfigResponse::fill()
{
    if (_filled) {
        LOG(info, "SlimeConfigResponse::fill() called twice, probably a bug");
        return;
    }
    Memory json((*_returnValues)[0]._string._str);
    auto data = std::make_unique<Slime>();
    JsonFormat::decode(json, *data);
    _data = std::move(data);
    _key = readKey();
    _state = readState();
    _value = readConfigValue();
    readTrace();
    _filled = true;
    if (LOG_WOULD_LOG(debug)) {
        LOG(debug, "trace at return(%s)", _trace.toString().c_str());
    }
}

void
SlimeConfigResponse::readTrace()
{
    Inspector & root(_data->get());
    _trace.deserialize(root[RESPONSE_TRACE]);
}

ConfigKey
SlimeConfigResponse::readKey() const
{
    Inspector & root(_data->get());
    return ConfigKey(root[RESPONSE_CONFIGID].asString().make_string(),
                     root[RESPONSE_DEF_NAME].asString().make_string(),
                     root[RESPONSE_DEF_NAMESPACE].asString().make_string(),
                     root[RESPONSE_DEF_MD5].asString().make_string());
}

ConfigState
SlimeConfigResponse::readState() const
{
    const Slime & data(*_data);
    return ConfigState(data.get()[RESPONSE_CONFIG_XXHASH64].asString().make_string(),
                       data.get()[RESPONSE_CONFIG_GENERATION].asLong(),
                       data.get()[RESPONSE_APPLY_ON_RESTART].asBool());
}

vespalib::string
SlimeConfigResponse::getHostName() const
{
    Inspector & root(_data->get());
    return root[RESPONSE_CLIENT_HOSTNAME].asString().make_string();
}

} // namespace config
