// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "debug_wait.h"
#include <vespa/vespalib/util/time.h>
#include <vespa/vespalib/util/stash.h>

using namespace search::fef;

namespace search::features {

//-----------------------------------------------------------------------------

class DebugWaitExecutor : public FeatureExecutor
{
private:
    DebugWaitParams _params;

public:
    DebugWaitExecutor(const IQueryEnvironment &env, const DebugWaitParams &params);
    void execute(uint32_t docId) override;
};

DebugWaitExecutor::DebugWaitExecutor(const IQueryEnvironment &, const DebugWaitParams &params)
    : _params(params)
{
}

using namespace std::chrono;

void
DebugWaitExecutor::execute(uint32_t)
{
    vespalib::Timer timer;
    vespalib::Timer::waitAtLeast(vespalib::from_s(_params.waitTime), _params.busyWait);
    outputs().set_number(0, vespalib::to_s(timer.elapsed()));
}

//-----------------------------------------------------------------------------

DebugWaitBlueprint::DebugWaitBlueprint()
    : Blueprint("debugWait"),
      _params()
{
}

void
DebugWaitBlueprint::visitDumpFeatures(const IIndexEnvironment &, IDumpFeatureVisitor &) const
{
}

Blueprint::UP
DebugWaitBlueprint::createInstance() const
{
    return std::make_unique<DebugWaitBlueprint>();
}

bool
DebugWaitBlueprint::setup(const IIndexEnvironment &env, const ParameterList &params)
{
    (void)env;
    _params.waitTime = params[0].asDouble();
    _params.busyWait = (params[1].asDouble() == 1.0);

    describeOutput("out", "actual time waited");
    return true;
}

FeatureExecutor &
DebugWaitBlueprint::createExecutor(const IQueryEnvironment &env, vespalib::Stash &stash) const
{
    return stash.create<DebugWaitExecutor>(env, _params);
}

//-----------------------------------------------------------------------------

}
