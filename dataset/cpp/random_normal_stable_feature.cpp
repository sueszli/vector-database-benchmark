// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "random_normal_stable_feature.h"
#include "utils.h"
#include <vespa/searchlib/fef/properties.h>
#include <vespa/vespalib/util/stash.h>
#include <cinttypes>

#include <vespa/log/log.h>
LOG_SETUP(".features.randomnormalstablefeature");

namespace search::features {

RandomNormalStableExecutor::RandomNormalStableExecutor(uint64_t seed, double mean, double stddev) :
    search::fef::FeatureExecutor(),
    _rnd(mean, stddev, false),  // don't use spares, as we reset seed on every generation
    _seed(seed)
{
    LOG(debug, "RandomNormalStableExecutor: seed=%" PRIu64 ", mean=%f, stddev=%f", seed, mean, stddev);
}

void
RandomNormalStableExecutor::execute(uint32_t docId)
{
    _rnd.seed(_seed + docId);
    outputs().set_number(0, _rnd.next());
}

RandomNormalStableBlueprint::RandomNormalStableBlueprint() :
    search::fef::Blueprint("randomNormalStable"),
    _seed(0),
    _mean(0.0),
    _stddev(1.0)
{
}

void
RandomNormalStableBlueprint::visitDumpFeatures(const search::fef::IIndexEnvironment &,
                                   search::fef::IDumpFeatureVisitor &) const
{
}

search::fef::Blueprint::UP
RandomNormalStableBlueprint::createInstance() const
{
    return std::make_unique<RandomNormalStableBlueprint>();
}

bool
RandomNormalStableBlueprint::setup(const search::fef::IIndexEnvironment & env,
                       const search::fef::ParameterList & params)
{
    search::fef::Property p = env.getProperties().lookup(getName(), "seed");
    if (p.found()) {
        _seed = util::strToNum<uint64_t>(p.get());
    }
    if (params.size() > 0) {
        _mean = params[0].asDouble();
    }
    if (params.size() > 1) {
        _stddev = params[1].asDouble();
    }

    describeOutput("out" , "A random value drawn from the Gaussian distribution that is stable for a given match (document and query)");

    return true;
}

search::fef::FeatureExecutor &
RandomNormalStableBlueprint::createExecutor(const search::fef::IQueryEnvironment &env, vespalib::Stash &stash) const
{
    uint64_t seed = _seed;
    if (seed == 0) {
        seed = util::strToNum<uint64_t>
                (env.getProperties().lookup(getName(), "seed").get("1024")); // default seed
    }
    return stash.create<RandomNormalStableExecutor>(seed, _mean, _stddev);
}

}
