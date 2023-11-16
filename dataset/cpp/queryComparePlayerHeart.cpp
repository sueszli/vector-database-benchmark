#include "Game/AI/Query/queryComparePlayerHeart.h"
#include <evfl/Query.h>
#include "KingSystem/ActorSystem/actPlayerInfo.h"

namespace uking::query {

ComparePlayerHeart::ComparePlayerHeart(const InitArg& arg) : ksys::act::ai::Query(arg) {}

ComparePlayerHeart::~ComparePlayerHeart() = default;

int ComparePlayerHeart::doQuery() {
    auto* pi = ksys::act::PlayerInfo::instance();
    if (pi == nullptr)
        return 0;

    return pi->getLife() >= *mThreshold;
}

void ComparePlayerHeart::loadParams(const evfl::QueryArg& arg) {
    loadInt(arg.param_accessor, "Threshold");
}

void ComparePlayerHeart::loadParams() {
    getDynamicParam(&mThreshold, "Threshold");
}

}  // namespace uking::query
