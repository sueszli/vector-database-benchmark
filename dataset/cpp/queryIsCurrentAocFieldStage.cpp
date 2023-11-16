#include "Game/AI/Query/queryIsCurrentAocFieldStage.h"
#include <evfl/Query.h>
#include "KingSystem/System/StageInfo.h"

namespace uking::query {

IsCurrentAocFieldStage::IsCurrentAocFieldStage(const InitArg& arg) : ksys::act::ai::Query(arg) {}

IsCurrentAocFieldStage::~IsCurrentAocFieldStage() = default;

int IsCurrentAocFieldStage::doQuery() {
    return ksys::StageInfo::sIsAocField;
}

void IsCurrentAocFieldStage::loadParams(const evfl::QueryArg& arg) {}

void IsCurrentAocFieldStage::loadParams() {}

}  // namespace uking::query
