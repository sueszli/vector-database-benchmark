#include "action/RecorderAction.hpp"
#include <vector>
#include "action/Action.hpp"
#include "mesh/SharedPointer.hpp"

namespace precice::action {

RecorderAction::RecorderAction(
    Timing               timing,
    const mesh::PtrMesh &mesh)
    : Action(timing, mesh) {}

void RecorderAction::performAction(double time)
{
  records.push_back(Record{
      getTiming(), time});
}

std::vector<RecorderAction::Record> RecorderAction::records{};

void RecorderAction::reset()
{
  records.clear();
}

} // namespace precice::action
