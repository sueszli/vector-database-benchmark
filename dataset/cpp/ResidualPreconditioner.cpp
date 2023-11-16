#include "acceleration/impl/ResidualPreconditioner.hpp"
#include <cstddef>
#include <vector>
#include "utils/IntraComm.hpp"
#include "utils/assertion.hpp"

namespace precice::acceleration::impl {

ResidualPreconditioner::ResidualPreconditioner(int maxNonConstTimeWindows)
    : Preconditioner(maxNonConstTimeWindows)
{
}

void ResidualPreconditioner::_update_(bool                   timeWindowComplete,
                                      const Eigen::VectorXd &oldValues,
                                      const Eigen::VectorXd &res)
{
  if (not timeWindowComplete) {
    std::vector<double> norms(_subVectorSizes.size(), 0.0);

    int offset = 0;
    for (size_t k = 0; k < _subVectorSizes.size(); k++) {
      Eigen::VectorXd part = Eigen::VectorXd::Zero(_subVectorSizes[k]);
      for (size_t i = 0; i < _subVectorSizes[k]; i++) {
        part(i) = res(i + offset);
      }
      norms[k] = utils::IntraComm::l2norm(part);
      offset += _subVectorSizes[k];
      PRECICE_ASSERT(norms[k] > 0.0);
    }

    offset = 0;
    for (size_t k = 0; k < _subVectorSizes.size(); k++) {
      for (size_t i = 0; i < _subVectorSizes[k]; i++) {
        _weights[i + offset]    = 1.0 / norms[k];
        _invWeights[i + offset] = norms[k];
      }
      offset += _subVectorSizes[k];
    }

    _requireNewQR = true;
  }
}

} // namespace precice::acceleration::impl
