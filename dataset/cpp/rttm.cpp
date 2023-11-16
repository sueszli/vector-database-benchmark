
#include <net/tcp/rttm.hpp>

using namespace net::tcp;

constexpr float RTTM::CLOCK_G;
constexpr float RTTM::K;
constexpr float RTTM::alpha;
constexpr float RTTM::beta;

/*
  When the first RTT measurement R is made, the host MUST set

  SRTT <- R
  RTTVAR <- R/2
  RTO <- SRTT + max (G, K*RTTVAR)

  where K = 4.

  When a subsequent RTT measurement R' is made, a host MUST set

  RTTVAR <- (1 - beta) * RTTVAR + beta * |SRTT - R'|
  SRTT <- (1 - alpha) * SRTT + alpha * R'

  The value of SRTT used in the update to RTTVAR is its value
  before updating SRTT itself using the second assignment.  That
  is, updating RTTVAR and SRTT MUST be computed in the above
  order.

  The above SHOULD be computed using alpha=1/8 and beta=1/4 (as
  suggested in [JK88]).

  After the computation, a host MUST update
  RTO <- SRTT + max (G, K*RTTVAR)
*/
void RTTM::rtt_measurement(milliseconds R)
{
  if(samples > 0)
  {
    RTTVAR = seconds{(1 - beta) * RTTVAR.count() + beta * std::abs((SRTT - R).count())};
    SRTT = (1 - alpha) * SRTT + alpha * R;
  }
  else
  {
    SRTT = R;
    RTTVAR = R/2;
  }
  ++samples;
  update_rto();
}
