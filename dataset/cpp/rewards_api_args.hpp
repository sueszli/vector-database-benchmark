#pragma once
#include <string>
#include <fc/uint128.hpp>
#include <steem/protocol/misc_utilities.hpp>
#include <steem/protocol/asset.hpp>
#include <steem/plugins/json_rpc/utility.hpp>

namespace steem { namespace plugins { namespace rewards_api {

struct simulate_curve_payouts_element {
   protocol::account_name_type  author;
   fc::string                   permlink;
   protocol::asset              payout;
};

struct simulate_curve_payouts_args
{
   protocol::curve_id curve;
   std::string        var1;
};

struct simulate_curve_payouts_return
{
   std::string                                   recent_claims;
   std::vector< simulate_curve_payouts_element > payouts;
};


} } } // steem::plugins::rewards_api

FC_REFLECT( steem::plugins::rewards_api::simulate_curve_payouts_element, (author)(permlink)(payout) )
FC_REFLECT( steem::plugins::rewards_api::simulate_curve_payouts_args, (curve)(var1) )
FC_REFLECT( steem::plugins::rewards_api::simulate_curve_payouts_return, (recent_claims)(payouts) )
