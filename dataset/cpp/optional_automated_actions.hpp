#pragma once

#include <steem/protocol/steem_optional_actions.hpp>

#include <steem/protocol/operation_util.hpp>

namespace steem { namespace protocol {

   /** NOTE: do not change the order of any actions or it will trigger a hardfork.
    */
   typedef fc::static_variant<
            smt_token_emission_action
#ifdef IS_TEST_NET
            ,example_optional_action
#endif
         > optional_automated_action;

} } // steem::protocol

STEEM_DECLARE_OPERATION_TYPE( steem::protocol::optional_automated_action );

FC_REFLECT_TYPENAME( steem::protocol::optional_automated_action );
