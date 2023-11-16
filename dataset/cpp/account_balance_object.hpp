#pragma once

#include <steem/chain/steem_object_types.hpp>
#include <steem/chain/util/manabar.hpp>

#include <steem/protocol/smt_operations.hpp>

namespace steem { namespace chain {

/**
 * Class responsible for holding regular (i.e. non-reward) balance of SMT for given account.
 * It has not been unified with reward balance object counterpart, due to different number
 * of fields needed to hold balances (2 for regular, 3 for reward).
 */
class account_regular_balance_object : public object< account_regular_balance_object_type, account_regular_balance_object >
{
   STEEM_STD_ALLOCATOR_CONSTRUCTOR( account_regular_balance_object );

public:
   template <typename Constructor, typename Allocator>
   account_regular_balance_object(Constructor&& c, allocator< Allocator > a)
   {
      c( *this );
   }

   id_type             id;
   account_name_type   name;
   asset               liquid;

   asset               vesting_shares;
   asset               delegated_vesting_shares;
   asset               received_vesting_shares;

   asset               vesting_withdraw_rate;
   time_point_sec      next_vesting_withdrawal = fc::time_point_sec::maximum();
   share_type          withdrawn               = 0;
   share_type          to_withdraw             = 0;

   util::manabar       voting_manabar;
   util::manabar       downvote_manabar;

   fc::time_point_sec  last_vote_time;

   asset_symbol_type get_liquid_symbol() const
   {
      return liquid.symbol;
   }

   asset_symbol_type get_stripped_symbol() const
   {
      return asset_symbol_type::from_asset_num( liquid.symbol.get_stripped_precision_smt_num() );
   }

   void initialize_assets( asset_symbol_type liquid_symbol )
   {
      liquid                   = asset( 0, liquid_symbol );
      vesting_shares           = asset( 0, liquid_symbol.get_paired_symbol() );
      delegated_vesting_shares = asset( 0, liquid_symbol.get_paired_symbol() );
      received_vesting_shares  = asset( 0, liquid_symbol.get_paired_symbol() );
      vesting_withdraw_rate    = asset( 0, liquid_symbol.get_paired_symbol() );
   }

   void add_vesting( const asset& shares, const asset& vesting_value )
   {
      // There's no need to store vesting value (in liquid SMT variant) in regular balance.
      vesting_shares += shares;
   }

   bool validate() const
   {
      return
         liquid.symbol         == vesting_shares.symbol.get_paired_symbol() &&
         vesting_shares.symbol == delegated_vesting_shares.symbol &&
         vesting_shares.symbol == received_vesting_shares.symbol &&
         vesting_shares.symbol == vesting_withdraw_rate.symbol;
   }
};

/**
 * Class responsible for holding reward balance of SMT for given account.
 * It has not been unified with regular balance object counterpart, due to different number
 * of fields needed to hold balances (2 for regular, 3 for reward).
 */
class account_rewards_balance_object : public object< account_rewards_balance_object_type, account_rewards_balance_object >
{
   STEEM_STD_ALLOCATOR_CONSTRUCTOR( account_rewards_balance_object );

public:
   template <typename Constructor, typename Allocator>
   account_rewards_balance_object(Constructor&& c, allocator< Allocator > a)
   {
      c( *this );
   }

   id_type             id;
   account_name_type   name;
   asset               pending_liquid;          /// 'reward_steem_balance' for pending STEEM
   asset               pending_vesting_shares;  /// 'reward_vesting_balance' for pending VESTS
   asset               pending_vesting_value;   /// 'reward_vesting_steem' for pending VESTS

   asset_symbol_type get_liquid_symbol() const
   {
      return pending_liquid.symbol;
   }

   asset_symbol_type get_stripped_symbol() const
   {
      return asset_symbol_type::from_asset_num( pending_liquid.symbol.get_stripped_precision_smt_num() );
   }

   void initialize_assets( asset_symbol_type liquid_symbol )
   {
      pending_liquid         = asset( 0, liquid_symbol );
      pending_vesting_shares = asset( 0, liquid_symbol.get_paired_symbol() );
      pending_vesting_value  = asset( 0, liquid_symbol );
   }

   void add_vesting( const asset& vesting_shares, const asset& vesting_value )
   {
      pending_vesting_shares += vesting_shares;
      pending_vesting_value  += vesting_value;
   }

   bool validate() const
   {
      return
         pending_liquid.symbol == pending_vesting_shares.symbol.get_paired_symbol() &&
         pending_liquid.symbol == pending_vesting_value.symbol;
   }
};

struct by_name_liquid_symbol;
struct by_next_vesting_withdrawal;
struct by_name_stripped_symbol;

typedef multi_index_container <
   account_regular_balance_object,
   indexed_by <
      ordered_unique< tag< by_id >,
         member< account_regular_balance_object, account_regular_balance_id_type, &account_regular_balance_object::id >
      >,
      ordered_unique< tag< by_name_liquid_symbol >,
         composite_key< account_regular_balance_object,
            member< account_regular_balance_object, account_name_type, &account_regular_balance_object::name >,
            const_mem_fun< account_regular_balance_object, asset_symbol_type, &account_regular_balance_object::get_liquid_symbol >
         >
      >,
      ordered_unique< tag< by_next_vesting_withdrawal >,
         composite_key< account_regular_balance_object,
            member< account_regular_balance_object, time_point_sec, &account_regular_balance_object::next_vesting_withdrawal >,
            member< account_regular_balance_object, account_name_type, &account_regular_balance_object::name >,
            const_mem_fun< account_regular_balance_object, asset_symbol_type, &account_regular_balance_object::get_liquid_symbol >
         >
      >,
      ordered_unique< tag< by_name_stripped_symbol >,
         composite_key< account_regular_balance_object,
            member< account_regular_balance_object, account_name_type, &account_regular_balance_object::name >,
            const_mem_fun< account_regular_balance_object, asset_symbol_type, &account_regular_balance_object::get_stripped_symbol >
         >
      >
   >,
   allocator< account_regular_balance_object >
> account_regular_balance_index;

typedef multi_index_container <
   account_rewards_balance_object,
   indexed_by <
      ordered_unique< tag< by_id >,
         member< account_rewards_balance_object, account_rewards_balance_id_type, &account_rewards_balance_object::id >
      >,
      ordered_unique< tag< by_name_liquid_symbol >,
         composite_key< account_rewards_balance_object,
            member< account_rewards_balance_object, account_name_type, &account_rewards_balance_object::name >,
            const_mem_fun< account_rewards_balance_object, asset_symbol_type, &account_rewards_balance_object::get_liquid_symbol >
         >
      >,
      ordered_unique< tag< by_name_stripped_symbol >,
         composite_key< account_rewards_balance_object,
            member< account_rewards_balance_object, account_name_type, &account_rewards_balance_object::name >,
            const_mem_fun< account_rewards_balance_object, asset_symbol_type, &account_rewards_balance_object::get_stripped_symbol >
         >
      >
   >,
   allocator< account_rewards_balance_object >
> account_rewards_balance_index;

} } // namespace steem::chain

FC_REFLECT( steem::chain::account_regular_balance_object,
   (id)
   (name)
   (liquid)
   (vesting_shares)
   (delegated_vesting_shares)
   (received_vesting_shares)
   (vesting_withdraw_rate)
   (next_vesting_withdrawal)
   (withdrawn)
   (to_withdraw)
   (voting_manabar)
   (downvote_manabar)
   (last_vote_time)
)

FC_REFLECT( steem::chain::account_rewards_balance_object,
   (id)
   (name)
   (pending_liquid)
   (pending_vesting_shares)
   (pending_vesting_value)
)

CHAINBASE_SET_INDEX_TYPE( steem::chain::account_regular_balance_object, steem::chain::account_regular_balance_index )
CHAINBASE_SET_INDEX_TYPE( steem::chain::account_rewards_balance_object, steem::chain::account_rewards_balance_index )
