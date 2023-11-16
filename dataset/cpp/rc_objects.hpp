#pragma once
#include <steem/chain/steem_fwd.hpp>
#include <steem/chain/util/manabar.hpp>

#include <steem/plugins/rc/rc_config.hpp>
#include <steem/plugins/rc/rc_utility.hpp>
#include <steem/plugins/rc/resource_count.hpp>

#include <steem/chain/steem_object_types.hpp>

#include <steem/protocol/asset.hpp>

#include <fc/int_array.hpp>

namespace steem { namespace plugins { namespace rc {

using namespace steem::chain;
using steem::protocol::asset;
using steem::protocol::asset_symbol_type;

#ifndef STEEM_RC_SPACE_ID
#define STEEM_RC_SPACE_ID 16
#endif

#define STEEM_RC_DRC_FLOAT_LEVEL   (20*STEEM_1_PERCENT)
#define STEEM_RC_MAX_DRC_RATE      1000

enum rc_object_types
{
   rc_resource_param_object_type          = ( STEEM_RC_SPACE_ID << 8 ),
   rc_pool_object_type                    = ( STEEM_RC_SPACE_ID << 8 ) + 1,
   rc_account_object_type                 = ( STEEM_RC_SPACE_ID << 8 ) + 2,
   rc_delegation_pool_object_type         = ( STEEM_RC_SPACE_ID << 8 ) + 3,
   rc_delegation_from_account_object_type = ( STEEM_RC_SPACE_ID << 8 ) + 4,
   rc_indel_edge_object_type              = ( STEEM_RC_SPACE_ID << 8 ) + 5,
   rc_outdel_drc_edge_object_type         = ( STEEM_RC_SPACE_ID << 8 ) + 6
};

class rc_resource_param_object : public object< rc_resource_param_object_type, rc_resource_param_object >
{
   public:
      template< typename Constructor, typename Allocator >
      rc_resource_param_object( Constructor&& c, allocator< Allocator > a )
      {
         c( *this );
      }

      rc_resource_param_object() {}

      id_type               id;
      fc::int_array< rc_resource_params, STEEM_NUM_RESOURCE_TYPES >
                            resource_param_array;
};

STEEM_OBJECT_ID_TYPE( rc_resource_param );

class rc_pool_object : public object< rc_pool_object_type, rc_pool_object >
{
   public:
      template< typename Constructor, typename Allocator >
      rc_pool_object( Constructor&& c, allocator< Allocator > a )
      {
         c( *this );
      }

      rc_pool_object() {}

      id_type               id;
      fc::int_array< int64_t, STEEM_NUM_RESOURCE_TYPES >
                            pool_array;
};

STEEM_OBJECT_ID_TYPE( rc_pool );

class rc_account_object : public object< rc_account_object_type, rc_account_object >
{
   public:
      template< typename Constructor, typename Allocator >
      rc_account_object( Constructor&& c, allocator< Allocator > a )
      {
         c( *this );
      }

      rc_account_object() {}

      id_type               id;

      account_name_type     account;
      account_name_type     creator;
      steem::chain::util::manabar   rc_manabar;
      asset                 max_rc_creation_adjustment = asset( 0, VESTS_SYMBOL );
      asset                 vests_delegated_to_pools = asset( 0, VESTS_SYMBOL );
      fc::array< account_name_type, STEEM_RC_MAX_SLOTS >
                            indel_slots;

      uint32_t              out_delegations = 0;

      // This is used for bug-catching, to match that the vesting shares in a
      // pre-op are equal to what they were at the last post-op.
      int64_t               last_max_rc = 0;
};

STEEM_OBJECT_ID_TYPE( rc_account );

/**
 * Represents a delegation pool.
 */
class rc_delegation_pool_object : public object< rc_delegation_pool_object_type, rc_delegation_pool_object >
{
   public:
      template< typename Constructor, typename Allocator >
      rc_delegation_pool_object( Constructor&& c, allocator< Allocator > a )
      {
         c( *this );
      }

      rc_delegation_pool_object() {}

      id_type                       id;

      account_name_type             account;
      asset_symbol_type             asset_symbol;
      steem::chain::util::manabar   rc_pool_manabar;
      int64_t                       max_rc = 0;
};

STEEM_OBJECT_ID_TYPE( rc_delegation_pool );

/**
 * Represents the total amount of an asset delegated by a user.
 *
 * Only used for SMT support.
 */
class rc_delegation_from_account_object : public object< rc_delegation_from_account_object_type, rc_delegation_from_account_object >
{
   public:
      template< typename Constructor, typename Allocator >
      rc_delegation_from_account_object( Constructor&& c, allocator< Allocator > a )
      {
         c( *this );
      }

      rc_delegation_from_account_object() {}

      id_type                       id;

      account_name_type             account;
      asset                         amount;

      asset_symbol_type get_asset_symbol()const
      {  return amount.symbol;                             }
};

STEEM_OBJECT_ID_TYPE( rc_delegation_from_account );

/**
 * Represents a delegation from a user to a pool.
 */
class rc_indel_edge_object : public object< rc_indel_edge_object_type, rc_indel_edge_object >
{
   public:
      template< typename Constructor, typename Allocator >
      rc_indel_edge_object( Constructor&& c, allocator< Allocator > a )
      {
         c( *this );
      }

      rc_indel_edge_object() {}

      asset_symbol_type get_asset_symbol()const
      {  return amount.symbol;                             }

      id_type                       id;
      account_name_type             from_account;
      account_name_type             to_pool;
      asset                         amount;
};

STEEM_OBJECT_ID_TYPE( rc_indel_edge );

/**
 * Represents a delegation from a pool to a user based on delegated resource credits (DRC).
 *
 * In the case of a pool that is not under heavy load, DRC:RC has a 1:1 exchange rate.
 *
 * However, if the pool drops below STEEM_RC_DRC_FLOAT_LEVEL, DRC:RC exchange rate starts
 * to rise according to `f(x) = 1/(a+b*x)` where `x` is the pool level, and coefficients `a`,
 * `b` are set such that `f(STEEM_RC_DRC_FLOAT_LEVEL) = 1` and `f(0) = STEEM_RC_MAX_DRC_RATE`.
 *
 * This ensures the limited RC of oversubscribed pools under heavy load are
 * shared "fairly" among their users proportionally to DRC.  This logic
 * provides a smooth transition between the "fiat regime" (i.e. DRC represent
 * a direct allocation of RC) and the "proportional regime" (i.e. DRC represent
 * the fraction of RC that the user is allowed).
 */
class rc_outdel_drc_edge_object : public object< rc_outdel_drc_edge_object_type, rc_outdel_drc_edge_object >
{
   public:
      template< typename Constructor, typename Allocator >
      rc_outdel_drc_edge_object( Constructor&& c, allocator< Allocator > a )
      {
         c( *this );
      }

      rc_outdel_drc_edge_object() {}

      id_type                       id;
      account_name_type             from_pool;
      account_name_type             to_account;
      asset_symbol_type             asset_symbol;
      steem::chain::util::manabar   drc_manabar;
      int64_t                       drc_max_mana = 0;
};

STEEM_OBJECT_ID_TYPE( rc_outdel_drc_edge );

int64_t get_maximum_rc( const steem::chain::account_object& account, const rc_account_object& rc_account );

struct by_edge;
struct by_account_symbol;
struct by_pool;

typedef multi_index_container<
   rc_resource_param_object,
   indexed_by<
      ordered_unique< tag< by_id >, member< rc_resource_param_object, rc_resource_param_object::id_type, &rc_resource_param_object::id > >
   >,
   allocator< rc_resource_param_object >
> rc_resource_param_index;

typedef multi_index_container<
   rc_pool_object,
   indexed_by<
      ordered_unique< tag< by_id >, member< rc_pool_object, rc_pool_object::id_type, &rc_pool_object::id > >
   >,
   allocator< rc_pool_object >
> rc_pool_index;

typedef multi_index_container<
   rc_account_object,
   indexed_by<
      ordered_unique< tag< by_id >, member< rc_account_object, rc_account_object::id_type, &rc_account_object::id > >,
      ordered_unique< tag< by_name >, member< rc_account_object, account_name_type, &rc_account_object::account > >
   >,
   allocator< rc_account_object >
> rc_account_index;

typedef multi_index_container<
   rc_delegation_pool_object,
   indexed_by<
      ordered_unique< tag< by_id >, member< rc_delegation_pool_object, rc_delegation_pool_object::id_type, &rc_delegation_pool_object::id > >,
      ordered_unique< tag< by_account_symbol >,
         composite_key< rc_delegation_pool_object,
            member< rc_delegation_pool_object, account_name_type, &rc_delegation_pool_object::account >,
            member< rc_delegation_pool_object, asset_symbol_type, &rc_delegation_pool_object::asset_symbol >
         >
      >
   >,
   allocator< rc_delegation_pool_object >
> rc_delegation_pool_index;

typedef multi_index_container<
   rc_delegation_from_account_object,
   indexed_by<
      ordered_unique< tag< by_id >, member< rc_delegation_from_account_object, rc_delegation_from_account_object::id_type, &rc_delegation_from_account_object::id > >,
      ordered_unique< tag< by_account_symbol >,
         composite_key< rc_delegation_from_account_object,
            member< rc_delegation_from_account_object, account_name_type, &rc_delegation_from_account_object::account >,
            const_mem_fun< rc_delegation_from_account_object, asset_symbol_type, &rc_delegation_from_account_object::get_asset_symbol >
         >
      >
   >,
   allocator< rc_delegation_from_account_object >
> rc_delegation_from_account_index;

typedef multi_index_container<
   rc_indel_edge_object,
   indexed_by<
      ordered_unique< tag< by_id >, member< rc_indel_edge_object, rc_indel_edge_object::id_type, &rc_indel_edge_object::id > >,
      ordered_unique< tag< by_edge >,
         composite_key< rc_indel_edge_object,
            member< rc_indel_edge_object, account_name_type, &rc_indel_edge_object::from_account >,
            const_mem_fun< rc_indel_edge_object, asset_symbol_type, &rc_indel_edge_object::get_asset_symbol >,
            member< rc_indel_edge_object, account_name_type, &rc_indel_edge_object::to_pool >
         >
      >,
      ordered_unique< tag< by_pool >,
         composite_key< rc_indel_edge_object,
            member< rc_indel_edge_object, account_name_type, &rc_indel_edge_object::to_pool >,
            const_mem_fun< rc_indel_edge_object, asset_symbol_type, &rc_indel_edge_object::get_asset_symbol >,
            member< rc_indel_edge_object, account_name_type, &rc_indel_edge_object::from_account >
         >
      >
   >,
   allocator< rc_indel_edge_object >
> rc_indel_edge_index;

typedef multi_index_container<
   rc_outdel_drc_edge_object,
   indexed_by<
      ordered_unique< tag< by_id >, member< rc_outdel_drc_edge_object, rc_outdel_drc_edge_id_type, &rc_outdel_drc_edge_object::id > >,
      ordered_unique< tag< by_edge >,
         composite_key< rc_outdel_drc_edge_object,
            member< rc_outdel_drc_edge_object, account_name_type, &rc_outdel_drc_edge_object::from_pool >,
            member< rc_outdel_drc_edge_object, account_name_type, &rc_outdel_drc_edge_object::to_account >,
            member< rc_outdel_drc_edge_object, asset_symbol_type, &rc_outdel_drc_edge_object::asset_symbol >
         >
      >,
      ordered_unique< tag< by_pool >,
         composite_key< rc_outdel_drc_edge_object,
            member< rc_outdel_drc_edge_object, account_name_type, &rc_outdel_drc_edge_object::from_pool >,
            member< rc_outdel_drc_edge_object, asset_symbol_type, &rc_outdel_drc_edge_object::asset_symbol >,
            member< rc_outdel_drc_edge_object, rc_outdel_drc_edge_id_type, &rc_outdel_drc_edge_object::id >
         >
      >
   >,
   allocator< rc_outdel_drc_edge_object >
> rc_outdel_drc_edge_index;

} } } // steem::plugins::rc

FC_REFLECT( steem::plugins::rc::rc_resource_param_object, (id)(resource_param_array) )
CHAINBASE_SET_INDEX_TYPE( steem::plugins::rc::rc_resource_param_object, steem::plugins::rc::rc_resource_param_index )

FC_REFLECT( steem::plugins::rc::rc_pool_object, (id)(pool_array) )
CHAINBASE_SET_INDEX_TYPE( steem::plugins::rc::rc_pool_object, steem::plugins::rc::rc_pool_index )

FC_REFLECT( steem::plugins::rc::rc_account_object,
   (id)
   (account)
   (creator)
   (rc_manabar)
   (max_rc_creation_adjustment)
   (vests_delegated_to_pools)
   (out_delegations)
   (indel_slots)
   (last_max_rc)
   )
CHAINBASE_SET_INDEX_TYPE( steem::plugins::rc::rc_account_object, steem::plugins::rc::rc_account_index )

FC_REFLECT( steem::plugins::rc::rc_delegation_pool_object,
   (id)
   (account)
   (asset_symbol)
   (rc_pool_manabar)
   (max_rc)
   )
CHAINBASE_SET_INDEX_TYPE( steem::plugins::rc::rc_delegation_pool_object, steem::plugins::rc::rc_delegation_pool_index )

FC_REFLECT( steem::plugins::rc::rc_delegation_from_account_object,
   (id)
   (account)
   (amount)
   )
CHAINBASE_SET_INDEX_TYPE( steem::plugins::rc::rc_delegation_from_account_object, steem::plugins::rc::rc_delegation_from_account_index )

FC_REFLECT( steem::plugins::rc::rc_indel_edge_object,
   (id)
   (from_account)
   (to_pool)
   (amount)
   )
CHAINBASE_SET_INDEX_TYPE( steem::plugins::rc::rc_indel_edge_object, steem::plugins::rc::rc_indel_edge_index )

FC_REFLECT( steem::plugins::rc::rc_outdel_drc_edge_object,
   (id)
   (from_pool)
   (to_account)
   (asset_symbol)
   (drc_manabar)
   (drc_max_mana)
   )
CHAINBASE_SET_INDEX_TYPE( steem::plugins::rc::rc_outdel_drc_edge_object, steem::plugins::rc::rc_outdel_drc_edge_index )
