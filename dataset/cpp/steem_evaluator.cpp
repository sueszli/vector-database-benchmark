#include <steem/chain/steem_fwd.hpp>

#include <steem/chain/steem_evaluator.hpp>
#include <steem/chain/database.hpp>
#include <steem/chain/custom_operation_interpreter.hpp>
#include <steem/chain/steem_objects.hpp>
#include <steem/chain/smt_objects.hpp>
#include <steem/chain/witness_objects.hpp>
#include <steem/chain/block_summary_object.hpp>
#include <steem/chain/smt_objects/account_balance_object.hpp>

#include <steem/chain/util/reward.hpp>
#include <steem/chain/util/manabar.hpp>

#include <fc/macros.hpp>

#ifndef IS_LOW_MEM
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic push
#if !defined( __clang__ )
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#include <diff_match_patch.h>
#pragma GCC diagnostic pop
#pragma GCC diagnostic pop
#include <boost/locale/encoding_utf.hpp>

using boost::locale::conv::utf_to_utf;

std::wstring utf8_to_wstring(const std::string& str)
{
    return utf_to_utf<wchar_t>(str.c_str(), str.c_str() + str.size());
}

std::string wstring_to_utf8(const std::wstring& str)
{
    return utf_to_utf<char>(str.c_str(), str.c_str() + str.size());
}

#endif

#include <fc/uint128.hpp>
#include <fc/utf8.hpp>

#include <limits>

namespace steem { namespace chain {
   using fc::uint128_t;

inline void validate_permlink_0_1( const string& permlink )
{
   FC_ASSERT( permlink.size() > STEEM_MIN_PERMLINK_LENGTH && permlink.size() < STEEM_MAX_PERMLINK_LENGTH, "Permlink is not a valid size." );

   for( const auto& c : permlink )
   {
      switch( c )
      {
         case 'a': case 'b': case 'c': case 'd': case 'e': case 'f': case 'g': case 'h': case 'i':
         case 'j': case 'k': case 'l': case 'm': case 'n': case 'o': case 'p': case 'q': case 'r':
         case 's': case 't': case 'u': case 'v': case 'w': case 'x': case 'y': case 'z': case '0':
         case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8': case '9':
         case '-':
            break;
         default:
            FC_ASSERT( false, "Invalid permlink character: ${s}", ("s", std::string() + c ) );
      }
   }
}

template< bool force_canon >
void copy_legacy_chain_properties( chain_properties& dest, const legacy_chain_properties& src )
{
   dest.account_creation_fee = src.account_creation_fee.to_asset< force_canon >();
   dest.maximum_block_size = src.maximum_block_size;
   dest.sbd_interest_rate = src.sbd_interest_rate;
}

void witness_update_evaluator::do_apply( const witness_update_operation& o )
{
   _db.get_account( o.owner ); // verify owner exists

   if ( _db.has_hardfork( STEEM_HARDFORK_0_14__410 ) )
   {
      FC_ASSERT( o.props.account_creation_fee.symbol.is_canon() );
      if( o.props.account_creation_fee.amount > STEEM_MAX_ACCOUNT_CREATION_FEE ) wlog( "NOTIFYALERT! account creation fee exceeds max in replay." );
      if( _db.has_hardfork( STEEM_HARDFORK_0_20__2651 ) || _db.is_producing() )
      {
         FC_TODO( "Move to validate() after HF20" );
         FC_ASSERT( o.props.account_creation_fee.amount <= STEEM_MAX_ACCOUNT_CREATION_FEE, "account_creation_fee greater than maximum account creation fee" );
      }
   }
   else if( !o.props.account_creation_fee.symbol.is_canon() )
   {
      // after HF, above check can be moved to validate() if reindex doesn't show this warning
      wlog( "Wrong fee symbol in block ${b}", ("b", _db.head_block_num()+1) );
   }

   FC_TODO( "Check and move this to validate after HF 20" );
   if( o.props.maximum_block_size > STEEM_SOFT_MAX_BLOCK_SIZE ) wlog( "NOTIFYALERT! max block size exceeds soft limit in replay" );
   if( _db.has_hardfork( STEEM_HARDFORK_0_20__2642 ))
   {
      FC_ASSERT( o.props.maximum_block_size <= STEEM_SOFT_MAX_BLOCK_SIZE, "Max block size cannot be more than 2MiB" );
   }

   const auto& by_witness_name_idx = _db.get_index< witness_index >().indices().get< by_name >();
   auto wit_itr = by_witness_name_idx.find( o.owner );
   if( wit_itr != by_witness_name_idx.end() )
   {
      _db.modify( *wit_itr, [&]( witness_object& w ) {
         from_string( w.url, o.url );
         w.signing_key        = o.block_signing_key;
         copy_legacy_chain_properties< false >( w.props, o.props );
      });
   }
   else
   {
      _db.create< witness_object >( [&]( witness_object& w ) {
         w.owner              = o.owner;
         from_string( w.url, o.url );
         w.signing_key        = o.block_signing_key;
         w.created            = _db.head_block_time();
         copy_legacy_chain_properties< false >( w.props, o.props );
      });
   }
}

struct witness_properties_change_flags
{
   uint32_t account_creation_changed       : 1;
   uint32_t max_block_changed              : 1;
   uint32_t sbd_interest_changed           : 1;
   uint32_t account_subsidy_budget_changed : 1;
   uint32_t account_subsidy_decay_changed  : 1;
   uint32_t key_changed                    : 1;
   uint32_t sbd_exchange_changed           : 1;
   uint32_t url_changed                    : 1;
};

void witness_set_properties_evaluator::do_apply( const witness_set_properties_operation& o )
{
   FC_ASSERT( _db.has_hardfork( STEEM_HARDFORK_0_20__1620 ), "witness_set_properties_evaluator not enabled until HF 20" );

   const auto& witness = _db.get< witness_object, by_name >( o.owner ); // verifies witness exists;

   // Capture old properties. This allows only updating the object once.
   chain_properties  props;
   public_key_type   signing_key;
   price             sbd_exchange_rate;
   time_point_sec    last_sbd_exchange_update;
   string            url;

   witness_properties_change_flags flags;

   auto itr = o.props.find( "key" );

   // This existence of 'key' is checked in witness_set_properties_operation::validate
   fc::raw::unpack_from_vector( itr->second, signing_key );
   FC_ASSERT( signing_key == witness.signing_key, "'key' does not match witness signing key.",
      ("key", signing_key)("signing_key", witness.signing_key) );

   itr = o.props.find( "account_creation_fee" );
   flags.account_creation_changed = itr != o.props.end();
   if( flags.account_creation_changed )
   {
      fc::raw::unpack_from_vector( itr->second, props.account_creation_fee );
      if( props.account_creation_fee.amount > STEEM_MAX_ACCOUNT_CREATION_FEE ) wlog( "NOTIFYALERT! account creation fee exceeds max in replay." );
      if( _db.has_hardfork( STEEM_HARDFORK_0_20__2651 ) || _db.is_producing() )
      {
         FC_TODO( "Move to validate() after HF20" );
         FC_ASSERT( props.account_creation_fee.amount <= STEEM_MAX_ACCOUNT_CREATION_FEE, "account_creation_fee greater than maximum account creation fee" );
      }
   }

   itr = o.props.find( "maximum_block_size" );
   flags.max_block_changed = itr != o.props.end();
   if( flags.max_block_changed )
   {
      fc::raw::unpack_from_vector( itr->second, props.maximum_block_size );
   }

   itr = o.props.find( "sbd_interest_rate" );
   flags.sbd_interest_changed = itr != o.props.end();
   if( flags.sbd_interest_changed )
   {
      fc::raw::unpack_from_vector( itr->second, props.sbd_interest_rate );
   }

   itr = o.props.find( "account_subsidy_budget" );
   flags.account_subsidy_budget_changed = itr != o.props.end();
   if( flags.account_subsidy_budget_changed )
   {
      fc::raw::unpack_from_vector( itr->second, props.account_subsidy_budget );
   }

   itr = o.props.find( "account_subsidy_decay" );
   flags.account_subsidy_decay_changed = itr != o.props.end();
   if( flags.account_subsidy_decay_changed )
   {
      fc::raw::unpack_from_vector( itr->second, props.account_subsidy_decay );
   }

   itr = o.props.find( "new_signing_key" );
   flags.key_changed = itr != o.props.end();
   if( flags.key_changed )
   {
      fc::raw::unpack_from_vector( itr->second, signing_key );
   }

   itr = o.props.find( "sbd_exchange_rate" );
   flags.sbd_exchange_changed = itr != o.props.end();
   if( flags.sbd_exchange_changed )
   {
      fc::raw::unpack_from_vector( itr->second, sbd_exchange_rate );
      last_sbd_exchange_update = _db.head_block_time();
   }

   itr = o.props.find( "url" );
   flags.url_changed = itr != o.props.end();
   if( flags.url_changed )
   {
      fc::raw::unpack_from_vector< std::string >( itr->second, url );
   }

   _db.modify( witness, [&]( witness_object& w )
   {
      if( flags.account_creation_changed )
      {
         w.props.account_creation_fee = props.account_creation_fee;
      }

      if( flags.max_block_changed )
      {
         w.props.maximum_block_size = props.maximum_block_size;
      }

      if( flags.sbd_interest_changed )
      {
         w.props.sbd_interest_rate = props.sbd_interest_rate;
      }

      if( flags.account_subsidy_budget_changed )
      {
         w.props.account_subsidy_budget = props.account_subsidy_budget;
      }

      if( flags.account_subsidy_decay_changed )
      {
         w.props.account_subsidy_decay = props.account_subsidy_decay;
      }

      if( flags.key_changed )
      {
         w.signing_key = signing_key;
      }

      if( flags.sbd_exchange_changed )
      {
         w.sbd_exchange_rate = sbd_exchange_rate;
         w.last_sbd_exchange_update = last_sbd_exchange_update;
      }

      if( flags.url_changed )
      {
         from_string( w.url, url );
      }
   });
}

void verify_authority_accounts_exist(
   const database& db,
   const authority& auth,
   const account_name_type& auth_account,
   authority::classification auth_class)
{
   for( const std::pair< account_name_type, weight_type >& aw : auth.account_auths )
   {
      const account_object* a = db.find_account( aw.first );
      FC_ASSERT( a != nullptr, "New ${ac} authority on account ${aa} references non-existing account ${aref}",
         ("aref", aw.first)("ac", auth_class)("aa", auth_account) );
   }
}

void initialize_account_object( account_object& acc, const account_name_type& name, const public_key_type& key,
   const dynamic_global_property_object& props, bool mined, const account_name_type& recovery_account, uint32_t hardfork )
{
   acc.name = name;
   acc.memo_key = key;
   acc.created = props.time;
   acc.voting_manabar.last_update_time = props.time.sec_since_epoch();
   acc.downvote_manabar.last_update_time = props.time.sec_since_epoch();
   acc.mined = mined;

   if( hardfork < STEEM_HARDFORK_0_20__2539 )
   {
      acc.voting_manabar.current_mana = STEEM_100_PERCENT;
   }

   if( hardfork >= STEEM_HARDFORK_0_11 )
   {
      FC_TODO( "If after HF 20, there are no temp account creations, the HF check can be removed." )
      if( recovery_account == STEEM_TEMP_ACCOUNT ) wlog( "NOTIFYALERT! Temp account creation during replay" );
      if( ( hardfork < STEEM_HARDFORK_0_20__1782 ) || ( recovery_account != STEEM_TEMP_ACCOUNT ) )
      {
         acc.recovery_account = recovery_account;
      }
   }
   else
   {
      acc.recovery_account = "steem";
   }
}

void account_create_evaluator::do_apply( const account_create_operation& o )
{
   const auto& creator = _db.get_account( o.creator );

   const auto& props = _db.get_dynamic_global_properties();

   const witness_schedule_object& wso = _db.get_witness_schedule_object();

   if( o.fee > asset( STEEM_MAX_ACCOUNT_CREATION_FEE, STEEM_SYMBOL ) ) wlog( "NOTIFYALERT! Account creation fee exceeds max in replay" );
   if( _db.has_hardfork( STEEM_HARDFORK_0_20__2651 ) || _db.is_producing() )
   {
      FC_TODO( "Move to validate() after HF20" );
      FC_ASSERT( o.fee <= asset( STEEM_MAX_ACCOUNT_CREATION_FEE, STEEM_SYMBOL ), "Account creation fee cannot be too large" );
   }

   if( _db.has_hardfork( STEEM_HARDFORK_0_20__1771 ) )
   {
      FC_ASSERT( o.fee == wso.median_props.account_creation_fee, "Must pay the exact account creation fee. paid: ${p} fee: ${f}",
                  ("p", o.fee)
                  ("f", wso.median_props.account_creation_fee) );
   }
   else if( !_db.has_hardfork( STEEM_HARDFORK_0_20__1761 ) && _db.has_hardfork( STEEM_HARDFORK_0_19__987 ) )
   {
      FC_ASSERT( o.fee >= asset( wso.median_props.account_creation_fee.amount * STEEM_CREATE_ACCOUNT_WITH_STEEM_MODIFIER, STEEM_SYMBOL ), "Insufficient Fee: ${f} required, ${p} provided.",
                 ("f", wso.median_props.account_creation_fee * asset( STEEM_CREATE_ACCOUNT_WITH_STEEM_MODIFIER, STEEM_SYMBOL ) )
                 ("p", o.fee) );
   }
   else if( _db.has_hardfork( STEEM_HARDFORK_0_1 ) )
   {
      FC_ASSERT( o.fee >= wso.median_props.account_creation_fee, "Insufficient Fee: ${f} required, ${p} provided.",
                 ("f", wso.median_props.account_creation_fee)
                 ("p", o.fee) );
   }

   FC_TODO( "Check and move to validate post HF20" );
   if( _db.has_hardfork( STEEM_HARDFORK_0_20 ) )
   {
      validate_auth_size( o.owner );
      validate_auth_size( o.active );
      validate_auth_size( o.posting );
   }

   if( _db.has_hardfork( STEEM_HARDFORK_0_15__465 ) )
   {
      verify_authority_accounts_exist( _db, o.owner, o.new_account_name, authority::owner );
      verify_authority_accounts_exist( _db, o.active, o.new_account_name, authority::active );
      verify_authority_accounts_exist( _db, o.posting, o.new_account_name, authority::posting );
   }

   _db.adjust_balance( creator, -o.fee );

   if( _db.has_hardfork( STEEM_HARDFORK_0_20__1762 ) )
   {
      _db.adjust_balance( _db.get< account_object, by_name >( STEEM_NULL_ACCOUNT ), o.fee );
   }

   const auto& new_account = _db.create< account_object >( [&]( account_object& acc )
   {
      initialize_account_object( acc, o.new_account_name, o.memo_key, props, false /*mined*/, o.creator, _db.get_hardfork() );
   });

#ifndef IS_LOW_MEM
   _db.create< account_metadata_object >( [&]( account_metadata_object& meta )
   {
      meta.account = new_account.id;
      from_string( meta.json_metadata, o.json_metadata );
   });
#else
   FC_UNUSED( new_account );
#endif

   _db.create< account_authority_object >( [&]( account_authority_object& auth )
   {
      auth.account = o.new_account_name;
      auth.owner = o.owner;
      auth.active = o.active;
      auth.posting = o.posting;
      auth.last_owner_update = fc::time_point_sec::min();
   });

   if( !_db.has_hardfork( STEEM_HARDFORK_0_20__1762 ) && o.fee.amount > 0 )
   {
      _db.create_vesting( new_account, o.fee );
   }
}

void account_create_with_delegation_evaluator::do_apply( const account_create_with_delegation_operation& o )
{
   FC_ASSERT( !_db.has_hardfork( STEEM_HARDFORK_0_20__1760 ), "Account creation with delegation is deprecated as of Hardfork 20" );

   if( o.fee > asset( STEEM_MAX_ACCOUNT_CREATION_FEE, STEEM_SYMBOL ) ) wlog( "NOTIFYALERT! Account creation fee exceeds max in replay" );
   if( _db.has_hardfork( STEEM_HARDFORK_0_20__2651 ) || _db.is_producing() )
   {
      FC_TODO( "Move to validate() after HF20" );
      FC_ASSERT( o.fee <= asset( STEEM_MAX_ACCOUNT_CREATION_FEE, STEEM_SYMBOL ), "Account creation fee cannot be too large" );
   }

   const auto& creator = _db.get_account( o.creator );
   const auto& props = _db.get_dynamic_global_properties();
   const witness_schedule_object& wso = _db.get_witness_schedule_object();

   FC_ASSERT( creator.balance >= o.fee, "Insufficient balance to create account.",
               ( "creator.balance", creator.balance )
               ( "required", o.fee ) );

   FC_ASSERT( creator.vesting_shares - creator.delegated_vesting_shares - asset( creator.to_withdraw - creator.withdrawn, VESTS_SYMBOL ) >= o.delegation, "Insufficient vesting shares to delegate to new account.",
               ( "creator.vesting_shares", creator.vesting_shares )
               ( "creator.delegated_vesting_shares", creator.delegated_vesting_shares )( "required", o.delegation ) );

   auto target_delegation = asset( wso.median_props.account_creation_fee.amount * STEEM_CREATE_ACCOUNT_WITH_STEEM_MODIFIER * STEEM_CREATE_ACCOUNT_DELEGATION_RATIO, STEEM_SYMBOL ) * props.get_vesting_share_price();

   auto current_delegation = asset( o.fee.amount * STEEM_CREATE_ACCOUNT_DELEGATION_RATIO, STEEM_SYMBOL ) * props.get_vesting_share_price() + o.delegation;

   FC_ASSERT( current_delegation >= target_delegation, "Insufficient Delegation ${f} required, ${p} provided.",
               ("f", target_delegation )
               ( "p", current_delegation )
               ( "account_creation_fee", wso.median_props.account_creation_fee )
               ( "o.fee", o.fee )
               ( "o.delegation", o.delegation ) );

   FC_ASSERT( o.fee >= wso.median_props.account_creation_fee, "Insufficient Fee: ${f} required, ${p} provided.",
               ("f", wso.median_props.account_creation_fee)
               ("p", o.fee) );

   FC_TODO( "Check and move to validate post HF20" );
   if( _db.has_hardfork( STEEM_HARDFORK_0_20 ) )
   {
      validate_auth_size( o.owner );
      validate_auth_size( o.active );
      validate_auth_size( o.posting );
   }

   for( const auto& a : o.owner.account_auths )
   {
      _db.get_account( a.first );
   }

   for( const auto& a : o.active.account_auths )
   {
      _db.get_account( a.first );
   }

   for( const auto& a : o.posting.account_auths )
   {
      _db.get_account( a.first );
   }

   _db.modify( creator, [&]( account_object& c )
   {
      c.balance -= o.fee;
      c.delegated_vesting_shares += o.delegation;
   });

   if( _db.has_hardfork( STEEM_HARDFORK_0_20__1762 ) )
   {
      _db.adjust_balance( _db.get< account_object, by_name >( STEEM_NULL_ACCOUNT ), o.fee );
   }

   const auto& new_account = _db.create< account_object >( [&]( account_object& acc )
   {
      initialize_account_object( acc, o.new_account_name, o.memo_key, props, false /*mined*/, o.creator, _db.get_hardfork() );
      acc.received_vesting_shares = o.delegation;
   });

#ifndef IS_LOW_MEM
   _db.create< account_metadata_object >( [&]( account_metadata_object& meta )
   {
      meta.account = new_account.id;
      from_string( meta.json_metadata, o.json_metadata );
   });
#else
   FC_UNUSED( new_account );
#endif

   _db.create< account_authority_object >( [&]( account_authority_object& auth )
   {
      auth.account = o.new_account_name;
      auth.owner = o.owner;
      auth.active = o.active;
      auth.posting = o.posting;
      auth.last_owner_update = fc::time_point_sec::min();
   });

   if( o.delegation.amount > 0 || !_db.has_hardfork( STEEM_HARDFORK_0_19__997 ) )
   {
      _db.create< vesting_delegation_object >( [&]( vesting_delegation_object& vdo )
      {
         vdo.delegator = o.creator;
         vdo.delegatee = o.new_account_name;
         vdo.vesting_shares = o.delegation;
         vdo.min_delegation_time = _db.head_block_time() + STEEM_CREATE_ACCOUNT_DELEGATION_TIME;
      });
   }

   if( !_db.has_hardfork( STEEM_HARDFORK_0_20__1762 ) && o.fee.amount > 0 )
   {
      _db.create_vesting( new_account, o.fee );
   }
}


void account_update_evaluator::do_apply( const account_update_operation& o )
{
   if( _db.has_hardfork( STEEM_HARDFORK_0_1 ) ) FC_ASSERT( o.account != STEEM_TEMP_ACCOUNT, "Cannot update temp account." );

   if( ( _db.has_hardfork( STEEM_HARDFORK_0_15__465 ) ) && o.posting )
      o.posting->validate();

   const auto& account = _db.get_account( o.account );
   const auto& account_auth = _db.get< account_authority_object, by_account >( o.account );

   if( _db.is_producing() || _db.has_hardfork( STEEM_HARDFORK_0_20 ) )
   {
      if( o.owner )
         validate_auth_size( *o.owner );
      if( o.active )
         validate_auth_size( *o.active );
      if( o.posting )
         validate_auth_size( *o.posting );
   }

   if( o.owner )
   {
#ifndef IS_TEST_NET
      if( _db.has_hardfork( STEEM_HARDFORK_0_11 ) )
         FC_ASSERT( _db.head_block_time() - account_auth.last_owner_update > STEEM_OWNER_UPDATE_LIMIT, "Owner authority can only be updated once an hour." );
#endif

      if( ( _db.has_hardfork( STEEM_HARDFORK_0_15__465 ) ) )
         verify_authority_accounts_exist( _db, *o.owner, o.account, authority::owner );

      _db.update_owner_authority( account, *o.owner );
   }
   if( o.active && ( _db.has_hardfork( STEEM_HARDFORK_0_15__465 ) ) )
      verify_authority_accounts_exist( _db, *o.active, o.account, authority::active );
   if( o.posting && ( _db.has_hardfork( STEEM_HARDFORK_0_15__465 ) ) )
      verify_authority_accounts_exist( _db, *o.posting, o.account, authority::posting );

   _db.modify( account, [&]( account_object& acc )
   {
      if( o.memo_key != public_key_type() )
            acc.memo_key = o.memo_key;

      acc.last_account_update = _db.head_block_time();
   });

   #ifndef IS_LOW_MEM
   if( o.json_metadata.size() > 0 )
   {
      _db.modify( _db.get< account_metadata_object, by_account >( account.id ), [&]( account_metadata_object& meta )
      {
         from_string( meta.json_metadata, o.json_metadata );
         if ( !_db.has_hardfork( STEEM_HARDFORK_0_21__3274 ) )
         {
            from_string( meta.posting_json_metadata, o.json_metadata );
         }
      });
   }
   #endif

   if( o.active || o.posting )
   {
      _db.modify( account_auth, [&]( account_authority_object& auth)
      {
         if( o.active )  auth.active  = *o.active;
         if( o.posting ) auth.posting = *o.posting;
      });
   }

}

void account_update2_evaluator::do_apply( const account_update2_operation& o )
{
   FC_ASSERT( _db.has_hardfork( STEEM_HARDFORK_0_21__3274 ), "Operation 'account_update2' is not enabled until HF 21" );
   FC_ASSERT( o.account != STEEM_TEMP_ACCOUNT, "Cannot update temp account." );

   if( o.posting )
      o.posting->validate();

   const auto& account = _db.get_account( o.account );
   const auto& account_auth = _db.get< account_authority_object, by_account >( o.account );

   if( o.owner )
      validate_auth_size( *o.owner );
   if( o.active )
      validate_auth_size( *o.active );
   if( o.posting )
      validate_auth_size( *o.posting );

   if( o.owner )
   {
#ifndef IS_TEST_NET
      FC_ASSERT( _db.head_block_time() - account_auth.last_owner_update > STEEM_OWNER_UPDATE_LIMIT, "Owner authority can only be updated once an hour." );
#endif

      verify_authority_accounts_exist( _db, *o.owner, o.account, authority::owner );

      _db.update_owner_authority( account, *o.owner );
   }
   if( o.active )
      verify_authority_accounts_exist( _db, *o.active, o.account, authority::active );
   if( o.posting )
      verify_authority_accounts_exist( _db, *o.posting, o.account, authority::posting );

   _db.modify( account, [&]( account_object& acc )
   {
      if( o.memo_key && *o.memo_key != public_key_type() )
            acc.memo_key = *o.memo_key;

      acc.last_account_update = _db.head_block_time();
   });

   #ifndef IS_LOW_MEM
   if( o.json_metadata.size() > 0 || o.posting_json_metadata.size() > 0 )
   {
      _db.modify( _db.get< account_metadata_object, by_account >( account.id ), [&]( account_metadata_object& meta )
      {
         if ( o.json_metadata.size() > 0 )
            from_string( meta.json_metadata, o.json_metadata );

         if ( o.posting_json_metadata.size() > 0 )
            from_string( meta.posting_json_metadata, o.posting_json_metadata );
      });
   }
   #endif

   if( o.active || o.posting )
   {
      _db.modify( account_auth, [&]( account_authority_object& auth)
      {
         if( o.active )  auth.active  = *o.active;
         if( o.posting ) auth.posting = *o.posting;
      });
   }

}

/**
 *  Because net_rshares is 0 there is no need to update any pending payout calculations or parent posts.
 */
void delete_comment_evaluator::do_apply( const delete_comment_operation& o )
{
   const auto& comment = _db.get_comment( o.author, o.permlink );
   FC_ASSERT( comment.children == 0, "Cannot delete a comment with replies." );

   if( _db.has_hardfork( STEEM_HARDFORK_0_19__876 ) )
      FC_ASSERT( comment.cashout_time != fc::time_point_sec::maximum() );

   if( _db.has_hardfork( STEEM_HARDFORK_0_19__977 ) )
      FC_ASSERT( comment.net_rshares <= 0, "Cannot delete a comment with net positive votes." );

   if( comment.net_rshares > 0 ) return;

   const auto& vote_idx = _db.get_index< comment_vote_index, by_comment_voter_symbol >();

   auto vote_itr = vote_idx.lower_bound( comment_id_type(comment.id) );
   while( vote_itr != vote_idx.end() && vote_itr->comment == comment.id ) {
      const auto& cur_vote = *vote_itr;
      ++vote_itr;
      _db.remove(cur_vote);
   }

   /// this loop can be skiped for validate-only nodes as it is merely gathering stats for indicies
   if( _db.has_hardfork( STEEM_HARDFORK_0_6__80 ) && comment.parent_author != STEEM_ROOT_POST_PARENT )
   {
      auto parent = &_db.get_comment( comment.parent_author, comment.parent_permlink );
      auto now = _db.head_block_time();
      while( parent )
      {
         _db.modify( *parent, [&]( comment_object& p ){
            p.children--;
            p.active = now;
         });
   #ifndef IS_LOW_MEM
         if( parent->parent_author != STEEM_ROOT_POST_PARENT )
            parent = &_db.get_comment( parent->parent_author, parent->parent_permlink );
         else
   #endif
            parent = nullptr;
      }
   }

   _db.remove( comment );
}

struct comment_options_extension_visitor
{
   comment_options_extension_visitor( const comment_object& c, database& db ) : _c( c ), _db( db ) {}

   typedef void result_type;

   const comment_object& _c;
   database& _db;

   void operator()( const allowed_vote_assets& va) const
   {
      FC_ASSERT( _db.has_hardfork( STEEM_SMT_HARDFORK ), "Specifying SMT Comment Options not available until SMT Hardfork." );
      FC_ASSERT( _c.abs_rshares == 0, "Comment must not have been voted on before specifying allowed vote assets." );

      const auto& root = _db.get< comment_object, by_id >( _c.root_comment );

      for( const auto& a : va.votable_assets )
      {
         if( _c.parent_author == STEEM_ROOT_POST_PARENT )
         {
            auto smt = _db.find< smt_token_object, by_symbol >( a.first );
            FC_ASSERT( smt != nullptr, "SMT ${s} was not found.", ("s", a.first) );
            FC_ASSERT( smt->phase == smt_phase::launch_success, "SMT ${s} must be in active phase to be a votable asset.", ("s", a.first) );
            FC_ASSERT( smt->allow_voting, "SMT ${s} does not support voting", ("s", a.first) );
         }
         else
         {
            auto smt = root.allowed_vote_assets.find( a.first );
            FC_ASSERT( smt != root.allowed_vote_assets.end(), "SMT ${s} is not a votable asset in the root comment ${a}/${p}",
               ("s", a.first)("a", root.author)("p", root.permlink) );
         }
      }

      for( const auto& a : va.votable_assets )
      {
         if( a.second.beneficiaries.beneficiaries.size() )
         {
            for( auto& b : a.second.beneficiaries.beneficiaries )
            {
               auto acc = _db.find< account_object, by_name >( b.account );
               FC_ASSERT( acc != nullptr, "Beneficiary \"${a}\" must exist.", ("a", b.account) );
            }

            auto beneficiaries = _db.find< comment_smt_beneficiaries_object, by_comment_symbol >( boost::make_tuple( _c.id, a.first ) );

            if( beneficiaries == nullptr )
            {
               _db.create< comment_smt_beneficiaries_object >( [&]( comment_smt_beneficiaries_object& b )
               {
                  b.comment = _c.id;
                  b.smt = a.first;
                  b.beneficiaries.insert(
                     b.beneficiaries.end(),
                     a.second.beneficiaries.beneficiaries.begin(),
                     a.second.beneficiaries.beneficiaries.end() );
               });
            }
            else
            {
               _db.modify( *beneficiaries, [&]( comment_smt_beneficiaries_object& b )
               {
                  b.beneficiaries.clear();
                  b.beneficiaries.insert(
                     b.beneficiaries.end(),
                     a.second.beneficiaries.beneficiaries.begin(),
                     a.second.beneficiaries.beneficiaries.end() );
               });
            }
         }
         else
         {
            auto beneficiaries = _db.find< comment_smt_beneficiaries_object, by_comment_symbol >( boost::make_tuple( _c.id, a.first ) );

            if( beneficiaries != nullptr )
            {
               _db.remove( *beneficiaries );
            }
         }
      }

      _db.modify( _c, [&]( comment_object& c )
      {
         for( const auto& a : va.votable_assets )
         {
            auto& vote_assets = c.allowed_vote_assets[ a.first ];
            vote_assets.max_accepted_payout = a.second.max_accepted_payout;
            vote_assets.allow_curation_rewards = a.second.allow_curation_rewards;
         }
      });
   }

   void operator()( const comment_payout_beneficiaries& cpb ) const
   {
      FC_ASSERT( _c.beneficiaries.size() == 0, "Comment already has beneficiaries specified." );
      FC_ASSERT( _c.abs_rshares == 0, "Comment must not have been voted on before specifying beneficiaries." );

      for( auto& b : cpb.beneficiaries )
      {
         auto acc = _db.find< account_object, by_name >( b.account );
         FC_ASSERT( acc != nullptr, "Beneficiary \"${a}\" must exist.", ("a", b.account) );
      }

      _db.modify( _c, [&]( comment_object& c )
      {
         c.beneficiaries.insert( c.beneficiaries.end(), cpb.beneficiaries.begin(), cpb.beneficiaries.end() );
      });
   }
};

void comment_options_evaluator::do_apply( const comment_options_operation& o )
{
   const auto& comment = _db.get_comment( o.author, o.permlink );
   if( !o.allow_curation_rewards || !o.allow_votes || o.max_accepted_payout < comment.max_accepted_payout )
      FC_ASSERT( comment.abs_rshares == 0, "One of the included comment options requires the comment to have no rshares allocated to it." );

   FC_ASSERT( comment.allow_curation_rewards >= o.allow_curation_rewards, "Curation rewards cannot be re-enabled." );
   FC_ASSERT( comment.allow_votes >= o.allow_votes, "Voting cannot be re-enabled." );
   FC_ASSERT( comment.max_accepted_payout >= o.max_accepted_payout, "A comment cannot accept a greater payout." );
   FC_ASSERT( comment.percent_steem_dollars >= o.percent_steem_dollars, "A comment cannot accept a greater percent SBD." );

   _db.modify( comment, [&]( comment_object& c ) {
       c.max_accepted_payout   = o.max_accepted_payout;
       c.percent_steem_dollars = o.percent_steem_dollars;
       c.allow_votes           = o.allow_votes;
       c.allow_curation_rewards = o.allow_curation_rewards;
   });

   for( auto& e : o.extensions )
   {
      e.visit( comment_options_extension_visitor( comment, _db ) );
   }
}

void comment_evaluator::do_apply( const comment_operation& o )
{ try {
   if( _db.has_hardfork( STEEM_HARDFORK_0_5__55 ) )
      FC_ASSERT( o.title.size() + o.body.size() + o.json_metadata.size(), "Cannot update comment because nothing appears to be changing." );

   const auto& by_permlink_idx = _db.get_index< comment_index >().indices().get< by_permlink >();
   auto itr = by_permlink_idx.find( boost::make_tuple( o.author, o.permlink ) );

   const auto& auth = _db.get_account( o.author ); /// prove it exists

   comment_id_type id;

   const comment_object* parent = nullptr;
   if( o.parent_author != STEEM_ROOT_POST_PARENT )
   {
      parent = &_db.get_comment( o.parent_author, o.parent_permlink );
      if( !_db.has_hardfork( STEEM_HARDFORK_0_17__767 ) )
         FC_ASSERT( parent->depth < STEEM_MAX_COMMENT_DEPTH_PRE_HF17, "Comment is nested ${x} posts deep, maximum depth is ${y}.", ("x",parent->depth)("y",STEEM_MAX_COMMENT_DEPTH_PRE_HF17) );
      else
         FC_ASSERT( parent->depth < STEEM_MAX_COMMENT_DEPTH, "Comment is nested ${x} posts deep, maximum depth is ${y}.", ("x",parent->depth)("y",STEEM_MAX_COMMENT_DEPTH) );
   }

   FC_ASSERT( fc::is_utf8( o.json_metadata ), "JSON Metadata must be UTF-8" );

   auto now = _db.head_block_time();

   if ( itr == by_permlink_idx.end() )
   {
      if( o.parent_author != STEEM_ROOT_POST_PARENT )
      {
         FC_ASSERT( _db.get( parent->root_comment ).allow_replies, "The parent comment has disabled replies." );
         if( _db.has_hardfork( STEEM_HARDFORK_0_12__177 ) && !_db.has_hardfork( STEEM_HARDFORK_0_17__869 ) )
            FC_ASSERT( _db.calculate_discussion_payout_time( *parent ) != fc::time_point_sec::maximum(), "Discussion is frozen." );
      }

      if( _db.has_hardfork( STEEM_HARDFORK_0_20__2019 ) )
      {
         if( o.parent_author == STEEM_ROOT_POST_PARENT )
             FC_ASSERT( ( now - auth.last_root_post ) > STEEM_MIN_ROOT_COMMENT_INTERVAL, "You may only post once every 5 minutes.", ("now",now)("last_root_post", auth.last_root_post) );
         else
             FC_ASSERT( (now - auth.last_post) >= STEEM_MIN_REPLY_INTERVAL_HF20, "You may only comment once every 3 seconds.", ("now",now)("auth.last_post",auth.last_post) );
      }

      uint16_t reward_weight = STEEM_100_PERCENT;
      uint64_t post_bandwidth = auth.post_bandwidth;

      if( _db.has_hardfork( STEEM_HARDFORK_0_12__176 ) && !_db.has_hardfork( STEEM_HARDFORK_0_17__733 ) && o.parent_author == STEEM_ROOT_POST_PARENT )
      {
         uint64_t post_delta_time = std::min( _db.head_block_time().sec_since_epoch() - auth.last_root_post.sec_since_epoch(), STEEM_POST_AVERAGE_WINDOW );
         uint32_t old_weight = uint32_t( ( post_bandwidth * ( STEEM_POST_AVERAGE_WINDOW - post_delta_time ) ) / STEEM_POST_AVERAGE_WINDOW );
         post_bandwidth = ( old_weight + STEEM_100_PERCENT );
         reward_weight = uint16_t( std::min( ( STEEM_POST_WEIGHT_CONSTANT * STEEM_100_PERCENT ) / ( post_bandwidth * post_bandwidth ), uint64_t( STEEM_100_PERCENT ) ) );
      }

      _db.modify( auth, [&]( account_object& a ) {
         if( o.parent_author == STEEM_ROOT_POST_PARENT )
         {
            a.last_root_post = now;
            a.post_bandwidth = uint32_t( post_bandwidth );
         }
         a.last_post = now;
         a.last_post_edit = now;
         a.post_count++;
      });

      const auto& new_comment = _db.create< comment_object >( [&]( comment_object& com )
      {
         if( _db.has_hardfork( STEEM_HARDFORK_0_1 ) )
         {
            validate_permlink_0_1( o.parent_permlink );
            validate_permlink_0_1( o.permlink );
         }

         com.author = o.author;
         from_string( com.permlink, o.permlink );
         com.last_update = _db.head_block_time();
         com.created = com.last_update;
         com.active = com.last_update;
         com.last_payout = fc::time_point_sec::min();
         com.max_cashout_time = fc::time_point_sec::maximum();
         com.reward_weight = reward_weight;

         if ( o.parent_author == STEEM_ROOT_POST_PARENT )
         {
            com.parent_author = "";
            from_string( com.parent_permlink, o.parent_permlink );
            from_string( com.category, o.parent_permlink );
            com.root_comment = com.id;
            com.cashout_time = _db.has_hardfork( STEEM_HARDFORK_0_12__177 ) ?
               _db.head_block_time() + STEEM_CASHOUT_WINDOW_SECONDS_PRE_HF17 :
               fc::time_point_sec::maximum();
         }
         else
         {
            com.parent_author = parent->author;
            com.parent_permlink = parent->permlink;
            com.depth = parent->depth + 1;
            com.category = parent->category;
            com.root_comment = parent->root_comment;
            com.cashout_time = fc::time_point_sec::maximum();
         }

         if( _db.has_hardfork( STEEM_HARDFORK_0_17__769 ) )
         {
            com.cashout_time = com.created + STEEM_CASHOUT_WINDOW_SECONDS;
         }
      });

      id = new_comment.id;

   #ifndef IS_LOW_MEM
      _db.create< comment_content_object >( [&]( comment_content_object& con )
      {
         con.comment = id;

         from_string( con.title, o.title );
         if( o.body.size() < 1024*1024*128 )
         {
            from_string( con.body, o.body );
         }
         from_string( con.json_metadata, o.json_metadata );
      });
   #endif


/// this loop can be skiped for validate-only nodes as it is merely gathering stats for indicies
      auto now = _db.head_block_time();
      while( parent ) {
         _db.modify( *parent, [&]( comment_object& p ){
            p.children++;
            p.active = now;
         });
#ifndef IS_LOW_MEM
         if( parent->parent_author != STEEM_ROOT_POST_PARENT )
            parent = &_db.get_comment( parent->parent_author, parent->parent_permlink );
         else
#endif
            parent = nullptr;
      }

   }
   else // start edit case
   {
      const auto& comment = *itr;

      if( _db.is_producing() || _db.has_hardfork( STEEM_HARDFORK_0_21__3313 ) )
      {
         FC_ASSERT( now - auth.last_post_edit >= STEEM_MIN_COMMENT_EDIT_INTERVAL, "Can only perform one comment edit per block." );
      }

      if( !_db.has_hardfork( STEEM_HARDFORK_0_17__772 ) )
      {
         if( _db.has_hardfork( STEEM_HARDFORK_0_14__306 ) )
            FC_ASSERT( _db.calculate_discussion_payout_time( comment ) != fc::time_point_sec::maximum(), "The comment is archived." );
         else if( _db.has_hardfork( STEEM_HARDFORK_0_10 ) )
            FC_ASSERT( comment.last_payout == fc::time_point_sec::min(), "Can only edit during the first 24 hours." );
      }
      _db.modify( comment, [&]( comment_object& com )
      {
         com.last_update   = _db.head_block_time();
         com.active        = com.last_update;
         std::function< bool( const shared_string& a, const string& b ) > equal;
         if ( _db.has_hardfork( STEEM_HARDFORK_0_21__2203 ) )
            equal = []( const shared_string& a, const string& b ) -> bool { return a.size() == b.size() && std::strcmp( a.c_str(), b.c_str() ) == 0; };
         else
            equal = []( const shared_string& a, const string& b ) -> bool { return a.size() == b.size() || std::strcmp( a.c_str(), b.c_str() ) == 0; };

         if( !parent )
         {
            FC_ASSERT( com.parent_author == account_name_type(), "The parent of a comment cannot change." );
            FC_ASSERT( equal( com.parent_permlink, o.parent_permlink ), "The permlink of a comment cannot change." );
         }
         else
         {
            FC_ASSERT( com.parent_author == o.parent_author, "The parent of a comment cannot change." );
            FC_ASSERT( equal( com.parent_permlink, o.parent_permlink ), "The permlink of a comment cannot change." );
         }
      });

      _db.modify( auth, [&]( account_object& a )
      {
         a.last_post_edit = now;
      });
   #ifndef IS_LOW_MEM
      _db.modify( _db.get< comment_content_object, by_comment >( comment.id ), [&]( comment_content_object& con )
      {
         if( o.title.size() )         from_string( con.title, o.title );
         if( o.json_metadata.size() )
            from_string( con.json_metadata, o.json_metadata );

         if( o.body.size() ) {
            try {
            diff_match_patch<std::wstring> dmp;
            auto patch = dmp.patch_fromText( utf8_to_wstring(o.body) );
            if( patch.size() ) {
               auto result = dmp.patch_apply( patch, utf8_to_wstring( to_string( con.body ) ) );
               auto patched_body = wstring_to_utf8(result.first);
               if( !fc::is_utf8( patched_body ) ) {
                  idump(("invalid utf8")(patched_body));
                  from_string( con.body, fc::prune_invalid_utf8(patched_body) );
               } else { from_string( con.body, patched_body ); }
            }
            else { // replace
               from_string( con.body, o.body );
            }
            } catch ( ... ) {
               from_string( con.body, o.body );
            }
         }
      });
   #endif



   } // end EDIT case

} FC_CAPTURE_AND_RETHROW( (o) ) }

void escrow_transfer_evaluator::do_apply( const escrow_transfer_operation& o )
{
   try
   {
      const auto& from_account = _db.get_account(o.from);
      _db.get_account(o.to);
      _db.get_account(o.agent);

      FC_ASSERT( o.ratification_deadline > _db.head_block_time(), "The escorw ratification deadline must be after head block time." );
      FC_ASSERT( o.escrow_expiration > _db.head_block_time(), "The escrow expiration must be after head block time." );

      asset steem_spent = o.steem_amount;
      asset sbd_spent = o.sbd_amount;
      if( o.fee.symbol == STEEM_SYMBOL )
         steem_spent += o.fee;
      else
         sbd_spent += o.fee;

      _db.adjust_balance( from_account, -steem_spent );
      _db.adjust_balance( from_account, -sbd_spent );

      _db.create<escrow_object>([&]( escrow_object& esc )
      {
         esc.escrow_id              = o.escrow_id;
         esc.from                   = o.from;
         esc.to                     = o.to;
         esc.agent                  = o.agent;
         esc.ratification_deadline  = o.ratification_deadline;
         esc.escrow_expiration      = o.escrow_expiration;
         esc.sbd_balance            = o.sbd_amount;
         esc.steem_balance          = o.steem_amount;
         esc.pending_fee            = o.fee;
      });
   }
   FC_CAPTURE_AND_RETHROW( (o) )
}

void escrow_approve_evaluator::do_apply( const escrow_approve_operation& o )
{
   try
   {

      const auto& escrow = _db.get_escrow( o.from, o.escrow_id );

      FC_ASSERT( escrow.to == o.to, "Operation 'to' (${o}) does not match escrow 'to' (${e}).", ("o", o.to)("e", escrow.to) );
      FC_ASSERT( escrow.agent == o.agent, "Operation 'agent' (${a}) does not match escrow 'agent' (${e}).", ("o", o.agent)("e", escrow.agent) );
      FC_ASSERT( escrow.ratification_deadline >= _db.head_block_time(), "The escrow ratification deadline has passed. Escrow can no longer be ratified." );

      bool reject_escrow = !o.approve;

      if( o.who == o.to )
      {
         FC_ASSERT( !escrow.to_approved, "Account 'to' (${t}) has already approved the escrow.", ("t", o.to) );

         if( !reject_escrow )
         {
            _db.modify( escrow, [&]( escrow_object& esc )
            {
               esc.to_approved = true;
            });
         }
      }
      if( o.who == o.agent )
      {
         FC_ASSERT( !escrow.agent_approved, "Account 'agent' (${a}) has already approved the escrow.", ("a", o.agent) );

         if( !reject_escrow )
         {
            _db.modify( escrow, [&]( escrow_object& esc )
            {
               esc.agent_approved = true;
            });
         }
      }

      if( reject_escrow )
      {
         _db.adjust_balance( o.from, escrow.steem_balance );
         _db.adjust_balance( o.from, escrow.sbd_balance );
         _db.adjust_balance( o.from, escrow.pending_fee );

         _db.remove( escrow );
      }
      else if( escrow.to_approved && escrow.agent_approved )
      {
         _db.adjust_balance( o.agent, escrow.pending_fee );

         _db.modify( escrow, [&]( escrow_object& esc )
         {
            esc.pending_fee.amount = 0;
         });
      }
   }
   FC_CAPTURE_AND_RETHROW( (o) )
}

void escrow_dispute_evaluator::do_apply( const escrow_dispute_operation& o )
{
   try
   {
      _db.get_account( o.from ); // Verify from account exists

      const auto& e = _db.get_escrow( o.from, o.escrow_id );
      FC_ASSERT( _db.head_block_time() < e.escrow_expiration, "Disputing the escrow must happen before expiration." );
      FC_ASSERT( e.to_approved && e.agent_approved, "The escrow must be approved by all parties before a dispute can be raised." );
      FC_ASSERT( !e.disputed, "The escrow is already under dispute." );
      FC_ASSERT( e.to == o.to, "Operation 'to' (${o}) does not match escrow 'to' (${e}).", ("o", o.to)("e", e.to) );
      FC_ASSERT( e.agent == o.agent, "Operation 'agent' (${a}) does not match escrow 'agent' (${e}).", ("o", o.agent)("e", e.agent) );

      _db.modify( e, [&]( escrow_object& esc )
      {
         esc.disputed = true;
      });
   }
   FC_CAPTURE_AND_RETHROW( (o) )
}

void escrow_release_evaluator::do_apply( const escrow_release_operation& o )
{
   try
   {
      _db.get_account(o.from); // Verify from account exists

      const auto& e = _db.get_escrow( o.from, o.escrow_id );
      FC_ASSERT( e.steem_balance >= o.steem_amount, "Release amount exceeds escrow balance. Amount: ${a}, Balance: ${b}", ("a", o.steem_amount)("b", e.steem_balance) );
      FC_ASSERT( e.sbd_balance >= o.sbd_amount, "Release amount exceeds escrow balance. Amount: ${a}, Balance: ${b}", ("a", o.sbd_amount)("b", e.sbd_balance) );
      FC_ASSERT( e.to == o.to, "Operation 'to' (${o}) does not match escrow 'to' (${e}).", ("o", o.to)("e", e.to) );
      FC_ASSERT( e.agent == o.agent, "Operation 'agent' (${a}) does not match escrow 'agent' (${e}).", ("o", o.agent)("e", e.agent) );
      FC_ASSERT( o.receiver == e.from || o.receiver == e.to, "Funds must be released to 'from' (${f}) or 'to' (${t})", ("f", e.from)("t", e.to) );
      FC_ASSERT( e.to_approved && e.agent_approved, "Funds cannot be released prior to escrow approval." );

      // If there is a dispute regardless of expiration, the agent can release funds to either party
      if( e.disputed )
      {
         FC_ASSERT( o.who == e.agent, "Only 'agent' (${a}) can release funds in a disputed escrow.", ("a", e.agent) );
      }
      else
      {
         FC_ASSERT( o.who == e.from || o.who == e.to, "Only 'from' (${f}) and 'to' (${t}) can release funds from a non-disputed escrow", ("f", e.from)("t", e.to) );

         if( e.escrow_expiration > _db.head_block_time() )
         {
            // If there is no dispute and escrow has not expired, either party can release funds to the other.
            if( o.who == e.from )
            {
               FC_ASSERT( o.receiver == e.to, "Only 'from' (${f}) can release funds to 'to' (${t}).", ("f", e.from)("t", e.to) );
            }
            else if( o.who == e.to )
            {
               FC_ASSERT( o.receiver == e.from, "Only 'to' (${t}) can release funds to 'from' (${t}).", ("f", e.from)("t", e.to) );
            }
         }
      }
      // If escrow expires and there is no dispute, either party can release funds to either party.

      _db.adjust_balance( o.receiver, o.steem_amount );
      _db.adjust_balance( o.receiver, o.sbd_amount );

      _db.modify( e, [&]( escrow_object& esc )
      {
         esc.steem_balance -= o.steem_amount;
         esc.sbd_balance -= o.sbd_amount;
      });

      if( e.steem_balance.amount == 0 && e.sbd_balance.amount == 0 )
      {
         _db.remove( e );
      }
   }
   FC_CAPTURE_AND_RETHROW( (o) )
}

void transfer_evaluator::do_apply( const transfer_operation& o )
{
   FC_TODO( "Remove is producing after HF 21" );
   if( o.to == STEEM_TREASURY_ACCOUNT && o.amount.symbol != SBD_SYMBOL ) wlog( "NOTIFYALERT! Non-SBD transfer to Treasury in replay" );
   if( _db.has_hardfork( STEEM_HARDFORK_0_21__3343 ) )
   {
      FC_ASSERT( o.amount.symbol == SBD_SYMBOL || o.to != STEEM_TREASURY_ACCOUNT,
         "Can only transfer SBD to ${s}", ("s", STEEM_TREASURY_ACCOUNT) );
   }

   _db.adjust_balance( o.from, -o.amount );
   _db.adjust_balance( o.to, o.amount );
}

void transfer_to_vesting_evaluator::do_apply( const transfer_to_vesting_operation& o )
{
   const auto& from_account = _db.get_account(o.from);
   const auto& to_account = o.to.size() ? _db.get_account(o.to) : from_account;

   FC_TODO( "Remove is producing after HF 21" );
   if( o.to == STEEM_TREASURY_ACCOUNT && o.amount.symbol != SBD_SYMBOL ) wlog( "NOTIFYALERT! Non-SBD transfer to Treasury in replay" );
   if( _db.has_hardfork( STEEM_HARDFORK_0_21__3343 ) )
   {
      FC_ASSERT( o.amount.symbol == SBD_SYMBOL || o.to != STEEM_TREASURY_ACCOUNT,
         "Can only transfer SBD to ${s}", ("s", STEEM_TREASURY_ACCOUNT) );
   }

   _db.adjust_balance( from_account, -o.amount );
   _db.create_vesting( to_account, o.amount );
}

void withdraw_vesting_evaluator::do_apply( const withdraw_vesting_operation& o )
{
   const auto& account = _db.get_account( o.account );

   if( o.vesting_shares.amount < 0 )
   {
      if( _db.has_hardfork( STEEM_HARDFORK_0_20 ) )
      {
         FC_ASSERT( false, "Cannot withdraw negative VESTS. account: ${account}, vests:${vests}",
            ("account", o.account)("vests", o.vesting_shares) );
      }

      return;
   }

   if ( o.vesting_shares.symbol.space() == asset_symbol_type::smt_nai_space )
   {
      FC_ASSERT( _db.has_hardfork( STEEM_SMT_HARDFORK ), "Withdraw vesting operation for SMTs is not enabled until the SMT hardfork." );
      FC_ASSERT( o.vesting_shares.symbol.is_vesting(), "Vesting shares must be a vesting symbol." );
      auto liquid_symbol = o.vesting_shares.symbol.get_paired_symbol();
      const auto* bal_obj = _db.find< account_regular_balance_object, by_name_liquid_symbol >( boost::make_tuple( account.name, liquid_symbol ) );

      FC_ASSERT( bal_obj != nullptr, "Account does not have sufficient Token Power for withdrawal." );
      FC_ASSERT( bal_obj->vesting_shares - bal_obj->delegated_vesting_shares >= o.vesting_shares, "Account does not have sufficient Token Power for withdrawal." );

      if ( o.vesting_shares.amount == 0 )
      {
         FC_ASSERT( bal_obj->vesting_withdraw_rate.amount != 0, "This operation would not change the vesting withdraw rate." );

         _db.modify( *bal_obj, [&]( account_regular_balance_object& a )
         {
            a.vesting_withdraw_rate = asset( 0, liquid_symbol.get_paired_symbol() );
            a.next_vesting_withdrawal = time_point_sec::maximum();
            a.to_withdraw = 0;
            a.withdrawn = 0;
         } );
      }
      else
      {
         int vesting_withdraw_intervals = STEEM_VESTING_WITHDRAW_INTERVALS;

         auto new_vesting_withdraw_rate = asset( o.vesting_shares.amount / vesting_withdraw_intervals, liquid_symbol.get_paired_symbol() );

         if ( new_vesting_withdraw_rate.amount == 0 )
            new_vesting_withdraw_rate.amount = 1;

         if ( new_vesting_withdraw_rate.amount * vesting_withdraw_intervals < o.vesting_shares.amount )
            new_vesting_withdraw_rate.amount += 1;

         FC_ASSERT( bal_obj->vesting_withdraw_rate != new_vesting_withdraw_rate, "This operation would not change the vesting withdraw rate." );

         _db.modify( *bal_obj, [&]( account_regular_balance_object& a )
         {
            a.vesting_withdraw_rate = new_vesting_withdraw_rate;
            a.next_vesting_withdrawal = _db.head_block_time() + fc::seconds( SMT_VESTING_WITHDRAW_INTERVAL_SECONDS );
            a.to_withdraw = o.vesting_shares.amount;
            a.withdrawn = 0;
         } );
      }
   }
   else
   {
      FC_ASSERT( o.vesting_shares.symbol == VESTS_SYMBOL, "Vesting shares must be a vesting symbol." );
      FC_ASSERT( account.vesting_shares >= asset( 0, VESTS_SYMBOL ), "Account does not have sufficient Steem Power for withdraw." );
      FC_ASSERT( account.vesting_shares - account.delegated_vesting_shares >= o.vesting_shares, "Account does not have sufficient Steem Power for withdraw." );

      if( o.vesting_shares.amount == 0 )
      {
         if( _db.has_hardfork( STEEM_HARDFORK_0_5__57 ) )
            FC_ASSERT( account.vesting_withdraw_rate.amount  != 0, "This operation would not change the vesting withdraw rate." );

         _db.modify( account, [&]( account_object& a ) {
            a.vesting_withdraw_rate = asset( 0, VESTS_SYMBOL );
            a.next_vesting_withdrawal = time_point_sec::maximum();
            a.to_withdraw = 0;
            a.withdrawn = 0;
         });
      }
      else
      {
         int vesting_withdraw_intervals = STEEM_VESTING_WITHDRAW_INTERVALS_PRE_HF_16;
         if( _db.has_hardfork( STEEM_HARDFORK_0_16__551 ) )
            vesting_withdraw_intervals = STEEM_VESTING_WITHDRAW_INTERVALS; /// 13 weeks = 1 quarter of a year

         auto new_vesting_withdraw_rate = asset( o.vesting_shares.amount / vesting_withdraw_intervals, VESTS_SYMBOL );

         if( new_vesting_withdraw_rate.amount == 0 )
            new_vesting_withdraw_rate.amount = 1;

         if( _db.has_hardfork( STEEM_HARDFORK_0_21 ) && new_vesting_withdraw_rate.amount * vesting_withdraw_intervals < o.vesting_shares.amount )
            new_vesting_withdraw_rate.amount += 1;

         if( _db.has_hardfork( STEEM_HARDFORK_0_5__57 ) )
            FC_ASSERT( account.vesting_withdraw_rate  != new_vesting_withdraw_rate, "This operation would not change the vesting withdraw rate." );

         _db.modify( account, [&]( account_object& a )
         {
            a.vesting_withdraw_rate = new_vesting_withdraw_rate;
            a.next_vesting_withdrawal = _db.head_block_time() + fc::seconds(STEEM_VESTING_WITHDRAW_INTERVAL_SECONDS);
            a.to_withdraw = o.vesting_shares.amount;
            a.withdrawn = 0;
         });
      }
   }
}

void set_withdraw_vesting_route_evaluator::do_apply( const set_withdraw_vesting_route_operation& o )
{
   try
   {
   const auto& from_account = _db.get_account( o.from_account );
   const auto& to_account = _db.get_account( o.to_account );
   const auto& wd_idx = _db.get_index< withdraw_vesting_route_index >().indices().get< by_withdraw_route >();
   auto itr = wd_idx.find( boost::make_tuple( from_account.name, to_account.name ) );

   FC_TODO( "Remove is producing after HF 21" );
   if( o.to_account == STEEM_TREASURY_ACCOUNT ) wlog( "NOTIFYALERT! Vesting withdrawal from treasury in replay" );
   if( _db.has_hardfork( STEEM_HARDFORK_0_21__3343 ) )
   {
      FC_ASSERT( o.to_account != STEEM_TREASURY_ACCOUNT,
         "Cannot withdraw vesting to ${s}", ("s", STEEM_TREASURY_ACCOUNT) );
   }

   if( itr == wd_idx.end() )
   {
      FC_ASSERT( o.percent != 0, "Cannot create a 0% destination." );
      FC_ASSERT( from_account.withdraw_routes < STEEM_MAX_WITHDRAW_ROUTES, "Account already has the maximum number of routes." );

      _db.create< withdraw_vesting_route_object >( [&]( withdraw_vesting_route_object& wvdo )
      {
         wvdo.from_account = from_account.name;
         wvdo.to_account = to_account.name;
         wvdo.percent = o.percent;
         wvdo.auto_vest = o.auto_vest;
      });

      _db.modify( from_account, [&]( account_object& a )
      {
         a.withdraw_routes++;
      });
   }
   else if( o.percent == 0 )
   {
      _db.remove( *itr );

      _db.modify( from_account, [&]( account_object& a )
      {
         a.withdraw_routes--;
      });
   }
   else
   {
      _db.modify( *itr, [&]( withdraw_vesting_route_object& wvdo )
      {
         wvdo.from_account = from_account.name;
         wvdo.to_account = to_account.name;
         wvdo.percent = o.percent;
         wvdo.auto_vest = o.auto_vest;
      });
   }

   itr = wd_idx.upper_bound( boost::make_tuple( from_account.name, account_name_type() ) );
   uint16_t total_percent = 0;

   while( itr != wd_idx.end() && itr->from_account == from_account.name )
   {
      total_percent += itr->percent;
      ++itr;
   }

   FC_ASSERT( total_percent <= STEEM_100_PERCENT, "More than 100% of vesting withdrawals allocated to destinations." );
   }
   FC_CAPTURE_AND_RETHROW()
}

void account_witness_proxy_evaluator::do_apply( const account_witness_proxy_operation& o )
{
   const auto& account = _db.get_account( o.account );
   FC_ASSERT( account.proxy != o.proxy, "Proxy must change." );

   FC_ASSERT( account.can_vote, "Account has declined the ability to vote and cannot proxy votes." );

   /// remove all current votes
   std::array<share_type, STEEM_MAX_PROXY_RECURSION_DEPTH+1> delta;
   delta[0] = -account.vesting_shares.amount;
   for( int i = 0; i < STEEM_MAX_PROXY_RECURSION_DEPTH; ++i )
      delta[i+1] = -account.proxied_vsf_votes[i];
   _db.adjust_proxied_witness_votes( account, delta );

   if( o.proxy.size() ) {
      const auto& new_proxy = _db.get_account( o.proxy );
      flat_set<account_id_type> proxy_chain( { account.id, new_proxy.id } );
      proxy_chain.reserve( STEEM_MAX_PROXY_RECURSION_DEPTH + 1 );

      /// check for proxy loops and fail to update the proxy if it would create a loop
      auto cprox = &new_proxy;
      while( cprox->proxy.size() != 0 ) {
         const auto next_proxy = _db.get_account( cprox->proxy );
         FC_ASSERT( proxy_chain.insert( next_proxy.id ).second, "This proxy would create a proxy loop." );
         cprox = &next_proxy;
         FC_ASSERT( proxy_chain.size() <= STEEM_MAX_PROXY_RECURSION_DEPTH, "Proxy chain is too long." );
      }

      /// clear all individual vote records
      _db.clear_witness_votes( account );

      _db.modify( account, [&]( account_object& a ) {
         a.proxy = o.proxy;
      });

      /// add all new votes
      for( int i = 0; i <= STEEM_MAX_PROXY_RECURSION_DEPTH; ++i )
         delta[i] = -delta[i];
      _db.adjust_proxied_witness_votes( account, delta );
   } else { /// we are clearing the proxy which means we simply update the account
      _db.modify( account, [&]( account_object& a ) {
          a.proxy = o.proxy;
      });
   }
}


void account_witness_vote_evaluator::do_apply( const account_witness_vote_operation& o )
{
   const auto& voter = _db.get_account( o.account );
   FC_ASSERT( voter.proxy.size() == 0, "A proxy is currently set, please clear the proxy before voting for a witness." );

   if( o.approve )
      FC_ASSERT( voter.can_vote, "Account has declined its voting rights." );

   const auto& witness = _db.get_witness( o.witness );

   const auto& by_account_witness_idx = _db.get_index< witness_vote_index >().indices().get< by_account_witness >();
   auto itr = by_account_witness_idx.find( boost::make_tuple( voter.name, witness.owner ) );

   if( itr == by_account_witness_idx.end() ) {
      FC_ASSERT( o.approve, "Vote doesn't exist, user must indicate a desire to approve witness." );

      if ( _db.has_hardfork( STEEM_HARDFORK_0_2 ) )
      {
         FC_ASSERT( voter.witnesses_voted_for < STEEM_MAX_ACCOUNT_WITNESS_VOTES, "Account has voted for too many witnesses." ); // TODO: Remove after hardfork 2

         _db.create<witness_vote_object>( [&]( witness_vote_object& v ) {
             v.witness = witness.owner;
             v.account = voter.name;
         });

         if( _db.has_hardfork( STEEM_HARDFORK_0_3 ) ) {
            _db.adjust_witness_vote( witness, voter.witness_vote_weight() );
         }
         else {
            _db.adjust_proxied_witness_votes( voter, voter.witness_vote_weight() );
         }

      } else {

         _db.create<witness_vote_object>( [&]( witness_vote_object& v ) {
             v.witness = witness.owner;
             v.account = voter.name;
         });
         _db.modify( witness, [&]( witness_object& w ) {
             w.votes += voter.witness_vote_weight();
         });

      }
      _db.modify( voter, [&]( account_object& a ) {
         a.witnesses_voted_for++;
      });

   } else {
      FC_ASSERT( !o.approve, "Vote currently exists, user must indicate a desire to reject witness." );

      if (  _db.has_hardfork( STEEM_HARDFORK_0_2 ) ) {
         if( _db.has_hardfork( STEEM_HARDFORK_0_3 ) )
            _db.adjust_witness_vote( witness, -voter.witness_vote_weight() );
         else
            _db.adjust_proxied_witness_votes( voter, -voter.witness_vote_weight() );
      } else  {
         _db.modify( witness, [&]( witness_object& w ) {
             w.votes -= voter.witness_vote_weight();
         });
      }
      _db.modify( voter, [&]( account_object& a ) {
         a.witnesses_voted_for--;
      });
      _db.remove( *itr );
   }
}

void pre_hf20_vote_evaluator( const vote_operation& o, database& _db )
{
   const auto& comment = _db.get_comment( o.author, o.permlink );
   const auto& voter   = _db.get_account( o.voter );

   FC_ASSERT( voter.can_vote, "Voter has declined their voting rights." );

   if( o.weight > 0 ) FC_ASSERT( comment.allow_votes, "Votes are not allowed on the comment." );

   if( _db.has_hardfork( STEEM_HARDFORK_0_12__177 ) && _db.calculate_discussion_payout_time( comment ) == fc::time_point_sec::maximum() )
   {
#ifndef CLEAR_VOTES
      const auto& comment_vote_idx = _db.get_index< comment_vote_index, by_comment_voter_symbol >();
      auto itr = comment_vote_idx.find( boost::make_tuple( comment.id, voter.id, STEEM_SYMBOL ) );

      if( itr == comment_vote_idx.end() )
         _db.create< comment_vote_object >( [&]( comment_vote_object& cvo )
         {
            cvo.voter = voter.id;
            cvo.comment = comment.id;
            cvo.vote_percent = o.weight;
            cvo.last_update = _db.head_block_time();
         });
      else
         _db.modify( *itr, [&]( comment_vote_object& cvo )
         {
            cvo.vote_percent = o.weight;
            cvo.last_update = _db.head_block_time();
         });
#endif
      return;
   }

   const auto& comment_vote_idx = _db.get_index< comment_vote_index, by_comment_voter_symbol >();
   auto itr = comment_vote_idx.find( boost::make_tuple( comment.id, voter.id, STEEM_SYMBOL ) );

   int64_t elapsed_seconds = _db.head_block_time().sec_since_epoch() - voter.voting_manabar.last_update_time;

   if( _db.has_hardfork( STEEM_HARDFORK_0_11 ) )
      FC_ASSERT( elapsed_seconds >= STEEM_MIN_VOTE_INTERVAL_SEC, "Can only vote once every 3 seconds." );

   int64_t regenerated_power = (STEEM_100_PERCENT * elapsed_seconds) / STEEM_VOTING_MANA_REGENERATION_SECONDS;
   int64_t current_power     = std::min( int64_t(voter.voting_manabar.current_mana) + regenerated_power, int64_t(STEEM_100_PERCENT) );
   FC_ASSERT( current_power > 0, "Account currently does not have voting power." );

   int64_t  abs_weight    = abs(o.weight);
   // Less rounding error would occur if we did the division last, but we need to maintain backward
   // compatibility with the previous implementation which was replaced in #1285
   int64_t  used_power  = ((current_power * abs_weight) / STEEM_100_PERCENT) * (60*60*24);

   const dynamic_global_property_object& dgpo = _db.get_dynamic_global_properties();

   // The second multiplication is rounded up as of HF 259
   int64_t max_vote_denom = dgpo.target_votes_per_period * STEEM_VOTING_MANA_REGENERATION_SECONDS;

   if( !_db.has_hardfork( STEEM_HARDFORK_0_14__259 ) )
   {
      used_power = (used_power / max_vote_denom)+1;
   }
   else
   {
      used_power = (used_power + max_vote_denom - 1) / max_vote_denom;
   }
   FC_ASSERT( used_power <= current_power, "Account does not have enough power to vote." );

   int64_t abs_rshares    = ((uint128_t( _db.get_effective_vesting_shares( voter, VESTS_SYMBOL ).amount.value ) * used_power) / (STEEM_100_PERCENT)).to_uint64();
   if( !_db.has_hardfork( STEEM_HARDFORK_0_14__259 ) && abs_rshares == 0 ) abs_rshares = 1;

   if( _db.has_hardfork( STEEM_HARDFORK_0_20__1764 ) )
   {
      abs_rshares -= STEEM_VOTE_DUST_THRESHOLD;
      abs_rshares = std::max( int64_t(0), abs_rshares );
   }
   else if( _db.has_hardfork( STEEM_HARDFORK_0_14__259 ) )
   {
      FC_ASSERT( abs_rshares > STEEM_VOTE_DUST_THRESHOLD || o.weight == 0, "Voting weight is too small, please accumulate more voting power or steem power." );
   }
   else if( _db.has_hardfork( STEEM_HARDFORK_0_13__248 ) )
   {
      FC_ASSERT( abs_rshares > STEEM_VOTE_DUST_THRESHOLD || abs_rshares == 1, "Voting weight is too small, please accumulate more voting power or steem power." );
   }



   // Lazily delete vote
   if( itr != comment_vote_idx.end() && itr->num_changes == -1 )
   {
      if( _db.has_hardfork( STEEM_HARDFORK_0_12__177 ) )
         FC_ASSERT( false, "Cannot vote again on a comment after payout." );

      _db.remove( *itr );
      itr = comment_vote_idx.end();
   }

   if( itr == comment_vote_idx.end() )
   {
      FC_ASSERT( o.weight != 0, "Vote weight cannot be 0." );
      /// this is the rshares voting for or against the post
      int64_t rshares        = o.weight < 0 ? -abs_rshares : abs_rshares;

      if( rshares > 0 )
      {
         if( _db.has_hardfork( STEEM_HARDFORK_0_17__900 ) )
            FC_ASSERT( _db.head_block_time() < comment.cashout_time - STEEM_UPVOTE_LOCKOUT_HF17, "Cannot increase payout within last twelve hours before payout." );
         else if( _db.has_hardfork( STEEM_HARDFORK_0_7 ) )
            FC_ASSERT( _db.head_block_time() < _db.calculate_discussion_payout_time( comment ) - STEEM_UPVOTE_LOCKOUT_HF7, "Cannot increase payout within last minute before payout." );
      }

      //used_power /= (50*7); /// a 100% vote means use .28% of voting power which should force users to spread their votes around over 50+ posts day for a week
      //if( used_power == 0 ) used_power = 1;

      _db.modify( voter, [&]( account_object& a ){
         a.voting_manabar.current_mana = current_power - used_power;
         a.last_vote_time = _db.head_block_time();
         a.voting_manabar.last_update_time = a.last_vote_time.sec_since_epoch();
      });

      /// if the current net_rshares is less than 0, the post is getting 0 rewards so it is not factored into total rshares^2
      fc::uint128_t old_rshares = std::max(comment.net_rshares.value, int64_t(0));
      const auto& root = _db.get( comment.root_comment );
      auto old_root_abs_rshares = root.children_abs_rshares.value;

      fc::uint128_t avg_cashout_sec;

      if( !_db.has_hardfork( STEEM_HARDFORK_0_17__769 ) )
      {
         fc::uint128_t cur_cashout_time_sec = _db.calculate_discussion_payout_time( comment ).sec_since_epoch();
         fc::uint128_t new_cashout_time_sec;

         if( _db.has_hardfork( STEEM_HARDFORK_0_12__177 ) && !_db.has_hardfork( STEEM_HARDFORK_0_13__257)  )
            new_cashout_time_sec = _db.head_block_time().sec_since_epoch() + STEEM_CASHOUT_WINDOW_SECONDS_PRE_HF17;
         else
            new_cashout_time_sec = _db.head_block_time().sec_since_epoch() + STEEM_CASHOUT_WINDOW_SECONDS_PRE_HF12;

         avg_cashout_sec = ( cur_cashout_time_sec * old_root_abs_rshares + new_cashout_time_sec * abs_rshares ) / ( old_root_abs_rshares + abs_rshares );
      }

      FC_ASSERT( abs_rshares > 0, "Cannot vote with 0 rshares." );

      auto old_vote_rshares = comment.vote_rshares;

      _db.modify( comment, [&]( comment_object& c ){
         c.net_rshares += rshares;
         c.abs_rshares += abs_rshares;
         if( rshares > 0 )
            c.vote_rshares += rshares;
         if( rshares > 0 )
            c.net_votes++;
         else
            c.net_votes--;
         if( !_db.has_hardfork( STEEM_HARDFORK_0_6__114 ) && c.net_rshares == -c.abs_rshares) FC_ASSERT( c.net_votes < 0, "Comment has negative net votes?" );
      });

      _db.modify( root, [&]( comment_object& c )
      {
         c.children_abs_rshares += abs_rshares;

         if( !_db.has_hardfork( STEEM_HARDFORK_0_17__769 ) )
         {
            if( _db.has_hardfork( STEEM_HARDFORK_0_12__177 ) && c.last_payout > fc::time_point_sec::min() )
               c.cashout_time = c.last_payout + STEEM_SECOND_CASHOUT_WINDOW;
            else
               c.cashout_time = fc::time_point_sec( std::min( uint32_t( avg_cashout_sec.to_uint64() ), c.max_cashout_time.sec_since_epoch() ) );

            if( c.max_cashout_time == fc::time_point_sec::maximum() )
               c.max_cashout_time = _db.head_block_time() + fc::seconds( STEEM_MAX_CASHOUT_WINDOW_SECONDS );
         }
      });

      fc::uint128_t new_rshares = std::max( comment.net_rshares.value, int64_t(0));

      /// calculate rshares2 value
      new_rshares = util::evaluate_reward_curve( new_rshares );
      old_rshares = util::evaluate_reward_curve( old_rshares );

      uint64_t max_vote_weight = 0;

      /** this verifies uniqueness of voter
       *
       *  cv.weight / c.total_vote_weight ==> % of rshares increase that is accounted for by the vote
       *
       *  W(R) = B * R / ( R + 2S )
       *  W(R) is bounded above by B. B is fixed at 2^64 - 1, so all weights fit in a 64 bit integer.
       *
       *  The equation for an individual vote is:
       *    W(R_N) - W(R_N-1), which is the delta increase of proportional weight
       *
       *  c.total_vote_weight =
       *    W(R_1) - W(R_0) +
       *    W(R_2) - W(R_1) + ...
       *    W(R_N) - W(R_N-1) = W(R_N) - W(R_0)
       *
       *  Since W(R_0) = 0, c.total_vote_weight is also bounded above by B and will always fit in a 64 bit integer.
       *
      **/
      _db.create<comment_vote_object>( [&]( comment_vote_object& cv ){
         cv.voter   = voter.id;
         cv.comment = comment.id;
         cv.rshares = rshares;
         cv.vote_percent = o.weight;
         cv.last_update = _db.head_block_time();

         bool curation_reward_eligible = rshares > 0 && (comment.last_payout == fc::time_point_sec()) && comment.allow_curation_rewards;

         if( curation_reward_eligible && _db.has_hardfork( STEEM_HARDFORK_0_17__774 ) )
            curation_reward_eligible = _db.get_curation_rewards_percent( comment ) > 0;

         if( curation_reward_eligible )
         {
            if( comment.created < fc::time_point_sec(STEEM_HARDFORK_0_6_REVERSE_AUCTION_TIME) ) {
               u512 rshares3(rshares);
               u256 total2( comment.abs_rshares.value );

               if( !_db.has_hardfork( STEEM_HARDFORK_0_1 ) )
               {
                  rshares3 *= 1000000;
                  total2 *= 1000000;
               }

               rshares3 = rshares3 * rshares3 * rshares3;

               total2 *= total2;
               cv.weight = static_cast<uint64_t>( rshares3 / total2 );
            } else {// cv.weight = W(R_1) - W(R_0)
               const uint128_t two_s = 2 * util::get_content_constant_s();
               if( _db.has_hardfork( STEEM_HARDFORK_0_17__774 ) )
               {
                  const auto& reward_fund = _db.get_reward_fund( comment );
                  auto curve = !_db.has_hardfork( STEEM_HARDFORK_0_19__1052 ) && comment.created > STEEM_HF_19_SQRT_PRE_CALC
                                 ? curve_id::square_root : reward_fund.curation_reward_curve;
                  uint64_t old_weight = util::evaluate_reward_curve( old_vote_rshares.value, curve, reward_fund.content_constant ).to_uint64();
                  uint64_t new_weight = util::evaluate_reward_curve( comment.vote_rshares.value, curve, reward_fund.content_constant ).to_uint64();
                  cv.weight = new_weight - old_weight;
               }
               else if ( _db.has_hardfork( STEEM_HARDFORK_0_1 ) )
               {
                  uint64_t old_weight = ( ( std::numeric_limits< uint64_t >::max() * fc::uint128_t( old_vote_rshares.value ) ) / ( two_s + old_vote_rshares.value ) ).to_uint64();
                  uint64_t new_weight = ( ( std::numeric_limits< uint64_t >::max() * fc::uint128_t( comment.vote_rshares.value ) ) / ( two_s + comment.vote_rshares.value ) ).to_uint64();
                  cv.weight = new_weight - old_weight;
               }
               else
               {
                  uint64_t old_weight = ( ( std::numeric_limits< uint64_t >::max() * fc::uint128_t( 1000000 * old_vote_rshares.value ) ) / ( two_s + ( 1000000 * old_vote_rshares.value ) ) ).to_uint64();
                  uint64_t new_weight = ( ( std::numeric_limits< uint64_t >::max() * fc::uint128_t( 1000000 * comment.vote_rshares.value ) ) / ( two_s + ( 1000000 * comment.vote_rshares.value ) ) ).to_uint64();
                  cv.weight = new_weight - old_weight;
               }
            }

            max_vote_weight = cv.weight;

            if( _db.head_block_time() > fc::time_point_sec(STEEM_HARDFORK_0_6_REVERSE_AUCTION_TIME) )  /// start enforcing this prior to the hardfork
            {
               /// discount weight by time
               uint128_t w(max_vote_weight);
               uint64_t delta_t = std::min( uint64_t((cv.last_update - comment.created).to_seconds()), dgpo.reverse_auction_seconds );

               w *= delta_t;
               w /= dgpo.reverse_auction_seconds;
               cv.weight = w.to_uint64();
            }
         }
         else
         {
            cv.weight = 0;
         }
      });

      if( max_vote_weight ) // Optimization
      {
         _db.modify( comment, [&]( comment_object& c )
         {
            c.total_vote_weight += max_vote_weight;
         });
      }
      if( !_db.has_hardfork( STEEM_HARDFORK_0_17__774) )
         _db.adjust_rshares2( comment, old_rshares, new_rshares );
   }
   else
   {
      FC_ASSERT( itr->num_changes < STEEM_MAX_VOTE_CHANGES, "Voter has used the maximum number of vote changes on this comment." );

      if( _db.has_hardfork( STEEM_HARDFORK_0_6__112 ) )
         FC_ASSERT( itr->vote_percent != o.weight, "You have already voted in a similar way." );

      /// this is the rshares voting for or against the post
      int64_t rshares        = o.weight < 0 ? -abs_rshares : abs_rshares;

      if( itr->rshares < rshares )
      {
         if( _db.has_hardfork( STEEM_HARDFORK_0_17__900 ) )
            FC_ASSERT( _db.head_block_time() < comment.cashout_time - STEEM_UPVOTE_LOCKOUT_HF17, "Cannot increase payout within last twelve hours before payout." );
         else if( _db.has_hardfork( STEEM_HARDFORK_0_7 ) )
            FC_ASSERT( _db.head_block_time() < _db.calculate_discussion_payout_time( comment ) - STEEM_UPVOTE_LOCKOUT_HF7, "Cannot increase payout within last minute before payout." );
      }

      _db.modify( voter, [&]( account_object& a ){
         a.voting_manabar.current_mana = current_power - used_power;
         a.last_vote_time = _db.head_block_time();
         a.voting_manabar.last_update_time = a.last_vote_time.sec_since_epoch();
      });

      /// if the current net_rshares is less than 0, the post is getting 0 rewards so it is not factored into total rshares^2
      fc::uint128_t old_rshares = std::max(comment.net_rshares.value, int64_t(0));
      const auto& root = _db.get( comment.root_comment );
      auto old_root_abs_rshares = root.children_abs_rshares.value;

      fc::uint128_t avg_cashout_sec;

      if( !_db.has_hardfork( STEEM_HARDFORK_0_17__769 ) )
      {
         fc::uint128_t cur_cashout_time_sec = _db.calculate_discussion_payout_time( comment ).sec_since_epoch();
         fc::uint128_t new_cashout_time_sec;

         if( _db.has_hardfork( STEEM_HARDFORK_0_12__177 ) && ! _db.has_hardfork( STEEM_HARDFORK_0_13__257 )  )
            new_cashout_time_sec = _db.head_block_time().sec_since_epoch() + STEEM_CASHOUT_WINDOW_SECONDS_PRE_HF17;
         else
            new_cashout_time_sec = _db.head_block_time().sec_since_epoch() + STEEM_CASHOUT_WINDOW_SECONDS_PRE_HF12;

         if( _db.has_hardfork( STEEM_HARDFORK_0_14__259 ) && abs_rshares == 0 )
            avg_cashout_sec = cur_cashout_time_sec;
         else
            avg_cashout_sec = ( cur_cashout_time_sec * old_root_abs_rshares + new_cashout_time_sec * abs_rshares ) / ( old_root_abs_rshares + abs_rshares );
      }

      _db.modify( comment, [&]( comment_object& c )
      {
         c.net_rshares -= itr->rshares;
         c.net_rshares += rshares;
         c.abs_rshares += abs_rshares;

         /// TODO: figure out how to handle remove a vote (rshares == 0 )
         if( rshares > 0 && itr->rshares < 0 )
            c.net_votes += 2;
         else if( rshares > 0 && itr->rshares == 0 )
            c.net_votes += 1;
         else if( rshares == 0 && itr->rshares < 0 )
            c.net_votes += 1;
         else if( rshares == 0 && itr->rshares > 0 )
            c.net_votes -= 1;
         else if( rshares < 0 && itr->rshares == 0 )
            c.net_votes -= 1;
         else if( rshares < 0 && itr->rshares > 0 )
            c.net_votes -= 2;
      });

      _db.modify( root, [&]( comment_object& c )
      {
         c.children_abs_rshares += abs_rshares;

         if( !_db.has_hardfork( STEEM_HARDFORK_0_17__769 ) )
         {
            if( _db.has_hardfork( STEEM_HARDFORK_0_12__177 ) && c.last_payout > fc::time_point_sec::min() )
               c.cashout_time = c.last_payout + STEEM_SECOND_CASHOUT_WINDOW;
            else
               c.cashout_time = fc::time_point_sec( std::min( uint32_t( avg_cashout_sec.to_uint64() ), c.max_cashout_time.sec_since_epoch() ) );

            if( c.max_cashout_time == fc::time_point_sec::maximum() )
               c.max_cashout_time = _db.head_block_time() + fc::seconds( STEEM_MAX_CASHOUT_WINDOW_SECONDS );
         }
      });

      fc::uint128_t new_rshares = std::max( comment.net_rshares.value, int64_t(0));

      /// calculate rshares2 value
      new_rshares = util::evaluate_reward_curve( new_rshares );
      old_rshares = util::evaluate_reward_curve( old_rshares );


      _db.modify( comment, [&]( comment_object& c )
      {
         c.total_vote_weight -= itr->weight;
      });

      _db.modify( *itr, [&]( comment_vote_object& cv )
      {
         cv.rshares = rshares;
         cv.vote_percent = o.weight;
         cv.last_update = _db.head_block_time();
         cv.weight = 0;
         cv.num_changes += 1;
      });

      if( !_db.has_hardfork( STEEM_HARDFORK_0_17__774) )
         _db.adjust_rshares2( comment, old_rshares, new_rshares );
   }
}

void hf20_vote_evaluator( const vote_operation& o, database& _db )
{
   const auto& comment = _db.get_comment( o.author, o.permlink );
   const auto& voter   = _db.get_account( o.voter );
   const auto& dgpo    = _db.get_dynamic_global_properties();

   FC_ASSERT( voter.can_vote, "Voter has declined their voting rights." );

   if( o.weight > 0 ) FC_ASSERT( comment.allow_votes, "Votes are not allowed on the comment." );

   if( _db.calculate_discussion_payout_time( comment ) == fc::time_point_sec::maximum() )
   {
#ifndef CLEAR_VOTES
      const auto& comment_vote_idx = _db.get_index< comment_vote_index, by_comment_voter_symbol >();
      auto itr = comment_vote_idx.find( boost::make_tuple( comment.id, voter.id, STEEM_SYMBOL ) );

      if( itr == comment_vote_idx.end() )
         _db.create< comment_vote_object >( [&]( comment_vote_object& cvo )
         {
            cvo.voter = voter.id;
            cvo.comment = comment.id;
            cvo.vote_percent = o.weight;
            cvo.last_update = _db.head_block_time();
         });
      else
         _db.modify( *itr, [&]( comment_vote_object& cvo )
         {
            cvo.vote_percent = o.weight;
            cvo.last_update = _db.head_block_time();
         });
#endif
      return;
   }
   else
   {
      FC_ASSERT( _db.head_block_time() < comment.cashout_time, "Comment is actively being rewarded. Cannot vote on comment." );
   }

   const auto& comment_vote_idx = _db.get_index< comment_vote_index, by_comment_voter_symbol >();
   auto itr = comment_vote_idx.find( boost::make_tuple( comment.id, voter.id, STEEM_SYMBOL ) );

   // Lazily delete vote
   if( itr != comment_vote_idx.end() && itr->num_changes == -1 )
   {
      FC_TODO( "This looks suspicious. We might not be deleting vote objects that we should be on nodes that are configured to clear votes" );
      FC_ASSERT( false, "Cannot vote again on a comment after payout." );

      _db.remove( *itr );
      itr = comment_vote_idx.end();
   }

   auto now = _db.head_block_time();
   FC_ASSERT( ( now - voter.last_vote_time ).to_seconds() >= STEEM_MIN_VOTE_INTERVAL_SEC, "Can only vote once every 3 seconds." );

   _db.modify( voter, [&]( account_object& a )
   {
      util::update_manabar( _db.get_dynamic_global_properties(), a, STEEM_VOTING_MANA_REGENERATION_SECONDS, _db.has_hardfork( STEEM_HARDFORK_0_21__3336 ) );
   });

   if ( _db.has_hardfork( STEEM_HARDFORK_0_21__3004 ) )
   {
      FC_ASSERT( voter.voting_manabar.current_mana >= 0, "Account does not have enough mana to vote." );
   }
   else
   {
      FC_ASSERT( voter.voting_manabar.current_mana > 0, "Account does not have enough mana to vote." );
   }

   int16_t abs_weight = abs( o.weight );
   uint128_t used_mana = 0;

   if( dgpo.downvote_pool_percent && o.weight < 0 )
   {
      if( _db.has_hardfork( STEEM_HARDFORK_0_22__3485 ) )
      {
         used_mana = ( std::max( ( ( uint128_t( voter.downvote_manabar.current_mana ) * STEEM_100_PERCENT ) / dgpo.downvote_pool_percent ),
                                uint128_t( voter.voting_manabar.current_mana ) )
                  * abs_weight * 60 * 60 * 24 ) / STEEM_100_PERCENT;
      }
      else
      {
         used_mana = ( std::max( ( uint128_t( voter.downvote_manabar.current_mana * STEEM_100_PERCENT ) / dgpo.downvote_pool_percent ),
                                uint128_t( voter.voting_manabar.current_mana ) )
                  * abs_weight * 60 * 60 * 24 ) / STEEM_100_PERCENT;
      }

   }
   else
   {
      used_mana = ( uint128_t( voter.voting_manabar.current_mana ) * abs_weight * 60 * 60 * 24 ) / STEEM_100_PERCENT;
   }

   int64_t max_vote_denom = dgpo.target_votes_per_period * STEEM_VOTING_MANA_REGENERATION_SECONDS;

   used_mana = ( used_mana + max_vote_denom - 1 ) / max_vote_denom;
   if( _db.has_hardfork( STEEM_HARDFORK_0_21__3336 ) && dgpo.downvote_pool_percent && o.weight < 0 )
   {
      FC_ASSERT( voter.voting_manabar.current_mana + voter.downvote_manabar.current_mana > used_mana.to_int64(),
         "Account does not have enough mana to downvote. voting_mana: ${v} downvote_mana: ${d} required_mana: ${r}",
         ("v", voter.voting_manabar.current_mana)("d", voter.downvote_manabar.current_mana)("r", used_mana.to_int64()) );
   }
   else
   {
      FC_ASSERT( voter.voting_manabar.has_mana( used_mana.to_int64() ), "Account does not have enough mana to vote." );
   }

   int64_t abs_rshares = used_mana.to_int64();

   abs_rshares -= STEEM_VOTE_DUST_THRESHOLD;
   abs_rshares = std::max( int64_t(0), abs_rshares );

   uint32_t cashout_delta = ( comment.cashout_time - _db.head_block_time() ).to_seconds();

   if( cashout_delta < STEEM_UPVOTE_LOCKOUT_SECONDS )
   {
      abs_rshares = (int64_t) ( ( uint128_t( abs_rshares ) * cashout_delta ) / STEEM_UPVOTE_LOCKOUT_SECONDS ).to_uint64();
   }

   if( itr == comment_vote_idx.end() )
   {
      FC_ASSERT( o.weight != 0, "Vote weight cannot be 0." );
      /// this is the rshares voting for or against the post

      int64_t rshares = o.weight < 0 ? -abs_rshares : abs_rshares;

      _db.modify( voter, [&]( account_object& a )
      {
         if( _db.has_hardfork( STEEM_HARDFORK_0_21__3336 ) && dgpo.downvote_pool_percent > 0 && o.weight < 0 )
         {
            if( used_mana.to_int64() > a.downvote_manabar.current_mana )
            {
               /* used mana is always less than downvote_mana + voting_mana because the amount used
                * is a fraction of max( downvote_mana, voting_mana ). If more mana is consumed than
                * there is downvote_mana, then it is because voting_mana is greater, and used_mana
                * is strictly smaller than voting_mana. This is the same reason why a check is not
                * required when using voting mana on its own as an upvote.
                */
               auto remainder = used_mana.to_int64() - a.downvote_manabar.current_mana;
               a.downvote_manabar.use_mana( a.downvote_manabar.current_mana );
               a.voting_manabar.use_mana( remainder );
            }
            else
            {
               a.downvote_manabar.use_mana( used_mana.to_int64() );
            }
         }
         else
         {
            a.voting_manabar.use_mana( used_mana.to_int64() );
         }

         a.last_vote_time = _db.head_block_time();
      });

      const auto& root = _db.get( comment.root_comment );

      auto old_vote_rshares = comment.vote_rshares;

      _db.modify( comment, [&]( comment_object& c )
      {
         c.net_rshares += rshares;
         c.abs_rshares += abs_rshares;
         if( rshares > 0 )
            c.vote_rshares += rshares;
         if( rshares > 0 )
            c.net_votes++;
         else
            c.net_votes--;
      });

      _db.modify( root, [&]( comment_object& c )
      {
         c.children_abs_rshares += abs_rshares;
      });

      uint64_t max_vote_weight = 0;

      /** this verifies uniqueness of voter
       *
       *  cv.weight / c.total_vote_weight ==> % of rshares increase that is accounted for by the vote
       *
       *  W(R) = B * R / ( R + 2S )
       *  W(R) is bounded above by B. B is fixed at 2^64 - 1, so all weights fit in a 64 bit integer.
       *
       *  The equation for an individual vote is:
       *    W(R_N) - W(R_N-1), which is the delta increase of proportional weight
       *
       *  c.total_vote_weight =
       *    W(R_1) - W(R_0) +
       *    W(R_2) - W(R_1) + ...
       *    W(R_N) - W(R_N-1) = W(R_N) - W(R_0)
       *
       *  Since W(R_0) = 0, c.total_vote_weight is also bounded above by B and will always fit in a 64 bit integer.
       *
      **/
      _db.create<comment_vote_object>( [&]( comment_vote_object& cv )
      {
         cv.voter   = voter.id;
         cv.comment = comment.id;
         cv.rshares = rshares;
         cv.vote_percent = o.weight;
         cv.last_update = _db.head_block_time();

         bool curation_reward_eligible = rshares > 0 && (comment.last_payout == fc::time_point_sec()) && comment.allow_curation_rewards;

         if( curation_reward_eligible )
         {
            curation_reward_eligible = _db.get_curation_rewards_percent( comment ) > 0;
         }

         if( curation_reward_eligible )
         {
            // cv.weight = W(R_1) - W(R_0)
            const auto& reward_fund = _db.get_reward_fund( comment );
            auto curve = reward_fund.curation_reward_curve;
            uint64_t old_weight = util::evaluate_reward_curve( old_vote_rshares.value, curve, reward_fund.content_constant ).to_uint64();
            uint64_t new_weight = util::evaluate_reward_curve( comment.vote_rshares.value, curve, reward_fund.content_constant ).to_uint64();

            if( old_weight >= new_weight ) // old_weight > new_weight should never happen
            {
               cv.weight = 0;
            }
            else
            {
               cv.weight = new_weight - old_weight;

               max_vote_weight = cv.weight;

               /// discount weight by time
               uint128_t w(max_vote_weight);
               uint64_t delta_t = std::min( uint64_t((cv.last_update - comment.created).to_seconds()), uint64_t( dgpo.reverse_auction_seconds ) );

               w *= delta_t;
               w /= dgpo.reverse_auction_seconds;
               cv.weight = w.to_uint64();
            }
         }
         else
         {
            cv.weight = 0;
         }
      });

      if( max_vote_weight ) // Optimization
      {
         _db.modify( comment, [&]( comment_object& c )
         {
            c.total_vote_weight += max_vote_weight;
         });
      }
   }
   else
   {
      FC_ASSERT( itr->num_changes < STEEM_MAX_VOTE_CHANGES, "Voter has used the maximum number of vote changes on this comment." );
      FC_ASSERT( itr->vote_percent != o.weight, "Your current vote on this comment is identical to this vote." );

      int64_t rshares = o.weight < 0 ? -abs_rshares : abs_rshares;

      _db.modify( voter, [&]( account_object& a )
      {
         if( _db.has_hardfork( STEEM_HARDFORK_0_21__3336 ) && dgpo.downvote_pool_percent > 0 && o.weight < 0 )
         {
            if( used_mana.to_int64() > a.downvote_manabar.current_mana )
            {
               /* used mana is always less than downvote_mana + voting_mana because the amount used
                * is a fraction of max( downvote_mana, voting_mana ). If more mana is consumed than
                * there is downvote_mana, then it is because voting_mana is greater, and used_mana
                * is strictly smaller than voting_mana. This is the same reason why a check is not
                * required when using voting mana on its own as an upvote.
                */
               auto remainder = used_mana.to_int64() - a.downvote_manabar.current_mana;
               a.downvote_manabar.use_mana( a.downvote_manabar.current_mana );
               a.voting_manabar.use_mana( remainder );
            }
            else
            {
               a.downvote_manabar.use_mana( used_mana.to_int64() );
            }
         }
         else
         {
            a.voting_manabar.use_mana( used_mana.to_int64() );
         }

         a.last_vote_time = _db.head_block_time();
      });

      const auto& root = _db.get( comment.root_comment );

      _db.modify( comment, [&]( comment_object& c )
      {
         c.net_rshares -= itr->rshares;
         c.net_rshares += rshares;
         c.abs_rshares += abs_rshares;

         /// TODO: figure out how to handle remove a vote (rshares == 0 )
         if( rshares > 0 && itr->rshares < 0 )
            c.net_votes += 2;
         else if( rshares > 0 && itr->rshares == 0 )
            c.net_votes += 1;
         else if( rshares == 0 && itr->rshares < 0 )
            c.net_votes += 1;
         else if( rshares == 0 && itr->rshares > 0 )
            c.net_votes -= 1;
         else if( rshares < 0 && itr->rshares == 0 )
            c.net_votes -= 1;
         else if( rshares < 0 && itr->rshares > 0 )
            c.net_votes -= 2;
      });

      _db.modify( root, [&]( comment_object& c )
      {
         c.children_abs_rshares += abs_rshares;
      });

      _db.modify( comment, [&]( comment_object& c )
      {
         c.total_vote_weight -= itr->weight;
      });

      _db.modify( *itr, [&]( comment_vote_object& cv )
      {
         cv.rshares = rshares;
         cv.vote_percent = o.weight;
         cv.last_update = _db.head_block_time();
         cv.weight = 0;
         cv.num_changes += 1;
      });
   }
}

template< typename VoteType >
struct vote_context
{
   vote_context(
      const VoteType& o,
      const comment_object& c,
      const dynamic_global_property_object& d,
      account_id_type v_id,
      int64_t r,
      asset_symbol_type s,
      int32_t mrp,
      int32_t vprp
   ) :
      op( o ),
      comment( c ),
      dgpo( d ),
      voter_id( v_id ),
      rshares( r ),
      symbol( s ),
      mana_regen_period( mrp ),
      votes_per_regen_period( vprp )
   {}

   vote_context(
      const VoteType& o,
      const comment_object& c,
      const dynamic_global_property_object& d,
      account_id_type v_id
   ) :
      op( o ),
      comment( c ),
      dgpo( d ),
      voter_id( v_id )
   {}

   const VoteType&                        op;
   const comment_object&                  comment;
   const dynamic_global_property_object&  dgpo;
   account_id_type                        voter_id;
   int64_t                                rshares = 0;
   asset_symbol_type                      symbol;
   int32_t                                mana_regen_period = 0;
   int32_t                                votes_per_regen_period = 0;
};

template< typename ContextType, typename AccountType >
void generic_vote_evaluator(
   const ContextType& ctx,
   const AccountType& voter,
   rshare_context& rshare_ctx,
   database& _db )
{
   const auto& comment_vote_idx = _db.get_index< comment_vote_index, by_comment_voter_symbol >();

   auto now = _db.head_block_time();

   if( _db.calculate_discussion_payout_time( ctx.comment ) == fc::time_point_sec::maximum() )
   {
#ifndef CLEAR_VOTES
      auto itr = comment_vote_idx.find( boost::make_tuple( ctx.comment.id, ctx.voter_id, ctx.symbol ) );

      if( itr == comment_vote_idx.end() )
         _db.create< comment_vote_object >( [&]( comment_vote_object& cvo )
         {
            cvo.voter = ctx.voter_id;
            cvo.comment = ctx.comment.id;
            cvo.last_update = now;
         });
      else
         _db.modify( *itr, [&]( comment_vote_object& cvo )
         {
            cvo.last_update = now;
         });
#endif
      return;
   }
   else
   {
      FC_ASSERT( now < ctx.comment.cashout_time, "Comment is actively being rewarded. Cannot vote on comment." );
   }

   auto itr = comment_vote_idx.find( boost::make_tuple( ctx.comment.id, ctx.voter_id, ctx.symbol ) );

   // Lazily delete vote
   if( itr != comment_vote_idx.end() && itr->num_changes == -1 )
   {
      FC_TODO( "This looks suspicious. We might not be deleting vote objects that we should be on nodes that are configured to clear votes" );
      FC_ASSERT( false, "Cannot vote again on a comment after payout." );

      _db.remove( *itr );
      itr = comment_vote_idx.end();
   }

   _db.modify( voter, [&]( AccountType& a )
   {
      util::update_manabar( _db.get_dynamic_global_properties(), a, ctx.mana_regen_period, true );
   });

   FC_ASSERT( voter.voting_manabar.current_mana >= 0, "Account does not have enough mana to vote." );

   int64_t max_abs_rshares = std::max( int64_t(0), ( voter.voting_manabar.current_mana + ctx.votes_per_regen_period - 1 ) / ctx.votes_per_regen_period );
   int64_t abs_rshares = abs( ctx.rshares );

   if( ctx.rshares < 0 )
   {
      max_abs_rshares = std::max(
                           max_abs_rshares,
                           ( ( fc::uint128_t ( voter.downvote_manabar.current_mana ) * STEEM_100_PERCENT + ctx.votes_per_regen_period - 1 ) / ( ctx.votes_per_regen_period * ctx.dgpo.downvote_pool_percent ) ).to_int64()
                        );
   }

   FC_ASSERT( abs_rshares <= max_abs_rshares, "Account cannot vote with more than ${m} abs rshares in a single vote with token ${t}. Attempted: ${r}",
      ("m", max_abs_rshares)("t", ctx.symbol)("r", ctx.rshares) );

   if( ctx.dgpo.downvote_pool_percent && ctx.rshares < 0 )
   {
      FC_ASSERT( voter.voting_manabar.current_mana + voter.downvote_manabar.current_mana > max_abs_rshares,
         "Account does not have enough mana to downvote. voting_mana: ${v} downvote_mana: ${d} required_mana: ${r}",
         ("v", voter.voting_manabar.current_mana)("d", voter.downvote_manabar.current_mana)("r", max_abs_rshares) );
   }
   else
   {
      FC_ASSERT( voter.voting_manabar.has_mana( max_abs_rshares ), "Account does not have enough mana to vote." );
   }

   FC_TODO( "Determine if we should use the same dust threshold for SMTs or allow the creators to set it themselves" );
   auto consumed_rshares = abs_rshares;
   abs_rshares -= STEEM_VOTE_DUST_THRESHOLD;
   abs_rshares = std::max( int64_t(0), abs_rshares );

   uint32_t cashout_delta = ( ctx.comment.cashout_time - _db.head_block_time() ).to_seconds();

   if( cashout_delta < STEEM_UPVOTE_LOCKOUT_SECONDS )
   {
      abs_rshares = (int64_t) ( ( uint128_t( abs_rshares ) * cashout_delta ) / STEEM_UPVOTE_LOCKOUT_SECONDS ).to_uint64();
   }

   if( itr == comment_vote_idx.end() )
   {
      FC_ASSERT( ctx.rshares != 0, "Cannot vote with 0 rshares." );
      /// this is the rshares voting for or against the post

      int64_t comment_rshares = ctx.rshares < 0 ? -abs_rshares : abs_rshares;

      _db.modify( voter, [&]( AccountType& a )
      {
         if( ctx.dgpo.downvote_pool_percent > 0 && ctx.rshares < 0 )
         {
            if( consumed_rshares > a.downvote_manabar.current_mana )
            {
               /* used mana is always less than downvote_mana + voting_mana because the amount used
                * is a fraction of max( downvote_mana, voting_mana ). If more mana is consumed than
                * there is downvote_mana, then it is because voting_mana is greater, and used_mana
                * is strictly smaller than voting_mana. This is the same reason why a check is not
                * required when using voting mana on its own as an upvote.
                */
               auto remainder = consumed_rshares - a.downvote_manabar.current_mana;

               a.downvote_manabar.use_mana( a.downvote_manabar.current_mana );
               a.voting_manabar.use_mana( remainder );
            }
            else
            {
               a.downvote_manabar.use_mana( consumed_rshares );
            }
         }
         else
         {
            a.voting_manabar.use_mana( consumed_rshares );
         }

         a.last_vote_time = _db.head_block_time();
      });

      auto old_vote_rshares = ctx.comment.vote_rshares;
      uint64_t max_vote_weight = 0;

      rshare_ctx.net_rshares += comment_rshares;
      rshare_ctx.abs_rshares += abs_rshares;
      if( comment_rshares > 0 )
         rshare_ctx.vote_rshares += comment_rshares;
      if( comment_rshares > 0 )
         rshare_ctx.net_votes++;
      else
         rshare_ctx.net_votes--;

      /** this verifies uniqueness of voter
       *
       *  cv.weight / c.total_vote_weight ==> % of rshares increase that is accounted for by the vote
       *
       *  W(R) = B * R / ( R + 2S )
       *  W(R) is bounded above by B. B is fixed at 2^64 - 1, so all weights fit in a 64 bit integer.
       *
       *  The equation for an individual vote is:
       *    W(R_N) - W(R_N-1), which is the delta increase of proportional weight
       *
       *  c.total_vote_weight =
       *    W(R_1) - W(R_0) +
       *    W(R_2) - W(R_1) + ...
       *    W(R_N) - W(R_N-1) = W(R_N) - W(R_0)
       *
       *  Since W(R_0) = 0, c.total_vote_weight is also bounded above by B and will always fit in a 64 bit integer.
       *
      **/
      _db.create< comment_vote_object >( [&]( comment_vote_object& cv )
      {
         cv.voter       = ctx.voter_id;
         cv.comment     = ctx.comment.id;
         cv.rshares     = comment_rshares;
         cv.last_update = _db.head_block_time();
         cv.symbol      = ctx.symbol;

         bool curation_reward_eligible = ctx.rshares > 0 && ( ctx.comment.last_payout == fc::time_point_sec() );
         if( ctx.symbol == STEEM_SYMBOL )
         {
            curation_reward_eligible &= ctx.comment.allow_curation_rewards;
         }
         else
         {
            auto smt_opts = ctx.comment.allowed_vote_assets.find( ctx.symbol );
            if( smt_opts != ctx.comment.allowed_vote_assets.end() )
            {
               curation_reward_eligible &= smt_opts->second.allow_curation_rewards;
            }
         }

         if( curation_reward_eligible )
         {
            if( ctx.symbol == STEEM_SYMBOL )
            {
               curation_reward_eligible = _db.get_curation_rewards_percent( ctx.comment ) > 0;
            }
            else
            {
               curation_reward_eligible = _db.get< smt_token_object, by_symbol >( ctx.symbol ).percent_curation_rewards;
            }
         }

         if( curation_reward_eligible )
         {
            // cv.weight = W(R_1) - W(R_0)
            if( ctx.symbol == STEEM_SYMBOL )
            {
               const auto& reward_fund = _db.get_reward_fund( ctx.comment );
               auto curve = reward_fund.curation_reward_curve;
               uint64_t old_weight = util::evaluate_reward_curve( old_vote_rshares.value, curve, reward_fund.content_constant ).to_uint64();
               uint64_t new_weight = util::evaluate_reward_curve( rshare_ctx.vote_rshares.value, curve, reward_fund.content_constant ).to_uint64();
               cv.weight = new_weight - old_weight;
            }
            else
            {
               const auto& smt = _db.get< smt_token_object, by_symbol >( ctx.symbol );
               uint64_t old_weight = util::evaluate_reward_curve( old_vote_rshares.value, smt.curation_reward_curve, smt.content_constant ).to_uint64();
               uint64_t new_weight = util::evaluate_reward_curve( rshare_ctx.vote_rshares.value, smt.curation_reward_curve, smt.content_constant ).to_uint64();
               cv.weight = new_weight - old_weight;
            }

            max_vote_weight = cv.weight;

            /// discount weight by time
            uint128_t w(max_vote_weight);
            uint64_t delta_t = std::min( uint64_t((cv.last_update - ctx.comment.created).to_seconds()), uint64_t( ctx.dgpo.reverse_auction_seconds ) );

            w *= delta_t;
            w /= ctx.dgpo.reverse_auction_seconds;
            cv.weight = w.to_uint64();
         }
         else
         {
            cv.weight = 0;
         }
      });

      rshare_ctx.total_vote_weight += max_vote_weight;

      FC_TODO( "Check if this needed anywhere. Possibly still used by Hivemind" );
      _db.modify( _db.get( ctx.comment.root_comment ), [&]( comment_object& c )
      {
         c.children_abs_rshares += abs_rshares;
      });
   }
   else
   {
      FC_ASSERT( itr->num_changes < STEEM_MAX_VOTE_CHANGES, "Voter has used the maximum number of vote changes on this comment." );

      int64_t comment_rshares = ctx.rshares < 0 ? -abs_rshares : abs_rshares;
      FC_ASSERT( itr->rshares != comment_rshares, "Your current vote on this comment is identical to the new vote." );

      // If they have the same sign
      if( ( itr->rshares ^ ctx.rshares ) >= 0 )
      {
         consumed_rshares = std::max( int64_t(0), abs_rshares - abs( itr->rshares ) );
      }

      _db.modify( voter, [&]( AccountType& a )
      {
         if( ctx.dgpo.downvote_pool_percent > 0 && comment_rshares < 0 )
         {
            if( consumed_rshares > a.downvote_manabar.current_mana )
            {
               /* used mana is always less than downvote_mana + voting_mana because the amount used
                * is a fraction of max( downvote_mana, voting_mana ). If more mana is consumed than
                * there is downvote_mana, then it is because voting_mana is greater, and used_mana
                * is strictly smaller than voting_mana. This is the same reason why a check is not
                * required when using voting mana on its own as an upvote.
                */
               auto remainder = consumed_rshares - a.downvote_manabar.current_mana;
               a.downvote_manabar.use_mana( a.downvote_manabar.current_mana );
               a.voting_manabar.use_mana( remainder );
            }
            else
            {
               a.downvote_manabar.use_mana( consumed_rshares );
            }
         }
         else
         {
            a.voting_manabar.use_mana( consumed_rshares );
         }

         a.last_vote_time = _db.head_block_time();
      });

      rshare_ctx.net_rshares -= itr->rshares;
      rshare_ctx.net_rshares += comment_rshares;
      rshare_ctx.abs_rshares += abs_rshares;

      /// TODO: figure out how to handle remove a vote (rshares == 0 )
      if( comment_rshares > 0 && itr->rshares < 0 )
         rshare_ctx.net_votes += 2;
      else if( comment_rshares > 0 && itr->rshares == 0 )
         rshare_ctx.net_votes += 1;
      else if( comment_rshares == 0 && itr->rshares < 0 )
         rshare_ctx.net_votes += 1;
      else if( comment_rshares == 0 && itr->rshares > 0 )
         rshare_ctx.net_votes -= 1;
      else if( comment_rshares < 0 && itr->rshares == 0 )
         rshare_ctx.net_votes -= 1;
      else if( comment_rshares < 0 && itr->rshares > 0 )
         rshare_ctx.net_votes -= 2;

      rshare_ctx.total_vote_weight -= itr->weight;

      _db.modify( *itr, [&]( comment_vote_object& cv )
      {
         cv.rshares = comment_rshares;
         cv.last_update = _db.head_block_time();
         cv.weight = 0;
         cv.num_changes += 1;
      });

      FC_TODO( "Check if this needed anywhere. Possibly still used by Hivemind" );
      _db.modify( _db.get( ctx.comment.root_comment ), [&]( comment_object& c )
      {
         c.children_abs_rshares += abs_rshares;
      });
   }
}

void vote_evaluator::do_apply( const vote_operation& o )
{ try {
   FC_TODO( "Deprecate vote_operation in a future hardfork" );
   if( _db.has_hardfork( STEEM_SMT_HARDFORK ) )
   {
      const auto& comment = _db.get_comment( o.author, o.permlink );
      const auto& voter   = _db.get_account( o.voter );
      const auto& dgpo    = _db.get_dynamic_global_properties();

      FC_ASSERT( voter.can_vote, "Voter has declined their voting rights." );

      FC_ASSERT( ( _db.head_block_time() - voter.last_vote_time ).to_seconds() >= STEEM_MIN_VOTE_INTERVAL_SEC, "Can only vote once every 3 seconds." );

      if( o.weight > 0 ) FC_ASSERT( comment.allow_votes, "Votes are not allowed on the comment." );

      _db.modify( voter, [&]( account_object& a )
      {
         util::update_manabar( _db.get_dynamic_global_properties(), a, STEEM_VOTING_MANA_REGENERATION_SECONDS, true );
      });

      int16_t abs_weight = abs( o.weight );
      int64_t rshares = ( ( uint128_t( voter.voting_manabar.current_mana ) * abs_weight + dgpo.target_votes_per_period - 1 ) / ( dgpo.target_votes_per_period * STEEM_100_PERCENT ) ).to_int64();

      if( o.weight < 0 )
      {
         rshares = -1 * std::max(
                     rshares,
                     // The full version of this calculation would have STEEM_100_PERCENT in the numerator and denominator. They have been removed
                     ( ( uint128_t( voter.downvote_manabar.current_mana ) * abs_weight + dgpo.target_votes_per_period - 1 ) / ( dgpo.target_votes_per_period * dgpo.downvote_pool_percent ) ).to_int64()
                  );
      }


      vote_context< vote_operation > ctx( o, comment, dgpo, voter.id, rshares, STEEM_SYMBOL, STEEM_VOTING_MANA_REGENERATION_SECONDS, dgpo.target_votes_per_period );
      rshare_context rshare_ctx { comment.net_rshares, comment.abs_rshares, comment.vote_rshares, comment.total_vote_weight, comment.net_votes, comment.author_rewards };
      generic_vote_evaluator( ctx, voter, rshare_ctx, _db );

      _db.modify( comment, [&]( comment_object& c )
      {
         c.net_rshares = rshare_ctx.net_rshares;
         c.abs_rshares = rshare_ctx.abs_rshares;
         c.vote_rshares = rshare_ctx.vote_rshares;
         c.total_vote_weight = rshare_ctx.total_vote_weight;
         c.net_votes = rshare_ctx.net_votes;
         c.author_rewards = rshare_ctx.author_rewards;
      });
   }
   else if( _db.has_hardfork( STEEM_HARDFORK_0_20__2539 ) )
   {
      hf20_vote_evaluator( o, _db );
   }
   else
   {
      pre_hf20_vote_evaluator( o, _db );
   }
} FC_CAPTURE_AND_RETHROW( (o) ) }

void vote2_evaluator::do_apply( const vote2_operation& o )
{
   FC_TODO( "Remove after SMT Hardfork" );
   FC_ASSERT( _db.has_hardfork( STEEM_SMT_HARDFORK ), "vote2_operation is not enabled until SMT Hardfork" );

   const auto& comment = _db.get_comment( o.author, o.permlink );
   const auto& root    = _db.get< comment_object, by_id >( comment.root_comment );
   const auto& voter   = _db.get_account( o.voter );
   const auto& dgpo    = _db.get_dynamic_global_properties();

   FC_ASSERT( voter.can_vote, "Voter has declined their voting rights." );

   vote_context< vote2_operation > ctx( o, comment, dgpo, voter.id );
   rshare_context rshare_ctx;

   for( auto& symbol_rshare : o.rshares )
   {
      ctx.rshares = symbol_rshare.second;
      ctx.symbol = symbol_rshare.first;

      if( symbol_rshare.first == STEEM_SYMBOL )
      {
         if( symbol_rshare.second > 0 ) FC_ASSERT( comment.allow_votes, "Votes are not allowed on the comment." );

         FC_ASSERT( ( _db.head_block_time() - voter.last_vote_time ).to_seconds() >= STEEM_MIN_VOTE_INTERVAL_SEC, "Can only vote once every 3 seconds." );

         ctx.mana_regen_period = STEEM_VOTING_MANA_REGENERATION_SECONDS;
         ctx.votes_per_regen_period = dgpo.target_votes_per_period;

         rshare_ctx.net_rshares = comment.net_rshares;
         rshare_ctx.abs_rshares = comment.abs_rshares;
         rshare_ctx.vote_rshares = comment.vote_rshares;
         rshare_ctx.total_vote_weight = comment.total_vote_weight;
         rshare_ctx.net_votes = comment.net_votes;
         rshare_ctx.author_rewards = comment.author_rewards;

         generic_vote_evaluator( ctx, voter, rshare_ctx, _db );

         _db.modify( comment, [&]( comment_object& c )
         {
            c.net_rshares = rshare_ctx.net_rshares;
            c.abs_rshares = rshare_ctx.abs_rshares;
            c.vote_rshares = rshare_ctx.vote_rshares;
            c.total_vote_weight = rshare_ctx.total_vote_weight;
            c.net_votes = rshare_ctx.net_votes;
            c.author_rewards = rshare_ctx.author_rewards;
         });
      }
      else
      {
         auto votable_smt = root.allowed_vote_assets.find( symbol_rshare.first );
         FC_ASSERT( votable_smt != root.allowed_vote_assets.end(), "SMT ${s} is not an allowed voting asset on comment ${a}/${p}",
            ("s", symbol_rshare.first)("a", o.author)("p", o.permlink) );

         const auto& smt = _db.get< smt_token_object, by_symbol >( symbol_rshare.first );
         ctx.mana_regen_period = smt.vote_regeneration_period_seconds;
         ctx.votes_per_regen_period = smt.votes_per_regeneration_period;

         const auto& smt_balance = _db.get< account_regular_balance_object, by_name_liquid_symbol >( boost::make_tuple( o.voter, symbol_rshare.first ) );

         FC_ASSERT( ( _db.head_block_time() - smt_balance.last_vote_time ).to_seconds() >= STEEM_MIN_VOTE_INTERVAL_SEC, "Can only vote once every 3 seconds." );

         auto itr = comment.smt_rshares.find( symbol_rshare.first );
         if( itr != comment.smt_rshares.end() )
            rshare_ctx = itr->second;
         else
            rshare_ctx = rshare_context();

         generic_vote_evaluator( ctx, smt_balance, rshare_ctx, _db );

         _db.modify( comment, [&]( comment_object& c )
         {
            c.smt_rshares[ symbol_rshare.first ] = rshare_ctx;
         });
      }
   }
}

void custom_evaluator::do_apply( const custom_operation& o )
{
   database& d = db();
   if( d.is_producing() )
      FC_ASSERT( o.data.size() <= STEEM_CUSTOM_OP_DATA_MAX_LENGTH,
         "Operation data must be less than ${bytes} bytes.", ("bytes", STEEM_CUSTOM_OP_DATA_MAX_LENGTH) );

   if( _db.is_producing() || _db.has_hardfork( STEEM_HARDFORK_0_20 ) )
   {
      FC_ASSERT( o.required_auths.size() <= STEEM_MAX_AUTHORITY_MEMBERSHIP,
         "Authority membership exceeded. Max: ${max} Current: ${n}", ("max", STEEM_MAX_AUTHORITY_MEMBERSHIP)("n", o.required_auths.size()) );
   }
}

void custom_json_evaluator::do_apply( const custom_json_operation& o )
{
   database& d = db();

   // ilog( "custom_json_evaluator   is_producing = ${p}    operation = ${o}", ("p", d.is_producing())("o", o) );

   if( d.is_producing() )
      FC_ASSERT( o.json.length() <= STEEM_CUSTOM_OP_DATA_MAX_LENGTH,
         "Operation JSON must be less than ${bytes} bytes.", ("bytes", STEEM_CUSTOM_OP_DATA_MAX_LENGTH) );

   if( _db.is_producing() || _db.has_hardfork( STEEM_HARDFORK_0_20 ) )
   {
      size_t num_auths = o.required_auths.size() + o.required_posting_auths.size();
      FC_ASSERT( num_auths <= STEEM_MAX_AUTHORITY_MEMBERSHIP,
         "Authority membership exceeded. Max: ${max} Current: ${n}", ("max", STEEM_MAX_AUTHORITY_MEMBERSHIP)("n", num_auths) );
   }

   std::shared_ptr< custom_operation_interpreter > eval = d.get_custom_json_evaluator( o.id );
   if( !eval )
   {
      // ilog( "Accepting, no evaluator registered" );
      return;
   }

   try
   {
      eval->apply( o );
   }
   catch( const fc::exception& e )
   {
      if( d.is_producing() )
      {
         // ilog( "Re-throwing exception ${e}", ("e", e) );
         throw e;
      }
      // ilog( "Suppressing exception ${e}", ("e", e) );
   }
   catch(...)
   {
      // elog( "Unexpected exception applying custom json evaluator." );
   }
}


void custom_binary_evaluator::do_apply( const custom_binary_operation& o )
{
   database& d = db();
   if( d.is_producing() )
   {
      FC_ASSERT( o.data.size() <= STEEM_CUSTOM_OP_DATA_MAX_LENGTH,
         "Operation data must be less than ${bytes} bytes.", ("bytes", STEEM_CUSTOM_OP_DATA_MAX_LENGTH) );
      FC_ASSERT( false, "custom_binary_operation is deprecated" );
   }
   FC_ASSERT( d.has_hardfork( STEEM_HARDFORK_0_14__317 ) );

   if( _db.is_producing() || _db.has_hardfork( STEEM_HARDFORK_0_20 ) )
   {
      size_t num_auths = o.required_owner_auths.size() + o.required_active_auths.size() + o.required_posting_auths.size();
      for( const auto& auth : o.required_auths )
      {
         num_auths += auth.key_auths.size() + auth.account_auths.size();
      }

      FC_ASSERT( num_auths <= STEEM_MAX_AUTHORITY_MEMBERSHIP,
         "Authority membership exceeded. Max: ${max} Current: ${n}", ("max", STEEM_MAX_AUTHORITY_MEMBERSHIP)("n", num_auths) );
   }

   std::shared_ptr< custom_operation_interpreter > eval = d.get_custom_json_evaluator( o.id );
   if( !eval )
      return;

   try
   {
      eval->apply( o );
   }
   catch( const fc::exception& e )
   {
      if( d.is_producing() )
         throw e;
   }
   catch(...)
   {
      elog( "Unexpected exception applying custom json evaluator." );
   }
}


template<typename Operation>
void pow_apply( database& db, Operation o )
{
   const auto& dgp = db.get_dynamic_global_properties();

   if( db.has_hardfork( STEEM_HARDFORK_0_5__59 ) )
   {
      const auto& witness_by_work = db.get_index<witness_index>().indices().get<by_work>();
      auto work_itr = witness_by_work.find( o.work.work );
      if( work_itr != witness_by_work.end() )
      {
          FC_ASSERT( !"DUPLICATE WORK DISCOVERED", "${w}  ${witness}",("w",o)("wit",*work_itr) );
      }
   }

   const auto& accounts_by_name = db.get_index<account_index>().indices().get<by_name>();

   auto itr = accounts_by_name.find(o.get_worker_account());
   if(itr == accounts_by_name.end())
   {
      const auto& new_account = db.create< account_object >( [&]( account_object& acc )
      {
         initialize_account_object( acc, o.get_worker_account(), o.work.worker, dgp, true /*mined*/, account_name_type(), db.get_hardfork() );
         // ^ empty recovery account parameter means highest voted witness at time of recovery
      });

#ifndef IS_LOW_MEM
      db.create< account_metadata_object >( [&]( account_metadata_object& meta )
      {
         meta.account = new_account.id;
      });
#else
      FC_UNUSED( new_account );
#endif

      db.create< account_authority_object >( [&]( account_authority_object& auth )
      {
         auth.account = o.get_worker_account();
         auth.owner = authority( 1, o.work.worker, 1);
         auth.active = auth.owner;
         auth.posting = auth.owner;
      });
   }

   const auto& worker_account = db.get_account( o.get_worker_account() ); // verify it exists
   const auto& worker_auth = db.get< account_authority_object, by_account >( o.get_worker_account() );
   FC_ASSERT( worker_auth.active.num_auths() == 1, "Miners can only have one key authority. ${a}", ("a",worker_auth.active) );
   FC_ASSERT( worker_auth.active.key_auths.size() == 1, "Miners may only have one key authority." );
   FC_ASSERT( worker_auth.active.key_auths.begin()->first == o.work.worker, "Work must be performed by key that signed the work." );
   FC_ASSERT( o.block_id == db.head_block_id(), "pow not for last block" );
   if( db.has_hardfork( STEEM_HARDFORK_0_13__256 ) )
      FC_ASSERT( worker_account.last_account_update < db.head_block_time(), "Worker account must not have updated their account this block." );

   fc::sha256 target = db.get_pow_target();

   FC_ASSERT( o.work.work < target, "Work lacks sufficient difficulty." );

   db.modify( dgp, [&]( dynamic_global_property_object& p )
   {
      p.total_pow++; // make sure this doesn't break anything...
      p.num_pow_witnesses++;
   });


   const witness_object* cur_witness = db.find_witness( worker_account.name );
   if( cur_witness ) {
      FC_ASSERT( cur_witness->pow_worker == 0, "This account is already scheduled for pow block production." );
      db.modify(*cur_witness, [&]( witness_object& w ){
          copy_legacy_chain_properties< true >( w.props, o.props );
          w.pow_worker        = dgp.total_pow;
          w.last_work         = o.work.work;
      });
   } else {
      db.create<witness_object>( [&]( witness_object& w )
      {
          w.owner             = o.get_worker_account();
          copy_legacy_chain_properties< true >( w.props, o.props );
          w.signing_key       = o.work.worker;
          w.pow_worker        = dgp.total_pow;
          w.last_work         = o.work.work;
      });
   }
   /// POW reward depends upon whether we are before or after MINER_VOTING kicks in
   asset pow_reward = db.get_pow_reward();
   if( db.head_block_num() < STEEM_START_MINER_VOTING_BLOCK )
      pow_reward.amount *= STEEM_MAX_WITNESSES;
   db.adjust_supply( pow_reward, true );

   /// pay the witness that includes this POW
   const auto& inc_witness = db.get_account( dgp.current_witness );
   if( db.head_block_num() < STEEM_START_MINER_VOTING_BLOCK )
      db.adjust_balance( inc_witness, pow_reward );
   else
      db.create_vesting( inc_witness, pow_reward );
}

void pow_evaluator::do_apply( const pow_operation& o ) {
   FC_ASSERT( !db().has_hardfork( STEEM_HARDFORK_0_13__256 ), "pow is deprecated. Use pow2 instead" );
   pow_apply( db(), o );
}


void pow2_evaluator::do_apply( const pow2_operation& o )
{
   database& db = this->db();
   FC_ASSERT( !db.has_hardfork( STEEM_HARDFORK_0_17__770 ), "mining is now disabled" );

   const auto& dgp = db.get_dynamic_global_properties();
   uint32_t target_pow = db.get_pow_summary_target();
   account_name_type worker_account;

   if( db.has_hardfork( STEEM_HARDFORK_0_16__551 ) )
   {
      const auto& work = o.work.get< equihash_pow >();
      FC_ASSERT( work.prev_block == db.head_block_id(), "Equihash pow op not for last block" );
      auto recent_block_num = protocol::block_header::num_from_id( work.input.prev_block );
      FC_ASSERT( recent_block_num > dgp.last_irreversible_block_num,
         "Equihash pow done for block older than last irreversible block num" );
      FC_ASSERT( work.pow_summary < target_pow, "Insufficient work difficulty. Work: ${w}, Target: ${t}", ("w",work.pow_summary)("t", target_pow) );
      worker_account = work.input.worker_account;
   }
   else
   {
      const auto& work = o.work.get< pow2 >();
      FC_ASSERT( work.input.prev_block == db.head_block_id(), "Work not for last block" );
      FC_ASSERT( work.pow_summary < target_pow, "Insufficient work difficulty. Work: ${w}, Target: ${t}", ("w",work.pow_summary)("t", target_pow) );
      worker_account = work.input.worker_account;
   }

   FC_ASSERT( o.props.maximum_block_size >= STEEM_MIN_BLOCK_SIZE_LIMIT * 2, "Voted maximum block size is too small." );

   db.modify( dgp, [&]( dynamic_global_property_object& p )
   {
      p.total_pow++;
      p.num_pow_witnesses++;
   });

   const auto& accounts_by_name = db.get_index<account_index>().indices().get<by_name>();
   auto itr = accounts_by_name.find( worker_account );
   if(itr == accounts_by_name.end())
   {
      FC_ASSERT( o.new_owner_key.valid(), "New owner key is not valid." );
      const auto& new_account = db.create< account_object >( [&]( account_object& acc )
      {
         initialize_account_object( acc, worker_account, *o.new_owner_key, dgp, true /*mined*/, account_name_type(), _db.get_hardfork() );
         // ^ empty recovery account parameter means highest voted witness at time of recovery
      });

#ifndef IS_LOW_MEM
      db.create< account_metadata_object >( [&]( account_metadata_object& meta )
      {
         meta.account = new_account.id;
      });
#else
      FC_UNUSED( new_account );
#endif

      db.create< account_authority_object >( [&]( account_authority_object& auth )
      {
         auth.account = worker_account;
         auth.owner = authority( 1, *o.new_owner_key, 1);
         auth.active = auth.owner;
         auth.posting = auth.owner;
      });

      db.create<witness_object>( [&]( witness_object& w )
      {
          w.owner             = worker_account;
          copy_legacy_chain_properties< true >( w.props, o.props );
          w.signing_key       = *o.new_owner_key;
          w.pow_worker        = dgp.total_pow;
      });
   }
   else
   {
      FC_ASSERT( !o.new_owner_key.valid(), "Cannot specify an owner key unless creating account." );
      const witness_object* cur_witness = db.find_witness( worker_account );
      FC_ASSERT( cur_witness, "Witness must be created for existing account before mining.");
      FC_ASSERT( cur_witness->pow_worker == 0, "This account is already scheduled for pow block production." );
      db.modify(*cur_witness, [&]( witness_object& w )
      {
          copy_legacy_chain_properties< true >( w.props, o.props );
          w.pow_worker        = dgp.total_pow;
      });
   }

   if( !db.has_hardfork( STEEM_HARDFORK_0_16__551) )
   {
      /// pay the witness that includes this POW
      asset inc_reward = db.get_pow_reward();
      db.adjust_supply( inc_reward, true );

      const auto& inc_witness = db.get_account( dgp.current_witness );
      db.create_vesting( inc_witness, inc_reward );
   }
}

void feed_publish_evaluator::do_apply( const feed_publish_operation& o )
{
   if( _db.has_hardfork( STEEM_HARDFORK_0_20__409 ) )
      FC_ASSERT( is_asset_type( o.exchange_rate.base, SBD_SYMBOL ) && is_asset_type( o.exchange_rate.quote, STEEM_SYMBOL ),
            "Price feed must be a SBD/STEEM price" );

   if ( _db.has_hardfork( STEEM_SMT_HARDFORK ) )
      validate_tick_pricing( o.exchange_rate );

   const auto& witness = _db.get_witness( o.publisher );
   _db.modify( witness, [&]( witness_object& w )
   {
      w.sbd_exchange_rate = o.exchange_rate;
      w.last_sbd_exchange_update = _db.head_block_time();
   });
}

void convert_evaluator::do_apply( const convert_operation& o )
{
  _db.adjust_balance( o.owner, -o.amount );

  const auto& fhistory = _db.get_feed_history();
  FC_ASSERT( !fhistory.current_median_history.is_null(), "Cannot convert SBD because there is no price feed." );

  auto steem_conversion_delay = STEEM_CONVERSION_DELAY_PRE_HF_16;
  if( _db.has_hardfork( STEEM_HARDFORK_0_16__551) )
     steem_conversion_delay = STEEM_CONVERSION_DELAY;

  _db.create<convert_request_object>( [&]( convert_request_object& obj )
  {
      obj.owner           = o.owner;
      obj.requestid       = o.requestid;
      obj.amount          = o.amount;
      obj.conversion_date = _db.head_block_time() + steem_conversion_delay;
  });

}

void limit_order_create_evaluator::do_apply( const limit_order_create_operation& o )
{
   FC_ASSERT( !_db.has_hardfork( STEEM_SMT_HARDFORK ), "The 'limit_order_create' operation is deprecated in favor of `limit_order_create2'." );

   FC_ASSERT( o.expiration > _db.head_block_time(), "Limit order has to expire after head block time." );

   if( _db.has_hardfork( STEEM_HARDFORK_0_20__1449) )
   {
      FC_ASSERT( o.expiration <= _db.head_block_time() + STEEM_MAX_LIMIT_ORDER_EXPIRATION, "Limit Order Expiration must not be more than 28 days in the future" );
   }

   _db.adjust_balance( o.owner, -o.amount_to_sell );

   const auto& order = _db.create<limit_order_object>( [&]( limit_order_object& obj )
   {
       obj.created    = _db.head_block_time();
       obj.seller     = o.owner;
       obj.orderid    = o.orderid;
       obj.for_sale   = o.amount_to_sell.amount;
       obj.sell_price = o.get_price();

       FC_TODO( "Check past order expirations and cleanup after HF 20" )
       if( _db.has_hardfork( STEEM_HARDFORK_0_20__1449 ) )
       {
          obj.expiration = o.expiration;
       }
       else
       {
          uint32_t rand_offset = _db.head_block_id()._hash[4] % 86400;
          obj.expiration = std::min( o.expiration, fc::time_point_sec( STEEM_HARDFORK_0_20_TIME + STEEM_MAX_LIMIT_ORDER_EXPIRATION + rand_offset ) );
       }
   });

   bool filled = _db.apply_order( order );

   if( o.fill_or_kill ) FC_ASSERT( filled, "Cancelling order because it was not filled." );
}

void limit_order_create2_evaluator::do_apply( const limit_order_create2_operation& o )
{
   FC_ASSERT( o.expiration > _db.head_block_time(), "Limit order has to expire after head block time." );

   if( _db.has_hardfork( STEEM_HARDFORK_0_20__1449) )
   {
      FC_ASSERT( o.expiration <= _db.head_block_time() + STEEM_MAX_LIMIT_ORDER_EXPIRATION, "Limit Order Expiration must not be more than 28 days in the future" );
   }

   price exchange_rate = o.exchange_rate;

   if ( _db.has_hardfork( STEEM_SMT_HARDFORK ) )
   {
      validate_tick_pricing( o.get_price() );

      if ( o.exchange_rate.quote.symbol.space() == asset_symbol_type::smt_nai_space )
      {
         const auto* token = _db.find< smt_token_object, by_symbol >( o.exchange_rate.quote.symbol );
         FC_ASSERT( token != nullptr, "Unable to find quote token. Symbol: ${symbol}", ("symbol", o.exchange_rate.quote.symbol) );
      }

      if ( o.amount_to_sell.symbol != o.exchange_rate.base.symbol )
         exchange_rate = ~exchange_rate;

      FC_ASSERT( o.amount_to_sell.symbol == exchange_rate.base.symbol,
         "Sell asset must be in the price. Sell symbol: ${symbol}, Exchange rate: ${exchange}",
            ("symbol", o.amount_to_sell.symbol)("exchange", o.exchange_rate) );
   }
   else
   {
      FC_ASSERT( o.amount_to_sell.symbol == o.exchange_rate.base.symbol, "Sell asset must be the base of the price" );
      FC_ASSERT(
         ( is_asset_type( o.amount_to_sell, STEEM_SYMBOL ) && is_asset_type( o.exchange_rate.quote, SBD_SYMBOL ) ) ||
         ( is_asset_type( o.amount_to_sell, SBD_SYMBOL ) && is_asset_type( o.exchange_rate.quote, STEEM_SYMBOL ) ),
         "Amount to sell symbol cannot be paired the given quote. Sell symbol: ${symbol}, Quote symbol: ${quote}",
            ("symbol", o.amount_to_sell.symbol)("quote", o.exchange_rate.quote.symbol) );
   }

   _db.adjust_balance( o.owner, -o.amount_to_sell );

   const auto& order = _db.create<limit_order_object>( [&]( limit_order_object& obj )
   {
       obj.created    = _db.head_block_time();
       obj.seller     = o.owner;
       obj.orderid    = o.orderid;
       obj.for_sale   = o.amount_to_sell.amount;
       obj.sell_price = exchange_rate;

       FC_TODO( "Check past order expirations and cleanup after HF 20" )
       if( _db.has_hardfork( STEEM_HARDFORK_0_20__1449 ) )
       {
          obj.expiration = o.expiration;
       }
       else
       {
          obj.expiration = std::min( o.expiration, fc::time_point_sec( STEEM_HARDFORK_0_20_TIME + STEEM_MAX_LIMIT_ORDER_EXPIRATION ) );
       }
   });

   bool filled = _db.apply_order( order );

   if( o.fill_or_kill ) FC_ASSERT( filled, "Cancelling order because it was not filled." );
}

void limit_order_cancel_evaluator::do_apply( const limit_order_cancel_operation& o )
{
   _db.cancel_order( _db.get_limit_order( o.owner, o.orderid ) );
}

void report_over_production_evaluator::do_apply( const report_over_production_operation& o )
{
   FC_ASSERT( !_db.has_hardfork( STEEM_HARDFORK_0_4 ), "report_over_production_operation is disabled." );
}

void claim_account_evaluator::do_apply( const claim_account_operation& o )
{
   FC_ASSERT( _db.has_hardfork( STEEM_HARDFORK_0_20__1771 ), "claim_account_operation is not enabled until hardfork 20." );

   const auto& creator = _db.get_account( o.creator );
   const auto& wso = _db.get_witness_schedule_object();

   FC_ASSERT( creator.balance >= o.fee, "Insufficient balance to create account.", ( "creator.balance", creator.balance )( "required", o.fee ) );

   if( o.fee.amount == 0 )
   {
      const auto& gpo = _db.get_dynamic_global_properties();

      // This block is a little weird. We want to enforce that only elected witnesses can include the transaction, but
      // we do not want to prevent the transaction from propogating on the p2p network. Because we do not know what type of
      // witness will have produced the including block when the tx is broadcast, we need to disregard this assertion when the tx
      // is propogating, but require it when applying the block.
      if( !_db.is_pending_tx() )
      {
         const auto& current_witness = _db.get_witness( gpo.current_witness );
         FC_ASSERT( current_witness.schedule == witness_object::elected, "Subsidized accounts can only be claimed by elected witnesses. current_witness:${w} witness_type:${t}",
            ("w",current_witness.owner)("t",current_witness.schedule) );

         FC_ASSERT( current_witness.available_witness_account_subsidies >= STEEM_ACCOUNT_SUBSIDY_PRECISION, "Witness ${w} does not have enough subsidized accounts to claim",
            ("w", current_witness.owner) );

         _db.modify( current_witness, [&]( witness_object& w )
         {
            w.available_witness_account_subsidies -= STEEM_ACCOUNT_SUBSIDY_PRECISION;
         });
      }

      FC_ASSERT( gpo.available_account_subsidies >= STEEM_ACCOUNT_SUBSIDY_PRECISION, "There are not enough subsidized accounts to claim" );

      _db.modify( gpo, [&]( dynamic_global_property_object& gpo )
      {
         gpo.available_account_subsidies -= STEEM_ACCOUNT_SUBSIDY_PRECISION;
      });
   }
   else
   {
      FC_ASSERT( o.fee == wso.median_props.account_creation_fee,
         "Must pay the exact account creation fee. paid: ${p} fee: ${f}",
         ("p", o.fee.amount.value)
         ("f", wso.median_props.account_creation_fee) );
   }

   _db.adjust_balance( _db.get_account( STEEM_NULL_ACCOUNT ), o.fee );

   _db.modify( creator, [&]( account_object& a )
   {
      a.balance -= o.fee;
      a.pending_claimed_accounts++;
   });
}

void create_claimed_account_evaluator::do_apply( const create_claimed_account_operation& o )
{
   FC_ASSERT( _db.has_hardfork( STEEM_HARDFORK_0_20__1771 ), "create_claimed_account_operation is not enabled until hardfork 20." );

   const auto& creator = _db.get_account( o.creator );
   const auto& props = _db.get_dynamic_global_properties();

   FC_ASSERT( creator.pending_claimed_accounts > 0, "${creator} has no claimed accounts to create", ( "creator", o.creator ) );

   verify_authority_accounts_exist( _db, o.owner, o.new_account_name, authority::owner );
   verify_authority_accounts_exist( _db, o.active, o.new_account_name, authority::active );
   verify_authority_accounts_exist( _db, o.posting, o.new_account_name, authority::posting );

   _db.modify( creator, [&]( account_object& a )
   {
      a.pending_claimed_accounts--;
   });

   const auto& new_account = _db.create< account_object >( [&]( account_object& acc )
   {
      initialize_account_object( acc, o.new_account_name, o.memo_key, props, false /*mined*/, o.creator, _db.get_hardfork() );
   });

#ifndef IS_LOW_MEM
   _db.create< account_metadata_object >( [&]( account_metadata_object& meta )
   {
      meta.account = new_account.id;
      from_string( meta.json_metadata, o.json_metadata );
   });
#else
   FC_UNUSED( new_account );
#endif

   _db.create< account_authority_object >( [&]( account_authority_object& auth )
   {
      auth.account = o.new_account_name;
      auth.owner = o.owner;
      auth.active = o.active;
      auth.posting = o.posting;
      auth.last_owner_update = fc::time_point_sec::min();
   });

}

void request_account_recovery_evaluator::do_apply( const request_account_recovery_operation& o )
{
   const auto& account_to_recover = _db.get_account( o.account_to_recover );

   if ( account_to_recover.recovery_account.length() )   // Make sure recovery matches expected recovery account
   {
      FC_ASSERT( account_to_recover.recovery_account == o.recovery_account, "Cannot recover an account that does not have you as there recovery partner." );
      if( o.recovery_account == STEEM_TEMP_ACCOUNT )
         wlog( "Recovery by temp account" );
   }
   else                                                  // Empty string recovery account defaults to top witness
      FC_ASSERT( _db.get_index< witness_index >().indices().get< by_vote_name >().begin()->owner == o.recovery_account, "Top witness must recover an account with no recovery partner." );

   const auto& recovery_request_idx = _db.get_index< account_recovery_request_index >().indices().get< by_account >();
   auto request = recovery_request_idx.find( o.account_to_recover );

   if( request == recovery_request_idx.end() ) // New Request
   {
      FC_ASSERT( !o.new_owner_authority.is_impossible(), "Cannot recover using an impossible authority." );
      FC_ASSERT( o.new_owner_authority.weight_threshold, "Cannot recover using an open authority." );

      if( _db.is_producing() || _db.has_hardfork( STEEM_HARDFORK_0_20 ) )
      {
         validate_auth_size( o.new_owner_authority );
      }

      // Check accounts in the new authority exist
      if( ( _db.has_hardfork( STEEM_HARDFORK_0_15__465 ) ) )
      {
         for( auto& a : o.new_owner_authority.account_auths )
         {
            _db.get_account( a.first );
         }
      }

      _db.create< account_recovery_request_object >( [&]( account_recovery_request_object& req )
      {
         req.account_to_recover = o.account_to_recover;
         req.new_owner_authority = o.new_owner_authority;
         req.expires = _db.head_block_time() + STEEM_ACCOUNT_RECOVERY_REQUEST_EXPIRATION_PERIOD;
      });
   }
   else if( o.new_owner_authority.weight_threshold == 0 ) // Cancel Request if authority is open
   {
      _db.remove( *request );
   }
   else // Change Request
   {
      FC_ASSERT( !o.new_owner_authority.is_impossible(), "Cannot recover using an impossible authority." );

      // Check accounts in the new authority exist
      if( ( _db.has_hardfork( STEEM_HARDFORK_0_15__465 ) ) )
      {
         for( auto& a : o.new_owner_authority.account_auths )
         {
            _db.get_account( a.first );
         }
      }

      _db.modify( *request, [&]( account_recovery_request_object& req )
      {
         req.new_owner_authority = o.new_owner_authority;
         req.expires = _db.head_block_time() + STEEM_ACCOUNT_RECOVERY_REQUEST_EXPIRATION_PERIOD;
      });
   }
}

void recover_account_evaluator::do_apply( const recover_account_operation& o )
{
   const auto& account = _db.get_account( o.account_to_recover );

   if( _db.has_hardfork( STEEM_HARDFORK_0_12 ) )
      FC_ASSERT( _db.head_block_time() - account.last_account_recovery > STEEM_OWNER_UPDATE_LIMIT, "Owner authority can only be updated once an hour." );

   const auto& recovery_request_idx = _db.get_index< account_recovery_request_index >().indices().get< by_account >();
   auto request = recovery_request_idx.find( o.account_to_recover );

   FC_ASSERT( request != recovery_request_idx.end(), "There are no active recovery requests for this account." );
   FC_ASSERT( request->new_owner_authority == o.new_owner_authority, "New owner authority does not match recovery request." );

   const auto& recent_auth_idx = _db.get_index< owner_authority_history_index >().indices().get< by_account >();
   auto hist = recent_auth_idx.lower_bound( o.account_to_recover );
   bool found = false;

   while( hist != recent_auth_idx.end() && hist->account == o.account_to_recover && !found )
   {
      found = hist->previous_owner_authority == o.recent_owner_authority;
      if( found ) break;
      ++hist;
   }

   FC_ASSERT( found, "Recent authority not found in authority history." );

   _db.remove( *request ); // Remove first, update_owner_authority may invalidate iterator
   _db.update_owner_authority( account, o.new_owner_authority );
   _db.modify( account, [&]( account_object& a )
   {
      a.last_account_recovery = _db.head_block_time();
   });
}

void change_recovery_account_evaluator::do_apply( const change_recovery_account_operation& o )
{
   _db.get_account( o.new_recovery_account ); // Simply validate account exists
   const auto& account_to_recover = _db.get_account( o.account_to_recover );

   const auto& change_recovery_idx = _db.get_index< change_recovery_account_request_index >().indices().get< by_account >();
   auto request = change_recovery_idx.find( o.account_to_recover );

   if( request == change_recovery_idx.end() ) // New request
   {
      _db.create< change_recovery_account_request_object >( [&]( change_recovery_account_request_object& req )
      {
         req.account_to_recover = o.account_to_recover;
         req.recovery_account = o.new_recovery_account;
         req.effective_on = _db.head_block_time() + STEEM_OWNER_AUTH_RECOVERY_PERIOD;
      });
   }
   else if( account_to_recover.recovery_account != o.new_recovery_account ) // Change existing request
   {
      _db.modify( *request, [&]( change_recovery_account_request_object& req )
      {
         req.recovery_account = o.new_recovery_account;
         req.effective_on = _db.head_block_time() + STEEM_OWNER_AUTH_RECOVERY_PERIOD;
      });
   }
   else // Request exists and changing back to current recovery account
   {
      _db.remove( *request );
   }
}

void transfer_to_savings_evaluator::do_apply( const transfer_to_savings_operation& op )
{
   const auto& from = _db.get_account( op.from );
   const auto& to   = _db.get_account(op.to);

   if( _db.has_hardfork( STEEM_HARDFORK_0_21__3343 ) )
   {
      FC_ASSERT( op.to != STEEM_TREASURY_ACCOUNT,
         "Cannot transfer savings to ${s}", ("s", STEEM_TREASURY_ACCOUNT) );
   }

   _db.adjust_balance( from, -op.amount );
   _db.adjust_savings_balance( to, op.amount );
}

void transfer_from_savings_evaluator::do_apply( const transfer_from_savings_operation& op )
{
   const auto& from = _db.get_account( op.from );
   _db.get_account(op.to); // Verify to account exists

   FC_ASSERT( from.savings_withdraw_requests < STEEM_SAVINGS_WITHDRAW_REQUEST_LIMIT, "Account has reached limit for pending withdraw requests." );

   if( _db.has_hardfork( STEEM_HARDFORK_0_21__3343 ) )
   {
      FC_ASSERT( op.amount.symbol == SBD_SYMBOL || op.to != STEEM_TREASURY_ACCOUNT,
         "Can only transfer SBD to ${s}", ("s", STEEM_TREASURY_ACCOUNT) );
   }

   FC_ASSERT( _db.get_savings_balance( from, op.amount.symbol ) >= op.amount );
   _db.adjust_savings_balance( from, -op.amount );
   _db.create<savings_withdraw_object>( [&]( savings_withdraw_object& s ) {
      s.from   = op.from;
      s.to     = op.to;
      s.amount = op.amount;
#ifndef IS_LOW_MEM
      from_string( s.memo, op.memo );
#endif
      s.request_id = op.request_id;
      s.complete = _db.head_block_time() + STEEM_SAVINGS_WITHDRAW_TIME;
   });

   _db.modify( from, [&]( account_object& a )
   {
      a.savings_withdraw_requests++;
   });
}

void cancel_transfer_from_savings_evaluator::do_apply( const cancel_transfer_from_savings_operation& op )
{
   const auto& swo = _db.get_savings_withdraw( op.from, op.request_id );
   _db.adjust_savings_balance( _db.get_account( swo.from ), swo.amount );
   _db.remove( swo );

   const auto& from = _db.get_account( op.from );
   _db.modify( from, [&]( account_object& a )
   {
      a.savings_withdraw_requests--;
   });
}

void decline_voting_rights_evaluator::do_apply( const decline_voting_rights_operation& o )
{
   FC_ASSERT( _db.has_hardfork( STEEM_HARDFORK_0_14__324 ) );

   const auto& account = _db.get_account( o.account );
   const auto& request_idx = _db.get_index< decline_voting_rights_request_index >().indices().get< by_account >();
   auto itr = request_idx.find( account.name );

   if( o.decline )
   {
      FC_ASSERT( itr == request_idx.end(), "Cannot create new request because one already exists." );

      _db.create< decline_voting_rights_request_object >( [&]( decline_voting_rights_request_object& req )
      {
         req.account = account.name;
         req.effective_date = _db.head_block_time() + STEEM_OWNER_AUTH_RECOVERY_PERIOD;
      });
   }
   else
   {
      FC_ASSERT( itr != request_idx.end(), "Cannot cancel the request because it does not exist." );
      _db.remove( *itr );
   }
}

void reset_account_evaluator::do_apply( const reset_account_operation& op )
{
   FC_ASSERT( false, "Reset Account Operation is currently disabled." );
/*
   const auto& acnt = _db.get_account( op.account_to_reset );
   auto band = _db.find< account_bandwidth_object, by_account_bandwidth_type >( boost::make_tuple( op.account_to_reset, bandwidth_type::old_forum ) );
   if( band != nullptr )
      FC_ASSERT( ( _db.head_block_time() - band->last_bandwidth_update ) > fc::days(60), "Account must be inactive for 60 days to be eligible for reset" );
   FC_ASSERT( acnt.reset_account == op.reset_account, "Reset account does not match reset account on account." );

   _db.update_owner_authority( acnt, op.new_owner_authority );
*/
}

void set_reset_account_evaluator::do_apply( const set_reset_account_operation& op )
{
   FC_ASSERT( false, "Set Reset Account Operation is currently disabled." );
/*
   const auto& acnt = _db.get_account( op.account );
   _db.get_account( op.reset_account );

   FC_ASSERT( acnt.reset_account == op.current_reset_account, "Current reset account does not match reset account on account." );
   FC_ASSERT( acnt.reset_account != op.reset_account, "Reset account must change" );

   _db.modify( acnt, [&]( account_object& a )
   {
       a.reset_account = op.reset_account;
   });
*/
}

void claim_reward_balance_evaluator::do_apply( const claim_reward_balance_operation& op )
{
   const auto& acnt = _db.get_account( op.account );

   FC_ASSERT( op.reward_steem <= acnt.reward_steem_balance, "Cannot claim that much STEEM. Claim: ${c} Actual: ${a}",
      ("c", op.reward_steem)("a", acnt.reward_steem_balance) );
   FC_ASSERT( op.reward_sbd <= acnt.reward_sbd_balance, "Cannot claim that much SBD. Claim: ${c} Actual: ${a}",
      ("c", op.reward_sbd)("a", acnt.reward_sbd_balance) );
   FC_ASSERT( op.reward_vests <= acnt.reward_vesting_balance, "Cannot claim that much VESTS. Claim: ${c} Actual: ${a}",
      ("c", op.reward_vests)("a", acnt.reward_vesting_balance) );

   asset reward_vesting_steem_to_move = asset( 0, STEEM_SYMBOL );
   if( op.reward_vests == acnt.reward_vesting_balance )
      reward_vesting_steem_to_move = acnt.reward_vesting_steem;
   else
      reward_vesting_steem_to_move = asset( ( ( uint128_t( op.reward_vests.amount.value ) * uint128_t( acnt.reward_vesting_steem.amount.value ) )
         / uint128_t( acnt.reward_vesting_balance.amount.value ) ).to_uint64(), STEEM_SYMBOL );

   _db.adjust_reward_balance( acnt, -op.reward_steem );
   _db.adjust_reward_balance( acnt, -op.reward_sbd );
   _db.adjust_balance( acnt, op.reward_steem );
   _db.adjust_balance( acnt, op.reward_sbd );

   _db.modify( acnt, [&]( account_object& a )
   {
      if( _db.has_hardfork( STEEM_HARDFORK_0_20__2539 ) )
      {
         util::update_manabar( _db.get_dynamic_global_properties(), a, STEEM_VOTING_MANA_REGENERATION_SECONDS, _db.has_hardfork( STEEM_HARDFORK_0_21__3336 ), op.reward_vests.amount.value );
      }

      a.vesting_shares += op.reward_vests;
      a.reward_vesting_balance -= op.reward_vests;
      a.reward_vesting_steem -= reward_vesting_steem_to_move;
   });

   _db.modify( _db.get_dynamic_global_properties(), [&]( dynamic_global_property_object& gpo )
   {
      gpo.total_vesting_shares += op.reward_vests;
      gpo.total_vesting_fund_steem += reward_vesting_steem_to_move;

      gpo.pending_rewarded_vesting_shares -= op.reward_vests;
      gpo.pending_rewarded_vesting_steem -= reward_vesting_steem_to_move;
   });

   _db.adjust_proxied_witness_votes( acnt, op.reward_vests.amount );
}

void claim_reward_balance2_evaluator::do_apply( const claim_reward_balance2_operation& op )
{
   FC_ASSERT( _db.has_hardfork( STEEM_SMT_HARDFORK ), "claim_reward_balance2_operation is not enabled until hardfork ${hf}", ("hf", STEEM_SMT_HARDFORK) );
   for( const asset& token : op.reward_tokens )
   {
      if( token.amount == 0 )
         continue;

      if( token.symbol.space() == asset_symbol_type::smt_nai_space )
      {
         const auto* bal = _db.find< account_rewards_balance_object, by_name_liquid_symbol >( boost::make_tuple( op.account, token.symbol.get_liquid_symbol() ) );

         FC_ASSERT( bal != nullptr, "Account ${a} has no rewards balance for SMT ${s}", ("a", op.account)("s", token.symbol) );
         if( token.symbol.is_vesting() )
         {
            FC_ASSERT( token <= bal->pending_liquid, "Account only has ${p} pending in their rewards balance. Claimed: ${c}",
               ("p", bal->pending_vesting_shares)("c", token) );

            asset reward_vesting_token_to_move = asset( 0, token.symbol.get_vesting_symbol() );

            if( token == bal->pending_vesting_shares )
               reward_vesting_token_to_move = bal->pending_vesting_value;
            else
               reward_vesting_token_to_move = asset( ( ( uint128_t( token.amount.value ) * uint128_t( bal->pending_vesting_value.amount.value ) )
                  / uint128_t( bal->pending_vesting_shares.amount.value ) ).to_uint64(), token.symbol.get_vesting_symbol() );

            _db.modify( *bal, [&]( account_rewards_balance_object& arbo )
            {
               arbo.pending_vesting_shares -= token;
               arbo.pending_vesting_value -= reward_vesting_token_to_move;
            });

            _db.modify( _db.get< smt_token_object, by_symbol >( token.symbol.get_vesting_symbol() ), [&]( smt_token_object& smt )
            {
               smt.total_vesting_shares += token.amount;
               smt.total_vesting_fund_smt += reward_vesting_token_to_move.amount;
               smt.pending_rewarded_vesting_shares -= token.amount;
               smt.pending_rewarded_vesting_smt -= reward_vesting_token_to_move.amount;
            });

         }
         else
         {
            FC_ASSERT( token <= bal->pending_liquid, "Account only has ${p} pending in their rewards balance. Claimed: ${c}",
               ("p", bal->pending_liquid)("c", token) );

            _db.modify( *bal, [&]( account_rewards_balance_object& arbo )
            {
               arbo.pending_liquid -= token;
            });
         }

         _db.adjust_balance( op.account, token );
      }
      else
      {
         const account_object& a = _db.get_account( op.account );

         if( token.symbol == VESTS_SYMBOL)
         {
            FC_ASSERT( token <= a.reward_vesting_balance, "Cannot claim that much VESTS. Claim: ${c} Actual: ${a}",
               ("c", token)("a", a.reward_vesting_balance) );

            asset reward_vesting_steem_to_move = asset( 0, STEEM_SYMBOL );
            if( token == a.reward_vesting_balance )
               reward_vesting_steem_to_move = a.reward_vesting_steem;
            else
               reward_vesting_steem_to_move = asset( ( ( uint128_t( token.amount.value ) * uint128_t( a.reward_vesting_steem.amount.value ) )
                  / uint128_t( a.reward_vesting_balance.amount.value ) ).to_uint64(), STEEM_SYMBOL );

            _db.modify( a, [&]( account_object& a )
            {
               a.vesting_shares += token;
               a.reward_vesting_balance -= token;
               a.reward_vesting_steem -= reward_vesting_steem_to_move;
            });

            _db.modify( _db.get_dynamic_global_properties(), [&]( dynamic_global_property_object& gpo )
            {
               gpo.total_vesting_shares += token;
               gpo.total_vesting_fund_steem += reward_vesting_steem_to_move;

               gpo.pending_rewarded_vesting_shares -= token;
               gpo.pending_rewarded_vesting_steem -= reward_vesting_steem_to_move;
            });

            _db.adjust_proxied_witness_votes( a, token.amount );
         }
         else if( token.symbol == STEEM_SYMBOL || token.symbol == SBD_SYMBOL )
         {
            FC_ASSERT( is_asset_type( token, STEEM_SYMBOL ) == false || token <= a.reward_steem_balance,
                       "Cannot claim that much STEEM. Claim: ${c} Actual: ${a}", ("c", token)("a", a.reward_steem_balance) );
            FC_ASSERT( is_asset_type( token, SBD_SYMBOL ) == false || token <= a.reward_sbd_balance,
                       "Cannot claim that much SBD. Claim: ${c} Actual: ${a}", ("c", token)("a", a.reward_sbd_balance) );
            _db.adjust_reward_balance( a, -token );
            _db.adjust_balance( a, token );
         }
         else
            FC_ASSERT( false, "Unknown asset symbol" );
      } // non-SMT token
   } // for( const auto& token : op.reward_tokens )
}

template < class AccountType >
void generic_delegate_vesting_shares_evaluator(
   database& _db,
   const AccountType& delegator,
   const AccountType& delegatee,
   const asset& vesting_shares,
   uint32_t vote_regeneration_period_seconds  )
{
   auto delegation = _db.find< vesting_delegation_object, by_delegation >( boost::make_tuple( delegator.name, delegatee.name, vesting_shares.symbol.get_paired_symbol() ) );

   const auto& gpo = _db.get_dynamic_global_properties();

   asset available_shares;
   asset available_downvote_shares;

   if( _db.has_hardfork( STEEM_HARDFORK_0_20__2539 ) )
   {
      auto max_mana = util::get_effective_vesting_shares( delegator );

      _db.modify( delegator, [&]( AccountType& a )
      {
         util::update_manabar( gpo, a, vote_regeneration_period_seconds, _db.has_hardfork( STEEM_HARDFORK_0_21__3336 ) );
      });

      available_shares = asset( delegator.voting_manabar.current_mana, vesting_shares.symbol );
      if( gpo.downvote_pool_percent )
      {
         if( _db.has_hardfork( STEEM_HARDFORK_0_22__3485 ) )
         {
            available_downvote_shares = asset(
               ( ( uint128_t( delegator.downvote_manabar.current_mana ) * STEEM_100_PERCENT ) / gpo.downvote_pool_percent
               + ( STEEM_100_PERCENT / gpo.downvote_pool_percent ) - 1 ).to_int64(), vesting_shares.symbol );
         }
         else
         {
            available_downvote_shares = asset(
               ( delegator.downvote_manabar.current_mana * STEEM_100_PERCENT ) / gpo.downvote_pool_percent
               + ( STEEM_100_PERCENT / gpo.downvote_pool_percent ) - 1, vesting_shares.symbol );
         }
      }
      else
      {
         available_downvote_shares = available_shares;
      }

      // Assume delegated VESTS are used first when consuming mana. You cannot delegate received vesting shares
      available_shares.amount = std::min( available_shares.amount, max_mana - delegator.received_vesting_shares.amount );
      available_downvote_shares.amount = std::min( available_downvote_shares.amount, max_mana - delegator.received_vesting_shares.amount );

      if( delegator.next_vesting_withdrawal < fc::time_point_sec::maximum()
         && delegator.to_withdraw - delegator.withdrawn > delegator.vesting_withdraw_rate.amount )
      {
         /*
         current voting mana does not include the current week's power down:

         std::min(
            account.vesting_withdraw_rate.amount.value,           // Weekly amount
            account.to_withdraw.value - account.withdrawn.value   // Or remainder
            );

         But an account cannot delegate **any** VESTS that they are powering down.
         The remaining withdrawal needs to be added in but then the current week is double counted.
         */

         auto weekly_withdraw = asset( std::min(
            delegator.vesting_withdraw_rate.amount.value,           // Weekly amount
            delegator.to_withdraw.value - delegator.withdrawn.value   // Or remainder
            ), vesting_shares.symbol );

         available_shares += weekly_withdraw - asset( delegator.to_withdraw - delegator.withdrawn, vesting_shares.symbol );
         available_downvote_shares += weekly_withdraw - asset( delegator.to_withdraw - delegator.withdrawn, vesting_shares.symbol );
      }
   }
   else
   {
      available_shares = delegator.vesting_shares - delegator.delegated_vesting_shares - asset( delegator.to_withdraw - delegator.withdrawn, vesting_shares.symbol );
   }

   // HF 20 increase fee meaning by 30x, reduce these thresholds to compensate.
   asset min_delegation = asset( 0, vesting_shares.symbol );
   asset min_update     = asset( 0, vesting_shares.symbol );
   if ( vesting_shares.symbol == VESTS_SYMBOL )
   {
      const auto& wso = _db.get_witness_schedule_object();
      min_delegation = _db.has_hardfork( STEEM_HARDFORK_0_20__1761 ) ?
         asset( wso.median_props.account_creation_fee.amount / 3, STEEM_SYMBOL ) * gpo.get_vesting_share_price() :
         asset( wso.median_props.account_creation_fee.amount * 10, STEEM_SYMBOL ) * gpo.get_vesting_share_price();
      min_update = _db.has_hardfork( STEEM_HARDFORK_0_20__1761 ) ?
         asset( wso.median_props.account_creation_fee.amount / 30, STEEM_SYMBOL ) * gpo.get_vesting_share_price() :
         wso.median_props.account_creation_fee * gpo.get_vesting_share_price();
   }

   // If delegation doesn't exist, create it
   if( delegation == nullptr )
   {
      FC_ASSERT( available_shares >= vesting_shares, "Account ${acc} does not have enough mana to delegate. required: ${r} available: ${a}",
         ("acc", delegator.name)("r", vesting_shares)("a", available_shares) );
      FC_TODO( "This hardfork check should not be needed. Remove after HF21 if that is the case." );
      if( _db.has_hardfork( STEEM_HARDFORK_0_21__3336 ) )
         FC_ASSERT( available_downvote_shares >= vesting_shares, "Account ${acc} does not have enough downvote mana to delegate. required: ${r} available: ${a}",
         ("acc", delegator.name)("r", vesting_shares)("a", available_downvote_shares) );
      FC_ASSERT( vesting_shares >= min_delegation, "Account must delegate a minimum of ${v}", ("v", min_delegation) );

      _db.create< vesting_delegation_object >( [&]( vesting_delegation_object& obj )
      {
         obj.delegator = delegator.name;
         obj.delegatee = delegatee.name;
         obj.vesting_shares = vesting_shares;
         obj.min_delegation_time = _db.head_block_time();
      });

      _db.modify( delegator, [&]( AccountType& a )
      {
         a.delegated_vesting_shares += vesting_shares;

         if( _db.has_hardfork( STEEM_HARDFORK_0_20__2539 ) )
         {
            a.voting_manabar.use_mana( vesting_shares.amount.value );

            if( _db.has_hardfork( STEEM_HARDFORK_0_22__3485 ) )
            {
               a.downvote_manabar.use_mana( ( ( uint128_t( vesting_shares.amount.value ) * gpo.downvote_pool_percent ) / STEEM_100_PERCENT ).to_int64() );
            }
            else if( _db.has_hardfork( STEEM_HARDFORK_0_21__3336 ) )
            {
               a.downvote_manabar.use_mana( vesting_shares.amount.value );
            }
         }
      });

      _db.modify( delegatee, [&]( AccountType& a )
      {
         if( _db.has_hardfork( STEEM_HARDFORK_0_20__2539 ) )
         {
            util::update_manabar( gpo, a, vote_regeneration_period_seconds, _db.has_hardfork( STEEM_HARDFORK_0_21__3336 ), vesting_shares.amount.value );
         }

         a.received_vesting_shares += vesting_shares;
      });
   }
   // Else if the delegation is increasing
   else if( vesting_shares >= delegation->vesting_shares )
   {
      auto delta = vesting_shares - delegation->vesting_shares;

      FC_ASSERT( delta >= min_update, "Steem Power increase is not enough of a difference. min_update: ${min}", ("min", min_update) );
      FC_ASSERT( available_shares >= delta, "Account ${acc} does not have enough mana to delegate. required: ${r} available: ${a}",
         ("acc", delegator.name)("r", delta)("a", available_shares) );
      FC_TODO( "This hardfork check should not be needed. Remove after HF21 if that is the case." );
      if( _db.has_hardfork( STEEM_HARDFORK_0_21__3336 ) )
         FC_ASSERT( available_downvote_shares >= delta, "Account ${acc} does not have enough downvote mana to delegate. required: ${r} available: ${a}",
         ("acc", delegator.name)("r", delta)("a", available_downvote_shares) );

      _db.modify( delegator, [&]( AccountType& a )
      {
         a.delegated_vesting_shares += delta;

         if( _db.has_hardfork( STEEM_HARDFORK_0_20__2539 ) )
         {
            a.voting_manabar.use_mana( delta.amount.value );

            if( _db.has_hardfork( STEEM_HARDFORK_0_22__3485 ) )
            {
               a.downvote_manabar.use_mana( ( ( uint128_t( delta.amount.value ) * gpo.downvote_pool_percent ) / STEEM_100_PERCENT ).to_int64() );
            }
            else if( _db.has_hardfork( STEEM_HARDFORK_0_21__3336 ) )
            {
               a.downvote_manabar.use_mana( delta.amount.value );
            }
         }
      });

      _db.modify( delegatee, [&]( AccountType& a )
      {
         if( _db.has_hardfork( STEEM_HARDFORK_0_20__2539 ) )
         {
            util::update_manabar( gpo, a, vote_regeneration_period_seconds, _db.has_hardfork( STEEM_HARDFORK_0_21__3336 ), delta.amount.value );
         }

         a.received_vesting_shares += delta;
      });

      _db.modify( *delegation, [&]( vesting_delegation_object& obj )
      {
         obj.vesting_shares = vesting_shares;
      });
   }
   // Else the delegation is decreasing
   else /* delegation->vesting_shares > op.vesting_shares */
   {
      auto delta = delegation->vesting_shares - vesting_shares;

      if( vesting_shares.amount > 0 )
      {
         FC_ASSERT( delta >= min_update, "Steem Power decrease is not enough of a difference. min_update: ${min}", ("min", min_update) );
         FC_ASSERT( vesting_shares >= min_delegation, "Delegation must be removed or leave minimum delegation amount of ${v}", ("v", min_delegation) );
      }
      else
      {
         FC_ASSERT( delegation->vesting_shares.amount > 0, "Delegation would set vesting_shares to zero, but it is already zero");
      }

      _db.create< vesting_delegation_expiration_object >( [&]( vesting_delegation_expiration_object& obj )
      {
         obj.delegator = delegator.name;
         obj.vesting_shares = delta;
         obj.expiration = std::max( _db.head_block_time() + gpo.delegation_return_period, delegation->min_delegation_time );
      });

      _db.modify( delegatee, [&]( AccountType& a )
      {
         if( _db.has_hardfork( STEEM_HARDFORK_0_22__3485 ) )
         {
            util::update_manabar( gpo, a, vote_regeneration_period_seconds, _db.has_hardfork( STEEM_HARDFORK_0_21__3336 ) );
         }

         a.received_vesting_shares -= delta;

         if( _db.has_hardfork( STEEM_HARDFORK_0_20__2539 ) )
         {
            a.voting_manabar.use_mana( delta.amount.value );

            if( a.voting_manabar.current_mana < 0 )
            {
               a.voting_manabar.current_mana = 0;
            }

            if( _db.has_hardfork( STEEM_HARDFORK_0_21__3336 ) )
            {
               if( _db.has_hardfork( STEEM_HARDFORK_0_22__3485 ) )
               {
                  a.downvote_manabar.use_mana( ( ( uint128_t( delta.amount.value ) * gpo.downvote_pool_percent ) / STEEM_100_PERCENT ).to_int64() );
               }
               else
               {
                  a.downvote_manabar.use_mana( vesting_shares.amount.value );
               }

               if( a.downvote_manabar.current_mana < 0 )
               {
                  a.downvote_manabar.current_mana = 0;
               }
            }
         }
      });

      if( vesting_shares.amount > 0 )
      {
         _db.modify( *delegation, [&]( vesting_delegation_object& obj )
         {
            obj.vesting_shares = vesting_shares;
         });
      }
      else
      {
         _db.remove( *delegation );
      }
   }
}

void delegate_vesting_shares_evaluator::do_apply( const delegate_vesting_shares_operation& op )
{
   try
   {
      if ( op.vesting_shares.symbol.space() == asset_symbol_type::legacy_space )
      {
         const auto& delegator = _db.get_account( op.delegator );
         const auto& delegatee = _db.get_account( op.delegatee );
         generic_delegate_vesting_shares_evaluator( _db, delegator, delegatee, op.vesting_shares, STEEM_VOTING_MANA_REGENERATION_SECONDS );
      }
      else
      {
         const auto& token = _db.get< smt_token_object, by_symbol >( op.vesting_shares.symbol.get_paired_symbol() );
         const auto* delegator = _db.find< account_regular_balance_object, by_name_liquid_symbol >( boost::make_tuple( op.delegator, token.liquid_symbol ) );
         FC_ASSERT( delegator != nullptr, "Account ${acc} does not have a balance for the given symbol. Symbol: ${sym}", ("acc", op.delegator)("sym", token.liquid_symbol) );

         const auto* delegatee = _db.find< account_regular_balance_object, by_name_liquid_symbol >( boost::make_tuple( op.delegatee, token.liquid_symbol ) );

         if ( delegatee == nullptr )
         {
            delegatee = &_db.create< account_regular_balance_object >( [&]( account_regular_balance_object& account_balance )
            {
               account_balance.initialize_assets( token.liquid_symbol );
               account_balance.name = op.delegatee;
               account_balance.validate();
            } );
         }

         generic_delegate_vesting_shares_evaluator( _db, *delegator, *delegatee, op.vesting_shares, token.vote_regeneration_period_seconds );
      }
   } FC_CAPTURE_AND_RETHROW( (op) )
}

} } // steem::chain
