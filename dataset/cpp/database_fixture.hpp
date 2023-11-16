#pragma once

#include <appbase/application.hpp>
#include <steem/chain/database.hpp>
#include <fc/io/json.hpp>
#include <fc/smart_ref_impl.hpp>

#include <steem/plugins/debug_node/debug_node_plugin.hpp>

#include <steem/utilities/key_conversion.hpp>

#include <steem/plugins/block_api/block_api_plugin.hpp>
#include <steem/plugins/condenser_api/condenser_api_legacy_asset.hpp>
#include <steem/plugins/database_api/database_api_plugin.hpp>

#include <fc/network/http/connection.hpp>
#include <fc/network/ip.hpp>

#include <array>
#include <iostream>

#define INITIAL_TEST_SUPPLY (10000000000ll)
#define SBD_INITIAL_TEST_SUPPLY (300000000ll)

extern uint32_t STEEM_TESTING_GENESIS_TIMESTAMP;

#define PUSH_TX \
   steem::chain::test::_push_transaction

#define PUSH_BLOCK \
   steem::chain::test::_push_block

// See below
#define REQUIRE_OP_VALIDATION_SUCCESS( op, field, value ) \
{ \
   const auto temp = op.field; \
   op.field = value; \
   op.validate(); \
   op.field = temp; \
}
#define REQUIRE_OP_EVALUATION_SUCCESS( op, field, value ) \
{ \
   const auto temp = op.field; \
   op.field = value; \
   trx.operations.back() = op; \
   op.field = temp; \
   db.push_transaction( trx, ~0 ); \
}

/*#define STEEM_REQUIRE_THROW( expr, exc_type )          \
{                                                         \
   std::string req_throw_info = fc::json::to_string(      \
      fc::mutable_variant_object()                        \
      ("source_file", __FILE__)                           \
      ("source_lineno", __LINE__)                         \
      ("expr", #expr)                                     \
      ("exc_type", #exc_type)                             \
      );                                                  \
   if( fc::enable_record_assert_trip )                    \
      std::cout << "STEEM_REQUIRE_THROW begin "        \
         << req_throw_info << std::endl;                  \
   BOOST_REQUIRE_THROW( expr, exc_type );                 \
   if( fc::enable_record_assert_trip )                    \
      std::cout << "STEEM_REQUIRE_THROW end "          \
         << req_throw_info << std::endl;                  \
}*/

#define STEEM_REQUIRE_THROW( expr, exc_type )          \
   BOOST_REQUIRE_THROW( expr, exc_type );

#define STEEM_CHECK_THROW( expr, exc_type )            \
{                                                         \
   std::string req_throw_info = fc::json::to_string(      \
      fc::mutable_variant_object()                        \
      ("source_file", __FILE__)                           \
      ("source_lineno", __LINE__)                         \
      ("expr", #expr)                                     \
      ("exc_type", #exc_type)                             \
      );                                                  \
   if( fc::enable_record_assert_trip )                    \
      std::cout << "STEEM_CHECK_THROW begin "          \
         << req_throw_info << std::endl;                  \
   BOOST_CHECK_THROW( expr, exc_type );                   \
   if( fc::enable_record_assert_trip )                    \
      std::cout << "STEEM_CHECK_THROW end "            \
         << req_throw_info << std::endl;                  \
}

#define REQUIRE_OP_VALIDATION_FAILURE_2( op, field, value, exc_type ) \
{ \
   const auto temp = op.field; \
   op.field = value; \
   STEEM_REQUIRE_THROW( op.validate(), exc_type ); \
   op.field = temp; \
}
#define REQUIRE_OP_VALIDATION_FAILURE( op, field, value ) \
   REQUIRE_OP_VALIDATION_FAILURE_2( op, field, value, fc::exception )

#define REQUIRE_THROW_WITH_VALUE_2(op, field, value, exc_type) \
{ \
   auto bak = op.field; \
   op.field = value; \
   trx.operations.back() = op; \
   op.field = bak; \
   STEEM_REQUIRE_THROW(db.push_transaction(trx, ~0), exc_type); \
}

#define REQUIRE_THROW_WITH_VALUE( op, field, value ) \
   REQUIRE_THROW_WITH_VALUE_2( op, field, value, fc::exception )

///This simply resets v back to its default-constructed value. Requires v to have a working assingment operator and
/// default constructor.
#define RESET(v) v = decltype(v)()
///This allows me to build consecutive test cases. It's pretty ugly, but it works well enough for unit tests.
/// i.e. This allows a test on update_account to begin with the database at the end state of create_account.
#define INVOKE(test) ((struct test*)this)->test_method(); trx.clear()

#define PREP_ACTOR(name) \
   fc::ecc::private_key name ## _private_key = generate_private_key(BOOST_PP_STRINGIZE(name));   \
   fc::ecc::private_key name ## _post_key = generate_private_key(std::string( BOOST_PP_STRINGIZE(name) ) + "_post" ); \
   public_key_type name ## _public_key = name ## _private_key.get_public_key();

#define ACTOR(name) \
   PREP_ACTOR(name) \
   const auto& name = account_create(BOOST_PP_STRINGIZE(name), name ## _public_key, name ## _post_key.get_public_key()); \
   account_id_type name ## _id = name.id; (void)name ## _id;

#define GET_ACTOR(name) \
   fc::ecc::private_key name ## _private_key = generate_private_key(BOOST_PP_STRINGIZE(name)); \
   const account_object& name = get_account(BOOST_PP_STRINGIZE(name)); \
   account_id_type name ## _id = name.id; \
   (void)name ##_id

#define ACTORS_IMPL(r, data, elem) ACTOR(elem)
#define ACTORS(names) BOOST_PP_SEQ_FOR_EACH(ACTORS_IMPL, ~, names) \
   validate_database();

#define SMT_SYMBOL( name, decimal_places, db ) \
   asset_symbol_type name ## _symbol = get_new_smt_symbol( decimal_places, db );

#define ASSET( s ) \
   steem::plugins::condenser_api::legacy_asset::from_string( s ).to_asset()

#define FUND( account_name, amount ) \
   fund( account_name, amount ); \
   generate_block();

// To be incorporated into fund() method if deemed appropriate.
// 'SMT' would be dropped from the name then.
#define FUND_SMT_REWARDS( account_name, amount ) \
   db->adjust_reward_balance( account_name, amount ); \
   db->adjust_supply( amount ); \
   generate_block();

#define OP2TX(OP,TX,KEY) \
TX.operations.push_back( OP ); \
TX.set_expiration( db->head_block_time() + STEEM_MAX_TIME_UNTIL_EXPIRATION ); \
TX.sign( KEY, db->get_chain_id(), fc::ecc::bip_0062 );

#define PUSH_OP(OP,KEY) \
{ \
   signed_transaction tx; \
   OP2TX(OP,tx,KEY) \
   db->push_transaction( tx, 0 ); \
}

#define PUSH_OP_TWICE(OP,KEY) \
{ \
   signed_transaction tx; \
   OP2TX(OP,tx,KEY) \
   db->push_transaction( tx, 0 ); \
   db->push_transaction( tx, database::skip_transaction_dupe_check ); \
}

#define FAIL_WITH_OP(OP,KEY,EXCEPTION) \
{ \
   signed_transaction tx; \
   OP2TX(OP,tx,KEY) \
   STEEM_REQUIRE_THROW( db->push_transaction( tx, 0 ), EXCEPTION ); \
}

namespace steem { namespace chain {

using namespace steem::protocol;

struct database_fixture {
   // the reason we use an app is to exercise the indexes of built-in
   //   plugins
   chain::database* db = nullptr;
   signed_transaction trx;
   public_key_type committee_key;
   account_id_type committee_account;
   fc::ecc::private_key private_key = fc::ecc::private_key::generate();
   fc::ecc::private_key init_account_priv_key = fc::ecc::private_key::regenerate( fc::sha256::hash( string( "init_key" ) ) );
   string debug_key = steem::utilities::key_to_wif( init_account_priv_key );
   public_key_type init_account_pub_key = init_account_priv_key.get_public_key();
   uint32_t default_skip = 0 | database::skip_undo_history_check | database::skip_authority_check;
   fc::ecc::canonical_signature_type default_sig_canon = fc::ecc::fc_canonical;

   plugins::debug_node::debug_node_plugin* db_plugin;

   optional<fc::temp_directory> data_dir;
   bool skip_key_index_test = false;

   database_fixture() {}
   virtual ~database_fixture() { appbase::reset(); }

   using units = flat_map< unit_target_type, uint16_t >;

   static fc::ecc::private_key generate_private_key( string seed = "init_key" );
   static asset_symbol_type get_new_smt_symbol( uint8_t token_decimal_places, chain::database* db );
   void open_database( uint16_t shared_file_size_in_mb = 8 );
   void generate_block(uint32_t skip = 0,
                               const fc::ecc::private_key& key = generate_private_key("init_key"),
                               int miss_blocks = 0);

   /**
    * @brief Generates block_count blocks
    * @param block_count number of blocks to generate
    */
   void generate_blocks(uint32_t block_count);

   /**
    * @brief Generates blocks until the head block time matches or exceeds timestamp
    * @param timestamp target time to generate blocks until
    */
   void generate_blocks(fc::time_point_sec timestamp, bool miss_intermediate_blocks = true);

   const account_object& account_create(
      const string& name,
      const string& creator,
      const private_key_type& creator_key,
      const share_type& fee,
      const public_key_type& key,
      const public_key_type& post_key,
      const string& json_metadata
   );

   const account_object& account_create(
      const string& name,
      const public_key_type& key,
      const public_key_type& post_key
   );

   const account_object& account_create(
      const string& name,
      const public_key_type& key
   );

   const witness_object& witness_create(
      const string& owner,
      const private_key_type& owner_key,
      const string& url,
      const public_key_type& signing_key,
      const share_type& fee
   );

   void fund( const string& account_name, const share_type& amount = 500000 );
   void fund( const string& account_name, const asset& amount );
   void transfer( const string& from, const string& to, const asset& amount );
   void convert( const string& account_name, const asset& amount );
   void vest( const string& from, const string& to, const asset& amount );
   void vest( const string& from, const share_type& amount );
   void vest( const string& from, const asset& amount );
   void proxy( const string& account, const string& proxy );
   void set_price_feed( const price& new_price );
   void set_witness_props( const flat_map< string, vector< char > >& new_props );
   const asset& get_balance( const string& account_name )const;
   void sign( signed_transaction& trx, const fc::ecc::private_key& key );

   vector< operation > get_last_operations( uint32_t ops );

   asset_symbol_type create_smt( const string& account_name, const fc::ecc::private_key& key, uint8_t token_decimal_places );
   asset_symbol_type create_smt_with_nai( const string& account_name, const fc::ecc::private_key& key, uint32_t nai, uint8_t token_decimal_places );

   /// Creates 3 different SMTs for provided control account, one with 0 precision, the other two with the same non-zero precision.
   std::array<asset_symbol_type, 3> create_smt_3(const char* control_account_name, const fc::ecc::private_key& key);
   /// Tries to create SMTs with too big precision or invalid name.
   void create_invalid_smt( const char* control_account_name, const fc::ecc::private_key& key );
   /// Tries to create SMTs matching existing one. First attempt with matching precision, second one with different (but valid) precision.
   void create_conflicting_smt( const asset_symbol_type existing_smt, const char* control_account_name, const fc::ecc::private_key& key );

   //smt_setup_operation
   smt_generation_unit get_generation_unit ( const units& steem_unit = units(), const units& token_unit = units() );
   smt_capped_generation_policy get_capped_generation_policy
   (
      const smt_generation_unit& pre_soft_cap_unit = smt_generation_unit(),
      const smt_generation_unit& post_soft_cap_unit = smt_generation_unit(),
      uint32_t min_unit_ratio = 0,
      uint32_t max_unit_ratio = 0
   );

   void validate_database();
};

struct clean_database_fixture : public database_fixture
{
   clean_database_fixture( uint16_t shared_file_size_in_mb = 8 );
   virtual ~clean_database_fixture();

   void resize_shared_mem( uint64_t size );
   void validate_database();
};

struct live_database_fixture : public database_fixture
{
   live_database_fixture();
   virtual ~live_database_fixture();

   fc::path _chain_dir;
};

struct sps_proposal_database_fixture : public clean_database_fixture
{
   sps_proposal_database_fixture( uint16_t shared_file_size_in_mb = 8 )
                           : clean_database_fixture( shared_file_size_in_mb ){}
   virtual ~sps_proposal_database_fixture(){}

   void plugin_prepare();

   int64_t create_proposal(   std::string creator, std::string receiver,
                              time_point_sec start_date, time_point_sec end_date,
                              asset daily_pay, const fc::ecc::private_key& key );

   void vote_proposal( std::string voter, const std::vector< int64_t >& id_proposals, bool approve, const fc::ecc::private_key& key );

   bool exist_proposal( int64_t id );
   const proposal_object* find_proposal( int64_t id );

   void remove_proposal(account_name_type _deleter, flat_set<int64_t> _proposal_id, const fc::ecc::private_key& _key);

   bool find_vote_for_proposal(const std::string& _user, int64_t _proposal_id);

   uint64_t get_nr_blocks_until_maintenance_block();

   void post_comment( std::string _authro, std::string _permlink, std::string _title, std::string _body, std::string _parent_permlink, const fc::ecc::private_key& _key);
};

struct sps_proposal_database_fixture_performance : public sps_proposal_database_fixture
{
   sps_proposal_database_fixture_performance( uint16_t shared_file_size_in_mb = 512 )
                           : sps_proposal_database_fixture( shared_file_size_in_mb )
   {
      db->get_benchmark_dumper().set_enabled( true );
      db->set_sps_remove_threshold( -1 );
   }
};

struct json_rpc_database_fixture : public database_fixture
{
   private:
      steem::plugins::json_rpc::json_rpc_plugin* rpc_plugin;

      fc::variant get_answer( std::string& request );
      void review_answer( fc::variant& answer, int64_t code, bool is_warning, bool is_fail, fc::optional< fc::variant > id );

   public:

      json_rpc_database_fixture();
      virtual ~json_rpc_database_fixture();

      void make_array_request( std::string& request, int64_t code = 0, bool is_warning = false, bool is_fail = true );
      fc::variant make_request( std::string& request, int64_t code = 0, bool is_warning = false, bool is_fail = true );
      void make_positive_request( std::string& request );
};

namespace test
{
   bool _push_block( database& db, const signed_block& b, uint32_t skip_flags = 0 );
   void _push_transaction( database& db, const signed_transaction& tx, uint32_t skip_flags = 0 );
}

} }
