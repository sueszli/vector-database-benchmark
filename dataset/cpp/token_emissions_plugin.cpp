#include <steem/chain/steem_fwd.hpp>

#include <steem/plugins/token_emissions/token_emissions_plugin.hpp>
#include <steem/plugins/token_emissions/token_emissions_objects.hpp>
#include <steem/chain/database.hpp>
#include <steem/chain/index.hpp>
#include <steem/chain/smt_objects.hpp>
#include <steem/chain/util/smt_token.hpp>
#include <steem/protocol/config.hpp>

#include <fc/io/json.hpp>
#include <cmath>

namespace steem { namespace plugins { namespace token_emissions {

namespace detail {

class token_emissions_impl
{
public:
   token_emissions_impl() : _db( appbase::app().get_plugin< steem::plugins::chain::chain_plugin >().db() ) {}
   virtual ~token_emissions_impl() {}

   void on_post_apply_required_action( const required_action_notification& note );
   void on_post_apply_optional_action( const optional_action_notification& note );
   void on_generate_optional_actions( const generate_optional_actions_notification& note );

   chain::database&              _db;
   boost::signals2::connection   post_apply_required_action_connection;
   boost::signals2::connection   post_apply_optional_action_connection;
   boost::signals2::connection   generate_optional_action_connection;
};

void token_emissions_impl::on_post_apply_required_action( const required_action_notification& note )
{
   if ( note.action.which() != required_automated_action::tag< smt_token_launch_action >::value )
      return;

   if ( _db.is_pending_tx() )
      return;

   smt_token_launch_action token_launch = note.action.get< smt_token_launch_action >();

   auto next_emission = util::smt::next_emission_time( _db, token_launch.symbol );

   if ( next_emission )
   {
      _db.create< token_emission_schedule_object >( [ token_launch, next_emission ]( token_emission_schedule_object& o )
      {
         o.symbol = token_launch.symbol;
         o.next_consensus_emission = *next_emission;
         o.next_scheduled_emission = *next_emission;
      } );
   }
}

void token_emissions_impl::on_post_apply_optional_action( const optional_action_notification& note )
{
   if ( note.action.which() != optional_automated_action::tag< smt_token_emission_action >::value )
      return;

   if ( _db.is_pending_tx() )
      return;

   smt_token_emission_action emission_action = note.action.get< smt_token_emission_action >();
   auto next = util::smt::next_emission_time( _db, emission_action.symbol, _db.get< smt_token_object, steem::chain::by_symbol >( emission_action.symbol ).last_virtual_emission_time );
   const auto& emission_obj = _db.get< token_emission_schedule_object, by_symbol >( emission_action.symbol );

   if ( next )
   {
      _db.modify( emission_obj, [ next ]( token_emission_schedule_object& o )
      {
         o.next_consensus_emission = *next;
         o.next_scheduled_emission = *next;
      } );
   }
   else
   {
      _db.remove( emission_obj );
   }
}

void token_emissions_impl::on_generate_optional_actions( const generate_optional_actions_notification& note )
{
   time_point_sec next_emission_time = _db.head_block_time();

   const auto& next_emission_schedule_idx = _db.get_index< token_emission_schedule_index, by_next_emission_symbol >();
   const auto& token_emissions_idx = _db.get_index< smt_token_emissions_index, by_symbol_end_time >();

   for ( auto itr = next_emission_schedule_idx.begin(); itr != next_emission_schedule_idx.end() && itr->next_scheduled_emission <= next_emission_time; ++itr )
   {
      auto emission = token_emissions_idx.lower_bound( boost::make_tuple( itr->symbol, itr->next_consensus_emission ) );

      if ( emission != token_emissions_idx.end() && emission->symbol == itr->symbol )
      {
         const auto& token = _db.get< smt_token_object, chain::by_symbol >( itr->symbol );

         auto emissions = util::smt::generate_emissions( token, *emission, itr->next_consensus_emission );

         if ( !emissions.empty() )
         {
            smt_token_emission_action action;
            action.control_account = token.control_account;
            action.symbol          = token.liquid_symbol;
            action.emission_time   = itr->next_consensus_emission;
            action.emissions       = std::move( emissions );

            _db.push_optional_action( action, next_emission_time );
         }

         _db.modify( *itr, [&]( token_emission_schedule_object& o )
         {
            o.next_scheduled_emission += fc::seconds( emission->interval_seconds );
         } );
      }
   }
}

} // detail

token_emissions_plugin::token_emissions_plugin() {}
token_emissions_plugin::~token_emissions_plugin() {}

void token_emissions_plugin::set_program_options( boost::program_options::options_description& cli, boost::program_options::options_description& cfg ) {}

void token_emissions_plugin::plugin_initialize( const boost::program_options::variables_map& options )
{
   try
   {
      ilog( "token_emissions: plugin_initialize() begin" );

      my = std::make_unique< detail::token_emissions_impl >();
      STEEM_ADD_PLUGIN_INDEX(my->_db, token_emission_schedule_index);

      my->post_apply_optional_action_connection = my->_db.add_post_apply_optional_action_handler(
         [&]( const optional_action_notification& note )
         {
            try
            {
               my->on_post_apply_optional_action( note );
            } FC_LOG_AND_RETHROW()
         }, *this, 0 );

      my->generate_optional_action_connection = my->_db.add_generate_optional_actions_handler(
         [&]( const generate_optional_actions_notification& note )
         {
            try
            {
               my->on_generate_optional_actions( note );
            } FC_LOG_AND_RETHROW()
         }, *this, 0 );

      my->post_apply_required_action_connection = my->_db.add_post_apply_required_action_handler(
         [&]( const required_action_notification& note )
         {
            try
            {
               my->on_post_apply_required_action( note );
            } FC_LOG_AND_RETHROW()
         }, *this, 0 );

      ilog( "token_emissions: plugin_initialize() end" );
   } FC_CAPTURE_AND_RETHROW()
}

void token_emissions_plugin::plugin_startup() {}

void token_emissions_plugin::plugin_shutdown()
{
   chain::util::disconnect_signal( my->post_apply_required_action_connection );
   chain::util::disconnect_signal( my->post_apply_optional_action_connection );
   chain::util::disconnect_signal( my->generate_optional_action_connection );
}

} } } // steem::plugins::token_emissions

