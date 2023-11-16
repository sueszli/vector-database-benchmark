#include <steem/chain/steem_fwd.hpp>
#include <steem/chain/database.hpp>
#include <steem/protocol/asset_symbol.hpp>
#include <steem/chain/smt_objects.hpp>
#include <steem/chain/util/nai_generator.hpp>
#include <steem/chain/util/smt_token.hpp>

#define NAI_GENERATION_SEED_BLOCK_ID_HASH_INDEX 4

namespace steem { namespace chain {

static void refill_nais(
   database& db,
   const block_id_type& last_block_id,
   uint32_t& collisions_per_block,
   uint32_t& attempts_per_block,
   uint8_t& num_available_nais,
   nai_pool_object::pool_type& nais );

/**
 * Refill the NAI pool with newly generated values
 *
 * We will attempt to fill the pool with randomly generated NAIs up until
 * the pool is full (SMT_MAX_NAI_POOL_COUNT) or we encounter the maximum
 * acceptable collisions (SMT_MAX_NAI_GENERATION_TRIES). If we hit the
 * maximum acceptable collisions, the pool will once again attempt to
 * replenish when called during the next block.
 */
void replenish_nai_pool( database& db )
{
   try
   {
      const nai_pool_object& npo = db.get< nai_pool_object >();
      auto head_block_id = db.head_block_id();

      block_id_type last_block_id     = npo.last_block_id;
      uint32_t collisions_per_block   = npo.collisions_per_block;
      uint32_t attempts_per_block     = npo.attempts_per_block;
      uint8_t num_available_nais      = npo.num_available_nais;
      nai_pool_object::pool_type nais = npo.nais;

      // If this is the first time we're encountering this block, reset our variables and set the last block */
      if ( last_block_id != head_block_id )
      {
         last_block_id = head_block_id;
         collisions_per_block = 0;
         attempts_per_block = 0;
      }

      /*
       * No reason to attempt NAI generation or inform the user again if we have already
       * reached the maximum acceptable collisions for this particular block.
       */
      if ( collisions_per_block < SMT_MAX_NAI_GENERATION_TRIES )
      {
         refill_nais( db, last_block_id, collisions_per_block, attempts_per_block, num_available_nais, nais );

         db.modify( npo, [&]( nai_pool_object& obj )
         {
            obj.last_block_id = last_block_id;
            obj.collisions_per_block = collisions_per_block;
            obj.attempts_per_block = attempts_per_block;
            obj.nais = nais;
            obj.num_available_nais = num_available_nais;
         });
      }
   }
   FC_CAPTURE_AND_RETHROW()
}

static void refill_nais(
   database& db,
   const block_id_type& last_block_id,
   uint32_t& collisions_per_block,
   uint32_t& attempts_per_block,
   uint8_t& num_available_nais,
   nai_pool_object::pool_type& nais )
{
   while ( num_available_nais < SMT_MAX_NAI_POOL_COUNT )
   {
      asset_symbol_type next_sym;
      for (;;)
      {
         if ( collisions_per_block >= SMT_MAX_NAI_GENERATION_TRIES )
         {
            ilog( "Encountered ${collisions} collisions while attempting to generate NAI, generation will resume at the next block",
               ("collisions", collisions_per_block) );
            return;
         }

         next_sym = util::nai_generator::generate( last_block_id._hash[ NAI_GENERATION_SEED_BLOCK_ID_HASH_INDEX ] + attempts_per_block );

         attempts_per_block++;

         // We must ensure the NAI is not an SMT, and it is not already contained within the NAI pool
         if ( !util::smt::find_token( db, next_sym, true ) && std::find( std::begin( nais ), std::begin( nais ) + num_available_nais, next_sym ) != std::end( nais ) )
            break;

         collisions_per_block++;
      }

      nais[ num_available_nais ] = next_sym;
      num_available_nais++;
   }
}

void remove_from_nai_pool( database &db, const asset_symbol_type& a )
{
   const nai_pool_object& npo = db.get< nai_pool_object >();
   const auto& nais = npo.nais;
   const auto end = nais.begin() + npo.num_available_nais;
   auto it = std::find( nais.begin(), end, asset_symbol_type::from_asset_num( a.get_stripped_precision_smt_num() ) );
   if ( it != end )
   {
      auto index = std::distance( nais.begin(), it );

      db.modify( npo, [&] ( nai_pool_object& obj )
      {
         obj.nais[ index ] = asset_symbol_type();
         std::swap( obj.nais[ index ], obj.nais[ obj.num_available_nais - 1 ] );
         obj.num_available_nais--;
      });
   }
}

} } // steem::chain

