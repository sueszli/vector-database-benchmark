// Copyright (c) 2019-2023 Dr. Colin Hirsch and Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/json/

#include <tao/json.hpp>

#include "bench_mark.hpp"

int main( int argc, char** argv )
{
   for( int i = 1; i < argc; ++i ) {
      tao::json::events::to_value consumer;
      tao::json::events::from_file( consumer, argv[ i ] );

      tao::bench::mark( "json", argv[ i ], [ & ]() {
         (void)tao::json::to_string( consumer.value, 4 );
      } );
   }
   return 0;
}
