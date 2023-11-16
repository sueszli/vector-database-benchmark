#pragma once

#include <fc/time.hpp>
#include <fc/variant.hpp>
#include <fc/reflect/variant.hpp>
#include <fc/exception/exception.hpp>
#include <fc/io/json.hpp>

#include <sys/time.h>

#include <utility>

namespace steem { namespace chain { namespace util {

template <typename TCntr>
struct emplace_ret_value
{
   using type = decltype(std::declval<TCntr>().emplace(std::declval<typename TCntr::value_type>()));
};


class advanced_benchmark_dumper
{
   public:

      struct item
      {
         std::string op_name;
         mutable fc::microseconds time;
         mutable uint64_t count = 0;

         item( std::string _op_name, fc::microseconds _time, uint64_t _count ): op_name( _op_name ), time( _time ), count( _count ) {}

         bool operator<( const item& obj ) const { return op_name < obj.op_name; }
         void inc( fc::microseconds _time ) const { time += _time; ++count; }
      };

      struct ritem
      {
         std::string op_name;
         fc::microseconds time;
         uint64_t count = 0;

         ritem( std::string _op_name, fc::microseconds _time, uint64_t _count ): op_name( _op_name ), time( _time ), count( _count ) {}

         bool operator<( const ritem& obj ) const { return time > obj.time; }
      };

      template< typename COLLECTION >
      struct total_info
      {
         fc::microseconds total_time;

         COLLECTION items;

         total_info(){}
         total_info( fc::microseconds _total_time ): total_time( _total_time ) {}

         void inc( fc::microseconds _time ) { total_time += _time; }

         template <typename... TArgs>
         typename emplace_ret_value<COLLECTION>::type emplace(TArgs&&... args)
            { return items.emplace(std::forward<TArgs>(args)...); }
      };

   private:

      static uint32_t cnt;
      static std::string virtual_operation_name;
      static std::string apply_context_name;

      bool enabled = false;

      uint32_t flush_cnt = 0;
      uint32_t flush_max = 500000;

      fc::time_point time_begin;

      std::string file_name;

      total_info< std::set< item > > info;

      template< typename COLLECTION >
      void dump_impl( const total_info< COLLECTION >& src, const std::string& src_file_name );

   public:

      advanced_benchmark_dumper();
      ~advanced_benchmark_dumper();

      static std::string& get_virtual_operation_name(){ return virtual_operation_name; }

      template< bool IS_PRE_OPERATION >
      static std::string generate_desc( const std::string& desc1, const std::string& desc2 )
      {
         std::stringstream s;
         s << ( IS_PRE_OPERATION ? "pre--->" : "post--->" ) << desc1 << "--->" << desc2;

         return s.str();
      }

      void set_enabled( bool val ) { enabled = val; }
      bool is_enabled() { return enabled; }

      void begin();
      template< bool APPLY_CONTEXT = false >
      void end( const std::string& str );

      void dump();
};

} } } // steem::chain::util

FC_REFLECT( steem::chain::util::advanced_benchmark_dumper::item, (op_name)(time)(count) )
FC_REFLECT( steem::chain::util::advanced_benchmark_dumper::ritem, (op_name)(time)(count) )

FC_REFLECT( steem::chain::util::advanced_benchmark_dumper::total_info< std::set< steem::chain::util::advanced_benchmark_dumper::item > >, (total_time)(items) )
FC_REFLECT( steem::chain::util::advanced_benchmark_dumper::total_info< std::multiset< steem::chain::util::advanced_benchmark_dumper::ritem > >, (total_time)(items) )

