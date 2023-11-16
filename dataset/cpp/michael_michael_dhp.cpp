/*
    This file is a part of libcds - Concurrent Data Structures library

    (C) Copyright Maxim Khizhinsky (libcds.dev@gmail.com) 2006-2016

    Source code repo: http://github.com/khizmax/libcds/
    Download: http://sourceforge.net/projects/libcds/files/
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.     
*/

#include "test_set_hp.h"

#include <cds/container/michael_list_dhp.h>
#include <cds/container/michael_set.h>

namespace {
    namespace cc = cds::container;
    typedef cds::gc::DHP gc_type;

    class MichaelSet_DHP : public cds_test::container_set_hp
    {
    protected:
        typedef cds_test::container_set_hp base_class;

        void SetUp()
        {
            typedef cc::MichaelList< gc_type, int_item > list_type;
            typedef cc::MichaelHashSet< gc_type, list_type >   set_type;

            cds::gc::dhp::GarbageCollector::Construct( 16, set_type::c_nHazardPtrCount );
            cds::threading::Manager::attachThread();
        }

        void TearDown()
        {
            cds::threading::Manager::detachThread();
            cds::gc::dhp::GarbageCollector::Destruct();
        }
    };

    TEST_F( MichaelSet_DHP, compare )
    {
        typedef cc::MichaelList< gc_type, int_item,
            typename cc::michael_list::make_traits<
                cds::opt::compare< cmp >
            >::type
        > list_type;

        typedef cc::MichaelHashSet< gc_type, list_type, 
            typename cc::michael_set::make_traits<
                cds::opt::hash< hash_int >
            >::type
        > set_type;

        set_type s( kSize, 2 );
        test( s );
    }

    TEST_F( MichaelSet_DHP, less )
    {
        typedef cc::MichaelList< gc_type, int_item,
            typename cc::michael_list::make_traits<
                cds::opt::less< base_class::less >
            >::type
        > list_type;

        typedef cc::MichaelHashSet< gc_type, list_type, 
            typename cc::michael_set::make_traits<
                cds::opt::hash< hash_int >
            >::type
        > set_type;

        set_type s( kSize, 2 );
        test( s );
    }

    TEST_F( MichaelSet_DHP, cmpmix )
    {
        struct list_traits : public cc::michael_list::traits
        {
            typedef base_class::less less;
            typedef cmp compare;
        };
        typedef cc::MichaelList< gc_type, int_item, list_traits > list_type;

        typedef cc::MichaelHashSet< gc_type, list_type, 
            typename cc::michael_set::make_traits<
                cds::opt::hash< hash_int >
            >::type
        > set_type;

        set_type s( kSize, 2 );
        test( s );
    }

    TEST_F( MichaelSet_DHP, item_counting )
    {
        struct list_traits : public cc::michael_list::traits
        {
            typedef cmp compare;
        };
        typedef cc::MichaelList< gc_type, int_item, list_traits > list_type;

        struct set_traits: public cc::michael_set::traits
        {
            typedef hash_int hash;
            typedef simple_item_counter item_counter;
        };
        typedef cc::MichaelHashSet< gc_type, list_type, set_traits >set_type;

        set_type s( kSize, 3 );
        test( s );
    }

    TEST_F( MichaelSet_DHP, backoff )
    {
        struct list_traits : public cc::michael_list::traits
        {
            typedef cmp compare;
            typedef cds::backoff::exponential<cds::backoff::pause, cds::backoff::yield> back_off;
        };
        typedef cc::MichaelList< gc_type, int_item, list_traits > list_type;

        struct set_traits : public cc::michael_set::traits
        {
            typedef hash_int hash;
            typedef cds::atomicity::item_counter item_counter;
        };
        typedef cc::MichaelHashSet< gc_type, list_type, set_traits >set_type;

        set_type s( kSize, 4 );
        test( s );
    }

    TEST_F( MichaelSet_DHP, seq_cst )
    {
        struct list_traits : public cc::michael_list::traits
        {
            typedef base_class::less less;
            typedef cds::backoff::pause back_off;
            typedef cds::opt::v::sequential_consistent memory_model;
        };
        typedef cc::MichaelList< gc_type, int_item, list_traits > list_type;

        struct set_traits : public cc::michael_set::traits
        {
            typedef hash_int hash;
            typedef cds::atomicity::item_counter item_counter;
        };
        typedef cc::MichaelHashSet< gc_type, list_type, set_traits >set_type;

        set_type s( kSize, 4 );
        test( s );
    }

} // namespace
