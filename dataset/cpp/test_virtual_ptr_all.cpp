// Copyright (c) 2021 Jean-Louis Leroy
// Distributed under the Boost Software License, Version 1.0.
// See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <yorel/yomm2/keywords.hpp>
#include <yorel/yomm2/runtime.hpp>
#include <yorel/yomm2/templates.hpp>

#include "test_helpers.hpp"

#define BOOST_TEST_MODULE yomm2
#include <boost/test/included/unit_test.hpp>
#include <boost/utility/identity_type.hpp>

using namespace yorel::yomm2;

namespace yorel {
namespace yomm2 {
struct direct {};
struct indirect {};
} // namespace yomm2
} // namespace yorel

struct Player {
    virtual ~Player() {
    }
};

struct Warrior : Player {};
struct Wizard : Player {};

struct Bear : Player {};

struct Object {
    virtual ~Object() {
    }
};

struct Axe : Object {};

template<class VirtualBearPtr>
auto kick_bear(VirtualBearPtr) {
    return std::string("growl");
}

template<class VirtualWarriorPtr, class VirtualAxePtr, class VirtualBearPtr>
auto fight_bear(VirtualWarriorPtr, VirtualAxePtr, VirtualBearPtr) {
    return "kill bear";
}

template<int Key>
struct indirect_test_policy : test_policy_<Key> {
    static constexpr bool use_indirect_method_pointers = true;
};

template<int Key>
using policy_types = types<test_policy_<Key>, indirect_test_policy<Key>>;

namespace YOMM2_GENSYM {

BOOST_AUTO_TEST_CASE_TEMPLATE(
    test_virtual_ptr, Policy, policy_types<__COUNTER__>) {
    using namespace detail;

    static use_classes<Policy, Player, Warrior, Object, Axe, Bear> YOMM2_GENSYM;
    using kick = method<void, std::string(virtual_ptr<Player, Policy>), Policy>;
    static typename kick::template add_function<
        kick_bear<virtual_ptr<Player, Policy>>>
        YOMM2_GENSYM;

    update<Policy>();

    using vptr_player = virtual_ptr<Player, Policy>;
    static_assert(detail::is_virtual_ptr<vptr_player>);
    using vptr_cat = virtual_ptr<Bear, Policy>;

    Player player;
    auto virtual_player = vptr_player::final(player);
    BOOST_TEST(&*virtual_player == &player);
    BOOST_TEST((virtual_player.method_table() == Policy::template method_table<Player>));

    Bear bear;
    BOOST_TEST((&*vptr_cat::final(bear)) == &bear);
    BOOST_TEST(
        (vptr_cat::final(bear).method_table() == Policy::template method_table<Bear>));

    BOOST_TEST(
        (vptr_player(bear).method_table() == Policy::template method_table<Bear>));

    vptr_cat virtual_cat_ptr(bear);
    vptr_player virtual_player_ptr = virtual_cat_ptr;

    struct upcast {
        static void fn(vptr_player) {
        }
    };

    upcast::fn(virtual_cat_ptr);

    auto data = Policy::context.gv.data();

    while (data == Policy::context.gv.data()) {
        Policy::context.gv.resize(2 * Policy::context.gv.size());
    }

    update<Policy>();

    BOOST_TEST(
        (virtual_cat_ptr.method_table() == Policy::template method_table<Bear>) ==
        Policy::use_indirect_method_pointers);
}
} // namespace test_virtual_ptr

namespace test_virtual_ptr_dispatch {

BOOST_AUTO_TEST_CASE_TEMPLATE(
    test_virtual_ptr_dispatch, Policy, policy_types<__COUNTER__>) {

    static use_classes<Policy, Player, Warrior, Object, Axe, Bear> YOMM2_GENSYM;

    using kick = method<void, std::string(virtual_ptr<Player, Policy>), Policy>;
    static typename kick::template add_function<
        kick_bear<virtual_ptr<Player, Policy>>>
        YOMM2_GENSYM;

    using fight = method<
        void,
        std::string(
            virtual_ptr<Player, Policy>, virtual_ptr<Object, Policy>,
            virtual_ptr<Player, Policy>),
        Policy>;
    static typename fight::template add_function<fight_bear<
        virtual_ptr<Player, Policy>, virtual_ptr<Object, Policy>,
        virtual_ptr<Player, Policy>>>
        YOMM2_GENSYM;

    update<Policy>();

    Bear bear;
    BOOST_TEST(kick::fn(virtual_ptr<Player, Policy>(bear)) == "growl");

    Warrior warrior;
    Axe axe;
    BOOST_TEST(
        fight::fn(
            virtual_ptr<Player, Policy>(warrior),
            virtual_ptr<Object, Policy>(axe),
            virtual_ptr<Player, Policy>(bear)) == "kill bear");
}

} // namespace test_virtual_ptr_dispatch

namespace test_virtual_shared_ptr {

BOOST_AUTO_TEST_CASE(test_virtual_shared_ptr) {
    auto bear = std::make_shared<Bear>();
    virtual_shared_ptr<Bear> vbear(bear);
}

} // namespace test_virtual_shared_ptr

namespace test_virtual_shared_ptr_dispatch {

BOOST_AUTO_TEST_CASE_TEMPLATE(
    test_virtual_ptr_dispatch, Policy, policy_types<__COUNTER__>) {

    static use_classes<Policy, Player, Warrior, Object, Axe, Bear> YOMM2_GENSYM;

    using kick =
        method<void, std::string(virtual_shared_ptr<Player, Policy>), Policy>;

    static typename kick::template add_function<
        kick_bear<virtual_shared_ptr<Player, Policy>>>
        YOMM2_GENSYM;

    using fight = method<
        void,
        std::string(
            virtual_shared_ptr<Player, Policy>,
            virtual_shared_ptr<Object, Policy>,
            virtual_shared_ptr<Player, Policy>),
        Policy>;

    static typename fight::template add_function<fight_bear<
        virtual_shared_ptr<Player, Policy>, virtual_shared_ptr<Object, Policy>,
        virtual_shared_ptr<Player, Policy>>>
        YOMM2_GENSYM;

    update<Policy>();

    auto bear = make_virtual_shared<Bear, Policy>();
    auto warrior = make_virtual_shared<Warrior, Policy>();
    auto axe = make_virtual_shared<Axe, Policy>();

    BOOST_TEST(kick::fn(bear) == "growl");

    BOOST_TEST(fight::fn(warrior, axe, bear) == "kill bear");
}

} // namespace test_virtual_shared_ptr_dispatch
