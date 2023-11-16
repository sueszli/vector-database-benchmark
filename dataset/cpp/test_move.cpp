#include <yorel/yomm2/keywords.hpp>

#include "test_helpers.hpp"

#define BOOST_TEST_MODULE yomm2
#include <boost/test/included/unit_test.hpp>
#include <boost/utility/identity_type.hpp>

enum Status { ORIGINAL, COPY, MOVED, DEAD };

struct Base {
    int id;
    Status status;
    Base(int id) : id(id), status(ORIGINAL) {
    }
    Base(const Base& other) : id(other.id), status(COPY) {
    }
    Base(Base&& other) : id(other.id), status(MOVED) {
        other.status = DEAD;
    }
    virtual ~Base() {
    }
};

struct Derived : Base {
    using Base::Base;
};

// -----------------------------------------------------------------------------
// Test test apparatus.

BOOST_AUTO_TEST_CASE_TEMPLATE(
    test_moveable, Class, BOOST_IDENTITY_TYPE((std::tuple<Base, Derived>))) {
    Class a(1);

    BOOST_REQUIRE(a.status == ORIGINAL);

    Class b = a;
    BOOST_REQUIRE(a.status == ORIGINAL);
    BOOST_REQUIRE(b.status == COPY);

    Class c = std::move(a);
    BOOST_REQUIRE(a.status == DEAD);
    BOOST_REQUIRE(c.status == MOVED);
}

// -----------------------------------------------------------------------------

declare_method(void, move_non_virtual_arg, (virtual_<Base&>, Base&&));

define_method(void, move_non_virtual_arg, (Base & ref_arg, Base&& moving_arg)) {
    BOOST_TEST(ref_arg.status == ORIGINAL);
    BOOST_TEST(moving_arg.status == ORIGINAL);
    Base moved = std::move(moving_arg);
}

BOOST_AUTO_TEST_CASE(test_move_non_virtual_arg) {
    Base ref_arg(1), moving_arg(2);
    move_non_virtual_arg(ref_arg, std::move(moving_arg));
    BOOST_TEST(moving_arg.status == DEAD);
}

// -----------------------------------------------------------------------------

declare_method(void, move_virtual_arg, (Base&, virtual_<Base&&>));

define_method(void, move_virtual_arg, (Base & ref_arg, Base&& moving_arg)) {
    BOOST_TEST(ref_arg.status == ORIGINAL);
    BOOST_TEST(moving_arg.status == ORIGINAL);
    Base moved = std::move(moving_arg);
}

BOOST_AUTO_TEST_CASE(test_move_virtual_arg) {
    Base ref_arg(1), moving_arg(2);
    move_virtual_arg(ref_arg, std::move(moving_arg));
    BOOST_TEST(moving_arg.status == DEAD);
}

// -----------------------------------------------------------------------------
// Global initialization.

register_classes(Base, Derived);

BOOST_TEST_GLOBAL_FIXTURE(yomm2_update);
