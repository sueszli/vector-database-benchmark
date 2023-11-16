// Copyright (c) 2021 Jean-Louis Leroy
// Distributed under the Boost Software License, Version 1.0.
// See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <yorel/yomm2/keywords.hpp>

#include <iostream>
#include <memory>
#include <string>

using namespace yorel::yomm2;
using std::cout;

struct Role {
    virtual ~Role() {
    }
};

struct Employee : Role {};

struct Manager : Employee {};

register_classes(Role, Employee, Manager);

struct Payroll {
    int balance{10000};

    void pay(const Role& role) {
        pay_method::fn(this, role);
    }

  private:
    struct YOMM2_SYMBOL(pay);
    using pay_method =
        method<YOMM2_SYMBOL(pay), void(Payroll*, virtual_<const Role&>)>;

    void pay_employee(const Employee&) {
        balance -= 2000;
    }
    void pay_manager(const Manager&) {
        balance -= 3000;
    }

  public:
    using pay_functions = Payroll::pay_method::add_member_functions<
        &Payroll::pay_employee, &Payroll::pay_manager>;
};

Payroll::pay_functions YOMM2_GENSYM;

#define BOOST_TEST_MODULE yomm2
#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_CASE(member_method) {
    update();

    Payroll pr;
    const Employee& alice = Employee();
    const Manager& bob = Manager();

    pr.pay(alice);
    BOOST_TEST(pr.balance == 8000);
    pr.pay(bob);
    BOOST_TEST(pr.balance == 5000);
}
