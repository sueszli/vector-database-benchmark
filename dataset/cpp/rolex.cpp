#include <yorel/yomm2/keywords.hpp>

struct Role {
    virtual ~Role() {}
};

struct Employee : Role {
    virtual double pay() const;
};

struct Manager : Employee {
    virtual double pay() const;
};

struct Founder : Role {};

struct Expense {
    virtual ~Expense() {}
};

struct Public : Expense {};
struct Bus : Public {};
struct Metro : Public {};
struct Taxi : Expense {};
struct Jet : Expense {};

register_class(Role);
register_class(Employee, Role);
register_class(Manager, Employee);
register_class(Founder, Role);
register_class(Expense);
register_class(Public, Expense);
register_class(Bus, Public);
register_class(Metro, Public);
register_class(Taxi, Expense);
register_class(Jet, Expense);

declare_method(double, pay, (virtual_<const Employee&>));
declare_method(bool, approve, (virtual_<const Role&>, virtual_<const Expense&>, double));

define_method(double, pay, (const Employee&)) {
    return 3000;
}

define_method(double, pay, (const Manager& exec)) {
    return next(exec) + 2000;
}

define_method(bool, approve, (const Role& r, const Expense& e, double amount)) {
    return false;
}

define_method(bool, approve, (const Employee& r, const Public& e, double amount)) {
    return true;
}

define_method(bool, approve, (const Manager& r, const Taxi& e, double amount)) {
    return true;
}

define_method(bool, approve, (const Founder& r, const Expense& e, double amount)) {
    return true;
}

double call_pay(const Employee& emp) {
    return pay(emp);
}

double Employee::pay() const {
    return 3000;
}

double Manager::pay() const {
    return Employee::pay() + 2000;
}

double call_pay_vfunc(const Employee& emp) {
    return emp.pay();
}
