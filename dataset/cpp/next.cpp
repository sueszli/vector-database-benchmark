// next.cpp
// Copyright (c) 2018-2021 Jean-Louis Leroy
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

// Example taken Dylan's documentation, see
// http://opendylan.org/documentation/intro-dylan/multiple-dispatch.html

#include <iostream>

#include <yorel/yomm2/keywords.hpp>

using namespace std;

struct Vehicle {
    virtual ~Vehicle() {
    }
};

//[ car
struct Car : Vehicle {};

struct Truck : Vehicle {};

struct Inspector {
    virtual ~Inspector() {
    }
};

struct StateInspector : Inspector {};

register_classes(Vehicle, Car, Truck, Inspector, StateInspector);

declare_method(
    void, inspect, (virtual_<const Vehicle&>, virtual_<const Inspector&>));

define_method(void, inspect, (const Vehicle& v, const Inspector& i)) {
    cout << "Inspect vehicle.\n";
}

define_method(void, inspect, (const Car& v, const Inspector& i)) {
    next(v, i);
    cout << "Inspect seat belts.\n";
}

define_method(void, inspect, (const Car& v, const StateInspector& i)) {
    next(v, i);
    cout << "Check road tax.\n";
}

int main() {
    yorel::yomm2::update();

    const Vehicle& vehicle1 = Car();
    const Inspector& inspector1 = StateInspector();
    const Vehicle& vehicle2 = Truck();
    const Inspector& inspector2 = Inspector();

    cout << "First inspection:\n";
    inspect(vehicle1, inspector1);

    cout << "\nSecond inspection:\n";
    inspect(vehicle2, inspector2);

    return 0;
}
