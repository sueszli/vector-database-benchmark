#include <iostream>
#include <vector>
#include <yorel/yomm2/keywords.hpp>

struct Animal {
    const char* name;
    Animal(const char* name) : name(name) {
    }
};

struct Dog : Animal {
    using Animal::Animal;
};

struct Cat : Animal {
    using Animal::Animal;
};

register_classes(Animal, Dog, Cat);

using yorel::yomm2::virtual_ptr;

declare_method(void, kick, (virtual_ptr<Animal>, std::ostream&));

define_method(void, kick, (virtual_ptr<Cat> animal, std::ostream& os)) {
    os << animal->name << " hisses.\n";
}

define_method(void, kick, (virtual_ptr<Dog> animal, std::ostream& os)) {
    os << animal->name << " barks.\n";
}

void kick_animals(
    const std::vector<virtual_ptr<Animal>>& animals, std::ostream& os) {
    for (auto animal : animals) {
        kick(animal, os);
    }
}

int main() {
    yorel::yomm2::update();

    Dog hector{"Hector"}, snoopy{"Snoopy"};
    Cat felix{"Felix"}, sylvester{"Sylvester"};
    std::vector<virtual_ptr<Animal>> animals = {
        virtual_ptr<Dog>::final(hector), virtual_ptr<Cat>::final(felix),
        virtual_ptr<Cat>::final(sylvester), virtual_ptr<Dog>::final(snoopy)};

    kick_animals(animals, std::cout);
}
