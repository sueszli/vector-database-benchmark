#ifdef YOMM2_MD

hrefs: YOMM2_DECLARE_METHOD_CONTAINER

<sub>/ ->home / ->reference </sub>

entry: method_container
headers: yorel/yomm2/cute.hpp, yorel/yomm2/keywords.hpp

---
```
#define method_container(container) /*unspecified*/
#define method_container(container, return_type, name,
(function-parameter-list)) /*unspecified*/
```
---
Declare a method container, and optionally a method definition inside that
container.

Method containers are collections of definitions for a method, wrapped in a
template, parameterized by the definition's signature. This makes it possible
for a class to grant friendship to a set of definitions, or to all the
definitions, of a method. It also makes it possible to retrieve a specific
method, for example to call it. This is especially useful for definitions
created with ->define_method_inline.

This macro only creates declarations, and thus can be placed in a header
file. The four argument form makes it possible to access a method definition
across translation units.

A definition placed in a container is a static member function, called `fn`,
of the a specialisation of the container. Consequently, it must be created in
the same namespace as the container itself.

## example

#endif

#define BOOST_TEST_MODULE yomm2
#include <boost/test/included/unit_test.hpp>

#ifdef YOMM2_CODE

#include <string>

#include <yorel/yomm2/keywords.hpp>

method_container(kicks);

class Animal {
    std::string name;

  public:
    explicit Animal(const std::string& name) : name(name) {}
    virtual ~Animal() {}

    friend_method(kicks);
};

class Dog : public Animal {
    using Animal::Animal;
};

class Bulldog : public Dog {
    using Dog::Dog;
};

register_classes(Animal, Dog, Bulldog);

declare_method(std::string, kick, (virtual_<Animal*>));

define_method_inline(kicks, std::string, kick, (Dog* dog)) {
    return dog->name + " barks";
}

define_method(kicks, std::string, kick, (Bulldog* dog)) {
    return kicks<std::string(Dog*)>::fn(dog) + " and bites";
}

BOOST_AUTO_TEST_CASE(reference_example) {
    yorel::yomm2::update();

    Dog snoopy("Snoopy");
    Bulldog hector("Hector");
    Animal* animal;

    animal = &snoopy;
    BOOST_TEST(kick(animal) == "Snoopy barks");

    animal = &hector;
    BOOST_TEST(kick(animal) == "Hector barks and bites");
}

#endif

#ifdef YOMM2_MD

## see also

The [containers example](../examples/containers) uses
multiple containers across multiple namespaces and source files.

#endif
