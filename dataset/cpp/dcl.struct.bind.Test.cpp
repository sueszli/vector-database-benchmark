
// p1
// cv := denote the cv-qualifiers in the decl-specifier-seq
// S  := consist of the storage-class-specifiers of the decl-specifier-seq (if any)
//
// 1) a variable with a unique name e is introduced. 
//    If the assignment-expression in the initializer has array type A and no ref-qualifier is present, e is defined by:
// 
//    attribute-specifier-seq_{opt} S cv A e ;
//
//    otherwise e is defined as-if by:
//
//   attribute-specifier-seq_{opt} decl-specifier-seq ref-qualifier_{opt} e initializer ;
//
//   The type of the id-expression e is called E.
//
//   Note: E is never a reference type
//
//
// -- tuple --
// Let i be an index of type std::size_t corresponding to vi
// either:
//  e.get<i>()
//  get<i>(e)
//
//  In either case, 
//      - e is an lvalue if the type of the entity e is an lvalue reference and 
//      - an xvalue otherwise. 
//
//  -> auto& [a ,b ] -> e := lvalue
//  -> auto  [a ,b ] -> e := xvalue
//
//  T_i  := std::tuple_element<i, E>::type 
//  U_i  := either
//           - T_i&:  if the initializer is an lvalue,
//           - T_i&&: an rvalue reference otherwise, 
//
//  variables are introduced with unique names r_i as follows:
//
//      S U_i r_i = initializer ;
//
//  Each vi is the name of an lvalue of type Ti that refers to the object bound to ri; the referenced type is Ti.
//


// The lvalue is a bit-field if that member is a bit-field. [Example:
//      struct S { int x1 : 2; volatile double y1; };
//      S f();
//      const auto [ x, y ] = f();
// The type of the id-expression x is “const int”, the type of the id-expression y is “const volatile double”. —end example]


// For each identifier, a variable whose type is "reference to std::tuple_element<i, E>::type" is introduced: lvalue reference if its corresponding initializer is an lvalue, rvalue reference otherwise. The initializer for the i-th variable is
// - e.get<i>(), if lookup for the identifier get in the scope of E by class member access lookup finds at least one declaration that is a function template whose first template parameter is a non-type parameter
// - Otherwise, get<i>(e), where get is looked up by argument-dependent lookup only, ignoring non-ADL lookup. 
// 
// The initializer for the new variable is e.get<i> or get<i>(e). 
// Here the overload of get that is called is a rvalue in case we use auto and an lvalue in case we use auto&



#include <cassert>
#include <tuple>

// https://en.cppreference.com/w/cpp/language/structured_binding
float x{};
char  y{};
int   z{};
 
std::tuple<float&,char&&,int> tpl(x,std::move(y),z);
//auto tpl = std::tuple{x,std::move(y),z};
auto& [a,b,c] = tpl;
// a names a structured binding that refers to x; decltype(a) is float&
// b names a structured binding that refers to y; decltype(b) is char&&
// c names a structured binding that refers to the 3rd element of tpl; decltype(c) is int

int main() {
    a = 4.5;
    c = 5;

//    assert(4.5 == x);
//    assert(0 == z);

//    std::cout << a << '\n';
//    std::cout << z << '\n';
}
