// bdlb_transformiterator.t.cpp                                       -*-C++-*-
#include <bdlb_transformiterator.h>

#include <bslim_testutil.h>

#include <bslma_allocator.h>
#include <bslma_default.h>
#include <bslma_testallocator.h>

#include <bsls_asserttest.h>
#include <bsls_nameof.h>

#include <bsl_algorithm.h>
#include <bsl_cmath.h>
#include <bsl_cstring.h>
#include <bsl_iostream.h>
#include <bsl_list.h>
#include <bsl_map.h>
#include <bsl_numeric.h>
#include <bsl_string.h>

using namespace BloombergLP;
using namespace bdlb;
using namespace bsl;

// ============================================================================
//                             TEST PLAN
// ----------------------------------------------------------------------------
//                              Overview
//                              --------
// The type under test, 'TransformIterator', is implemented using contained
// constructor proxy objects for a generic functor type and a generic iterator
// type.  (Use of the proxy objects allows the functor and iterator to be
// initialized with an allocator even if the allocator is not used.)
//
// Because of the possibility that neither contained object uses allocators,
// 'TransformIterator' uses a form of implementation inheritance that supplies
// an 'allocator' method only when at least one of the contained objects does
// use an allocator.  Testing is required to demonstrate that the 'allocator'
// method is or is not present as appropriate, and that it returns the
// allocator supplied to the constructor of the object.  Additionally,
// 'TransformIterator' sets its 'UsesBslmaAllocator' allocator trait to true if
// either of its contained objects uses allocators, and to false if neither
// does.  Testing is required to demonstrate that this is done correctly.
//
// 'TransformIterator' is an iterator adapter providing appropriate typedefs
// for 'iterator_traits' to match the category of iterator it adapts.  For
// iterator categories other than input iterator, dereferencing an iterator
// must provide a reference type.  Accordingly, 'TransformIterator' sets its
// iterator category to input iterator if the functor does not return a
// reference type, otherwise passing through the iterator category of the
// underlying iterator.  Testing is required to demonstrate that the category
// and other iterator parameters are correctly determined.
//
// As per the above, 'TransformIterator' must be able to determine the return
// type of the functor.  In C++11 and onward, language features allow this to
// be determined, but in C++03 the return type must be supplied as a
// 'result_type' member of the functor.  'TransformIterator' is specialized for
// pointers to functions, from which it can determine the result type, and
// otherwise uses 'bslmf::ResultType' in C++03.  Testing is required to
// demonstrate that 'TransformIterator' picks up the correct result type in
// both C++03 and C++11.
//
// The remit of 'TransformIterator' is to pass on iterator operations to the
// contained iterator, and to apply the functor to the dereferenced contained
// iterator when 'TransformIterator' is itself dereferenced.  Testing is
// required to demonstrate that this occurs correctly, including for indexed
// dereference ('operator[](difference_type)') when the underlying iterator
// supports it, and pointer dereference ('operator->()') when the functor
// returns a reference.
//
// Functors used by 'TransformIterator' can maintain state.  Testing is
// required to demonstrate that such stateful functors maintain their state
// throughout the iteration process.
//
// Pairwise comparison operations of 'TransformIterator' objects are supported,
// comparing only the contained iterators.  Testing is required to demonstrate
// that contained functors do not participate.
//
// ----------------------------------------------------------------------------
// CREATORS
// [ 2] TransformIterator();
// [ 2] TransformIterator(Allocator *);
// [ 2] TransformIterator(const ITERATOR&, FUNCTOR, Allocator * = 0);
// [ 2] TransformIterator(const TransformIterator&, Allocator * = 0);
//
// CLASS METHODS
// [ 2] TransformIterator TransformIteratorUtil::make(...);
//
// MANIPULATORS
// [ 3] TransformIterator& operator=(const TransformIterator&);
// [ 3] TransformIterator& operator+=(difference_type);
// [ 3] TransformIterator& operator-=(difference_type);
// [ 3] TransformIterator& operator++();
// [ 3] TransformIterator& operator--();
// [ 3] Traits::reference operator*();
// [ 3] pointer operator->();
// [ 3] reference operator[](difference_type);
// [ 3] FUNCTOR& functor();
// [ 3] ITERATOR& iterator();
// [ 3] void swap(TransformIterator&);
//
// ACCESSORS
// [ 4] reference operator*() const;
// [ 4] pointer operator->() const;
// [ 4] reference operator[](difference_type n) const;
// [ 4] const FUNCTOR& functor() const;
// [ 4] const ITERATOR& iterator() const;
//
// FREE OPERATORS
// [ 5] bool operator==(const TI&, const TI&);
// [ 5] bool operator!=(const TI&, const TI&);
// [ 5] bool operator<(const TI&, const TI&);
// [ 5] bool operator>(const TI&, const TI&);
// [ 5] bool operator<=(const TI&, const TI&);
// [ 5] bool operator>=(const TI&, const TI&);
// [ 3] TransformIterator& operator++(TransformIterator&, int);
// [ 3] TransformIterator& operator--(TransformIterator&, int);
// [ 4] TransformIterator operator+(TransformIterator, difference_type);
// [ 4] TransformIterator operator+(difference_type, TransformIterator);
// [ 4] TransformIterator operator-(TransformIterator, difference_type);
// [ 4] difference_type operator-(TransformIterator, TransformIterator);
//
// FREE FUNCTIONS
// [ 3] void swap(TransformIterator&, TransformIterator&);
//
// ALLOCATOR TRAITS
// [ 6] bslma::UsesBslmaAllocator
// [ 6] bslma::Allocator *allocator() const;
// ----------------------------------------------------------------------------
// [ 1] BREATHING TEST
// [ 7] USAGE EXAMPLE

// ============================================================================
//                     STANDARD BDE ASSERT TEST FUNCTION
// ----------------------------------------------------------------------------

namespace {

int testStatus = 0;

void aSsErT(bool condition, const char *message, int line)
{
    if (condition) {
        cout << "Error " __FILE__ "(" << line << "): " << message
             << "    (failed)" << endl;

        if (0 <= testStatus && testStatus <= 100) {
            ++testStatus;
        }
    }
}

}  // close unnamed namespace

// ============================================================================
//               STANDARD BDE TEST DRIVER MACRO ABBREVIATIONS
// ----------------------------------------------------------------------------

#define ASSERT       BSLIM_TESTUTIL_ASSERT
#define ASSERTV      BSLIM_TESTUTIL_ASSERTV

#define LOOP_ASSERT  BSLIM_TESTUTIL_LOOP_ASSERT
#define LOOP0_ASSERT BSLIM_TESTUTIL_LOOP0_ASSERT
#define LOOP1_ASSERT BSLIM_TESTUTIL_LOOP1_ASSERT
#define LOOP2_ASSERT BSLIM_TESTUTIL_LOOP2_ASSERT
#define LOOP3_ASSERT BSLIM_TESTUTIL_LOOP3_ASSERT
#define LOOP4_ASSERT BSLIM_TESTUTIL_LOOP4_ASSERT
#define LOOP5_ASSERT BSLIM_TESTUTIL_LOOP5_ASSERT
#define LOOP6_ASSERT BSLIM_TESTUTIL_LOOP6_ASSERT

#define Q            BSLIM_TESTUTIL_Q   // Quote identifier literally.
#define P            BSLIM_TESTUTIL_P   // Print identifier and value.
#define P_           BSLIM_TESTUTIL_P_  // P(X) without '\n'.
#define T_           BSLIM_TESTUTIL_T_  // Print a tab (w/o newline).
#define L_           BSLIM_TESTUTIL_L_  // current Line number

// ============================================================================
//                  NEGATIVE-TEST MACRO ABBREVIATIONS
// ----------------------------------------------------------------------------

#define ASSERT_SAFE_PASS(EXPR) BSLS_ASSERTTEST_ASSERT_SAFE_PASS(EXPR)
#define ASSERT_SAFE_FAIL(EXPR) BSLS_ASSERTTEST_ASSERT_SAFE_FAIL(EXPR)
#define ASSERT_PASS(EXPR)      BSLS_ASSERTTEST_ASSERT_PASS(EXPR)
#define ASSERT_FAIL(EXPR)      BSLS_ASSERTTEST_ASSERT_FAIL(EXPR)
#define ASSERT_OPT_PASS(EXPR)  BSLS_ASSERTTEST_ASSERT_OPT_PASS(EXPR)
#define ASSERT_OPT_FAIL(EXPR)  BSLS_ASSERTTEST_ASSERT_OPT_FAIL(EXPR)

#define ASSERT_SAFE_PASS_RAW(EXPR) BSLS_ASSERTTEST_ASSERT_SAFE_PASS_RAW(EXPR)
#define ASSERT_SAFE_FAIL_RAW(EXPR) BSLS_ASSERTTEST_ASSERT_SAFE_FAIL_RAW(EXPR)
#define ASSERT_PASS_RAW(EXPR)      BSLS_ASSERTTEST_ASSERT_PASS_RAW(EXPR)
#define ASSERT_FAIL_RAW(EXPR)      BSLS_ASSERTTEST_ASSERT_FAIL_RAW(EXPR)
#define ASSERT_OPT_PASS_RAW(EXPR)  BSLS_ASSERTTEST_ASSERT_OPT_PASS_RAW(EXPR)
#define ASSERT_OPT_FAIL_RAW(EXPR)  BSLS_ASSERTTEST_ASSERT_OPT_FAIL_RAW(EXPR)

// ============================================================================
//                  HELPER CLASSES AND FUNCTIONS FOR TESTING
// ----------------------------------------------------------------------------

                            // ===================
                            // class Parenthesizer
                            // ===================

class Parenthesizer {
    // Wrap strings in increasingly nested levels of parentheses.

  private:
    // DATA
    mutable bsl::string d_before;  // opening parentheses
    mutable bsl::string d_after;   // closing parentheses

  public:
    // PUBLIC TYPES
    #ifndef BSLS_COMPILERFEATURES_SUPPORT_DECLTYPE
    typedef bsl::string& result_type;
    #endif

    // CREATORS
    explicit Parenthesizer(bslma::Allocator *basicAllocator);
        // Create a Parenthesizer object using the specified 'basicAllocator'
        // to supply memory.

    Parenthesizer(const Parenthesizer&  other,
                  bslma::Allocator     *basicAllocator);
        // Create a copy of the specified 'other' object using the specified
        // 'basicAllocator' to supply memory.

    // ACCESSORS
    bsl::string& operator()(bsl::string& s) const;
        // Increment the parentheses nesting level, modify the specified string
        // 's' to be wrapped in those parentheses, and return a reference to
        // 's'.

    bslma::Allocator *allocator() const;
        // Return the allocator used to supply memory for this object.

    // TRAITS
    BSLMF_NESTED_TRAIT_DECLARATION(Parenthesizer, bslma::UsesBslmaAllocator);
};

                            // -------------------
                            // class Parenthesizer
                            // -------------------

// CREATORS
Parenthesizer::Parenthesizer(bslma::Allocator *basicAllocator)
: d_before(basicAllocator)
, d_after(basicAllocator)
{
}

Parenthesizer::Parenthesizer(const Parenthesizer&  other,
                             bslma::Allocator     *basicAllocator)
: d_before(other.d_before, basicAllocator)
, d_after(other.d_after, basicAllocator)
{
}

// ACCESSORS
bsl::string& Parenthesizer::operator()(bsl::string& s) const
{
    return s = (d_before += "(") + s + (d_after += ")");
}

bslma::Allocator *Parenthesizer::allocator() const
{
    return d_before.allocator().mechanism();
}

                       // =============================
                       // class FunctorWithoutAllocator
                       // =============================

class FunctorWithoutAllocator {
    // A functor that does not use allocators.
  public:
    // PUBLIC TYPES
    typedef int result_type;

    // ACCESSORS
    int operator()(int n) const;
        // Return the specified 'n'.
};

                       // -----------------------------
                       // class FunctorWithoutAllocator
                       // -----------------------------

// ACCESSORS
int FunctorWithoutAllocator::operator()(int n) const
{
    return n;
}

                         // ==========================
                         // class FunctorWithAllocator
                         // ==========================

class FunctorWithAllocator {
    // A functor that does not use allocators.

  private:
    bslma::Allocator *d_allocator_p;  // the allocator for this object

  public:
    // PUBLIC TYPES
    typedef int result_type;

    // CREATORS
    FunctorWithAllocator();
    explicit FunctorWithAllocator(bslma::Allocator *basicAllocator);
        // Create an object of this type.  Optionally specify 'basicAllocator'
        // to set as the allocator of the new object.

    FunctorWithAllocator(const FunctorWithAllocator&  ,
                         bslma::Allocator            *basicAllocator = 0);
        // Create an object of this type.  Optionally specify 'basicAllocator'
        // to set as the allocator of the new object.

    // ACCESSORS
    int operator()(int n) const;
        // Return the specified 'n'.

    bslma::Allocator *allocator() const;
        // Return the allocator of this object.

    // TRAITS
    BSLMF_NESTED_TRAIT_DECLARATION(FunctorWithAllocator,
                                   bslma::UsesBslmaAllocator);
};

                       // --------------------------
                       // class FunctorWithAllocator
                       // --------------------------

// CREATORS
FunctorWithAllocator::FunctorWithAllocator()
: d_allocator_p(0)
{
}

FunctorWithAllocator::FunctorWithAllocator(bslma::Allocator *basicAllocator)
: d_allocator_p(basicAllocator)
{
}

FunctorWithAllocator::FunctorWithAllocator(
                                   const FunctorWithAllocator&  ,
                                   bslma::Allocator            *basicAllocator)
: d_allocator_p(basicAllocator)
{
}

// ACCESSORS
int FunctorWithAllocator::operator()(int n) const
{
    return n;
}

bslma::Allocator *FunctorWithAllocator::allocator() const
{
    return d_allocator_p;
}

                       // ==============================
                       // class IteratorWithoutAllocator
                       // ==============================

class IteratorWithoutAllocator : public bsl::reverse_iterator<int *> {
    // An iterator that does not use allocators.

  public:
    IteratorWithoutAllocator();
        // Create an object of this type.

    explicit IteratorWithoutAllocator(int *iterator);
        // Create an object of this type using the specified 'iterator'.
};

                       // ------------------------------
                       // class IteratorWithoutAllocator
                       // ------------------------------

IteratorWithoutAllocator::IteratorWithoutAllocator()
{
}

IteratorWithoutAllocator::IteratorWithoutAllocator(int *iterator)
: bsl::reverse_iterator<int *>(iterator)
{
}

                        // ===========================
                        // class IteratorWithAllocator
                        // ===========================

class IteratorWithAllocator : public bsl::reverse_iterator<int *> {
    // An iterator that uses allocators.

  private:
    bslma::Allocator *d_allocator_p;  // the allocator for this object

  public:
    // CREATORS
    IteratorWithAllocator();
    explicit IteratorWithAllocator(bslma::Allocator *basicAllocator);
        // Create an object of this type.  Optionally specify 'basicAllocator'
        // to set as the allocator of the new object.

    explicit IteratorWithAllocator(int *iterator);
    IteratorWithAllocator(int *iterator, bslma::Allocator *basicAllocator);
        // Create an object of this type using the specified 'iterator'.
        // Optionally specify 'basicAllocator' to set as the allocator of the
        // new object.

    IteratorWithAllocator(const IteratorWithAllocator&  other,
                          bslma::Allocator             *basicAllocator = 0);
        // Create a copy of the specified 'other' object.  Optionally specify
        // 'basicAllocator' to set as the allocator of the new object.

    // ACCESSORS
    bslma::Allocator *allocator() const;
        // Return the allocator of this object.

    // TRAITS
    BSLMF_NESTED_TRAIT_DECLARATION(IteratorWithAllocator,
                                   bslma::UsesBslmaAllocator);
};

                        // ---------------------------
                        // class IteratorWithAllocator
                        // ---------------------------

// CREATORS
IteratorWithAllocator::IteratorWithAllocator()
: d_allocator_p(0)
{
}

IteratorWithAllocator::IteratorWithAllocator(bslma::Allocator *basicAllocator)
: d_allocator_p(basicAllocator)
{
}

IteratorWithAllocator::IteratorWithAllocator(int *iterator)
: bsl::reverse_iterator<int *>(iterator)
, d_allocator_p(0)
{
}

IteratorWithAllocator::IteratorWithAllocator(int              *iterator,
                                             bslma::Allocator *basicAllocator)
: bsl::reverse_iterator<int *>(iterator)
, d_allocator_p(basicAllocator)
{
}

IteratorWithAllocator::IteratorWithAllocator(
                                  const IteratorWithAllocator&  other,
                                  bslma::Allocator             *basicAllocator)
: bsl::reverse_iterator<int *>(other)
, d_allocator_p(basicAllocator)
{
}

// ACCESSORS
bslma::Allocator *IteratorWithAllocator::allocator() const
{
    return d_allocator_p;
}

                                // ======================
                                // struct IteratorSupport
                                // ======================

template<class Category,
         class ValueType,
         class DistanceType = ptrdiff_t,
         class Pointer      = ValueType *,
         class Reference    = ValueType& >
struct IteratorSupport {
    // This class provides the five type names needed for an iterator to meet
    // the requirements of 'iterator_traits'.

    // PUBLIC TYPEDEFS
    typedef Category     iterator_category;
    typedef ValueType    value_type;
    typedef DistanceType difference_type;
    typedef Pointer      pointer;
    typedef Reference    reference;
};

                                // ===================
                                // struct FakeIterator
                                // ===================

template <class ITERATOR_TYPE>
struct FakeIterator : public ITERATOR_TYPE {
    // This class is used to test how the transform iterator sets traits.

    // ACCESSORS
    typename ITERATOR_TYPE::reference operator*() const;
        // Unimplemented dereference operator used in C++11 'declval' test.
};

                                   // =============
                                   // USAGE EXAMPLE
                                   // =============

// Next, we create a functor that will return a price given a product.  The
// following prolix functor at namespace scope is necessary for C++03:
//..
    #ifndef BSLS_COMPILERFEATURES_SUPPORT_DECLTYPE
    class Pricer {
      private:
        // DATA
        const bsl::map<bsl::string, double> *d_prices_p;  // price list

      public:
        // PUBLIC TYPES
        typedef double result_type;

        // CREATORS
        explicit Pricer(const bsl::map<bsl::string, double> *prices);
            // Create a 'Pricer' object using the specified 'prices'.  The
            // lifetime of 'prices' must be at least as long as this object.

        // ACCESSORS
        double operator()(const bsl::string& product) const;
            // Return the price of the specified 'product'.
    };

    // CREATORS
    Pricer::Pricer(const bsl::map<bsl::string, double> *prices)
    : d_prices_p(prices)
    {
    }

    double Pricer::operator()(const bsl::string& product) const
    {
        bsl::map<bsl::string, double>::const_iterator i =
                                                     d_prices_p->find(product);
        return i == d_prices_p->end() ? 0.0 : i->second;
    }
    #endif
//..

//=============================================================================
//                                 MAIN PROGRAM
//-----------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    int test                = argc > 1 ? atoi(argv[1]) : 0;
    int verbose             = argc > 2; (void)verbose;
    int veryVerbose         = argc > 3; (void)veryVerbose;
    int veryVeryVerbose     = argc > 4; (void)veryVeryVerbose;
    int veryVeryVeryVerbose = argc > 5; (void)veryVeryVeryVerbose;

    cout << "TEST " << __FILE__ << " CASE " << test << endl;

    bslma::TestAllocator globalAllocator("global", veryVeryVeryVerbose);
    bslma::Default::setGlobalAllocator(&globalAllocator);

    switch (test) { case 0:  // zero is always the leading case.
      case 7: {
        // --------------------------------------------------------------------
        // USAGE EXAMPLE
        //   Extracted from component header file.
        //
        // Concerns:
        //: 1 The usage example provided in the component header file compiles,
        //:   links, and runs as shown.
        //
        // Plan:
        //: 1 Incorporate usage example from header into test driver, remove
        //:   leading comment characters, and replace 'assert' with 'ASSERT'.
        //:   (C-1)
        //
        // Testing:
        //   USAGE EXAMPLE
        // --------------------------------------------------------------------
        if (verbose) cout << "\nUSAGE EXAMPLE"
                             "\n=============\n";
///Usage
///-----
// This section illustrates intended use of this component.
//
///Example 1: Totaling a Grocery List
/// - - - - - - - - - - - - - - - - -
// Suppose we have a shopping list of products and we want to compute how much
// it will cost to buy selected items.  We can use 'bdlb::TransformIterator' to
// do the computation, looking up the price of each item.
//
// First, we set up the price list:
//..
    bsl::map<bsl::string, double> prices;
    prices["pudding"] = 1.25;
    prices["apple"] = 0.33;
    prices["milk"] = 2.50;
//..
// Then, we set up our shopping list:
//..
    bsl::list<bsl::string> list;
    list.push_back("milk");
    list.push_back("milk");
    list.push_back("pudding");
//..
// Then, we create the functor object.  In C++11 or later, the explicit functor
// class above is unnecessary since we can use a lambda:
//..
    #ifndef BSLS_COMPILERFEATURES_SUPPORT_DECLTYPE
    Pricer pricer(&prices);
    #else
    auto pricer = [&](const bsl::string &product) { return prices[product]; };
    #endif
//..
// Now, we need a pair of transform iterators to process our grocery list.  We
// can use 'TransformIteratorUtil::make' to create those iterators, avoiding
// the need to explicitly name types.  We create the iterators and process the
// list in one step, as follows:
//..
    double total = bsl::accumulate(
        bdlb::TransformIteratorUtil::make(list.begin(), pricer),
        bdlb::TransformIteratorUtil::make(list.end(), pricer),
        0.0);
//..
// Finally, we verify that we have the correct total:
//..
    ASSERT(6.25 == total);
//..
//
///Example 2: Totaling the Grocery List Again
/// - - - - - - - - - - - - - - - - - - - - -
// In the previous example, we did not explicitly name our iterator type.  We
// may want to do so, however, if we intend to reuse iterators, or store them
// in data structures.  We will rework the previous example using explicitly
// typed iterators.  We also demonstrate how a single iterator type can deal
// with multiple functors.
//
// First, we notice that we have two different functor types depending on
// whether we compile as C++03 or C++11.  To abstract away the difference, we
// will use a 'bsl::function' functor type that is conformable to both:
//..
    typedef bdlb::TransformIterator<bsl::function<double(const bsl::string&)>,
                                    bsl::list<bsl::string>::iterator> Iterator;
//..
// Then, we create a pair of these iterators to traverse our list:
//..
    Iterator groceryBegin(list.begin(), pricer);
    Iterator groceryEnd(list.end(), pricer);
//..
// Now, we add up the prices of our groceries:
//..
    double retotal = bsl::accumulate(groceryBegin, groceryEnd, 0.0);
//..
// Finally, we verify that we have the correct total:
//..
    ASSERT(6.25 == retotal);
//..
//
///Example 3: Summing Absolute Values
/// - - - - - - - - - - - - - - - - -
// Suppose we have a sequence of numbers and we would like to sum their
// absolute values.  We can use 'bdlb::TransformIterator' for this purpose.
//
// First, we set up the numbers:
//..
    int data[5] = { 1, -1, 2, -2, 3 };
//..
// Then, we need a functor that will return the absolute value of a number.
// Rather than write a functor object, we can use a simple pointer to function
// as a functor:
//..
    int (*abs)(int) = &bsl::abs;
//..
// Next, we create the transform iterators that will convert a number to its
// absolute value.  We need iterators for both the beginning and end of the
// sequence:
//..
    bdlb::TransformIterator<int(*)(int), int *> dataBegin(data + 0, abs);
    bdlb::TransformIterator<int(*)(int), int *> dataEnd  (data + 5, abs);
//..
// Now, we compute the sum of the absolute values of the numbers:
//..
    int sum = bsl::accumulate(dataBegin, dataEnd, 0);
//..
// Finally, we verify that we have computed the sum correctly:
//..
    ASSERT(9 == sum);
//..
      } break;
      case 6: {
        // --------------------------------------------------------------------
        // TESTING TRAITS AND 'allocator()'
        //
        // Concerns:
        //: 1 The transform iterator should have an affirmative allocator trait
        //:   if either its functor or underlying iterator uses allocators, and
        //:   a negative one if neither do.
        //
        //: 2 The transform iterator should have an 'allocator()' method if
        //:   if either its functor or underlying iterator uses allocators, and
        //:   not have such a method if neither use allocators.
        //:
        //: 3 If the transform iterator has an 'allocator()' method, it should
        //:   return the allocator with which the object was initialized, if
        //:   one was provided.
        //:
        //: 4 The allocator of the subobjects should be the same as the
        //:   allocator with which the object was initialized, if one was
        //:   provided.
        //:
        //: 5 The iterator traits of the transform iterator should correspond
        //:   correctly to the traits of the underlying iterator and the return
        //:   type of the functor.
        //
        // Plan:
        //: 1 Create all four possible versions of the transform iterator with
        //:   its members using or not using allocators, and verify that the
        //:   trait is correct.  (C-1)
        //:
        //: 2 For versions that have an 'allocator()' method, construct an
        //:   object using a test allocator and verify that the object itself
        //:   and its allocator-using subobjects return that allocator.
        //:   (C-2,3,4)
        //:
        //: 3 For versions that should not have an 'allocator()' method, use
        //:   inheritance to cause a situation where the test would fail to
        //:   compile if the 'allocator()' method were present.  (C-2)
        //:
        //: 4 Create transform iterators based on a variety of iterator
        //:   categories and functor return types and verify that the trait
        //:   types of the iterators are set correctly.  (C-5)
        //
        // Testing:
        //   bslma::UsesBslmaAllocator
        //   bslma::Allocator *allocator() const;
        // --------------------------------------------------------------------
        if (verbose) cout << "\nTESTING TRAITS AND 'allocator()'"
                             "\n================================\n";

        if (veryVerbose) {
            cout << "functor with allocator, iterator with allocator\n";
        }
        {
            typedef bdlb::TransformIterator<FunctorWithAllocator,
                                            IteratorWithAllocator>
                Obj;

            ASSERT(bslma::UsesBslmaAllocator<Obj>::value);

            bslma::TestAllocator ta(veryVeryVeryVerbose);

            Obj        mX(&ta);
            const Obj& X = mX;

            ASSERT(&ta == X.allocator());
            ASSERT(&ta == X.functor().allocator());
            ASSERT(&ta == X.iterator().allocator());
            ASSERT(&ta == mX.allocator());
            ASSERT(&ta == mX.functor().allocator());
            ASSERT(&ta == mX.iterator().allocator());
        }

        if (veryVerbose) {
            cout << "functor without allocator, iterator with allocator\n";
        }
        {
            typedef bdlb::TransformIterator<FunctorWithoutAllocator,
                                            IteratorWithAllocator>
                Obj;

            ASSERT(bslma::UsesBslmaAllocator<Obj>::value);

            bslma::TestAllocator ta(veryVeryVeryVerbose);

            Obj        mX(&ta);
            const Obj& X = mX;

            ASSERT(&ta == X.allocator());
            ASSERT(&ta == X.iterator().allocator());
            ASSERT(&ta == mX.allocator());
            ASSERT(&ta == mX.iterator().allocator());
        }

        if (veryVerbose) {
            cout << "functor with allocator, iterator without allocator\n";
        }
        {
            typedef bdlb::TransformIterator<FunctorWithAllocator,
                                            IteratorWithoutAllocator>
                Obj;

            ASSERT(bslma::UsesBslmaAllocator<Obj>::value);

            bslma::TestAllocator ta(veryVeryVeryVerbose);

            Obj        mX(&ta);
            const Obj& X = mX;

            ASSERT(&ta == X.allocator());
            ASSERT(&ta == X.functor().allocator());
            ASSERT(&ta == mX.allocator());
            ASSERT(&ta == mX.functor().allocator());
        }

        if (veryVerbose) {
            cout << "functor without allocator, iterator without allocator\n";
        }
        {
            typedef bdlb::TransformIterator<FunctorWithoutAllocator,
                                            IteratorWithoutAllocator>
                Obj;

            ASSERT(!bslma::UsesBslmaAllocator<Obj>::value);

            struct FakeAllocatorMethod {
                // This object contains a member named 'allocator' that is not
                // a method.

              public:
                // PUBLIC TYPES
                enum { allocator = -1 };
            };

            struct DerivedTransformIterator : public FakeAllocatorMethod,
                                              public Obj {
                // This object inherits from both the transform iterator and
                // the fake allocator class.  If the transform iterator has a
                // public 'allocator()' method, asking for the 'allocator'
                // member of this class will be ambiguous and will fail to
                // compile.
            };

            ASSERT(-1 == DerivedTransformIterator::allocator);
        }

        if (veryVerbose) {
            cout << "input iterator, functor returns reference\n";
        }
        {
            typedef
              FakeIterator<IteratorSupport<bsl::input_iterator_tag, int> > Itr;
            typedef bdlb::TransformIterator<char& (*)(int), Itr> Obj;
            ASSERT((bsl::is_same<Itr::iterator_category,
                                 Obj::iterator_category>::value));
            ASSERT((bsl::is_same<Itr::difference_type,
                                 Obj::difference_type>::value));
            LOOP_ASSERT(bsls::NameOf<Obj::value_type>().name(),
                        (bsl::is_same<Obj::value_type, char>::value));
            LOOP_ASSERT(bsls::NameOf<Obj::pointer>().name(),
                        (bsl::is_same<Obj::pointer, char *>::value));
            LOOP_ASSERT(bsls::NameOf<Obj::reference>().name(),
                        (bsl::is_same<Obj::reference, char&>::value));
        }

        if (veryVerbose) {
            cout << "input iterator, functor returns value\n";
        }
        {
            typedef
              FakeIterator<IteratorSupport<bsl::input_iterator_tag, int> > Itr;
            typedef bdlb::TransformIterator<char (*)(int), Itr> Obj;
            ASSERT((bsl::is_same<bsl::input_iterator_tag,
                                 Obj::iterator_category>::value));
            ASSERT((bsl::is_same<Itr::difference_type,
                                 Obj::difference_type>::value));
            LOOP_ASSERT(bsls::NameOf<Obj::value_type>().name(),
                        (bsl::is_same<Obj::value_type, char>::value));
            LOOP_ASSERT(bsls::NameOf<Obj::pointer>().name(),
                        (bsl::is_same<Obj::pointer, char *>::value));
            LOOP_ASSERT(bsls::NameOf<Obj::reference>().name(),
                        (bsl::is_same<Obj::reference, char>::value));
        }

        if (veryVerbose) {
            cout << "forward iterator, functor returns reference\n";
        }
        {
            typedef FakeIterator<
                IteratorSupport<bsl::forward_iterator_tag, int> >  Itr;
            typedef bdlb::TransformIterator<char& (*)(int), Itr> Obj;
            ASSERT((bsl::is_same<Itr::iterator_category,
                                 Obj::iterator_category>::value));
            ASSERT((bsl::is_same<Itr::difference_type,
                                 Obj::difference_type>::value));
            LOOP_ASSERT(bsls::NameOf<Obj::value_type>().name(),
                        (bsl::is_same<Obj::value_type, char>::value));
            LOOP_ASSERT(bsls::NameOf<Obj::pointer>().name(),
                        (bsl::is_same<Obj::pointer, char *>::value));
            LOOP_ASSERT(bsls::NameOf<Obj::reference>().name(),
                        (bsl::is_same<Obj::reference, char&>::value));
        }

        if (veryVerbose) {
            cout << "forward iterator, functor returns value\n";
        }
        {
            typedef FakeIterator<
                IteratorSupport<bsl::forward_iterator_tag, int> > Itr;
            typedef bdlb::TransformIterator<char (*)(int), Itr> Obj;
            ASSERT((bsl::is_same<bsl::input_iterator_tag,
                                 Obj::iterator_category>::value));
            ASSERT((bsl::is_same<Itr::difference_type,
                                 Obj::difference_type>::value));
            LOOP_ASSERT(bsls::NameOf<Obj::value_type>().name(),
                        (bsl::is_same<Obj::value_type, char>::value));
            LOOP_ASSERT(bsls::NameOf<Obj::pointer>().name(),
                        (bsl::is_same<Obj::pointer, char *>::value));
            LOOP_ASSERT(bsls::NameOf<Obj::reference>().name(),
                        (bsl::is_same<Obj::reference, char>::value));
        }

        if (veryVerbose) {
            cout << "bidirectional iterator, functor returns reference\n";
        }
        {
            typedef FakeIterator<
                IteratorSupport<bsl::bidirectional_iterator_tag, int> > Itr;
            typedef bdlb::TransformIterator<char& (*)(int), Itr>      Obj;
            ASSERT((bsl::is_same<Itr::iterator_category,
                                 Obj::iterator_category>::value));
            ASSERT((bsl::is_same<Itr::difference_type,
                                 Obj::difference_type>::value));
            LOOP_ASSERT(bsls::NameOf<Obj::value_type>().name(),
                        (bsl::is_same<Obj::value_type, char>::value));
            LOOP_ASSERT(bsls::NameOf<Obj::pointer>().name(),
                        (bsl::is_same<Obj::pointer, char *>::value));
            LOOP_ASSERT(bsls::NameOf<Obj::reference>().name(),
                        (bsl::is_same<Obj::reference, char&>::value));
        }

        if (veryVerbose) {
            cout << "bidirectional iterator, functor returns value\n";
        }
        {
            typedef FakeIterator<
                IteratorSupport<bsl::bidirectional_iterator_tag, int> > Itr;
            typedef bdlb::TransformIterator<char (*)(int), Itr>       Obj;
            ASSERT((bsl::is_same<bsl::input_iterator_tag,
                                 Obj::iterator_category>::value));
            ASSERT((bsl::is_same<Itr::difference_type,
                                 Obj::difference_type>::value));
            LOOP_ASSERT(bsls::NameOf<Obj::value_type>().name(),
                        (bsl::is_same<Obj::value_type, char>::value));
            LOOP_ASSERT(bsls::NameOf<Obj::pointer>().name(),
                        (bsl::is_same<Obj::pointer, char *>::value));
            LOOP_ASSERT(bsls::NameOf<Obj::reference>().name(),
                        (bsl::is_same<Obj::reference, char>::value));
        }

        if (veryVerbose) {
            cout << "random access iterator, functor returns reference\n";
        }
        {
            typedef FakeIterator<
                IteratorSupport<bsl::random_access_iterator_tag, int> > Itr;
            typedef bdlb::TransformIterator<char& (*)(int), Itr>      Obj;
            ASSERT((bsl::is_same<Itr::iterator_category,
                                 Obj::iterator_category>::value));
            ASSERT((bsl::is_same<Itr::difference_type,
                                 Obj::difference_type>::value));
            LOOP_ASSERT(bsls::NameOf<Obj::value_type>().name(),
                        (bsl::is_same<Obj::value_type, char>::value));
            LOOP_ASSERT(bsls::NameOf<Obj::pointer>().name(),
                        (bsl::is_same<Obj::pointer, char *>::value));
            LOOP_ASSERT(bsls::NameOf<Obj::reference>().name(),
                        (bsl::is_same<Obj::reference, char&>::value));
        }

        if (veryVerbose) {
            cout << "random access iterator, functor returns value\n";
        }
        {
            typedef FakeIterator<
                IteratorSupport<bsl::random_access_iterator_tag, int> > Itr;
            typedef bdlb::TransformIterator<char (*)(int), Itr>       Obj;
            ASSERT((bsl::is_same<bsl::input_iterator_tag,
                                 Obj::iterator_category>::value));
            ASSERT((bsl::is_same<Itr::difference_type,
                                 Obj::difference_type>::value));
            LOOP_ASSERT(bsls::NameOf<Obj::value_type>().name(),
                        (bsl::is_same<Obj::value_type, char>::value));
            LOOP_ASSERT(bsls::NameOf<Obj::pointer>().name(),
                        (bsl::is_same<Obj::pointer, char *>::value));
            LOOP_ASSERT(bsls::NameOf<Obj::reference>().name(),
                        (bsl::is_same<Obj::reference, char>::value));
        }
      } break;
      case 5: {
        // --------------------------------------------------------------------
        // TESTING FREE FUNCTIONS
        //
        // Concerns:
        //: 1 'TransformIterator' objects can be compared when the underlying
        //:   iterators can be compared.
        //:
        //: 2 Such comparisons disregard the functors of the objects.
        //
        // Plan:
        //: 1 Create 'TransformIterator' objects holding reverse iterators into
        //:   an array of integers and verify that comparisons are correct.
        //:   (C-1)
        //:
        //: 2 When comparing these objects, use different functors and verify
        //:   that comparisons are not affected.
        //:   (C-2)
        //
        // Testing:
        //   bool operator==(const TI&, const TI&);
        //   bool operator!=(const TI&, const TI&);
        //   bool operator<(const TI&, const TI&);
        //   bool operator<=(const TI&, const TI&);
        //   bool operator>(const TI&, const TI&);
        //   bool operator>=(const TI&, const TI&);
        // --------------------------------------------------------------------
        if (verbose) cout << "\nTESTING FREE FUNCTIONS"
                             "\n======================\n";

        double a[1] = {.785};

        typedef bsl::reverse_iterator<double *>           ri;
        typedef TransformIterator<double (*)(double), ri> Obj;

        double (*sin)(double) = &bsl::sin;
        double (*cos)(double) = &bsl::cos;

        Obj iterators[4] = {
            Obj(ri(a + 0), sin),
            Obj(ri(a + 0), cos),
            Obj(ri(a + 1), sin),
            Obj(ri(a + 1), cos),
        };

        bool expected_eq[4][4] = {
            true,  true,  false, false,
            true,  true,  false, false,
            false, false, true,  true,
            false, false, true,  true,
        };

        bool expected_lt[4][4] = {
            false, false, false, false,
            false, false, false, false,
            true,  true,  false, false,
            true,  true,  false, false,
        };

        if (veryVerbose) {
            cout << "\toperator==\n";
        }
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                bool expected = expected_eq[i][j];
                if (veryVeryVerbose) {
                    P_(i) P_(j) P(expected)
                }
                ASSERT(expected == (iterators[i] == iterators[j]));
            }
        }

        if (veryVerbose) {
            cout << "\toperator!=\n";
        }
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                bool expected = !expected_eq[i][j];
                if (veryVeryVerbose) {
                    P_(i) P_(j) P(expected)
                }
                ASSERT(expected == (iterators[i] != iterators[j]));
            }
        }

        if (veryVerbose) {
            cout << "\toperator<\n";
        }
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                bool expected = expected_lt[i][j];
                if (veryVeryVerbose) {
                    P_(i) P_(j) P(expected)
                }
                ASSERT(expected == (iterators[i] < iterators[j]));
            }
        }

        if (veryVerbose) {
            cout << "\toperator<=\n";
        }
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                bool expected = expected_lt[i][j] || expected_eq[i][j];
                if (veryVeryVerbose) {
                    P_(i) P_(j) P(expected)
                }
                ASSERT(expected == (iterators[i] <= iterators[j]));
            }
        }

        if (veryVerbose) {
            cout << "\toperator>\n";
        }
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                bool expected = !expected_lt[i][j] && !expected_eq[i][j];
                if (veryVeryVerbose) {
                    P_(i) P_(j) P(expected)
                }
                ASSERT(expected == (iterators[i] > iterators[j]));
            }
        }

        if (veryVerbose) {
            cout << "\toperator>=\n";
        }
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                bool expected = !expected_lt[i][j];
                if (veryVeryVerbose) {
                    P_(i) P_(j) P(expected)
                }
                ASSERT(expected == (iterators[i] >= iterators[j]));
            }
        }
      } break;
      case 4: {
        // --------------------------------------------------------------------
        // TESTING ACCESSORS
        //
        // Concerns:
        //: 1 The accessor methods should do what they are supposed to, some
        //:   forwarding to the underlying iterator, some querying the object
        //:   itself, and the dereference operators invoking the functor.
        //
        // Plan:
        //: 1 Create a transform iterator, apply the various accessors, and
        //:   verify that the results are correct.  (C-1)
        //
        // Testing:
        //   reference operator*() const;
        //   pointer operator->() const;
        //   reference operator[](difference_type n) const;
        //   TransformIterator operator+(TransformIterator, difference_type);
        //   TransformIterator operator+(difference_type, TransformIterator);
        //   TransformIterator operator-(TransformIterator, difference_type);
        //   difference_type operator-(TransformIterator, TransformIterator);
        //   const FUNCTOR& functor() const;
        //   const ITERATOR& iterator() const;
        // --------------------------------------------------------------------
        if (verbose) cout << "\nTESTING ACCESSORS"
                             "\n=================\n";

        typedef TransformIterator<Parenthesizer, bsl::string *> Obj;

        bsl::string DATA[] = { "a", "b", "c" };

        bslma::TestAllocator ta;

        Obj        mX(DATA + 1, Parenthesizer(&ta), &ta);
        const Obj& X = mX;

        if (veryVerbose) {
            cout << "operator*() const\n";
        }
        ASSERT("(b)" == *X);

        if (veryVerbose) {
            cout << "operator->() const\n";
        }
        ASSERT(0 == bsl::strcmp("(((b)))", X->c_str()));

        if (veryVerbose) {
            cout << "operator[](difference_type n) const\n";
        }
        ASSERT("(((c)))" == X[1]);

        if (veryVerbose) {
            cout << "operator+(TransformIterator, difference_type)\n";
        }
        ASSERT(&DATA[2] == (X + 1).iterator());

        if (veryVerbose) {
            cout << "operator+(difference_type, TransformIterator)\n";
        }
        ASSERT(&DATA[2] == (1 + X).iterator());

        if (veryVerbose) {
            cout << "operator-(TransformIterator, difference_type)\n";
        }
        ASSERT(&DATA[0] == (X - 1).iterator());

        if (veryVerbose) {
            cout << "operator-(TransformIterator, TransformIterator)\n";
        }
        ASSERT(-2 == (X - 1) - (X + 1));

        if (veryVerbose) {
            cout << "functor() const\n";
        }
        ASSERT("((((a))))" == X.functor()(DATA[0]));

        if (veryVerbose) {
            cout << "iterator() const\n";
        }
        ASSERT(&DATA[1] == X.iterator());
      } break;
      case 3: {
        // --------------------------------------------------------------------
        // TESTING MANIPULATORS
        //
        // Concerns:
        //: 1 The manipulator methods should do what they are supposed to, some
        //:   forwarding to the underlying iterator, some querying or modifying
        //:   the object itself, and the dereference operators invoking the
        //:   functor.
        //
        // Plan:
        //: 1 Create a transform iterator, apply the various manipulators, and
        //:   verify that the results are correct.  (C-1)
        //
        // Testing:
        //   TransformIterator& operator=(const TransformIterator&);
        //   TransformIterator& operator++();
        //   TransformIterator& operator++(TransformIterator&, int);
        //   TransformIterator& operator--();
        //   TransformIterator& operator--(TransformIterator&, int);
        //   TransformIterator& operator+=(difference_type);
        //   TransformIterator& operator-=(difference_type);
        //   Traits::reference operator*();
        //   pointer operator->();
        //   reference operator[](difference_type);
        //   FUNCTOR& functor();
        //   ITERATOR& iterator();
        //   void swap(TransformIterator&);
        //   void swap(TransformIterator&, TransformIterator&);
        // --------------------------------------------------------------------
        if (verbose) cout << "\nTESTING MANIPULATORS"
                             "\n====================\n";

        typedef TransformIterator<Parenthesizer, bsl::string *> Obj;

        bsl::string DATA[] = { "a", "b", "c" };

        bslma::TestAllocator ta;

        Obj        mX(DATA + 1, Parenthesizer(&ta), &ta);
        const Obj& X = mX;

        if (veryVerbose) {
            cout << "operator=(const TransformIterator&)\n";
        }
        {
            Obj        mY(&ta);
            const Obj& Y = mY;
            mY = X;
            ASSERT(&DATA[1] == Y.iterator());
        }

        if (veryVerbose) {
            cout << "operator++()\n";
        }
        {
            const Obj &Y = ++mX;
            ASSERT(bsls::Util::addressOf(Y) == bsls::Util::addressOf(X));
            ASSERT(&DATA[2] == X.iterator());
        }

        if (veryVerbose) {
            cout << "operator--()\n";
        }
        {
            const Obj &Y = --mX;
            ASSERT(bsls::Util::addressOf(Y) == bsls::Util::addressOf(X));
            ASSERT(&DATA[1] == X.iterator());
        }

        if (veryVerbose) {
            cout << "operator++(int)\n";
        }
        {
            const Obj Y = mX++;
            ASSERT(&DATA[1] == Y.iterator());
            ASSERT(&DATA[2] == X.iterator());
        }

        if (veryVerbose) {
            cout << "operator--(int)\n";
        }
        {
            const Obj Y = mX--;
            ASSERT(&DATA[2] == Y.iterator());
            ASSERT(&DATA[1] == X.iterator());
        }

        if (veryVerbose) {
            cout << "operator+=(difference_type)\n";
        }
        {
            const Obj &Y = mX += 1;
            ASSERT(bsls::Util::addressOf(Y) == bsls::Util::addressOf(X));
            ASSERT(&DATA[2] == X.iterator());
        }

        if (veryVerbose) {
            cout << "operator-=(difference_type)\n";
        }
        {
            const Obj &Y = mX -= 1;
            ASSERT(bsls::Util::addressOf(Y) == bsls::Util::addressOf(X));
            ASSERT(&DATA[1] == X.iterator());
        }

        if (veryVerbose) {
            cout << "operator*()\n";
        }
        ASSERT("(b)" == *mX);

        if (veryVerbose) {
            cout << "operator->()\n";
        }
        ASSERT(0 == bsl::strcmp("(((b)))", mX->c_str()));

        if (veryVerbose) {
            cout << "operator[](difference_type n)\n";
        }
        ASSERT("(((c)))" == mX[1]);

        if (veryVerbose) {
            cout << "functor()\n";
        }
        mX.functor() = Parenthesizer(&ta);
        ASSERT("(a)" == X.functor()(DATA[0]));

        if (veryVerbose) {
            cout << "iterator()\n";
        }
        mX.iterator() = &DATA[0];
        ASSERT(&DATA[0] == X.iterator());

        if (veryVerbose) {
            cout << "swap(TransformIterator&)\n";
        }
        {
            Obj        mY(&DATA[2], Parenthesizer(&ta), &ta);
            const Obj& Y = mY;

            mX.swap(mY);
            ASSERT("((((c))))" == *X);
            ASSERT("(((a)))" == *Y);
        }

        if (veryVerbose) {
            cout << "swap(TransformIterator&, (TransformIterator&)\n";
        }
        {
            Obj        mY(&DATA[0], Parenthesizer(&ta), &ta);
            const Obj& Y = mY;

            using bsl::swap;
            swap(mX, mY);
            ASSERT("((((a))))" == *X);
            ASSERT("((((((c))))))" == *Y);
        }
      } break;
      case 2: {
        // --------------------------------------------------------------------
        // TESTING CREATORS
        //
        // Concerns:
        //: 1 The various forms of constructor for the transform iterator
        //:   correctly initialize the new object, and pass allocators to the
        //:   constructors of the subobjects that need them.
        //:
        //: 2 The transform iterator utility 'make' class method correctly
        //:   creates transform iterator objects.
        //
        // Plan:
        //: 1 For each of the four combinations of the two subobjects having or
        //:   not having allocators, create an object using each constructor
        //:   and verify that the objects are correct.  (C-1)
        //:
        //: 2 For each of the four combinations of the two subobjects having or
        //:   not having allocators, create an object using the utility 'make'
        //:   method and verify that the objects are correct.  (C-2)
        //
        // Testing:
        //   TransformIterator();
        //   TransformIterator(Allocator *);
        //   TransformIterator(const ITERATOR&, FUNCTOR, Allocator * = 0);
        //   TransformIterator(const TransformIterator&, Allocator * = 0);
        //   TransformIterator TransformIteratorUtil::make(...);
        // --------------------------------------------------------------------
        if (verbose) cout << "\nTESTING CREATORS"
                             "\n================\n";

        bslma::TestAllocator ta;

        int DATA[] = { 1, 2, 3 };

        if (veryVerbose) {
            cout << "functor with allocator, iterator with allocator\n";
        }
        {
            typedef TransformIterator<FunctorWithAllocator,
                                      IteratorWithAllocator>
                Obj;

            if (veryVerbose) {
                cout << "\tTI()\n";
            }
            {
                Obj        mX;
                const Obj& X = mX;

                ASSERT(bslma::Default::allocator(0) ==
                       bslma::Default::allocator(X.allocator()));
                ASSERT(bslma::Default::allocator(0) ==
                       bslma::Default::allocator(X.functor().allocator()));
                ASSERT(bslma::Default::allocator(0) ==
                       bslma::Default::allocator(X.iterator().allocator()));
            }

            if (veryVerbose) {
                cout << "\tTI(Allocator *)\n";
            }
            {
                Obj        mX(&ta);
                const Obj& X = mX;

                ASSERT(&ta == X.allocator());
                ASSERT(&ta == X.functor().allocator());
                ASSERT(&ta == X.iterator().allocator());
            }

            if (veryVerbose) {
                cout << "\tTI(const ITERATOR&, FUNCTOR, Allocator *)\n";
            }
            {
                Obj        mX(IteratorWithAllocator(&DATA[1], &ta),
                              FunctorWithAllocator(&ta),
                              &ta);
                const Obj& X = mX;

                ASSERT(&ta == X.allocator());
                ASSERT(&ta == X.functor().allocator());
                ASSERT(&ta == X.iterator().allocator());
                ASSERT(bsl::reverse_iterator<int *>(&DATA[1]) == X.iterator());
                ASSERT(DATA[0] == *X);

                const Obj& Y = TransformIteratorUtil::make(
                    X.iterator(), X.functor(), X.allocator());

                ASSERT(&ta == Y.allocator() || 0 == Y.allocator());
                ASSERT(bsl::reverse_iterator<int *>(&DATA[1]) == Y.iterator());
                ASSERT(DATA[0] == *Y);
            }

            if (veryVerbose) {
                cout << "\tTI(const TI&, Allocator *)\n";
            }
            {
                Obj        mY(IteratorWithAllocator(&DATA[1], &ta),
                              FunctorWithAllocator(&ta),
                              &ta);
                const Obj& Y = mY;
                Obj        mX(Y, &ta);
                const Obj& X = mX;

                ASSERT(&ta == X.allocator());
                ASSERT(&ta == X.functor().allocator());
                ASSERT(&ta == X.iterator().allocator());
                ASSERT(bsl::reverse_iterator<int *>(&DATA[1]) == X.iterator());
                ASSERT(DATA[0] == *X);
            }
        }

        if (veryVerbose) {
            cout << "functor without allocator, iterator with allocator\n";
        }
        {
            typedef TransformIterator<FunctorWithoutAllocator,
                                      IteratorWithAllocator>
                Obj;

            if (veryVerbose) {
                cout << "\tTI()\n";
            }
            {
                Obj        mX;
                const Obj& X = mX;

                ASSERT(bslma::Default::allocator(0) ==
                       bslma::Default::allocator(X.allocator()));
                ASSERT(bslma::Default::allocator(0) ==
                       bslma::Default::allocator(X.iterator().allocator()));
            }

            if (veryVerbose) {
                cout << "\tTI(Allocator *)\n";
            }
            {
                Obj        mX(&ta);
                const Obj& X = mX;

                ASSERT(&ta == X.allocator());
                ASSERT(&ta == X.iterator().allocator());
            }

            if (veryVerbose) {
                cout << "\tTI(const ITERATOR&, FUNCTOR, Allocator *)\n";
            }
            {
                Obj        mX(IteratorWithAllocator(&DATA[1], &ta),
                              FunctorWithoutAllocator(),
                              &ta);
                const Obj& X = mX;

                ASSERT(&ta == X.allocator());
                ASSERT(&ta == X.iterator().allocator());
                ASSERT(bsl::reverse_iterator<int *>(&DATA[1]) == X.iterator());
                ASSERT(DATA[0] == *X);

                const Obj& Y = TransformIteratorUtil::make(
                    X.iterator(), X.functor(), X.allocator());

                ASSERT(&ta == Y.allocator() || 0 == Y.allocator());
                ASSERT(bsl::reverse_iterator<int *>(&DATA[1]) == Y.iterator());
                ASSERT(DATA[0] == *Y);
            }

            if (veryVerbose) {
                cout << "\tTI(const TI&, Allocator *)\n";
            }
            {
                Obj        mY(IteratorWithAllocator(&DATA[1], &ta),
                              FunctorWithoutAllocator(),
                              &ta);
                const Obj& Y = mY;
                Obj        mX(Y, &ta);
                const Obj& X = mX;

                ASSERT(&ta == X.allocator());
                ASSERT(&ta == X.iterator().allocator());
                ASSERT(bsl::reverse_iterator<int *>(&DATA[1]) == X.iterator());
                ASSERT(DATA[0] == *X);
            }
        }

        if (veryVerbose) {
            cout << "functor with allocator, iterator without allocator\n";
        }
        {
            typedef TransformIterator<FunctorWithAllocator,
                                      IteratorWithoutAllocator>
                Obj;

            if (veryVerbose) {
                cout << "\tTI()\n";
            }
            {
                Obj        mX;
                const Obj& X = mX;

                ASSERT(bslma::Default::allocator(0) ==
                       bslma::Default::allocator(X.allocator()));
                ASSERT(bslma::Default::allocator(0) ==
                       bslma::Default::allocator(X.functor().allocator()));
            }

            if (veryVerbose) {
                cout << "\tTI(Allocator *)\n";
            }
            {
                Obj        mX(&ta);
                const Obj& X = mX;

                ASSERT(&ta == X.allocator());
                ASSERT(&ta == X.functor().allocator());
            }

            if (veryVerbose) {
                cout << "\tTI(const ITERATOR&, FUNCTOR, Allocator *)\n";
            }
            {
                Obj        mX((IteratorWithoutAllocator(&DATA[1])),
                              FunctorWithAllocator(&ta),
                              &ta);
                const Obj& X = mX;

                ASSERT(&ta == X.allocator());
                ASSERT(&ta == X.functor().allocator());
                ASSERT(bsl::reverse_iterator<int *>(&DATA[1]) == X.iterator());
                ASSERT(DATA[0] == *X);

                const Obj& Y = TransformIteratorUtil::make(
                    X.iterator(), X.functor(), X.allocator());

                ASSERT(&ta == Y.allocator() || 0 == Y.allocator());
                ASSERT(bsl::reverse_iterator<int *>(&DATA[1]) == Y.iterator());
                ASSERT(DATA[0] == *Y);
            }

            if (veryVerbose) {
                cout << "\tTI(const TI&, Allocator *)\n";
            }
            {
                Obj        mY((IteratorWithoutAllocator(&DATA[1])),
                              FunctorWithAllocator(&ta),
                              &ta);
                const Obj& Y = mY;
                Obj        mX(Y, &ta);
                const Obj& X = mX;

                ASSERT(&ta == X.allocator());
                ASSERT(&ta == X.functor().allocator());
                ASSERT(bsl::reverse_iterator<int *>(&DATA[1]) == X.iterator());
                ASSERT(DATA[0] == *X);
            }
        }

        if (veryVerbose) {
            cout << "functor without allocator, iterator without allocator\n";
        }
        {
            typedef TransformIterator<FunctorWithoutAllocator,
                                      IteratorWithoutAllocator>
                Obj;

            if (veryVerbose) {
                cout << "\tTI()\n";
            }
            {
                Obj mX;  (void)mX;
            }

            if (veryVerbose) {
                cout << "\tTI(Allocator *)\n";
            }
            {
                Obj mX(&ta);  (void)mX;
            }

            if (veryVerbose) {
                cout << "\tTI(const ITERATOR&, FUNCTOR, Allocator *)\n";
            }
            {
                Obj        mX((IteratorWithoutAllocator(&DATA[1])),
                              FunctorWithoutAllocator(),
                              &ta);
                const Obj& X = mX;

                ASSERT(bsl::reverse_iterator<int *>(&DATA[1]) == X.iterator());
                ASSERT(DATA[0] == *X);

                Obj        mY =
                      TransformIteratorUtil::make(mX.iterator(), mX.functor());
                const Obj& Y  = mY;

                ASSERT(bsl::reverse_iterator<int *>(&DATA[1]) == Y.iterator());
                ASSERT(DATA[0] == *Y);
            }

            if (veryVerbose) {
                cout << "\tTI(const TI&, Allocator *)\n";
            }
            {
                Obj        mY((IteratorWithoutAllocator(&DATA[1])),
                              FunctorWithoutAllocator(),
                              &ta);
                const Obj& Y = mY;
                Obj        mX(Y, &ta);
                const Obj& X = mX;

                ASSERT(bsl::reverse_iterator<int *>(&DATA[1]) == X.iterator());
                ASSERT(DATA[0] == *X);
            }
        }
      } break;
      case 1: {
        // --------------------------------------------------------------------
        // BREATHING TEST
        //
        // Concerns:
        //: 1 Basic functionality of this component works as expected.
        //
        // Plan:
        //: 1 Exercise some instances and verify their operation.  (C-1)
        //
        // Testing:
        //   BREATHING TEST
        // --------------------------------------------------------------------
        if (verbose) cout << "\nBREATHING TEST"
                             "\n==============\n";

        if (veryVerbose) {
            cout << "Simple Transformation\n";
        }
        {
            int    DATA[]   = {-1, 1, -2, 2, 3};
            size_t NUM_DATA = sizeof DATA / sizeof *DATA;

            int (*abs)(int) = &bsl::abs;

            if (veryVerbose) {
                cout << "\tusing function pointer\n";
            }
            {
                typedef TransformIterator<int (*)(int), int *> Obj;

                Obj begin(DATA + 0, abs);
                Obj end(DATA + NUM_DATA, abs);

                ASSERT(9 == bsl::accumulate(begin, end, 0));
            }
            if (veryVerbose) {
                cout << "\tusing bsl::function\n";
            }
            {
                typedef TransformIterator<bsl::function<int(int)>, int *> Obj;

                Obj begin(DATA + 0, abs);
                Obj end(DATA + NUM_DATA, abs);

                ASSERT(9 == bsl::accumulate(begin, end, 0));
            }
        }

        if (veryVerbose) {
            cout << "Stateful Functor\n";
        }
        {
            bslma::TestAllocator ta(veryVeryVeryVerbose);
            if (veryVerbose) {
                cout << "\tusing function object and allocators\n";
            }
            {
                typedef TransformIterator<Parenthesizer, bsl::string *> Obj;

                bsl::string DATA[]   = {"1", "2", "3"};
                size_t      NUM_DATA = sizeof DATA / sizeof *DATA;

                Parenthesizer parenthesizer(&ta);

                Obj begin(DATA + 0, parenthesizer, &ta);
                Obj end(DATA + NUM_DATA, parenthesizer, &ta);

                bsl::string result;
                for (; begin != end; ++begin) {
                    result += *begin;
                }
                ASSERT("(1)((2))(((3)))" == result);
                LOOP_ASSERT(DATA[0], "(1)" == DATA[0]);
                LOOP_ASSERT(DATA[1], "((2))" == DATA[1]);
                LOOP_ASSERT(DATA[2], "(((3)))" == DATA[2]);
            }
            if (veryVerbose) {
                cout << "\tusing bsl::function and allocators\n";
            }
            {
                typedef TransformIterator<
                    bsl::function<bsl::string&(bsl::string&)>,
                    bsl::string *>
                    Obj;

                bsl::string DATA[]   = {"1", "2", "3"};
                size_t      NUM_DATA = sizeof DATA / sizeof *DATA;

                Parenthesizer parenthesizer(&ta);

                Obj begin(DATA + 0, parenthesizer, &ta);
                Obj end(DATA + NUM_DATA, parenthesizer, &ta);

                bsl::string result;
                for (; begin != end; ++begin) {
                    result += *begin;
                }
                ASSERT("(1)((2))(((3)))" == result);
                LOOP_ASSERT(DATA[0], "(1)" == DATA[0]);
                LOOP_ASSERT(DATA[1], "((2))" == DATA[1]);
                LOOP_ASSERT(DATA[2], "(((3)))" == DATA[2]);
            }
        }
      } break;
      default: {
         cerr << "WARNING: CASE '" << test << "' NOT FOUND." << endl;
         testStatus = -1;
      }
    }

    LOOP_ASSERT(globalAllocator.numBlocksTotal(),
                0 == globalAllocator.numBlocksTotal());

    if (testStatus > 0) {
        cerr << "Error, non-zero test status = " << testStatus << "." << endl;
    }

    return testStatus;
}

// ----------------------------------------------------------------------------
// Copyright 2018 Bloomberg Finance L.P.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ----------------------------- END-OF-FILE ----------------------------------
