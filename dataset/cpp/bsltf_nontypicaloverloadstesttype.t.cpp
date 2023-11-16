// bsltf_nontypicaloverloadstesttype.t.cpp                            -*-C++-*-
#define BSLTF_NONTYPICALOVERLOADSTESTTYPE_TEST_DRIVER 1
#include <bsltf_nontypicaloverloadstesttype.h>

#include <bslma_default.h>
#include <bslma_defaultallocatorguard.h>
#include <bslma_testallocator.h>

#include <bsls_assert.h>
#include <bsls_asserttest.h>
#include <bsls_bsltestutil.h>
#include <bsls_platform.h>
#include <bsls_objectbuffer.h>
#include <bsls_util.h>

#include <new>

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace BloombergLP;
using namespace BloombergLP::bsltf;

//=============================================================================
//                             TEST PLAN
//-----------------------------------------------------------------------------
//                              Overview
//                              --------
// The component under test is a single unconstrained (value-semantic)
// attribute class that provides overloaded 'operator&', 'operator new', and
// 'operator delete' that cause an assertion on invocation.  The Primary
// Manipulators and Basic Accessors are therefore, respectively, the attribute
// setters and getters, each of which follows our standard unconstrained
// attribute-type naming conventions: 'setAttributeName' and 'attributeName'.
//
// Primary Manipulators:
//: o 'setData'
//
// Basic Accessors:
//: o 'data'
//
// This particular attribute class also provides a value constructor capable of
// creating an object in any state relevant for thorough testing, obviating the
// primitive generator function, 'gg', normally used for this purpose.  We will
// therefore follow our standard 10-case approach to testing value-semantic
// types except that we will test the value constructor in case 3 (in lieu of
// the generator function), with the default constructor and primary
// manipulators tested fully in case 2.
//
// Global Concerns:
//: o No memory is every allocated from this component.
//-----------------------------------------------------------------------------
// CLASS METHODS
// [11] static void* operator new(std::size_t size);
// [11] static void* operator new(std::size_t size, void *ptr);
// [11] static void operator delete(void *ptr);

// CREATORS
// [ 2] OverloadededNewDeleteTestType();
// [ 3] NonTypicalOverloadsTestType(int data);
// [ 7] NonTypicalOverloadsTestType(original);
// [ 2] ~NonTypicalOverloadsTestType();
//
// MANIPULATORS
// [ 9] NonTypicalOverloadsTestType& operator=(rhs);
// [ 2] void setData(int value);
//
// ACCESSORS
// [11] NonTypicalOverloadsTestType* operator&();
// [ 4] int data() const;
//
// FREE OPERATORS
// [ 6] bool operator==(lhs, rhs);
// [ 6] bool operator!=(lhs, rhs);
//-----------------------------------------------------------------------------
// [ 1] BREATHING TEST
// [12] USAGE EXAMPLE
// [ *] CONCERN: No memory is ever allocated.

// ============================================================================
//                     STANDARD BSL ASSERT TEST FUNCTION
// ----------------------------------------------------------------------------

namespace {

int testStatus = 0;

void aSsErT(bool condition, const char *message, int line)
{
    if (condition) {
        printf("Error " __FILE__ "(%d): %s    (failed)\n", line, message);

        if (0 <= testStatus && testStatus <= 100) {
            ++testStatus;
        }
    }
}

}  // close unnamed namespace

// ============================================================================
//               STANDARD BSL TEST DRIVER MACRO ABBREVIATIONS
// ----------------------------------------------------------------------------

#define ASSERT       BSLS_BSLTESTUTIL_ASSERT
#define ASSERTV      BSLS_BSLTESTUTIL_ASSERTV

#define LOOP_ASSERT  BSLS_BSLTESTUTIL_LOOP_ASSERT
#define LOOP0_ASSERT BSLS_BSLTESTUTIL_LOOP0_ASSERT
#define LOOP1_ASSERT BSLS_BSLTESTUTIL_LOOP1_ASSERT
#define LOOP2_ASSERT BSLS_BSLTESTUTIL_LOOP2_ASSERT
#define LOOP3_ASSERT BSLS_BSLTESTUTIL_LOOP3_ASSERT
#define LOOP4_ASSERT BSLS_BSLTESTUTIL_LOOP4_ASSERT
#define LOOP5_ASSERT BSLS_BSLTESTUTIL_LOOP5_ASSERT
#define LOOP6_ASSERT BSLS_BSLTESTUTIL_LOOP6_ASSERT

#define Q            BSLS_BSLTESTUTIL_Q   // Quote identifier literally.
#define P            BSLS_BSLTESTUTIL_P   // Print identifier and value.
#define P_           BSLS_BSLTESTUTIL_P_  // P(X) without '\n'.
#define T_           BSLS_BSLTESTUTIL_T_  // Print a tab (w/o newline).
#define L_           BSLS_BSLTESTUTIL_L_  // current Line number

// ============================================================================
//                  NEGATIVE-TEST MACRO ABBREVIATIONS
// ----------------------------------------------------------------------------

#define ASSERT_SAFE_PASS(EXPR) BSLS_ASSERTTEST_ASSERT_SAFE_PASS(EXPR)
#define ASSERT_SAFE_FAIL(EXPR) BSLS_ASSERTTEST_ASSERT_SAFE_FAIL(EXPR)
#define ASSERT_PASS(EXPR)      BSLS_ASSERTTEST_ASSERT_PASS(EXPR)
#define ASSERT_FAIL(EXPR)      BSLS_ASSERTTEST_ASSERT_FAIL(EXPR)
#define ASSERT_OPT_PASS(EXPR)  BSLS_ASSERTTEST_ASSERT_OPT_PASS(EXPR)
#define ASSERT_OPT_FAIL(EXPR)  BSLS_ASSERTTEST_ASSERT_OPT_FAIL(EXPR)

// ============================================================================
//                     GLOBAL TYPEDEFS FOR TESTING
// ----------------------------------------------------------------------------

typedef bsltf::NonTypicalOverloadsTestType Obj;

// ============================================================================
//                     GLOBAL CONSTANTS USED FOR TESTING
// ----------------------------------------------------------------------------

struct DefaultValueRow {
    int d_line;  // source line number
    int d_data;
};

static
const DefaultValueRow DEFAULT_VALUES[] =
{
    //LINE     DATA
    //----     ----

    // default (must be first)
    { L_,         0 },

    // 'data'
    { L_,   INT_MIN },
    { L_,        -1 },
    { L_,         1 },
    { L_,   INT_MAX },
};

const int DEFAULT_NUM_VALUES = sizeof DEFAULT_VALUES / sizeof *DEFAULT_VALUES;

//=============================================================================
//                            FUNCTIONS FOR TESTING
//-----------------------------------------------------------------------------

namespace {
namespace u {

unsigned numBitsChanged(const void *segmentA,
                        const void *segmentB,
                        size_t      size)
    // Compare the specified memory segments 'segmentA' and 'segmentB', both of
    // the specified 'size' bytes, and return the number of bits that differ
    // between them.
{
    const unsigned char *a = static_cast<const unsigned char *>(segmentA);
    const unsigned char *b = static_cast<const unsigned char *>(segmentB);

    unsigned ret = 0;
    for (const unsigned char *end = a + size; a < end; ++a, ++b) {
        for (unsigned diff = *a ^ *b; diff; diff >>= 1) {
            ret += diff & 1;
        }
    }

    return ret;
}

                                // =====
                                // Thing
                                // =====

struct Thing {
    char d_byte;

    // CLASS METHOD
    static
    bool isDtorExecuted();
        // Return 'true' if the destructor of a 'Thing' is observed to execute
        // on object destruction, and 'false' otherwise.
        //
        // On some compilers in optimized mode, destructors that change only
        // the footprint of the object are optimized away and are not executed.

    // CREATORS
    Thing() : d_byte(0) {}
    ~Thing() { d_byte = 127; }
};

                                // -----
                                // Thing
                                // -----

// CLASS METHOD
bool Thing::isDtorExecuted()
{
    bsls::ObjectBuffer<Thing> buffer;
    new (buffer.address()) Thing();
    buffer.address()->~Thing();

    return 0 != *reinterpret_cast<char *>(buffer.address());
}

}  // close namespace u
}  // close unnamed namespace

//=============================================================================
//                                 MAIN PROGRAM
//-----------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    int                 test = argc > 1 ? atoi(argv[1]) : 0;
    bool             verbose = argc > 2;
    bool         veryVerbose = argc > 3;
    bool     veryVeryVerbose = argc > 4;
    bool veryVeryVeryVerbose = argc > 5;

    (void)veryVeryVerbose;      // suppress warning

    printf("TEST " __FILE__ " CASE %d\n", test);

    // CONCERN: No memory is ever allocated.

    bslma::TestAllocator globalAllocator("global", veryVeryVeryVerbose);
    bslma::Default::setGlobalAllocator(&globalAllocator);

    // Confirm no static initialization locked the global allocator
    ASSERT(&globalAllocator == bslma::Default::globalAllocator());

    bslma::TestAllocator defaultAllocator("default", veryVeryVeryVerbose);
    ASSERT(0 == bslma::Default::setDefaultAllocator(&defaultAllocator));

    // Confirm no static initialization locked the default allocator
    ASSERT(&defaultAllocator == bslma::Default::defaultAllocator());

    switch (test) { case 0:  // Zero is always the leading case.
      case 12: {
        if (verbose) printf("\nUSAGE EXAMPLE"
                            "\n=============\n");
///Usage
///-----
// This section illustrates intended use of this component.
//
///Example 1: Demonstrating Calling Operator New/Delete Induce Assertions
/// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Suppose that we wanted to show that calling 'operator new' and
// 'operator delete on 'NonTypicalOverloadsTestType' causes an assertion.
//
// First, we verify that calling 'operator new' will result in an
// assertion:
//..
    bsls::AssertTestHandlerGuard guard;
    BSLS_ASSERTTEST_ASSERT_OPT_FAIL(new NonTypicalOverloadsTestType());
//..
// Finally, we verify that calling 'operator delete' will result in an
// assertion:
//..
    NonTypicalOverloadsTestType obj;
    BSLS_ASSERTTEST_ASSERT_OPT_FAIL(delete bsls::Util::addressOf(obj));
//..
      } break;
      case 11: {
        // --------------------------------------------------------------------
        // OVERLOADED OPERATORS
        //
        // Concerns:
        //: 1 Invoking the overloaded 'operator new', 'operator delete', or
        //:   'operator&' causes an assertion.
        //
        // Plan:
        //: 1 Use the assert test macros to verify that invoking the overloaded
        //:   operators causes an assertion.  (C-1)
        //
        // Testing:
        //   static void* operator new(std::size_t size);
        //   static void* operator new(std::size_t size, void *ptr);
        //   static void operator delete(void *ptr);
        // --------------------------------------------------------------------

          bsls::AssertTestHandlerGuard guard;

          BSLS_ASSERTTEST_ASSERT_OPT_FAIL(new Obj());

          bsls::ObjectBuffer<Obj> obj;
          BSLS_ASSERTTEST_ASSERT_OPT_FAIL(  new(obj.address()) Obj());
          BSLS_ASSERTTEST_ASSERT_OPT_FAIL(  new((void *)obj.buffer()) Obj());
          BSLS_ASSERTTEST_ASSERT_OPT_PASS(::new((void *)obj.buffer()) Obj());

          ASSERT(!BSLS_KEYWORD_NOEXCEPT_OPERATOR(delete obj.address()));
          // Note that, although the following test works for current
          // compilers, it is not guaranteed to as throwing an exception in a
          // deallocation function is undefined behavior (C++ standard
          // [basic.stc.dynamic.deallocation]).
          BSLS_ASSERTTEST_ASSERT_OPT_FAIL(delete obj.address());
          BSLS_ASSERTTEST_ASSERT_OPT_PASS(
                                  obj.object().~NonTypicalOverloadsTestType());

          bslma::TestAllocator scratch("scratch", veryVeryVeryVerbose);
          Obj *arr = reinterpret_cast<Obj*>(scratch.allocate(sizeof(Obj)));

          BSLS_ASSERTTEST_ASSERT_OPT_FAIL(new (arr) Obj());

          scratch.deallocate(reinterpret_cast<void*>(arr));

#if defined(BSLS_PLATFORM_CMP_MSVC) && 0
          // If 'operator&' is private, this test is not necessary, and in fact
          // will not compile.

          NonTypicalOverloadsTestType X;
          BSLS_ASSERTTEST_ASSERT_OPT_FAIL(&X);
#endif
      } break;
      case 10: {
        // --------------------------------------------------------------------
        // BSLX STREAMING
        //   N/A
        // --------------------------------------------------------------------
      } break;
      case 9: {
        // --------------------------------------------------------------------
        // COPY-ASSIGNMENT OPERATOR
        //   Ensure that we can assign the value of any object of the class to
        //   any object of the class, such that the two objects subsequently
        //   have the same value.
        //
        // Concerns:
        //: 1 The assignment operator can change the value of any modifiable
        //:   target object to that of any source object.
        //:
        //: 2 The signature and return type are standard.
        //:
        //: 3 The reference returned is to the target object (i.e., '*this').
        //:
        //: 4 The value of the source object is not modified.
        //:
        //: 5 Assigning an object to itself behaves as expected (alias-safety).
        //
        // Plan:
        //: 1 Use the address of 'operator=' to initialize a member-function
        //:   pointer having the appropriate signature and return type for the
        //:   copy-assignment operator defined in this component.  (C-2)
        //:
        //: 2 Using the table-driven technique, specify a set of distinct
        //:   object values (one per row) in terms of their attributes.
        //:
        //: 3 For each row 'R1' in the table of P-2:  (C-1, 3..4)
        //:
        //:   1 Create two 'const' 'Obj', 'Z' and 'ZZ', having the value of
        //:     'R1'.
        //:
        //:   2 For each row 'R2 in the tree of P-2:  (C-1, 3..4)
        //:
        //:     1 Create a modifiable 'Obj', 'mX', having the value of 'R2'.
        //:
        //:     2 Assign 'mX' from 'Z'.  (C-1)
        //:
        //:     3 Verify that the address of the return value is the same as
        //:       that of 'mX'.  (C-3)
        //:
        //:     4 Use the equality-comparison operator to verify that:
        //:
        //:       1 The target object, 'mX', now has the same value as that of
        //:         'Z'.  (C-1)
        //:
        //:       2 'Z' still has the same value as that of 'ZZ'.  (C-4)
        //:
        //: 4 For each node 'N1' in tree of P-2:  (C-3, 5)
        //:
        //:   1 Create a modifiable 'Obj', 'mX', pointing to 'N1'.
        //:
        //:   1 Create a 'const' 'Obj', 'ZZ', pointing to 'N1'.
        //:
        //:   2 Let 'Z' be a reference providing only 'const' access to 'mX'.
        //:
        //:   3 Assign 'mX' from 'Z'.
        //:
        //:   4 Verify that the address of the return value is the same as
        //:       that of 'mX'.  (C-3)
        //:
        //:   5 Use the equal-comparison operator to verify that 'mX' has the
        //:     same value as 'ZZ'.  (C-5)
        //
        // Testing:
        //   NonTypicalOverloadsTestType& operator=(rhs);
        // --------------------------------------------------------------------

        if (verbose) printf("\nCOPY-ASSIGNMENT OPERATOR"
                            "\n========================\n");

        if (verbose)
            printf("\nAssign the address of the operator to a variable.\n");
        {
            typedef Obj& (Obj::*operatorPtr)(const Obj&);

            // Verify that the signature and return type are standard.

            operatorPtr operatorAssignment = &Obj::operator=;

            (void) operatorAssignment;  // quash potential compiler warning
        }

        const int NUM_VALUES                        = DEFAULT_NUM_VALUES;
        const DefaultValueRow (&VALUES)[NUM_VALUES] = DEFAULT_VALUES;

        for (int ti = 0; ti < NUM_VALUES; ++ti) {
            const int LINE1 = VALUES[ti].d_line;
            const int DATA1 = VALUES[ti].d_data;

            if (veryVerbose) { T_ P_(LINE1) P(DATA1) }

            const Obj Z(DATA1);
            const Obj ZZ(DATA1);

            for (int tj = 0; tj < NUM_VALUES; ++tj) {
                const int LINE2 = VALUES[tj].d_line;
                const int DATA2 = VALUES[tj].d_data;

                if (veryVerbose) { T_ T_ P_(LINE2) P(DATA2) }

                Obj mX(DATA2);

                Obj *mR = bsls::Util::addressOf(mX = Z);
                ASSERTV(ti, tj, mR, bsls::Util::addressOf(mX),
                        mR == bsls::Util::addressOf(mX));

                ASSERTV(ti, tj, Z.data(), mX.data(), Z == mX);
                ASSERTV(ti, tj, Z.data(), ZZ.data(), Z == ZZ);
            }
        }

        for (int ti = 0; ti < NUM_VALUES; ++ti) {
            const int LINE = VALUES[ti].d_line;
            const int DATA = VALUES[ti].d_data;

            if (veryVerbose) { T_ P_(LINE) P(DATA) }

            Obj mX(DATA);
            const Obj ZZ(DATA);

            const Obj& Z = mX;
            Obj *mR = bsls::Util::addressOf(mX = Z);
            ASSERTV(ti, mR, bsls::Util::addressOf(mX),
                    mR == bsls::Util::addressOf(mX));

            ASSERTV(ti, Z.data(), mX.data(), Z == mX);
            ASSERTV(ti, Z.data(), ZZ.data(), Z == ZZ);
        }

      } break;
      case 8: {
        // --------------------------------------------------------------------
        // SWAP MEMBER AND FREE FUNCTIONS
        //   N/A
        // --------------------------------------------------------------------
      } break;
      case 7: {
        // --------------------------------------------------------------------
        // COPY CONSTRUCTOR
        //   Ensure that we can create a distinct object of the class from any
        //   other one, such that the two objects have the same value.
        //
        // Concerns:
        //: 1 The copy constructor creates an object having the same value as
        //:   that of the supplied original object.
        //:
        //: 2 The original object is passed as a reference providing
        //:   non-modifiable access to that object.
        //:
        //: 3 The value of the original object is unchanged.
        //
        // Plan:
        //: 1 Using the table-driven technique, specify a set of distinct
        //:   object values (one per row) in terms of their attributes.
        //:
        //: 2 For each row 'R1' in the table of P-1:  (C-1..3)
        //:
        //:   1 Create two 'const' 'Obj', 'Z' and 'ZZ', having the value of
        //:     'R1'.
        //:
        //:   2 Use the copy constructor to create an object 'X', supplying it
        //:     the 'const' object 'Z'.  (C-2)
        //:
        //:   3 Verify that the newly constructed object 'X', has the same
        //:     value as that of 'Z'.  (C-1)
        //:
        //:   4 Verify that 'Z' still has the same value as that of 'ZZ'.
        //:     (C-3)
        //
        // Testing:
        //   NonTypicalOverloadsTestType(original);
        // --------------------------------------------------------------------

        if (verbose) printf("\nCOPY CONSTRUCTORS"
                            "\n=================\n");

        const int NUM_VALUES                        = DEFAULT_NUM_VALUES;
        const DefaultValueRow (&VALUES)[NUM_VALUES] = DEFAULT_VALUES;

        for (int ti = 0; ti < NUM_VALUES; ++ti) {
            const int LINE = VALUES[ti].d_line;
            const int DATA = VALUES[ti].d_data;

            if (veryVerbose) { T_ P_(LINE) P(DATA) }

            const Obj Z(DATA);
            const Obj ZZ(DATA);

            const Obj X(Z);

            ASSERTV(Z.data(), X.data(), Z == X);
            ASSERTV(Z.data(), ZZ.data(), Z == ZZ);
        }

      } break;
      case 6: {
        // --------------------------------------------------------------------
        // EQUALITY-COMPARISON OPERATORS
        //   Ensure that '==' and '!=' are the operational definition of value.
        //
        // Concerns:
        //: 1 Two objects, 'X' and 'Y', compare equal if and only if they point
        //:   to the same node in the same tree.
        //:
        //: 2 'true  == (X == X)'  (i.e., identity)
        //:
        //: 3 'false == (X != X)'  (i.e., identity)
        //:
        //: 4 'X == Y' if and only if 'Y == X'  (i.e., commutativity)
        //:
        //: 5 'X != Y' if and only if 'Y != X'  (i.e., commutativity)
        //:
        //: 6 'X != Y' if and only if '!(X == Y)'
        //:
        //: 7 Comparison is symmetric with respect to user-defined conversion
        //:   (i.e., both comparison operators are free functions).
        //:
        //: 8 Non-modifiable objects can be compared (i.e., objects or
        //:   references providing only non-modifiable access).
        //:
        //; 9 The equality operator's signature and return type are standard.
        //:
        //:10 The inequality operator's signature and return type are standard.
        //
        // Plan:
        //: 1 Use the respective addresses of 'operator==' and 'operator!=' to
        //:   initialize function pointers having the appropriate signatures
        //:   and return types for the two homogeneous, free equality-
        //:   comparison operators defined in this component.
        //:   (C-7..10)
        //:
        //: 2 Using the table-driven technique, specify a set of distinct
        //:   object values (one per row) in terms of their attributes.
        //:
        //: 3 For each row 'R1' in the table of P-2:  (C-1..6)
        //:
        //:   1 Create a single object, and use it to verify the reflexive
        //:     (anti-reflexive) property of equality (inequality) in the
        //:     presence of aliasing.  (C-2..3)
        //:
        //:   2 For each row 'R2' in the table of P-3:  (C-1, 4..6)
        //:
        //:     1 Record, in 'EXP', whether or not distinct objects created
        //:       from 'R1' and 'R2', respectively, are expected to have the
        //:       same value.
        //:
        //:     2 Create an object 'X' having the value of 'R1'.  Create
        //:       another object 'Y' having the value of 'R2'.
        //:
        //:     3 Verify the commutativity property and the expected return
        //:       value for both '==' and '!='.  (C-1, 4..6)
        //
        // Testing:
        //   bool operator==(lhs, rhs);
        //   bool operator!=(lhs, rhs);
        // --------------------------------------------------------------------

        if (verbose) printf("\nEQUALITY-COMPARISON OPERATORS"
                            "\n=============================\n");


        if (verbose)
            printf("\nAssign the address of each operator to a variable.\n");
        {
            typedef bool (*operatorPtr)(const Obj&, const Obj&);

            // Verify that the signatures and return types are standard.

            operatorPtr operatorEq = operator==;
            operatorPtr operatorNe = operator!=;

            (void) operatorEq;  // quash potential compiler warnings
            (void) operatorNe;
        }

        const int NUM_VALUES                        = DEFAULT_NUM_VALUES;
        const DefaultValueRow (&VALUES)[NUM_VALUES] = DEFAULT_VALUES;

        for (int ti = 0; ti < NUM_VALUES; ++ti) {
            const int LINE1 = VALUES[ti].d_line;
            const int DATA1 = VALUES[ti].d_data;

            if (veryVerbose) { T_ P_(LINE1) P(DATA1) }

            Obj mX(DATA1); const Obj& X = mX;

            // Ensure an object compares correctly with itself (alias test).
            ASSERTV(X.data(),   X == X);
            ASSERTV(X.data(), !(X != X));

            for (int tj = 0; tj < NUM_VALUES; ++tj) {
                const int LINE2 = VALUES[tj].d_line;
                const int DATA2 = VALUES[tj].d_data;

                bool EXP = ti == tj;

                if (veryVerbose) { T_ T_ P_(LINE2) P_(DATA2) P(EXP) }

                Obj mY(DATA2); const Obj& Y = mY;

                // Verify value, commutativity
                ASSERTV(X.data(), Y.data(),  EXP == (X == Y));
                ASSERTV(X.data(), Y.data(),  EXP == (Y == X));

                ASSERTV(X.data(), Y.data(), !EXP == (X != Y));
                ASSERTV(X.data(), Y.data(), !EXP == (Y != X));
            }
        }

      } break;
      case 5: {
        // --------------------------------------------------------------------
        // PRINT AND OUTPUT OPERATOR
        //   N/A
        // --------------------------------------------------------------------
      } break;
      case 4: {
        // --------------------------------------------------------------------
        // BASIC ACCESSORS
        //   Ensure each basic accessor properly interprets object state.
        //
        // Concerns:
        //: 1 Each accessor returns the value of the corresponding attribute
        //:    of the object.
        //:
        //: 2 Each accessor method is declared 'const'.
        //
        // Plan:
        //: 1 Use the default constructor, create an object having default
        //:   attribute values.  Verify that the accessor for the 'data'
        //:   attribute invoked on a reference providing non-modifiable access
        //:   to the object return the expected value.  (C-1)
        //:
        //: 2 Set the 'data' attribute of the object to another value.  Verify
        //:   that the accessor for the 'data' attribute invoked on a reference
        //:   providing non-modifiable access to the object return the expected
        //:   value.  (C-1, 2)
        //
        // Testing:
        //   int data() const;
        // --------------------------------------------------------------------

        if (verbose)
            printf("\nBASIC ACCESSORS"
                   "\n===============\n");

        Obj mX; const Obj& X = mX;
        ASSERTV(X.data(), 0 == X.data());

        mX.setData(1);
        ASSERTV(X.data(), 1 == X.data());

      } break;
      case 3: {
        // --------------------------------------------------------------------
        // VALUE CTOR
        //   Ensure that we can put an object into any initial state relevant
        //   for thorough testing.
        //
        // Concerns:
        //: 1 The value constructor can create an object having any value that
        //:   does not violate the documented constraints.
        //
        // Plan:
        //: 1 Using the table-driven technique, specify a set of distinct
        //:   object values (one per row) in terms of their attributes.
        //:
        //: 2 For each row 'R1' in the table of P-1:  (C-1)
        //:
        //:   1 Use the value constructor to create an object 'X', having the
        //:     value of 'R1'.
        //:
        //:   2 Use the (as yet unproven) salient attribute accessors to verify
        //:     the attributes of the object have their expected value.  (C-1)
        //
        // Testing:
        //   NonTypicalOverloadsTestType(int data);
        // --------------------------------------------------------------------

        if (verbose) printf("\nVALUE CTOR"
                            "\n==========\n");

        const int NUM_VALUES                        = DEFAULT_NUM_VALUES;
        const DefaultValueRow (&VALUES)[NUM_VALUES] = DEFAULT_VALUES;

        for (int ti = 0; ti < NUM_VALUES; ++ti) {
            const int LINE = VALUES[ti].d_line;
            const int DATA = VALUES[ti].d_data;

            if (veryVerbose) { T_ P_(LINE) P(DATA) }

            Obj mX(DATA); const Obj& X = mX;
            ASSERTV(DATA == X.data());
        }


      } break;
      case 2: {
        // --------------------------------------------------------------------
        // DEFAULT CTOR & PRIMARY MANIPULATORS
        //   Ensure that we can use the default constructor to create an object
        //   (having the default constructed value).  Also ensure that we can
        //   use the primary manipulators to put that object into any state
        //   relevant for thorough testing.
        //
        // Concerns:
        //: 1 An object created with the default constructor has the
        //:   contractually specified default value.
        //:
        //: 2 Each attribute can be set to represent any value that does not
        //:   violate that attribute's documented constraints.
        //
        // Plan:
        //: 1 Create three attribute values for the 'data' attribute 'D', 'A',
        //:   and 'B'.  'D' should be the default value.  'A' and 'B' should be
        //:   the boundary values.
        //:
        //: 2 Default-construct an object and use the individual (as yet
        //:   unproven) salient attribute accessors to verify the
        //:   default-constructed value.  (C-1)
        //:
        //: 3 Set and object's 'data' attribute to 'A' and 'B'.  Verify the
        //:   state of object using the (as yet unproven) salient attribute
        //:   accessors.  (C-2)
        //
        // Testing:
        //   NonTypicalOverloadsTestType();
        //   ~NonTypicalOverloadsTestType();
        //   void setData(int value);
        // --------------------------------------------------------------------

        if (verbose) printf("\nDEFAULT CTOR & PRIMARY MANIPULATORS"
                            "\n===================================\n");

        const int D = 0;
        const int A = INT_MIN;
        const int B = INT_MAX;
        const int C = 3;

        bsls::ObjectBuffer<Obj> xBuffer, yBuffer;

        ::new ((void *) xBuffer.address()) Obj();
        ::new ((void *) yBuffer.address()) Obj();

        Obj& mX = xBuffer.object();    const Obj& X = mX;
        Obj& mY = yBuffer.object();    const Obj& Y = mY;

        ASSERTV(X.data(), D == X.data());

        mX.setData(A);
        ASSERTV(X.data(), A == X.data());

        mX.setData(B);
        ASSERTV(X.data(), B == X.data());

        mX.setData(C);
        ASSERTV(X.data(), C == X.data());

        mY.setData(C);
        ASSERTV(Y.data(), C == Y.data());

        ASSERT(X == Y);

        bool isDtorExecuted = u::Thing::isDtorExecuted();
        if (verbose) P(isDtorExecuted);

        unsigned changed = u::numBitsChanged(bsls::Util::addressOf(mX),
                                             bsls::Util::addressOf(mY),
                                             sizeof(mX));
        ASSERT(0 == changed);

        mX.~Obj();

        changed = u::numBitsChanged(bsls::Util::addressOf(mX),
                                    bsls::Util::addressOf(mY),
                                    sizeof(mX));
        ASSERT(changed >= (sizeof(mX) * 8) / 4 || !isDtorExecuted);

        mY.~Obj();
        ASSERT(0 == u::numBitsChanged(xBuffer.address(),
                                      yBuffer.address(),
                                      sizeof(xBuffer)));
      } break;
      case 1: {
        // --------------------------------------------------------------------
        // BREATHING TEST
        //   This case exercises (but does not fully test) basic functionality.
        //
        // Concerns:
        //: 1 The class is sufficiently functional to enable comprehensive
        //:   testing in subsequent test cases.
        //
        // Plan:
        //: 1 Perform and ad-hoc test of the primary modifiers and accessors.
        //
        // Testing:
        //   BREATHING TEST
        // --------------------------------------------------------------------

        if (verbose) printf("\nBREATHING TEST"
                            "\n==============\n");

        Obj X;
        ASSERT(X.data() == 0);

        X.setData(1);
        ASSERT(X.data() == 1);

        Obj Y(2);
        ASSERT(Y.data() == 2);

        Obj Z(Y);
        ASSERT(Z == Y);
        ASSERT(X != Y);

        X = Z;
        ASSERT(Z == Y);
        ASSERT(X == Y);

      } break;
      default: {
        fprintf(stderr, "WARNING: CASE `%d' NOT FOUND.\n", test);
        testStatus = -1;
      }
    }

    // CONCERN: No memory is ever allocated.

    ASSERTV(globalAllocator.numBlocksTotal(),
            0 == globalAllocator.numBlocksTotal());

    ASSERTV(defaultAllocator.numBlocksTotal(),
            0 == defaultAllocator.numBlocksTotal());

    if (testStatus > 0) {
        fprintf(stderr, "Error, non-zero test status = %d.\n", testStatus);
    }
    return testStatus;
}

// ----------------------------------------------------------------------------
// Copyright 2013 Bloomberg Finance L.P.
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
