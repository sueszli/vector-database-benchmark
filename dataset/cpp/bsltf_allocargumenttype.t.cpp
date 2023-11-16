// bsltf_allocargumenttype.t.cpp                                      -*-C++-*-
#include <bsltf_allocargumenttype.h>

#include <bslma_allocator.h>
#include <bslma_default.h>
#include <bslma_defaultallocatorguard.h>
#include <bslma_testallocator.h>
#include <bslma_testallocatormonitor.h>
#include <bslma_usesbslmaallocator.h>

#include <bsls_assert.h>
#include <bsls_asserttest.h>
#include <bsls_bsltestutil.h>

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

using namespace BloombergLP;
using namespace BloombergLP::bsltf;

//=============================================================================
//                             TEST PLAN
//-----------------------------------------------------------------------------
//                              Overview
//                              --------
// The component under test is a single unconstrained (value-semantic)
// attribute class.
//
// This particular attribute class also provides a value constructor capable of
// creating an object in any state relevant for thorough testing, obviating the
// primitive generator function, 'gg', normally used for this purpose.  We will
// therefore follow our standard 10-case approach to testing value-semantic
// types except that we will test the value constructor in case 3 (in lieu of
// the generator function), with the default constructor and primary
// manipulators tested fully in case 2.
//
// This particular test class does not allocate any memory from the default
// constructor.  Futhermore, when copying or moving the default-constructed
// object, there should not be no memory allocated ether.
//
// Global Concerns:
//: o In no case does memory come from the global allocator.
//-----------------------------------------------------------------------------
// CREATORS
// [ 2] AllocArgumentType(bslma::Allocator *alloc = 0);
// [ 3] AllocArgumentType(int data, bslma::Allocator *alloc = 0);
// [ 7] AllocArgumentType(const AAT& original, bslma::Allocator *alloc = 0);
// [11] AllocArgumentType(bslmf::MovableRef<AAT> original);
// [11] AllocArgumentType(bslmf::MovableRef<AAT> original, Allocator *al);
// [ 3] ~AllocArgumentType();
//
// MANIPULATORS
// [ 9] AllocArgumentType& operator=(const AllocArgumentType& rhs);
// [12] AllocArgumentType& operator=(bslmf::MovableRef<AAT> rhs);
//
// ACCESSORS
// [ 4] operator int() const;
// [ 4] bslma::Allocator *allocator() const;
// [13] MoveState::Enum movedInto() const;
// [13] MoveState::Enum movedFrom() const;
//-----------------------------------------------------------------------------
// [ 1] BREATHING TEST
// [  ] USAGE EXAMPLE
// [ 2] CONCERN: Default constructor does not allocate.
// [ *] CONCERN: In no case does memory come from the global allocator.
// [10] CONCERN: The object has the necessary type traits

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

typedef bsltf::AllocArgumentType<1> Obj;

typedef bslmf::MovableRefUtil       MoveUtil;

// ============================================================================
//                     GLOBAL CONSTANTS USED FOR TESTING
// ----------------------------------------------------------------------------

struct DefaultDataRow {
    int d_line;  // source line number
    int d_value;
};

static
const DefaultDataRow DEFAULT_DATA[] =
{
    //LINE  VALUE
    //----  --------
    { L_,         0 },
    { L_,         1 },
    { L_,       512 },
    { L_,   INT_MAX },
};
const size_t DEFAULT_NUM_DATA = sizeof DEFAULT_DATA / sizeof *DEFAULT_DATA;

//=============================================================================
//                                USAGE EXAMPLE
//-----------------------------------------------------------------------------


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

    (void)veryVerbose;          // suppress warning
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
      case 13: {
        // --------------------------------------------------------------------
        // TESTING 'movedFrom' AND 'movedInto' METHODS
        //   Ensure that move attributes are set to the correct values after
        //   object construction, copy-assignement, and move-assignment
        //   operations.
        //
        // Concerns:
        //: 1 Each accessor returns the value of the corresponding attribute
        //:   of the object.
        //:
        //: 2 Each accessor method is declared 'const'.
        //
        // Plan:
        //: 1 Manually create and/or assign a value to an object.  Verify that
        //:   the accessors for the 'movedFrom' and 'movedInto' attributes
        //:   invoked on a reference providing non-modifiable access to the
        //:   object return the expected value.  (C-1,2)
        //
        // Testing:
        //   MoveState::Enum movedInto() const;
        //   MoveState::Enum movedFrom() const;
        // --------------------------------------------------------------------
        if (verbose) printf("\nTESTING 'movedFrom' AND 'movedInto' METHODS"
                            "\n===========================================\n");

        if (verbose) printf("\nTesting default and value constructor.\n");
        {
            Obj mX; const Obj& X = mX;

            ASSERTV(MoveState::e_NOT_MOVED == X.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == X.movedFrom());
        }
        {
            bslma::TestAllocator oa("object", veryVeryVeryVerbose);
            Obj mX(&oa); const Obj& X = mX;

            ASSERTV(MoveState::e_NOT_MOVED == X.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == X.movedFrom());
        }
        {
            bslma::TestAllocator oa("object", veryVeryVeryVerbose);
            Obj mX(1, &oa); const Obj& X = mX;

            ASSERTV(MoveState::e_NOT_MOVED == X.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == X.movedFrom());
        }

        if (verbose) printf("\nTesting copy constructor.\n");
        {
            Obj mX;    const Obj& X = mX;
            Obj mY(X); const Obj& Y = mY;

            ASSERTV(MoveState::e_NOT_MOVED == X.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == X.movedFrom());
            ASSERTV(MoveState::e_NOT_MOVED == Y.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == Y.movedFrom());
        }
        {
            bslma::TestAllocator oa("object", veryVeryVeryVerbose);
            Obj mX(&oa); const Obj& X = mX;
            Obj mY(X);   const Obj& Y = mY;

            ASSERTV(MoveState::e_NOT_MOVED == X.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == X.movedFrom());
            ASSERTV(MoveState::e_NOT_MOVED == Y.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == Y.movedFrom());
        }
        {
            bslma::TestAllocator oa("object", veryVeryVeryVerbose);
            Obj mX(1, &oa); const Obj& X = mX;
            Obj mY(X);      const Obj& Y = mY;

            ASSERTV(MoveState::e_NOT_MOVED == X.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == X.movedFrom());
            ASSERTV(MoveState::e_NOT_MOVED == Y.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == Y.movedFrom());
        }

        if (verbose) printf("\nTesting copy-assignment.\n");
        {
            Obj mX; const Obj& X = mX;
            Obj mY; const Obj& Y = mY;

            mY = X;

            ASSERTV(MoveState::e_NOT_MOVED == X.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == X.movedFrom());
            ASSERTV(MoveState::e_NOT_MOVED == Y.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == Y.movedFrom());
        }
        {
            bslma::TestAllocator oa("object", veryVeryVeryVerbose);
            Obj mX(&oa); const Obj& X = mX;
            Obj mY;      const Obj& Y = mY;

            mY = X;

            ASSERTV(MoveState::e_NOT_MOVED == X.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == X.movedFrom());
            ASSERTV(MoveState::e_NOT_MOVED == Y.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == Y.movedFrom());
        }
        {
            bslma::TestAllocator oa("object", veryVeryVeryVerbose);
            Obj mX(1, &oa); const Obj& X = mX;
            Obj mY;         const Obj& Y = mY;

            mY = X;

            ASSERTV(MoveState::e_NOT_MOVED == X.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == X.movedFrom());
            ASSERTV(MoveState::e_NOT_MOVED == Y.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == Y.movedFrom());
        }
        {
            bslma::TestAllocator oa("object", veryVeryVeryVerbose);
            Obj mX(1);      const Obj& X = mX;
            Obj mY(2, &oa); const Obj& Y = mY;

            mY = X;

            ASSERTV(MoveState::e_NOT_MOVED == X.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == X.movedFrom());
            ASSERTV(MoveState::e_NOT_MOVED == Y.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == Y.movedFrom());
        }
        {
            bslma::TestAllocator oa1("object1", veryVeryVeryVerbose);
            bslma::TestAllocator oa2("object2", veryVeryVeryVerbose);
            Obj mX(1, &oa1); const Obj& X = mX;
            Obj mY(2, &oa2); const Obj& Y = mY;

            mY = X;

            ASSERTV(MoveState::e_NOT_MOVED == X.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == X.movedFrom());
            ASSERTV(MoveState::e_NOT_MOVED == Y.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == Y.movedFrom());
        }

        if (verbose) printf("\nTesting move constructor.\n");
        {
            Obj mX;                     const Obj& X = mX;
            Obj mY(MoveUtil::move(mX)); const Obj& Y = mY;

            ASSERT(MoveState::e_NOT_MOVED == X.movedInto());
            ASSERT(MoveState::e_MOVED     == X.movedFrom());
            ASSERT(MoveState::e_MOVED     == Y.movedInto());
            ASSERT(MoveState::e_NOT_MOVED == Y.movedFrom());
        }
        {
            bslma::TestAllocator oa("object", veryVeryVeryVerbose);
            Obj mX(&oa);                const Obj& X = mX;
            Obj mY(MoveUtil::move(mX)); const Obj& Y = mY;

            ASSERT(MoveState::e_NOT_MOVED == X.movedInto());
            ASSERT(MoveState::e_MOVED     == X.movedFrom());
            ASSERT(MoveState::e_MOVED     == Y.movedInto());
            ASSERT(MoveState::e_NOT_MOVED == Y.movedFrom());
        }
        {
            bslma::TestAllocator oa("object", veryVeryVeryVerbose);
            Obj mX(1, &oa);             const Obj& X = mX;
            Obj mY(MoveUtil::move(mX)); const Obj& Y = mY;

            ASSERT(MoveState::e_NOT_MOVED == X.movedInto());
            ASSERT(MoveState::e_MOVED     == X.movedFrom());
            ASSERT(MoveState::e_MOVED     == Y.movedInto());
            ASSERT(MoveState::e_NOT_MOVED == Y.movedFrom());
        }

        if (verbose) printf("\nTesting move-assignment.\n");
        {
            Obj mX; const Obj& X = mX;
            Obj mY; const Obj& Y = mY;

            mY = MoveUtil::move(mX);

            ASSERTV(MoveState::e_NOT_MOVED == X.movedInto());
            ASSERTV(MoveState::e_MOVED     == X.movedFrom());
            ASSERTV(MoveState::e_MOVED     == Y.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == Y.movedFrom());

            mX = MoveUtil::move(mY);

            ASSERTV(MoveState::e_MOVED     == X.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == X.movedFrom());
            ASSERTV(MoveState::e_NOT_MOVED == Y.movedInto());
            ASSERTV(MoveState::e_MOVED     == Y.movedFrom());
        }
        {
            bslma::TestAllocator oa("object", veryVeryVeryVerbose);
            Obj mX(&oa); const Obj& X = mX;
            Obj mY;      const Obj& Y = mY;

            mY = MoveUtil::move(mX);

            ASSERTV(MoveState::e_NOT_MOVED == X.movedInto());
            ASSERTV(MoveState::e_MOVED     == X.movedFrom());
            ASSERTV(MoveState::e_MOVED     == Y.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == Y.movedFrom());

            mX = MoveUtil::move(mY);

            ASSERTV(MoveState::e_MOVED     == X.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == X.movedFrom());
            ASSERTV(MoveState::e_NOT_MOVED == Y.movedInto());
            ASSERTV(MoveState::e_MOVED     == Y.movedFrom());
        }
        {
            bslma::TestAllocator oa("object", veryVeryVeryVerbose);
            Obj mX(1, &oa); const Obj& X = mX;
            Obj mY;         const Obj& Y = mY;

            mY = MoveUtil::move(mX);

            ASSERTV(MoveState::e_NOT_MOVED == X.movedInto());
            ASSERTV(MoveState::e_MOVED     == X.movedFrom());
            ASSERTV(MoveState::e_MOVED     == Y.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == Y.movedFrom());

            mX = MoveUtil::move(mY);

            ASSERTV(MoveState::e_MOVED     == X.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == X.movedFrom());
            ASSERTV(MoveState::e_NOT_MOVED == Y.movedInto());
            ASSERTV(MoveState::e_MOVED     == Y.movedFrom());
        }
        {
            bslma::TestAllocator oa("object", veryVeryVeryVerbose);
            Obj mX(1);      const Obj& X = mX;
            Obj mY(2, &oa); const Obj& Y = mY;

            mY = MoveUtil::move(mX);

            ASSERTV(MoveState::e_NOT_MOVED == X.movedInto());
            ASSERTV(MoveState::e_MOVED     == X.movedFrom());
            ASSERTV(MoveState::e_MOVED     == Y.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == Y.movedFrom());

            mX = MoveUtil::move(mY);

            ASSERTV(MoveState::e_MOVED     == X.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == X.movedFrom());
            ASSERTV(MoveState::e_NOT_MOVED == Y.movedInto());
            ASSERTV(MoveState::e_MOVED     == Y.movedFrom());
        }
        {
            bslma::TestAllocator oa("object", veryVeryVeryVerbose);
            Obj mX(1, &oa); const Obj& X = mX;
            Obj mY(2, &oa); const Obj& Y = mY;

            mY = MoveUtil::move(mX);

            ASSERTV(MoveState::e_NOT_MOVED == X.movedInto());
            ASSERTV(MoveState::e_MOVED     == X.movedFrom());
            ASSERTV(MoveState::e_MOVED     == Y.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == Y.movedFrom());

            mX = MoveUtil::move(mY);

            ASSERTV(MoveState::e_MOVED     == X.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == X.movedFrom());
            ASSERTV(MoveState::e_NOT_MOVED == Y.movedInto());
            ASSERTV(MoveState::e_MOVED     == Y.movedFrom());
        }
        {
            bslma::TestAllocator oa1("object1", veryVeryVeryVerbose);
            bslma::TestAllocator oa2("object2", veryVeryVeryVerbose);
            Obj mX(1, &oa1); const Obj& X = mX;
            Obj mY(2, &oa2); const Obj& Y = mY;

            mY = MoveUtil::move(mX);

            ASSERTV(MoveState::e_NOT_MOVED == X.movedInto());
            ASSERTV(MoveState::e_MOVED     == X.movedFrom());
            ASSERTV(MoveState::e_MOVED     == Y.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == Y.movedFrom());

            mX = MoveUtil::move(mY);

            ASSERTV(MoveState::e_MOVED     == X.movedInto());
            ASSERTV(MoveState::e_NOT_MOVED == X.movedFrom());
            ASSERTV(MoveState::e_NOT_MOVED == Y.movedInto());
            ASSERTV(MoveState::e_MOVED     == Y.movedFrom());
        }
      } break;
      case 12: {
        // --------------------------------------------------------------------
        // MOVE-ASSIGNMENT OPERATOR
        //   Ensure that we can assign the value of any object of the class to
        //   any object of the class, such that the two objects subsequently
        //   have the same value.
        //
        // Concerns:
        //: 1 The assignment operator can change the value of any modifiable
        //:   target object to that of any source object.
        //:
        //: 2 The allocator address held by the target object is unchanged.
        //:
        //: 3 Any memory allocation is from the target object's allocator.
        //:
        //: 4 The signature and return type are standard.
        //:
        //: 5 The reference returned is to the target object (i.e., '*this').
        //:
        //: 6 The value of the source object is not modified.
        //:
        //: 7 The allocator address held by the source object is unchanged.
        //:
        //: 8 QoI: Assigning a source object having the default-constructed
        //:   value allocates no memory.
        //:
        //: 9 Any memory allocation is exception neutral.
        //:
        //:10 Assigning an object to itself behaves as expected (alias-safety).
        //:
        //:11 Every object releases any allocated memory at destruction.
        //
        // Plan:
        //: 1 Use the address of 'operator=' to initialize a member-function
        //:   pointer having the appropriate signature and return type for the
        //:   move-assignment operator defined in this component.  (C-4)
        //:
        //: 2 Create a 'bslma::TestAllocator' object, and install it as the
        //:   default allocator (note that a ubiquitous test allocator is
        //:   already installed as the global allocator).
        //:
        //: 3 Using the table-driven technique, specify a set of distinct
        //:   object values (one per row) in terms of their attributes.
        //:
        //: 4 For each row 'R1' (representing a distinct object value, 'V') in
        //:   the table described in P-3:  (C-1..2, 5..8, 11)
        //:
        //:   1 Use the value constructor and a "scratch" allocator to create
        //:     two 'const' 'Obj', 'Z' and 'ZZ', each having the value 'V'.
        //:
        //:   2 Execute an inner loop that iterates over each row 'R2'
        //:     (representing a distinct object value, 'W') in the table
        //:     described in P-3:
        //:
        //:   3 For each of the iterations (P-4.2):  (C-1..2, 5..8, 11)
        //:
        //:     1 Create a 'bslma::TestAllocator' object, 'oa'.
        //:
        //:     2 Use the value constructor and 'oa' to create a modifiable
        //:       'Obj', 'mX', having the value 'W'.
        //:
        //:     3 Assign 'mX' from 'Z' in the presence of injected exceptions
        //:       (using the 'BSLMA_TESTALLOCATOR_EXCEPTION_TEST_*' macros).
        //:
        //:     4 Verify that the address of the return value is the same as
        //:       that of 'mX'.  (C-5)
        //:
        //:     5 Use the equality-comparison operator to verify that: (C-1, 6)
        //:
        //:       1 The target object, 'mX', now has the same value as that of
        //:         'Z'.  (C-1)
        //:
        //:       2 'Z' still has the same value as that of 'ZZ'.  (C-6)
        //:
        //:     6 Use the 'allocator' accessor of both 'mX' and 'Z' to verify
        //:       that the respective allocator addresses held by the target
        //:       and source objects are unchanged.  (C-2, 7)
        //:
        //:     7 Use the appropriate test allocators to verify that:
        //:       (C-8, 11)
        //:
        //:       1 For an object that (a) is initialized with a value that did
        //:         NOT require memory allocation, and (b) is then assigned a
        //:         value that DID require memory allocation, the target object
        //:         DOES allocate memory from its object allocator only
        //:         (irrespective of the specific number of allocations or the
        //:         total amount of memory allocated); also cross check with
        //:         what is expected for 'mX' and 'Z'.
        //:
        //:       2 An object that is assigned a value that did NOT require
        //:         memory allocation, does NOT allocate memory from its object
        //:         allocator; also cross check with what is expected for 'Z'.
        //:
        //:       3 No additional memory is allocated by the source object.
        //:         (C-8)
        //:
        //:       4 All object memory is released when the object is destroyed.
        //:         (C-11)
        //:
        //: 5 Repeat steps similar to those described in P-2 except that, this
        //:   time, there is no inner loop (as in P-4.2); instead, the source
        //:   object, 'Z', is a reference to the target object, 'mX', and both
        //:   'mX' and 'ZZ' are initialized to have the value 'V'.  For each
        //:   row (representing a distinct object value, 'V') in the table
        //:   described in P-3:  (C-9)
        //:
        //:   1 Create a 'bslma::TestAllocator' object, 'oa'.
        //:
        //:   2 Use the value constructor and 'oa' to create a modifiable 'Obj'
        //:     'mX'; also use the value constructor and a distinct "scratch"
        //:     allocator to create a 'const' 'Obj' 'ZZ'.
        //:
        //:   3 Let 'Z' be a reference providing only 'const' access to 'mX'.
        //:
        //:   4 Assign 'mX' from 'Z' in the presence of injected exceptions
        //:     (using the 'BSLMA_TESTALLOCATOR_EXCEPTION_TEST_*' macros).
        //:     (C-9)
        //:
        //:   5 Verify that the address of the return value is the same as that
        //:     of 'mX'.
        //:
        //:   6 Use the equality-comparison operator to verify that the
        //:     target object, 'mX', still has the same value as that of 'ZZ'.
        //:
        //:   7 Use the 'allocator' accessor of 'mX' to verify that it is still
        //:     the object allocator.
        //:
        //:   8 Use the appropriate test allocators to verify that:
        //:
        //:     1 Any memory that is allocated is from the object allocator.
        //:
        //:     2 No additional (e.g., temporary) object memory is allocated
        //:       when assigning an object value that did NOT initially require
        //:       allocated memory.
        //:
        //:     3 All object memory is released when the object is destroyed.
        //:
        //: 6 Use the test allocator from P-2 to verify that no memory is ever
        //:   allocated from the default allocator.  (C-3)
        //
        // Testing:
        //   AllocArgumentType& operator=(AllocArgumentType&&);
        // --------------------------------------------------------------------

        if (verbose) printf("\nMOVE-ASSIGNMENT OPERATOR"
                            "\n========================\n");

        if (verbose)
            printf("\nAssign the address of the operator to a variable.\n");
        {
            typedef Obj& (Obj::*operatorPtr)(bslmf::MovableRef<Obj>);

            // Verify that the signature and return type are standard.

            operatorPtr operatorAssignment = &Obj::operator=;

            (void) operatorAssignment;  // quash potential compiler warning
        }

        bslma::TestAllocator         da("default", veryVeryVeryVerbose);
        bslma::DefaultAllocatorGuard dag(&da);

        const int NUM_DATA                     = DEFAULT_NUM_DATA;
        const DefaultDataRow (&DATA)[NUM_DATA] = DEFAULT_DATA;

        for (int ti = 0; ti < NUM_DATA; ++ti) {
            const int LINE1  = DATA[ti].d_line;
            const int VALUE1 = DATA[ti].d_value;

            if (veryVerbose) { T_ P_(LINE1) P(VALUE1) }

            bslma::TestAllocator scratch("scratch", veryVeryVeryVerbose);

            const Obj ZZ(VALUE1, &scratch);

            for (int tj = 0; tj < NUM_DATA; ++tj) {
                const int LINE2  = DATA[tj].d_line;
                const int VALUE2 = DATA[tj].d_value;

                if (veryVerbose) { T_ T_ P_(LINE2) P(VALUE2) }

                bslma::TestAllocator oa("object", veryVeryVeryVerbose);

                {
                    Obj mX(VALUE2, &oa); const Obj& X = mX;

                    if (veryVerbose) { T_ P_(LINE2) P(X) }

                    bslma::TestAllocatorMonitor oam(&oa);

                    BSLMA_TESTALLOCATOR_EXCEPTION_TEST_BEGIN(oa) {
                        Obj mZ(VALUE1, &scratch); const Obj& Z = mZ;

                        ASSERTV(LINE1, LINE2, Z, X,
                                (Z == X) == (LINE1 == LINE2));
                        ASSERTV(LINE1, LINE2, &scratch, Z.allocator(),
                                &scratch == Z.allocator());

                        if (veryVeryVerbose) { T_ T_ Q(ExceptionTestBody) }

                        Obj *mR = &(mX = bslmf::MovableRefUtil::move(mZ));

                        ASSERTV(LINE1, LINE2, mR, &mX, mR == &mX);
                    } BSLMA_TESTALLOCATOR_EXCEPTION_TEST_END;

                    ASSERTV(LINE1, LINE2, int(ZZ), int(X), ZZ == X);

                    ASSERTV(LINE1, LINE2, &oa, X.allocator(),
                            &oa == X.allocator());

                    ASSERTV(LINE1, LINE2, oam.isInUseSame());

                    ASSERTV(LINE1, LINE2, 0 == da.numBlocksTotal());
                }

                // Verify all memory is released on object destruction.

                ASSERTV(LINE1, LINE2, oa.numBlocksInUse(),
                        0 == oa.numBlocksInUse());
            }

            // self-assignment

            bslma::TestAllocator oa("object", veryVeryVeryVerbose);

            {
                bslma::TestAllocator scratch("scratch", veryVeryVeryVerbose);

                Obj       mX(VALUE1, &oa);
                const Obj ZZ(VALUE1, &scratch);

                Obj& mZ = mX; const Obj& Z = mX;

                ASSERTV(LINE1, int(ZZ), int(Z), ZZ == Z);

                bslma::TestAllocatorMonitor oam(&oa), sam(&scratch);

                BSLMA_TESTALLOCATOR_EXCEPTION_TEST_BEGIN(oa) {
                    if (veryVeryVerbose) { T_ T_ Q(ExceptionTestBody) }

                    Obj *mR = &(mX = bslmf::MovableRefUtil::move(mZ));

                    ASSERTV(LINE1, mR, &mX, mR == &mX);
                    ASSERTV(LINE1, int(ZZ), int(Z), ZZ == Z);
                } BSLMA_TESTALLOCATOR_EXCEPTION_TEST_END

                ASSERTV(LINE1, &oa, Z.allocator(), &oa == Z.allocator());

                ASSERTV(LINE1, oam.isInUseSame());

                ASSERTV(LINE1, sam.isInUseSame());

                ASSERTV(LINE1, 0 == da.numBlocksTotal());
            }

            // Verify all object memory is released on destruction.

            ASSERTV(LINE1, oa.numBlocksInUse(), 0 == oa.numBlocksInUse());
        }
      } break;
      case 11: {
        // --------------------------------------------------------------------
        // TESTING MOVE CONSTRUCTORS
        //
        // Concerns:
        //: 1 The newly created object has the same value (using the equality
        //:   operator) as that of the original object before the call.
        //:
        //: 2 All internal representations of a given value can be used to
        //:   create a new object of equivalent value.
        //:
        //: 3 The allocator is propagated to the newly created object if (and
        //:   only if) no allocator is specified in the call to the move
        //:   constructor.
        //:
        //: 4 The original object is always left in a valid state; the
        //:   allocator address held by the original object is unchanged.
        //:
        //: 5 Subsequent changes to or destruction of the original object have
        //:   no effect on the move-constructed object and vice-versa.
        //:
        //: 6 The object has its internal memory management system hooked up
        //:   properly so that *all* internally allocated memory draws from a
        //:   user-supplied allocator whenever one is specified.
        //:
        //: 7 Every object releases any allocated memory at destruction.
        //
        //: 8 Any memory allocation is exception neutral.
        //:
        // Plan:
        //: 1 Specify a set 'S' of object values with substantial and varied
        //:   differences to be used sequentially in the following tests; for
        //:   each entry, create a control object.   (C-2)
        //:
        //: 2 Call the move constructor to create the object in all relevant
        //:   use cases involving the allocator: 1) no allocator passed in,
        //:   2) a '0' is explicitly passed in as the allocator argument,
        //:   3) the same allocator as that of the original object is
        //:   explicitly passed in, and 4) a different allocator than that of
        //:   the original object is passed in.
        //:
        //: 3 For each of the object values (P-1) and for each configuration
        //:   (P-2), verify the following:
        //:
        //:   1 Verify the newly created object has the same value as that of
        //:     the original object before the call to the move constructor
        //:     (control value).  (C-1)
        //:
        //:   2 Where a move is expected, ensure that no memory was allocated,
        //:     and that the original object is left in the default state.
        //:
        //:   3 Ensure that the new original, and control object continue to
        //:     have the correct allocator and that all memory allocations come
        //:     from the appropriate allocator.  (C-3..6)
        //:
        //:   4 Manipulate the original object (after the move construction) to
        //:     ensure it is in a valid state, destroy it, and then manipulate
        //:     the newly created object to ensure that it is in a valid state.
        //:     (C-5)
        //:
        //:   5 Verify all memory is released when the object is destroyed.
        //:     (C-7)
        //;
        //: 4 Perform tests in the presence of exceptions during memory
        //:   allocations using a 'bslma::TestAllocator' and varying its
        //:   *allocation* *limit*.  (C-8)
        //
        // Testing:
        //   AllocArgumentType(bslmf::MovableRef<AAT> original);
        //   AllocArgumentType(bslmf::MovableRef<AAT> original, Allocator *al);
        // --------------------------------------------------------------------
        if (verbose) printf("\nMOVE CONSTRUCTORS"
                            "\n=================\n");

        if (verbose) printf("\nTesting both versions of move constructor.\n");

        if (veryVerbose)
            printf("\n\tTesting move on default-constructed object.\n");
        {
            // Create control object ZZ with the scratch allocator.

            bslma::TestAllocator scratch("scratch", veryVeryVeryVerbose);
            Obj mZZ(&scratch); const Obj& ZZ = mZZ;
            ASSERTV(scratch.numBytesInUse(), 0 == scratch.numBytesInUse());

            for (char cfg = 'a'; cfg <= 'd'; ++cfg) {
                const char CONFIG = cfg;  // how we specify the allocator

                bslma::TestAllocator fa("footprint", veryVeryVeryVerbose);
                bslma::TestAllocator da("default",   veryVeryVeryVerbose);
                bslma::TestAllocator sa("supplied",  veryVeryVeryVerbose);
                bslma::TestAllocator za("different", veryVeryVeryVerbose);

                bslma::DefaultAllocatorGuard dag(&da);

                // Create source object 'Z'.
                Obj *srcPtr = new (fa) Obj(&sa);
                Obj& mZ = *srcPtr; const Obj& Z = mZ;

                ASSERTV(CONFIG, sa.numBytesInUse(), 0 == sa.numBytesInUse());

                Obj                  *objPtr = 0;
                bslma::TestAllocator *objAllocatorPtr;
                bslma::TestAllocator *othAllocatorPtr;

                switch (CONFIG) {
                  case 'a': {
                    objAllocatorPtr = &sa;
                    othAllocatorPtr = &da;
                  } break;
                  case 'b': {
                    objAllocatorPtr = &da;
                    othAllocatorPtr = &sa;
                  } break;
                  case 'c': {
                    objAllocatorPtr = &sa;
                    othAllocatorPtr = &da;
                  } break;
                  case 'd': {
                    objAllocatorPtr = &za;
                    othAllocatorPtr = &sa;
                  } break;
                  default: {
                    BSLS_ASSERT_INVOKE_NORETURN("Bad allocator config.");
                  }
                }

                bslma::TestAllocator&  oa = *objAllocatorPtr;
                bslma::TestAllocator& noa = *othAllocatorPtr;

                BSLMA_TESTALLOCATOR_EXCEPTION_TEST_BEGIN(oa) {
                    if (veryVeryVerbose) { T_ T_ Q(ExceptionTestBody) }

                    switch (CONFIG) {
                      case 'a': {
                        objPtr = new (fa) Obj(MoveUtil::move(mZ));
                      } break;
                      case 'b': {
                        objPtr = new (fa) Obj(MoveUtil::move(mZ), 0);
                      } break;
                      case 'c': {
                        objPtr = new (fa) Obj(MoveUtil::move(mZ), &sa);
                      } break;
                      case 'd': {
                        objPtr = new (fa) Obj(MoveUtil::move(mZ), &za);
                      } break;
                      default: {
                        BSLS_ASSERT_INVOKE_NORETURN("Bad allocator config.");
                      } break;
                    }
                    // Note that move-constructing from a default-constructed
                    // object does not allocate.
                    ASSERTV(CONFIG, 0 == oa.numBytesInUse());
                } BSLMA_TESTALLOCATOR_EXCEPTION_TEST_END;

                ASSERTV(CONFIG, 2 * sizeof(Obj) == fa.numBytesInUse());

                Obj& mX = *objPtr; const Obj& X = mX;

                // Verify the value of the object.
                ASSERTV(CONFIG, X, ZZ, X == ZZ);

                // original object is movedFrom
                ASSERTV(CONFIG, &sa == &oa,
                        MoveState::e_MOVED == Z.movedFrom());

                // new object is movedInto
                ASSERTV(CONFIG, &sa == &oa,
                        MoveState::e_MOVED == X.movedInto());

                // Verify that 'X', 'Z', and 'ZZ' have the correct allocator.
                ASSERTV(CONFIG, &scratch == ZZ.allocator());
                ASSERTV(CONFIG,      &sa ==  Z.allocator());
                ASSERTV(CONFIG,      &oa ==  X.allocator());

                // Verify no allocation from the non-object allocator.
                ASSERTV(CONFIG, 0 == noa.numBlocksTotal());

                fa.deleteObject(srcPtr);

                ASSERTV(CONFIG, X, ZZ, X == ZZ);

                fa.deleteObject(objPtr);

                // Verify all memory is released on object destruction.
                ASSERTV(0 == fa.numBlocksInUse());
                ASSERTV(0 == da.numBlocksInUse());
                ASSERTV(0 == sa.numBlocksInUse());
                ASSERTV(0 == za.numBlocksInUse());
            }
        }

        if (veryVerbose)
            printf("\n\tTesting move on value-constructed objects.\n");

        const size_t NUM_DATA                  = DEFAULT_NUM_DATA;
        const DefaultDataRow (&DATA)[NUM_DATA] = DEFAULT_DATA;

        {
            for (size_t ti = 0; ti < NUM_DATA; ++ti) {
                const int LINE   = DATA[ti].d_line;
                const int VALUE  = DATA[ti].d_value;

                if (veryVerbose) { T_ P_(LINE) P(VALUE) }

                // Create control object ZZ with the scratch allocator.

                bslma::TestAllocator scratch("scratch", veryVeryVeryVerbose);
                Obj mZZ(VALUE, &scratch); const Obj& ZZ = mZZ;

                for (char cfg = 'a'; cfg <= 'd'; ++cfg) {
                    const char CONFIG = cfg;  // how we specify the allocator

                    bslma::TestAllocator fa("footprint", veryVeryVeryVerbose);
                    bslma::TestAllocator da("default",   veryVeryVeryVerbose);
                    bslma::TestAllocator sa("supplied",  veryVeryVeryVerbose);
                    bslma::TestAllocator za("different", veryVeryVeryVerbose);

                    bslma::DefaultAllocatorGuard dag(&da);

                    // Create source object 'Z'.
                    Obj *srcPtr = new (fa) Obj(VALUE, &sa);
                    Obj& mZ = *srcPtr; const Obj& Z = mZ;

                    Obj                  *objPtr = 0;
                    bslma::TestAllocator *objAllocatorPtr;
                    bslma::TestAllocator *othAllocatorPtr;

                    switch (CONFIG) {
                      case 'a': {
                        objAllocatorPtr = &sa;
                        othAllocatorPtr = &da;
                      } break;
                      case 'b': {
                        objAllocatorPtr = &da;
                        othAllocatorPtr = &za;
                      } break;
                      case 'c': {
                        objAllocatorPtr = &sa;
                        othAllocatorPtr = &da;
                      } break;
                      case 'd': {
                        objAllocatorPtr = &za;
                        othAllocatorPtr = &da;
                      } break;
                      default: {
                        BSLS_ASSERT_INVOKE_NORETURN("Bad allocator config.");
                      }
                    }

                    bslma::TestAllocator&  oa = *objAllocatorPtr;
                    bslma::TestAllocator& noa = *othAllocatorPtr;

                    BSLMA_TESTALLOCATOR_EXCEPTION_TEST_BEGIN(oa) {
                        if (veryVeryVerbose) { T_ T_ Q(ExceptionTestBody) }

                        bslma::TestAllocatorMonitor tam(&oa);

                        switch (CONFIG) {
                          case 'a': {
                            objPtr = new (fa) Obj(MoveUtil::move(mZ));
                          } break;
                          case 'b': {
                            objPtr = new (fa) Obj(MoveUtil::move(mZ), 0);
                          } break;
                          case 'c': {
                            objPtr = new (fa) Obj(MoveUtil::move(mZ), &sa);
                          } break;
                          case 'd': {
                            objPtr = new (fa) Obj(MoveUtil::move(mZ), &za);
                          } break;
                          default: {
                            BSLS_ASSERT_INVOKE_NORETURN(
                                                      "Bad allocator config.");
                          } break;
                        }
                        ASSERTV(CONFIG, (&sa != &oa) == tam.isInUseUp());
                    } BSLMA_TESTALLOCATOR_EXCEPTION_TEST_END;

                    ASSERTV(CONFIG, 2 * sizeof(Obj) == fa.numBytesInUse());

                    Obj& mX = *objPtr; const Obj& X = mX;

                    // Verify the value of the object.
                    ASSERTV(VALUE, CONFIG, X, ZZ, X == ZZ);

                    // original object is movedFrom
                    ASSERTV(VALUE, CONFIG, &sa == &oa,
                            MoveState::e_MOVED == Z.movedFrom());

                    // new object is movedInto
                    ASSERTV(VALUE, CONFIG, &sa == &oa,
                            MoveState::e_MOVED == X.movedInto());

                    // Verify that 'X', 'Z', and 'ZZ' have the correct
                    // allocator.
                    ASSERTV(VALUE, CONFIG, &scratch == ZZ.allocator());
                    ASSERTV(VALUE, CONFIG,      &sa ==  Z.allocator());
                    ASSERTV(VALUE, CONFIG,      &oa ==  X.allocator());

                    // Verify no allocation from the non-object allocator and
                    // that object allocator is hooked up.
                    ASSERTV(VALUE, CONFIG, 0 == noa.numBlocksTotal());
                    ASSERTV(VALUE, CONFIG, 0 < oa.numBlocksTotal());

                    fa.deleteObject(srcPtr);

                    ASSERTV(VALUE, CONFIG, X, ZZ, X == ZZ);

                    fa.deleteObject(objPtr);

                    // Verify all memory is released on object destruction.
                    ASSERTV(VALUE, 0 == fa.numBlocksInUse());
                    ASSERTV(VALUE, 0 == da.numBlocksInUse());
                    ASSERTV(VALUE, 0 == sa.numBlocksInUse());
                    ASSERTV(VALUE, 0 == za.numBlocksInUse());
                }
            }
        }
      } break;
      case 10: {
        // --------------------------------------------------------------------
        // TESTING TYPE TRAITS
        //   Ensure that 'AllocArgumentType' has the necessary trait values
        //   to guarantee its expected behavior.
        //
        // Concerns:
        //: 1 The object has the 'bslma::UsesBslmaAllocator' trait.
        //:
        //: 2 The object does not have the 'bslmf::IsBitwiseMoveable' trait.
        //
        // Plan:
        //: 1 Use 'BSLMF_ASSERT' to verify all the type traits exists.  (C-1,2)
        //
        // Testing:
        //   CONCERN: The object has the necessary type traits
        // --------------------------------------------------------------------

        if (verbose) printf("\nTESTING TYPE TRAITS"
                            "\n===================\n");

        BSLMF_ASSERT( bslma::UsesBslmaAllocator<Obj>::value);
        BSLMF_ASSERT(!bslmf::IsBitwiseMoveable<Obj>::value);
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
        //: 2 The allocator address held by the target object is unchanged.
        //:
        //: 3 Any memory allocation is from the target object's allocator.
        //:
        //: 4 The signature and return type are standard.
        //:
        //: 5 The reference returned is to the target object (i.e., '*this').
        //:
        //: 6 The value of the source object is not modified.
        //:
        //: 7 The allocator address held by the source object is unchanged.
        //:
        //: 8 QoI: Assigning a source object having the default-constructed
        //:   value allocates no memory.
        //:
        //: 9 Any memory allocation is exception neutral.
        //:
        //:10 Assigning an object to itself behaves as expected (alias-safety).
        //:
        //:11 Every object releases any allocated memory at destruction.
        //
        // Plan:
        //: 1 Use the address of 'operator=' to initialize a member-function
        //:   pointer having the appropriate signature and return type for the
        //:   copy-assignment operator defined in this component.  (C-4)
        //:
        //: 2 Create a 'bslma::TestAllocator' object, and install it as the
        //:   default allocator (note that a ubiquitous test allocator is
        //:   already installed as the global allocator).
        //:
        //: 3 Using the table-driven technique, specify a set of distinct
        //:   object values (one per row) in terms of their attributes.
        //:
        //: 4 For each row 'R1' (representing a distinct object value, 'V') in
        //:   the table described in P-3:  (C-1..2, 5..8, 11)
        //:
        //:   1 Use the value constructor and a "scratch" allocator to create
        //:     two 'const' 'Obj', 'Z' and 'ZZ', each having the value 'V'.
        //:
        //:   2 Execute an inner loop that iterates over each row 'R2'
        //:     (representing a distinct object value, 'W') in the table
        //:     described in P-3:
        //:
        //:   3 For each of the iterations (P-4.2):  (C-1..2, 5..8, 11)
        //:
        //:     1 Create a 'bslma::TestAllocator' object, 'oa'.
        //:
        //:     2 Use the value constructor and 'oa' to create a modifiable
        //:       'Obj', 'mX', having the value 'W'.
        //:
        //:     3 Assign 'mX' from 'Z' in the presence of injected exceptions
        //:       (using the 'BSLMA_TESTALLOCATOR_EXCEPTION_TEST_*' macros).
        //:
        //:     4 Verify that the address of the return value is the same as
        //:       that of 'mX'.  (C-5)
        //:
        //:     5 Use the equality-comparison operator to verify that: (C-1, 6)
        //:
        //:       1 The target object, 'mX', now has the same value as that of
        //:         'Z'.  (C-1)
        //:
        //:       2 'Z' still has the same value as that of 'ZZ'.  (C-6)
        //:
        //:     6 Use the 'allocator' accessor of both 'mX' and 'Z' to verify
        //:       that the respective allocator addresses held by the target
        //:       and source objects are unchanged.  (C-2, 7)
        //:
        //:     7 Use the appropriate test allocators to verify that:
        //:       (C-8, 11)
        //:
        //:       1 For an object that (a) is initialized with a value that did
        //:         NOT require memory allocation, and (b) is then assigned a
        //:         value that DID require memory allocation, the target object
        //:         DOES allocate memory from its object allocator only
        //:         (irrespective of the specific number of allocations or the
        //:         total amount of memory allocated); also cross check with
        //:         what is expected for 'mX' and 'Z'.
        //:
        //:       2 An object that is assigned a value that did NOT require
        //:         memory allocation, does NOT allocate memory from its object
        //:         allocator; also cross check with what is expected for 'Z'.
        //:
        //:       3 No additional memory is allocated by the source object.
        //:         (C-8)
        //:
        //:       4 All object memory is released when the object is destroyed.
        //:         (C-11)
        //:
        //: 5 Repeat steps similar to those described in P-2 except that, this
        //:   time, there is no inner loop (as in P-4.2); instead, the source
        //:   object, 'Z', is a reference to the target object, 'mX', and both
        //:   'mX' and 'ZZ' are initialized to have the value 'V'.  For each
        //:   row (representing a distinct object value, 'V') in the table
        //:   described in P-3:  (C-9)
        //:
        //:   1 Create a 'bslma::TestAllocator' object, 'oa'.
        //:
        //:   2 Use the value constructor and 'oa' to create a modifiable 'Obj'
        //:     'mX'; also use the value constructor and a distinct "scratch"
        //:     allocator to create a 'const' 'Obj' 'ZZ'.
        //:
        //:   3 Let 'Z' be a reference providing only 'const' access to 'mX'.
        //:
        //:   4 Assign 'mX' from 'Z' in the presence of injected exceptions
        //:     (using the 'BSLMA_TESTALLOCATOR_EXCEPTION_TEST_*' macros).
        //:     (C-9)
        //:
        //:   5 Verify that the address of the return value is the same as that
        //:     of 'mX'.
        //:
        //:   6 Use the equality-comparison operator to verify that the
        //:     target object, 'mX', still has the same value as that of 'ZZ'.
        //:
        //:   7 Use the 'allocator' accessor of 'mX' to verify that it is still
        //:     the object allocator.
        //:
        //:   8 Use the appropriate test allocators to verify that:
        //:
        //:     1 Any memory that is allocated is from the object allocator.
        //:
        //:     2 No additional (e.g., temporary) object memory is allocated
        //:       when assigning an object value that did NOT initially require
        //:       allocated memory.
        //:
        //:     3 All object memory is released when the object is destroyed.
        //:
        //: 6 Use the test allocator from P-2 to verify that no memory is ever
        //:   allocated from the default allocator.  (C-3)
        //
        // Testing:
        //   AllocArgumentType& operator=(const AllocArgumentType& rhs);
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

        if (verbose) printf("\nTesting operator.\n");

        bslma::TestAllocator         da("default", veryVeryVeryVerbose);
        bslma::DefaultAllocatorGuard dag(&da);

        const size_t NUM_DATA                  = DEFAULT_NUM_DATA;
        const DefaultDataRow (&DATA)[NUM_DATA] = DEFAULT_DATA;

        for (size_t ti = 0; ti < NUM_DATA; ++ti) {
            const int LINE1  = DATA[ti].d_line;
            const int VALUE1 = DATA[ti].d_value;

            if (veryVerbose) { T_ P_(LINE1) P(VALUE1) }

            bslma::TestAllocator scratch("scratch", veryVeryVeryVerbose);

            const Obj Z(VALUE1,  &scratch);
            const Obj ZZ(VALUE1, &scratch);

            for (size_t tj = 0; tj < NUM_DATA; ++tj) {
                const int LINE2  = DATA[tj].d_line;
                const int VALUE2 = DATA[tj].d_value;

                if (veryVerbose) { T_ T_ P_(LINE2) P(VALUE2) }

                bslma::TestAllocator oa("object", veryVeryVeryVerbose);
                {
                    Obj mX(VALUE2, &oa); const Obj& X = mX;

                    if (veryVerbose) { T_ P_(LINE2) P(int(X)) }

                    ASSERTV(LINE1, LINE2, int(Z), int(X),
                            (Z == X) == (LINE1 == LINE2));

                    bslma::TestAllocatorMonitor oam(&oa), sam(&scratch);

                    BSLMA_TESTALLOCATOR_EXCEPTION_TEST_BEGIN(oa) {
                        if (veryVeryVerbose) { T_ T_ Q(ExceptionTestBody) }

                        Obj *mR = &(mX = Z);
                        ASSERTV(LINE1, LINE2, int(Z), int(X), Z == X);
                        ASSERTV(LINE1, LINE2, mR, &mX, mR == &mX);
                    } BSLMA_TESTALLOCATOR_EXCEPTION_TEST_END;

                    ASSERTV(LINE1, LINE2, int(ZZ), int(Z), ZZ == Z);

                    ASSERTV(LINE1, LINE2, &oa, X.allocator(),
                            &oa == X.allocator());
                    ASSERTV(LINE1, LINE2, &scratch, Z.allocator(),
                            &scratch == Z.allocator());

                    ASSERTV(LINE1, LINE2, oam.isInUseSame());

                    ASSERTV(LINE1, LINE2, sam.isInUseSame());

                    ASSERTV(LINE1, LINE2, 0 == da.numBlocksTotal());
                }

                // Verify all memory is released on object destruction.
                ASSERTV(LINE1, LINE2, oa.numBlocksInUse(),
                        0 == oa.numBlocksInUse());
            }

            // self-assignment
            bslma::TestAllocator oa("object", veryVeryVeryVerbose);

            {
                bslma::TestAllocator scratch("scratch", veryVeryVeryVerbose);

                      Obj mX(VALUE1, &oa);
                const Obj ZZ(VALUE1, &scratch);

                const Obj& Z = mX;

                ASSERTV(LINE1, int(ZZ), int(Z), ZZ == Z);

                bslma::TestAllocatorMonitor oam(&oa), sam(&scratch);

                BSLMA_TESTALLOCATOR_EXCEPTION_TEST_BEGIN(oa) {
                    if (veryVeryVerbose) { T_ T_ Q(ExceptionTestBody) }

                    Obj *mR = &(mX = Z);
                    ASSERTV(LINE1, int(ZZ), int(Z), ZZ == Z);
                    ASSERTV(LINE1, mR, &mX, mR == &mX);
                } BSLMA_TESTALLOCATOR_EXCEPTION_TEST_END

                ASSERTV(LINE1, &oa, Z.allocator(), &oa == Z.allocator());

                ASSERTV(LINE1, oam.isInUseSame());

                ASSERTV(LINE1, sam.isInUseSame());

                ASSERTV(LINE1, 0 == da.numBlocksTotal());
            }

            // Verify all object memory is released on destruction.
            ASSERTV(LINE1, oa.numBlocksInUse(), 0 == oa.numBlocksInUse());
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
        // COPY CONSTRUCTORS
        //   Ensure that we can create a distinct object of the class from any
        //   other one, such that the two objects have the same value.
        //
        // Concerns:
        //: 1 The copy constructor creates an object having the same value as
        //:   that of the supplied original object.
        //:
        //: 2 If an allocator is NOT supplied to the copy constructor, the
        //:   default allocator in effect at the time of construction becomes
        //:   the object allocator for the resulting object (i.e., the
        //:   allocator of the original object is never copied).
        //:
        //: 3 If an allocator IS supplied to the copy constructor, that
        //:   allocator becomes the object allocator for the resulting object.
        //:
        //: 4 Supplying a null allocator address has the same effect as not
        //:   supplying an allocator.
        //:
        //: 5 Supplying an allocator to the copy constructor has no effect
        //:   on subsequent object values.
        //:
        //: 6 Any memory allocation is from the object allocator.
        //:
        //: 7 The copy constructor is exception-neutral w.r.t. memory
        //:   allocation.
        //:
        //: 8 The original object is passed as a reference providing
        //:   non-modifiable access to that object.
        //:
        //: 9 The value of the original object is unchanged.
        //:
        //:10 The allocator address held by the original object is unchanged.
        //
        // Plan:
        //: 1 Using the table-driven technique, specify a set of distinct
        //:   object values (one per row) in terms of their attributes.
        //:
        //: 2 For each row (representing a distinct object value, 'V') in the
        //:   table described in P-1:  (C-1..10)
        //:
        //:   1 Use the value constructor and a "scratch" allocator to create
        //:     two 'const' 'Obj', 'Z' and 'ZZ', each having the value 'V'.
        //:
        //:   2 Execute an inner loop creating three distinct objects in turn,
        //:     each using the copy constructor in the presence of injected
        //:     exceptions (using the 'BSLMA_TESTALLOCATOR_EXCEPTION_TEST_*'
        //:     macros) on 'Z' from P-2.1, but configured differently: (a)
        //:     without passing an allocator, (b) passing a null allocator
        //:     address explicitly, and (c) passing the address of a test
        //:     allocator distinct from the default.  (C-7)
        //:
        //:   3 For each of these three iterations (P-2.2):  (C-1..10)
        //:
        //:     1 Create three 'bslma::TestAllocator' objects, and install one
        //:       as the current default allocator (note that a ubiquitous test
        //:       allocator is already installed as the global allocator).
        //:
        //:     2 Use the copy constructor to dynamically create an object 'X',
        //:       with its object allocator configured appropriately (see
        //:       P-2.2), supplying it the 'const' object 'Z' (see P-2.1); use
        //:       a distinct test allocator for the object's footprint.  (C-8)
        //:
        //:     3 Use the equality-comparison operator to verify that:
        //:       (C-1, 5, 9)
        //:
        //:       1 The newly constructed object, 'X', has the same value as
        //:         that of 'Z'.  (C-1, 5)
        //:
        //:       2 'Z' still has the same value as that of 'ZZ'.  (C-9
        //:
        //:     4 Use the 'allocator' accessor of each underlying attribute
        //:       capable of allocating memory to ensure that its object
        //:       allocator is properly installed; also use the 'allocator'
        //:       accessor of 'X' to verify that its object allocator is
        //:       properly installed, and use the 'allocator' accessor of
        //:       'Z' to verify that the allocator address that it holds is
        //:       unchanged.  (C-6, 10)
        //:
        //:     5 Use the appropriate test allocators to verify that:  (C-2..4,
        //:       7..8)
        //:
        //:       1 An object allocates memory from the object allocator only.
        //:         (C-2..4)
        //:
        //:       3 If an allocator was supplied at construction (P-2.1c), the
        //:         current default allocator doesn't allocate any memory.
        //:         (C-3)
        //:
        //:       4 No temporary memory is allocated from the object allocator.
        //:         (C-7)
        //:
        //:       5 All object memory is released when the object is destroyed.
        //:         (C-8)
        //
        // Testing:
        //   AllocArgumentType(const AAT& original, bslma::Allocator *al = 0);
        // --------------------------------------------------------------------

        if (verbose) printf("\nCOPY CONSTRUCTORS"
                            "\n=================\n");

        const size_t NUM_DATA                  = DEFAULT_NUM_DATA;
        const DefaultDataRow (&DATA)[NUM_DATA] = DEFAULT_DATA;

        for (size_t ti = 0; ti < NUM_DATA; ++ti) {
            const int LINE  = DATA[ti].d_line;
            const int VALUE = DATA[ti].d_value;

            if (veryVerbose) { T_ P_(LINE) P(VALUE) }

            bslma::TestAllocator scratch("scratch", veryVeryVeryVerbose);

            const Obj Z(VALUE,  &scratch);
            const Obj ZZ(VALUE, &scratch);

            for (char cfg = 'a'; cfg <= 'c'; ++cfg) {

                const char CONFIG = cfg;  // how we specify the allocator

                bslma::TestAllocator da("default",   veryVeryVeryVerbose);
                bslma::TestAllocator fa("footprint", veryVeryVeryVerbose);
                bslma::TestAllocator sa("supplied",  veryVeryVeryVerbose);

                bslma::DefaultAllocatorGuard dag(&da);

                Obj                  *objPtr = 0;
                bslma::TestAllocator *objAllocatorPtr = 0;

                switch (CONFIG) {
                  case 'a': {
                    objAllocatorPtr = &da;
                  } break;
                  case 'b': {
                    objAllocatorPtr = &da;
                  } break;
                  case 'c': {
                    objAllocatorPtr = &sa;
                  } break;
                  default: {
                    BSLS_ASSERT_INVOKE_NORETURN("Bad allocator config.");
                  } break;
                }

                bslma::TestAllocator&  oa = *objAllocatorPtr;
                bslma::TestAllocator& noa = 'c' != CONFIG ? sa : da;

                BSLMA_TESTALLOCATOR_EXCEPTION_TEST_BEGIN(oa) {
                    if (veryVeryVerbose) { T_ T_ Q(ExceptionTestBody) }

                    bslma::TestAllocatorMonitor tam(&oa);
                    switch (CONFIG) {
                      case 'a': {
                        objPtr = new (fa) Obj(Z);
                      } break;
                      case 'b': {
                        objPtr = new (fa) Obj(Z, 0);
                      } break;
                      case 'c': {
                        objPtr = new (fa) Obj(Z, &sa);
                      } break;
                      default: {
                        BSLS_ASSERT_INVOKE_NORETURN("Bad allocator config.");
                      } break;
                    }
                    ASSERTV(CONFIG, tam.isInUseUp());
                } BSLMA_TESTALLOCATOR_EXCEPTION_TEST_END;

                ASSERTV(LINE, CONFIG, sizeof(Obj) == fa.numBytesInUse());

                Obj& mX = *objPtr;  const Obj& X = mX;

                ASSERTV(int(Z), int(X),  Z == X);
                ASSERTV(int(Z), int(ZZ), Z == ZZ);

                ASSERTV(LINE, CONFIG, &oa, X.allocator(),
                             &oa == X.allocator());

                ASSERTV(LINE, CONFIG, &scratch, Z.allocator(),
                             &scratch == Z.allocator());

                // Verify no allocation from the non-object allocator.
                ASSERTV(LINE, CONFIG, noa.numBlocksTotal(),
                             0 == noa.numBlocksTotal());

                // Verify no temporary memory is allocated from the object
                // allocator.
                ASSERTV(LINE, CONFIG, oa.numBlocksTotal(),
                             oa.numBlocksInUse(),
                             oa.numBlocksTotal() == oa.numBlocksInUse());

                // Verify expected object-memory allocations.
                ASSERTV(LINE, CONFIG, 1, oa.numBlocksInUse(),
                             oa.numBlocksInUse() == 1);


                // Reclaim dynamically allocated object under test.
                fa.deleteObject(objPtr);

                // Verify all memory is released on object destruction.
                ASSERTV(LINE, CONFIG, da.numBlocksInUse(),
                        0 == da.numBlocksInUse());
                ASSERTV(LINE, CONFIG, fa.numBlocksInUse(),
                        0 == fa.numBlocksInUse());
                ASSERTV(LINE, CONFIG, sa.numBlocksInUse(),
                        0 == sa.numBlocksInUse());
            }
        }

      } break;
      case 6: {
        // --------------------------------------------------------------------
        // EQUALITY-COMPARISON OPERATORS
        //   Objects of the test type are compared by their values [implicitly]
        //   converted to 'int'.
        // --------------------------------------------------------------------
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
        //:   of the object.
        //:
        //: 2 Each accessor method is declared 'const'.
        //:
        //: 3 No accessor allocates any memory.
        //
        // Plan:
        //: 1 Use the default constructor, create an object having default
        //:   attribute values.  Verify that the accessor 'operator int'
        //:   invoked on a reference providing non-modifiable access to the
        //:   object return the expected value.  (C-1)
        //:
        //: 2 Verify that no memory is ever allocated after construction.
        //:   (C-3)
        //
        // Testing:
        //   operator int() const;
        //   bslma::Allocator *allocator() const;
        // --------------------------------------------------------------------

        if (verbose)
            printf("\nBASIC ACCESSORS"
                   "\n===============\n");

        bslma::TestAllocator         da("default", veryVeryVeryVerbose);
        bslma::DefaultAllocatorGuard dag(&da);

        bslma::TestAllocatorMonitor tam(&da);
        Obj mX(0); const Obj& X = mX;

        ASSERT(tam.isInUseUp());

        tam.reset(&da);
        ASSERTV(int(X), 0 == int(X));
        ASSERTV(tam.isTotalSame());

        tam.reset(&da);
        ASSERTV(&da == X.allocator());
        ASSERTV(tam.isTotalSame());
      } break;
      case 3: {
        // --------------------------------------------------------------------
        // VALUE CONSTRUCTOR
        //   Ensure that we can put an object into any initial state relevant
        //   for thorough testing.
        //
        // Concerns:
        //: 1 The value constructor can create an object having any value that
        //:   does not violate the documented constraints.
        //:
        //: 2 If an allocator is NOT supplied to the value constructor, the
        //:   default allocator in effect at the time of construction becomes
        //:   the object allocator for the resulting object.
        //:
        //: 3 If an allocator IS supplied to the value constructor, that
        //:   allocator becomes the object allocator for the resulting object.
        //:
        //: 4 Supplying a null allocator address has the same effect as not
        //:   supplying an allocator.
        //:
        //: 5 Supplying an allocator to the value constructor has no effect
        //:   on subsequent object values.
        //:
        //: 6 Any memory allocation is from the object allocator.
        //:
        //: 7 There is no temporary memory allocation from any allocator.
        //:
        //: 8 Every object releases any allocated memory at destruction.
        //:
        //: 9 Any memory allocation is exception neutral.
        //
        // Plan:
        //: 1 Using the table-driven technique, specify a set of distinct
        //:   object values (one per row) in terms of their attributes.
        //:
        //: 2 For each row (representing a distinct object value, 'V') in the
        //:   table of P-1:
        //:
        //:   1 Execute an inner loop creating three distinct objects, in turn,
        //:     each object having the same value of 'R1', but configured
        //:     differently: (a) without passing an allocator, (b) passing a
        //:     null allocator address explicitly, and (c) passing the address
        //:     of a test allocator distinct from the default allocator.
        //:
        //:     1 Create three 'bslma::TestAllocator' objects, and install one
        //:       as the current default allocator (note that a ubiquitous test
        //:       allocator is already installed as the global allocator).
        //:
        //:     2 Use the value constructor to dynamically create an object
        //:       having the value 'V' in the presence of injected exceptions
        //:       (using the 'BSLMA_TESTALLOCATOR_EXCEPTION_TEST_*' macros)
        //:       with its object allocator configured appropriately (see
        //:       P-2.1); use a distinct test allocator for the object's
        //:       footprint.  (C-9)
        //:
        //:     3 Use the (as yet unproven) salient attribute accessors to
        //:       verify that all of the attributes of each object have their
        //:       expected values.  (C-1, 5)
        //:
        //:     4 Use the 'allocator' accessor of each underlying attribute
        //:       capable of allocating memory to ensure that its object
        //:       allocator is properly installed; also invoke the (as yet
        //:       unproven) 'allocator' accessor of the object under test.
        //:       (C-6)
        //:
        //:     5 Use the appropriate test allocators to verify that:
        //:       (C-2..4, 7..8)
        //:
        //:       1 An object that IS expected to allocate memory does so
        //:         from the object allocator only (irrespective of the
        //:         specific number of allocations or the total amount of
        //:         memory allocated).  (C-2..4)
        //:
        //:       4 No temporary memory is allocated from the object allocator.
        //:         (C-7)
        //:
        //:       5 All object memory is released when the object is destroyed.
        //:         (C-8)
        //
        // Testing:
        //   AllocArgumentTypeType(int data, bslma::Allocator *alloc = 0);
        // --------------------------------------------------------------------

        if (verbose) printf("\nVALUE CONSTRUCTOR"
                            "\n=================\n");

        const size_t NUM_DATA                  = DEFAULT_NUM_DATA;
        const DefaultDataRow (&DATA)[NUM_DATA] = DEFAULT_DATA;

        for (size_t ti = 0; ti < NUM_DATA; ++ti) {
            const int LINE  = DATA[ti].d_line;
            const int VALUE = DATA[ti].d_value;


            for (char cfg = 'a'; cfg <= 'c'; ++cfg) {
                const char CONFIG = cfg;  // how we specify the allocator

                if (veryVeryVerbose) { T_ P_(LINE) P_(VALUE) P(CONFIG) }

                bslma::TestAllocator da("default",   veryVeryVeryVerbose);
                bslma::TestAllocator fa("footprint", veryVeryVeryVerbose);
                bslma::TestAllocator sa("supplied",  veryVeryVeryVerbose);

                bslma::DefaultAllocatorGuard dag(&da);

                Obj                  *objPtr = 0;
                bslma::TestAllocator *objAllocatorPtr = 0;

                switch (CONFIG) {
                  case 'a': {
                    objAllocatorPtr = &da;
                  } break;
                  case 'b': {
                    objAllocatorPtr = &da;
                  } break;
                  case 'c': {
                    objAllocatorPtr = &sa;
                  } break;
                  default: {
                    BSLS_ASSERT_INVOKE_NORETURN("Bad allocator config.");
                  } return testStatus;                                // RETURN
                }

                bslma::TestAllocator&  oa = *objAllocatorPtr;
                bslma::TestAllocator& noa = 'c' != CONFIG ? sa : da;

                BSLMA_TESTALLOCATOR_EXCEPTION_TEST_BEGIN(oa) {
                    if (veryVeryVerbose) { T_ T_ Q(ExceptionTestBody) }

                    bslma::TestAllocatorMonitor tam(&oa);
                    switch (CONFIG) {
                      case 'a': {
                        objPtr = new (fa) Obj(VALUE);
                      } break;
                      case 'b': {
                        objPtr = new (fa) Obj(VALUE, 0);
                      } break;
                      case 'c': {
                        objPtr = new (fa) Obj(VALUE, &sa);
                      } break;
                      default: {
                        BSLS_ASSERT_INVOKE_NORETURN("Bad allocator config.");
                      } return testStatus;                            // RETURN
                    }
                    ASSERTV(CONFIG, tam.isInUseUp());
                } BSLMA_TESTALLOCATOR_EXCEPTION_TEST_END;

                ASSERTV(LINE, CONFIG, sizeof(Obj) == fa.numBytesInUse());

                Obj& mX = *objPtr;  const Obj& X = mX;

                // Verify the object's attribute values.
                ASSERTV(CONFIG, VALUE, int(X), VALUE == int(X));

                // Verify the object's allocator is installed properly.
                ASSERTV(CONFIG, &oa, X.allocator(),&oa == X.allocator());

                // Verify no allocation from the non-object allocators.
                ASSERTV(CONFIG, noa.numBlocksTotal(),
                        0 == noa.numBlocksTotal());

                // Verify no temporary memory is allocated from the object
                // allocator.
                ASSERTV(CONFIG, oa.numBlocksTotal(), oa.numBlocksInUse(),
                        oa.numBlocksTotal() == oa.numBlocksInUse());

                // Reclaim dynamically allocated object under test.
                fa.deleteObject(objPtr);

                // Verify all memory is released on object destruction.
                ASSERTV(LINE, CONFIG, da.numBlocksInUse(),
                        0 == da.numBlocksInUse());
                ASSERTV(LINE, CONFIG, fa.numBlocksInUse(),
                        0 == fa.numBlocksInUse());
                ASSERTV(LINE, CONFIG, sa.numBlocksInUse(),
                        0 == sa.numBlocksInUse());
            }
        }

        if (verbose) printf("\tNegative Testing.\n");
        {
            bsls::AssertTestHandlerGuard hG;

            ASSERT_SAFE_PASS(Obj(0));
            ASSERT_SAFE_PASS(Obj(1));
            ASSERT_SAFE_PASS(Obj(INT_MAX));
            ASSERT_SAFE_FAIL(Obj(-1));
        }

      } break;
      case 2: {
        // --------------------------------------------------------------------
        // DEFAULT CTOR
        //   Ensure that we can use the default constructor to create an object
        //   (having the default constructed value).
        //
        // Concerns:
        //: 1 An object created with the default constructor has the
        //:   contractually specified default value.
        //:
        //: 2 If an allocator is NOT supplied to the default constructor, the
        //:   default allocator in effect at the time of construction becomes
        //:   the object allocator for the resulting object.
        //:
        //: 3 If an allocator IS supplied to the default constructor, that
        //:   allocator becomes the object allocator for the resulting object.
        //:
        //: 4 Supplying a null allocator address has the same effect as not
        //:   supplying an allocator.
        //:
        //: 5 Supplying an allocator to the default constructor has no effect
        //:   on subsequent object values.
        //:
        //: 6 There is no memory allocation from the object allocator.
        //:
        //: 7 There is no temporary allocation from any allocator.
        //:
        //: 8 Every object releases any allocated memory at destruction.
        //:
        //: 9 The default constructor is exception-neutral w.r.t. memory
        //:   allocation.
        //
        // Plan:
        //: 1 Using a loop-based approach, default-construct three distinct
        //:   objects, in turn, but configured differently: (a) without passing
        //:   an allocator, (b) passing a null allocator address explicitly,
        //:   and (c) passing the address of a test allocator distinct from the
        //:   default.  For each of these three iterations:
        //:
        //:   1 Create three 'bslma::TestAllocator' objects, and install one as
        //:     as the current default allocator (note that a ubiquitous test
        //:     allocator is already installed as the global allocator).
        //:
        //:   2 Use the default constructor to dynamically create an object 'X'
        //:     in the presence of exception (using the
        //:     'BSLMA_TESTALLOCATOR_EXCEPTION_TEST_*' macros), with its object
        //:     allocator configured appropriately (see P-2); use a distinct
        //:     test allocator for the object's footprint.  (C-9)
        //:
        //:   3 Use the 'allocator' accessor of each underlying attribute
        //:     capable of allocating memory to ensure that its object
        //:     allocator is properly installed; also invoke the (as yet
        //:     unproven) 'getAallocator' accessor of the object under test.
        //:     (C-2..4)
        //:
        //:   4 Use the appropriate test allocators to verify that no memory
        //:     is allocated by the default constructor.  (C-9)
        //:
        //:   5 Use the individual (as yet unproven) salient attribute
        //:     accessors to verify the default-constructed value.  (C-1)
        //:
        //:   7 Verify that no memory is allocated from the object allocator.
        //:     (C-6)
        //:
        //:   8 Verify that all object memory is released when the object is
        //:     destroyed.  (C-8)
        //
        // Testing:
        //   AllocArgumentType(bslma::Allocator *bA = 0);
        //   ~AllocArgumentType();
        //   CONCERN: Default constructor does not allocate.
        // --------------------------------------------------------------------

        if (verbose) printf("\nDEFAULT CTOR"
                            "\n============\n");

        const int D = -1;

        if (verbose)
            printf("\nTesting with various allocator configurations.\n");

        for (char cfg = 'a'; cfg <= 'c'; ++cfg) {

            const char CONFIG = cfg;  // how we specify the allocator

            bslma::TestAllocator da("default",   veryVeryVeryVerbose);
            bslma::TestAllocator fa("footprint", veryVeryVeryVerbose);
            bslma::TestAllocator sa("supplied",  veryVeryVeryVerbose);

            bslma::DefaultAllocatorGuard dag(&da);

            Obj                  *objPtr = 0;
            bslma::TestAllocator *objAllocatorPtr = 0;

            switch (CONFIG) {
              case 'a': {
                objAllocatorPtr = &da;
              } break;
              case 'b': {
                objAllocatorPtr = &da;
              } break;
              case 'c': {
                objAllocatorPtr = &sa;
              } break;
              default: {
                BSLS_ASSERT_INVOKE_NORETURN("Bad allocator config.");
              } return testStatus;                                    // RETURN
            }

            bslma::TestAllocator&  oa = *objAllocatorPtr;
            bslma::TestAllocator& noa = 'c' != CONFIG ? sa : da;

            BSLMA_TESTALLOCATOR_EXCEPTION_TEST_BEGIN(oa) {
                if (veryVeryVerbose) { T_ T_ Q(ExceptionTestBody) }

                bslma::TestAllocatorMonitor tam(&oa);
                switch (CONFIG) {
                  case 'a': {
                    objPtr = new (fa) Obj();
                  } break;
                  case 'b': {
                    objPtr = new (fa) Obj((bslma::Allocator *) 0);
                  } break;
                  case 'c': {
                    objPtr = new (fa) Obj(&sa);
                  } break;
                  default: {
                    BSLS_ASSERT_INVOKE_NORETURN("Bad allocator config.");
                  } return testStatus;                                // RETURN
                }
                ASSERTV(CONFIG,  tam.isInUseSame());
            } BSLMA_TESTALLOCATOR_EXCEPTION_TEST_END;

            ASSERTV(CONFIG, sizeof(Obj) == fa.numBytesInUse());

            Obj& mX = *objPtr;  const Obj& X = mX;

            // Verify any attribute allocators are installed properly.
            ASSERTV(CONFIG, &oa, X.allocator(), &oa == X.allocator());

            // Verify no allocation from the non-object allocators.
            ASSERTV(CONFIG, noa.numBlocksTotal(), 0 == noa.numBlocksTotal());

            // Verify the object's attribute values.
            ASSERTV(CONFIG, D, int(X), D == int(X));

            // Verify no temporary memory is allocated from the object
            // allocator.
            ASSERTV(CONFIG, oa.numBlocksTotal(), oa.numBlocksInUse(),
                         oa.numBlocksTotal() == oa.numBlocksInUse());

            // Reclaim dynamically allocated object under test.
            fa.deleteObject(objPtr);

            // Verify all memory is released on object destruction.
            ASSERTV(fa.numBlocksInUse(),  0 ==  fa.numBlocksInUse());
            ASSERTV(oa.numBlocksInUse(),  0 ==  oa.numBlocksInUse());
            ASSERTV(noa.numBlocksTotal(), 0 == noa.numBlocksTotal());

            // Double check that no object memory was allocated.
            ASSERTV(CONFIG, 0 == oa.numBlocksTotal());
        }

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

        {
            Obj X;
            ASSERT(-1 == X);
            Obj Y(0);
            ASSERT(0 == Y);
            Obj Z(1);
            ASSERT(1 == Z);
        }
      } break;
      default: {
        fprintf(stderr, "WARNING: CASE `%d' NOT FOUND.\n", test);
        testStatus = -1;
      }
    }

    // CONCERN: In no case does memory come from the global allocator.

    ASSERTV(globalAllocator.numBlocksTotal(),
            0 == globalAllocator.numBlocksTotal());

    if (testStatus > 0) {
        fprintf(stderr, "Error, non-zero test status = %d.\n", testStatus);
    }
    return testStatus;
}

// ----------------------------------------------------------------------------
// Copyright 2016 Bloomberg Finance L.P.
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
