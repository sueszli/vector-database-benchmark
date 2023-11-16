// bsls_cpp11.t.cpp                                                   -*-C++-*-

#include <bsls_cpp11.h>

#include <bsls_bsltestutil.h>

#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>

// Warning: the following 'using' declarations interfere with the testing of
// the macros defined in this component.  Please do not uncomment them.
//  using namespace BloombergLP;
//  using namespace std;

//=============================================================================
//                                 TEST PLAN
//-----------------------------------------------------------------------------
//                                  Overview
//                                  --------
//-----------------------------------------------------------------------------
// [ 1] BSLS_CPP11_CONSTEXPR
// [ 2] BSLS_CPP11_EXPLICIT
// [ 3] BSLS_CPP11_FINAL (class)
// [ 4] BSLS_CPP11_FINAL (function)
// [ 5] BSLS_CPP11_NOEXCEPT
// [ 5] BSLS_CPP11_NOEXCEPT_SPECIFICATION
// [ 5] BSLS_CPP11_NOEXCEPT_OPERATOR
// [ 6] BSLS_CPP11_OVERRIDE
// [ 5] BSLS_CPP11_PROVISIONALLY_FALSE
//-----------------------------------------------------------------------------
// [ 7] MACRO SAFETY
// [ 8] USAGE EXAMPLE

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

#define STRINGIFY2(...) "" #__VA_ARGS__
#define STRINGIFY(a) STRINGIFY2(a)

//=============================================================================

namespace
{
    template <class TYPE>
    class Optional
    {
        TYPE* d_value;
    public:
        Optional(): d_value() {}
            // The default constructor zero initializes the object.
        explicit Optional(const TYPE& value): d_value(new TYPE(value)) {}
            // This constructor sets the object to the specified 'value'.
        ~Optional() { delete d_value; }
            // The destructor destructs the object.

        // ...

        BSLS_CPP11_EXPLICIT
        operator bool() const { return d_value; }
            // The conversion operators returns the object's value.
    };

    template <class, bool Pred>
    struct TestMetafunction {
        enum { value = Pred };
          // Used to construct 'void foo() noexcept(expr-with-commas)'
    };

    void noThrow1() BSLS_CPP11_NOEXCEPT                           {}
    void noThrow2() BSLS_CPP11_NOEXCEPT_SPECIFICATION(true)       {}
    void noThrow3() BSLS_CPP11_NOEXCEPT_SPECIFICATION(
                        BSLS_CPP11_NOEXCEPT_OPERATOR(noThrow1())) {}
    void noThrow4() BSLS_CPP11_NOEXCEPT_SPECIFICATION(
                        TestMetafunction<void, true>::value)      {}
    void throws1()  BSLS_CPP11_NOEXCEPT_SPECIFICATION(false)      {}
    void throws2()  BSLS_CPP11_NOEXCEPT_SPECIFICATION(
                        BSLS_CPP11_NOEXCEPT_OPERATOR(throws1()))  {}
    void throws3()  BSLS_CPP11_NOEXCEPT_SPECIFICATION(
                        TestMetafunction<void, false>::value)     {}
    template <class, class>
    void throws4() BSLS_CPP11_NOEXCEPT_SPECIFICATION(false)       {}
      // 'throws4<T, U>()' is used to test the operator
      // 'noexcept(expr-with-commas)'

    void throws5() BSLS_CPP11_NOEXCEPT_SPECIFICATION(
                                                BSLS_CPP11_PROVISIONALLY_FALSE)
                                                                  {}
}  // close unnamed namespace

// ============================================================================
//                              HELPER FUNCTIONS
// ----------------------------------------------------------------------------

static void printFlags()
    // Print a diagnostic message to standard output if any of the preprocessor
    // flags of interest are defined, and their value if a value had been set.
    // An "Enter" and "Leave" message is printed unconditionally so there is
    // some report even if all of the flags are undefined.
{
    printf("printFlags: Enter\n");

    printf("\n  printFlags: bsls_cpp11 Macros\n");

    printf("\n  BSLS_CPP11_CONSTEXPR: ");
#ifdef BSLS_CPP11_CONSTEXPR
    printf("%s\n", STRINGIFY(BSLS_CPP11_CONSTEXPR) );
#else
    printf("UNDEFINED\n");
#endif

    printf("\n  BSLS_CPP11_DELETED: ");
#ifdef BSLS_CPP11_DELETED
    printf("%s\n", STRINGIFY(BSLS_CPP11_DELETED) );
#else
    printf("UNDEFINED\n");
#endif

    printf("\n  BSLS_CPP11_EXPLICIT: ");
#ifdef BSLS_CPP11_EXPLICIT
    printf("%s\n", STRINGIFY(BSLS_CPP11_EXPLICIT) );
#else
    printf("UNDEFINED\n");
#endif

    printf("\n  BSLS_CPP11_FINAL: ");
#ifdef BSLS_CPP11_FINAL
    printf("%s\n", STRINGIFY(BSLS_CPP11_FINAL) );
#else
    printf("UNDEFINED\n");
#endif

    printf("\n  BSLS_CPP11_NOEXCEPT: ");
#ifdef BSLS_CPP11_NOEXCEPT
    printf("%s\n", STRINGIFY(BSLS_CPP11_NOEXCEPT) );
#else
    printf("UNDEFINED\n");
#endif

    printf("\n  BSLS_CPP11_NOEXCEPT_AVAILABLE: ");
#ifdef BSLS_CPP11_NOEXCEPT_AVAILABLE
    printf("%s\n", STRINGIFY(BSLS_CPP11_NOEXCEPT_AVAILABLE) );
#else
    printf("UNDEFINED\n");
#endif

    printf("\n  BSLS_CPP11_NOEXCEPT_OPERATOR(...): ");
#ifdef BSLS_CPP11_NOEXCEPT_OPERATOR
    printf("%s\n", STRINGIFY(BSLS_CPP11_NOEXCEPT_OPERATOR(...)) );
#else
    printf("UNDEFINED\n");
#endif

    printf("\n  BSLS_CPP11_NOEXCEPT_SPECIFICATION(...): ");
#ifdef BSLS_CPP11_NOEXCEPT_SPECIFICATION
    printf("%s\n", STRINGIFY(BSLS_CPP11_NOEXCEPT_SPECIFICATION(...)) );
#else
    printf("UNDEFINED\n");
#endif

    printf("\n  BSLS_CPP11_OVERRIDE: ");
#ifdef BSLS_CPP11_OVERRIDE
    printf("%s\n", STRINGIFY(BSLS_CPP11_OVERRIDE) );
#else
    printf("UNDEFINED\n");
#endif

    printf("\n  BSLS_CPP11_PROVISIONALLY_FALSE: ");
#ifdef BSLS_CPP11_PROVISIONALLY_FALSE
    printf("%s\n", STRINGIFY(BSLS_CPP11_PROVISIONALLY_FALSE) );
#else
    printf("UNDEFINED\n");
#endif

    printf("\n\nprintFlags: Leave\n");
}

//=============================================================================
//                                MAIN PROGRAM
//-----------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    int                 test = argc > 1 ? atoi(argv[1]) : 0;
    bool             verbose = argc > 2;
    bool         veryVerbose = argc > 3;
    bool     veryVeryVerbose = argc > 4;

    (void)        veryVerbose;  // unused variable warning
    (void)    veryVeryVerbose;  // unused variable warning

    printf( "TEST %s CASE %d\n", __FILE__, test);

    if (veryVeryVerbose) {
        printFlags();
    }

    switch (test) { case 0:
      case 8: {
        // --------------------------------------------------------------------
        // TESTING USAGE EXAMPLE
        //
        // Concerns:
        //: 1 The usage example provided in the component header file must
        //:   compile, link, and run on all platforms as shown.
        //
        // Plan:
        //: 1 Run the usage example.
        //
        // Testing:
        //   USAGE EXAMPLE
        // --------------------------------------------------------------------

        if (verbose) printf("\nTESTING USAGE EXAMPLE"
                            "\n=====================\n");

#undef    FAIL_USAGE_OVERRIDE
//#define FAIL_USAGE_OVERRIDE 1

        Optional<int> value;
        ASSERT(bool(value) == false);
        if (value) { /*... */ }
#if !defined(BSLS_COMPILERFEATURES_SUPPORT_OPERATOR_EXPLICIT) \
 || defined(FAIL_USAGE_EXPLICIT)
        bool flag = value;
        ASSERT(flag == false);
#endif

        class FinalClass BSLS_CPP11_FINAL
        {
            int d_value;
        public:
            explicit FinalClass(int value = 0)
                // Initialize the object with the optionally specified 'value'.
                : d_value(value) {}
            int value() const { return d_value; }
                // Returns the object's value.
        };
        class FinalClassDerived
#if !defined(BSLS_COMPILERFEATURES_SUPPORT_FINAL) \
 || defined(FAIL_USAGE_FINAL_CLASS)
            : public FinalClass
#endif
        {
            int d_anotherValue;
        public:
            explicit FinalClassDerived(int value)
                // Initialize the object with the specified 'value'.
                : d_anotherValue(2 * value) {
            }
            int anotherValue() const { return d_anotherValue; }
                // Returns another value for the object.
        };

        FinalClass finalValue(1);
        ASSERT(finalValue.value() == 1);
        FinalClassDerived derivedValue(2);
        ASSERT(derivedValue.anotherValue() == 4);

        struct FinalFunctionBase
        {
            virtual int f() { return 0; }
                // Returns a value associated with the class's type.
        };
        struct FinalFunctionDerived: FinalFunctionBase
        {
            int f() BSLS_CPP11_FINAL
                // Returns a value associated with the class's type.
            { return 1; }
        };
        struct FinalFunctionFailure: FinalFunctionDerived
        {
#if !defined(BSLS_COMPILERFEATURES_SUPPORT_FINAL) \
 || defined(FAIL_USAGE_FINAL_FUNCTION)
            int f()
                // Returns a value associated with the class's type.
            { return 2; }
#endif
        };

        FinalFunctionBase finalFunctionBase;
        ASSERT(finalFunctionBase.f()    == 0);

        FinalFunctionDerived finalFunctionDerived;
        ASSERT(finalFunctionDerived.f() == 1);

        FinalFunctionFailure finalFunctionFailure;
#if defined(BSLS_COMPILERFEATURES_SUPPORT_FINAL)
        ASSERT(finalFunctionFailure.f() == 1);
#else
        ASSERT(finalFunctionFailure.f() == 2);
#endif

        struct OverrideBase
        {
            virtual int f() const
                // Returns a value associated with the type.
            { return 0; }
        };
        struct OverrideSuccess: OverrideBase
        {
            int f() const BSLS_CPP11_OVERRIDE
                // Returns a value associated with the type.
            { return 1; }
        };
        struct OverrideFailure: OverrideBase
        {
            int f()
                // Returns a value associated with the type.
#if !defined(FAIL_USAGE_OVERRIDE)
                    const BSLS_CPP11_OVERRIDE
#endif
            { return 2; }
        };

        OverrideBase overrideBase;
        ASSERT(overrideBase.f() == 0);
        OverrideSuccess overrideSuccess;
        ASSERT(overrideSuccess.f() == 1);
        ASSERT(static_cast<const OverrideBase&>(overrideSuccess).f() == 1);
        OverrideFailure overrideFailure;
        ASSERT(overrideFailure.f() == 2);
#if defined(FAIL_USAGE_OVERRIDE)
        ASSERT(static_cast<const OverrideBase&>(overrideFailure).f() == 0);
#endif

#undef    FAIL_USAGE_OVERRIDE
      } break;
      case 7: {
        // --------------------------------------------------------------------
        // TESTING MACRO SAFETY
        //
        // Concerns:
        //: 1 The macros should be usable outside 'namespace' 'BloombergLP' and
        //:   'namespace' 'std'.
        //
        // Plan:
        //: 1 The 'using' declarations at the top of the file are specifically
        //:   commented out to test the concern.  If the concerns is violated,
        //:   the test driver should fail to compile.
        //
        // Testing:
        //   MACRO SAFETY
        // --------------------------------------------------------------------

        if (verbose) printf("\nTESTING MACRO SAFETY"
                            "\n====================\n");

        struct TestBase {
            virtual void f() = 0;
                // Used to verify the derived type overrides the function.
            virtual void g() = 0;
                // Used to verify the derived type overrides the function.
        };
        struct TestIntermediate
            : TestBase {
            void f() BSLS_CPP11_OVERRIDE BSLS_CPP11_FINAL
                // Override the abstraction function.
            {}
            void g() BSLS_CPP11_FINAL BSLS_CPP11_OVERRIDE
                // Override the abstraction function.
            {}
        };
        struct TestDerived BSLS_CPP11_FINAL
            : TestIntermediate {
            bool test()
                // Returns 'true'.
            { return true; }
            BSLS_CPP11_EXPLICIT
            operator bool() const
                // The conversion operator always returns 'true'.
            { return true; }
        };

        TestDerived object;
        ASSERT(object.test());
        ASSERT(bool(object));
      } break;
      case 6: {
        // --------------------------------------------------------------------
        // TESTING: BSLS_CPP11_OVERRIDE
        //
        // Concerns:
        //: 1 Marking an overriding function as 'override' using
        //:   'BSLS_CPP11_OVERRIDE' shall compile OK both with C++03 and C++11
        //:   mode.
        //
        //: 2 Marking a function which isn't overriding a 'virtual' function as
        //:   'override' using 'BSLS_CPP11_OVERRIDE' shall fail to compile when
        //:   compiling using C++11 mode.  It may fail when using C++03 mode
        //:   due to a warning about an overload hiding base class functions.
        //
        // Plan:
        //: 1 Define a base class with a 'virtual' function and two derived
        //:   classes which override the function correctly and incorrectly.
        //:   In both cases use the 'BSLS_CPP11_OVERRIDE' macro and determine
        //:   if the compilation is successful.  The incorrect use is guarded
        //:   by a the macro 'FAIL_OVERRIDE' to restrict compilation failure to
        //:   compilations with this macro defined.
        //
        // Testing:
        //   BSLS_CPP11_OVERRIDE
        // --------------------------------------------------------------------

        if (verbose) printf("\nTESTING: BSLS_CPP11_OVERRIDE"
                            "\n============================\n");

#undef    FAIL_OVERRIDE
//#define FAIL_OVERRIDE 1

        struct Base
        {
            virtual int f() const
                // Returns a value for each type.
            {
                return 0;
            }
        };
        struct OverrideOK
            : Base
        {
            int f() const BSLS_CPP11_OVERRIDE
                // Returns a value specific to this type.
            {
                return 1;
            }
        };
        struct OverrideFail
            : Base
        {
            int f()
                // Returns a value specific to this type.
#if !defined(FAIL_OVERRIDE)
                    const BSLS_CPP11_OVERRIDE
#endif
            {
                return 2;
            }
        };

        OverrideOK ok;
        ASSERT(ok.f() == 1);
        ASSERT(static_cast<const Base&>(ok).f() == 1);
        OverrideFail fail;
        ASSERT(fail.f() == 2);
#if defined(FAIL_OVERRIDE)
        ASSERT(static_cast<const Base&>(fail).f() == 0);
#endif
#undef FAIL_OVERRIDE
      } break;
      case 5: {
        // --------------------------------------------------------------------
        // TESTING BSLS_CPP11_NOEXCEPT AND VARIATIONS
        //
        // Concerns:
        //: 1 Marking a function 'noexcept' using 'BSLS_CPP11_NOEXCEPT' or
        //:   'BSLS_CPP11_NOEXCEPT_SPECIFICATION(pred)' or
        //:   BSLS_CPP11_NOEXCEPT_SPECIFICATION(
        //:       BSLS_CPP11_NOEXCEPT_OPERATOR(expr))' should result in a
        //:   successful compilation in C++03 mode.
        //:
        //: 2 Marking a function 'noexcept' or 'noexcept(bool)' using
        //:   'BSLS_CPP11_NOEXCEPT' or
        //:   'BSLS_CPP11_NOEXCEPT_SPECIFICATION(pred)' or
        //:   BSLS_CPP11_NOEXCEPT_SPECIFICATION(
        //:       BSLS_CPP11_NOEXCEPT_OPERATOR(expr))' should be detectable
        //:   using BSLS_CPP11_NOEXCEPT_OPERATOR(function(...))`.
        //:
        //: 3 The `BSLS_CPP11_NOEXCEPT_SPECIFICATION(pred)` and
        //:   'BSLS_CPP11_NOEXCEPT_OPERATOR(expr)' macros both allow commas in
        //:   template parameter lists.
        //
        // Plan:
        //: 1 Define a function marking it 'noexcept' using the various forms
        //:   of the macro.  Then use
        //:   `BSLS_CPP11_NOEXCEPT_OPERATOR(function(...))` to check that the
        //:   function's 'noexcept' specification matches the expected
        //:   specification.
        //
        // NOTE: The test functions are called only to prevent
        //  '-Wunused-function' warning.
        //
        // Testing:
        //   BSLS_CPP11_NOEXCEPT
        //   BSLS_CPP11_NOEXCEPT_SPECIFICATION
        //   BSLS_CPP11_NOEXCEPT_OPERATOR
        //   BSLS_CPP11_PROVISIONALLY_FALSE
        // --------------------------------------------------------------------

        if (verbose) printf("\nTESTING BSLS_CPP11_NOEXCEPT AND VARIATIONS"
                            "\n==========================================\n");

#if defined(BSLS_COMPILERFEATURES_SUPPORT_NOEXCEPT)
        const bool hasNoexceptSupport = true;
#else
        const bool hasNoexceptSupport = false;
#endif
        noThrow1();
        const bool isNoThrow1 = BSLS_CPP11_NOEXCEPT_OPERATOR(noThrow1());
        ASSERT(isNoThrow1 == hasNoexceptSupport);

        noThrow2();
        const bool isNoThrow2 = BSLS_CPP11_NOEXCEPT_OPERATOR(noThrow2());
        ASSERT(isNoThrow2 == hasNoexceptSupport);

        noThrow3();
        const bool isNoThrow3 = BSLS_CPP11_NOEXCEPT_OPERATOR(noThrow3());
        ASSERT(isNoThrow3 == hasNoexceptSupport);

        noThrow4();
        const bool isNoThrow4 = BSLS_CPP11_NOEXCEPT_OPERATOR(noThrow4());
        ASSERT(isNoThrow4 == hasNoexceptSupport);

        throws1();
        const bool isNoThrow5 = BSLS_CPP11_NOEXCEPT_OPERATOR(throws1());
        ASSERT(isNoThrow5 == false);

        throws2();
        const bool isNoThrow6 = BSLS_CPP11_NOEXCEPT_OPERATOR(throws2());
        ASSERT(isNoThrow6 == false);

        throws3();
        const bool isNoThrow7 = BSLS_CPP11_NOEXCEPT_OPERATOR(throws3());
        ASSERT(isNoThrow7 == false);

        throws4<void, void>();
        const bool isNoThrow8 = BSLS_CPP11_NOEXCEPT_OPERATOR(
                                                        throws4<void, void>());
        ASSERT(isNoThrow8 == false);

        throws5();
        const bool isNoThrow9 = BSLS_CPP11_NOEXCEPT_OPERATOR(throws5());
        ASSERT(isNoThrow9 == false);

        ASSERT(hasNoexceptSupport == BSLS_CPP11_NOEXCEPT_AVAILABLE);
        ASSERT(false              == BSLS_CPP11_PROVISIONALLY_FALSE);
      } break;
      case 4: {
        // --------------------------------------------------------------------
        // TESTING BSLS_CPP11_FINAL (fFUNCTION)
        //
        // Concerns:
        //: 1 Marking a 'virtual' function as 'final' should compile.
        //:
        //: 2 Trying to override a function marked as 'final' shall fail to to
        //:   compile when compiling with C++11 mode.  Since 'BSLS_CPP11_FINAL'
        //:   is replaced by nothing when compiling with C++03 mode the could
        //:   should compile in this case.
        //
        // Plan:
        //: 1 Define a base class with a 'virtual' function and mark it 'final'
        //:   using 'BSLS_CPP11_FINAL' in a derived class.  Creating a further
        //:   derived class which also overrides the function marked as 'final'
        //:   should fail compilation when compiling with C++11 mode.
        //
        // Testing:
        //   BSLS_CPP11_FINAL (function)
        // --------------------------------------------------------------------

        if (verbose) printf("\nTESTING BSLS_CPP11_FINAL (FUNCTION)"
                            "\n===================================\n");

        struct FinalFunctionBase
        {
            virtual int f()
                // Returns a value for each type.
            { return 0; }
        };
        struct FinalFunctionDerived: FinalFunctionBase
        {
            int f()
                // Returns a value for the specific type.
                BSLS_CPP11_FINAL
            { return 1; }
        };
        struct FinalFunctionFailure: FinalFunctionDerived
        {
#if !defined(BSLS_COMPILERFEATURES_SUPPORT_FINAL) \
 || defined(FAIL_FINAL_FUNCTION)
            int f()
                // Returns a value for the specific type.
            { return 2; }
#endif
        };

        FinalFunctionBase finalFunctionBase;
        ASSERT(finalFunctionBase.f() == 0);

        FinalFunctionDerived finalFunctionDerived;
        ASSERT(finalFunctionDerived.f() == 1);

        FinalFunctionFailure finalFunctionFailure;
#if defined(BSLS_COMPILERFEATURES_SUPPORT_FINAL)
        ASSERT(finalFunctionFailure.f() == 1);
#else
        ASSERT(finalFunctionFailure.f() == 2);
#endif
      } break;
      case 3: {
        // --------------------------------------------------------------------
        // TESTING: BSLS_CPP11_FINAL (CLASS)
        //
        // Concerns:
        //: 1 Marking a class 'final' using 'BSLS_CPP11_FINAL' should result in
        //:   a successful compilation.
        //:
        //: 2 Trying to further derive from a function marked as 'final' shall
        //:   fail to compile when compiling with C++11 mode.  Since
        //:   'BSLS_CPP11_FINAL' is replaced by nothing when compiling with
        //:   C++03 mode the could should compile in this case.
        //
        // Plan:
        //: 1 Define a class marking it 'final' using 'BSLS_CPP11_FINAL'.
        //:   Creating a derived class from the 'final' class should fail
        //:   compilation when compiling with C++11 mode.
        //
        // Testing:
        //   BSLS_CPP11_FINAL (class)
        // --------------------------------------------------------------------

        if (verbose) printf("\nTESTING: BSLS_CPP11_FINAL (CLASS)"
                            "\n=================================\n");

        class FinalClass BSLS_CPP11_FINAL
        {
            int d_value;
        public:
            explicit FinalClass(int value = 0)
                // Initialize with the optionally specified 'value'.
                : d_value(value) {}
            int value() const
                // Returns the object's value.
            { return d_value; }
        };
        class FinalClassDerived
#if !defined(BSLS_COMPILERFEATURES_SUPPORT_FINAL) || defined(FAIL_FINAL_CLASS)
            : public FinalClass
#endif
        {
            int d_anotherValue;
        public:
            explicit FinalClassDerived(int value)
                // Initialize with the specified 'value'.
                : d_anotherValue(2 * value) {
            }
            int anotherValue() const
                // Returns another value for the object.
            { return d_anotherValue; }
        };

        FinalClass finalValue(1);
        ASSERT(finalValue.value() == 1);
        FinalClassDerived derivedValue(2);
        ASSERT(derivedValue.anotherValue() == 4);

      } break;
      case 2: {
        // --------------------------------------------------------------------
        // TESTING: BSLS_CPP11_EXPLICIT
        //
        // Concerns:
        //: 1 Marking a conversion operator 'explicit' using
        //:   'BSLS_CPP11_EXPLICIT' needs to allow explicit conversions.
        //:
        //: 2 Marking a conversion operator 'explicit' using
        //:   'BSLS_CPP11_EXPLICIT' should prevent attempts of implicit
        //:   conversion when compiling with C++11 mode.  When compiling with
        //:   C++03 mode compilation will succeed.
        //
        // Plan:
        //: 1 Define a class with an explicit conversion operator and verify
        //:   that explicit and implicit conversions succeed when using C++03
        //:   mode.  When compiling with C++11 mode the implicit conversion
        //:   should fail.
        //
        // Testing:
        //   BSLS_CPP11_EXPLICIT
        // --------------------------------------------------------------------

        if (verbose) printf("\nTESTING: BSLS_CPP11_EXPLICIT"
                            "\n============================\n");

        struct Explicit
        {
            BSLS_CPP11_EXPLICIT
            operator int() const
                // Returns a value for the object.
            { return 3; }
        };

        Explicit explicitObject;

        int explicitResult(explicitObject);
        ASSERT(explicitResult == 3);
#if !defined(BSLS_COMPILERFEATURES_SUPPORT_OPERATOR_EXPLICIT) \
 || defined(FAIL_EXPLICIT)
        int implicitResult = explicitObject;
        ASSERT(implicitResult == 3);
#endif
      } break;
      case 1:{
        // --------------------------------------------------------------------
        // TESTING BSLS_CPP11_CONSTEXPR
        //
        // Concerns:
        //: 1 Marking a function 'constexpr' using 'BSLS_CPP11_CONSTEXPR'
        //:   should result in a successful compilation.
        //:
        //: 2 Marking a function 'constexpr' using 'BSLS_CPP11_CONSTEXPR'
        //:   should make the test driver not compile if the use of the
        //:   resulting constexpr function is used illegally.
        //
        // Plan:
        //: 1 Define a struct marking its various member functions as constexpr
        //:   functions.  Verify that if the constexpr member functions are not
        //:   used appropriately the program will fail to compile in cpp11
        //:   mode.
        //:
        //: 2 Since the correct behaviour will case the program to not compile,
        //:   it is rather difficult to create test cases that actually tests
        //:   the feature and still have the test driver pass.  As such, this
        //:   tests must be manually checked to ensure that the program does
        //:   not compile if testStruct is not used correctly.
        //
        // Testing:
        //   BSLS_CPP11_CONSTEXPR
        // --------------------------------------------------------------------

        if (verbose) printf("\nTESTING BSLS_CPP11_CONSTEXPR"
                            "\n============================\n");

        struct testStruct {
            BSLS_CPP11_CONSTEXPR testStruct (int i) : value(i){}
            BSLS_CPP11_CONSTEXPR operator int() const {return value; }
            BSLS_CPP11_CONSTEXPR operator long() const {return 1.0; }
            private:
                int value;
        };

        BSLS_CPP11_CONSTEXPR testStruct B (15);
#if defined(BSLS_COMPILERFEATURES_SUPPORT_CONSTEXPR)
        BSLS_CPP11_CONSTEXPR int X (B);
#else
        int X (B);
#endif
        (void)X;  // unused variable warning
      } break;
      default: {
        fprintf( stderr, "WARNING: CASE `%d` NOT FOUND.\n" , test);
        testStatus = -1;
      }
    }

    if (testStatus > 0) {
        fprintf( stderr, "Error, non-zero test status = %d.\n", testStatus );
    }

    return testStatus;
}

// ----------------------------------------------------------------------------
// Copyright 2015 Bloomberg Finance L.P.
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
