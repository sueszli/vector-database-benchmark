// balb_testmessages.cpp           *DO NOT EDIT*           @generated -*-C++-*-

#include <bsls_ident.h>
BSLS_IDENT_RCSID(balb_testmessages_cpp,"$Id$ $CSID$")

#include <balb_testmessages.h>

#include <bdlat_formattingmode.h>
#include <bdlat_valuetypefunctions.h>
#include <bdlde_utf8util.h>
#include <bdlb_print.h>
#include <bdlb_printmethods.h>
#include <bdlb_string.h>

#include <bdlb_nullableallocatedvalue.h>
#include <bdlb_nullablevalue.h>
#include <bdlt_datetimetz.h>
#include <bsl_string.h>
#include <bsl_vector.h>
#include <bsls_types.h>
#include <bdlb_print.h>
#include <bslim_printer.h>
#include <bsls_assert.h>

#include <bsl_iomanip.h>
#include <bsl_limits.h>
#include <bsl_ostream.h>

namespace BloombergLP {
namespace balb {

                               // -------------
                               // class Choice4
                               // -------------

// CONSTANTS

const char Choice4::CLASS_NAME[] = "Choice4";

const bdlat_SelectionInfo Choice4::SELECTION_INFO_ARRAY[] = {
    {
        SELECTION_ID_SELECTION1,
        "selection1",
        sizeof("selection1") - 1,
        "",
        bdlat_FormattingMode::e_TEXT
    },
    {
        SELECTION_ID_SELECTION2,
        "selection2",
        sizeof("selection2") - 1,
        "",
        bdlat_FormattingMode::e_DEC
    }
};

// CLASS METHODS

const bdlat_SelectionInfo *Choice4::lookupSelectionInfo(
        const char *name,
        int         nameLength)
{
    for (int i = 0; i < 2; ++i) {
        const bdlat_SelectionInfo& selectionInfo =
                    Choice4::SELECTION_INFO_ARRAY[i];

        if (nameLength == selectionInfo.d_nameLength
        &&  0 == bsl::memcmp(selectionInfo.d_name_p, name, nameLength))
        {
            return &selectionInfo;
        }
    }

    return 0;
}

const bdlat_SelectionInfo *Choice4::lookupSelectionInfo(int id)
{
    switch (id) {
      case SELECTION_ID_SELECTION1:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION1];
      case SELECTION_ID_SELECTION2:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION2];
      default:
        return 0;
    }
}

// CREATORS

Choice4::Choice4(
    const Choice4& original,
    bslma::Allocator *basicAllocator)
: d_selectionId(original.d_selectionId)
, d_allocator_p(bslma::Default::allocator(basicAllocator))
{
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION1: {
        new (d_selection1.buffer())
            bsl::vector<bsl::string>(
                original.d_selection1.object(), d_allocator_p);
      } break;
      case SELECTION_ID_SELECTION2: {
        new (d_selection2.buffer())
            int(original.d_selection2.object());
      } break;
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
    }
}

// MANIPULATORS

Choice4&
Choice4::operator=(const Choice4& rhs)
{
    if (this != &rhs) {
        switch (rhs.d_selectionId) {
          case SELECTION_ID_SELECTION1: {
            makeSelection1(rhs.d_selection1.object());
          } break;
          case SELECTION_ID_SELECTION2: {
            makeSelection2(rhs.d_selection2.object());
          } break;
          default:
            BSLS_ASSERT(SELECTION_ID_UNDEFINED == rhs.d_selectionId);
            reset();
        }
    }

    return *this;
}

void Choice4::reset()
{
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION1: {
        typedef bsl::vector<bsl::string> Type;
        d_selection1.object().~Type();
      } break;
      case SELECTION_ID_SELECTION2: {
        // no destruction required
      } break;
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
    }

    d_selectionId = SELECTION_ID_UNDEFINED;
}

int Choice4::makeSelection(int selectionId)
{
    switch (selectionId) {
      case SELECTION_ID_SELECTION1: {
        makeSelection1();
      } break;
      case SELECTION_ID_SELECTION2: {
        makeSelection2();
      } break;
      case SELECTION_ID_UNDEFINED: {
        reset();
      } break;
      default:
        return -1;
    }
    return 0;
}

int Choice4::makeSelection(const char *name, int nameLength)
{
    const bdlat_SelectionInfo *selectionInfo =
                                         lookupSelectionInfo(name, nameLength);
    if (0 == selectionInfo) {
       return -1;
    }

    return makeSelection(selectionInfo->d_id);
}

bsl::vector<bsl::string>& Choice4::makeSelection1()
{
    if (SELECTION_ID_SELECTION1 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_selection1.object());
    }
    else {
        reset();
        new (d_selection1.buffer())
                bsl::vector<bsl::string>(d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION1;
    }

    return d_selection1.object();
}

bsl::vector<bsl::string>& Choice4::makeSelection1(const bsl::vector<bsl::string>& value)
{
    if (SELECTION_ID_SELECTION1 == d_selectionId) {
        d_selection1.object() = value;
    }
    else {
        reset();
        new (d_selection1.buffer())
                bsl::vector<bsl::string>(value, d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION1;
    }

    return d_selection1.object();
}

int& Choice4::makeSelection2()
{
    if (SELECTION_ID_SELECTION2 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_selection2.object());
    }
    else {
        reset();
        new (d_selection2.buffer())
            int();
        d_selectionId = SELECTION_ID_SELECTION2;
    }

    return d_selection2.object();
}

int& Choice4::makeSelection2(int value)
{
    if (SELECTION_ID_SELECTION2 == d_selectionId) {
        d_selection2.object() = value;
    }
    else {
        reset();
        new (d_selection2.buffer())
                int(value);
        d_selectionId = SELECTION_ID_SELECTION2;
    }

    return d_selection2.object();
}

// ACCESSORS

bsl::ostream& Choice4::print(
    bsl::ostream& stream,
    int           level,
    int           spacesPerLevel) const
{
    bslim::Printer printer(&stream, level, spacesPerLevel);
    printer.start();
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION1: {
        printer.printAttribute("selection1", d_selection1.object());
      }  break;
      case SELECTION_ID_SELECTION2: {
        printer.printAttribute("selection2", d_selection2.object());
      }  break;
      default:
        stream << "SELECTION UNDEFINED\n";
    }
    printer.end();
    return stream;
}


const char *Choice4::selectionName() const
{
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION1:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION1].name();
      case SELECTION_ID_SELECTION2:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION2].name();
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
        return "(* UNDEFINED *)";
    }
}

                              // ---------------
                              // class CustomInt
                              // ---------------

// PUBLIC CLASS METHODS

int CustomInt::checkRestrictions(const int& value)
{
    if (1000 < value) {
        return -1;
    }

    return 0;
}

// CONSTANTS

const char CustomInt::CLASS_NAME[] = "CustomInt";



                             // ------------------
                             // class CustomString
                             // ------------------

// PUBLIC CLASS METHODS

int CustomString::checkRestrictions(const bsl::string& value)
{
    const char              *invalid = 0;
    bdlde::Utf8Util::IntPtr  result  = bdlde::Utf8Util::numCodePointsIfValid(
           &invalid, value.c_str(), value.length());

    if (result < 0 || 8 < result) {
        return -1;                                                    // RETURN
    }

    return 0;
}

// CONSTANTS

const char CustomString::CLASS_NAME[] = "CustomString";



                              // ----------------
                              // class Enumerated
                              // ----------------

// CONSTANTS

const char Enumerated::CLASS_NAME[] = "Enumerated";

const bdlat_EnumeratorInfo Enumerated::ENUMERATOR_INFO_ARRAY[] = {
    {
        Enumerated::NEW_YORK,
        "NEW_YORK",
        sizeof("NEW_YORK") - 1,
        ""
    },
    {
        Enumerated::NEW_JERSEY,
        "NEW_JERSEY",
        sizeof("NEW_JERSEY") - 1,
        ""
    },
    {
        Enumerated::LONDON,
        "LONDON",
        sizeof("LONDON") - 1,
        ""
    }
};

// CLASS METHODS

int Enumerated::fromInt(Enumerated::Value *result, int number)
{
    switch (number) {
      case Enumerated::NEW_YORK:
      case Enumerated::NEW_JERSEY:
      case Enumerated::LONDON:
        *result = (Enumerated::Value)number;
        return 0;
      default:
        return -1;
    }
}

int Enumerated::fromString(
        Enumerated::Value *result,
        const char         *string,
        int                 stringLength)
{
    for (int i = 0; i < 3; ++i) {
        const bdlat_EnumeratorInfo& enumeratorInfo =
                    Enumerated::ENUMERATOR_INFO_ARRAY[i];

        if (stringLength == enumeratorInfo.d_nameLength
        &&  0 == bsl::memcmp(enumeratorInfo.d_name_p, string, stringLength))
        {
            *result = (Enumerated::Value)enumeratorInfo.d_value;
            return 0;
        }
    }

    return -1;
}

const char *Enumerated::toString(Enumerated::Value value)
{
    switch (value) {
      case NEW_YORK: {
        return "NEW_YORK";
      } break;
      case NEW_JERSEY: {
        return "NEW_JERSEY";
      } break;
      case LONDON: {
        return "LONDON";
      } break;
    }

    BSLS_ASSERT(!"invalid enumerator");
    return 0;
}


                     // ----------------------------------
                     // class SequenceWithAnonymityChoice1
                     // ----------------------------------

// CONSTANTS

const char SequenceWithAnonymityChoice1::CLASS_NAME[] = "SequenceWithAnonymityChoice1";

const bdlat_SelectionInfo SequenceWithAnonymityChoice1::SELECTION_INFO_ARRAY[] = {
    {
        SELECTION_ID_SELECTION5,
        "selection5",
        sizeof("selection5") - 1,
        "",
        bdlat_FormattingMode::e_TEXT
    },
    {
        SELECTION_ID_SELECTION6,
        "selection6",
        sizeof("selection6") - 1,
        "",
        bdlat_FormattingMode::e_TEXT
    }
};

// CLASS METHODS

const bdlat_SelectionInfo *SequenceWithAnonymityChoice1::lookupSelectionInfo(
        const char *name,
        int         nameLength)
{
    for (int i = 0; i < 2; ++i) {
        const bdlat_SelectionInfo& selectionInfo =
                    SequenceWithAnonymityChoice1::SELECTION_INFO_ARRAY[i];

        if (nameLength == selectionInfo.d_nameLength
        &&  0 == bsl::memcmp(selectionInfo.d_name_p, name, nameLength))
        {
            return &selectionInfo;
        }
    }

    return 0;
}

const bdlat_SelectionInfo *SequenceWithAnonymityChoice1::lookupSelectionInfo(int id)
{
    switch (id) {
      case SELECTION_ID_SELECTION5:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION5];
      case SELECTION_ID_SELECTION6:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION6];
      default:
        return 0;
    }
}

// CREATORS

SequenceWithAnonymityChoice1::SequenceWithAnonymityChoice1(
    const SequenceWithAnonymityChoice1& original,
    bslma::Allocator *basicAllocator)
: d_selectionId(original.d_selectionId)
, d_allocator_p(bslma::Default::allocator(basicAllocator))
{
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION5: {
        new (d_selection5.buffer())
            bool(original.d_selection5.object());
      } break;
      case SELECTION_ID_SELECTION6: {
        new (d_selection6.buffer())
            bsl::string(
                original.d_selection6.object(), d_allocator_p);
      } break;
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
    }
}

// MANIPULATORS

SequenceWithAnonymityChoice1&
SequenceWithAnonymityChoice1::operator=(const SequenceWithAnonymityChoice1& rhs)
{
    if (this != &rhs) {
        switch (rhs.d_selectionId) {
          case SELECTION_ID_SELECTION5: {
            makeSelection5(rhs.d_selection5.object());
          } break;
          case SELECTION_ID_SELECTION6: {
            makeSelection6(rhs.d_selection6.object());
          } break;
          default:
            BSLS_ASSERT(SELECTION_ID_UNDEFINED == rhs.d_selectionId);
            reset();
        }
    }

    return *this;
}

void SequenceWithAnonymityChoice1::reset()
{
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION5: {
        // no destruction required
      } break;
      case SELECTION_ID_SELECTION6: {
        typedef bsl::string Type;
        d_selection6.object().~Type();
      } break;
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
    }

    d_selectionId = SELECTION_ID_UNDEFINED;
}

int SequenceWithAnonymityChoice1::makeSelection(int selectionId)
{
    switch (selectionId) {
      case SELECTION_ID_SELECTION5: {
        makeSelection5();
      } break;
      case SELECTION_ID_SELECTION6: {
        makeSelection6();
      } break;
      case SELECTION_ID_UNDEFINED: {
        reset();
      } break;
      default:
        return -1;
    }
    return 0;
}

int SequenceWithAnonymityChoice1::makeSelection(const char *name, int nameLength)
{
    const bdlat_SelectionInfo *selectionInfo =
                                         lookupSelectionInfo(name, nameLength);
    if (0 == selectionInfo) {
       return -1;
    }

    return makeSelection(selectionInfo->d_id);
}

bool& SequenceWithAnonymityChoice1::makeSelection5()
{
    if (SELECTION_ID_SELECTION5 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_selection5.object());
    }
    else {
        reset();
        new (d_selection5.buffer())
            bool();
        d_selectionId = SELECTION_ID_SELECTION5;
    }

    return d_selection5.object();
}

bool& SequenceWithAnonymityChoice1::makeSelection5(bool value)
{
    if (SELECTION_ID_SELECTION5 == d_selectionId) {
        d_selection5.object() = value;
    }
    else {
        reset();
        new (d_selection5.buffer())
                bool(value);
        d_selectionId = SELECTION_ID_SELECTION5;
    }

    return d_selection5.object();
}

bsl::string& SequenceWithAnonymityChoice1::makeSelection6()
{
    if (SELECTION_ID_SELECTION6 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_selection6.object());
    }
    else {
        reset();
        new (d_selection6.buffer())
                bsl::string(d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION6;
    }

    return d_selection6.object();
}

bsl::string& SequenceWithAnonymityChoice1::makeSelection6(const bsl::string& value)
{
    if (SELECTION_ID_SELECTION6 == d_selectionId) {
        d_selection6.object() = value;
    }
    else {
        reset();
        new (d_selection6.buffer())
                bsl::string(value, d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION6;
    }

    return d_selection6.object();
}

// ACCESSORS

bsl::ostream& SequenceWithAnonymityChoice1::print(
    bsl::ostream& stream,
    int           level,
    int           spacesPerLevel) const
{
    bslim::Printer printer(&stream, level, spacesPerLevel);
    printer.start();
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION5: {
        printer.printAttribute("selection5", d_selection5.object());
      }  break;
      case SELECTION_ID_SELECTION6: {
        printer.printAttribute("selection6", d_selection6.object());
      }  break;
      default:
        stream << "SELECTION UNDEFINED\n";
    }
    printer.end();
    return stream;
}


const char *SequenceWithAnonymityChoice1::selectionName() const
{
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION5:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION5].name();
      case SELECTION_ID_SELECTION6:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION6].name();
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
        return "(* UNDEFINED *)";
    }
}

                            // -------------------
                            // class SimpleRequest
                            // -------------------

// CONSTANTS

const char SimpleRequest::CLASS_NAME[] = "SimpleRequest";

const bdlat_AttributeInfo SimpleRequest::ATTRIBUTE_INFO_ARRAY[] = {
    {
        ATTRIBUTE_ID_DATA,
        "data",
        sizeof("data") - 1,
        "",
        bdlat_FormattingMode::e_TEXT
    },
    {
        ATTRIBUTE_ID_RESPONSE_LENGTH,
        "responseLength",
        sizeof("responseLength") - 1,
        "",
        bdlat_FormattingMode::e_DEC
    }
};

// CLASS METHODS

const bdlat_AttributeInfo *SimpleRequest::lookupAttributeInfo(
        const char *name,
        int         nameLength)
{
    for (int i = 0; i < 2; ++i) {
        const bdlat_AttributeInfo& attributeInfo =
                    SimpleRequest::ATTRIBUTE_INFO_ARRAY[i];

        if (nameLength == attributeInfo.d_nameLength
        &&  0 == bsl::memcmp(attributeInfo.d_name_p, name, nameLength))
        {
            return &attributeInfo;
        }
    }

    return 0;
}

const bdlat_AttributeInfo *SimpleRequest::lookupAttributeInfo(int id)
{
    switch (id) {
      case ATTRIBUTE_ID_DATA:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_DATA];
      case ATTRIBUTE_ID_RESPONSE_LENGTH:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_RESPONSE_LENGTH];
      default:
        return 0;
    }
}

// CREATORS

SimpleRequest::SimpleRequest(bslma::Allocator *basicAllocator)
: d_data(basicAllocator)
, d_responseLength()
{
}

SimpleRequest::SimpleRequest(const SimpleRequest& original,
                             bslma::Allocator *basicAllocator)
: d_data(original.d_data, basicAllocator)
, d_responseLength(original.d_responseLength)
{
}

SimpleRequest::~SimpleRequest()
{
}

// MANIPULATORS

SimpleRequest&
SimpleRequest::operator=(const SimpleRequest& rhs)
{
    if (this != &rhs) {
        d_data = rhs.d_data;
        d_responseLength = rhs.d_responseLength;
    }

    return *this;
}

void SimpleRequest::reset()
{
    bdlat_ValueTypeFunctions::reset(&d_data);
    bdlat_ValueTypeFunctions::reset(&d_responseLength);
}

// ACCESSORS

bsl::ostream& SimpleRequest::print(
        bsl::ostream& stream,
        int           level,
        int           spacesPerLevel) const
{
    bslim::Printer printer(&stream, level, spacesPerLevel);
    printer.start();
    printer.printAttribute("data", d_data);
    printer.printAttribute("responseLength", d_responseLength);
    printer.end();
    return stream;
}



                           // ----------------------
                           // class UnsignedSequence
                           // ----------------------

// CONSTANTS

const char UnsignedSequence::CLASS_NAME[] = "UnsignedSequence";

const bdlat_AttributeInfo UnsignedSequence::ATTRIBUTE_INFO_ARRAY[] = {
    {
        ATTRIBUTE_ID_ELEMENT1,
        "element1",
        sizeof("element1") - 1,
        "",
        bdlat_FormattingMode::e_DEC
    },
    {
        ATTRIBUTE_ID_ELEMENT2,
        "element2",
        sizeof("element2") - 1,
        "",
        bdlat_FormattingMode::e_DEC
    },
    {
        ATTRIBUTE_ID_ELEMENT3,
        "element3",
        sizeof("element3") - 1,
        "",
        bdlat_FormattingMode::e_DEC
    }
};

// CLASS METHODS

const bdlat_AttributeInfo *UnsignedSequence::lookupAttributeInfo(
        const char *name,
        int         nameLength)
{
    for (int i = 0; i < 3; ++i) {
        const bdlat_AttributeInfo& attributeInfo =
                    UnsignedSequence::ATTRIBUTE_INFO_ARRAY[i];

        if (nameLength == attributeInfo.d_nameLength
        &&  0 == bsl::memcmp(attributeInfo.d_name_p, name, nameLength))
        {
            return &attributeInfo;
        }
    }

    return 0;
}

const bdlat_AttributeInfo *UnsignedSequence::lookupAttributeInfo(int id)
{
    switch (id) {
      case ATTRIBUTE_ID_ELEMENT1:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT1];
      case ATTRIBUTE_ID_ELEMENT2:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT2];
      case ATTRIBUTE_ID_ELEMENT3:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT3];
      default:
        return 0;
    }
}

// CREATORS

UnsignedSequence::UnsignedSequence()
: d_element3()
, d_element1()
, d_element2()
{
}

UnsignedSequence::UnsignedSequence(const UnsignedSequence& original)
: d_element3(original.d_element3)
, d_element1(original.d_element1)
, d_element2(original.d_element2)
{
}

UnsignedSequence::~UnsignedSequence()
{
}

// MANIPULATORS

UnsignedSequence&
UnsignedSequence::operator=(const UnsignedSequence& rhs)
{
    if (this != &rhs) {
        d_element1 = rhs.d_element1;
        d_element2 = rhs.d_element2;
        d_element3 = rhs.d_element3;
    }

    return *this;
}

void UnsignedSequence::reset()
{
    bdlat_ValueTypeFunctions::reset(&d_element1);
    bdlat_ValueTypeFunctions::reset(&d_element2);
    bdlat_ValueTypeFunctions::reset(&d_element3);
}

// ACCESSORS

bsl::ostream& UnsignedSequence::print(
        bsl::ostream& stream,
        int           level,
        int           spacesPerLevel) const
{
    bslim::Printer printer(&stream, level, spacesPerLevel);
    printer.start();
    printer.printAttribute("element1", d_element1);
    printer.printAttribute("element2", d_element2);
    printer.printAttribute("element3", d_element3);
    printer.end();
    return stream;
}



                             // ------------------
                             // class VoidSequence
                             // ------------------

// CONSTANTS

const char VoidSequence::CLASS_NAME[] = "VoidSequence";


// CLASS METHODS

const bdlat_AttributeInfo *VoidSequence::lookupAttributeInfo(
        const char *name,
        int         nameLength)
{
    (void)name;
    (void)nameLength;
    return 0;
}

const bdlat_AttributeInfo *VoidSequence::lookupAttributeInfo(int id)
{
    switch (id) {
      default:
        return 0;
    }
}

// CREATORS

VoidSequence::VoidSequence()
{
}

VoidSequence::VoidSequence(const VoidSequence& original)
{
    (void)original;
}

VoidSequence::~VoidSequence()
{
}

// MANIPULATORS

VoidSequence&
VoidSequence::operator=(const VoidSequence& rhs)
{
    (void)rhs;
    return *this;
}

void VoidSequence::reset()
{
}

// ACCESSORS

bsl::ostream& VoidSequence::print(
        bsl::ostream& stream,
        int           level,
        int           spacesPerLevel) const
{
    (void)level;
    (void)spacesPerLevel;
    return stream;
}



                               // -------------
                               // class Choice5
                               // -------------

// CONSTANTS

const char Choice5::CLASS_NAME[] = "Choice5";

const bdlat_SelectionInfo Choice5::SELECTION_INFO_ARRAY[] = {
    {
        SELECTION_ID_SELECTION1,
        "selection1",
        sizeof("selection1") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        SELECTION_ID_SELECTION2,
        "selection2",
        sizeof("selection2") - 1,
        "",
        bdlat_FormattingMode::e_DEC
    }
};

// CLASS METHODS

const bdlat_SelectionInfo *Choice5::lookupSelectionInfo(
        const char *name,
        int         nameLength)
{
    for (int i = 0; i < 2; ++i) {
        const bdlat_SelectionInfo& selectionInfo =
                    Choice5::SELECTION_INFO_ARRAY[i];

        if (nameLength == selectionInfo.d_nameLength
        &&  0 == bsl::memcmp(selectionInfo.d_name_p, name, nameLength))
        {
            return &selectionInfo;
        }
    }

    return 0;
}

const bdlat_SelectionInfo *Choice5::lookupSelectionInfo(int id)
{
    switch (id) {
      case SELECTION_ID_SELECTION1:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION1];
      case SELECTION_ID_SELECTION2:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION2];
      default:
        return 0;
    }
}

// CREATORS

Choice5::Choice5(
    const Choice5& original,
    bslma::Allocator *basicAllocator)
: d_selectionId(original.d_selectionId)
, d_allocator_p(bslma::Default::allocator(basicAllocator))
{
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION1: {
        new (d_selection1.buffer())
            Choice4(
                original.d_selection1.object(), d_allocator_p);
      } break;
      case SELECTION_ID_SELECTION2: {
        new (d_selection2.buffer())
            int(original.d_selection2.object());
      } break;
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
    }
}

// MANIPULATORS

Choice5&
Choice5::operator=(const Choice5& rhs)
{
    if (this != &rhs) {
        switch (rhs.d_selectionId) {
          case SELECTION_ID_SELECTION1: {
            makeSelection1(rhs.d_selection1.object());
          } break;
          case SELECTION_ID_SELECTION2: {
            makeSelection2(rhs.d_selection2.object());
          } break;
          default:
            BSLS_ASSERT(SELECTION_ID_UNDEFINED == rhs.d_selectionId);
            reset();
        }
    }

    return *this;
}

void Choice5::reset()
{
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION1: {
        d_selection1.object().~Choice4();
      } break;
      case SELECTION_ID_SELECTION2: {
        // no destruction required
      } break;
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
    }

    d_selectionId = SELECTION_ID_UNDEFINED;
}

int Choice5::makeSelection(int selectionId)
{
    switch (selectionId) {
      case SELECTION_ID_SELECTION1: {
        makeSelection1();
      } break;
      case SELECTION_ID_SELECTION2: {
        makeSelection2();
      } break;
      case SELECTION_ID_UNDEFINED: {
        reset();
      } break;
      default:
        return -1;
    }
    return 0;
}

int Choice5::makeSelection(const char *name, int nameLength)
{
    const bdlat_SelectionInfo *selectionInfo =
                                         lookupSelectionInfo(name, nameLength);
    if (0 == selectionInfo) {
       return -1;
    }

    return makeSelection(selectionInfo->d_id);
}

Choice4& Choice5::makeSelection1()
{
    if (SELECTION_ID_SELECTION1 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_selection1.object());
    }
    else {
        reset();
        new (d_selection1.buffer())
                Choice4(d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION1;
    }

    return d_selection1.object();
}

Choice4& Choice5::makeSelection1(const Choice4& value)
{
    if (SELECTION_ID_SELECTION1 == d_selectionId) {
        d_selection1.object() = value;
    }
    else {
        reset();
        new (d_selection1.buffer())
                Choice4(value, d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION1;
    }

    return d_selection1.object();
}

int& Choice5::makeSelection2()
{
    if (SELECTION_ID_SELECTION2 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_selection2.object());
    }
    else {
        reset();
        new (d_selection2.buffer())
            int();
        d_selectionId = SELECTION_ID_SELECTION2;
    }

    return d_selection2.object();
}

int& Choice5::makeSelection2(int value)
{
    if (SELECTION_ID_SELECTION2 == d_selectionId) {
        d_selection2.object() = value;
    }
    else {
        reset();
        new (d_selection2.buffer())
                int(value);
        d_selectionId = SELECTION_ID_SELECTION2;
    }

    return d_selection2.object();
}

// ACCESSORS

bsl::ostream& Choice5::print(
    bsl::ostream& stream,
    int           level,
    int           spacesPerLevel) const
{
    bslim::Printer printer(&stream, level, spacesPerLevel);
    printer.start();
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION1: {
        printer.printAttribute("selection1", d_selection1.object());
      }  break;
      case SELECTION_ID_SELECTION2: {
        printer.printAttribute("selection2", d_selection2.object());
      }  break;
      default:
        stream << "SELECTION UNDEFINED\n";
    }
    printer.end();
    return stream;
}


const char *Choice5::selectionName() const
{
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION1:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION1].name();
      case SELECTION_ID_SELECTION2:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION2].name();
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
        return "(* UNDEFINED *)";
    }
}

                              // ---------------
                              // class Sequence3
                              // ---------------

// CONSTANTS

const char Sequence3::CLASS_NAME[] = "Sequence3";

const bdlat_AttributeInfo Sequence3::ATTRIBUTE_INFO_ARRAY[] = {
    {
        ATTRIBUTE_ID_ELEMENT1,
        "element1",
        sizeof("element1") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        ATTRIBUTE_ID_ELEMENT2,
        "element2",
        sizeof("element2") - 1,
        "",
        bdlat_FormattingMode::e_TEXT
    },
    {
        ATTRIBUTE_ID_ELEMENT3,
        "element3",
        sizeof("element3") - 1,
        "",
        bdlat_FormattingMode::e_TEXT
    },
    {
        ATTRIBUTE_ID_ELEMENT4,
        "element4",
        sizeof("element4") - 1,
        "",
        bdlat_FormattingMode::e_TEXT
    },
    {
        ATTRIBUTE_ID_ELEMENT5,
        "element5",
        sizeof("element5") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        ATTRIBUTE_ID_ELEMENT6,
        "element6",
        sizeof("element6") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
      | bdlat_FormattingMode::e_NILLABLE
    }
};

// CLASS METHODS

const bdlat_AttributeInfo *Sequence3::lookupAttributeInfo(
        const char *name,
        int         nameLength)
{
    for (int i = 0; i < 6; ++i) {
        const bdlat_AttributeInfo& attributeInfo =
                    Sequence3::ATTRIBUTE_INFO_ARRAY[i];

        if (nameLength == attributeInfo.d_nameLength
        &&  0 == bsl::memcmp(attributeInfo.d_name_p, name, nameLength))
        {
            return &attributeInfo;
        }
    }

    return 0;
}

const bdlat_AttributeInfo *Sequence3::lookupAttributeInfo(int id)
{
    switch (id) {
      case ATTRIBUTE_ID_ELEMENT1:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT1];
      case ATTRIBUTE_ID_ELEMENT2:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT2];
      case ATTRIBUTE_ID_ELEMENT3:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT3];
      case ATTRIBUTE_ID_ELEMENT4:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT4];
      case ATTRIBUTE_ID_ELEMENT5:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT5];
      case ATTRIBUTE_ID_ELEMENT6:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT6];
      default:
        return 0;
    }
}

// CREATORS

Sequence3::Sequence3(bslma::Allocator *basicAllocator)
: d_element2(basicAllocator)
, d_element4(basicAllocator)
, d_element5(basicAllocator)
, d_element6(basicAllocator)
, d_element1(basicAllocator)
, d_element3()
{
}

Sequence3::Sequence3(const Sequence3& original,
                     bslma::Allocator *basicAllocator)
: d_element2(original.d_element2, basicAllocator)
, d_element4(original.d_element4, basicAllocator)
, d_element5(original.d_element5, basicAllocator)
, d_element6(original.d_element6, basicAllocator)
, d_element1(original.d_element1, basicAllocator)
, d_element3(original.d_element3)
{
}

Sequence3::~Sequence3()
{
}

// MANIPULATORS

Sequence3&
Sequence3::operator=(const Sequence3& rhs)
{
    if (this != &rhs) {
        d_element1 = rhs.d_element1;
        d_element2 = rhs.d_element2;
        d_element3 = rhs.d_element3;
        d_element4 = rhs.d_element4;
        d_element5 = rhs.d_element5;
        d_element6 = rhs.d_element6;
    }

    return *this;
}

void Sequence3::reset()
{
    bdlat_ValueTypeFunctions::reset(&d_element1);
    bdlat_ValueTypeFunctions::reset(&d_element2);
    bdlat_ValueTypeFunctions::reset(&d_element3);
    bdlat_ValueTypeFunctions::reset(&d_element4);
    bdlat_ValueTypeFunctions::reset(&d_element5);
    bdlat_ValueTypeFunctions::reset(&d_element6);
}

// ACCESSORS

bsl::ostream& Sequence3::print(
        bsl::ostream& stream,
        int           level,
        int           spacesPerLevel) const
{
    bslim::Printer printer(&stream, level, spacesPerLevel);
    printer.start();
    printer.printAttribute("element1", d_element1);
    printer.printAttribute("element2", d_element2);
    printer.printAttribute("element3", d_element3);
    printer.printAttribute("element4", d_element4);
    printer.printAttribute("element5", d_element5);
    printer.printAttribute("element6", d_element6);
    printer.end();
    return stream;
}



                              // ---------------
                              // class Sequence5
                              // ---------------

// CONSTANTS

const char Sequence5::CLASS_NAME[] = "Sequence5";

const bdlat_AttributeInfo Sequence5::ATTRIBUTE_INFO_ARRAY[] = {
    {
        ATTRIBUTE_ID_ELEMENT1,
        "element1",
        sizeof("element1") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        ATTRIBUTE_ID_ELEMENT2,
        "element2",
        sizeof("element2") - 1,
        "",
        bdlat_FormattingMode::e_TEXT
      | bdlat_FormattingMode::e_NILLABLE
    },
    {
        ATTRIBUTE_ID_ELEMENT3,
        "element3",
        sizeof("element3") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
      | bdlat_FormattingMode::e_NILLABLE
    },
    {
        ATTRIBUTE_ID_ELEMENT4,
        "element4",
        sizeof("element4") - 1,
        "",
        bdlat_FormattingMode::e_HEX
      | bdlat_FormattingMode::e_NILLABLE
    },
    {
        ATTRIBUTE_ID_ELEMENT5,
        "element5",
        sizeof("element5") - 1,
        "",
        bdlat_FormattingMode::e_DEC
      | bdlat_FormattingMode::e_NILLABLE
    },
    {
        ATTRIBUTE_ID_ELEMENT6,
        "element6",
        sizeof("element6") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
      | bdlat_FormattingMode::e_NILLABLE
    },
    {
        ATTRIBUTE_ID_ELEMENT7,
        "element7",
        sizeof("element7") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
      | bdlat_FormattingMode::e_NILLABLE
    }
};

// CLASS METHODS

const bdlat_AttributeInfo *Sequence5::lookupAttributeInfo(
        const char *name,
        int         nameLength)
{
    for (int i = 0; i < 7; ++i) {
        const bdlat_AttributeInfo& attributeInfo =
                    Sequence5::ATTRIBUTE_INFO_ARRAY[i];

        if (nameLength == attributeInfo.d_nameLength
        &&  0 == bsl::memcmp(attributeInfo.d_name_p, name, nameLength))
        {
            return &attributeInfo;
        }
    }

    return 0;
}

const bdlat_AttributeInfo *Sequence5::lookupAttributeInfo(int id)
{
    switch (id) {
      case ATTRIBUTE_ID_ELEMENT1:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT1];
      case ATTRIBUTE_ID_ELEMENT2:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT2];
      case ATTRIBUTE_ID_ELEMENT3:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT3];
      case ATTRIBUTE_ID_ELEMENT4:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT4];
      case ATTRIBUTE_ID_ELEMENT5:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT5];
      case ATTRIBUTE_ID_ELEMENT6:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT6];
      case ATTRIBUTE_ID_ELEMENT7:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT7];
      default:
        return 0;
    }
}

// CREATORS

Sequence5::Sequence5(bslma::Allocator *basicAllocator)
: d_allocator_p(bslma::Default::allocator(basicAllocator))
, d_element5(basicAllocator)
, d_element3(basicAllocator)
, d_element4(basicAllocator)
, d_element2(basicAllocator)
, d_element6(basicAllocator)
, d_element7(basicAllocator)
{
    d_element1 = new (*d_allocator_p)
            Sequence3(d_allocator_p);
}

Sequence5::Sequence5(const Sequence5& original,
                     bslma::Allocator *basicAllocator)
: d_allocator_p(bslma::Default::allocator(basicAllocator))
, d_element5(original.d_element5, basicAllocator)
, d_element3(original.d_element3, basicAllocator)
, d_element4(original.d_element4, basicAllocator)
, d_element2(original.d_element2, basicAllocator)
, d_element6(original.d_element6, basicAllocator)
, d_element7(original.d_element7, basicAllocator)
{
    d_element1 = new (*d_allocator_p)
            Sequence3(*original.d_element1, d_allocator_p);
}

Sequence5::~Sequence5()
{
    d_allocator_p->deleteObject(d_element1);
}

// MANIPULATORS

Sequence5&
Sequence5::operator=(const Sequence5& rhs)
{
    if (this != &rhs) {
        *d_element1 = *rhs.d_element1;
        d_element2 = rhs.d_element2;
        d_element3 = rhs.d_element3;
        d_element4 = rhs.d_element4;
        d_element5 = rhs.d_element5;
        d_element6 = rhs.d_element6;
        d_element7 = rhs.d_element7;
    }

    return *this;
}

void Sequence5::reset()
{
    bdlat_ValueTypeFunctions::reset(d_element1);
    bdlat_ValueTypeFunctions::reset(&d_element2);
    bdlat_ValueTypeFunctions::reset(&d_element3);
    bdlat_ValueTypeFunctions::reset(&d_element4);
    bdlat_ValueTypeFunctions::reset(&d_element5);
    bdlat_ValueTypeFunctions::reset(&d_element6);
    bdlat_ValueTypeFunctions::reset(&d_element7);
}

// ACCESSORS

bsl::ostream& Sequence5::print(
        bsl::ostream& stream,
        int           level,
        int           spacesPerLevel) const
{
    bslim::Printer printer(&stream, level, spacesPerLevel);
    printer.start();
    printer.printAttribute("element1", *d_element1);
    printer.printAttribute("element2", d_element2);
    printer.printAttribute("element3", d_element3);
    printer.printAttribute("element4", d_element4);
    printer.printAttribute("element5", d_element5);
    printer.printAttribute("element6", d_element6);
    printer.printAttribute("element7", d_element7);
    printer.end();
    return stream;
}



                              // ---------------
                              // class Sequence6
                              // ---------------

// CONSTANTS

const char Sequence6::CLASS_NAME[] = "Sequence6";

const bdlat_AttributeInfo Sequence6::ATTRIBUTE_INFO_ARRAY[] = {
    {
        ATTRIBUTE_ID_ELEMENT1,
        "element1",
        sizeof("element1") - 1,
        "",
        bdlat_FormattingMode::e_DEC
    },
    {
        ATTRIBUTE_ID_ELEMENT2,
        "element2",
        sizeof("element2") - 1,
        "",
        bdlat_FormattingMode::e_TEXT
    },
    {
        ATTRIBUTE_ID_ELEMENT3,
        "element3",
        sizeof("element3") - 1,
        "",
        bdlat_FormattingMode::e_DEC
    },
    {
        ATTRIBUTE_ID_ELEMENT4,
        "element4",
        sizeof("element4") - 1,
        "",
        bdlat_FormattingMode::e_DEC
    },
    {
        ATTRIBUTE_ID_ELEMENT5,
        "element5",
        sizeof("element5") - 1,
        "",
        bdlat_FormattingMode::e_DEC
    },
    {
        ATTRIBUTE_ID_ELEMENT6,
        "element6",
        sizeof("element6") - 1,
        "",
        bdlat_FormattingMode::e_DEC
      | bdlat_FormattingMode::e_NILLABLE
    },
    {
        ATTRIBUTE_ID_ELEMENT7,
        "element7",
        sizeof("element7") - 1,
        "",
        bdlat_FormattingMode::e_TEXT
    },
    {
        ATTRIBUTE_ID_ELEMENT8,
        "element8",
        sizeof("element8") - 1,
        "",
        bdlat_FormattingMode::e_DEC
    },
    {
        ATTRIBUTE_ID_ELEMENT9,
        "element9",
        sizeof("element9") - 1,
        "",
        bdlat_FormattingMode::e_DEC
    },
    {
        ATTRIBUTE_ID_ELEMENT10,
        "element10",
        sizeof("element10") - 1,
        "",
        bdlat_FormattingMode::e_DEC
    },
    {
        ATTRIBUTE_ID_ELEMENT11,
        "element11",
        sizeof("element11") - 1,
        "",
        bdlat_FormattingMode::e_TEXT
    },
    {
        ATTRIBUTE_ID_ELEMENT12,
        "element12",
        sizeof("element12") - 1,
        "",
        bdlat_FormattingMode::e_DEC
    },
    {
        ATTRIBUTE_ID_ELEMENT13,
        "element13",
        sizeof("element13") - 1,
        "",
        bdlat_FormattingMode::e_DEC
      | bdlat_FormattingMode::e_NILLABLE
    },
    {
        ATTRIBUTE_ID_ELEMENT14,
        "element14",
        sizeof("element14") - 1,
        "",
        bdlat_FormattingMode::e_DEC
    },
    {
        ATTRIBUTE_ID_ELEMENT15,
        "element15",
        sizeof("element15") - 1,
        "",
        bdlat_FormattingMode::e_DEC
      | bdlat_FormattingMode::e_NILLABLE
    }
};

// CLASS METHODS

const bdlat_AttributeInfo *Sequence6::lookupAttributeInfo(
        const char *name,
        int         nameLength)
{
    for (int i = 0; i < 15; ++i) {
        const bdlat_AttributeInfo& attributeInfo =
                    Sequence6::ATTRIBUTE_INFO_ARRAY[i];

        if (nameLength == attributeInfo.d_nameLength
        &&  0 == bsl::memcmp(attributeInfo.d_name_p, name, nameLength))
        {
            return &attributeInfo;
        }
    }

    return 0;
}

const bdlat_AttributeInfo *Sequence6::lookupAttributeInfo(int id)
{
    switch (id) {
      case ATTRIBUTE_ID_ELEMENT1:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT1];
      case ATTRIBUTE_ID_ELEMENT2:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT2];
      case ATTRIBUTE_ID_ELEMENT3:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT3];
      case ATTRIBUTE_ID_ELEMENT4:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT4];
      case ATTRIBUTE_ID_ELEMENT5:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT5];
      case ATTRIBUTE_ID_ELEMENT6:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT6];
      case ATTRIBUTE_ID_ELEMENT7:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT7];
      case ATTRIBUTE_ID_ELEMENT8:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT8];
      case ATTRIBUTE_ID_ELEMENT9:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT9];
      case ATTRIBUTE_ID_ELEMENT10:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT10];
      case ATTRIBUTE_ID_ELEMENT11:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT11];
      case ATTRIBUTE_ID_ELEMENT12:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT12];
      case ATTRIBUTE_ID_ELEMENT13:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT13];
      case ATTRIBUTE_ID_ELEMENT14:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT14];
      case ATTRIBUTE_ID_ELEMENT15:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT15];
      default:
        return 0;
    }
}

// CREATORS

Sequence6::Sequence6(bslma::Allocator *basicAllocator)
: d_element12(basicAllocator)
, d_element10(basicAllocator)
, d_element15(basicAllocator)
, d_element13(basicAllocator)
, d_element11(basicAllocator)
, d_element2(basicAllocator)
, d_element7(basicAllocator)
, d_element4()
, d_element6(basicAllocator)
, d_element14(basicAllocator)
, d_element9()
, d_element3()
, d_element8()
, d_element5()
, d_element1()
{
}

Sequence6::Sequence6(const Sequence6& original,
                     bslma::Allocator *basicAllocator)
: d_element12(original.d_element12, basicAllocator)
, d_element10(original.d_element10, basicAllocator)
, d_element15(original.d_element15, basicAllocator)
, d_element13(original.d_element13, basicAllocator)
, d_element11(original.d_element11, basicAllocator)
, d_element2(original.d_element2, basicAllocator)
, d_element7(original.d_element7, basicAllocator)
, d_element4(original.d_element4)
, d_element6(original.d_element6, basicAllocator)
, d_element14(original.d_element14, basicAllocator)
, d_element9(original.d_element9)
, d_element3(original.d_element3)
, d_element8(original.d_element8)
, d_element5(original.d_element5)
, d_element1(original.d_element1)
{
}

Sequence6::~Sequence6()
{
}

// MANIPULATORS

Sequence6&
Sequence6::operator=(const Sequence6& rhs)
{
    if (this != &rhs) {
        d_element1 = rhs.d_element1;
        d_element2 = rhs.d_element2;
        d_element3 = rhs.d_element3;
        d_element4 = rhs.d_element4;
        d_element5 = rhs.d_element5;
        d_element6 = rhs.d_element6;
        d_element7 = rhs.d_element7;
        d_element8 = rhs.d_element8;
        d_element9 = rhs.d_element9;
        d_element10 = rhs.d_element10;
        d_element11 = rhs.d_element11;
        d_element12 = rhs.d_element12;
        d_element13 = rhs.d_element13;
        d_element14 = rhs.d_element14;
        d_element15 = rhs.d_element15;
    }

    return *this;
}

void Sequence6::reset()
{
    bdlat_ValueTypeFunctions::reset(&d_element1);
    bdlat_ValueTypeFunctions::reset(&d_element2);
    bdlat_ValueTypeFunctions::reset(&d_element3);
    bdlat_ValueTypeFunctions::reset(&d_element4);
    bdlat_ValueTypeFunctions::reset(&d_element5);
    bdlat_ValueTypeFunctions::reset(&d_element6);
    bdlat_ValueTypeFunctions::reset(&d_element7);
    bdlat_ValueTypeFunctions::reset(&d_element8);
    bdlat_ValueTypeFunctions::reset(&d_element9);
    bdlat_ValueTypeFunctions::reset(&d_element10);
    bdlat_ValueTypeFunctions::reset(&d_element11);
    bdlat_ValueTypeFunctions::reset(&d_element12);
    bdlat_ValueTypeFunctions::reset(&d_element13);
    bdlat_ValueTypeFunctions::reset(&d_element14);
    bdlat_ValueTypeFunctions::reset(&d_element15);
}

// ACCESSORS

bsl::ostream& Sequence6::print(
        bsl::ostream& stream,
        int           level,
        int           spacesPerLevel) const
{
    bslim::Printer printer(&stream, level, spacesPerLevel);
    printer.start();
    printer.printAttribute("element1", d_element1);
    printer.printAttribute("element2", d_element2);
    printer.printAttribute("element3", d_element3);
    printer.printAttribute("element4", d_element4);
    printer.printAttribute("element5", (int)d_element5);
    printer.printAttribute("element6", d_element6);
    printer.printAttribute("element7", d_element7);
    printer.printAttribute("element8", d_element8);
    printer.printAttribute("element9", d_element9);
    printer.printAttribute("element10", d_element10);
    printer.printAttribute("element11", d_element11);
    printer.printAttribute("element12", d_element12);
    printer.printAttribute("element13", d_element13);
    printer.printAttribute("element14", d_element14);
    printer.printAttribute("element15", d_element15);
    printer.end();
    return stream;
}



                               // -------------
                               // class Choice3
                               // -------------

// CONSTANTS

const char Choice3::CLASS_NAME[] = "Choice3";

const bdlat_SelectionInfo Choice3::SELECTION_INFO_ARRAY[] = {
    {
        SELECTION_ID_SELECTION1,
        "selection1",
        sizeof("selection1") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        SELECTION_ID_SELECTION2,
        "selection2",
        sizeof("selection2") - 1,
        "",
        bdlat_FormattingMode::e_DEC
    },
    {
        SELECTION_ID_SELECTION3,
        "selection3",
        sizeof("selection3") - 1,
        "",
        bdlat_FormattingMode::e_TEXT
    },
    {
        SELECTION_ID_SELECTION4,
        "selection4",
        sizeof("selection4") - 1,
        "",
        bdlat_FormattingMode::e_DEC
    }
};

// CLASS METHODS

const bdlat_SelectionInfo *Choice3::lookupSelectionInfo(
        const char *name,
        int         nameLength)
{
    for (int i = 0; i < 4; ++i) {
        const bdlat_SelectionInfo& selectionInfo =
                    Choice3::SELECTION_INFO_ARRAY[i];

        if (nameLength == selectionInfo.d_nameLength
        &&  0 == bsl::memcmp(selectionInfo.d_name_p, name, nameLength))
        {
            return &selectionInfo;
        }
    }

    return 0;
}

const bdlat_SelectionInfo *Choice3::lookupSelectionInfo(int id)
{
    switch (id) {
      case SELECTION_ID_SELECTION1:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION1];
      case SELECTION_ID_SELECTION2:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION2];
      case SELECTION_ID_SELECTION3:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION3];
      case SELECTION_ID_SELECTION4:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION4];
      default:
        return 0;
    }
}

// CREATORS

Choice3::Choice3(
    const Choice3& original,
    bslma::Allocator *basicAllocator)
: d_selectionId(original.d_selectionId)
, d_allocator_p(bslma::Default::allocator(basicAllocator))
{
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION1: {
        new (d_selection1.buffer())
            Sequence6(
                original.d_selection1.object(), d_allocator_p);
      } break;
      case SELECTION_ID_SELECTION2: {
        new (d_selection2.buffer())
            unsigned char(original.d_selection2.object());
      } break;
      case SELECTION_ID_SELECTION3: {
        new (d_selection3.buffer())
            CustomString(
                original.d_selection3.object(), d_allocator_p);
      } break;
      case SELECTION_ID_SELECTION4: {
        new (d_selection4.buffer())
            CustomInt(original.d_selection4.object());
      } break;
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
    }
}

// MANIPULATORS

Choice3&
Choice3::operator=(const Choice3& rhs)
{
    if (this != &rhs) {
        switch (rhs.d_selectionId) {
          case SELECTION_ID_SELECTION1: {
            makeSelection1(rhs.d_selection1.object());
          } break;
          case SELECTION_ID_SELECTION2: {
            makeSelection2(rhs.d_selection2.object());
          } break;
          case SELECTION_ID_SELECTION3: {
            makeSelection3(rhs.d_selection3.object());
          } break;
          case SELECTION_ID_SELECTION4: {
            makeSelection4(rhs.d_selection4.object());
          } break;
          default:
            BSLS_ASSERT(SELECTION_ID_UNDEFINED == rhs.d_selectionId);
            reset();
        }
    }

    return *this;
}

void Choice3::reset()
{
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION1: {
        d_selection1.object().~Sequence6();
      } break;
      case SELECTION_ID_SELECTION2: {
        // no destruction required
      } break;
      case SELECTION_ID_SELECTION3: {
        d_selection3.object().~CustomString();
      } break;
      case SELECTION_ID_SELECTION4: {
        d_selection4.object().~CustomInt();
      } break;
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
    }

    d_selectionId = SELECTION_ID_UNDEFINED;
}

int Choice3::makeSelection(int selectionId)
{
    switch (selectionId) {
      case SELECTION_ID_SELECTION1: {
        makeSelection1();
      } break;
      case SELECTION_ID_SELECTION2: {
        makeSelection2();
      } break;
      case SELECTION_ID_SELECTION3: {
        makeSelection3();
      } break;
      case SELECTION_ID_SELECTION4: {
        makeSelection4();
      } break;
      case SELECTION_ID_UNDEFINED: {
        reset();
      } break;
      default:
        return -1;
    }
    return 0;
}

int Choice3::makeSelection(const char *name, int nameLength)
{
    const bdlat_SelectionInfo *selectionInfo =
                                         lookupSelectionInfo(name, nameLength);
    if (0 == selectionInfo) {
       return -1;
    }

    return makeSelection(selectionInfo->d_id);
}

Sequence6& Choice3::makeSelection1()
{
    if (SELECTION_ID_SELECTION1 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_selection1.object());
    }
    else {
        reset();
        new (d_selection1.buffer())
                Sequence6(d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION1;
    }

    return d_selection1.object();
}

Sequence6& Choice3::makeSelection1(const Sequence6& value)
{
    if (SELECTION_ID_SELECTION1 == d_selectionId) {
        d_selection1.object() = value;
    }
    else {
        reset();
        new (d_selection1.buffer())
                Sequence6(value, d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION1;
    }

    return d_selection1.object();
}

unsigned char& Choice3::makeSelection2()
{
    if (SELECTION_ID_SELECTION2 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_selection2.object());
    }
    else {
        reset();
        new (d_selection2.buffer())
            unsigned char();
        d_selectionId = SELECTION_ID_SELECTION2;
    }

    return d_selection2.object();
}

unsigned char& Choice3::makeSelection2(unsigned char value)
{
    if (SELECTION_ID_SELECTION2 == d_selectionId) {
        d_selection2.object() = value;
    }
    else {
        reset();
        new (d_selection2.buffer())
                unsigned char(value);
        d_selectionId = SELECTION_ID_SELECTION2;
    }

    return d_selection2.object();
}

CustomString& Choice3::makeSelection3()
{
    if (SELECTION_ID_SELECTION3 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_selection3.object());
    }
    else {
        reset();
        new (d_selection3.buffer())
                CustomString(d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION3;
    }

    return d_selection3.object();
}

CustomString& Choice3::makeSelection3(const CustomString& value)
{
    if (SELECTION_ID_SELECTION3 == d_selectionId) {
        d_selection3.object() = value;
    }
    else {
        reset();
        new (d_selection3.buffer())
                CustomString(value, d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION3;
    }

    return d_selection3.object();
}

CustomInt& Choice3::makeSelection4()
{
    if (SELECTION_ID_SELECTION4 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_selection4.object());
    }
    else {
        reset();
        new (d_selection4.buffer())
            CustomInt();
        d_selectionId = SELECTION_ID_SELECTION4;
    }

    return d_selection4.object();
}

CustomInt& Choice3::makeSelection4(const CustomInt& value)
{
    if (SELECTION_ID_SELECTION4 == d_selectionId) {
        d_selection4.object() = value;
    }
    else {
        reset();
        new (d_selection4.buffer())
                CustomInt(value);
        d_selectionId = SELECTION_ID_SELECTION4;
    }

    return d_selection4.object();
}

// ACCESSORS

bsl::ostream& Choice3::print(
    bsl::ostream& stream,
    int           level,
    int           spacesPerLevel) const
{
    bslim::Printer printer(&stream, level, spacesPerLevel);
    printer.start();
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION1: {
        printer.printAttribute("selection1", d_selection1.object());
      }  break;
      case SELECTION_ID_SELECTION2: {
        printer.printAttribute("selection2", (int)d_selection2.object());
      }  break;
      case SELECTION_ID_SELECTION3: {
        printer.printAttribute("selection3", d_selection3.object());
      }  break;
      case SELECTION_ID_SELECTION4: {
        printer.printAttribute("selection4", d_selection4.object());
      }  break;
      default:
        stream << "SELECTION UNDEFINED\n";
    }
    printer.end();
    return stream;
}


const char *Choice3::selectionName() const
{
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION1:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION1].name();
      case SELECTION_ID_SELECTION2:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION2].name();
      case SELECTION_ID_SELECTION3:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION3].name();
      case SELECTION_ID_SELECTION4:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION4].name();
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
        return "(* UNDEFINED *)";
    }
}

                     // ---------------------------------
                     // class SequenceWithAnonymityChoice
                     // ---------------------------------

// CONSTANTS

const char SequenceWithAnonymityChoice::CLASS_NAME[] = "SequenceWithAnonymityChoice";

const bdlat_SelectionInfo SequenceWithAnonymityChoice::SELECTION_INFO_ARRAY[] = {
    {
        SELECTION_ID_SELECTION1,
        "selection1",
        sizeof("selection1") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        SELECTION_ID_SELECTION2,
        "selection2",
        sizeof("selection2") - 1,
        "",
        bdlat_FormattingMode::e_DEC
    },
    {
        SELECTION_ID_SELECTION3,
        "selection3",
        sizeof("selection3") - 1,
        "",
        bdlat_FormattingMode::e_TEXT
    },
    {
        SELECTION_ID_SELECTION4,
        "selection4",
        sizeof("selection4") - 1,
        "",
        bdlat_FormattingMode::e_DEC
    }
};

// CLASS METHODS

const bdlat_SelectionInfo *SequenceWithAnonymityChoice::lookupSelectionInfo(
        const char *name,
        int         nameLength)
{
    for (int i = 0; i < 4; ++i) {
        const bdlat_SelectionInfo& selectionInfo =
                    SequenceWithAnonymityChoice::SELECTION_INFO_ARRAY[i];

        if (nameLength == selectionInfo.d_nameLength
        &&  0 == bsl::memcmp(selectionInfo.d_name_p, name, nameLength))
        {
            return &selectionInfo;
        }
    }

    return 0;
}

const bdlat_SelectionInfo *SequenceWithAnonymityChoice::lookupSelectionInfo(int id)
{
    switch (id) {
      case SELECTION_ID_SELECTION1:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION1];
      case SELECTION_ID_SELECTION2:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION2];
      case SELECTION_ID_SELECTION3:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION3];
      case SELECTION_ID_SELECTION4:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION4];
      default:
        return 0;
    }
}

// CREATORS

SequenceWithAnonymityChoice::SequenceWithAnonymityChoice(
    const SequenceWithAnonymityChoice& original,
    bslma::Allocator *basicAllocator)
: d_selectionId(original.d_selectionId)
, d_allocator_p(bslma::Default::allocator(basicAllocator))
{
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION1: {
        new (d_selection1.buffer())
            Sequence6(
                original.d_selection1.object(), d_allocator_p);
      } break;
      case SELECTION_ID_SELECTION2: {
        new (d_selection2.buffer())
            unsigned char(original.d_selection2.object());
      } break;
      case SELECTION_ID_SELECTION3: {
        new (d_selection3.buffer())
            CustomString(
                original.d_selection3.object(), d_allocator_p);
      } break;
      case SELECTION_ID_SELECTION4: {
        new (d_selection4.buffer())
            CustomInt(original.d_selection4.object());
      } break;
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
    }
}

// MANIPULATORS

SequenceWithAnonymityChoice&
SequenceWithAnonymityChoice::operator=(const SequenceWithAnonymityChoice& rhs)
{
    if (this != &rhs) {
        switch (rhs.d_selectionId) {
          case SELECTION_ID_SELECTION1: {
            makeSelection1(rhs.d_selection1.object());
          } break;
          case SELECTION_ID_SELECTION2: {
            makeSelection2(rhs.d_selection2.object());
          } break;
          case SELECTION_ID_SELECTION3: {
            makeSelection3(rhs.d_selection3.object());
          } break;
          case SELECTION_ID_SELECTION4: {
            makeSelection4(rhs.d_selection4.object());
          } break;
          default:
            BSLS_ASSERT(SELECTION_ID_UNDEFINED == rhs.d_selectionId);
            reset();
        }
    }

    return *this;
}

void SequenceWithAnonymityChoice::reset()
{
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION1: {
        d_selection1.object().~Sequence6();
      } break;
      case SELECTION_ID_SELECTION2: {
        // no destruction required
      } break;
      case SELECTION_ID_SELECTION3: {
        d_selection3.object().~CustomString();
      } break;
      case SELECTION_ID_SELECTION4: {
        d_selection4.object().~CustomInt();
      } break;
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
    }

    d_selectionId = SELECTION_ID_UNDEFINED;
}

int SequenceWithAnonymityChoice::makeSelection(int selectionId)
{
    switch (selectionId) {
      case SELECTION_ID_SELECTION1: {
        makeSelection1();
      } break;
      case SELECTION_ID_SELECTION2: {
        makeSelection2();
      } break;
      case SELECTION_ID_SELECTION3: {
        makeSelection3();
      } break;
      case SELECTION_ID_SELECTION4: {
        makeSelection4();
      } break;
      case SELECTION_ID_UNDEFINED: {
        reset();
      } break;
      default:
        return -1;
    }
    return 0;
}

int SequenceWithAnonymityChoice::makeSelection(const char *name, int nameLength)
{
    const bdlat_SelectionInfo *selectionInfo =
                                         lookupSelectionInfo(name, nameLength);
    if (0 == selectionInfo) {
       return -1;
    }

    return makeSelection(selectionInfo->d_id);
}

Sequence6& SequenceWithAnonymityChoice::makeSelection1()
{
    if (SELECTION_ID_SELECTION1 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_selection1.object());
    }
    else {
        reset();
        new (d_selection1.buffer())
                Sequence6(d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION1;
    }

    return d_selection1.object();
}

Sequence6& SequenceWithAnonymityChoice::makeSelection1(const Sequence6& value)
{
    if (SELECTION_ID_SELECTION1 == d_selectionId) {
        d_selection1.object() = value;
    }
    else {
        reset();
        new (d_selection1.buffer())
                Sequence6(value, d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION1;
    }

    return d_selection1.object();
}

unsigned char& SequenceWithAnonymityChoice::makeSelection2()
{
    if (SELECTION_ID_SELECTION2 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_selection2.object());
    }
    else {
        reset();
        new (d_selection2.buffer())
            unsigned char();
        d_selectionId = SELECTION_ID_SELECTION2;
    }

    return d_selection2.object();
}

unsigned char& SequenceWithAnonymityChoice::makeSelection2(unsigned char value)
{
    if (SELECTION_ID_SELECTION2 == d_selectionId) {
        d_selection2.object() = value;
    }
    else {
        reset();
        new (d_selection2.buffer())
                unsigned char(value);
        d_selectionId = SELECTION_ID_SELECTION2;
    }

    return d_selection2.object();
}

CustomString& SequenceWithAnonymityChoice::makeSelection3()
{
    if (SELECTION_ID_SELECTION3 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_selection3.object());
    }
    else {
        reset();
        new (d_selection3.buffer())
                CustomString(d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION3;
    }

    return d_selection3.object();
}

CustomString& SequenceWithAnonymityChoice::makeSelection3(const CustomString& value)
{
    if (SELECTION_ID_SELECTION3 == d_selectionId) {
        d_selection3.object() = value;
    }
    else {
        reset();
        new (d_selection3.buffer())
                CustomString(value, d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION3;
    }

    return d_selection3.object();
}

CustomInt& SequenceWithAnonymityChoice::makeSelection4()
{
    if (SELECTION_ID_SELECTION4 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_selection4.object());
    }
    else {
        reset();
        new (d_selection4.buffer())
            CustomInt();
        d_selectionId = SELECTION_ID_SELECTION4;
    }

    return d_selection4.object();
}

CustomInt& SequenceWithAnonymityChoice::makeSelection4(const CustomInt& value)
{
    if (SELECTION_ID_SELECTION4 == d_selectionId) {
        d_selection4.object() = value;
    }
    else {
        reset();
        new (d_selection4.buffer())
                CustomInt(value);
        d_selectionId = SELECTION_ID_SELECTION4;
    }

    return d_selection4.object();
}

// ACCESSORS

bsl::ostream& SequenceWithAnonymityChoice::print(
    bsl::ostream& stream,
    int           level,
    int           spacesPerLevel) const
{
    bslim::Printer printer(&stream, level, spacesPerLevel);
    printer.start();
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION1: {
        printer.printAttribute("selection1", d_selection1.object());
      }  break;
      case SELECTION_ID_SELECTION2: {
        printer.printAttribute("selection2", (int)d_selection2.object());
      }  break;
      case SELECTION_ID_SELECTION3: {
        printer.printAttribute("selection3", d_selection3.object());
      }  break;
      case SELECTION_ID_SELECTION4: {
        printer.printAttribute("selection4", d_selection4.object());
      }  break;
      default:
        stream << "SELECTION UNDEFINED\n";
    }
    printer.end();
    return stream;
}


const char *SequenceWithAnonymityChoice::selectionName() const
{
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION1:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION1].name();
      case SELECTION_ID_SELECTION2:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION2].name();
      case SELECTION_ID_SELECTION3:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION3].name();
      case SELECTION_ID_SELECTION4:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION4].name();
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
        return "(* UNDEFINED *)";
    }
}

                               // -------------
                               // class Choice1
                               // -------------

// CONSTANTS

const char Choice1::CLASS_NAME[] = "Choice1";

const bdlat_SelectionInfo Choice1::SELECTION_INFO_ARRAY[] = {
    {
        SELECTION_ID_SELECTION1,
        "selection1",
        sizeof("selection1") - 1,
        "",
        bdlat_FormattingMode::e_DEC
    },
    {
        SELECTION_ID_SELECTION2,
        "selection2",
        sizeof("selection2") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        SELECTION_ID_SELECTION3,
        "selection3",
        sizeof("selection3") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        SELECTION_ID_SELECTION4,
        "selection4",
        sizeof("selection4") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    }
};

// CLASS METHODS

const bdlat_SelectionInfo *Choice1::lookupSelectionInfo(
        const char *name,
        int         nameLength)
{
    for (int i = 0; i < 4; ++i) {
        const bdlat_SelectionInfo& selectionInfo =
                    Choice1::SELECTION_INFO_ARRAY[i];

        if (nameLength == selectionInfo.d_nameLength
        &&  0 == bsl::memcmp(selectionInfo.d_name_p, name, nameLength))
        {
            return &selectionInfo;
        }
    }

    return 0;
}

const bdlat_SelectionInfo *Choice1::lookupSelectionInfo(int id)
{
    switch (id) {
      case SELECTION_ID_SELECTION1:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION1];
      case SELECTION_ID_SELECTION2:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION2];
      case SELECTION_ID_SELECTION3:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION3];
      case SELECTION_ID_SELECTION4:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION4];
      default:
        return 0;
    }
}

// CREATORS

Choice1::Choice1(
    const Choice1& original,
    bslma::Allocator *basicAllocator)
: d_selectionId(original.d_selectionId)
, d_allocator_p(bslma::Default::allocator(basicAllocator))
{
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION1: {
        new (d_selection1.buffer())
            int(original.d_selection1.object());
      } break;
      case SELECTION_ID_SELECTION2: {
        new (d_selection2.buffer())
            double(original.d_selection2.object());
      } break;
      case SELECTION_ID_SELECTION3: {
        d_selection3 = new (*d_allocator_p)
                Sequence4(*original.d_selection3, d_allocator_p);
      } break;
      case SELECTION_ID_SELECTION4: {
        d_selection4 = new (*d_allocator_p)
                Choice2(*original.d_selection4, d_allocator_p);
      } break;
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
    }
}

// MANIPULATORS

Choice1&
Choice1::operator=(const Choice1& rhs)
{
    if (this != &rhs) {
        switch (rhs.d_selectionId) {
          case SELECTION_ID_SELECTION1: {
            makeSelection1(rhs.d_selection1.object());
          } break;
          case SELECTION_ID_SELECTION2: {
            makeSelection2(rhs.d_selection2.object());
          } break;
          case SELECTION_ID_SELECTION3: {
            makeSelection3(*rhs.d_selection3);
          } break;
          case SELECTION_ID_SELECTION4: {
            makeSelection4(*rhs.d_selection4);
          } break;
          default:
            BSLS_ASSERT(SELECTION_ID_UNDEFINED == rhs.d_selectionId);
            reset();
        }
    }

    return *this;
}

void Choice1::reset()
{
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION1: {
        // no destruction required
      } break;
      case SELECTION_ID_SELECTION2: {
        // no destruction required
      } break;
      case SELECTION_ID_SELECTION3: {
        d_allocator_p->deleteObject(d_selection3);
      } break;
      case SELECTION_ID_SELECTION4: {
        d_allocator_p->deleteObject(d_selection4);
      } break;
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
    }

    d_selectionId = SELECTION_ID_UNDEFINED;
}

int Choice1::makeSelection(int selectionId)
{
    switch (selectionId) {
      case SELECTION_ID_SELECTION1: {
        makeSelection1();
      } break;
      case SELECTION_ID_SELECTION2: {
        makeSelection2();
      } break;
      case SELECTION_ID_SELECTION3: {
        makeSelection3();
      } break;
      case SELECTION_ID_SELECTION4: {
        makeSelection4();
      } break;
      case SELECTION_ID_UNDEFINED: {
        reset();
      } break;
      default:
        return -1;
    }
    return 0;
}

int Choice1::makeSelection(const char *name, int nameLength)
{
    const bdlat_SelectionInfo *selectionInfo =
                                         lookupSelectionInfo(name, nameLength);
    if (0 == selectionInfo) {
       return -1;
    }

    return makeSelection(selectionInfo->d_id);
}

int& Choice1::makeSelection1()
{
    if (SELECTION_ID_SELECTION1 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_selection1.object());
    }
    else {
        reset();
        new (d_selection1.buffer())
            int();
        d_selectionId = SELECTION_ID_SELECTION1;
    }

    return d_selection1.object();
}

int& Choice1::makeSelection1(int value)
{
    if (SELECTION_ID_SELECTION1 == d_selectionId) {
        d_selection1.object() = value;
    }
    else {
        reset();
        new (d_selection1.buffer())
                int(value);
        d_selectionId = SELECTION_ID_SELECTION1;
    }

    return d_selection1.object();
}

double& Choice1::makeSelection2()
{
    if (SELECTION_ID_SELECTION2 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_selection2.object());
    }
    else {
        reset();
        new (d_selection2.buffer())
            double();
        d_selectionId = SELECTION_ID_SELECTION2;
    }

    return d_selection2.object();
}

double& Choice1::makeSelection2(double value)
{
    if (SELECTION_ID_SELECTION2 == d_selectionId) {
        d_selection2.object() = value;
    }
    else {
        reset();
        new (d_selection2.buffer())
                double(value);
        d_selectionId = SELECTION_ID_SELECTION2;
    }

    return d_selection2.object();
}

Sequence4& Choice1::makeSelection3()
{
    if (SELECTION_ID_SELECTION3 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(d_selection3);
    }
    else {
        reset();
        d_selection3 = new (*d_allocator_p)
                Sequence4(d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION3;
    }

    return *d_selection3;
}

Sequence4& Choice1::makeSelection3(const Sequence4& value)
{
    if (SELECTION_ID_SELECTION3 == d_selectionId) {
        *d_selection3 = value;
    }
    else {
        reset();
        d_selection3 = new (*d_allocator_p)
                Sequence4(value, d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION3;
    }

    return *d_selection3;
}

Choice2& Choice1::makeSelection4()
{
    if (SELECTION_ID_SELECTION4 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(d_selection4);
    }
    else {
        reset();
        d_selection4 = new (*d_allocator_p)
                Choice2(d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION4;
    }

    return *d_selection4;
}

Choice2& Choice1::makeSelection4(const Choice2& value)
{
    if (SELECTION_ID_SELECTION4 == d_selectionId) {
        *d_selection4 = value;
    }
    else {
        reset();
        d_selection4 = new (*d_allocator_p)
                Choice2(value, d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION4;
    }

    return *d_selection4;
}

// ACCESSORS

bsl::ostream& Choice1::print(
    bsl::ostream& stream,
    int           level,
    int           spacesPerLevel) const
{
    bslim::Printer printer(&stream, level, spacesPerLevel);
    printer.start();
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION1: {
        printer.printAttribute("selection1", d_selection1.object());
      }  break;
      case SELECTION_ID_SELECTION2: {
        printer.printAttribute("selection2", d_selection2.object());
      }  break;
      case SELECTION_ID_SELECTION3: {
    printer.printAttribute("selection3", *d_selection3);
      }  break;
      case SELECTION_ID_SELECTION4: {
    printer.printAttribute("selection4", *d_selection4);
      }  break;
      default:
        stream << "SELECTION UNDEFINED\n";
    }
    printer.end();
    return stream;
}


const char *Choice1::selectionName() const
{
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION1:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION1].name();
      case SELECTION_ID_SELECTION2:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION2].name();
      case SELECTION_ID_SELECTION3:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION3].name();
      case SELECTION_ID_SELECTION4:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION4].name();
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
        return "(* UNDEFINED *)";
    }
}

                               // -------------
                               // class Choice2
                               // -------------

// CONSTANTS

const char Choice2::CLASS_NAME[] = "Choice2";

const bdlat_SelectionInfo Choice2::SELECTION_INFO_ARRAY[] = {
    {
        SELECTION_ID_SELECTION1,
        "selection1",
        sizeof("selection1") - 1,
        "",
        bdlat_FormattingMode::e_TEXT
    },
    {
        SELECTION_ID_SELECTION2,
        "selection2",
        sizeof("selection2") - 1,
        "",
        bdlat_FormattingMode::e_TEXT
    },
    {
        SELECTION_ID_SELECTION3,
        "selection3",
        sizeof("selection3") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        SELECTION_ID_SELECTION4,
        "selection4",
        sizeof("selection4") - 1,
        "",
        bdlat_FormattingMode::e_DEC
    }
};

// CLASS METHODS

const bdlat_SelectionInfo *Choice2::lookupSelectionInfo(
        const char *name,
        int         nameLength)
{
    for (int i = 0; i < 4; ++i) {
        const bdlat_SelectionInfo& selectionInfo =
                    Choice2::SELECTION_INFO_ARRAY[i];

        if (nameLength == selectionInfo.d_nameLength
        &&  0 == bsl::memcmp(selectionInfo.d_name_p, name, nameLength))
        {
            return &selectionInfo;
        }
    }

    return 0;
}

const bdlat_SelectionInfo *Choice2::lookupSelectionInfo(int id)
{
    switch (id) {
      case SELECTION_ID_SELECTION1:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION1];
      case SELECTION_ID_SELECTION2:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION2];
      case SELECTION_ID_SELECTION3:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION3];
      case SELECTION_ID_SELECTION4:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION4];
      default:
        return 0;
    }
}

// CREATORS

Choice2::Choice2(
    const Choice2& original,
    bslma::Allocator *basicAllocator)
: d_selectionId(original.d_selectionId)
, d_allocator_p(bslma::Default::allocator(basicAllocator))
{
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION1: {
        new (d_selection1.buffer())
            bool(original.d_selection1.object());
      } break;
      case SELECTION_ID_SELECTION2: {
        new (d_selection2.buffer())
            bsl::string(
                original.d_selection2.object(), d_allocator_p);
      } break;
      case SELECTION_ID_SELECTION3: {
        d_selection3 = new (*d_allocator_p)
                Choice1(*original.d_selection3, d_allocator_p);
      } break;
      case SELECTION_ID_SELECTION4: {
        new (d_selection4.buffer())
            unsigned int(original.d_selection4.object());
      } break;
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
    }
}

// MANIPULATORS

Choice2&
Choice2::operator=(const Choice2& rhs)
{
    if (this != &rhs) {
        switch (rhs.d_selectionId) {
          case SELECTION_ID_SELECTION1: {
            makeSelection1(rhs.d_selection1.object());
          } break;
          case SELECTION_ID_SELECTION2: {
            makeSelection2(rhs.d_selection2.object());
          } break;
          case SELECTION_ID_SELECTION3: {
            makeSelection3(*rhs.d_selection3);
          } break;
          case SELECTION_ID_SELECTION4: {
            makeSelection4(rhs.d_selection4.object());
          } break;
          default:
            BSLS_ASSERT(SELECTION_ID_UNDEFINED == rhs.d_selectionId);
            reset();
        }
    }

    return *this;
}

void Choice2::reset()
{
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION1: {
        // no destruction required
      } break;
      case SELECTION_ID_SELECTION2: {
        typedef bsl::string Type;
        d_selection2.object().~Type();
      } break;
      case SELECTION_ID_SELECTION3: {
        d_allocator_p->deleteObject(d_selection3);
      } break;
      case SELECTION_ID_SELECTION4: {
        // no destruction required
      } break;
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
    }

    d_selectionId = SELECTION_ID_UNDEFINED;
}

int Choice2::makeSelection(int selectionId)
{
    switch (selectionId) {
      case SELECTION_ID_SELECTION1: {
        makeSelection1();
      } break;
      case SELECTION_ID_SELECTION2: {
        makeSelection2();
      } break;
      case SELECTION_ID_SELECTION3: {
        makeSelection3();
      } break;
      case SELECTION_ID_SELECTION4: {
        makeSelection4();
      } break;
      case SELECTION_ID_UNDEFINED: {
        reset();
      } break;
      default:
        return -1;
    }
    return 0;
}

int Choice2::makeSelection(const char *name, int nameLength)
{
    const bdlat_SelectionInfo *selectionInfo =
                                         lookupSelectionInfo(name, nameLength);
    if (0 == selectionInfo) {
       return -1;
    }

    return makeSelection(selectionInfo->d_id);
}

bool& Choice2::makeSelection1()
{
    if (SELECTION_ID_SELECTION1 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_selection1.object());
    }
    else {
        reset();
        new (d_selection1.buffer())
            bool();
        d_selectionId = SELECTION_ID_SELECTION1;
    }

    return d_selection1.object();
}

bool& Choice2::makeSelection1(bool value)
{
    if (SELECTION_ID_SELECTION1 == d_selectionId) {
        d_selection1.object() = value;
    }
    else {
        reset();
        new (d_selection1.buffer())
                bool(value);
        d_selectionId = SELECTION_ID_SELECTION1;
    }

    return d_selection1.object();
}

bsl::string& Choice2::makeSelection2()
{
    if (SELECTION_ID_SELECTION2 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_selection2.object());
    }
    else {
        reset();
        new (d_selection2.buffer())
                bsl::string(d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION2;
    }

    return d_selection2.object();
}

bsl::string& Choice2::makeSelection2(const bsl::string& value)
{
    if (SELECTION_ID_SELECTION2 == d_selectionId) {
        d_selection2.object() = value;
    }
    else {
        reset();
        new (d_selection2.buffer())
                bsl::string(value, d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION2;
    }

    return d_selection2.object();
}

Choice1& Choice2::makeSelection3()
{
    if (SELECTION_ID_SELECTION3 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(d_selection3);
    }
    else {
        reset();
        d_selection3 = new (*d_allocator_p)
                Choice1(d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION3;
    }

    return *d_selection3;
}

Choice1& Choice2::makeSelection3(const Choice1& value)
{
    if (SELECTION_ID_SELECTION3 == d_selectionId) {
        *d_selection3 = value;
    }
    else {
        reset();
        d_selection3 = new (*d_allocator_p)
                Choice1(value, d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION3;
    }

    return *d_selection3;
}

unsigned int& Choice2::makeSelection4()
{
    if (SELECTION_ID_SELECTION4 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_selection4.object());
    }
    else {
        reset();
        new (d_selection4.buffer())
            unsigned int();
        d_selectionId = SELECTION_ID_SELECTION4;
    }

    return d_selection4.object();
}

unsigned int& Choice2::makeSelection4(unsigned int value)
{
    if (SELECTION_ID_SELECTION4 == d_selectionId) {
        d_selection4.object() = value;
    }
    else {
        reset();
        new (d_selection4.buffer())
                unsigned int(value);
        d_selectionId = SELECTION_ID_SELECTION4;
    }

    return d_selection4.object();
}

// ACCESSORS

bsl::ostream& Choice2::print(
    bsl::ostream& stream,
    int           level,
    int           spacesPerLevel) const
{
    bslim::Printer printer(&stream, level, spacesPerLevel);
    printer.start();
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION1: {
        printer.printAttribute("selection1", d_selection1.object());
      }  break;
      case SELECTION_ID_SELECTION2: {
        printer.printAttribute("selection2", d_selection2.object());
      }  break;
      case SELECTION_ID_SELECTION3: {
    printer.printAttribute("selection3", *d_selection3);
      }  break;
      case SELECTION_ID_SELECTION4: {
        printer.printAttribute("selection4", d_selection4.object());
      }  break;
      default:
        stream << "SELECTION UNDEFINED\n";
    }
    printer.end();
    return stream;
}


const char *Choice2::selectionName() const
{
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION1:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION1].name();
      case SELECTION_ID_SELECTION2:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION2].name();
      case SELECTION_ID_SELECTION3:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION3].name();
      case SELECTION_ID_SELECTION4:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION4].name();
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
        return "(* UNDEFINED *)";
    }
}

                              // ---------------
                              // class Sequence4
                              // ---------------

// CONSTANTS

const char Sequence4::CLASS_NAME[] = "Sequence4";

const bdlat_AttributeInfo Sequence4::ATTRIBUTE_INFO_ARRAY[] = {
    {
        ATTRIBUTE_ID_ELEMENT1,
        "element1",
        sizeof("element1") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        ATTRIBUTE_ID_ELEMENT2,
        "element2",
        sizeof("element2") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        ATTRIBUTE_ID_ELEMENT3,
        "element3",
        sizeof("element3") - 1,
        "",
        bdlat_FormattingMode::e_HEX
    },
    {
        ATTRIBUTE_ID_ELEMENT4,
        "element4",
        sizeof("element4") - 1,
        "",
        bdlat_FormattingMode::e_DEC
    },
    {
        ATTRIBUTE_ID_ELEMENT5,
        "element5",
        sizeof("element5") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        ATTRIBUTE_ID_ELEMENT6,
        "element6",
        sizeof("element6") - 1,
        "",
        bdlat_FormattingMode::e_TEXT
    },
    {
        ATTRIBUTE_ID_ELEMENT7,
        "element7",
        sizeof("element7") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        ATTRIBUTE_ID_ELEMENT8,
        "element8",
        sizeof("element8") - 1,
        "",
        bdlat_FormattingMode::e_TEXT
    },
    {
        ATTRIBUTE_ID_ELEMENT9,
        "element9",
        sizeof("element9") - 1,
        "",
        bdlat_FormattingMode::e_TEXT
    },
    {
        ATTRIBUTE_ID_ELEMENT10,
        "element10",
        sizeof("element10") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        ATTRIBUTE_ID_ELEMENT11,
        "element11",
        sizeof("element11") - 1,
        "",
        bdlat_FormattingMode::e_HEX
    },
    {
        ATTRIBUTE_ID_ELEMENT12,
        "element12",
        sizeof("element12") - 1,
        "",
        bdlat_FormattingMode::e_DEC
    },
    {
        ATTRIBUTE_ID_ELEMENT13,
        "element13",
        sizeof("element13") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        ATTRIBUTE_ID_ELEMENT14,
        "element14",
        sizeof("element14") - 1,
        "",
        bdlat_FormattingMode::e_TEXT
    },
    {
        ATTRIBUTE_ID_ELEMENT15,
        "element15",
        sizeof("element15") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        ATTRIBUTE_ID_ELEMENT16,
        "element16",
        sizeof("element16") - 1,
        "",
        bdlat_FormattingMode::e_HEX
    },
    {
        ATTRIBUTE_ID_ELEMENT17,
        "element17",
        sizeof("element17") - 1,
        "",
        bdlat_FormattingMode::e_DEC
    },
    {
        ATTRIBUTE_ID_ELEMENT18,
        "element18",
        sizeof("element18") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        ATTRIBUTE_ID_ELEMENT19,
        "element19",
        sizeof("element19") - 1,
        "",
        bdlat_FormattingMode::e_TEXT
    }
};

// CLASS METHODS

const bdlat_AttributeInfo *Sequence4::lookupAttributeInfo(
        const char *name,
        int         nameLength)
{
    for (int i = 0; i < 19; ++i) {
        const bdlat_AttributeInfo& attributeInfo =
                    Sequence4::ATTRIBUTE_INFO_ARRAY[i];

        if (nameLength == attributeInfo.d_nameLength
        &&  0 == bsl::memcmp(attributeInfo.d_name_p, name, nameLength))
        {
            return &attributeInfo;
        }
    }

    return 0;
}

const bdlat_AttributeInfo *Sequence4::lookupAttributeInfo(int id)
{
    switch (id) {
      case ATTRIBUTE_ID_ELEMENT1:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT1];
      case ATTRIBUTE_ID_ELEMENT2:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT2];
      case ATTRIBUTE_ID_ELEMENT3:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT3];
      case ATTRIBUTE_ID_ELEMENT4:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT4];
      case ATTRIBUTE_ID_ELEMENT5:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT5];
      case ATTRIBUTE_ID_ELEMENT6:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT6];
      case ATTRIBUTE_ID_ELEMENT7:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT7];
      case ATTRIBUTE_ID_ELEMENT8:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT8];
      case ATTRIBUTE_ID_ELEMENT9:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT9];
      case ATTRIBUTE_ID_ELEMENT10:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT10];
      case ATTRIBUTE_ID_ELEMENT11:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT11];
      case ATTRIBUTE_ID_ELEMENT12:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT12];
      case ATTRIBUTE_ID_ELEMENT13:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT13];
      case ATTRIBUTE_ID_ELEMENT14:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT14];
      case ATTRIBUTE_ID_ELEMENT15:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT15];
      case ATTRIBUTE_ID_ELEMENT16:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT16];
      case ATTRIBUTE_ID_ELEMENT17:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT17];
      case ATTRIBUTE_ID_ELEMENT18:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT18];
      case ATTRIBUTE_ID_ELEMENT19:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT19];
      default:
        return 0;
    }
}

// CREATORS

Sequence4::Sequence4(bslma::Allocator *basicAllocator)
: d_element10()
, d_element17(basicAllocator)
, d_element15(basicAllocator)
, d_element11(basicAllocator)
, d_element16(basicAllocator)
, d_element14(basicAllocator)
, d_element18(basicAllocator)
, d_element1(basicAllocator)
, d_element19(basicAllocator)
, d_element2(basicAllocator)
, d_element9(basicAllocator)
, d_element3(basicAllocator)
, d_element5()
, d_element6(basicAllocator)
, d_element12()
, d_element4()
, d_element7()
, d_element13(static_cast<Enumerated::Value>(0))
, d_element8()
{
}

Sequence4::Sequence4(const Sequence4& original,
                     bslma::Allocator *basicAllocator)
: d_element10(original.d_element10)
, d_element17(original.d_element17, basicAllocator)
, d_element15(original.d_element15, basicAllocator)
, d_element11(original.d_element11, basicAllocator)
, d_element16(original.d_element16, basicAllocator)
, d_element14(original.d_element14, basicAllocator)
, d_element18(original.d_element18, basicAllocator)
, d_element1(original.d_element1, basicAllocator)
, d_element19(original.d_element19, basicAllocator)
, d_element2(original.d_element2, basicAllocator)
, d_element9(original.d_element9, basicAllocator)
, d_element3(original.d_element3, basicAllocator)
, d_element5(original.d_element5)
, d_element6(original.d_element6, basicAllocator)
, d_element12(original.d_element12)
, d_element4(original.d_element4)
, d_element7(original.d_element7)
, d_element13(original.d_element13)
, d_element8(original.d_element8)
{
}

Sequence4::~Sequence4()
{
}

// MANIPULATORS

Sequence4&
Sequence4::operator=(const Sequence4& rhs)
{
    if (this != &rhs) {
        d_element1 = rhs.d_element1;
        d_element2 = rhs.d_element2;
        d_element3 = rhs.d_element3;
        d_element4 = rhs.d_element4;
        d_element5 = rhs.d_element5;
        d_element6 = rhs.d_element6;
        d_element7 = rhs.d_element7;
        d_element8 = rhs.d_element8;
        d_element9 = rhs.d_element9;
        d_element10 = rhs.d_element10;
        d_element11 = rhs.d_element11;
        d_element12 = rhs.d_element12;
        d_element13 = rhs.d_element13;
        d_element14 = rhs.d_element14;
        d_element15 = rhs.d_element15;
        d_element16 = rhs.d_element16;
        d_element17 = rhs.d_element17;
        d_element18 = rhs.d_element18;
        d_element19 = rhs.d_element19;
    }

    return *this;
}

void Sequence4::reset()
{
    bdlat_ValueTypeFunctions::reset(&d_element1);
    bdlat_ValueTypeFunctions::reset(&d_element2);
    bdlat_ValueTypeFunctions::reset(&d_element3);
    bdlat_ValueTypeFunctions::reset(&d_element4);
    bdlat_ValueTypeFunctions::reset(&d_element5);
    bdlat_ValueTypeFunctions::reset(&d_element6);
    bdlat_ValueTypeFunctions::reset(&d_element7);
    bdlat_ValueTypeFunctions::reset(&d_element8);
    bdlat_ValueTypeFunctions::reset(&d_element9);
    bdlat_ValueTypeFunctions::reset(&d_element10);
    bdlat_ValueTypeFunctions::reset(&d_element11);
    bdlat_ValueTypeFunctions::reset(&d_element12);
    bdlat_ValueTypeFunctions::reset(&d_element13);
    bdlat_ValueTypeFunctions::reset(&d_element14);
    bdlat_ValueTypeFunctions::reset(&d_element15);
    bdlat_ValueTypeFunctions::reset(&d_element16);
    bdlat_ValueTypeFunctions::reset(&d_element17);
    bdlat_ValueTypeFunctions::reset(&d_element18);
    bdlat_ValueTypeFunctions::reset(&d_element19);
}

// ACCESSORS

bsl::ostream& Sequence4::print(
        bsl::ostream& stream,
        int           level,
        int           spacesPerLevel) const
{
    bslim::Printer printer(&stream, level, spacesPerLevel);
    printer.start();
    printer.printAttribute("element1", d_element1);
    printer.printAttribute("element2", d_element2);
    printer.printAttribute("element3", d_element3);
    printer.printAttribute("element4", d_element4);
    printer.printAttribute("element5", d_element5);
    printer.printAttribute("element6", d_element6);
    printer.printAttribute("element7", d_element7);
    printer.printAttribute("element8", d_element8);
    printer.printAttribute("element9", d_element9);
    printer.printAttribute("element10", d_element10);
    {
        bool multilineFlag = (0 <= level);
        bdlb::Print::indent(stream, level + 1, spacesPerLevel);
        stream << (multilineFlag ? "" : " ");
        stream << "element11 = [ ";
        bdlb::Print::singleLineHexDump(
            stream, d_element11.begin(), d_element11.end());
        stream << " ]" << (multilineFlag ? "\n" : "");
    }
    printer.printAttribute("element12", d_element12);
    printer.printAttribute("element13", d_element13);
    printer.printAttribute("element14", d_element14);
    printer.printAttribute("element15", d_element15);
    printer.printAttribute("element16", d_element16);
    printer.printAttribute("element17", d_element17);
    printer.printAttribute("element18", d_element18);
    printer.printAttribute("element19", d_element19);
    printer.end();
    return stream;
}



                              // ---------------
                              // class Sequence1
                              // ---------------

// CONSTANTS

const char Sequence1::CLASS_NAME[] = "Sequence1";

const bdlat_AttributeInfo Sequence1::ATTRIBUTE_INFO_ARRAY[] = {
    {
        ATTRIBUTE_ID_ELEMENT1,
        "element1",
        sizeof("element1") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        ATTRIBUTE_ID_ELEMENT2,
        "element2",
        sizeof("element2") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        ATTRIBUTE_ID_ELEMENT3,
        "element3",
        sizeof("element3") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        ATTRIBUTE_ID_ELEMENT4,
        "element4",
        sizeof("element4") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
      | bdlat_FormattingMode::e_NILLABLE
    },
    {
        ATTRIBUTE_ID_ELEMENT5,
        "element5",
        sizeof("element5") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    }
};

// CLASS METHODS

const bdlat_AttributeInfo *Sequence1::lookupAttributeInfo(
        const char *name,
        int         nameLength)
{
    for (int i = 0; i < 5; ++i) {
        const bdlat_AttributeInfo& attributeInfo =
                    Sequence1::ATTRIBUTE_INFO_ARRAY[i];

        if (nameLength == attributeInfo.d_nameLength
        &&  0 == bsl::memcmp(attributeInfo.d_name_p, name, nameLength))
        {
            return &attributeInfo;
        }
    }

    return 0;
}

const bdlat_AttributeInfo *Sequence1::lookupAttributeInfo(int id)
{
    switch (id) {
      case ATTRIBUTE_ID_ELEMENT1:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT1];
      case ATTRIBUTE_ID_ELEMENT2:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT2];
      case ATTRIBUTE_ID_ELEMENT3:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT3];
      case ATTRIBUTE_ID_ELEMENT4:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT4];
      case ATTRIBUTE_ID_ELEMENT5:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT5];
      default:
        return 0;
    }
}

// CREATORS

Sequence1::Sequence1(bslma::Allocator *basicAllocator)
: d_allocator_p(bslma::Default::allocator(basicAllocator))
, d_element4(basicAllocator)
, d_element5(basicAllocator)
, d_element2(basicAllocator)
, d_element1(basicAllocator)
{
    d_element3 = new (*d_allocator_p)
            Choice2(d_allocator_p);
}

Sequence1::Sequence1(const Sequence1& original,
                     bslma::Allocator *basicAllocator)
: d_allocator_p(bslma::Default::allocator(basicAllocator))
, d_element4(original.d_element4, basicAllocator)
, d_element5(original.d_element5, basicAllocator)
, d_element2(original.d_element2, basicAllocator)
, d_element1(original.d_element1, basicAllocator)
{
    d_element3 = new (*d_allocator_p)
            Choice2(*original.d_element3, d_allocator_p);
}

Sequence1::~Sequence1()
{
    d_allocator_p->deleteObject(d_element3);
}

// MANIPULATORS

Sequence1&
Sequence1::operator=(const Sequence1& rhs)
{
    if (this != &rhs) {
        d_element1 = rhs.d_element1;
        d_element2 = rhs.d_element2;
        *d_element3 = *rhs.d_element3;
        d_element4 = rhs.d_element4;
        d_element5 = rhs.d_element5;
    }

    return *this;
}

void Sequence1::reset()
{
    bdlat_ValueTypeFunctions::reset(&d_element1);
    bdlat_ValueTypeFunctions::reset(&d_element2);
    bdlat_ValueTypeFunctions::reset(d_element3);
    bdlat_ValueTypeFunctions::reset(&d_element4);
    bdlat_ValueTypeFunctions::reset(&d_element5);
}

// ACCESSORS

bsl::ostream& Sequence1::print(
        bsl::ostream& stream,
        int           level,
        int           spacesPerLevel) const
{
    bslim::Printer printer(&stream, level, spacesPerLevel);
    printer.start();
    printer.printAttribute("element1", d_element1);
    printer.printAttribute("element2", d_element2);
    printer.printAttribute("element3", *d_element3);
    printer.printAttribute("element4", d_element4);
    printer.printAttribute("element5", d_element5);
    printer.end();
    return stream;
}



                              // ---------------
                              // class Sequence2
                              // ---------------

// CONSTANTS

const char Sequence2::CLASS_NAME[] = "Sequence2";

const bdlat_AttributeInfo Sequence2::ATTRIBUTE_INFO_ARRAY[] = {
    {
        ATTRIBUTE_ID_ELEMENT1,
        "element1",
        sizeof("element1") - 1,
        "",
        bdlat_FormattingMode::e_TEXT
    },
    {
        ATTRIBUTE_ID_ELEMENT2,
        "element2",
        sizeof("element2") - 1,
        "",
        bdlat_FormattingMode::e_DEC
    },
    {
        ATTRIBUTE_ID_ELEMENT3,
        "element3",
        sizeof("element3") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        ATTRIBUTE_ID_ELEMENT4,
        "element4",
        sizeof("element4") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        ATTRIBUTE_ID_ELEMENT5,
        "element5",
        sizeof("element5") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    }
};

// CLASS METHODS

const bdlat_AttributeInfo *Sequence2::lookupAttributeInfo(
        const char *name,
        int         nameLength)
{
    for (int i = 0; i < 5; ++i) {
        const bdlat_AttributeInfo& attributeInfo =
                    Sequence2::ATTRIBUTE_INFO_ARRAY[i];

        if (nameLength == attributeInfo.d_nameLength
        &&  0 == bsl::memcmp(attributeInfo.d_name_p, name, nameLength))
        {
            return &attributeInfo;
        }
    }

    return 0;
}

const bdlat_AttributeInfo *Sequence2::lookupAttributeInfo(int id)
{
    switch (id) {
      case ATTRIBUTE_ID_ELEMENT1:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT1];
      case ATTRIBUTE_ID_ELEMENT2:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT2];
      case ATTRIBUTE_ID_ELEMENT3:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT3];
      case ATTRIBUTE_ID_ELEMENT4:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT4];
      case ATTRIBUTE_ID_ELEMENT5:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT5];
      default:
        return 0;
    }
}

// CREATORS

Sequence2::Sequence2(bslma::Allocator *basicAllocator)
: d_element3()
, d_element5()
, d_element4(basicAllocator)
, d_element1(basicAllocator)
, d_element2()
{
}

Sequence2::Sequence2(const Sequence2& original,
                     bslma::Allocator *basicAllocator)
: d_element3(original.d_element3)
, d_element5(original.d_element5)
, d_element4(original.d_element4, basicAllocator)
, d_element1(original.d_element1, basicAllocator)
, d_element2(original.d_element2)
{
}

Sequence2::~Sequence2()
{
}

// MANIPULATORS

Sequence2&
Sequence2::operator=(const Sequence2& rhs)
{
    if (this != &rhs) {
        d_element1 = rhs.d_element1;
        d_element2 = rhs.d_element2;
        d_element3 = rhs.d_element3;
        d_element4 = rhs.d_element4;
        d_element5 = rhs.d_element5;
    }

    return *this;
}

void Sequence2::reset()
{
    bdlat_ValueTypeFunctions::reset(&d_element1);
    bdlat_ValueTypeFunctions::reset(&d_element2);
    bdlat_ValueTypeFunctions::reset(&d_element3);
    bdlat_ValueTypeFunctions::reset(&d_element4);
    bdlat_ValueTypeFunctions::reset(&d_element5);
}

// ACCESSORS

bsl::ostream& Sequence2::print(
        bsl::ostream& stream,
        int           level,
        int           spacesPerLevel) const
{
    bslim::Printer printer(&stream, level, spacesPerLevel);
    printer.start();
    printer.printAttribute("element1", d_element1);
    printer.printAttribute("element2", (int)d_element2);
    printer.printAttribute("element3", d_element3);
    printer.printAttribute("element4", d_element4);
    printer.printAttribute("element5", d_element5);
    printer.end();
    return stream;
}



                     // ----------------------------------
                     // class SequenceWithAnonymityChoice2
                     // ----------------------------------

// CONSTANTS

const char SequenceWithAnonymityChoice2::CLASS_NAME[] = "SequenceWithAnonymityChoice2";

const bdlat_SelectionInfo SequenceWithAnonymityChoice2::SELECTION_INFO_ARRAY[] = {
    {
        SELECTION_ID_SELECTION7,
        "selection7",
        sizeof("selection7") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        SELECTION_ID_SELECTION8,
        "selection8",
        sizeof("selection8") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    }
};

// CLASS METHODS

const bdlat_SelectionInfo *SequenceWithAnonymityChoice2::lookupSelectionInfo(
        const char *name,
        int         nameLength)
{
    for (int i = 0; i < 2; ++i) {
        const bdlat_SelectionInfo& selectionInfo =
                    SequenceWithAnonymityChoice2::SELECTION_INFO_ARRAY[i];

        if (nameLength == selectionInfo.d_nameLength
        &&  0 == bsl::memcmp(selectionInfo.d_name_p, name, nameLength))
        {
            return &selectionInfo;
        }
    }

    return 0;
}

const bdlat_SelectionInfo *SequenceWithAnonymityChoice2::lookupSelectionInfo(int id)
{
    switch (id) {
      case SELECTION_ID_SELECTION7:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION7];
      case SELECTION_ID_SELECTION8:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION8];
      default:
        return 0;
    }
}

// CREATORS

SequenceWithAnonymityChoice2::SequenceWithAnonymityChoice2(
    const SequenceWithAnonymityChoice2& original,
    bslma::Allocator *basicAllocator)
: d_selectionId(original.d_selectionId)
, d_allocator_p(bslma::Default::allocator(basicAllocator))
{
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION7: {
        d_selection7 = new (*d_allocator_p)
                Sequence4(*original.d_selection7, d_allocator_p);
      } break;
      case SELECTION_ID_SELECTION8: {
        d_selection8 = new (*d_allocator_p)
                Choice2(*original.d_selection8, d_allocator_p);
      } break;
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
    }
}

// MANIPULATORS

SequenceWithAnonymityChoice2&
SequenceWithAnonymityChoice2::operator=(const SequenceWithAnonymityChoice2& rhs)
{
    if (this != &rhs) {
        switch (rhs.d_selectionId) {
          case SELECTION_ID_SELECTION7: {
            makeSelection7(*rhs.d_selection7);
          } break;
          case SELECTION_ID_SELECTION8: {
            makeSelection8(*rhs.d_selection8);
          } break;
          default:
            BSLS_ASSERT(SELECTION_ID_UNDEFINED == rhs.d_selectionId);
            reset();
        }
    }

    return *this;
}

void SequenceWithAnonymityChoice2::reset()
{
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION7: {
        d_allocator_p->deleteObject(d_selection7);
      } break;
      case SELECTION_ID_SELECTION8: {
        d_allocator_p->deleteObject(d_selection8);
      } break;
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
    }

    d_selectionId = SELECTION_ID_UNDEFINED;
}

int SequenceWithAnonymityChoice2::makeSelection(int selectionId)
{
    switch (selectionId) {
      case SELECTION_ID_SELECTION7: {
        makeSelection7();
      } break;
      case SELECTION_ID_SELECTION8: {
        makeSelection8();
      } break;
      case SELECTION_ID_UNDEFINED: {
        reset();
      } break;
      default:
        return -1;
    }
    return 0;
}

int SequenceWithAnonymityChoice2::makeSelection(const char *name, int nameLength)
{
    const bdlat_SelectionInfo *selectionInfo =
                                         lookupSelectionInfo(name, nameLength);
    if (0 == selectionInfo) {
       return -1;
    }

    return makeSelection(selectionInfo->d_id);
}

Sequence4& SequenceWithAnonymityChoice2::makeSelection7()
{
    if (SELECTION_ID_SELECTION7 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(d_selection7);
    }
    else {
        reset();
        d_selection7 = new (*d_allocator_p)
                Sequence4(d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION7;
    }

    return *d_selection7;
}

Sequence4& SequenceWithAnonymityChoice2::makeSelection7(const Sequence4& value)
{
    if (SELECTION_ID_SELECTION7 == d_selectionId) {
        *d_selection7 = value;
    }
    else {
        reset();
        d_selection7 = new (*d_allocator_p)
                Sequence4(value, d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION7;
    }

    return *d_selection7;
}

Choice2& SequenceWithAnonymityChoice2::makeSelection8()
{
    if (SELECTION_ID_SELECTION8 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(d_selection8);
    }
    else {
        reset();
        d_selection8 = new (*d_allocator_p)
                Choice2(d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION8;
    }

    return *d_selection8;
}

Choice2& SequenceWithAnonymityChoice2::makeSelection8(const Choice2& value)
{
    if (SELECTION_ID_SELECTION8 == d_selectionId) {
        *d_selection8 = value;
    }
    else {
        reset();
        d_selection8 = new (*d_allocator_p)
                Choice2(value, d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION8;
    }

    return *d_selection8;
}

// ACCESSORS

bsl::ostream& SequenceWithAnonymityChoice2::print(
    bsl::ostream& stream,
    int           level,
    int           spacesPerLevel) const
{
    bslim::Printer printer(&stream, level, spacesPerLevel);
    printer.start();
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION7: {
    printer.printAttribute("selection7", *d_selection7);
      }  break;
      case SELECTION_ID_SELECTION8: {
    printer.printAttribute("selection8", *d_selection8);
      }  break;
      default:
        stream << "SELECTION UNDEFINED\n";
    }
    printer.end();
    return stream;
}


const char *SequenceWithAnonymityChoice2::selectionName() const
{
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION7:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION7].name();
      case SELECTION_ID_SELECTION8:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION8].name();
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
        return "(* UNDEFINED *)";
    }
}

                        // ---------------------------
                        // class SequenceWithAnonymity
                        // ---------------------------

// CONSTANTS

const char SequenceWithAnonymity::CLASS_NAME[] = "SequenceWithAnonymity";

const bdlat_AttributeInfo SequenceWithAnonymity::ATTRIBUTE_INFO_ARRAY[] = {
    {
        ATTRIBUTE_ID_CHOICE,
        "Choice",
        sizeof("Choice") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
      | bdlat_FormattingMode::e_UNTAGGED
    },
    {
        ATTRIBUTE_ID_CHOICE1,
        "Choice-1",
        sizeof("Choice-1") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
      | bdlat_FormattingMode::e_UNTAGGED
    },
    {
        ATTRIBUTE_ID_CHOICE2,
        "Choice-2",
        sizeof("Choice-2") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
      | bdlat_FormattingMode::e_UNTAGGED
    },
    {
        ATTRIBUTE_ID_ELEMENT4,
        "element4",
        sizeof("element4") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    }
};

// CLASS METHODS

const bdlat_AttributeInfo *SequenceWithAnonymity::lookupAttributeInfo(
        const char *name,
        int         nameLength)
{
    if (bdlb::String::areEqualCaseless("selection1", name, nameLength)) {
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_CHOICE];
    }

    if (bdlb::String::areEqualCaseless("selection2", name, nameLength)) {
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_CHOICE];
    }

    if (bdlb::String::areEqualCaseless("selection3", name, nameLength)) {
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_CHOICE];
    }

    if (bdlb::String::areEqualCaseless("selection4", name, nameLength)) {
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_CHOICE];
    }

    if (bdlb::String::areEqualCaseless("selection5", name, nameLength)) {
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_CHOICE1];
    }

    if (bdlb::String::areEqualCaseless("selection6", name, nameLength)) {
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_CHOICE1];
    }

    if (bdlb::String::areEqualCaseless("selection7", name, nameLength)) {
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_CHOICE2];
    }

    if (bdlb::String::areEqualCaseless("selection8", name, nameLength)) {
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_CHOICE2];
    }

    for (int i = 0; i < 4; ++i) {
        const bdlat_AttributeInfo& attributeInfo =
                    SequenceWithAnonymity::ATTRIBUTE_INFO_ARRAY[i];

        if (nameLength == attributeInfo.d_nameLength
        &&  0 == bsl::memcmp(attributeInfo.d_name_p, name, nameLength))
        {
            return &attributeInfo;
        }
    }

    return 0;
}

const bdlat_AttributeInfo *SequenceWithAnonymity::lookupAttributeInfo(int id)
{
    switch (id) {
      case ATTRIBUTE_ID_CHOICE:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_CHOICE];
      case ATTRIBUTE_ID_CHOICE1:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_CHOICE1];
      case ATTRIBUTE_ID_CHOICE2:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_CHOICE2];
      case ATTRIBUTE_ID_ELEMENT4:
        return &ATTRIBUTE_INFO_ARRAY[ATTRIBUTE_INDEX_ELEMENT4];
      default:
        return 0;
    }
}

// CREATORS

SequenceWithAnonymity::SequenceWithAnonymity(bslma::Allocator *basicAllocator)
: d_choice2(basicAllocator)
, d_choice1(basicAllocator)
, d_choice(basicAllocator)
, d_element4(basicAllocator)
{
}

SequenceWithAnonymity::SequenceWithAnonymity(const SequenceWithAnonymity& original,
                                             bslma::Allocator *basicAllocator)
: d_choice2(original.d_choice2, basicAllocator)
, d_choice1(original.d_choice1, basicAllocator)
, d_choice(original.d_choice, basicAllocator)
, d_element4(original.d_element4, basicAllocator)
{
}

SequenceWithAnonymity::~SequenceWithAnonymity()
{
}

// MANIPULATORS

SequenceWithAnonymity&
SequenceWithAnonymity::operator=(const SequenceWithAnonymity& rhs)
{
    if (this != &rhs) {
        d_choice = rhs.d_choice;
        d_choice1 = rhs.d_choice1;
        d_choice2 = rhs.d_choice2;
        d_element4 = rhs.d_element4;
    }

    return *this;
}

void SequenceWithAnonymity::reset()
{
    bdlat_ValueTypeFunctions::reset(&d_choice);
    bdlat_ValueTypeFunctions::reset(&d_choice1);
    bdlat_ValueTypeFunctions::reset(&d_choice2);
    bdlat_ValueTypeFunctions::reset(&d_element4);
}

// ACCESSORS

bsl::ostream& SequenceWithAnonymity::print(
        bsl::ostream& stream,
        int           level,
        int           spacesPerLevel) const
{
    bslim::Printer printer(&stream, level, spacesPerLevel);
    printer.start();
    printer.printAttribute("choice", d_choice);
    printer.printAttribute("choice1", d_choice1);
    printer.printAttribute("choice2", d_choice2);
    printer.printAttribute("element4", d_element4);
    printer.end();
    return stream;
}



                          // ------------------------
                          // class FeatureTestMessage
                          // ------------------------

// CONSTANTS

const char FeatureTestMessage::CLASS_NAME[] = "FeatureTestMessage";

const bdlat_SelectionInfo FeatureTestMessage::SELECTION_INFO_ARRAY[] = {
    {
        SELECTION_ID_SELECTION1,
        "selection1",
        sizeof("selection1") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        SELECTION_ID_SELECTION2,
        "selection2",
        sizeof("selection2") - 1,
        "",
        bdlat_FormattingMode::e_HEX
    },
    {
        SELECTION_ID_SELECTION3,
        "selection3",
        sizeof("selection3") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        SELECTION_ID_SELECTION4,
        "selection4",
        sizeof("selection4") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        SELECTION_ID_SELECTION5,
        "selection5",
        sizeof("selection5") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        SELECTION_ID_SELECTION6,
        "selection6",
        sizeof("selection6") - 1,
        "",
        bdlat_FormattingMode::e_TEXT
    },
    {
        SELECTION_ID_SELECTION7,
        "selection7",
        sizeof("selection7") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        SELECTION_ID_SELECTION8,
        "selection8",
        sizeof("selection8") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        SELECTION_ID_SELECTION9,
        "selection9",
        sizeof("selection9") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        SELECTION_ID_SELECTION10,
        "selection10",
        sizeof("selection10") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        SELECTION_ID_SELECTION11,
        "selection11",
        sizeof("selection11") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    }
};

// CLASS METHODS

const bdlat_SelectionInfo *FeatureTestMessage::lookupSelectionInfo(
        const char *name,
        int         nameLength)
{
    for (int i = 0; i < 11; ++i) {
        const bdlat_SelectionInfo& selectionInfo =
                    FeatureTestMessage::SELECTION_INFO_ARRAY[i];

        if (nameLength == selectionInfo.d_nameLength
        &&  0 == bsl::memcmp(selectionInfo.d_name_p, name, nameLength))
        {
            return &selectionInfo;
        }
    }

    return 0;
}

const bdlat_SelectionInfo *FeatureTestMessage::lookupSelectionInfo(int id)
{
    switch (id) {
      case SELECTION_ID_SELECTION1:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION1];
      case SELECTION_ID_SELECTION2:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION2];
      case SELECTION_ID_SELECTION3:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION3];
      case SELECTION_ID_SELECTION4:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION4];
      case SELECTION_ID_SELECTION5:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION5];
      case SELECTION_ID_SELECTION6:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION6];
      case SELECTION_ID_SELECTION7:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION7];
      case SELECTION_ID_SELECTION8:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION8];
      case SELECTION_ID_SELECTION9:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION9];
      case SELECTION_ID_SELECTION10:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION10];
      case SELECTION_ID_SELECTION11:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION11];
      default:
        return 0;
    }
}

// CREATORS

FeatureTestMessage::FeatureTestMessage(
    const FeatureTestMessage& original,
    bslma::Allocator *basicAllocator)
: d_selectionId(original.d_selectionId)
, d_allocator_p(bslma::Default::allocator(basicAllocator))
{
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION1: {
        new (d_selection1.buffer())
            Sequence1(
                original.d_selection1.object(), d_allocator_p);
      } break;
      case SELECTION_ID_SELECTION2: {
        new (d_selection2.buffer())
            bsl::vector<char>(
                original.d_selection2.object(), d_allocator_p);
      } break;
      case SELECTION_ID_SELECTION3: {
        new (d_selection3.buffer())
            Sequence2(
                original.d_selection3.object(), d_allocator_p);
      } break;
      case SELECTION_ID_SELECTION4: {
        d_selection4 = new (*d_allocator_p)
                Sequence3(*original.d_selection4, d_allocator_p);
      } break;
      case SELECTION_ID_SELECTION5: {
        new (d_selection5.buffer())
            bdlt::DatetimeTz(original.d_selection5.object());
      } break;
      case SELECTION_ID_SELECTION6: {
        new (d_selection6.buffer())
            CustomString(
                original.d_selection6.object(), d_allocator_p);
      } break;
      case SELECTION_ID_SELECTION7: {
        new (d_selection7.buffer())
            Enumerated::Value(original.d_selection7.object());
      } break;
      case SELECTION_ID_SELECTION8: {
        new (d_selection8.buffer())
            Choice3(
                original.d_selection8.object(), d_allocator_p);
      } break;
      case SELECTION_ID_SELECTION9: {
        new (d_selection9.buffer())
            VoidSequence(original.d_selection9.object());
      } break;
      case SELECTION_ID_SELECTION10: {
        new (d_selection10.buffer())
            UnsignedSequence(original.d_selection10.object());
      } break;
      case SELECTION_ID_SELECTION11: {
        new (d_selection11.buffer())
            SequenceWithAnonymity(
                original.d_selection11.object(), d_allocator_p);
      } break;
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
    }
}

// MANIPULATORS

FeatureTestMessage&
FeatureTestMessage::operator=(const FeatureTestMessage& rhs)
{
    if (this != &rhs) {
        switch (rhs.d_selectionId) {
          case SELECTION_ID_SELECTION1: {
            makeSelection1(rhs.d_selection1.object());
          } break;
          case SELECTION_ID_SELECTION2: {
            makeSelection2(rhs.d_selection2.object());
          } break;
          case SELECTION_ID_SELECTION3: {
            makeSelection3(rhs.d_selection3.object());
          } break;
          case SELECTION_ID_SELECTION4: {
            makeSelection4(*rhs.d_selection4);
          } break;
          case SELECTION_ID_SELECTION5: {
            makeSelection5(rhs.d_selection5.object());
          } break;
          case SELECTION_ID_SELECTION6: {
            makeSelection6(rhs.d_selection6.object());
          } break;
          case SELECTION_ID_SELECTION7: {
            makeSelection7(rhs.d_selection7.object());
          } break;
          case SELECTION_ID_SELECTION8: {
            makeSelection8(rhs.d_selection8.object());
          } break;
          case SELECTION_ID_SELECTION9: {
            makeSelection9(rhs.d_selection9.object());
          } break;
          case SELECTION_ID_SELECTION10: {
            makeSelection10(rhs.d_selection10.object());
          } break;
          case SELECTION_ID_SELECTION11: {
            makeSelection11(rhs.d_selection11.object());
          } break;
          default:
            BSLS_ASSERT(SELECTION_ID_UNDEFINED == rhs.d_selectionId);
            reset();
        }
    }

    return *this;
}

void FeatureTestMessage::reset()
{
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION1: {
        d_selection1.object().~Sequence1();
      } break;
      case SELECTION_ID_SELECTION2: {
        typedef bsl::vector<char> Type;
        d_selection2.object().~Type();
      } break;
      case SELECTION_ID_SELECTION3: {
        d_selection3.object().~Sequence2();
      } break;
      case SELECTION_ID_SELECTION4: {
        d_allocator_p->deleteObject(d_selection4);
      } break;
      case SELECTION_ID_SELECTION5: {
        // no destruction required
      } break;
      case SELECTION_ID_SELECTION6: {
        d_selection6.object().~CustomString();
      } break;
      case SELECTION_ID_SELECTION7: {
        typedef Enumerated::Value Type;
        d_selection7.object().~Type();
      } break;
      case SELECTION_ID_SELECTION8: {
        d_selection8.object().~Choice3();
      } break;
      case SELECTION_ID_SELECTION9: {
        d_selection9.object().~VoidSequence();
      } break;
      case SELECTION_ID_SELECTION10: {
        d_selection10.object().~UnsignedSequence();
      } break;
      case SELECTION_ID_SELECTION11: {
        d_selection11.object().~SequenceWithAnonymity();
      } break;
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
    }

    d_selectionId = SELECTION_ID_UNDEFINED;
}

int FeatureTestMessage::makeSelection(int selectionId)
{
    switch (selectionId) {
      case SELECTION_ID_SELECTION1: {
        makeSelection1();
      } break;
      case SELECTION_ID_SELECTION2: {
        makeSelection2();
      } break;
      case SELECTION_ID_SELECTION3: {
        makeSelection3();
      } break;
      case SELECTION_ID_SELECTION4: {
        makeSelection4();
      } break;
      case SELECTION_ID_SELECTION5: {
        makeSelection5();
      } break;
      case SELECTION_ID_SELECTION6: {
        makeSelection6();
      } break;
      case SELECTION_ID_SELECTION7: {
        makeSelection7();
      } break;
      case SELECTION_ID_SELECTION8: {
        makeSelection8();
      } break;
      case SELECTION_ID_SELECTION9: {
        makeSelection9();
      } break;
      case SELECTION_ID_SELECTION10: {
        makeSelection10();
      } break;
      case SELECTION_ID_SELECTION11: {
        makeSelection11();
      } break;
      case SELECTION_ID_UNDEFINED: {
        reset();
      } break;
      default:
        return -1;
    }
    return 0;
}

int FeatureTestMessage::makeSelection(const char *name, int nameLength)
{
    const bdlat_SelectionInfo *selectionInfo =
                                         lookupSelectionInfo(name, nameLength);
    if (0 == selectionInfo) {
       return -1;
    }

    return makeSelection(selectionInfo->d_id);
}

Sequence1& FeatureTestMessage::makeSelection1()
{
    if (SELECTION_ID_SELECTION1 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_selection1.object());
    }
    else {
        reset();
        new (d_selection1.buffer())
                Sequence1(d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION1;
    }

    return d_selection1.object();
}

Sequence1& FeatureTestMessage::makeSelection1(const Sequence1& value)
{
    if (SELECTION_ID_SELECTION1 == d_selectionId) {
        d_selection1.object() = value;
    }
    else {
        reset();
        new (d_selection1.buffer())
                Sequence1(value, d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION1;
    }

    return d_selection1.object();
}

bsl::vector<char>& FeatureTestMessage::makeSelection2()
{
    if (SELECTION_ID_SELECTION2 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_selection2.object());
    }
    else {
        reset();
        new (d_selection2.buffer())
                bsl::vector<char>(d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION2;
    }

    return d_selection2.object();
}

bsl::vector<char>& FeatureTestMessage::makeSelection2(const bsl::vector<char>& value)
{
    if (SELECTION_ID_SELECTION2 == d_selectionId) {
        d_selection2.object() = value;
    }
    else {
        reset();
        new (d_selection2.buffer())
                bsl::vector<char>(value, d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION2;
    }

    return d_selection2.object();
}

Sequence2& FeatureTestMessage::makeSelection3()
{
    if (SELECTION_ID_SELECTION3 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_selection3.object());
    }
    else {
        reset();
        new (d_selection3.buffer())
                Sequence2(d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION3;
    }

    return d_selection3.object();
}

Sequence2& FeatureTestMessage::makeSelection3(const Sequence2& value)
{
    if (SELECTION_ID_SELECTION3 == d_selectionId) {
        d_selection3.object() = value;
    }
    else {
        reset();
        new (d_selection3.buffer())
                Sequence2(value, d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION3;
    }

    return d_selection3.object();
}

Sequence3& FeatureTestMessage::makeSelection4()
{
    if (SELECTION_ID_SELECTION4 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(d_selection4);
    }
    else {
        reset();
        d_selection4 = new (*d_allocator_p)
                Sequence3(d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION4;
    }

    return *d_selection4;
}

Sequence3& FeatureTestMessage::makeSelection4(const Sequence3& value)
{
    if (SELECTION_ID_SELECTION4 == d_selectionId) {
        *d_selection4 = value;
    }
    else {
        reset();
        d_selection4 = new (*d_allocator_p)
                Sequence3(value, d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION4;
    }

    return *d_selection4;
}

bdlt::DatetimeTz& FeatureTestMessage::makeSelection5()
{
    if (SELECTION_ID_SELECTION5 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_selection5.object());
    }
    else {
        reset();
        new (d_selection5.buffer())
            bdlt::DatetimeTz();
        d_selectionId = SELECTION_ID_SELECTION5;
    }

    return d_selection5.object();
}

bdlt::DatetimeTz& FeatureTestMessage::makeSelection5(const bdlt::DatetimeTz& value)
{
    if (SELECTION_ID_SELECTION5 == d_selectionId) {
        d_selection5.object() = value;
    }
    else {
        reset();
        new (d_selection5.buffer())
                bdlt::DatetimeTz(value);
        d_selectionId = SELECTION_ID_SELECTION5;
    }

    return d_selection5.object();
}

CustomString& FeatureTestMessage::makeSelection6()
{
    if (SELECTION_ID_SELECTION6 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_selection6.object());
    }
    else {
        reset();
        new (d_selection6.buffer())
                CustomString(d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION6;
    }

    return d_selection6.object();
}

CustomString& FeatureTestMessage::makeSelection6(const CustomString& value)
{
    if (SELECTION_ID_SELECTION6 == d_selectionId) {
        d_selection6.object() = value;
    }
    else {
        reset();
        new (d_selection6.buffer())
                CustomString(value, d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION6;
    }

    return d_selection6.object();
}

Enumerated::Value& FeatureTestMessage::makeSelection7()
{
    if (SELECTION_ID_SELECTION7 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_selection7.object());
    }
    else {
        reset();
        new (d_selection7.buffer())
                    Enumerated::Value(static_cast<Enumerated::Value>(0));
        d_selectionId = SELECTION_ID_SELECTION7;
    }

    return d_selection7.object();
}

Enumerated::Value& FeatureTestMessage::makeSelection7(Enumerated::Value value)
{
    if (SELECTION_ID_SELECTION7 == d_selectionId) {
        d_selection7.object() = value;
    }
    else {
        reset();
        new (d_selection7.buffer())
                Enumerated::Value(value);
        d_selectionId = SELECTION_ID_SELECTION7;
    }

    return d_selection7.object();
}

Choice3& FeatureTestMessage::makeSelection8()
{
    if (SELECTION_ID_SELECTION8 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_selection8.object());
    }
    else {
        reset();
        new (d_selection8.buffer())
                Choice3(d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION8;
    }

    return d_selection8.object();
}

Choice3& FeatureTestMessage::makeSelection8(const Choice3& value)
{
    if (SELECTION_ID_SELECTION8 == d_selectionId) {
        d_selection8.object() = value;
    }
    else {
        reset();
        new (d_selection8.buffer())
                Choice3(value, d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION8;
    }

    return d_selection8.object();
}

VoidSequence& FeatureTestMessage::makeSelection9()
{
    if (SELECTION_ID_SELECTION9 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_selection9.object());
    }
    else {
        reset();
        new (d_selection9.buffer())
            VoidSequence();
        d_selectionId = SELECTION_ID_SELECTION9;
    }

    return d_selection9.object();
}

VoidSequence& FeatureTestMessage::makeSelection9(const VoidSequence& value)
{
    if (SELECTION_ID_SELECTION9 == d_selectionId) {
        d_selection9.object() = value;
    }
    else {
        reset();
        new (d_selection9.buffer())
                VoidSequence(value);
        d_selectionId = SELECTION_ID_SELECTION9;
    }

    return d_selection9.object();
}

UnsignedSequence& FeatureTestMessage::makeSelection10()
{
    if (SELECTION_ID_SELECTION10 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_selection10.object());
    }
    else {
        reset();
        new (d_selection10.buffer())
            UnsignedSequence();
        d_selectionId = SELECTION_ID_SELECTION10;
    }

    return d_selection10.object();
}

UnsignedSequence& FeatureTestMessage::makeSelection10(const UnsignedSequence& value)
{
    if (SELECTION_ID_SELECTION10 == d_selectionId) {
        d_selection10.object() = value;
    }
    else {
        reset();
        new (d_selection10.buffer())
                UnsignedSequence(value);
        d_selectionId = SELECTION_ID_SELECTION10;
    }

    return d_selection10.object();
}

SequenceWithAnonymity& FeatureTestMessage::makeSelection11()
{
    if (SELECTION_ID_SELECTION11 == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_selection11.object());
    }
    else {
        reset();
        new (d_selection11.buffer())
                SequenceWithAnonymity(d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION11;
    }

    return d_selection11.object();
}

SequenceWithAnonymity& FeatureTestMessage::makeSelection11(const SequenceWithAnonymity& value)
{
    if (SELECTION_ID_SELECTION11 == d_selectionId) {
        d_selection11.object() = value;
    }
    else {
        reset();
        new (d_selection11.buffer())
                SequenceWithAnonymity(value, d_allocator_p);
        d_selectionId = SELECTION_ID_SELECTION11;
    }

    return d_selection11.object();
}

// ACCESSORS

bsl::ostream& FeatureTestMessage::print(
    bsl::ostream& stream,
    int           level,
    int           spacesPerLevel) const
{
    bslim::Printer printer(&stream, level, spacesPerLevel);
    printer.start();
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION1: {
        printer.printAttribute("selection1", d_selection1.object());
      }  break;
      case SELECTION_ID_SELECTION2: {
        bool multilineFlag = (0 <= level);
        bdlb::Print::indent(stream, level + 1, spacesPerLevel);
        stream << (multilineFlag ? "" : " ");
        stream << "selection2 = [ ";
        bdlb::Print::singleLineHexDump(
            stream, d_selection2.object().begin(), d_selection2.object().end());
        stream << " ]" << (multilineFlag ? "\n" : "");
      }  break;
      case SELECTION_ID_SELECTION3: {
        printer.printAttribute("selection3", d_selection3.object());
      }  break;
      case SELECTION_ID_SELECTION4: {
    printer.printAttribute("selection4", *d_selection4);
      }  break;
      case SELECTION_ID_SELECTION5: {
        printer.printAttribute("selection5", d_selection5.object());
      }  break;
      case SELECTION_ID_SELECTION6: {
        printer.printAttribute("selection6", d_selection6.object());
      }  break;
      case SELECTION_ID_SELECTION7: {
        printer.printAttribute("selection7", d_selection7.object());
      }  break;
      case SELECTION_ID_SELECTION8: {
        printer.printAttribute("selection8", d_selection8.object());
      }  break;
      case SELECTION_ID_SELECTION9: {
        printer.printAttribute("selection9", d_selection9.object());
      }  break;
      case SELECTION_ID_SELECTION10: {
        printer.printAttribute("selection10", d_selection10.object());
      }  break;
      case SELECTION_ID_SELECTION11: {
        printer.printAttribute("selection11", d_selection11.object());
      }  break;
      default:
        stream << "SELECTION UNDEFINED\n";
    }
    printer.end();
    return stream;
}


const char *FeatureTestMessage::selectionName() const
{
    switch (d_selectionId) {
      case SELECTION_ID_SELECTION1:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION1].name();
      case SELECTION_ID_SELECTION2:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION2].name();
      case SELECTION_ID_SELECTION3:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION3].name();
      case SELECTION_ID_SELECTION4:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION4].name();
      case SELECTION_ID_SELECTION5:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION5].name();
      case SELECTION_ID_SELECTION6:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION6].name();
      case SELECTION_ID_SELECTION7:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION7].name();
      case SELECTION_ID_SELECTION8:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION8].name();
      case SELECTION_ID_SELECTION9:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION9].name();
      case SELECTION_ID_SELECTION10:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION10].name();
      case SELECTION_ID_SELECTION11:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SELECTION11].name();
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
        return "(* UNDEFINED *)";
    }
}

                               // -------------
                               // class Request
                               // -------------

// CONSTANTS

const char Request::CLASS_NAME[] = "Request";

const bdlat_SelectionInfo Request::SELECTION_INFO_ARRAY[] = {
    {
        SELECTION_ID_SIMPLE_REQUEST,
        "simpleRequest",
        sizeof("simpleRequest") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    },
    {
        SELECTION_ID_FEATURE_REQUEST,
        "featureRequest",
        sizeof("featureRequest") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    }
};

// CLASS METHODS

const bdlat_SelectionInfo *Request::lookupSelectionInfo(
        const char *name,
        int         nameLength)
{
    for (int i = 0; i < 2; ++i) {
        const bdlat_SelectionInfo& selectionInfo =
                    Request::SELECTION_INFO_ARRAY[i];

        if (nameLength == selectionInfo.d_nameLength
        &&  0 == bsl::memcmp(selectionInfo.d_name_p, name, nameLength))
        {
            return &selectionInfo;
        }
    }

    return 0;
}

const bdlat_SelectionInfo *Request::lookupSelectionInfo(int id)
{
    switch (id) {
      case SELECTION_ID_SIMPLE_REQUEST:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_SIMPLE_REQUEST];
      case SELECTION_ID_FEATURE_REQUEST:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_FEATURE_REQUEST];
      default:
        return 0;
    }
}

// CREATORS

Request::Request(
    const Request& original,
    bslma::Allocator *basicAllocator)
: d_selectionId(original.d_selectionId)
, d_allocator_p(bslma::Default::allocator(basicAllocator))
{
    switch (d_selectionId) {
      case SELECTION_ID_SIMPLE_REQUEST: {
        new (d_simpleRequest.buffer())
            SimpleRequest(
                original.d_simpleRequest.object(), d_allocator_p);
      } break;
      case SELECTION_ID_FEATURE_REQUEST: {
        new (d_featureRequest.buffer())
            FeatureTestMessage(
                original.d_featureRequest.object(), d_allocator_p);
      } break;
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
    }
}

// MANIPULATORS

Request&
Request::operator=(const Request& rhs)
{
    if (this != &rhs) {
        switch (rhs.d_selectionId) {
          case SELECTION_ID_SIMPLE_REQUEST: {
            makeSimpleRequest(rhs.d_simpleRequest.object());
          } break;
          case SELECTION_ID_FEATURE_REQUEST: {
            makeFeatureRequest(rhs.d_featureRequest.object());
          } break;
          default:
            BSLS_ASSERT(SELECTION_ID_UNDEFINED == rhs.d_selectionId);
            reset();
        }
    }

    return *this;
}

void Request::reset()
{
    switch (d_selectionId) {
      case SELECTION_ID_SIMPLE_REQUEST: {
        d_simpleRequest.object().~SimpleRequest();
      } break;
      case SELECTION_ID_FEATURE_REQUEST: {
        d_featureRequest.object().~FeatureTestMessage();
      } break;
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
    }

    d_selectionId = SELECTION_ID_UNDEFINED;
}

int Request::makeSelection(int selectionId)
{
    switch (selectionId) {
      case SELECTION_ID_SIMPLE_REQUEST: {
        makeSimpleRequest();
      } break;
      case SELECTION_ID_FEATURE_REQUEST: {
        makeFeatureRequest();
      } break;
      case SELECTION_ID_UNDEFINED: {
        reset();
      } break;
      default:
        return -1;
    }
    return 0;
}

int Request::makeSelection(const char *name, int nameLength)
{
    const bdlat_SelectionInfo *selectionInfo =
                                         lookupSelectionInfo(name, nameLength);
    if (0 == selectionInfo) {
       return -1;
    }

    return makeSelection(selectionInfo->d_id);
}

SimpleRequest& Request::makeSimpleRequest()
{
    if (SELECTION_ID_SIMPLE_REQUEST == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_simpleRequest.object());
    }
    else {
        reset();
        new (d_simpleRequest.buffer())
                SimpleRequest(d_allocator_p);
        d_selectionId = SELECTION_ID_SIMPLE_REQUEST;
    }

    return d_simpleRequest.object();
}

SimpleRequest& Request::makeSimpleRequest(const SimpleRequest& value)
{
    if (SELECTION_ID_SIMPLE_REQUEST == d_selectionId) {
        d_simpleRequest.object() = value;
    }
    else {
        reset();
        new (d_simpleRequest.buffer())
                SimpleRequest(value, d_allocator_p);
        d_selectionId = SELECTION_ID_SIMPLE_REQUEST;
    }

    return d_simpleRequest.object();
}

FeatureTestMessage& Request::makeFeatureRequest()
{
    if (SELECTION_ID_FEATURE_REQUEST == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_featureRequest.object());
    }
    else {
        reset();
        new (d_featureRequest.buffer())
                FeatureTestMessage(d_allocator_p);
        d_selectionId = SELECTION_ID_FEATURE_REQUEST;
    }

    return d_featureRequest.object();
}

FeatureTestMessage& Request::makeFeatureRequest(const FeatureTestMessage& value)
{
    if (SELECTION_ID_FEATURE_REQUEST == d_selectionId) {
        d_featureRequest.object() = value;
    }
    else {
        reset();
        new (d_featureRequest.buffer())
                FeatureTestMessage(value, d_allocator_p);
        d_selectionId = SELECTION_ID_FEATURE_REQUEST;
    }

    return d_featureRequest.object();
}

// ACCESSORS

bsl::ostream& Request::print(
    bsl::ostream& stream,
    int           level,
    int           spacesPerLevel) const
{
    bslim::Printer printer(&stream, level, spacesPerLevel);
    printer.start();
    switch (d_selectionId) {
      case SELECTION_ID_SIMPLE_REQUEST: {
        printer.printAttribute("simpleRequest", d_simpleRequest.object());
      }  break;
      case SELECTION_ID_FEATURE_REQUEST: {
        printer.printAttribute("featureRequest", d_featureRequest.object());
      }  break;
      default:
        stream << "SELECTION UNDEFINED\n";
    }
    printer.end();
    return stream;
}


const char *Request::selectionName() const
{
    switch (d_selectionId) {
      case SELECTION_ID_SIMPLE_REQUEST:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_SIMPLE_REQUEST].name();
      case SELECTION_ID_FEATURE_REQUEST:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_FEATURE_REQUEST].name();
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
        return "(* UNDEFINED *)";
    }
}

                               // --------------
                               // class Response
                               // --------------

// CONSTANTS

const char Response::CLASS_NAME[] = "Response";

const bdlat_SelectionInfo Response::SELECTION_INFO_ARRAY[] = {
    {
        SELECTION_ID_RESPONSE_DATA,
        "responseData",
        sizeof("responseData") - 1,
        "",
        bdlat_FormattingMode::e_TEXT
    },
    {
        SELECTION_ID_FEATURE_RESPONSE,
        "featureResponse",
        sizeof("featureResponse") - 1,
        "",
        bdlat_FormattingMode::e_DEFAULT
    }
};

// CLASS METHODS

const bdlat_SelectionInfo *Response::lookupSelectionInfo(
        const char *name,
        int         nameLength)
{
    for (int i = 0; i < 2; ++i) {
        const bdlat_SelectionInfo& selectionInfo =
                    Response::SELECTION_INFO_ARRAY[i];

        if (nameLength == selectionInfo.d_nameLength
        &&  0 == bsl::memcmp(selectionInfo.d_name_p, name, nameLength))
        {
            return &selectionInfo;
        }
    }

    return 0;
}

const bdlat_SelectionInfo *Response::lookupSelectionInfo(int id)
{
    switch (id) {
      case SELECTION_ID_RESPONSE_DATA:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_RESPONSE_DATA];
      case SELECTION_ID_FEATURE_RESPONSE:
        return &SELECTION_INFO_ARRAY[SELECTION_INDEX_FEATURE_RESPONSE];
      default:
        return 0;
    }
}

// CREATORS

Response::Response(
    const Response& original,
    bslma::Allocator *basicAllocator)
: d_selectionId(original.d_selectionId)
, d_allocator_p(bslma::Default::allocator(basicAllocator))
{
    switch (d_selectionId) {
      case SELECTION_ID_RESPONSE_DATA: {
        new (d_responseData.buffer())
            bsl::string(
                original.d_responseData.object(), d_allocator_p);
      } break;
      case SELECTION_ID_FEATURE_RESPONSE: {
        new (d_featureResponse.buffer())
            FeatureTestMessage(
                original.d_featureResponse.object(), d_allocator_p);
      } break;
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
    }
}

// MANIPULATORS

Response&
Response::operator=(const Response& rhs)
{
    if (this != &rhs) {
        switch (rhs.d_selectionId) {
          case SELECTION_ID_RESPONSE_DATA: {
            makeResponseData(rhs.d_responseData.object());
          } break;
          case SELECTION_ID_FEATURE_RESPONSE: {
            makeFeatureResponse(rhs.d_featureResponse.object());
          } break;
          default:
            BSLS_ASSERT(SELECTION_ID_UNDEFINED == rhs.d_selectionId);
            reset();
        }
    }

    return *this;
}

void Response::reset()
{
    switch (d_selectionId) {
      case SELECTION_ID_RESPONSE_DATA: {
        typedef bsl::string Type;
        d_responseData.object().~Type();
      } break;
      case SELECTION_ID_FEATURE_RESPONSE: {
        d_featureResponse.object().~FeatureTestMessage();
      } break;
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
    }

    d_selectionId = SELECTION_ID_UNDEFINED;
}

int Response::makeSelection(int selectionId)
{
    switch (selectionId) {
      case SELECTION_ID_RESPONSE_DATA: {
        makeResponseData();
      } break;
      case SELECTION_ID_FEATURE_RESPONSE: {
        makeFeatureResponse();
      } break;
      case SELECTION_ID_UNDEFINED: {
        reset();
      } break;
      default:
        return -1;
    }
    return 0;
}

int Response::makeSelection(const char *name, int nameLength)
{
    const bdlat_SelectionInfo *selectionInfo =
                                         lookupSelectionInfo(name, nameLength);
    if (0 == selectionInfo) {
       return -1;
    }

    return makeSelection(selectionInfo->d_id);
}

bsl::string& Response::makeResponseData()
{
    if (SELECTION_ID_RESPONSE_DATA == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_responseData.object());
    }
    else {
        reset();
        new (d_responseData.buffer())
                bsl::string(d_allocator_p);
        d_selectionId = SELECTION_ID_RESPONSE_DATA;
    }

    return d_responseData.object();
}

bsl::string& Response::makeResponseData(const bsl::string& value)
{
    if (SELECTION_ID_RESPONSE_DATA == d_selectionId) {
        d_responseData.object() = value;
    }
    else {
        reset();
        new (d_responseData.buffer())
                bsl::string(value, d_allocator_p);
        d_selectionId = SELECTION_ID_RESPONSE_DATA;
    }

    return d_responseData.object();
}

FeatureTestMessage& Response::makeFeatureResponse()
{
    if (SELECTION_ID_FEATURE_RESPONSE == d_selectionId) {
        bdlat_ValueTypeFunctions::reset(&d_featureResponse.object());
    }
    else {
        reset();
        new (d_featureResponse.buffer())
                FeatureTestMessage(d_allocator_p);
        d_selectionId = SELECTION_ID_FEATURE_RESPONSE;
    }

    return d_featureResponse.object();
}

FeatureTestMessage& Response::makeFeatureResponse(const FeatureTestMessage& value)
{
    if (SELECTION_ID_FEATURE_RESPONSE == d_selectionId) {
        d_featureResponse.object() = value;
    }
    else {
        reset();
        new (d_featureResponse.buffer())
                FeatureTestMessage(value, d_allocator_p);
        d_selectionId = SELECTION_ID_FEATURE_RESPONSE;
    }

    return d_featureResponse.object();
}

// ACCESSORS

bsl::ostream& Response::print(
    bsl::ostream& stream,
    int           level,
    int           spacesPerLevel) const
{
    bslim::Printer printer(&stream, level, spacesPerLevel);
    printer.start();
    switch (d_selectionId) {
      case SELECTION_ID_RESPONSE_DATA: {
        printer.printAttribute("responseData", d_responseData.object());
      }  break;
      case SELECTION_ID_FEATURE_RESPONSE: {
        printer.printAttribute("featureResponse", d_featureResponse.object());
      }  break;
      default:
        stream << "SELECTION UNDEFINED\n";
    }
    printer.end();
    return stream;
}


const char *Response::selectionName() const
{
    switch (d_selectionId) {
      case SELECTION_ID_RESPONSE_DATA:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_RESPONSE_DATA].name();
      case SELECTION_ID_FEATURE_RESPONSE:
        return SELECTION_INFO_ARRAY[SELECTION_INDEX_FEATURE_RESPONSE].name();
      default:
        BSLS_ASSERT(SELECTION_ID_UNDEFINED == d_selectionId);
        return "(* UNDEFINED *)";
    }
}
}  // close package namespace
}  // close enterprise namespace

// GENERATED BY BLP_BAS_CODEGEN_2018.06.17.1 Wed Jun 27 13:12:23 2018
// USING bas_codegen.pl -mmsg -Ctestmessages --noAggregateConversion balb_testmessages.xsd
// ----------------------------------------------------------------------------
// NOTICE:
//      Copyright (C) Bloomberg L.P., 2018
//      All Rights Reserved.
//      Property of Bloomberg L.P. (BLP)
//      This software is made available solely pursuant to the
//      terms of a BLP license agreement which governs its use.
// ------------------------------ END-OF-FILE ---------------------------------
