//
//
// Copyright (C) 2007 Pingtel Corp., certain elements licensed under a Contributor Agreement.
// Contributors retain copyright to elements licensed under a Contributor Agreement.
// Licensed to the User under the LGPL license.
//
// $$
//////////////////////////////////////////////////////////////////////////////

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestCase.h>
#include <sipxunit/TestUtilities.h>

#include "net/Url.h"
#include "sipdb/ResultSet.h"

typedef struct
{
  const char* uri;
  const char* callid;
  const char* contact;
  const char* expires;
  const char* cseq;
  const char* qvalue;
  const char* instance_id;
  const char* gruu;
  const char* path;
} RegistrationRow;

RegistrationRow regdata[] =
{
  {
    "uri",
    "callid",
    "contact",
    "expires",
    "cseq",
    "qvalue",
    "instance_id",
    "gruu",
    "path"
  },
  {
    "sip:user1@example.com",
    "6745637808245563@TmVhbC1sYXB0b3Ay",
    "sip:181@192.168.0.2:6012",
    "1133218054",
    "3",
    "",
    "1111",
    "sip:181@example.com;gr",
    "<sip:visitme.com>,<sip:overhere.com>,<sip:comemyway.com>"
  },
  {
    "sip:user@example.com",
    "8d2d9c70405f4e66@TmVhbC1sYXB0b3Ay",
    "sip:181@66.30.139.170:24907",
    "1133221655",
    "2",
    "0.8",
    "2222",
    "sip:182@example.com;gr",
    "<sip:visitme.com>,<sip:overhere.com>,<sip:comemyway.com>"
  },
  {
    "sip:user3@example.com",
    "fa294244984e0c3f@TmVhbC1sYXB0b3Ay",
    "sip:181@192.168.0.2:6000",
    "1133221680",
    "1",
    "0.2",
    "3333",
    "sip:183@example.com;gr",
    "<sip:visitme.com>,<sip:overhere.com>,<sip:comemyway.com>"
  }
};

class ResultSetTest : public CppUnit::TestCase
{
  CPPUNIT_TEST_SUITE(ResultSetTest);
  CPPUNIT_TEST(testResultSet_AddValueAndGetIndex);
  CPPUNIT_TEST_SUITE_END();

public:

  void addValues(ResultSet& registrations)
  {
    // Row 0 contain the key names. The values begins from row 1.
    size_t row;
    for (row = 1; row < sizeof(regdata)/sizeof(RegistrationRow); row++)
    {
      UtlHashMap regRow;

      UtlString* uriKey = new UtlString(regdata[0].uri);
      UtlString* uriValue = new UtlString(regdata[row].uri);
      regRow.insertKeyAndValue(uriKey, uriValue);

      UtlString* callidKey = new UtlString(regdata[0].callid);
      UtlString* callidValue = new UtlString(regdata[row].callid);
      regRow.insertKeyAndValue(callidKey, callidValue);

      UtlString* contactKey = new UtlString(regdata[0].contact);
      UtlString* contactValue = new UtlString(regdata[row].contact);
      regRow.insertKeyAndValue(contactKey, contactValue);

      UtlString* expiresKey = new UtlString(regdata[0].expires);
      UtlString* expiresValue = new UtlString(regdata[row].expires);
      regRow.insertKeyAndValue(expiresKey, expiresValue);

      UtlString* cseqKey = new UtlString(regdata[0].cseq);
      UtlString* cseqValue = new UtlString(regdata[row].cseq);
      regRow.insertKeyAndValue(cseqKey, cseqValue);

      UtlString* qvalueKey = new UtlString(regdata[0].qvalue);
      UtlString* qvalueValue = new UtlString(regdata[row].qvalue);
      regRow.insertKeyAndValue(qvalueKey, qvalueValue);

      UtlString* instanceIdKey = new UtlString(regdata[0].instance_id);
      UtlString* instanceIdValue = new UtlString(regdata[row].instance_id);
      regRow.insertKeyAndValue(instanceIdKey, instanceIdValue);

      UtlString* gruuKey = new UtlString(regdata[0].gruu);
      UtlString* gruuValue = new UtlString(regdata[row].gruu);
      regRow.insertKeyAndValue(gruuKey, gruuValue);

      UtlString* pathKey = new UtlString(regdata[0].path);
      UtlString* pathValue = new UtlString(regdata[row].path);
      regRow.insertKeyAndValue(pathKey, pathValue);

      registrations.addValue(regRow);
    }
  }

  void testValues(ResultSet& registrations)
  {
    // Row 0 contain the key names. The values begins from row 1.
    for (int row = 1; row < registrations.getSize(); row++)
    {
      UtlHashMap regRow;

      // registrations begins with index 0
      OsStatus status = registrations.getIndex(row - 1, regRow);

      CPPUNIT_ASSERT(status == OS_SUCCESS);

      if (status == OS_SUCCESS)
      {
        UtlString uriKey(regdata[0].uri);
        UtlString* uriValue = (UtlString*)regRow.findValue(&uriKey);

        // TEST: Verify inserted value
        CPPUNIT_ASSERT(*uriValue == UtlString(regdata[row].uri));

        UtlString callidKey(regdata[0].callid);
        UtlString* callidValue = (UtlString*)regRow.findValue(&callidKey);

        // TEST: Verify inserted value
        CPPUNIT_ASSERT(*callidValue == UtlString(regdata[row].callid));

        UtlString contactKey(regdata[0].contact);
        UtlString* contactValue = (UtlString*)regRow.findValue(&contactKey);

        // TEST: Verify inserted value
        CPPUNIT_ASSERT(*contactValue == UtlString(regdata[row].contact));

        UtlString expiresKey(regdata[0].expires);
        UtlString* expiresValue = (UtlString*)regRow.findValue(&expiresKey);

        // TEST: Verify inserted value
        CPPUNIT_ASSERT(*expiresValue == UtlString(regdata[row].expires));

        UtlString cseqKey(regdata[0].cseq);
        UtlString* cseqValue = (UtlString*)regRow.findValue(&cseqKey);

        // TEST: Verify inserted value
        CPPUNIT_ASSERT(*cseqValue == UtlString(regdata[row].cseq));

        UtlString qvalueKey(regdata[0].qvalue);
        UtlString* qvalueValue = (UtlString*)regRow.findValue(&qvalueKey);

        // TEST: Verify inserted value
        CPPUNIT_ASSERT(*qvalueValue == UtlString(regdata[row].qvalue));

        UtlString instanceIdKey(regdata[0].instance_id);
        UtlString* instanceIdValue = (UtlString*)regRow.findValue(&instanceIdKey);

        // TEST: Verify inserted value
        CPPUNIT_ASSERT(*instanceIdValue == UtlString(regdata[row].instance_id));

        UtlString gruuKey(regdata[0].gruu);
        UtlString* gruuValue = (UtlString*)regRow.findValue(&gruuKey);

        // TEST: Verify inserted value
        CPPUNIT_ASSERT(*gruuValue == UtlString(regdata[row].gruu));

        UtlString pathKey(regdata[0].path);
        UtlString* pathValue = (UtlString*)regRow.findValue(&pathKey);

        // TEST: Verify inserted value
        CPPUNIT_ASSERT(*pathValue == UtlString(regdata[row].path));
      }
    }
  }

  void testResultSet_AddValueAndGetIndex()
  {
    ResultSet registrations;

    // add default values
    addValues(registrations);

    // check added values
    testValues(registrations);
  };
};

CPPUNIT_TEST_SUITE_REGISTRATION(ResultSetTest);
