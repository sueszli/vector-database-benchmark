/*
 * BitMeterOS
 * http://codebox.org.uk/bitmeterOS
 *
 * Copyright (c) 2011 Rob Dawson
 *
 * Licensed under the GNU General Public License
 * http://www.gnu.org/licenses/gpl.txt
 *
 * This file is part of BitMeterOS.
 *
 * BitMeterOS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * BitMeterOS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with BitMeterOS.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "common.h"
#include "test.h"
#include "client.h"
#include "CuTest.h"

/*
Contains unit tests for the clientDump utility.
*/

static void onDumpRow(int ignored, struct Data* data);
struct Data* dumpResult;

void testDumpEmptyDb(CuTest *tc) {
 // Check that we behave correctly if the data table is empty
    emptyDb();
    dumpResult = NULL;
    getDumpValues(0, &onDumpRow);
    CuAssertTrue(tc, dumpResult == NULL);
}

void testDumpOneEntry(CuTest *tc) {
 // Check the results if we only have single row in the table
    emptyDb();
    dumpResult = NULL;

    addDbRow(1234, 1, "eth0", 1, 2, "");
    getDumpValues(0, &onDumpRow);
    checkData(tc, dumpResult, 1234, 1, "eth0", 1, 2, "");

    dumpResult = dumpResult->next;
    CuAssertTrue(tc, dumpResult == NULL);
}

void testDumpMultipleEntries(CuTest *tc) {
 // Check the results if we have multiple rows in the table
    emptyDb();
    dumpResult = NULL;

    addDbRow(1233, 1, "eth0", 1, 2, "");
    addDbRow(1234, 2, "eth1", 2, 3, "");
    addDbRow(1235, 3, "eth0", 3, 4, "");

    getDumpValues(0, &onDumpRow);

    checkData(tc, dumpResult, 1235, 3, "eth0", 3, 4, "");

	dumpResult = dumpResult->next;
	checkData(tc, dumpResult, 1234, 2, "eth1", 2, 3, "");

	dumpResult = dumpResult->next;
	checkData(tc, dumpResult, 1233, 1, "eth0", 1, 2, "");

    dumpResult = dumpResult->next;
    CuAssertTrue(tc, dumpResult == NULL);
}

static void onDumpRow(int ignored, struct Data* data) {
 // Helper callback function used by tests to record each Data struct that is returned
    appendData(&dumpResult, data);
}

CuSuite* clientDumpGetSuite() {
    CuSuite* suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, testDumpEmptyDb);
    SUITE_ADD_TEST(suite, testDumpOneEntry);
    SUITE_ADD_TEST(suite, testDumpMultipleEntries);
    return suite;
}
