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

#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include "bmdb.h"
#include "string.h"
#include "test.h"
#include "CuTest.h"

/*
Contains unit tests for the bmdb config module.
*/

static void testConfigDump(CuTest *tc){
    executeSql("delete from config;", NULL);
    addConfigRow("key1", "value1");
    addConfigRow("key2", "value2");

    FILE* f = makeTmpFileStream();
    doListConfig(f, 0, 0);
    char* result = readTmpFile();

    CuAssertStrEquals(tc, INFO_DUMPING_CONFIG EOL "key1=value1" EOL "key2=value2" EOL, result);
    populateConfigTable();
}

static void testConfigUpdate(CuTest *tc){
    executeSql("delete from config;", NULL);
    addConfigRow("key1", "value1");
    addConfigRow("key2", "value2");

	char* args[] = {"key1", "value3"};
	doSetConfig(NULL, 2, args);

    FILE* f = makeTmpFileStream();
    doListConfig(f, 0, 0);
    char* result = readTmpFile();

    CuAssertStrEquals(tc, INFO_DUMPING_CONFIG EOL "key1=value3" EOL "key2=value2" EOL, result);
    populateConfigTable();
}

static void testConfigDelete(CuTest *tc){
    executeSql("delete from config;", NULL);
    addConfigRow("key1", "value1");
    addConfigRow("key2", "value2");

	char* args[] = {"key1"};
	doRmConfig(NULL, 1, args);

    FILE* f = makeTmpFileStream();
    doListConfig(f, 0, 0);
    char* result = readTmpFile();

    CuAssertStrEquals(tc, INFO_DUMPING_CONFIG EOL "key2=value2" EOL, result);
    populateConfigTable();
}

CuSuite* bmdbConfigGetSuite() {
    CuSuite* suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, testConfigDump);
    SUITE_ADD_TEST(suite, testConfigUpdate);
    SUITE_ADD_TEST(suite, testConfigDelete);
    return suite;
}
