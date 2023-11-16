/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include <open62541/plugin/log_stdout.h>
#include <open62541/client_config_default.h>
#include <open62541/client_highlevel.h>
#include <check.h>
#include <stdio.h>
#include <stdlib.h>
#include "thread_wrapper.h"
#include "test_helpers.h"
#include "deviceObjectType.h"

#define NUMBER_OF_WORKERS 10
#define ITERATIONS_PER_WORKER 10
#define NUMBER_OF_CLIENTS 10
#define ITERATIONS_PER_CLIENT 10

static void setup(void) {
    tc.running = true;
    tc.server = UA_Server_newForUnitTest();
    ck_assert(tc.server != NULL);
    defineObjectTypes();
    addPumpTypeConstructor(tc.server);
    UA_Server_run_startup(tc.server);
    THREAD_CREATE(server_thread, serverloop);
}

static
void checkServer(void) {
    for (size_t i = 0; i < NUMBER_OF_WORKERS * ITERATIONS_PER_WORKER; i++) {
        char string_buf[20];
        snprintf(string_buf, sizeof(string_buf), "Server %u", (unsigned)i);
        UA_NodeId reqNodeId = UA_NODEID_STRING(1, string_buf);
        UA_NodeId resNodeId;
        UA_StatusCode ret = UA_Server_readNodeId(tc.server, reqNodeId, &resNodeId);

        ck_assert_int_eq(ret, UA_STATUSCODE_GOOD);
        ck_assert(UA_NodeId_equal(&reqNodeId, &resNodeId) == UA_TRUE);
        UA_NodeId_clear(&resNodeId);
    }

    for (size_t i = 0; i < NUMBER_OF_CLIENTS * ITERATIONS_PER_CLIENT; i++) {
        char string_buf[20];
        snprintf(string_buf, sizeof(string_buf), "Client %u", (unsigned)i);
        UA_NodeId reqNodeId = UA_NODEID_STRING(1, string_buf);
        UA_NodeId resNodeId;
        UA_StatusCode ret = UA_Server_readNodeId(tc.server, reqNodeId, &resNodeId);

        ck_assert_int_eq(ret, UA_STATUSCODE_GOOD);
        ck_assert(UA_NodeId_equal(&reqNodeId, &resNodeId) == UA_TRUE);
        UA_NodeId_clear(&resNodeId);
    }
}

static
void server_addObject(void* value) {
    ThreadContext tmp = (*(ThreadContext *) value);
    size_t offset = tmp.index * tmp.upperBound;
    size_t number = offset + tmp.counter;
    char string_buf[20];
    snprintf(string_buf, sizeof(string_buf), "Server %u", (unsigned)number);
    UA_NodeId reqId = UA_NODEID_STRING(1, string_buf);
    UA_ObjectAttributes oAttr = UA_ObjectAttributes_default;
    oAttr.displayName = UA_LOCALIZEDTEXT("en-US", string_buf);
    UA_StatusCode res = UA_Server_addObjectNode(tc.server, reqId,
                                UA_NODEID_NUMERIC(0, UA_NS0ID_OBJECTSFOLDER),
                                UA_NODEID_NUMERIC(0, UA_NS0ID_ORGANIZES),
                                UA_QUALIFIEDNAME(1, string_buf),
                                pumpTypeId,
                                oAttr, NULL, NULL);

    ck_assert_int_eq(UA_STATUSCODE_GOOD, res);
}

static
void client_addObject(void* value){
    ThreadContext tmp = (*(ThreadContext *) value);
    size_t offset = tmp.index * tmp.upperBound;
    size_t number = offset + tmp.counter;
    char string_buf[20];
    snprintf(string_buf, sizeof(string_buf), "Client %u", (unsigned)number);
    UA_NodeId reqId = UA_NODEID_STRING(1, string_buf);
    UA_ObjectAttributes oAttr = UA_ObjectAttributes_default;
    oAttr.displayName = UA_LOCALIZEDTEXT("en-US", string_buf);
    UA_StatusCode retval = UA_Client_addObjectNode(tc.clients[tmp.index], reqId,
                                         UA_NODEID_NUMERIC(0, UA_NS0ID_OBJECTSFOLDER),
                                         UA_NODEID_NUMERIC(0, UA_NS0ID_ORGANIZES),
                                         UA_QUALIFIEDNAME(1, string_buf), pumpTypeId, oAttr, NULL);

    ck_assert_uint_eq(retval, UA_STATUSCODE_GOOD);
}

static
void initTest(void) {
    for (size_t i = 0; i < tc.numberOfWorkers; i++) {
        setThreadContext(&tc.workerContext[i], i, ITERATIONS_PER_WORKER, server_addObject);
    }

    for (size_t i = 0; i < tc.numberofClients; i++) {
        setThreadContext(&tc.clientContext[i], i, ITERATIONS_PER_CLIENT, client_addObject);
    }
}

START_TEST(addObjectTypeNodes) {
        startMultithreading();
    }
END_TEST


static Suite* testSuite_immutableNodes(void) {
    Suite *s = suite_create("Multithreading");
    TCase *valueCallback = tcase_create("Add object nodes");
    tcase_add_checked_fixture(valueCallback, setup, teardown);
    tcase_add_test(valueCallback, addObjectTypeNodes);
    suite_add_tcase(s,valueCallback);
    return s;
}

int main(void) {
    Suite *s = testSuite_immutableNodes();
    SRunner *sr = srunner_create(s);
    srunner_set_fork_status(sr, CK_NOFORK);

    createThreadContext(NUMBER_OF_WORKERS, NUMBER_OF_CLIENTS, checkServer);
    initTest();
    srunner_run_all(sr, CK_NORMAL);
    deleteThreadContext();

    int number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
