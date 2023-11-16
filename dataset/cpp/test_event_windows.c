/***************************************************************************
 *   Copyright (C) 2022 by Kyle Hayes  & @Joylei                           *
 *   Author Kyle Hayes  kyle.hayes@gmail.com, @Joylei                      *
 *                                                                         *
 * This software is available under either the Mozilla Public License      *
 * version 2.0 or the GNU LGPL version 2 (or later) license, whichever     *
 * you choose.                                                             *
 *                                                                         *
 * MPL 2.0:                                                                *
 *                                                                         *
 *   This Source Code Form is subject to the terms of the Mozilla Public   *
 *   License, v. 2.0. If a copy of the MPL was not distributed with this   *
 *   file, You can obtain one at http://mozilla.org/MPL/2.0/.              *
 *                                                                         *
 *                                                                         *
 * LGPL 2:                                                                 *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU Library General Public License as       *
 *   published by the Free Software Foundation; either version 2 of the    *
 *   License, or (at your option) any later version.                       *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU Library General Public     *
 *   License along with this program; if not, write to the                 *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#ifdef _WIN32
#include <Windows.h>
#else
#include <pthread.h>
#include <signal.h>
#endif
#include "../lib/libplctag.h"
#include "utils.h"

#define REQUIRED_VERSION 2, 5, 0

#define TAG_PATH "protocol=ab_eip&gateway=10.206.1.40&path=1,4&cpu=ControlLogix&elem_count=1&name=TestBigArray[%d]"
#define ELEM_COUNT 1
#define ELEM_SIZE 4
#define DATA_TIMEOUT 500

#define MAX_THREADS (20)

typedef struct
{
    volatile int tid;
    volatile int32_t tag;
    volatile HANDLE hEvent;
} tag_state;

/*
 * This test program creates a lot of threads that read the same tag in
 * the plc.  They all hit the exact same underlying tag data structure.
 * This tests, to some extent, whether the library can handle multi-threaded
 * access.
 */

#ifdef _WIN32
volatile int done = 0;

/* straight from MS' web site :-) */
BOOL WINAPI CtrlHandler(DWORD fdwCtrlType)
{
    switch (fdwCtrlType)
    {
        // Handle the CTRL-C signal.
    case CTRL_C_EVENT:
        done = 1;
        return TRUE;

        // CTRL-CLOSE: confirm that the user wants to exit.
    case CTRL_CLOSE_EVENT:
        done = 1;
        return TRUE;

        // Pass other signals to the next handler.
    case CTRL_BREAK_EVENT:
        done = 1;
        return FALSE;

    case CTRL_LOGOFF_EVENT:
        done = 1;
        return FALSE;

    case CTRL_SHUTDOWN_EVENT:
        done = 1;
        return FALSE;

    default:
        return FALSE;
    }
}

void setup_break_handler(void)
{
    if (!SetConsoleCtrlHandler(CtrlHandler, TRUE))
    {
        printf("\nERROR: Could not set control handler!\n");
    }
}

#else
volatile sig_atomic_t done = 0;

void SIGINT_handler(int not_used)
{
    (void)not_used;

    done = 1;
}

void setup_break_handler(void)
{
    struct sigaction act;

    /* set up signal handler. */
    memset(&act, 0, sizeof(act));
    act.sa_handler = SIGINT_handler;
    sigaction(SIGINT, &act, NULL);
}

#endif

static int num_threads = 0;
static tag_state states[MAX_THREADS];

void tag_callback(int32_t tag_id, int event, int status, void *arg)
{
    int tid = (int)(intptr_t)arg;

    if (event != PLCTAG_EVENT_READ_COMPLETED)
    {
        return;
    }
    fprintf(stderr, "callback tag(%d), tag id(%d), event(%d), status(%d)\n", tid, tag_id, event, status);

    SetEvent(states[tid].hEvent);

    /*
    I think this logic is incorrect.  Not a Windows programmer, but I think this wakes all
    waiting threads, including the one for which this callback was called.  Some of those other
    threads may legitimately be waiting for a read to complete.

    for (int i = 0; i < num_threads; i++)
    {
        if (states[i].tag == tag_id)
        {
            SetEvent(states[i].hEvent);
        }
    }

    */
}

/*
 * Thread function.  Just read until killed.
 */

#ifdef _WIN32
DWORD __stdcall thread_func(LPVOID data)
#else
void *thread_func(void *data)
#endif
{
    int rc = PLCTAG_STATUS_OK;
    int tid = (int)(intptr_t)data;
    int value;
    HANDLE hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
    char buf[250] = {
        0,
    };
    snprintf(buf, sizeof(buf), TAG_PATH, tid);
    /* create the tag */
    int tag = plc_tag_create(buf, 0);
    states[tid].tag = tag;
    states[tid].hEvent = hEvent;
    /* everything OK? */
    if (tag < 0)
    {
        fprintf(stderr, "ERROR %s: Could not create tag!\n", plc_tag_decode_error(tag));
        return 0;
    }

    while ((rc = plc_tag_status(tag)) == PLCTAG_STATUS_PENDING)
    {
        if (done)
        {
            break;
        }
        util_sleep_ms(0); // yield cpu
    }

    if (rc != PLCTAG_STATUS_OK)
    {
        fprintf(stderr, "Error setting up tag internal state. %s\n", plc_tag_decode_error(rc));
        plc_tag_destroy(tag);
        return 0;
    }

    /* use extended callback to pass the thread index/id */
    plc_tag_register_callback_ex(tag, tag_callback, (void *)(intptr_t)tid);

    while (!done)
    {
        int64_t start;
        int64_t end;

        /* capture the starting time */
        start = util_time_ms();

        do
        {
            rc = plc_tag_read(tag, 0);
            if (rc < 0)
            {
                fprintf(stderr, "Error setting up tag internal state. %s\n", plc_tag_decode_error(rc));
            }
            if (rc == PLCTAG_STATUS_PENDING)
            {
                WaitForSingleObject(hEvent, -1);
                if ((rc = plc_tag_status(tag)) == PLCTAG_STATUS_PENDING)
                {
                    fprintf(stderr, "something is wrong for tag(%d), status(%d)\n", tag, rc);
                    plc_tag_destroy(tag);
                    return 0;
                }
            }
            value = plc_tag_get_int32(tag, 0);
        } while (0);

        end = util_time_ms();

        fprintf(stderr, "Thread %d got result %d with return code %s in %dms\n", tid, value, plc_tag_decode_error(rc), (int)(end - start));

        util_sleep_ms(0); // yield cpu
    }
    plc_tag_destroy(tag);

#ifdef _WIN32
    return (DWORD)0;
#else
    return NULL;
#endif
}

int main(int argc, char **argv)
{

#ifdef _WIN32
    HANDLE thread[MAX_THREADS];
#else
    pthread_t thread[MAX_THREADS];
#endif

    int thread_id = 0;

    /* set up handler for ^C etc. */
    setup_break_handler();

    fprintf(stderr, "Hit ^C to terminate the test.\n");

    /* check the library version. */
    if (plc_tag_check_lib_version(REQUIRED_VERSION) != PLCTAG_STATUS_OK)
    {
        fprintf(stderr, "Required compatible library version %d.%d.%d not available!", REQUIRED_VERSION);
        exit(1);
    }

    if (argc != 2)
    {
        fprintf(stderr, "ERROR: Must provide number of threads to run (between 1 and 300) argc=%d!\n", argc);
        return 0;
    }

    plc_tag_set_debug_level(PLCTAG_DEBUG_DETAIL);

    num_threads = (int)strtol(argv[1], NULL, 10);

    if (num_threads < 1 || num_threads > MAX_THREADS)
    {
        fprintf(stderr, "ERROR: %d (%s) is not a valid number. Must provide number of threads to run (between 1 and 300)!\n", num_threads, argv[1]);
        return 0;
    }

    /* create the read threads */
    fprintf(stderr, "Creating %d threads.\n", num_threads);

    for (thread_id = 0; thread_id < num_threads; thread_id++)
    {
#ifdef _WIN32
        thread[thread_id] = CreateThread(
            NULL,                        /* default security attributes */
            0,                           /* use default stack size      */
            thread_func,                 /* thread function             */
            (void *)(intptr_t)thread_id, /* argument to thread function */
            (DWORD)0,                    /* use default creation flags  */
            (LPDWORD)NULL);              /* do not need thread ID       */
#else
        pthread_create(&thread[thread_id], NULL, thread_func, (void *)(intptr_t)thread_id);
#endif
    }

    /* wait until ^C */
    while (!done)
    {
        util_sleep_ms(100);
    }

    for (thread_id = 0; thread_id < num_threads; thread_id++)
    {
#ifdef _WIN32
        WaitForSingleObject(thread[thread_id], (DWORD)INFINITE);
#else
        pthread_join(thread[thread_id], NULL);
#endif
    }
    return 0;
}
