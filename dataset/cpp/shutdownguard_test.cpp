// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#include <vespa/vespalib/testkit/testapp.h>
#include <vespa/vespalib/util/shutdownguard.h>
#include <vespa/vespalib/util/malloc_mmap_guard.h>
#include <thread>
#include <unistd.h>
#include <sys/wait.h>
#include <cstdlib>

using namespace vespalib;

TEST("test shutdown guard")
{
    {
        ShutdownGuard farFuture(1000000s);
        std::this_thread::sleep_for(20ms);
    }
    EXPECT_TRUE(true);
    pid_t child = fork();
    if (child == 0) {
        ShutdownGuard soon(30ms);
        for (int i = 0; i < 1000; ++i) {
            std::this_thread::sleep_for(20ms);
        }
        std::_Exit(0);
    }
    for (int i = 0; i < 1000; ++i) {
        std::this_thread::sleep_for(20ms);
        int stat = 0;
        if (waitpid(child, &stat, WNOHANG) == child) {
            EXPECT_TRUE(WIFEXITED(stat));
            EXPECT_EQUAL(1, WEXITSTATUS(stat));
            break;
        }
        EXPECT_TRUE(i < 800);
    }
}

TEST("test malloc mmap guard") {
    MallocMmapGuard guard(0x100000);
}

TEST_MAIN() { TEST_RUN_ALL(); }
