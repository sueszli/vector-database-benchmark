/* SPDX-License-Identifier: LGPL-2.1-or-later */

#include "journald-server.h"
#include "test-tables.h"

int main(int argc, char **argv) {
        test_table(split_mode, SPLIT);
        test_table(storage, STORAGE);

        return EXIT_SUCCESS;
}
