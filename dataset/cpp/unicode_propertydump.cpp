// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/fastlib/text/unicodeutil.h>

int main(int argc, char **argv) {
    for (ucs4_t testchar = 0; testchar < 0x10000; testchar++) {
        printf("%08x %04x\n", testchar, Fast_UnicodeUtil::GetProperty(testchar));
    }
    return 0;
}
