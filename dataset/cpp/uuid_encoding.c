#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <traildb.h>
#include "tdb_test.h"

int main(int argc, char** argv)
{
    for ( int i1 = 0; i1 < 10000; ++i1 )
    {
        uint8_t hex_uuid[33];
        uint8_t hex_uuid2[33];
        uint8_t uuid[16];
        uint8_t uuid2[16];

        for ( int i2 = 0; i2 < 16; ++i2 ) {
            uuid[i2] = rand();
        }
        tdb_uuid_hex(uuid, hex_uuid);
        tdb_uuid_raw(hex_uuid, uuid2);
        assert(!memcmp(uuid, uuid2, 16));

        for ( int i2 = 0; i2 < 32; ++i2 ) {
            hex_uuid[i2] = rand() % 16;
            hex_uuid[i2] += '0';
            if ( hex_uuid[i2] > '9' ) {
                hex_uuid[i2] -= ('9'+1);
                hex_uuid[i2] += 'a';
            }
        }
        tdb_uuid_raw(hex_uuid, uuid);
        tdb_uuid_hex(uuid, hex_uuid2);
        assert(!memcmp(hex_uuid, hex_uuid2, 32));
    }
    return 0;
}

