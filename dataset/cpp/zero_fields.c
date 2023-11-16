#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include <traildb.h>
#include "tdb_test.h"

int main(int argc, char** argv)
{
    uint8_t uuid[16];
    const char *fields[] = {};
    tdb_cons* c = tdb_cons_init();
    test_cons_settings(c);
    assert(tdb_cons_open(c, getenv("TDB_TMP_DIR"), fields, 0) == 0);
    uint64_t i, j, cmp, sum = 0;
    uint64_t zero = 0;
    tdb_field field;

    memset(uuid, 0, 16);

    for (i = 0; i < 1000; i++){
        memcpy(uuid, &i, 4);
        for (j = 0; j < 10 + i; j++){
            sum += j;
            assert(tdb_cons_add(c, uuid, j, fields, &zero) == 0);
        }
    }

    assert(tdb_cons_finalize(c) == 0);
    tdb_cons_close(c);

    tdb* t = tdb_init();
    assert(tdb_open(t, getenv("TDB_TMP_DIR")) == 0);
    tdb_cursor *cursor = tdb_cursor_new(t);

    assert(tdb_num_fields(t) == 1);

    assert(tdb_get_field(t, "world", &field) == TDB_ERR_UNKNOWN_FIELD);
    assert(tdb_get_field(t, "hello", &field) == TDB_ERR_UNKNOWN_FIELD);
    assert(tdb_get_field(t, "what", &field) == TDB_ERR_UNKNOWN_FIELD);
    assert(tdb_get_field(t, "is", &field) == TDB_ERR_UNKNOWN_FIELD);
    assert(tdb_get_field(t, "this", &field) == TDB_ERR_UNKNOWN_FIELD);
    assert(tdb_get_field(t, "time", &field) == 0);
    assert(field == 0);
    assert(tdb_get_field(t, "blah", &field) == TDB_ERR_UNKNOWN_FIELD);
    assert(tdb_get_field(t, "bloh", &field) == TDB_ERR_UNKNOWN_FIELD);

    for (cmp = 0, i = 0; i < tdb_num_trails(t); i++){
        const tdb_event *event;
        assert(tdb_get_trail(cursor, i) == 0);
        while ((event = tdb_cursor_next(cursor)))
            cmp += event->timestamp;
    }
    assert(cmp == sum);

    tdb_close(t);
    return 0;
}

