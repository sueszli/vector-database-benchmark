#include <stdlib.h>
#include <assert.h>

#include <traildb.h>
#include "tdb_test.h"

int main(int argc, char** argv)
{
    const char *fields[] = {};
    tdb_cons* c = tdb_cons_init();
    test_cons_settings(c);
    assert(tdb_cons_open(c, getenv("TDB_TMP_DIR"), fields, 0) == 0);
    assert(tdb_cons_finalize(c) == 0);
    tdb_cons_close(c);

    tdb* t = tdb_init();
    assert(tdb_open(t, getenv("TDB_TMP_DIR")) == 0);
    assert(tdb_version(t) == TDB_VERSION_LATEST);
    tdb_close(t);

    return 0;
}
