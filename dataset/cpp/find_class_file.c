/**
 *
 * Phantom OS
 *
 * Copyright (C) 2005-2012 Dmitry Zavalishin, dz@dz.ru
 *
 * Load class from filesystem file.
 *
 * Strict Phantom kernel environment works withount file system,
 * but debug env lets us access host file system from within Phantom.
 * It is handy to have ability to load class files from there.
 *
**/


#define DEBUG_MSG_PREFIX "vm.fcf"
#include <debug_ext.h>
#define debug_level_flow 10
#define debug_level_error 10
#define debug_level_info 10

#include <vm/bulk.h>
#include "main.h"


#define DEBUG 0

static int do_load_class_from_file(const char *fn, pvm_object_t *out)
{
    void *code;
    unsigned int size;
    int rc = load_code( &code, &size, fn);

    if(rc)
        return rc;

    //pvm_object_t out;
    rc = pvm_load_class_from_memory( code, size, out );

    free(code);
    return rc;
}


int load_class_from_file(const char *cn, pvm_object_t *out)
{
    char * have_suffix = (char *)strstr( cn, ".pc" );

    if(*cn == '.') cn++;

    // TODO check all automounts?
    char *path[] =
    {
        ".", // Reserved for getenv search

//        ".",
        "/amnt0/class",
        "/amnt1/class",
        "/amnt2/class",
        "/amnt3/class",
        "../../plib/bin",
//        "../compiler",
//        "./pcode",
        0
    };

/*
    char *dir = getenv("PHANTOM_HOME");
    char *rest = "plib/bin";

    if( dir == NULL )
    {
        dir = "pcode";
        rest = "classes";
    }

    char fn[1024];
    snprintf( fn, 1024, "%s/%s", dir, rest );
    path[0] = fn;
*/

#define BS 1024
    char       buf[BS+1];

    char **prefix;
    for( prefix = path; *prefix; prefix++ )
    {
        snprintf( buf, BS, "%s/%s%s", *prefix, cn, have_suffix ? "" : ".pc" );

        //printf("try '%s'\n", buf );
        if(!do_load_class_from_file(buf, out))
        {
            if(DEBUG) printf("OK: File found for class '%s'\n", cn );
            return 0;
        }
    }

    if(DEBUG) printf("ERR: File not found for class '%s'\n", cn );

    return 1;
}


//phantom_is_a_real_kernel()
