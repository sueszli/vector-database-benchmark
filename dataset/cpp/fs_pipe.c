/**
 *
 * Phantom OS
 *
 * Copyright (C) 2005-2011 Dmitry Zavalishin, dz@dz.ru
 *
 * Pipe FS
 *
 *
**/

#if HAVE_UNIX


#define DEBUG_MSG_PREFIX "pipefs"
#include <debug_ext.h>
#define debug_level_flow 0
#define debug_level_error 10
#define debug_level_info 10

#include <unix/uufile.h>
//#include <unix/uuprocess.h>
#include <malloc.h>
#include <string.h>
#include "device.h"

#include "unix/fs_pipe.h"


#define PIPE_BUFS       1024


// TODO stat

// -----------------------------------------------------------------------
// Generic impl
// -----------------------------------------------------------------------


static size_t      pipe_read(    struct uufile *f, void *dest, size_t bytes);
static size_t      pipe_write(   struct uufile *f, const void *src, size_t bytes);
static size_t      pipe_getpath( struct uufile *f, void *dest, size_t bytes);
static ssize_t     pipe_getsize( struct uufile *f);
static void *      pipe_copyimpl( void *impl );


static struct uufileops pipe_fops =
{
    .read 	= pipe_read,
    .write 	= pipe_write,

    .getpath 	= pipe_getpath,
    .getsize 	= pipe_getsize,

    .copyimpl   = pipe_copyimpl,

    //.stat       = pipe_stat,
    //.ioctl      = pipe_ioctl,
};



// -----------------------------------------------------------------------
// FS struct
// -----------------------------------------------------------------------


static errno_t     pipe_open(struct uufile *, int create, int write);
static errno_t     pipe_close(struct uufile *);
static uufile_t *  pipe_namei(uufs_t *fs, const char *filename);
static uufile_t *  pipe_getRoot(uufs_t *fs);
static errno_t     pipe_dismiss(uufs_t *fs);


struct uufs pipe_fs =
{
    .name       = "pipe",
    .open 	= pipe_open,
    .close 	= pipe_close,
    .namei 	= pipe_namei,
    .root 	= pipe_getRoot,
    .dismiss    = pipe_dismiss,

    .impl       = 0,
};


static struct uufile pipe_root =
{
    .ops 	= &pipe_fops,
    .pos        = 0,
    .fs         = &pipe_fs,
    .name       = "/",
    .flags      = UU_FILE_FLAG_DIR|UU_FILE_FLAG_NODESTROY,
    .impl       = 0, // dir will create if needed
};



// -----------------------------------------------------------------------
// FS methods
// -----------------------------------------------------------------------


static errno_t     pipe_open(struct uufile *f, int create, int write)
{
    (void) create;
    (void) write;
    (void) f;
    return ENOENT;
}


// Create a file struct for given path
static uufile_t *  pipe_namei(uufs_t *fs, const char *filename)
{
    (void) fs;
    (void) filename;

    return 0;
}

// Return a file struct for fs root
static uufile_t *  pipe_getRoot(uufs_t *fs)
{
    (void) fs;
    return &pipe_root;
}


static errno_t     pipe_dismiss(uufs_t *fs)
{
    (void) fs;
    return 0;
}


// -----------------------------------------------------------------------
// Generic impl
// -----------------------------------------------------------------------

//static errno_t     pipe_stat(    struct uufile *f, struct ??);
//static errno_t     pipe_ioctl(   struct uufile *f, struct ??);

static size_t      pipe_getpath( struct uufile *f, void *dest, size_t bytes)
{
    if( bytes == 0 ) return 0;
    // TODO get mountpoint
    strncpy( dest, f->name, bytes );
    return strlen(dest);
}

// returns -1 for non-files
static ssize_t      pipe_getsize( struct uufile *f)
{
    (void) f;

    return -1;
}

static void *      pipe_copyimpl( void *impl )
{
    return impl; // Just a pointer to dev, can copy
}








struct pipe_buf
{
    char        buf[PIPE_BUFS];

    size_t      contains; // n bytes in buf

    size_t	putpos;
    size_t      getpos;

    int         kill; // = 1 to break pipe

    int         ref; // refcount

    hal_cond_t  cond;
    hal_mutex_t mutex;
};


static struct pipe_buf * init_pb(void)
{
    struct pipe_buf     *pb = calloc( 1, sizeof(struct pipe_buf) );
    hal_mutex_init( &pb->mutex, "pipe" );
    hal_cond_init( &pb->cond, "pipe" );
    pb->ref = 2;
    return pb;
}

void pipefs_make_pipe( uufile_t **f1, uufile_t **f2 )
{
    struct pipe_buf *pb1 = init_pb();
    //struct pipe_buf *pb2 = init_pb();

    *f1 = create_uufile();
    *f2 = create_uufile();

    (*f1)->ops = &pipe_fops;
    (*f1)->pos = 0;
    (*f1)->fs = &pipe_fs;
    (*f1)->name = "(unnamed pipe)";
    (*f1)->impl = pb1;
    (*f1)->flags = UU_FILE_FLAG_OPEN|UU_FILE_FLAG_PIPE;

    (*f2)->ops = &pipe_fops;
    (*f2)->pos = 0;
    (*f2)->fs = &pipe_fs;
    (*f2)->name = "(unnamed pipe)";
    (*f2)->impl = pb1;
    (*f2)->flags = UU_FILE_FLAG_OPEN|UU_FILE_FLAG_PIPE;

}





// TODO use pools!
static errno_t     pipe_close(struct uufile *f)
{
    struct pipe_buf *pb = f->impl;

    SHOW_FLOW( 10, "pb @%p", pb );

    assert(pb->ref > 0);
    pb->ref--;
    if(pb->ref == 0)
    {
        hal_mutex_destroy(&pb->mutex);
        hal_cond_destroy(&pb->cond);
        free(pb);
    }
    return 0;
}



// TODO use ssize_t! use cbuf?
static size_t      pipe_read(    struct uufile *f, void *dest, size_t bytes)
{
    struct pipe_buf *pb = f->impl;
    size_t ret = 0;

    if(pb->kill)
        return 0;

    SHOW_FLOW( 10, "pb @%p", pb );

    hal_mutex_lock(&pb->mutex);

    while(pb->contains == 0)
    {
        SHOW_FLOW( 9, "have %d bytes, sleep", pb->contains );
        hal_cond_wait(&pb->cond, &pb->mutex);
        if(pb->kill)
            goto ret;
    }

    if( bytes > pb->contains )
        bytes = pb->contains;

    size_t part = bytes;
    if( part > PIPE_BUFS - pb->getpos )
        part = PIPE_BUFS - pb->getpos;

    SHOW_FLOW( 9, "req %d bytes, part %d", bytes, part );

    memcpy( dest, pb->buf + pb->getpos, part );
    ret += part;
    dest += part;
    bytes -= part;
    pb->getpos += part;
    pb->contains -= part;

    if(pb->getpos >= PIPE_BUFS)
        pb->getpos = 0;

    if( bytes > 0 )
    {
        assert(pb->getpos == 0);
        part = bytes;

        memcpy( dest, pb->buf + pb->getpos, part );
        ret += part;
        //dest += part;
        bytes -= part;
        pb->getpos += part;
        pb->contains -= part;
    }

    assert(bytes == 0);
    assert(ret != 0);

    hal_cond_signal(&pb->cond);

ret:
    hal_mutex_unlock(&pb->mutex);
    return ret;
}

static size_t      pipe_write(   struct uufile *f, const void *src, size_t bytes)
{
    struct pipe_buf *pb = f->impl;
    size_t ret = 0;

    if(pb->kill)
        return 0;

    SHOW_FLOW( 10, "pb @%p", pb );

    hal_mutex_lock(&pb->mutex);

    while(pb->contains >= PIPE_BUFS)
    {
        SHOW_FLOW( 9, "have %d bytes, sleep", pb->contains );
        hal_cond_wait(&pb->cond,&pb->mutex);
        if(pb->kill)
            goto ret;
    }

    if( bytes > PIPE_BUFS - pb->contains )
        bytes = PIPE_BUFS - pb->contains;

    size_t part = bytes;
    if( part > PIPE_BUFS - pb->putpos )
        part = PIPE_BUFS - pb->putpos;

    SHOW_FLOW( 9, "req %d bytes, part %d", bytes, part );

    memcpy( pb->buf + pb->putpos, src, part );
    ret += part;
    src += part;
    bytes -= part;
    pb->putpos += part;
    pb->contains += part;

    if(pb->putpos >= PIPE_BUFS)
        pb->putpos = 0;

    if( bytes > 0 )
    {
        assert(pb->putpos == 0);
        part = bytes;

        memcpy( pb->buf + pb->putpos, src, part );
        ret += part;
        //src += part;
        //bytes -= part;
        pb->putpos += part;
        pb->contains += part;
    }

    // TODO separate r/w conds
    hal_cond_signal(&pb->cond);

    SHOW_FLOW( 9, "ret %d", ret );

ret:
    hal_mutex_unlock(&pb->mutex);
    return ret;
}




#endif // HAVE_UNIX

