#include "include/stdstream.h"
#include "general.h"
#include "include/vfs.h"
#include "rtlib.h"
#include "thread.h"

#define STREAM_DEFAULT_LEN 512

native_string STDIN_PATH = "/sys/stdin";
native_string STDOUT_PATH = "/sys/stdout";

static native_string STDIN_PART = "STDIN";
static native_string STDOUT_PART = "STDOUT";

static raw_stream stdin;
static raw_stream stdout;

static file_vtable __std_rw_file_stream_vtable;

static error_code new_raw_stream(raw_stream* rs, bool autoresize);
static error_code new_stream_file(stream_file* rs, file_mode mode,
                                  raw_stream* source);

static error_code stream_move_cursor(file* f, int32 n);
static error_code stream_set_to_absolute_position(file* f, uint32 position);
static size_t stream_len(file* f);

static error_code stream_close(file* f);
static error_code stream_write(file* f, void* buff, uint32 count);
static error_code stream_read(file* f, void* buf, uint32 count);

// -------------------------------------------------------------
// Methods that don't make sense on a stream
// -------------------------------------------------------------
void stream_reset_cursor(file* f) { return; }

static error_code stream_move_cursor(file* f, int32 n) { return ARG_ERROR; }

static error_code stream_set_to_absolute_position(file* f, uint32 position) {
  return ARG_ERROR;
}

static size_t stream_len(file* f) { return 0; }


// -------------------------------------------------------------
// Stream management
// -------------------------------------------------------------

static error_code new_raw_stream(raw_stream* rs, bool autoresize) {
  if (NULL == rs) return ARG_ERROR;

  error_code err = NO_ERROR;
  rs->len = STREAM_DEFAULT_LEN;
  rs->buff = kmalloc(sizeof(uint8) * rs->len);

  if (NULL == rs->buff) return MEM_ERROR;

  rs->autoresize = autoresize;
  rs->high = 0;

  condvar* readycv = CAST(condvar*, kmalloc(sizeof(condvar)));

  if (NULL == readycv) {
    kfree(rs->buff);
    return MEM_ERROR;
  }

  rs->readycv = new_condvar(readycv);

  return err;
}

static error_code new_stream_file(stream_file* rs, file_mode mode,
                                  raw_stream* source) {
  error_code err = NO_ERROR;

  if (NULL == rs) {
    return ARG_ERROR;
  }

  if (NULL == source) {
    return ARG_ERROR;
  }

  bool write_only;

  if (IS_MODE_WRITE_ONLY(mode)) {
    write_only = TRUE;
  } else {
    write_only = FALSE;
  }

  rs->header.mode = mode;
  rs->header._vtable = &__std_rw_file_stream_vtable;
  rs->_source = source;
  rs->_lo = 0;
  rs->_reset = source->_reset;

  if (!write_only) source->readers++;

  return err;
}

static error_code stream_close(file* ff) {
  error_code err = NO_ERROR;
  stream_file* f = CAST(stream_file*, ff);
  bool write_only;
  file_mode mode = ff->mode;

  if (IS_MODE_WRITE_ONLY(mode)) {
    write_only = TRUE;
  } else {
    write_only = FALSE;
  }

  if (!write_only) f->_source->readers--;

  kfree(f);
  return err;
}

#define stream_reset(rs)                     \
  do {                                       \
    (rs)->_reset = !(rs)->_reset;            \
    (rs)->high = 0;                          \
    condvar_mutexless_signal((rs)->readycv); \
  } while (0);

static error_code stream_write(file* ff, void* buff, uint32 count) {
  // if (!IS_MODE_WRITE(ff->type)) return PERMISSION_ERROR;

  error_code err = NO_ERROR;
  stream_file* f = CAST(stream_file*, ff);
  raw_stream* rs = f->_source;
  condvar* streamcv = rs->readycv;
  uint8* stream_buff = CAST(uint8*, rs->buff);
  uint8* source_buff = CAST(uint8*, buff);

  bool inter_disabled = ARE_INTERRUPTS_ENABLED();

  {
      /*
       * This implementation of standards streams is terrible. I'm allowed
       * to say it, cause I wrote it. The idea behind it was that every
       * reader would be able to read everything, sort of like a multiplexed
       * stream. I had some concerns over duplicating the same bytes in every
       * stream reader (as one should have) and so I made this fixed-size
       * buffer with multiples readers. The buffer would wait until every
       * read had caught up before moving the cursor backwards (it would move
       * it forward as much as it needed / could). 
       * It did what I wanted perfectly, but I did not want what I needed and 
       * so it mostly is a mix between this broken dream of a multiplexed
       * stream and a good ol' circular buffer. 
       *
       * I will not bother redoing it correctly unless it breaks because
       * it's soon (TM) gonna be implemented in scheme.
       *
       */
    if (inter_disabled) disable_interrupts();

    // Loop like this and do not use mem copy because
    // we want to signal every new character)
    uint32 i;
    for (i = 0; i < count; /*++i is in the write branch*/) {
      uint32 next_hi = (rs->high + 1);
      if (next_hi < rs->len) {
        stream_buff[rs->high] = source_buff[i];
        rs->high = next_hi;
        ++i;
        // No one is caught up: there's a new char
        rs->late = rs->readers;
        condvar_mutexless_signal(streamcv);
      } else if (rs->late == 0) { // this allows a not reader stream to never overflow
        stream_reset(rs);
      } else if (rs->autoresize) {
        // Resize
        panic(L"STD stream resize not implemented yet");
      } else {
        // This ia a temp fix
        rs->high = 0;
        /* panic(L"OOM"); */
        err = MEM_ERROR;
        break;
      }
    }

    if (HAS_NO_ERROR(err) && i == count) {
      err = count;
    }

    if (inter_disabled) enable_interrupts();
  }

  return err;
}

static error_code stream_read(file* ff, void* buff, uint32 count) {
  error_code err = NO_ERROR;
  stream_file* f = CAST(stream_file*, ff);
  raw_stream* rs = f->_source;
  uint8* stream_buff = CAST(uint8*, rs->buff);
  uint8* read_buff = CAST(uint8*, buff);

  bool inter_disabled = ARE_INTERRUPTS_ENABLED();

  if (inter_disabled) disable_interrupts();

  if (f->header.mode & MODE_NONBLOCK_ACCESS) {
      if (f->_reset != rs->_reset) {
          f->_lo = 0;
          f->_reset = rs->_reset;
      }

      // Read as much as possible
      uint32 i = 0;
      for (; f->_lo != rs->high && i < count; ++i) {
          if (NULL != read_buff) read_buff[i] = stream_buff[f->_lo];
          f->_lo = (f->_lo + 1) % rs->len;
      }

      if (f->_lo == rs->high) {
          rs->late = rs->late - 1;
          if (rs->late == 0) {
              stream_reset(rs);
          }
      }

      err = i;
      // Signal afterwards in non blocking mode
      // We want to read in one shot and stop
      // afterwards
  } else {
      panic(L"!NOT IMPL");
  }

  if (inter_disabled) enable_interrupts();

  return err;
}

error_code stream_open_file(uint32 id, file_mode mode, file** result) {
  error_code err = NO_ERROR;
  stream_file* strm = NULL;

  if (0 == id) {
    strm = CAST(stream_file*, kmalloc(sizeof(stream_file)));
    if (NULL == strm) {
      err = MEM_ERROR;
    } else {
      err = new_stream_file(strm, mode, &stdin);
    }
  } else if (1 == id) {
    strm = CAST(stream_file*, kmalloc(sizeof(stream_file)));
    if (NULL == strm) {
      err = MEM_ERROR;
    } else {
      err = new_stream_file(strm, mode, &stdout);
    }
  } else {
    err = FNF_ERROR;
  }

  *result = CAST(file*, strm);

  return err;
}

error_code mount_streams(vfnode* parent) {
  error_code err = NO_ERROR;

  __std_rw_file_stream_vtable._file_close = stream_close;
  __std_rw_file_stream_vtable._file_len = stream_len;
  __std_rw_file_stream_vtable._file_move_cursor = stream_move_cursor;
  __std_rw_file_stream_vtable._file_read = stream_read;
  __std_rw_file_stream_vtable._file_set_to_absolute_position =
      stream_set_to_absolute_position;
  __std_rw_file_stream_vtable._file_write = stream_write;

  // Init streams
  if (ERROR(err = new_raw_stream(&stdin, FALSE))) return err;
  if (ERROR(err = new_raw_stream(&stdout, FALSE))) return err;

  // Init mount point
  vfnode* sys_node = CAST(vfnode*, kmalloc(sizeof(vfnode)));
  new_vfnode(sys_node, "SYS", TYPE_VFOLDER);
  vfnode_add_child(parent, sys_node);
  
  vfnode* stdin_node = CAST(vfnode*, kmalloc(sizeof(vfnode)));
  new_vfnode(stdin_node, STDIN_PART, TYPE_VFILE);
  stdin_node->_value.file_gate.identifier = 0;
  stdin_node->_value.file_gate._vf_node_open = stream_open_file;
  vfnode_add_child(sys_node, stdin_node);

  vfnode* stdout_node = CAST(vfnode*, kmalloc(sizeof(vfnode)));
  new_vfnode(stdout_node, STDOUT_PART, TYPE_VFILE);
  stdout_node->_value.file_gate.identifier = 1;
  stdout_node->_value.file_gate._vf_node_open = stream_open_file;
  vfnode_add_child(sys_node, stdout_node); 

  return err;
}
