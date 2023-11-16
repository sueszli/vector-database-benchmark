/*
 * Copyright (C) 2020 Waldemar Kozaczuk
 *
 * This work is open source software, licensed under the terms of the
 * BSD license as described in the LICENSE file in the top-level directory.
 */

#ifndef VIRTIO_FS_DRIVER_H
#define VIRTIO_FS_DRIVER_H

#include <functional>

#include <osv/mmio.hh>
#include <osv/mutex.h>
#include <osv/sched.hh>
#include <osv/wait_record.hh>

#include "drivers/device.hh"
#include "drivers/driver.hh"
#include "drivers/virtio.hh"
#include "drivers/virtio-device.hh"
#include "fs/virtiofs/fuse_kernel.h"

namespace virtio {

enum {
    VQ_HIPRIO = 0,
    VQ_REQUEST = 1
};

class fs : public virtio_driver {
public:
    // A fuse_request encapsulates a low-level virtio-fs device request. This is
    // a single-use object.
    struct fuse_request {
        struct fuse_in_header in_header;
        struct fuse_out_header out_header;

        void* input_args_data;
        size_t input_args_size;

        void* output_args_data;
        size_t output_args_size;

        // Constructs a fuse_request, determining that wait() on it will be
        // called by @t.
        fuse_request(sched::thread* t): processed{t} {}

        // Waits for the request to be marked as completed. Should only be
        // called once, from the thread specified in the constructor.
        void wait() { processed.wait(); }
        // Marks the request as completed. Should only be called once.
        void done() { processed.wake(); }

    private:
        // Signifies whether the request has been processed by the device
        waiter processed;
    };

    struct fs_config {
        char tag[36];
        u32 num_queues;
    } __attribute__((packed));

    struct dax_window {
        mmioaddr_t addr;
        u64 len;
    };

    // Helper enabling the use of fs* as key type in an unordered_* container.
    struct hasher {
        size_t operator()(fs* const _fs) const noexcept {
            return std::hash<int>{}(_fs->_id);
        }
    };

    explicit fs(virtio_device& dev);
    virtual ~fs();

    virtual std::string get_name() const { return _driver_name; }
    void read_config();

    virtual u64 get_driver_features();

    int make_request(fuse_request&);
    dax_window* get_dax() {
        return (_dax.addr != mmio_nullptr) ? &_dax : nullptr;
    }
    // Set map alignment for DAX window. @map_align should be
    // log2(byte_alignment), e.g. 12 for a 4096 byte alignment.
    void set_map_alignment(int map_align) { _map_align = map_align; }
    // Returns the map alignment for the DAX window as preiously set with
    // set_map_alignment(), or < 0 if it has not been set.
    int get_map_alignment() const { return _map_align; }

    void req_done();
    int64_t size();

    bool ack_irq();

    static hw_driver* probe(hw_device* dev);

private:
    std::string _driver_name;
    fs_config _config;
    dax_window _dax;
    int _map_align;

    // maintains the virtio instance number for multiple drives
    static int _instance;
    int _id;

    // This mutex protects parallel make_request invocations
    mutex _lock;    // TODO: Maintain one mutex per virtqueue
};

}
#endif
