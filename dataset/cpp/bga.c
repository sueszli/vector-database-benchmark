/*
 * Copyright (C) 2020-2022 The opuntiaOS Project Authors.
 *  + Contributed by Nikita Melekhin <nimelehin@gmail.com>
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include <drivers/bus/x86/pci.h>
#include <drivers/driver_manager.h>
#include <drivers/graphics/x86/bga.h>
#include <fs/devfs/devfs.h>
#include <fs/vfs.h>
#include <libkern/bits/errno.h>
#include <libkern/libkern.h>
#include <libkern/log.h>
#include <tasking/proc.h>
#include <tasking/tasking.h>

#define VBE_DISPI_IOPORT_INDEX 0x01CE
#define VBE_DISPI_IOPORT_DATA 0x01CF

#define VBE_DISPI_INDEX_ID 0x0
#define VBE_DISPI_INDEX_XRES 0x1
#define VBE_DISPI_INDEX_YRES 0x2
#define VBE_DISPI_INDEX_BPP 0x3
#define VBE_DISPI_INDEX_ENABLE 0x4
#define VBE_DISPI_INDEX_BANK 0x5
#define VBE_DISPI_INDEX_VIRT_WIDTH 0x6
#define VBE_DISPI_INDEX_VIRT_HEIGHT 0x7
#define VBE_DISPI_INDEX_X_OFFSET 0x8
#define VBE_DISPI_INDEX_Y_OFFSET 0x9

#define VBE_DISPI_DISABLED 0x00
#define VBE_DISPI_ENABLED 0x01
#define VBE_DISPI_LFB_ENABLED 0x40

static uint16_t bga_screen_width, bga_screen_height;
static uint32_t bga_screen_line_size, bga_screen_buffer_size;
static uint32_t bga_buf_paddr;

static int _bga_swap_page_mode(struct memzone* zone, uintptr_t vaddr)
{
    return SWAP_NOT_ALLOWED;
}

static vm_ops_t mmap_file_vm_ops = {
    .load_page_content = NULL,
    .restore_swapped_page = NULL,
    .swap_page_mode = _bga_swap_page_mode,
};

static inline void _bga_write_reg(uint16_t cmd, uint16_t data)
{
    port_write16(VBE_DISPI_IOPORT_INDEX, cmd);
    port_write16(VBE_DISPI_IOPORT_DATA, data);
}

static inline uint16_t _bga_read_reg(uint16_t cmd)
{
    port_write16(VBE_DISPI_IOPORT_INDEX, cmd);
    return port_read16(VBE_DISPI_IOPORT_DATA);
}

static void _bga_set_resolution(uint16_t width, uint16_t height)
{
    _bga_write_reg(VBE_DISPI_INDEX_ENABLE, VBE_DISPI_DISABLED);
    _bga_write_reg(VBE_DISPI_INDEX_XRES, width);
    _bga_write_reg(VBE_DISPI_INDEX_YRES, height);
    _bga_write_reg(VBE_DISPI_INDEX_VIRT_WIDTH, width);
    _bga_write_reg(VBE_DISPI_INDEX_VIRT_HEIGHT, (uint16_t)height * 2);
    _bga_write_reg(VBE_DISPI_INDEX_BPP, 32);
    _bga_write_reg(VBE_DISPI_INDEX_X_OFFSET, 0);
    _bga_write_reg(VBE_DISPI_INDEX_Y_OFFSET, 0);
    _bga_write_reg(VBE_DISPI_INDEX_ENABLE, VBE_DISPI_ENABLED | VBE_DISPI_LFB_ENABLED);
    _bga_write_reg(VBE_DISPI_INDEX_BANK, 0);

    bga_screen_line_size = (uint32_t)width * 4;
}

static int _bga_ioctl(file_t* file, uintptr_t cmd, uintptr_t arg)
{
    uint32_t y_offset = 0;
    switch (cmd) {
    case BGA_GET_HEIGHT:
        return bga_screen_height;
    case BGA_GET_WIDTH:
        return bga_screen_width;
    case BGA_GET_SCALE:
        return 1;
    case BGA_SWAP_BUFFERS:
        y_offset = bga_screen_height * (arg & 1);
        _bga_write_reg(VBE_DISPI_INDEX_Y_OFFSET, (uint16_t)y_offset);
        return 0;
    default:
        return -EINVAL;
    }
}

static memzone_t* _bga_mmap(file_t* file, mmap_params_t* params)
{
    bool map_shared = ((params->flags & MAP_SHARED) > 0);

    if (!map_shared) {
        return 0;
    }

    memzone_t* zone = memzone_new_random(RUNNING_THREAD->process->address_space, bga_screen_buffer_size);
    if (!zone) {
        return 0;
    }

    zone->mmu_flags |= MMU_FLAG_PERM_WRITE | MMU_FLAG_PERM_READ | MMU_FLAG_UNCACHED;
    zone->type |= ZONE_TYPE_DEVICE;
    zone->file = file_duplicate(file);
    zone->ops = &mmap_file_vm_ops;

    for (int offset = 0; offset < bga_screen_buffer_size; offset += VMM_PAGE_SIZE) {
        vmm_map_page(zone->vaddr + offset, bga_buf_paddr + offset, zone->mmu_flags);
    }

    return zone;
}

static void bga_recieve_notification(uintptr_t msg, uintptr_t param)
{
    if (msg == DEVMAN_NOTIFICATION_DEVFS_READY) {
        path_t vfspth;
        if (vfs_resolve_path("/dev", &vfspth) < 0) {
            kpanic("Can't init bga in /dev");
        }

        file_ops_t fops = { 0 };
        fops.ioctl = _bga_ioctl;
        fops.mmap = _bga_mmap;
        devfs_inode_t* res = devfs_register(&vfspth, MKDEV(10, 156), "bga", 3, S_IFBLK | 0777, &fops);

        path_put(&vfspth);
    }
}

static void bga_set_resolution(uint32_t width, uint32_t height)
{
    _bga_set_resolution(width, height);
    bga_screen_width = width;
    bga_screen_height = height;
    bga_screen_buffer_size = bga_screen_line_size * height * 2;
}

static inline driver_desc_t _bga_driver_info()
{
    driver_desc_t bga_desc = { 0 };
    bga_desc.type = DRIVER_VIDEO_DEVICE;

    bga_desc.listened_device_mask = DEVICE_DISPLAY;
    bga_desc.system_funcs.init_with_dev = bga_init_with_dev;
    bga_desc.system_funcs.recieve_notification = bga_recieve_notification;

    bga_desc.functions[DRIVER_VIDEO_INIT] = NULL;
    bga_desc.functions[DRIVER_VIDEO_SET_RESOLUTION] = bga_set_resolution;
    return bga_desc;
}

void bga_install()
{
    devman_register_driver(_bga_driver_info(), "bga86");
}
devman_register_driver_installation(bga_install);

int bga_init_with_dev(device_t* dev)
{
    if (dev->device_desc.type != DEVICE_DESC_PCI) {
        return -1;
    }
    if (dev->device_desc.pci.class_id != 0x03) {
        return -1;
    }
    if (dev->device_desc.pci.vendor_id != 0x1234) {
        return -1;
    }
    if (dev->device_desc.pci.device_id != 0x1111) {
        return -1;
    }

    bga_buf_paddr = pci_read_bar(dev, 0) & 0xfffffff0;
#ifdef TARGET_DESKTOP
    bga_set_resolution(1024, 768);
#elif TARGET_MOBILE
    bga_set_resolution(320, 568);
#endif

    return 0;
}
