/*
 * This file and its contents are supplied under the terms of the
 * Common Development and Distribution License ("CDDL"), version 1.0.
 * You may only use this file in accordance with the terms of version
 * 1.0 of the CDDL.
 *
 * A full copy of the text of the CDDL should have accompanied this
 * source.  A copy of the CDDL is also available via the Internet at
 * http://www.illumos.org/license/CDDL.
 */

/*
 * Copyright (c) 2016 The MathWorks, Inc.  All rights reserved.
 * Copyright 2019 Unix Software Ltd.
 * Copyright 2020 Joyent, Inc.
 * Copyright 2020 Racktop Systems.
 * Copyright 2023 Oxide Computer Company.
 * Copyright 2022 OmniOS Community Edition (OmniOSce) Association.
 * Copyright 2022 Tintri by DDN, Inc. All rights reserved.
 */

/*
 * blkdev driver for NVMe compliant storage devices
 *
 * This driver targets and is designed to support all NVMe 1.x devices.
 * Features are added to the driver as we encounter devices that require them
 * and our needs, so some commands or log pages may not take advantage of newer
 * features that devices support at this time. When you encounter such a case,
 * it is generally fine to add that support to the driver as long as you take
 * care to ensure that the requisite device version is met before using it.
 *
 * The driver has only been tested on x86 systems and will not work on big-
 * endian systems without changes to the code accessing registers and data
 * structures used by the hardware.
 *
 *
 * Interrupt Usage:
 *
 * The driver will use a single interrupt while configuring the device as the
 * specification requires, but contrary to the specification it will try to use
 * a single-message MSI(-X) or FIXED interrupt. Later in the attach process it
 * will switch to multiple-message MSI(-X) if supported. The driver wants to
 * have one interrupt vector per CPU, but it will work correctly if less are
 * available. Interrupts can be shared by queues, the interrupt handler will
 * iterate through the I/O queue array by steps of n_intr_cnt. Usually only
 * the admin queue will share an interrupt with one I/O queue. The interrupt
 * handler will retrieve completed commands from all queues sharing an interrupt
 * vector and will post them to a taskq for completion processing.
 *
 *
 * Command Processing:
 *
 * NVMe devices can have up to 65535 I/O queue pairs, with each queue holding up
 * to 65536 I/O commands. The driver will configure one I/O queue pair per
 * available interrupt vector, with the queue length usually much smaller than
 * the maximum of 65536. If the hardware doesn't provide enough queues, fewer
 * interrupt vectors will be used.
 *
 * Additionally the hardware provides a single special admin queue pair that can
 * hold up to 4096 admin commands.
 *
 * From the hardware perspective both queues of a queue pair are independent,
 * but they share some driver state: the command array (holding pointers to
 * commands currently being processed by the hardware) and the active command
 * counter. Access to a submission queue and the shared state is protected by
 * nq_mutex; completion queue is protected by ncq_mutex.
 *
 * When a command is submitted to a queue pair the active command counter is
 * incremented and a pointer to the command is stored in the command array. The
 * array index is used as command identifier (CID) in the submission queue
 * entry. Some commands may take a very long time to complete, and if the queue
 * wraps around in that time a submission may find the next array slot to still
 * be used by a long-running command. In this case the array is sequentially
 * searched for the next free slot. The length of the command array is the same
 * as the configured queue length. Queue overrun is prevented by the semaphore,
 * so a command submission may block if the queue is full.
 *
 *
 * Polled I/O Support:
 *
 * For kernel core dump support the driver can do polled I/O. As interrupts are
 * turned off while dumping the driver will just submit a command in the regular
 * way, and then repeatedly attempt a command retrieval until it gets the
 * command back.
 *
 *
 * Namespace Support:
 *
 * NVMe devices can have multiple namespaces, each being a independent data
 * store. The driver supports multiple namespaces and creates a blkdev interface
 * for each namespace found. Namespaces can have various attributes to support
 * protection information. This driver does not support any of this and ignores
 * namespaces that have these attributes.
 *
 * As of NVMe 1.1 namespaces can have an 64bit Extended Unique Identifier
 * (EUI64), and NVMe 1.2 introduced an additional 128bit Namespace Globally
 * Unique Identifier (NGUID). This driver uses either the NGUID or the EUI64
 * if present to generate the devid, and passes the EUI64 to blkdev to use it
 * in the device node names.
 *
 * We currently support only (2 << NVME_MINOR_INST_SHIFT) - 2 namespaces in a
 * single controller. This is an artificial limit imposed by the driver to be
 * able to address a reasonable number of controllers and namespaces using a
 * 32bit minor node number.
 *
 *
 * Minor nodes:
 *
 * For each NVMe device the driver exposes one minor node for the controller and
 * one minor node for each namespace. The only operations supported by those
 * minor nodes are open(9E), close(9E), and ioctl(9E). This serves as the
 * interface for the nvmeadm(8) utility.
 *
 * Exclusive opens are required for certain ioctl(9E) operations that alter
 * controller and/or namespace state. While different namespaces may be opened
 * exclusively in parallel, an exclusive open of the controller minor node
 * requires that no namespaces are currently open (exclusive or otherwise).
 * Opening any namespace minor node (exclusive or otherwise) will fail while
 * the controller minor node is opened exclusively by any other thread. Thus it
 * is possible for one thread at a time to open the controller minor node
 * exclusively, and keep it open while opening any namespace minor node of the
 * same controller, exclusively or otherwise.
 *
 *
 *
 * Blkdev Interface:
 *
 * This driver uses blkdev to do all the heavy lifting involved with presenting
 * a disk device to the system. As a result, the processing of I/O requests is
 * relatively simple as blkdev takes care of partitioning, boundary checks, DMA
 * setup, and splitting of transfers into manageable chunks.
 *
 * I/O requests coming in from blkdev are turned into NVM commands and posted to
 * an I/O queue. The queue is selected by taking the CPU id modulo the number of
 * queues. There is currently no timeout handling of I/O commands.
 *
 * Blkdev also supports querying device/media information and generating a
 * devid. The driver reports the best block size as determined by the namespace
 * format back to blkdev as physical block size to support partition and block
 * alignment. The devid is either based on the namespace GUID or EUI64, if
 * present, or composed using the device vendor ID, model number, serial number,
 * and the namespace ID.
 *
 *
 * Error Handling:
 *
 * Error handling is currently limited to detecting fatal hardware errors,
 * either by asynchronous events, or synchronously through command status or
 * admin command timeouts. In case of severe errors the device is fenced off,
 * all further requests will return EIO. FMA is then called to fault the device.
 *
 * The hardware has a limit for outstanding asynchronous event requests. Before
 * this limit is known the driver assumes it is at least 1 and posts a single
 * asynchronous request. Later when the limit is known more asynchronous event
 * requests are posted to allow quicker reception of error information. When an
 * asynchronous event is posted by the hardware the driver will parse the error
 * status fields and log information or fault the device, depending on the
 * severity of the asynchronous event. The asynchronous event request is then
 * reused and posted to the admin queue again.
 *
 * On command completion the command status is checked for errors. In case of
 * errors indicating a driver bug the driver panics. Almost all other error
 * status values just cause EIO to be returned.
 *
 * Command timeouts are currently detected for all admin commands except
 * asynchronous event requests. If a command times out and the hardware appears
 * to be healthy the driver attempts to abort the command. The original command
 * timeout is also applied to the abort command. If the abort times out too the
 * driver assumes the device to be dead, fences it off, and calls FMA to retire
 * it. In all other cases the aborted command should return immediately with a
 * status indicating it was aborted, and the driver will wait indefinitely for
 * that to happen. No timeout handling of normal I/O commands is presently done.
 *
 * Any command that times out due to the controller dropping dead will be put on
 * nvme_lost_cmds list if it references DMA memory. This will prevent the DMA
 * memory being reused by the system and later be written to by a "dead" NVMe
 * controller.
 *
 *
 * Locking:
 *
 * Each queue pair has a nq_mutex and ncq_mutex. The nq_mutex must be held
 * when accessing shared state and submission queue registers, ncq_mutex
 * is held when accessing completion queue state and registers.
 * Callers of nvme_unqueue_cmd() must make sure that nq_mutex is held, while
 * nvme_submit_{admin,io}_cmd() and nvme_retrieve_cmd() take care of both
 * mutexes themselves.
 *
 * Each command also has its own nc_mutex, which is associated with the
 * condition variable nc_cv. It is only used on admin commands which are run
 * synchronously. In that case it must be held across calls to
 * nvme_submit_{admin,io}_cmd() and nvme_wait_cmd(), which is taken care of by
 * nvme_admin_cmd(). It must also be held whenever the completion state of the
 * command is changed or while a admin command timeout is handled.
 *
 * If both nc_mutex and nq_mutex must be held, nc_mutex must be acquired first.
 * More than one nc_mutex may only be held when aborting commands. In this case,
 * the nc_mutex of the command to be aborted must be held across the call to
 * nvme_abort_cmd() to prevent the command from completing while the abort is in
 * progress.
 *
 * If both nq_mutex and ncq_mutex need to be held, ncq_mutex must be
 * acquired first. More than one nq_mutex is never held by a single thread.
 * The ncq_mutex is only held by nvme_retrieve_cmd() and
 * nvme_process_iocq(). nvme_process_iocq() is only called from the
 * interrupt thread and nvme_retrieve_cmd() during polled I/O, so the
 * mutex is non-contentious but is required for implementation completeness
 * and safety.
 *
 * There is one mutex n_minor_mutex which protects all open flags nm_open and
 * exclusive-open thread pointers nm_oexcl of each minor node associated with a
 * controller and its namespaces.
 *
 * In addition, there is one mutex n_mgmt_mutex which must be held whenever the
 * driver state for any namespace is changed, especially across calls to
 * nvme_init_ns(), nvme_attach_ns() and nvme_detach_ns(). Except when detaching
 * nvme, it should also be held across calls that modify the blkdev handle of a
 * namespace. Command and queue mutexes may be acquired and released while
 * n_mgmt_mutex is held, n_minor_mutex should not.
 *
 *
 * Quiesce / Fast Reboot:
 *
 * The driver currently does not support fast reboot. A quiesce(9E) entry point
 * is still provided which is used to send a shutdown notification to the
 * device.
 *
 *
 * NVMe Hotplug:
 *
 * The driver supports hot removal. The driver uses the NDI event framework
 * to register a callback, nvme_remove_callback, to clean up when a disk is
 * removed. In particular, the driver will unqueue outstanding I/O commands and
 * set n_dead on the softstate to true so that other operations, such as ioctls
 * and command submissions, fail as well.
 *
 * While the callback registration relies on the NDI event framework, the
 * removal event itself is kicked off in the PCIe hotplug framework, when the
 * PCIe bridge driver ("pcieb") gets a hotplug interrupt indicating that a
 * device was removed from the slot.
 *
 * The NVMe driver instance itself will remain until the final close of the
 * device.
 *
 *
 * DDI UFM Support
 *
 * The driver supports the DDI UFM framework for reporting information about
 * the device's firmware image and slot configuration. This data can be
 * queried by userland software via ioctls to the ufm driver. For more
 * information, see ddi_ufm(9E).
 *
 *
 * Driver Configuration:
 *
 * The following driver properties can be changed to control some aspects of the
 * drivers operation:
 * - strict-version: can be set to 0 to allow devices conforming to newer
 *   major versions to be used
 * - ignore-unknown-vendor-status: can be set to 1 to not handle any vendor
 *   specific command status as a fatal error leading device faulting
 * - admin-queue-len: the maximum length of the admin queue (16-4096)
 * - io-squeue-len: the maximum length of the I/O submission queues (16-65536)
 * - io-cqueue-len: the maximum length of the I/O completion queues (16-65536)
 * - async-event-limit: the maximum number of asynchronous event requests to be
 *   posted by the driver
 * - volatile-write-cache-enable: can be set to 0 to disable the volatile write
 *   cache
 * - min-phys-block-size: the minimum physical block size to report to blkdev,
 *   which is among other things the basis for ZFS vdev ashift
 * - max-submission-queues: the maximum number of I/O submission queues.
 * - max-completion-queues: the maximum number of I/O completion queues,
 *   can be less than max-submission-queues, in which case the completion
 *   queues are shared.
 *
 * In addition to the above properties, some device-specific tunables can be
 * configured using the nvme-config-list global property. The value of this
 * property is a list of triplets. The formal syntax is:
 *
 *   nvme-config-list ::= <triplet> [, <triplet>]* ;
 *   <triplet>        ::= "<model>" , "<rev-list>" , "<tuple-list>"
 *   <rev-list>       ::= [ <fwrev> [, <fwrev>]*]
 *   <tuple-list>     ::= <tunable> [, <tunable>]*
 *   <tunable>        ::= <name> : <value>
 *
 * The <model> and <fwrev> are the strings in nvme_identify_ctrl_t`id_model and
 * nvme_identify_ctrl_t`id_fwrev, respectively. The remainder of <tuple-list>
 * contains one or more tunables to apply to all controllers that match the
 * specified model number and optionally firmware revision. Each <tunable> is a
 * <name> : <value> pair.  Supported tunables are:
 *
 * - ignore-unknown-vendor-status:  can be set to "on" to not handle any vendor
 *   specific command status as a fatal error leading device faulting
 *
 * - min-phys-block-size: the minimum physical block size to report to blkdev,
 *   which is among other things the basis for ZFS vdev ashift
 *
 * - volatile-write-cache: can be set to "on" or "off" to enable or disable the
 *   volatile write cache, if present
 *
 *
 * TODO:
 * - figure out sane default for I/O queue depth reported to blkdev
 * - FMA handling of media errors
 * - support for devices supporting very large I/O requests using chained PRPs
 * - support for configuring hardware parameters like interrupt coalescing
 * - support for media formatting and hard partitioning into namespaces
 * - support for big-endian systems
 * - support for fast reboot
 * - support for NVMe Subsystem Reset (1.1)
 * - support for Scatter/Gather lists (1.1)
 * - support for Reservations (1.1)
 * - support for power management
 */

#include <sys/byteorder.h>
#ifdef _BIG_ENDIAN
#error nvme driver needs porting for big-endian platforms
#endif

#include <sys/modctl.h>
#include <sys/conf.h>
#include <sys/devops.h>
#include <sys/ddi.h>
#include <sys/ddi_ufm.h>
#include <sys/sunddi.h>
#include <sys/sunndi.h>
#include <sys/bitmap.h>
#include <sys/sysmacros.h>
#include <sys/param.h>
#include <sys/varargs.h>
#include <sys/cpuvar.h>
#include <sys/disp.h>
#include <sys/blkdev.h>
#include <sys/atomic.h>
#include <sys/archsystm.h>
#include <sys/sata/sata_hba.h>
#include <sys/stat.h>
#include <sys/policy.h>
#include <sys/list.h>
#include <sys/dkio.h>
#include <sys/pci.h>

#include <sys/nvme.h>

#ifdef __x86
#include <sys/x86_archext.h>
#endif

#include "nvme_reg.h"
#include "nvme_var.h"

/*
 * Assertions to make sure that we've properly captured various aspects of the
 * packed structures and haven't broken them during updates.
 */
CTASSERT(sizeof (nvme_identify_ctrl_t) == NVME_IDENTIFY_BUFSIZE);
CTASSERT(offsetof(nvme_identify_ctrl_t, id_oacs) == 256);
CTASSERT(offsetof(nvme_identify_ctrl_t, id_sqes) == 512);
CTASSERT(offsetof(nvme_identify_ctrl_t, id_oncs) == 520);
CTASSERT(offsetof(nvme_identify_ctrl_t, id_subnqn) == 768);
CTASSERT(offsetof(nvme_identify_ctrl_t, id_nvmof) == 1792);
CTASSERT(offsetof(nvme_identify_ctrl_t, id_psd) == 2048);
CTASSERT(offsetof(nvme_identify_ctrl_t, id_vs) == 3072);

CTASSERT(sizeof (nvme_identify_nsid_t) == NVME_IDENTIFY_BUFSIZE);
CTASSERT(offsetof(nvme_identify_nsid_t, id_fpi) == 32);
CTASSERT(offsetof(nvme_identify_nsid_t, id_anagrpid) == 92);
CTASSERT(offsetof(nvme_identify_nsid_t, id_nguid) == 104);
CTASSERT(offsetof(nvme_identify_nsid_t, id_lbaf) == 128);
CTASSERT(offsetof(nvme_identify_nsid_t, id_vs) == 384);

CTASSERT(sizeof (nvme_identify_nsid_list_t) == NVME_IDENTIFY_BUFSIZE);
CTASSERT(sizeof (nvme_identify_ctrl_list_t) == NVME_IDENTIFY_BUFSIZE);

CTASSERT(sizeof (nvme_identify_primary_caps_t) == NVME_IDENTIFY_BUFSIZE);
CTASSERT(offsetof(nvme_identify_primary_caps_t, nipc_vqfrt) == 32);
CTASSERT(offsetof(nvme_identify_primary_caps_t, nipc_vifrt) == 64);

CTASSERT(sizeof (nvme_nschange_list_t) == 4096);


/* NVMe spec version supported */
static const int nvme_version_major = 2;

/* tunable for admin command timeout in seconds, default is 1s */
int nvme_admin_cmd_timeout = 1;

/* tunable for FORMAT NVM command timeout in seconds, default is 600s */
int nvme_format_cmd_timeout = 600;

/* tunable for firmware commit with NVME_FWC_SAVE, default is 15s */
int nvme_commit_save_cmd_timeout = 15;

/*
 * tunable for the size of arbitrary vendor specific admin commands,
 * default is 16MiB.
 */
uint32_t nvme_vendor_specific_admin_cmd_size = 1 << 24;

/*
 * tunable for the max timeout of arbitary vendor specific admin commands,
 * default is 60s.
 */
uint_t nvme_vendor_specific_admin_cmd_max_timeout = 60;

static int nvme_attach(dev_info_t *, ddi_attach_cmd_t);
static int nvme_detach(dev_info_t *, ddi_detach_cmd_t);
static int nvme_quiesce(dev_info_t *);
static int nvme_fm_errcb(dev_info_t *, ddi_fm_error_t *, const void *);
static int nvme_setup_interrupts(nvme_t *, int, int);
static void nvme_release_interrupts(nvme_t *);
static uint_t nvme_intr(caddr_t, caddr_t);

static void nvme_shutdown(nvme_t *, boolean_t);
static boolean_t nvme_reset(nvme_t *, boolean_t);
static int nvme_init(nvme_t *);
static nvme_cmd_t *nvme_alloc_cmd(nvme_t *, int);
static void nvme_free_cmd(nvme_cmd_t *);
static nvme_cmd_t *nvme_create_nvm_cmd(nvme_namespace_t *, uint8_t,
    bd_xfer_t *);
static void nvme_admin_cmd(nvme_cmd_t *, int);
static void nvme_submit_admin_cmd(nvme_qpair_t *, nvme_cmd_t *);
static int nvme_submit_io_cmd(nvme_qpair_t *, nvme_cmd_t *);
static void nvme_submit_cmd_common(nvme_qpair_t *, nvme_cmd_t *);
static nvme_cmd_t *nvme_unqueue_cmd(nvme_t *, nvme_qpair_t *, int);
static nvme_cmd_t *nvme_retrieve_cmd(nvme_t *, nvme_qpair_t *);
static void nvme_wait_cmd(nvme_cmd_t *, uint_t);
static void nvme_wakeup_cmd(void *);
static void nvme_async_event_task(void *);

static int nvme_check_unknown_cmd_status(nvme_cmd_t *);
static int nvme_check_vendor_cmd_status(nvme_cmd_t *);
static int nvme_check_integrity_cmd_status(nvme_cmd_t *);
static int nvme_check_specific_cmd_status(nvme_cmd_t *);
static int nvme_check_generic_cmd_status(nvme_cmd_t *);
static inline int nvme_check_cmd_status(nvme_cmd_t *);

static int nvme_abort_cmd(nvme_cmd_t *, uint_t);
static void nvme_async_event(nvme_t *);
static int nvme_format_nvm(nvme_t *, boolean_t, uint32_t, uint8_t, boolean_t,
    uint8_t, boolean_t, uint8_t);
static int nvme_get_logpage(nvme_t *, boolean_t, void **, size_t *, uint8_t,
    ...);
static int nvme_identify(nvme_t *, boolean_t, uint32_t, uint8_t, void **);
static int nvme_set_features(nvme_t *, boolean_t, uint32_t, uint8_t, uint32_t,
    uint32_t *);
static int nvme_get_features(nvme_t *, boolean_t, uint32_t, uint8_t, uint32_t *,
    void **, size_t *);
static int nvme_write_cache_set(nvme_t *, boolean_t);
static int nvme_set_nqueues(nvme_t *);

static void nvme_free_dma(nvme_dma_t *);
static int nvme_zalloc_dma(nvme_t *, size_t, uint_t, ddi_dma_attr_t *,
    nvme_dma_t **);
static int nvme_zalloc_queue_dma(nvme_t *, uint32_t, uint16_t, uint_t,
    nvme_dma_t **);
static void nvme_free_qpair(nvme_qpair_t *);
static int nvme_alloc_qpair(nvme_t *, uint32_t, nvme_qpair_t **, uint_t);
static int nvme_create_io_qpair(nvme_t *, nvme_qpair_t *, uint16_t);

static inline void nvme_put64(nvme_t *, uintptr_t, uint64_t);
static inline void nvme_put32(nvme_t *, uintptr_t, uint32_t);
static inline uint64_t nvme_get64(nvme_t *, uintptr_t);
static inline uint32_t nvme_get32(nvme_t *, uintptr_t);

static boolean_t nvme_check_regs_hdl(nvme_t *);
static boolean_t nvme_check_dma_hdl(nvme_dma_t *);

static int nvme_fill_prp(nvme_cmd_t *, ddi_dma_handle_t);

static void nvme_bd_xfer_done(void *);
static void nvme_bd_driveinfo(void *, bd_drive_t *);
static int nvme_bd_mediainfo(void *, bd_media_t *);
static int nvme_bd_cmd(nvme_namespace_t *, bd_xfer_t *, uint8_t);
static int nvme_bd_read(void *, bd_xfer_t *);
static int nvme_bd_write(void *, bd_xfer_t *);
static int nvme_bd_sync(void *, bd_xfer_t *);
static int nvme_bd_devid(void *, dev_info_t *, ddi_devid_t *);
static int nvme_bd_free_space(void *, bd_xfer_t *);

static int nvme_prp_dma_constructor(void *, void *, int);
static void nvme_prp_dma_destructor(void *, void *);

static void nvme_prepare_devid(nvme_t *, uint32_t);

/* DDI UFM callbacks */
static int nvme_ufm_fill_image(ddi_ufm_handle_t *, void *, uint_t,
    ddi_ufm_image_t *);
static int nvme_ufm_fill_slot(ddi_ufm_handle_t *, void *, uint_t, uint_t,
    ddi_ufm_slot_t *);
static int nvme_ufm_getcaps(ddi_ufm_handle_t *, void *, ddi_ufm_cap_t *);

static int nvme_open(dev_t *, int, int, cred_t *);
static int nvme_close(dev_t, int, int, cred_t *);
static int nvme_ioctl(dev_t, int, intptr_t, int, cred_t *, int *);

static int nvme_init_ns(nvme_t *, int);
static int nvme_attach_ns(nvme_t *, int);
static int nvme_detach_ns(nvme_t *, int);

#define	NVME_NSID2NS(nvme, nsid)	(&((nvme)->n_ns[(nsid) - 1]))

static ddi_ufm_ops_t nvme_ufm_ops = {
	NULL,
	nvme_ufm_fill_image,
	nvme_ufm_fill_slot,
	nvme_ufm_getcaps
};

#define	NVME_MINOR_INST_SHIFT	9
#define	NVME_MINOR(inst, nsid)	(((inst) << NVME_MINOR_INST_SHIFT) | (nsid))
#define	NVME_MINOR_INST(minor)	((minor) >> NVME_MINOR_INST_SHIFT)
#define	NVME_MINOR_NSID(minor)	((minor) & ((1 << NVME_MINOR_INST_SHIFT) - 1))
#define	NVME_MINOR_MAX		(NVME_MINOR(1, 0) - 2)
#define	NVME_IS_VENDOR_SPECIFIC_CMD(x)	(((x) >= 0xC0) && ((x) <= 0xFF))
#define	NVME_VENDOR_SPECIFIC_LOGPAGE_MIN	0xC0
#define	NVME_VENDOR_SPECIFIC_LOGPAGE_MAX	0xFF
#define	NVME_IS_VENDOR_SPECIFIC_LOGPAGE(x)	\
		(((x) >= NVME_VENDOR_SPECIFIC_LOGPAGE_MIN) && \
		((x) <= NVME_VENDOR_SPECIFIC_LOGPAGE_MAX))

/*
 * NVMe versions 1.3 and later actually support log pages up to UINT32_MAX
 * DWords in size. However, revision 1.3 also modified the layout of the Get Log
 * Page command significantly relative to version 1.2, including changing
 * reserved bits, adding new bitfields, and requiring the use of command DWord
 * 11 to fully specify the size of the log page (the lower and upper 16 bits of
 * the number of DWords in the page are split between DWord 10 and DWord 11,
 * respectively).
 *
 * All of these impose significantly different layout requirements on the
 * `nvme_getlogpage_t` type. This could be solved with two different types, or a
 * complicated/nested union with the two versions as the overlying members. Both
 * of these are reasonable, if a bit convoluted. However, these is no current
 * need for such large pages, or a way to test them, as most log pages actually
 * fit within the current size limit. So for simplicity, we retain the size cap
 * from version 1.2.
 *
 * Note that the number of DWords is zero-based, so we add 1. It is subtracted
 * to form a zero-based value in `nvme_get_logpage`.
 */
#define	NVME_VENDOR_SPECIFIC_LOGPAGE_MAX_SIZE	\
		(((1 << 12) + 1) * sizeof (uint32_t))

static void *nvme_state;
static kmem_cache_t *nvme_cmd_cache;

/*
 * DMA attributes for queue DMA memory
 *
 * Queue DMA memory must be page aligned. The maximum length of a queue is
 * 65536 entries, and an entry can be 64 bytes long.
 */
static const ddi_dma_attr_t nvme_queue_dma_attr = {
	.dma_attr_version	= DMA_ATTR_V0,
	.dma_attr_addr_lo	= 0,
	.dma_attr_addr_hi	= 0xffffffffffffffffULL,
	.dma_attr_count_max	= (UINT16_MAX + 1) * sizeof (nvme_sqe_t) - 1,
	.dma_attr_align		= 0x1000,
	.dma_attr_burstsizes	= 0x7ff,
	.dma_attr_minxfer	= 0x1000,
	.dma_attr_maxxfer	= (UINT16_MAX + 1) * sizeof (nvme_sqe_t),
	.dma_attr_seg		= 0xffffffffffffffffULL,
	.dma_attr_sgllen	= 1,
	.dma_attr_granular	= 1,
	.dma_attr_flags		= 0,
};

/*
 * DMA attributes for transfers using Physical Region Page (PRP) entries
 *
 * A PRP entry describes one page of DMA memory using the page size specified
 * in the controller configuration's memory page size register (CC.MPS). It uses
 * a 64bit base address aligned to this page size. There is no limitation on
 * chaining PRPs together for arbitrarily large DMA transfers. These DMA
 * attributes will be copied into the nvme_t during nvme_attach() and the
 * dma_attr_maxxfer will be updated.
 */
static const ddi_dma_attr_t nvme_prp_dma_attr = {
	.dma_attr_version	= DMA_ATTR_V0,
	.dma_attr_addr_lo	= 0,
	.dma_attr_addr_hi	= 0xffffffffffffffffULL,
	.dma_attr_count_max	= 0xfff,
	.dma_attr_align		= 0x1000,
	.dma_attr_burstsizes	= 0x7ff,
	.dma_attr_minxfer	= 0x1000,
	.dma_attr_maxxfer	= 0x1000,
	.dma_attr_seg		= 0xfff,
	.dma_attr_sgllen	= -1,
	.dma_attr_granular	= 1,
	.dma_attr_flags		= 0,
};

/*
 * DMA attributes for transfers using scatter/gather lists
 *
 * A SGL entry describes a chunk of DMA memory using a 64bit base address and a
 * 32bit length field. SGL Segment and SGL Last Segment entries require the
 * length to be a multiple of 16 bytes. While the SGL DMA attributes are copied
 * into the nvme_t, they are not currently used for any I/O.
 */
static const ddi_dma_attr_t nvme_sgl_dma_attr = {
	.dma_attr_version	= DMA_ATTR_V0,
	.dma_attr_addr_lo	= 0,
	.dma_attr_addr_hi	= 0xffffffffffffffffULL,
	.dma_attr_count_max	= 0xffffffffUL,
	.dma_attr_align		= 1,
	.dma_attr_burstsizes	= 0x7ff,
	.dma_attr_minxfer	= 0x10,
	.dma_attr_maxxfer	= 0xfffffffffULL,
	.dma_attr_seg		= 0xffffffffffffffffULL,
	.dma_attr_sgllen	= -1,
	.dma_attr_granular	= 0x10,
	.dma_attr_flags		= 0
};

static ddi_device_acc_attr_t nvme_reg_acc_attr = {
	.devacc_attr_version	= DDI_DEVICE_ATTR_V0,
	.devacc_attr_endian_flags = DDI_STRUCTURE_LE_ACC,
	.devacc_attr_dataorder	= DDI_STRICTORDER_ACC
};

static struct cb_ops nvme_cb_ops = {
	.cb_open	= nvme_open,
	.cb_close	= nvme_close,
	.cb_strategy	= nodev,
	.cb_print	= nodev,
	.cb_dump	= nodev,
	.cb_read	= nodev,
	.cb_write	= nodev,
	.cb_ioctl	= nvme_ioctl,
	.cb_devmap	= nodev,
	.cb_mmap	= nodev,
	.cb_segmap	= nodev,
	.cb_chpoll	= nochpoll,
	.cb_prop_op	= ddi_prop_op,
	.cb_str		= 0,
	.cb_flag	= D_NEW | D_MP,
	.cb_rev		= CB_REV,
	.cb_aread	= nodev,
	.cb_awrite	= nodev
};

static struct dev_ops nvme_dev_ops = {
	.devo_rev	= DEVO_REV,
	.devo_refcnt	= 0,
	.devo_getinfo	= ddi_no_info,
	.devo_identify	= nulldev,
	.devo_probe	= nulldev,
	.devo_attach	= nvme_attach,
	.devo_detach	= nvme_detach,
	.devo_reset	= nodev,
	.devo_cb_ops	= &nvme_cb_ops,
	.devo_bus_ops	= NULL,
	.devo_power	= NULL,
	.devo_quiesce	= nvme_quiesce,
};

static struct modldrv nvme_modldrv = {
	.drv_modops	= &mod_driverops,
	.drv_linkinfo	= "NVMe v1.1b",
	.drv_dev_ops	= &nvme_dev_ops
};

static struct modlinkage nvme_modlinkage = {
	.ml_rev		= MODREV_1,
	.ml_linkage	= { &nvme_modldrv, NULL }
};

static bd_ops_t nvme_bd_ops = {
	.o_version	= BD_OPS_CURRENT_VERSION,
	.o_drive_info	= nvme_bd_driveinfo,
	.o_media_info	= nvme_bd_mediainfo,
	.o_devid_init	= nvme_bd_devid,
	.o_sync_cache	= nvme_bd_sync,
	.o_read		= nvme_bd_read,
	.o_write	= nvme_bd_write,
	.o_free_space	= nvme_bd_free_space,
};

/*
 * This list will hold commands that have timed out and couldn't be aborted.
 * As we don't know what the hardware may still do with the DMA memory we can't
 * free them, so we'll keep them forever on this list where we can easily look
 * at them with mdb.
 */
static struct list nvme_lost_cmds;
static kmutex_t nvme_lc_mutex;

int
_init(void)
{
	int error;

	error = ddi_soft_state_init(&nvme_state, sizeof (nvme_t), 1);
	if (error != DDI_SUCCESS)
		return (error);

	nvme_cmd_cache = kmem_cache_create("nvme_cmd_cache",
	    sizeof (nvme_cmd_t), 64, NULL, NULL, NULL, NULL, NULL, 0);

	mutex_init(&nvme_lc_mutex, NULL, MUTEX_DRIVER, NULL);
	list_create(&nvme_lost_cmds, sizeof (nvme_cmd_t),
	    offsetof(nvme_cmd_t, nc_list));

	bd_mod_init(&nvme_dev_ops);

	error = mod_install(&nvme_modlinkage);
	if (error != DDI_SUCCESS) {
		ddi_soft_state_fini(&nvme_state);
		mutex_destroy(&nvme_lc_mutex);
		list_destroy(&nvme_lost_cmds);
		bd_mod_fini(&nvme_dev_ops);
	}

	return (error);
}

int
_fini(void)
{
	int error;

	if (!list_is_empty(&nvme_lost_cmds))
		return (DDI_FAILURE);

	error = mod_remove(&nvme_modlinkage);
	if (error == DDI_SUCCESS) {
		ddi_soft_state_fini(&nvme_state);
		kmem_cache_destroy(nvme_cmd_cache);
		mutex_destroy(&nvme_lc_mutex);
		list_destroy(&nvme_lost_cmds);
		bd_mod_fini(&nvme_dev_ops);
	}

	return (error);
}

int
_info(struct modinfo *modinfop)
{
	return (mod_info(&nvme_modlinkage, modinfop));
}

static inline void
nvme_put64(nvme_t *nvme, uintptr_t reg, uint64_t val)
{
	ASSERT(((uintptr_t)(nvme->n_regs + reg) & 0x7) == 0);

	/*LINTED: E_BAD_PTR_CAST_ALIGN*/
	ddi_put64(nvme->n_regh, (uint64_t *)(nvme->n_regs + reg), val);
}

static inline void
nvme_put32(nvme_t *nvme, uintptr_t reg, uint32_t val)
{
	ASSERT(((uintptr_t)(nvme->n_regs + reg) & 0x3) == 0);

	/*LINTED: E_BAD_PTR_CAST_ALIGN*/
	ddi_put32(nvme->n_regh, (uint32_t *)(nvme->n_regs + reg), val);
}

static inline uint64_t
nvme_get64(nvme_t *nvme, uintptr_t reg)
{
	uint64_t val;

	ASSERT(((uintptr_t)(nvme->n_regs + reg) & 0x7) == 0);

	/*LINTED: E_BAD_PTR_CAST_ALIGN*/
	val = ddi_get64(nvme->n_regh, (uint64_t *)(nvme->n_regs + reg));

	return (val);
}

static inline uint32_t
nvme_get32(nvme_t *nvme, uintptr_t reg)
{
	uint32_t val;

	ASSERT(((uintptr_t)(nvme->n_regs + reg) & 0x3) == 0);

	/*LINTED: E_BAD_PTR_CAST_ALIGN*/
	val = ddi_get32(nvme->n_regh, (uint32_t *)(nvme->n_regs + reg));

	return (val);
}

static boolean_t
nvme_check_regs_hdl(nvme_t *nvme)
{
	ddi_fm_error_t error;

	ddi_fm_acc_err_get(nvme->n_regh, &error, DDI_FME_VERSION);

	if (error.fme_status != DDI_FM_OK)
		return (B_TRUE);

	return (B_FALSE);
}

static boolean_t
nvme_check_dma_hdl(nvme_dma_t *dma)
{
	ddi_fm_error_t error;

	if (dma == NULL)
		return (B_FALSE);

	ddi_fm_dma_err_get(dma->nd_dmah, &error, DDI_FME_VERSION);

	if (error.fme_status != DDI_FM_OK)
		return (B_TRUE);

	return (B_FALSE);
}

static void
nvme_free_dma_common(nvme_dma_t *dma)
{
	if (dma->nd_dmah != NULL)
		(void) ddi_dma_unbind_handle(dma->nd_dmah);
	if (dma->nd_acch != NULL)
		ddi_dma_mem_free(&dma->nd_acch);
	if (dma->nd_dmah != NULL)
		ddi_dma_free_handle(&dma->nd_dmah);
}

static void
nvme_free_dma(nvme_dma_t *dma)
{
	nvme_free_dma_common(dma);
	kmem_free(dma, sizeof (*dma));
}

/* ARGSUSED */
static void
nvme_prp_dma_destructor(void *buf, void *private)
{
	nvme_dma_t *dma = (nvme_dma_t *)buf;

	nvme_free_dma_common(dma);
}

static int
nvme_alloc_dma_common(nvme_t *nvme, nvme_dma_t *dma,
    size_t len, uint_t flags, ddi_dma_attr_t *dma_attr)
{
	if (ddi_dma_alloc_handle(nvme->n_dip, dma_attr, DDI_DMA_SLEEP, NULL,
	    &dma->nd_dmah) != DDI_SUCCESS) {
		/*
		 * Due to DDI_DMA_SLEEP this can't be DDI_DMA_NORESOURCES, and
		 * the only other possible error is DDI_DMA_BADATTR which
		 * indicates a driver bug which should cause a panic.
		 */
		dev_err(nvme->n_dip, CE_PANIC,
		    "!failed to get DMA handle, check DMA attributes");
		return (DDI_FAILURE);
	}

	/*
	 * ddi_dma_mem_alloc() can only fail when DDI_DMA_NOSLEEP is specified
	 * or the flags are conflicting, which isn't the case here.
	 */
	(void) ddi_dma_mem_alloc(dma->nd_dmah, len, &nvme->n_reg_acc_attr,
	    DDI_DMA_CONSISTENT, DDI_DMA_SLEEP, NULL, &dma->nd_memp,
	    &dma->nd_len, &dma->nd_acch);

	if (ddi_dma_addr_bind_handle(dma->nd_dmah, NULL, dma->nd_memp,
	    dma->nd_len, flags | DDI_DMA_CONSISTENT, DDI_DMA_SLEEP, NULL,
	    &dma->nd_cookie, &dma->nd_ncookie) != DDI_DMA_MAPPED) {
		dev_err(nvme->n_dip, CE_WARN,
		    "!failed to bind DMA memory");
		atomic_inc_32(&nvme->n_dma_bind_err);
		nvme_free_dma_common(dma);
		return (DDI_FAILURE);
	}

	return (DDI_SUCCESS);
}

static int
nvme_zalloc_dma(nvme_t *nvme, size_t len, uint_t flags,
    ddi_dma_attr_t *dma_attr, nvme_dma_t **ret)
{
	nvme_dma_t *dma = kmem_zalloc(sizeof (nvme_dma_t), KM_SLEEP);

	if (nvme_alloc_dma_common(nvme, dma, len, flags, dma_attr) !=
	    DDI_SUCCESS) {
		*ret = NULL;
		kmem_free(dma, sizeof (nvme_dma_t));
		return (DDI_FAILURE);
	}

	bzero(dma->nd_memp, dma->nd_len);

	*ret = dma;
	return (DDI_SUCCESS);
}

/* ARGSUSED */
static int
nvme_prp_dma_constructor(void *buf, void *private, int flags)
{
	nvme_dma_t *dma = (nvme_dma_t *)buf;
	nvme_t *nvme = (nvme_t *)private;

	dma->nd_dmah = NULL;
	dma->nd_acch = NULL;

	if (nvme_alloc_dma_common(nvme, dma, nvme->n_pagesize,
	    DDI_DMA_READ, &nvme->n_prp_dma_attr) != DDI_SUCCESS) {
		return (-1);
	}

	ASSERT(dma->nd_ncookie == 1);

	dma->nd_cached = B_TRUE;

	return (0);
}

static int
nvme_zalloc_queue_dma(nvme_t *nvme, uint32_t nentry, uint16_t qe_len,
    uint_t flags, nvme_dma_t **dma)
{
	uint32_t len = nentry * qe_len;
	ddi_dma_attr_t q_dma_attr = nvme->n_queue_dma_attr;

	len = roundup(len, nvme->n_pagesize);

	if (nvme_zalloc_dma(nvme, len, flags, &q_dma_attr, dma)
	    != DDI_SUCCESS) {
		dev_err(nvme->n_dip, CE_WARN,
		    "!failed to get DMA memory for queue");
		goto fail;
	}

	if ((*dma)->nd_ncookie != 1) {
		dev_err(nvme->n_dip, CE_WARN,
		    "!got too many cookies for queue DMA");
		goto fail;
	}

	return (DDI_SUCCESS);

fail:
	if (*dma) {
		nvme_free_dma(*dma);
		*dma = NULL;
	}

	return (DDI_FAILURE);
}

static void
nvme_free_cq(nvme_cq_t *cq)
{
	mutex_destroy(&cq->ncq_mutex);

	if (cq->ncq_cmd_taskq != NULL)
		taskq_destroy(cq->ncq_cmd_taskq);

	if (cq->ncq_dma != NULL)
		nvme_free_dma(cq->ncq_dma);

	kmem_free(cq, sizeof (*cq));
}

static void
nvme_free_qpair(nvme_qpair_t *qp)
{
	int i;

	mutex_destroy(&qp->nq_mutex);
	sema_destroy(&qp->nq_sema);

	if (qp->nq_sqdma != NULL)
		nvme_free_dma(qp->nq_sqdma);

	if (qp->nq_active_cmds > 0)
		for (i = 0; i != qp->nq_nentry; i++)
			if (qp->nq_cmd[i] != NULL)
				nvme_free_cmd(qp->nq_cmd[i]);

	if (qp->nq_cmd != NULL)
		kmem_free(qp->nq_cmd, sizeof (nvme_cmd_t *) * qp->nq_nentry);

	kmem_free(qp, sizeof (nvme_qpair_t));
}

/*
 * Destroy the pre-allocated cq array, but only free individual completion
 * queues from the given starting index.
 */
static void
nvme_destroy_cq_array(nvme_t *nvme, uint_t start)
{
	uint_t i;

	for (i = start; i < nvme->n_cq_count; i++)
		if (nvme->n_cq[i] != NULL)
			nvme_free_cq(nvme->n_cq[i]);

	kmem_free(nvme->n_cq, sizeof (*nvme->n_cq) * nvme->n_cq_count);
}

static int
nvme_alloc_cq(nvme_t *nvme, uint32_t nentry, nvme_cq_t **cqp, uint16_t idx,
    uint_t nthr)
{
	nvme_cq_t *cq = kmem_zalloc(sizeof (*cq), KM_SLEEP);
	char name[64];		/* large enough for the taskq name */

	mutex_init(&cq->ncq_mutex, NULL, MUTEX_DRIVER,
	    DDI_INTR_PRI(nvme->n_intr_pri));

	if (nvme_zalloc_queue_dma(nvme, nentry, sizeof (nvme_cqe_t),
	    DDI_DMA_READ, &cq->ncq_dma) != DDI_SUCCESS)
		goto fail;

	cq->ncq_cq = (nvme_cqe_t *)cq->ncq_dma->nd_memp;
	cq->ncq_nentry = nentry;
	cq->ncq_id = idx;
	cq->ncq_hdbl = NVME_REG_CQHDBL(nvme, idx);

	/*
	 * Each completion queue has its own command taskq.
	 */
	(void) snprintf(name, sizeof (name), "%s%d_cmd_taskq%u",
	    ddi_driver_name(nvme->n_dip), ddi_get_instance(nvme->n_dip), idx);

	cq->ncq_cmd_taskq = taskq_create(name, nthr, minclsyspri, 64, INT_MAX,
	    TASKQ_PREPOPULATE);

	if (cq->ncq_cmd_taskq == NULL) {
		dev_err(nvme->n_dip, CE_WARN, "!failed to create cmd "
		    "taskq for cq %u", idx);
		goto fail;
	}

	*cqp = cq;
	return (DDI_SUCCESS);

fail:
	nvme_free_cq(cq);
	*cqp = NULL;

	return (DDI_FAILURE);
}

/*
 * Create the n_cq array big enough to hold "ncq" completion queues.
 * If the array already exists it will be re-sized (but only larger).
 * The admin queue is included in this array, which boosts the
 * max number of entries to UINT16_MAX + 1.
 */
static int
nvme_create_cq_array(nvme_t *nvme, uint_t ncq, uint32_t nentry, uint_t nthr)
{
	nvme_cq_t **cq;
	uint_t i, cq_count;

	ASSERT3U(ncq, >, nvme->n_cq_count);

	cq = nvme->n_cq;
	cq_count = nvme->n_cq_count;

	nvme->n_cq = kmem_zalloc(sizeof (*nvme->n_cq) * ncq, KM_SLEEP);
	nvme->n_cq_count = ncq;

	for (i = 0; i < cq_count; i++)
		nvme->n_cq[i] = cq[i];

	for (; i < nvme->n_cq_count; i++)
		if (nvme_alloc_cq(nvme, nentry, &nvme->n_cq[i], i, nthr) !=
		    DDI_SUCCESS)
			goto fail;

	if (cq != NULL)
		kmem_free(cq, sizeof (*cq) * cq_count);

	return (DDI_SUCCESS);

fail:
	nvme_destroy_cq_array(nvme, cq_count);
	/*
	 * Restore the original array
	 */
	nvme->n_cq_count = cq_count;
	nvme->n_cq = cq;

	return (DDI_FAILURE);
}

static int
nvme_alloc_qpair(nvme_t *nvme, uint32_t nentry, nvme_qpair_t **nqp,
    uint_t idx)
{
	nvme_qpair_t *qp = kmem_zalloc(sizeof (*qp), KM_SLEEP);
	uint_t cq_idx;

	mutex_init(&qp->nq_mutex, NULL, MUTEX_DRIVER,
	    DDI_INTR_PRI(nvme->n_intr_pri));

	/*
	 * The NVMe spec defines that a full queue has one empty (unused) slot;
	 * initialize the semaphore accordingly.
	 */
	sema_init(&qp->nq_sema, nentry - 1, NULL, SEMA_DRIVER, NULL);

	if (nvme_zalloc_queue_dma(nvme, nentry, sizeof (nvme_sqe_t),
	    DDI_DMA_WRITE, &qp->nq_sqdma) != DDI_SUCCESS)
		goto fail;

	/*
	 * idx == 0 is adminq, those above 0 are shared io completion queues.
	 */
	cq_idx = idx == 0 ? 0 : 1 + (idx - 1) % (nvme->n_cq_count - 1);
	qp->nq_cq = nvme->n_cq[cq_idx];
	qp->nq_sq = (nvme_sqe_t *)qp->nq_sqdma->nd_memp;
	qp->nq_nentry = nentry;

	qp->nq_sqtdbl = NVME_REG_SQTDBL(nvme, idx);

	qp->nq_cmd = kmem_zalloc(sizeof (nvme_cmd_t *) * nentry, KM_SLEEP);
	qp->nq_next_cmd = 0;

	*nqp = qp;
	return (DDI_SUCCESS);

fail:
	nvme_free_qpair(qp);
	*nqp = NULL;

	return (DDI_FAILURE);
}

static nvme_cmd_t *
nvme_alloc_cmd(nvme_t *nvme, int kmflag)
{
	nvme_cmd_t *cmd = kmem_cache_alloc(nvme_cmd_cache, kmflag);

	if (cmd == NULL)
		return (cmd);

	bzero(cmd, sizeof (nvme_cmd_t));

	cmd->nc_nvme = nvme;

	mutex_init(&cmd->nc_mutex, NULL, MUTEX_DRIVER,
	    DDI_INTR_PRI(nvme->n_intr_pri));
	cv_init(&cmd->nc_cv, NULL, CV_DRIVER, NULL);

	return (cmd);
}

static void
nvme_free_cmd(nvme_cmd_t *cmd)
{
	/* Don't free commands on the lost commands list. */
	if (list_link_active(&cmd->nc_list))
		return;

	if (cmd->nc_dma) {
		nvme_free_dma(cmd->nc_dma);
		cmd->nc_dma = NULL;
	}

	if (cmd->nc_prp) {
		kmem_cache_free(cmd->nc_nvme->n_prp_cache, cmd->nc_prp);
		cmd->nc_prp = NULL;
	}

	cv_destroy(&cmd->nc_cv);
	mutex_destroy(&cmd->nc_mutex);

	kmem_cache_free(nvme_cmd_cache, cmd);
}

static void
nvme_submit_admin_cmd(nvme_qpair_t *qp, nvme_cmd_t *cmd)
{
	sema_p(&qp->nq_sema);
	nvme_submit_cmd_common(qp, cmd);
}

static int
nvme_submit_io_cmd(nvme_qpair_t *qp, nvme_cmd_t *cmd)
{
	if (cmd->nc_nvme->n_dead) {
		return (EIO);
	}

	if (sema_tryp(&qp->nq_sema) == 0)
		return (EAGAIN);

	nvme_submit_cmd_common(qp, cmd);
	return (0);
}

static void
nvme_submit_cmd_common(nvme_qpair_t *qp, nvme_cmd_t *cmd)
{
	nvme_reg_sqtdbl_t tail = { 0 };

	mutex_enter(&qp->nq_mutex);
	cmd->nc_completed = B_FALSE;

	/*
	 * Now that we hold the queue pair lock, we must check whether or not
	 * the controller has been listed as dead (e.g. was removed due to
	 * hotplug). This is necessary as otherwise we could race with
	 * nvme_remove_callback(). Because this has not been enqueued, we don't
	 * call nvme_unqueue_cmd(), which is why we must manually decrement the
	 * semaphore.
	 */
	if (cmd->nc_nvme->n_dead) {
		taskq_dispatch_ent(qp->nq_cq->ncq_cmd_taskq, cmd->nc_callback,
		    cmd, TQ_NOSLEEP, &cmd->nc_tqent);
		sema_v(&qp->nq_sema);
		mutex_exit(&qp->nq_mutex);
		return;
	}

	/*
	 * Try to insert the cmd into the active cmd array at the nq_next_cmd
	 * slot. If the slot is already occupied advance to the next slot and
	 * try again. This can happen for long running commands like async event
	 * requests.
	 */
	while (qp->nq_cmd[qp->nq_next_cmd] != NULL)
		qp->nq_next_cmd = (qp->nq_next_cmd + 1) % qp->nq_nentry;
	qp->nq_cmd[qp->nq_next_cmd] = cmd;

	qp->nq_active_cmds++;

	cmd->nc_sqe.sqe_cid = qp->nq_next_cmd;
	bcopy(&cmd->nc_sqe, &qp->nq_sq[qp->nq_sqtail], sizeof (nvme_sqe_t));
	(void) ddi_dma_sync(qp->nq_sqdma->nd_dmah,
	    sizeof (nvme_sqe_t) * qp->nq_sqtail,
	    sizeof (nvme_sqe_t), DDI_DMA_SYNC_FORDEV);
	qp->nq_next_cmd = (qp->nq_next_cmd + 1) % qp->nq_nentry;

	tail.b.sqtdbl_sqt = qp->nq_sqtail = (qp->nq_sqtail + 1) % qp->nq_nentry;
	nvme_put32(cmd->nc_nvme, qp->nq_sqtdbl, tail.r);

	mutex_exit(&qp->nq_mutex);
}

static nvme_cmd_t *
nvme_unqueue_cmd(nvme_t *nvme, nvme_qpair_t *qp, int cid)
{
	nvme_cmd_t *cmd;

	ASSERT(mutex_owned(&qp->nq_mutex));
	ASSERT3S(cid, <, qp->nq_nentry);

	cmd = qp->nq_cmd[cid];
	/*
	 * Some controllers will erroneously add things to the completion queue
	 * for which there is no matching outstanding command. If this happens,
	 * it is almost certainly a controller firmware bug since nq_mutex
	 * is held across command submission and ringing the queue doorbell,
	 * and is also held in this function.
	 *
	 * If we see such an unexpected command, there is not much we can do.
	 * These will be logged and counted in nvme_get_completed(), but
	 * otherwise ignored.
	 */
	if (cmd == NULL)
		return (NULL);
	qp->nq_cmd[cid] = NULL;
	ASSERT3U(qp->nq_active_cmds, >, 0);
	qp->nq_active_cmds--;
	sema_v(&qp->nq_sema);

	ASSERT3P(cmd, !=, NULL);
	ASSERT3P(cmd->nc_nvme, ==, nvme);
	ASSERT3S(cmd->nc_sqe.sqe_cid, ==, cid);

	return (cmd);
}

/*
 * Get the command tied to the next completed cqe and bump along completion
 * queue head counter.
 */
static nvme_cmd_t *
nvme_get_completed(nvme_t *nvme, nvme_cq_t *cq)
{
	nvme_qpair_t *qp;
	nvme_cqe_t *cqe;
	nvme_cmd_t *cmd;

	ASSERT(mutex_owned(&cq->ncq_mutex));

retry:
	cqe = &cq->ncq_cq[cq->ncq_head];

	/* Check phase tag of CQE. Hardware inverts it for new entries. */
	if (cqe->cqe_sf.sf_p == cq->ncq_phase)
		return (NULL);

	qp = nvme->n_ioq[cqe->cqe_sqid];

	mutex_enter(&qp->nq_mutex);
	cmd = nvme_unqueue_cmd(nvme, qp, cqe->cqe_cid);
	mutex_exit(&qp->nq_mutex);

	qp->nq_sqhead = cqe->cqe_sqhd;
	cq->ncq_head = (cq->ncq_head + 1) % cq->ncq_nentry;

	/* Toggle phase on wrap-around. */
	if (cq->ncq_head == 0)
		cq->ncq_phase = cq->ncq_phase != 0 ? 0 : 1;

	if (cmd == NULL) {
		dev_err(nvme->n_dip, CE_WARN,
		    "!received completion for unknown cid 0x%x", cqe->cqe_cid);
		atomic_inc_32(&nvme->n_unknown_cid);
		/*
		 * We want to ignore this unexpected completion entry as it
		 * is most likely a result of a bug in the controller firmware.
		 * However, if we return NULL, then callers will assume there
		 * are no more pending commands for this wakeup. Retry to keep
		 * enumerating commands until the phase tag indicates there are
		 * no more and we are really done.
		 */
		goto retry;
	}

	ASSERT3U(cmd->nc_sqid, ==, cqe->cqe_sqid);
	bcopy(cqe, &cmd->nc_cqe, sizeof (nvme_cqe_t));

	return (cmd);
}

/*
 * Process all completed commands on the io completion queue.
 */
static uint_t
nvme_process_iocq(nvme_t *nvme, nvme_cq_t *cq)
{
	nvme_reg_cqhdbl_t head = { 0 };
	nvme_cmd_t *cmd;
	uint_t completed = 0;

	if (ddi_dma_sync(cq->ncq_dma->nd_dmah, 0, 0, DDI_DMA_SYNC_FORKERNEL) !=
	    DDI_SUCCESS)
		dev_err(nvme->n_dip, CE_WARN, "!ddi_dma_sync() failed in %s",
		    __func__);

	mutex_enter(&cq->ncq_mutex);

	while ((cmd = nvme_get_completed(nvme, cq)) != NULL) {
		taskq_dispatch_ent(cq->ncq_cmd_taskq, cmd->nc_callback, cmd,
		    TQ_NOSLEEP, &cmd->nc_tqent);

		completed++;
	}

	if (completed > 0) {
		/*
		 * Update the completion queue head doorbell.
		 */
		head.b.cqhdbl_cqh = cq->ncq_head;
		nvme_put32(nvme, cq->ncq_hdbl, head.r);
	}

	mutex_exit(&cq->ncq_mutex);

	return (completed);
}

static nvme_cmd_t *
nvme_retrieve_cmd(nvme_t *nvme, nvme_qpair_t *qp)
{
	nvme_cq_t *cq = qp->nq_cq;
	nvme_reg_cqhdbl_t head = { 0 };
	nvme_cmd_t *cmd;

	if (ddi_dma_sync(cq->ncq_dma->nd_dmah, 0, 0, DDI_DMA_SYNC_FORKERNEL) !=
	    DDI_SUCCESS)
		dev_err(nvme->n_dip, CE_WARN, "!ddi_dma_sync() failed in %s",
		    __func__);

	mutex_enter(&cq->ncq_mutex);

	if ((cmd = nvme_get_completed(nvme, cq)) != NULL) {
		head.b.cqhdbl_cqh = cq->ncq_head;
		nvme_put32(nvme, cq->ncq_hdbl, head.r);
	}

	mutex_exit(&cq->ncq_mutex);

	return (cmd);
}

static int
nvme_check_unknown_cmd_status(nvme_cmd_t *cmd)
{
	nvme_cqe_t *cqe = &cmd->nc_cqe;

	dev_err(cmd->nc_nvme->n_dip, CE_WARN,
	    "!unknown command status received: opc = %x, sqid = %d, cid = %d, "
	    "sc = %x, sct = %x, dnr = %d, m = %d", cmd->nc_sqe.sqe_opc,
	    cqe->cqe_sqid, cqe->cqe_cid, cqe->cqe_sf.sf_sc, cqe->cqe_sf.sf_sct,
	    cqe->cqe_sf.sf_dnr, cqe->cqe_sf.sf_m);

	if (cmd->nc_xfer != NULL)
		bd_error(cmd->nc_xfer, BD_ERR_ILLRQ);

	if (cmd->nc_nvme->n_strict_version) {
		cmd->nc_nvme->n_dead = B_TRUE;
		ddi_fm_service_impact(cmd->nc_nvme->n_dip, DDI_SERVICE_LOST);
	}

	return (EIO);
}

static int
nvme_check_vendor_cmd_status(nvme_cmd_t *cmd)
{
	nvme_cqe_t *cqe = &cmd->nc_cqe;

	dev_err(cmd->nc_nvme->n_dip, CE_WARN,
	    "!unknown command status received: opc = %x, sqid = %d, cid = %d, "
	    "sc = %x, sct = %x, dnr = %d, m = %d", cmd->nc_sqe.sqe_opc,
	    cqe->cqe_sqid, cqe->cqe_cid, cqe->cqe_sf.sf_sc, cqe->cqe_sf.sf_sct,
	    cqe->cqe_sf.sf_dnr, cqe->cqe_sf.sf_m);
	if (!cmd->nc_nvme->n_ignore_unknown_vendor_status) {
		cmd->nc_nvme->n_dead = B_TRUE;
		ddi_fm_service_impact(cmd->nc_nvme->n_dip, DDI_SERVICE_LOST);
	}

	return (EIO);
}

static int
nvme_check_integrity_cmd_status(nvme_cmd_t *cmd)
{
	nvme_cqe_t *cqe = &cmd->nc_cqe;

	switch (cqe->cqe_sf.sf_sc) {
	case NVME_CQE_SC_INT_NVM_WRITE:
		/* write fail */
		/* TODO: post ereport */
		if (cmd->nc_xfer != NULL)
			bd_error(cmd->nc_xfer, BD_ERR_MEDIA);
		return (EIO);

	case NVME_CQE_SC_INT_NVM_READ:
		/* read fail */
		/* TODO: post ereport */
		if (cmd->nc_xfer != NULL)
			bd_error(cmd->nc_xfer, BD_ERR_MEDIA);
		return (EIO);

	default:
		return (nvme_check_unknown_cmd_status(cmd));
	}
}

static int
nvme_check_generic_cmd_status(nvme_cmd_t *cmd)
{
	nvme_cqe_t *cqe = &cmd->nc_cqe;

	switch (cqe->cqe_sf.sf_sc) {
	case NVME_CQE_SC_GEN_SUCCESS:
		return (0);

	/*
	 * Errors indicating a bug in the driver should cause a panic.
	 */
	case NVME_CQE_SC_GEN_INV_OPC:
		/* Invalid Command Opcode */
		if (!cmd->nc_dontpanic)
			dev_err(cmd->nc_nvme->n_dip, CE_PANIC,
			    "programming error: invalid opcode in cmd %p",
			    (void *)cmd);
		return (EINVAL);

	case NVME_CQE_SC_GEN_INV_FLD:
		/* Invalid Field in Command */
		if (!cmd->nc_dontpanic)
			dev_err(cmd->nc_nvme->n_dip, CE_PANIC,
			    "programming error: invalid field in cmd %p",
			    (void *)cmd);
		return (EIO);

	case NVME_CQE_SC_GEN_ID_CNFL:
		/* Command ID Conflict */
		dev_err(cmd->nc_nvme->n_dip, CE_PANIC, "programming error: "
		    "cmd ID conflict in cmd %p", (void *)cmd);
		return (0);

	case NVME_CQE_SC_GEN_INV_NS:
		/* Invalid Namespace or Format */
		if (!cmd->nc_dontpanic)
			dev_err(cmd->nc_nvme->n_dip, CE_PANIC,
			    "programming error: invalid NS/format in cmd %p",
			    (void *)cmd);
		return (EINVAL);

	case NVME_CQE_SC_GEN_NVM_LBA_RANGE:
		/* LBA Out Of Range */
		dev_err(cmd->nc_nvme->n_dip, CE_PANIC, "programming error: "
		    "LBA out of range in cmd %p", (void *)cmd);
		return (0);

	/*
	 * Non-fatal errors, handle gracefully.
	 */
	case NVME_CQE_SC_GEN_DATA_XFR_ERR:
		/* Data Transfer Error (DMA) */
		/* TODO: post ereport */
		atomic_inc_32(&cmd->nc_nvme->n_data_xfr_err);
		if (cmd->nc_xfer != NULL)
			bd_error(cmd->nc_xfer, BD_ERR_NTRDY);
		return (EIO);

	case NVME_CQE_SC_GEN_INTERNAL_ERR:
		/*
		 * Internal Error. The spec (v1.0, section 4.5.1.2) says
		 * detailed error information is returned as async event,
		 * so we pretty much ignore the error here and handle it
		 * in the async event handler.
		 */
		atomic_inc_32(&cmd->nc_nvme->n_internal_err);
		if (cmd->nc_xfer != NULL)
			bd_error(cmd->nc_xfer, BD_ERR_NTRDY);
		return (EIO);

	case NVME_CQE_SC_GEN_ABORT_REQUEST:
		/*
		 * Command Abort Requested. This normally happens only when a
		 * command times out.
		 */
		/* TODO: post ereport or change blkdev to handle this? */
		atomic_inc_32(&cmd->nc_nvme->n_abort_rq_err);
		return (ECANCELED);

	case NVME_CQE_SC_GEN_ABORT_PWRLOSS:
		/* Command Aborted due to Power Loss Notification */
		ddi_fm_service_impact(cmd->nc_nvme->n_dip, DDI_SERVICE_LOST);
		cmd->nc_nvme->n_dead = B_TRUE;
		return (EIO);

	case NVME_CQE_SC_GEN_ABORT_SQ_DEL:
		/* Command Aborted due to SQ Deletion */
		atomic_inc_32(&cmd->nc_nvme->n_abort_sq_del);
		return (EIO);

	case NVME_CQE_SC_GEN_NVM_CAP_EXC:
		/* Capacity Exceeded */
		atomic_inc_32(&cmd->nc_nvme->n_nvm_cap_exc);
		if (cmd->nc_xfer != NULL)
			bd_error(cmd->nc_xfer, BD_ERR_MEDIA);
		return (EIO);

	case NVME_CQE_SC_GEN_NVM_NS_NOTRDY:
		/* Namespace Not Ready */
		atomic_inc_32(&cmd->nc_nvme->n_nvm_ns_notrdy);
		if (cmd->nc_xfer != NULL)
			bd_error(cmd->nc_xfer, BD_ERR_NTRDY);
		return (EIO);

	case NVME_CQE_SC_GEN_NVM_FORMATTING:
		/* Format in progress (1.2) */
		if (!NVME_VERSION_ATLEAST(&cmd->nc_nvme->n_version, 1, 2))
			return (nvme_check_unknown_cmd_status(cmd));
		atomic_inc_32(&cmd->nc_nvme->n_nvm_ns_formatting);
		if (cmd->nc_xfer != NULL)
			bd_error(cmd->nc_xfer, BD_ERR_NTRDY);
		return (EIO);

	default:
		return (nvme_check_unknown_cmd_status(cmd));
	}
}

static int
nvme_check_specific_cmd_status(nvme_cmd_t *cmd)
{
	nvme_cqe_t *cqe = &cmd->nc_cqe;

	switch (cqe->cqe_sf.sf_sc) {
	case NVME_CQE_SC_SPC_INV_CQ:
		/* Completion Queue Invalid */
		ASSERT(cmd->nc_sqe.sqe_opc == NVME_OPC_CREATE_SQUEUE);
		atomic_inc_32(&cmd->nc_nvme->n_inv_cq_err);
		return (EINVAL);

	case NVME_CQE_SC_SPC_INV_QID:
		/* Invalid Queue Identifier */
		ASSERT(cmd->nc_sqe.sqe_opc == NVME_OPC_CREATE_SQUEUE ||
		    cmd->nc_sqe.sqe_opc == NVME_OPC_DELETE_SQUEUE ||
		    cmd->nc_sqe.sqe_opc == NVME_OPC_CREATE_CQUEUE ||
		    cmd->nc_sqe.sqe_opc == NVME_OPC_DELETE_CQUEUE);
		atomic_inc_32(&cmd->nc_nvme->n_inv_qid_err);
		return (EINVAL);

	case NVME_CQE_SC_SPC_MAX_QSZ_EXC:
		/* Max Queue Size Exceeded */
		ASSERT(cmd->nc_sqe.sqe_opc == NVME_OPC_CREATE_SQUEUE ||
		    cmd->nc_sqe.sqe_opc == NVME_OPC_CREATE_CQUEUE);
		atomic_inc_32(&cmd->nc_nvme->n_max_qsz_exc);
		return (EINVAL);

	case NVME_CQE_SC_SPC_ABRT_CMD_EXC:
		/* Abort Command Limit Exceeded */
		ASSERT(cmd->nc_sqe.sqe_opc == NVME_OPC_ABORT);
		dev_err(cmd->nc_nvme->n_dip, CE_PANIC, "programming error: "
		    "abort command limit exceeded in cmd %p", (void *)cmd);
		return (0);

	case NVME_CQE_SC_SPC_ASYNC_EVREQ_EXC:
		/* Async Event Request Limit Exceeded */
		ASSERT(cmd->nc_sqe.sqe_opc == NVME_OPC_ASYNC_EVENT);
		dev_err(cmd->nc_nvme->n_dip, CE_PANIC, "programming error: "
		    "async event request limit exceeded in cmd %p",
		    (void *)cmd);
		return (0);

	case NVME_CQE_SC_SPC_INV_INT_VECT:
		/* Invalid Interrupt Vector */
		ASSERT(cmd->nc_sqe.sqe_opc == NVME_OPC_CREATE_CQUEUE);
		atomic_inc_32(&cmd->nc_nvme->n_inv_int_vect);
		return (EINVAL);

	case NVME_CQE_SC_SPC_INV_LOG_PAGE:
		/* Invalid Log Page */
		ASSERT(cmd->nc_sqe.sqe_opc == NVME_OPC_GET_LOG_PAGE);
		atomic_inc_32(&cmd->nc_nvme->n_inv_log_page);
		return (EINVAL);

	case NVME_CQE_SC_SPC_INV_FORMAT:
		/* Invalid Format */
		ASSERT(cmd->nc_sqe.sqe_opc == NVME_OPC_NVM_FORMAT);
		atomic_inc_32(&cmd->nc_nvme->n_inv_format);
		if (cmd->nc_xfer != NULL)
			bd_error(cmd->nc_xfer, BD_ERR_ILLRQ);
		return (EINVAL);

	case NVME_CQE_SC_SPC_INV_Q_DEL:
		/* Invalid Queue Deletion */
		ASSERT(cmd->nc_sqe.sqe_opc == NVME_OPC_DELETE_CQUEUE);
		atomic_inc_32(&cmd->nc_nvme->n_inv_q_del);
		return (EINVAL);

	case NVME_CQE_SC_SPC_NVM_CNFL_ATTR:
		/* Conflicting Attributes */
		ASSERT(cmd->nc_sqe.sqe_opc == NVME_OPC_NVM_DSET_MGMT ||
		    cmd->nc_sqe.sqe_opc == NVME_OPC_NVM_READ ||
		    cmd->nc_sqe.sqe_opc == NVME_OPC_NVM_WRITE);
		atomic_inc_32(&cmd->nc_nvme->n_cnfl_attr);
		if (cmd->nc_xfer != NULL)
			bd_error(cmd->nc_xfer, BD_ERR_ILLRQ);
		return (EINVAL);

	case NVME_CQE_SC_SPC_NVM_INV_PROT:
		/* Invalid Protection Information */
		ASSERT(cmd->nc_sqe.sqe_opc == NVME_OPC_NVM_COMPARE ||
		    cmd->nc_sqe.sqe_opc == NVME_OPC_NVM_READ ||
		    cmd->nc_sqe.sqe_opc == NVME_OPC_NVM_WRITE);
		atomic_inc_32(&cmd->nc_nvme->n_inv_prot);
		if (cmd->nc_xfer != NULL)
			bd_error(cmd->nc_xfer, BD_ERR_ILLRQ);
		return (EINVAL);

	case NVME_CQE_SC_SPC_NVM_READONLY:
		/* Write to Read Only Range */
		ASSERT(cmd->nc_sqe.sqe_opc == NVME_OPC_NVM_WRITE);
		atomic_inc_32(&cmd->nc_nvme->n_readonly);
		if (cmd->nc_xfer != NULL)
			bd_error(cmd->nc_xfer, BD_ERR_ILLRQ);
		return (EROFS);

	case NVME_CQE_SC_SPC_INV_FW_SLOT:
		/* Invalid Firmware Slot */
		ASSERT(cmd->nc_sqe.sqe_opc == NVME_OPC_FW_ACTIVATE);
		return (EINVAL);

	case NVME_CQE_SC_SPC_INV_FW_IMG:
		/* Invalid Firmware Image */
		ASSERT(cmd->nc_sqe.sqe_opc == NVME_OPC_FW_ACTIVATE);
		return (EINVAL);

	case NVME_CQE_SC_SPC_FW_RESET:
		/* Conventional Reset Required */
		ASSERT(cmd->nc_sqe.sqe_opc == NVME_OPC_FW_ACTIVATE);
		return (0);

	case NVME_CQE_SC_SPC_FW_NSSR:
		/* NVMe Subsystem Reset Required */
		ASSERT(cmd->nc_sqe.sqe_opc == NVME_OPC_FW_ACTIVATE);
		return (0);

	case NVME_CQE_SC_SPC_FW_NEXT_RESET:
		/* Activation Requires Reset */
		ASSERT(cmd->nc_sqe.sqe_opc == NVME_OPC_FW_ACTIVATE);
		return (0);

	case NVME_CQE_SC_SPC_FW_MTFA:
		/* Activation Requires Maximum Time Violation */
		ASSERT(cmd->nc_sqe.sqe_opc == NVME_OPC_FW_ACTIVATE);
		return (EAGAIN);

	case NVME_CQE_SC_SPC_FW_PROHIBITED:
		/* Activation Prohibited */
		ASSERT(cmd->nc_sqe.sqe_opc == NVME_OPC_FW_ACTIVATE);
		return (EINVAL);

	case NVME_CQE_SC_SPC_FW_OVERLAP:
		/* Overlapping Firmware Ranges */
		ASSERT(cmd->nc_sqe.sqe_opc == NVME_OPC_FW_IMAGE_LOAD);
		return (EINVAL);

	default:
		return (nvme_check_unknown_cmd_status(cmd));
	}
}

static inline int
nvme_check_cmd_status(nvme_cmd_t *cmd)
{
	nvme_cqe_t *cqe = &cmd->nc_cqe;

	/*
	 * Take a shortcut if the controller is dead, or if
	 * command status indicates no error.
	 */
	if (cmd->nc_nvme->n_dead)
		return (EIO);

	if (cqe->cqe_sf.sf_sct == NVME_CQE_SCT_GENERIC &&
	    cqe->cqe_sf.sf_sc == NVME_CQE_SC_GEN_SUCCESS)
		return (0);

	if (cqe->cqe_sf.sf_sct == NVME_CQE_SCT_GENERIC)
		return (nvme_check_generic_cmd_status(cmd));
	else if (cqe->cqe_sf.sf_sct == NVME_CQE_SCT_SPECIFIC)
		return (nvme_check_specific_cmd_status(cmd));
	else if (cqe->cqe_sf.sf_sct == NVME_CQE_SCT_INTEGRITY)
		return (nvme_check_integrity_cmd_status(cmd));
	else if (cqe->cqe_sf.sf_sct == NVME_CQE_SCT_VENDOR)
		return (nvme_check_vendor_cmd_status(cmd));

	return (nvme_check_unknown_cmd_status(cmd));
}

static int
nvme_abort_cmd(nvme_cmd_t *abort_cmd, uint_t sec)
{
	nvme_t *nvme = abort_cmd->nc_nvme;
	nvme_cmd_t *cmd = nvme_alloc_cmd(nvme, KM_SLEEP);
	nvme_abort_cmd_t ac = { 0 };
	int ret = 0;

	sema_p(&nvme->n_abort_sema);

	ac.b.ac_cid = abort_cmd->nc_sqe.sqe_cid;
	ac.b.ac_sqid = abort_cmd->nc_sqid;

	cmd->nc_sqid = 0;
	cmd->nc_sqe.sqe_opc = NVME_OPC_ABORT;
	cmd->nc_callback = nvme_wakeup_cmd;
	cmd->nc_sqe.sqe_cdw10 = ac.r;

	/*
	 * Send the ABORT to the hardware. The ABORT command will return _after_
	 * the aborted command has completed (aborted or otherwise), but since
	 * we still hold the aborted command's mutex its callback hasn't been
	 * processed yet.
	 */
	nvme_admin_cmd(cmd, sec);
	sema_v(&nvme->n_abort_sema);

	if ((ret = nvme_check_cmd_status(cmd)) != 0) {
		dev_err(nvme->n_dip, CE_WARN,
		    "!ABORT failed with sct = %x, sc = %x",
		    cmd->nc_cqe.cqe_sf.sf_sct, cmd->nc_cqe.cqe_sf.sf_sc);
		atomic_inc_32(&nvme->n_abort_failed);
	} else {
		dev_err(nvme->n_dip, CE_WARN,
		    "!ABORT of command %d/%d %ssuccessful",
		    abort_cmd->nc_sqe.sqe_cid, abort_cmd->nc_sqid,
		    cmd->nc_cqe.cqe_dw0 & 1 ? "un" : "");
		if ((cmd->nc_cqe.cqe_dw0 & 1) == 0)
			atomic_inc_32(&nvme->n_cmd_aborted);
	}

	nvme_free_cmd(cmd);
	return (ret);
}

/*
 * nvme_wait_cmd -- wait for command completion or timeout
 *
 * In case of a serious error or a timeout of the abort command the hardware
 * will be declared dead and FMA will be notified.
 */
static void
nvme_wait_cmd(nvme_cmd_t *cmd, uint_t sec)
{
	clock_t timeout = ddi_get_lbolt() + drv_usectohz(sec * MICROSEC);
	nvme_t *nvme = cmd->nc_nvme;
	nvme_reg_csts_t csts;
	nvme_qpair_t *qp;

	ASSERT(mutex_owned(&cmd->nc_mutex));

	while (!cmd->nc_completed) {
		if (cv_timedwait(&cmd->nc_cv, &cmd->nc_mutex, timeout) == -1)
			break;
	}

	if (cmd->nc_completed)
		return;

	/*
	 * The command timed out.
	 *
	 * Check controller for fatal status, any errors associated with the
	 * register or DMA handle, or for a double timeout (abort command timed
	 * out). If necessary log a warning and call FMA.
	 */
	csts.r = nvme_get32(nvme, NVME_REG_CSTS);
	dev_err(nvme->n_dip, CE_WARN, "!command %d/%d timeout, "
	    "OPC = %x, CFS = %d", cmd->nc_sqe.sqe_cid, cmd->nc_sqid,
	    cmd->nc_sqe.sqe_opc, csts.b.csts_cfs);
	atomic_inc_32(&nvme->n_cmd_timeout);

	if (csts.b.csts_cfs ||
	    nvme_check_regs_hdl(nvme) ||
	    nvme_check_dma_hdl(cmd->nc_dma) ||
	    cmd->nc_sqe.sqe_opc == NVME_OPC_ABORT) {
		ddi_fm_service_impact(nvme->n_dip, DDI_SERVICE_LOST);
		nvme->n_dead = B_TRUE;
	} else if (nvme_abort_cmd(cmd, sec) == 0) {
		/*
		 * If the abort succeeded the command should complete
		 * immediately with an appropriate status.
		 */
		while (!cmd->nc_completed)
			cv_wait(&cmd->nc_cv, &cmd->nc_mutex);

		return;
	}

	qp = nvme->n_ioq[cmd->nc_sqid];

	mutex_enter(&qp->nq_mutex);
	(void) nvme_unqueue_cmd(nvme, qp, cmd->nc_sqe.sqe_cid);
	mutex_exit(&qp->nq_mutex);

	/*
	 * As we don't know what the presumed dead hardware might still do with
	 * the DMA memory, we'll put the command on the lost commands list if it
	 * has any DMA memory.
	 */
	if (cmd->nc_dma != NULL) {
		mutex_enter(&nvme_lc_mutex);
		list_insert_head(&nvme_lost_cmds, cmd);
		mutex_exit(&nvme_lc_mutex);
	}
}

static void
nvme_wakeup_cmd(void *arg)
{
	nvme_cmd_t *cmd = arg;

	mutex_enter(&cmd->nc_mutex);
	cmd->nc_completed = B_TRUE;
	cv_signal(&cmd->nc_cv);
	mutex_exit(&cmd->nc_mutex);
}

static void
nvme_async_event_task(void *arg)
{
	nvme_cmd_t *cmd = arg;
	nvme_t *nvme = cmd->nc_nvme;
	nvme_error_log_entry_t *error_log = NULL;
	nvme_health_log_t *health_log = NULL;
	nvme_nschange_list_t *nslist = NULL;
	size_t logsize = 0;
	nvme_async_event_t event;

	/*
	 * Check for errors associated with the async request itself. The only
	 * command-specific error is "async event limit exceeded", which
	 * indicates a programming error in the driver and causes a panic in
	 * nvme_check_cmd_status().
	 *
	 * Other possible errors are various scenarios where the async request
	 * was aborted, or internal errors in the device. Internal errors are
	 * reported to FMA, the command aborts need no special handling here.
	 *
	 * And finally, at least qemu nvme does not support async events,
	 * and will return NVME_CQE_SC_GEN_INV_OPC | DNR. If so, we
	 * will avoid posting async events.
	 */

	if (nvme_check_cmd_status(cmd) != 0) {
		dev_err(cmd->nc_nvme->n_dip, CE_WARN,
		    "!async event request returned failure, sct = 0x%x, "
		    "sc = 0x%x, dnr = %d, m = %d", cmd->nc_cqe.cqe_sf.sf_sct,
		    cmd->nc_cqe.cqe_sf.sf_sc, cmd->nc_cqe.cqe_sf.sf_dnr,
		    cmd->nc_cqe.cqe_sf.sf_m);

		if (cmd->nc_cqe.cqe_sf.sf_sct == NVME_CQE_SCT_GENERIC &&
		    cmd->nc_cqe.cqe_sf.sf_sc == NVME_CQE_SC_GEN_INTERNAL_ERR) {
			cmd->nc_nvme->n_dead = B_TRUE;
			ddi_fm_service_impact(cmd->nc_nvme->n_dip,
			    DDI_SERVICE_LOST);
		}

		if (cmd->nc_cqe.cqe_sf.sf_sct == NVME_CQE_SCT_GENERIC &&
		    cmd->nc_cqe.cqe_sf.sf_sc == NVME_CQE_SC_GEN_INV_OPC &&
		    cmd->nc_cqe.cqe_sf.sf_dnr == 1) {
			nvme->n_async_event_supported = B_FALSE;
		}

		nvme_free_cmd(cmd);
		return;
	}

	event.r = cmd->nc_cqe.cqe_dw0;

	/* Clear CQE and re-submit the async request. */
	bzero(&cmd->nc_cqe, sizeof (nvme_cqe_t));
	nvme_submit_admin_cmd(nvme->n_adminq, cmd);
	cmd = NULL;	/* cmd can no longer be used after resubmission */

	switch (event.b.ae_type) {
	case NVME_ASYNC_TYPE_ERROR:
		if (event.b.ae_logpage != NVME_LOGPAGE_ERROR) {
			dev_err(nvme->n_dip, CE_WARN,
			    "!wrong logpage in async event reply: "
			    "type=0x%x logpage=0x%x",
			    event.b.ae_type, event.b.ae_logpage);
			atomic_inc_32(&nvme->n_wrong_logpage);
			break;
		}

		(void) nvme_get_logpage(nvme, B_FALSE, (void **)&error_log,
		    &logsize, event.b.ae_logpage);

		switch (event.b.ae_info) {
		case NVME_ASYNC_ERROR_INV_SQ:
			dev_err(nvme->n_dip, CE_PANIC, "programming error: "
			    "invalid submission queue");
			return;

		case NVME_ASYNC_ERROR_INV_DBL:
			dev_err(nvme->n_dip, CE_PANIC, "programming error: "
			    "invalid doorbell write value");
			return;

		case NVME_ASYNC_ERROR_DIAGFAIL:
			dev_err(nvme->n_dip, CE_WARN, "!diagnostic failure");
			ddi_fm_service_impact(nvme->n_dip, DDI_SERVICE_LOST);
			nvme->n_dead = B_TRUE;
			atomic_inc_32(&nvme->n_diagfail_event);
			break;

		case NVME_ASYNC_ERROR_PERSISTENT:
			dev_err(nvme->n_dip, CE_WARN, "!persistent internal "
			    "device error");
			ddi_fm_service_impact(nvme->n_dip, DDI_SERVICE_LOST);
			nvme->n_dead = B_TRUE;
			atomic_inc_32(&nvme->n_persistent_event);
			break;

		case NVME_ASYNC_ERROR_TRANSIENT:
			dev_err(nvme->n_dip, CE_WARN, "!transient internal "
			    "device error");
			/* TODO: send ereport */
			atomic_inc_32(&nvme->n_transient_event);
			break;

		case NVME_ASYNC_ERROR_FW_LOAD:
			dev_err(nvme->n_dip, CE_WARN,
			    "!firmware image load error");
			atomic_inc_32(&nvme->n_fw_load_event);
			break;
		}
		break;

	case NVME_ASYNC_TYPE_HEALTH:
		if (event.b.ae_logpage != NVME_LOGPAGE_HEALTH) {
			dev_err(nvme->n_dip, CE_WARN,
			    "!wrong logpage in async event reply: "
			    "type=0x%x logpage=0x%x",
			    event.b.ae_type, event.b.ae_logpage);
			atomic_inc_32(&nvme->n_wrong_logpage);
			break;
		}

		(void) nvme_get_logpage(nvme, B_FALSE, (void **)&health_log,
		    &logsize, event.b.ae_logpage, -1);

		switch (event.b.ae_info) {
		case NVME_ASYNC_HEALTH_RELIABILITY:
			dev_err(nvme->n_dip, CE_WARN,
			    "!device reliability compromised");
			/* TODO: send ereport */
			atomic_inc_32(&nvme->n_reliability_event);
			break;

		case NVME_ASYNC_HEALTH_TEMPERATURE:
			dev_err(nvme->n_dip, CE_WARN,
			    "!temperature above threshold");
			/* TODO: send ereport */
			atomic_inc_32(&nvme->n_temperature_event);
			break;

		case NVME_ASYNC_HEALTH_SPARE:
			dev_err(nvme->n_dip, CE_WARN,
			    "!spare space below threshold");
			/* TODO: send ereport */
			atomic_inc_32(&nvme->n_spare_event);
			break;
		}
		break;

	case NVME_ASYNC_TYPE_NOTICE:
		switch (event.b.ae_info) {
		case NVME_ASYNC_NOTICE_NS_CHANGE:
			if (event.b.ae_logpage != NVME_LOGPAGE_NSCHANGE) {
				dev_err(nvme->n_dip, CE_WARN,
				    "!wrong logpage in async event reply: "
				    "type=0x%x logpage=0x%x",
				    event.b.ae_type, event.b.ae_logpage);
				atomic_inc_32(&nvme->n_wrong_logpage);
				break;
			}

			dev_err(nvme->n_dip, CE_NOTE,
			    "namespace attribute change event, "
			    "logpage = 0x%x", event.b.ae_logpage);
			atomic_inc_32(&nvme->n_notice_event);

			if (nvme_get_logpage(nvme, B_FALSE, (void **)&nslist,
			    &logsize, event.b.ae_logpage, -1) != 0) {
				break;
			}

			if (nslist->nscl_ns[0] == UINT32_MAX) {
				dev_err(nvme->n_dip, CE_CONT,
				    "more than %u namespaces have changed.\n",
				    NVME_NSCHANGE_LIST_SIZE);
				break;
			}

			mutex_enter(&nvme->n_mgmt_mutex);
			for (uint_t i = 0; i < NVME_NSCHANGE_LIST_SIZE; i++) {
				uint32_t nsid = nslist->nscl_ns[i];

				if (nsid == 0)	/* end of list */
					break;

				dev_err(nvme->n_dip, CE_NOTE,
				    "!namespace nvme%d/%u has changed.",
				    ddi_get_instance(nvme->n_dip), nsid);


				if (nvme_init_ns(nvme, nsid) != DDI_SUCCESS)
					continue;

				bd_state_change(
				    NVME_NSID2NS(nvme, nsid)->ns_bd_hdl);
			}
			mutex_exit(&nvme->n_mgmt_mutex);

			break;

		case NVME_ASYNC_NOTICE_FW_ACTIVATE:
			dev_err(nvme->n_dip, CE_NOTE,
			    "firmware activation starting, "
			    "logpage = 0x%x", event.b.ae_logpage);
			atomic_inc_32(&nvme->n_notice_event);
			break;

		case NVME_ASYNC_NOTICE_TELEMETRY:
			dev_err(nvme->n_dip, CE_NOTE,
			    "telemetry log changed, "
			    "logpage = 0x%x", event.b.ae_logpage);
			atomic_inc_32(&nvme->n_notice_event);
			break;

		case NVME_ASYNC_NOTICE_NS_ASYMM:
			dev_err(nvme->n_dip, CE_NOTE,
			    "asymmetric namespace access change, "
			    "logpage = 0x%x", event.b.ae_logpage);
			atomic_inc_32(&nvme->n_notice_event);
			break;

		case NVME_ASYNC_NOTICE_LATENCYLOG:
			dev_err(nvme->n_dip, CE_NOTE,
			    "predictable latency event aggregate log change, "
			    "logpage = 0x%x", event.b.ae_logpage);
			atomic_inc_32(&nvme->n_notice_event);
			break;

		case NVME_ASYNC_NOTICE_LBASTATUS:
			dev_err(nvme->n_dip, CE_NOTE,
			    "LBA status information alert, "
			    "logpage = 0x%x", event.b.ae_logpage);
			atomic_inc_32(&nvme->n_notice_event);
			break;

		case NVME_ASYNC_NOTICE_ENDURANCELOG:
			dev_err(nvme->n_dip, CE_NOTE,
			    "endurance group event aggregate log page change, "
			    "logpage = 0x%x", event.b.ae_logpage);
			atomic_inc_32(&nvme->n_notice_event);
			break;

		default:
			dev_err(nvme->n_dip, CE_WARN,
			    "!unknown notice async event received, "
			    "info = 0x%x, logpage = 0x%x", event.b.ae_info,
			    event.b.ae_logpage);
			atomic_inc_32(&nvme->n_unknown_event);
			break;
		}
		break;

	case NVME_ASYNC_TYPE_VENDOR:
		dev_err(nvme->n_dip, CE_WARN, "!vendor specific async event "
		    "received, info = 0x%x, logpage = 0x%x", event.b.ae_info,
		    event.b.ae_logpage);
		atomic_inc_32(&nvme->n_vendor_event);
		break;

	default:
		dev_err(nvme->n_dip, CE_WARN, "!unknown async event received, "
		    "type = 0x%x, info = 0x%x, logpage = 0x%x", event.b.ae_type,
		    event.b.ae_info, event.b.ae_logpage);
		atomic_inc_32(&nvme->n_unknown_event);
		break;
	}

	if (error_log != NULL)
		kmem_free(error_log, logsize);

	if (health_log != NULL)
		kmem_free(health_log, logsize);

	if (nslist != NULL)
		kmem_free(nslist, logsize);
}

static void
nvme_admin_cmd(nvme_cmd_t *cmd, int sec)
{
	mutex_enter(&cmd->nc_mutex);
	nvme_submit_admin_cmd(cmd->nc_nvme->n_adminq, cmd);
	nvme_wait_cmd(cmd, sec);
	mutex_exit(&cmd->nc_mutex);
}

static void
nvme_async_event(nvme_t *nvme)
{
	nvme_cmd_t *cmd;

	cmd = nvme_alloc_cmd(nvme, KM_SLEEP);
	cmd->nc_sqid = 0;
	cmd->nc_sqe.sqe_opc = NVME_OPC_ASYNC_EVENT;
	cmd->nc_callback = nvme_async_event_task;
	cmd->nc_dontpanic = B_TRUE;

	nvme_submit_admin_cmd(nvme->n_adminq, cmd);
}

static int
nvme_format_nvm(nvme_t *nvme, boolean_t user, uint32_t nsid, uint8_t lbaf,
    boolean_t ms, uint8_t pi, boolean_t pil, uint8_t ses)
{
	nvme_cmd_t *cmd = nvme_alloc_cmd(nvme, KM_SLEEP);
	nvme_format_nvm_t format_nvm = { 0 };
	int ret;

	format_nvm.b.fm_lbaf = lbaf & 0xf;
	format_nvm.b.fm_ms = ms ? 1 : 0;
	format_nvm.b.fm_pi = pi & 0x7;
	format_nvm.b.fm_pil = pil ? 1 : 0;
	format_nvm.b.fm_ses = ses & 0x7;

	cmd->nc_sqid = 0;
	cmd->nc_callback = nvme_wakeup_cmd;
	cmd->nc_sqe.sqe_nsid = nsid;
	cmd->nc_sqe.sqe_opc = NVME_OPC_NVM_FORMAT;
	cmd->nc_sqe.sqe_cdw10 = format_nvm.r;

	/*
	 * Some devices like Samsung SM951 don't allow formatting of all
	 * namespaces in one command. Handle that gracefully.
	 */
	if (nsid == (uint32_t)-1)
		cmd->nc_dontpanic = B_TRUE;
	/*
	 * If this format request was initiated by the user, then don't allow a
	 * programmer error to panic the system.
	 */
	if (user)
		cmd->nc_dontpanic = B_TRUE;

	nvme_admin_cmd(cmd, nvme_format_cmd_timeout);

	if ((ret = nvme_check_cmd_status(cmd)) != 0) {
		dev_err(nvme->n_dip, CE_WARN,
		    "!FORMAT failed with sct = %x, sc = %x",
		    cmd->nc_cqe.cqe_sf.sf_sct, cmd->nc_cqe.cqe_sf.sf_sc);
	}

	nvme_free_cmd(cmd);
	return (ret);
}

/*
 * The `bufsize` parameter is usually an output parameter, set by this routine
 * when filling in the supported types of logpages from the device. However, for
 * vendor-specific pages, it is an input parameter, and must be set
 * appropriately by callers.
 */
static int
nvme_get_logpage(nvme_t *nvme, boolean_t user, void **buf, size_t *bufsize,
    uint8_t logpage, ...)
{
	nvme_cmd_t *cmd = nvme_alloc_cmd(nvme, KM_SLEEP);
	nvme_getlogpage_t getlogpage = { 0 };
	va_list ap;
	int ret;

	va_start(ap, logpage);

	cmd->nc_sqid = 0;
	cmd->nc_callback = nvme_wakeup_cmd;
	cmd->nc_sqe.sqe_opc = NVME_OPC_GET_LOG_PAGE;

	if (user)
		cmd->nc_dontpanic = B_TRUE;

	getlogpage.b.lp_lid = logpage;

	switch (logpage) {
	case NVME_LOGPAGE_ERROR:
		cmd->nc_sqe.sqe_nsid = (uint32_t)-1;
		*bufsize = MIN(NVME_VENDOR_SPECIFIC_LOGPAGE_MAX_SIZE,
		    nvme->n_error_log_len * sizeof (nvme_error_log_entry_t));
		break;

	case NVME_LOGPAGE_HEALTH:
		cmd->nc_sqe.sqe_nsid = va_arg(ap, uint32_t);
		*bufsize = sizeof (nvme_health_log_t);
		break;

	case NVME_LOGPAGE_FWSLOT:
		cmd->nc_sqe.sqe_nsid = (uint32_t)-1;
		*bufsize = sizeof (nvme_fwslot_log_t);
		break;

	case NVME_LOGPAGE_NSCHANGE:
		cmd->nc_sqe.sqe_nsid = (uint32_t)-1;
		*bufsize = sizeof (nvme_nschange_list_t);
		break;

	default:
		/*
		 * This intentionally only checks against the minimum valid
		 * log page ID. `logpage` is a uint8_t, and `0xFF` is a valid
		 * page ID, so this one-sided check avoids a compiler error
		 * about a check that's always true.
		 */
		if (logpage < NVME_VENDOR_SPECIFIC_LOGPAGE_MIN) {
			dev_err(nvme->n_dip, CE_WARN,
			    "!unknown log page requested: %d", logpage);
			atomic_inc_32(&nvme->n_unknown_logpage);
			ret = EINVAL;
			goto fail;
		}
		cmd->nc_sqe.sqe_nsid = va_arg(ap, uint32_t);
	}

	va_end(ap);

	getlogpage.b.lp_numd = *bufsize / sizeof (uint32_t) - 1;

	cmd->nc_sqe.sqe_cdw10 = getlogpage.r;

	if (nvme_zalloc_dma(nvme, *bufsize,
	    DDI_DMA_READ, &nvme->n_prp_dma_attr, &cmd->nc_dma) != DDI_SUCCESS) {
		dev_err(nvme->n_dip, CE_WARN,
		    "!nvme_zalloc_dma failed for GET LOG PAGE");
		ret = ENOMEM;
		goto fail;
	}

	if ((ret = nvme_fill_prp(cmd, cmd->nc_dma->nd_dmah)) != 0)
		goto fail;
	nvme_admin_cmd(cmd, nvme_admin_cmd_timeout);

	if ((ret = nvme_check_cmd_status(cmd)) != 0) {
		dev_err(nvme->n_dip, CE_WARN,
		    "!GET LOG PAGE failed with sct = %x, sc = %x",
		    cmd->nc_cqe.cqe_sf.sf_sct, cmd->nc_cqe.cqe_sf.sf_sc);
		goto fail;
	}

	*buf = kmem_alloc(*bufsize, KM_SLEEP);
	bcopy(cmd->nc_dma->nd_memp, *buf, *bufsize);

fail:
	nvme_free_cmd(cmd);

	return (ret);
}

static int
nvme_identify(nvme_t *nvme, boolean_t user, uint32_t nsid, uint8_t cns,
    void **buf)
{
	nvme_cmd_t *cmd = nvme_alloc_cmd(nvme, KM_SLEEP);
	int ret;

	if (buf == NULL)
		return (EINVAL);

	cmd->nc_sqid = 0;
	cmd->nc_callback = nvme_wakeup_cmd;
	cmd->nc_sqe.sqe_opc = NVME_OPC_IDENTIFY;
	cmd->nc_sqe.sqe_nsid = nsid;
	cmd->nc_sqe.sqe_cdw10 = cns;

	if (nvme_zalloc_dma(nvme, NVME_IDENTIFY_BUFSIZE, DDI_DMA_READ,
	    &nvme->n_prp_dma_attr, &cmd->nc_dma) != DDI_SUCCESS) {
		dev_err(nvme->n_dip, CE_WARN,
		    "!nvme_zalloc_dma failed for IDENTIFY");
		ret = ENOMEM;
		goto fail;
	}

	if (cmd->nc_dma->nd_ncookie > 2) {
		dev_err(nvme->n_dip, CE_WARN,
		    "!too many DMA cookies for IDENTIFY");
		atomic_inc_32(&nvme->n_too_many_cookies);
		ret = ENOMEM;
		goto fail;
	}

	cmd->nc_sqe.sqe_dptr.d_prp[0] = cmd->nc_dma->nd_cookie.dmac_laddress;
	if (cmd->nc_dma->nd_ncookie > 1) {
		ddi_dma_nextcookie(cmd->nc_dma->nd_dmah,
		    &cmd->nc_dma->nd_cookie);
		cmd->nc_sqe.sqe_dptr.d_prp[1] =
		    cmd->nc_dma->nd_cookie.dmac_laddress;
	}

	if (user)
		cmd->nc_dontpanic = B_TRUE;

	nvme_admin_cmd(cmd, nvme_admin_cmd_timeout);

	if ((ret = nvme_check_cmd_status(cmd)) != 0) {
		dev_err(nvme->n_dip, CE_WARN,
		    "!IDENTIFY failed with sct = %x, sc = %x",
		    cmd->nc_cqe.cqe_sf.sf_sct, cmd->nc_cqe.cqe_sf.sf_sc);
		goto fail;
	}

	*buf = kmem_alloc(NVME_IDENTIFY_BUFSIZE, KM_SLEEP);
	bcopy(cmd->nc_dma->nd_memp, *buf, NVME_IDENTIFY_BUFSIZE);

fail:
	nvme_free_cmd(cmd);

	return (ret);
}

static int
nvme_set_features(nvme_t *nvme, boolean_t user, uint32_t nsid, uint8_t feature,
    uint32_t val, uint32_t *res)
{
	_NOTE(ARGUNUSED(nsid));
	nvme_cmd_t *cmd = nvme_alloc_cmd(nvme, KM_SLEEP);
	int ret = EINVAL;

	ASSERT(res != NULL);

	cmd->nc_sqid = 0;
	cmd->nc_callback = nvme_wakeup_cmd;
	cmd->nc_sqe.sqe_opc = NVME_OPC_SET_FEATURES;
	cmd->nc_sqe.sqe_cdw10 = feature;
	cmd->nc_sqe.sqe_cdw11 = val;

	if (user)
		cmd->nc_dontpanic = B_TRUE;

	switch (feature) {
	case NVME_FEAT_WRITE_CACHE:
		if (!nvme->n_write_cache_present)
			goto fail;
		break;

	case NVME_FEAT_NQUEUES:
		break;

	default:
		goto fail;
	}

	nvme_admin_cmd(cmd, nvme_admin_cmd_timeout);

	if ((ret = nvme_check_cmd_status(cmd)) != 0) {
		dev_err(nvme->n_dip, CE_WARN,
		    "!SET FEATURES %d failed with sct = %x, sc = %x",
		    feature, cmd->nc_cqe.cqe_sf.sf_sct,
		    cmd->nc_cqe.cqe_sf.sf_sc);
		goto fail;
	}

	*res = cmd->nc_cqe.cqe_dw0;

fail:
	nvme_free_cmd(cmd);
	return (ret);
}

static int
nvme_get_features(nvme_t *nvme, boolean_t user, uint32_t nsid, uint8_t feature,
    uint32_t *res, void **buf, size_t *bufsize)
{
	nvme_cmd_t *cmd = nvme_alloc_cmd(nvme, KM_SLEEP);
	int ret = EINVAL;

	ASSERT(res != NULL);

	if (bufsize != NULL)
		*bufsize = 0;

	cmd->nc_sqid = 0;
	cmd->nc_callback = nvme_wakeup_cmd;
	cmd->nc_sqe.sqe_opc = NVME_OPC_GET_FEATURES;
	cmd->nc_sqe.sqe_cdw10 = feature;
	cmd->nc_sqe.sqe_cdw11 = *res;

	/*
	 * For some of the optional features there doesn't seem to be a method
	 * of detecting whether it is supported other than using it.  This will
	 * cause "Invalid Field in Command" error, which is normally considered
	 * a programming error.  Set the nc_dontpanic flag to override the panic
	 * in nvme_check_generic_cmd_status().
	 */
	switch (feature) {
	case NVME_FEAT_ARBITRATION:
	case NVME_FEAT_POWER_MGMT:
	case NVME_FEAT_TEMPERATURE:
	case NVME_FEAT_ERROR:
	case NVME_FEAT_NQUEUES:
	case NVME_FEAT_INTR_COAL:
	case NVME_FEAT_INTR_VECT:
	case NVME_FEAT_WRITE_ATOM:
	case NVME_FEAT_ASYNC_EVENT:
		break;

	case NVME_FEAT_WRITE_CACHE:
		if (!nvme->n_write_cache_present)
			goto fail;
		break;

	case NVME_FEAT_LBA_RANGE:
		if (!nvme->n_lba_range_supported)
			goto fail;

		cmd->nc_dontpanic = B_TRUE;
		cmd->nc_sqe.sqe_nsid = nsid;
		ASSERT(bufsize != NULL);
		*bufsize = NVME_LBA_RANGE_BUFSIZE;
		break;

	case NVME_FEAT_AUTO_PST:
		if (!nvme->n_auto_pst_supported)
			goto fail;

		ASSERT(bufsize != NULL);
		*bufsize = NVME_AUTO_PST_BUFSIZE;
		break;

	case NVME_FEAT_PROGRESS:
		if (!nvme->n_progress_supported)
			goto fail;

		cmd->nc_dontpanic = B_TRUE;
		break;

	default:
		goto fail;
	}

	if (user)
		cmd->nc_dontpanic = B_TRUE;

	if (bufsize != NULL && *bufsize != 0) {
		if (nvme_zalloc_dma(nvme, *bufsize, DDI_DMA_READ,
		    &nvme->n_prp_dma_attr, &cmd->nc_dma) != DDI_SUCCESS) {
			dev_err(nvme->n_dip, CE_WARN,
			    "!nvme_zalloc_dma failed for GET FEATURES");
			ret = ENOMEM;
			goto fail;
		}

		if (cmd->nc_dma->nd_ncookie > 2) {
			dev_err(nvme->n_dip, CE_WARN,
			    "!too many DMA cookies for GET FEATURES");
			atomic_inc_32(&nvme->n_too_many_cookies);
			ret = ENOMEM;
			goto fail;
		}

		cmd->nc_sqe.sqe_dptr.d_prp[0] =
		    cmd->nc_dma->nd_cookie.dmac_laddress;
		if (cmd->nc_dma->nd_ncookie > 1) {
			ddi_dma_nextcookie(cmd->nc_dma->nd_dmah,
			    &cmd->nc_dma->nd_cookie);
			cmd->nc_sqe.sqe_dptr.d_prp[1] =
			    cmd->nc_dma->nd_cookie.dmac_laddress;
		}
	}

	nvme_admin_cmd(cmd, nvme_admin_cmd_timeout);

	if ((ret = nvme_check_cmd_status(cmd)) != 0) {
		boolean_t known = B_TRUE;

		/* Check if this is unsupported optional feature */
		if (cmd->nc_cqe.cqe_sf.sf_sct == NVME_CQE_SCT_GENERIC &&
		    cmd->nc_cqe.cqe_sf.sf_sc == NVME_CQE_SC_GEN_INV_FLD) {
			switch (feature) {
			case NVME_FEAT_LBA_RANGE:
				nvme->n_lba_range_supported = B_FALSE;
				break;
			case NVME_FEAT_PROGRESS:
				nvme->n_progress_supported = B_FALSE;
				break;
			default:
				known = B_FALSE;
				break;
			}
		} else {
			known = B_FALSE;
		}

		/* Report the error otherwise */
		if (!known) {
			dev_err(nvme->n_dip, CE_WARN,
			    "!GET FEATURES %d failed with sct = %x, sc = %x",
			    feature, cmd->nc_cqe.cqe_sf.sf_sct,
			    cmd->nc_cqe.cqe_sf.sf_sc);
		}

		goto fail;
	}

	if (bufsize != NULL && *bufsize != 0) {
		ASSERT(buf != NULL);
		*buf = kmem_alloc(*bufsize, KM_SLEEP);
		bcopy(cmd->nc_dma->nd_memp, *buf, *bufsize);
	}

	*res = cmd->nc_cqe.cqe_dw0;

fail:
	nvme_free_cmd(cmd);
	return (ret);
}

static int
nvme_write_cache_set(nvme_t *nvme, boolean_t enable)
{
	nvme_write_cache_t nwc = { 0 };

	if (enable)
		nwc.b.wc_wce = 1;

	return (nvme_set_features(nvme, B_FALSE, 0, NVME_FEAT_WRITE_CACHE,
	    nwc.r, &nwc.r));
}

static int
nvme_set_nqueues(nvme_t *nvme)
{
	nvme_nqueues_t nq = { 0 };
	int ret;

	/*
	 * The default is to allocate one completion queue per vector.
	 */
	if (nvme->n_completion_queues == -1)
		nvme->n_completion_queues = nvme->n_intr_cnt;

	/*
	 * There is no point in having more completion queues than
	 * interrupt vectors.
	 */
	nvme->n_completion_queues = MIN(nvme->n_completion_queues,
	    nvme->n_intr_cnt);

	/*
	 * The default is to use one submission queue per completion queue.
	 */
	if (nvme->n_submission_queues == -1)
		nvme->n_submission_queues = nvme->n_completion_queues;

	/*
	 * There is no point in having more compeletion queues than
	 * submission queues.
	 */
	nvme->n_completion_queues = MIN(nvme->n_completion_queues,
	    nvme->n_submission_queues);

	ASSERT(nvme->n_submission_queues > 0);
	ASSERT(nvme->n_completion_queues > 0);

	nq.b.nq_nsq = nvme->n_submission_queues - 1;
	nq.b.nq_ncq = nvme->n_completion_queues - 1;

	ret = nvme_set_features(nvme, B_FALSE, 0, NVME_FEAT_NQUEUES, nq.r,
	    &nq.r);

	if (ret == 0) {
		/*
		 * Never use more than the requested number of queues.
		 */
		nvme->n_submission_queues = MIN(nvme->n_submission_queues,
		    nq.b.nq_nsq + 1);
		nvme->n_completion_queues = MIN(nvme->n_completion_queues,
		    nq.b.nq_ncq + 1);
	}

	return (ret);
}

static int
nvme_create_completion_queue(nvme_t *nvme, nvme_cq_t *cq)
{
	nvme_cmd_t *cmd = nvme_alloc_cmd(nvme, KM_SLEEP);
	nvme_create_queue_dw10_t dw10 = { 0 };
	nvme_create_cq_dw11_t c_dw11 = { 0 };
	int ret;

	dw10.b.q_qid = cq->ncq_id;
	dw10.b.q_qsize = cq->ncq_nentry - 1;

	c_dw11.b.cq_pc = 1;
	c_dw11.b.cq_ien = 1;
	c_dw11.b.cq_iv = cq->ncq_id % nvme->n_intr_cnt;

	cmd->nc_sqid = 0;
	cmd->nc_callback = nvme_wakeup_cmd;
	cmd->nc_sqe.sqe_opc = NVME_OPC_CREATE_CQUEUE;
	cmd->nc_sqe.sqe_cdw10 = dw10.r;
	cmd->nc_sqe.sqe_cdw11 = c_dw11.r;
	cmd->nc_sqe.sqe_dptr.d_prp[0] = cq->ncq_dma->nd_cookie.dmac_laddress;

	nvme_admin_cmd(cmd, nvme_admin_cmd_timeout);

	if ((ret = nvme_check_cmd_status(cmd)) != 0) {
		dev_err(nvme->n_dip, CE_WARN,
		    "!CREATE CQUEUE failed with sct = %x, sc = %x",
		    cmd->nc_cqe.cqe_sf.sf_sct, cmd->nc_cqe.cqe_sf.sf_sc);
	}

	nvme_free_cmd(cmd);

	return (ret);
}

static int
nvme_create_io_qpair(nvme_t *nvme, nvme_qpair_t *qp, uint16_t idx)
{
	nvme_cq_t *cq = qp->nq_cq;
	nvme_cmd_t *cmd;
	nvme_create_queue_dw10_t dw10 = { 0 };
	nvme_create_sq_dw11_t s_dw11 = { 0 };
	int ret;

	/*
	 * It is possible to have more qpairs than completion queues,
	 * and when the idx > ncq_id, that completion queue is shared
	 * and has already been created.
	 */
	if (idx <= cq->ncq_id &&
	    nvme_create_completion_queue(nvme, cq) != DDI_SUCCESS)
		return (DDI_FAILURE);

	dw10.b.q_qid = idx;
	dw10.b.q_qsize = qp->nq_nentry - 1;

	s_dw11.b.sq_pc = 1;
	s_dw11.b.sq_cqid = cq->ncq_id;

	cmd = nvme_alloc_cmd(nvme, KM_SLEEP);
	cmd->nc_sqid = 0;
	cmd->nc_callback = nvme_wakeup_cmd;
	cmd->nc_sqe.sqe_opc = NVME_OPC_CREATE_SQUEUE;
	cmd->nc_sqe.sqe_cdw10 = dw10.r;
	cmd->nc_sqe.sqe_cdw11 = s_dw11.r;
	cmd->nc_sqe.sqe_dptr.d_prp[0] = qp->nq_sqdma->nd_cookie.dmac_laddress;

	nvme_admin_cmd(cmd, nvme_admin_cmd_timeout);

	if ((ret = nvme_check_cmd_status(cmd)) != 0) {
		dev_err(nvme->n_dip, CE_WARN,
		    "!CREATE SQUEUE failed with sct = %x, sc = %x",
		    cmd->nc_cqe.cqe_sf.sf_sct, cmd->nc_cqe.cqe_sf.sf_sc);
	}

	nvme_free_cmd(cmd);

	return (ret);
}

static boolean_t
nvme_reset(nvme_t *nvme, boolean_t quiesce)
{
	nvme_reg_csts_t csts;
	int i;

	nvme_put32(nvme, NVME_REG_CC, 0);

	csts.r = nvme_get32(nvme, NVME_REG_CSTS);
	if (csts.b.csts_rdy == 1) {
		nvme_put32(nvme, NVME_REG_CC, 0);

		/*
		 * The timeout value is from the Controller Capabilities
		 * register (CAP.TO, section 3.1.1). This is the worst case
		 * time to wait for CSTS.RDY to transition from 1 to 0 after
		 * CC.EN transitions from 1 to 0.
		 *
		 * The timeout units are in 500 ms units, and we are delaying
		 * in 50ms chunks, hence counting to n_timeout * 10.
		 */
		for (i = 0; i < nvme->n_timeout * 10; i++) {
			csts.r = nvme_get32(nvme, NVME_REG_CSTS);
			if (csts.b.csts_rdy == 0)
				break;

			/*
			 * Quiescing drivers should not use locks or timeouts,
			 * so if this is the quiesce path, use a quiesce-safe
			 * delay.
			 */
			if (quiesce) {
				drv_usecwait(50000);
			} else {
				delay(drv_usectohz(50000));
			}
		}
	}

	nvme_put32(nvme, NVME_REG_AQA, 0);
	nvme_put32(nvme, NVME_REG_ASQ, 0);
	nvme_put32(nvme, NVME_REG_ACQ, 0);

	csts.r = nvme_get32(nvme, NVME_REG_CSTS);
	return (csts.b.csts_rdy == 0 ? B_TRUE : B_FALSE);
}

static void
nvme_shutdown(nvme_t *nvme, boolean_t quiesce)
{
	nvme_reg_cc_t cc;
	nvme_reg_csts_t csts;
	int i;

	cc.r = nvme_get32(nvme, NVME_REG_CC);
	cc.b.cc_shn = NVME_CC_SHN_NORMAL;
	nvme_put32(nvme, NVME_REG_CC, cc.r);

	for (i = 0; i < 10; i++) {
		csts.r = nvme_get32(nvme, NVME_REG_CSTS);
		if (csts.b.csts_shst == NVME_CSTS_SHN_COMPLETE)
			break;

		if (quiesce) {
			drv_usecwait(100000);
		} else {
			delay(drv_usectohz(100000));
		}
	}
}

/*
 * Return length of string without trailing spaces.
 */
static int
nvme_strlen(const char *str, int len)
{
	if (len <= 0)
		return (0);

	while (str[--len] == ' ')
		;

	return (++len);
}

static void
nvme_config_min_block_size(nvme_t *nvme, char *model, char *val)
{
	ulong_t bsize = 0;
	char *msg = "";

	if (ddi_strtoul(val, NULL, 0, &bsize) != 0)
		goto err;

	if (!ISP2(bsize)) {
		msg = ": not a power of 2";
		goto err;
	}

	if (bsize < NVME_DEFAULT_MIN_BLOCK_SIZE) {
		msg = ": too low";
		goto err;
	}

	nvme->n_min_block_size = bsize;
	return;

err:
	dev_err(nvme->n_dip, CE_WARN,
	    "!nvme-config-list: ignoring invalid min-phys-block-size '%s' "
	    "for model '%s'%s", val, model, msg);

	nvme->n_min_block_size = NVME_DEFAULT_MIN_BLOCK_SIZE;
}

static void
nvme_config_boolean(nvme_t *nvme, char *model, char *name, char *val,
    boolean_t *b)
{
	if (strcmp(val, "on") == 0 ||
	    strcmp(val, "true") == 0)
		*b = B_TRUE;
	else if (strcmp(val, "off") == 0 ||
	    strcmp(val, "false") == 0)
		*b = B_FALSE;
	else
		dev_err(nvme->n_dip, CE_WARN,
		    "!nvme-config-list: invalid value for %s '%s'"
		    " for model '%s', ignoring", name, val, model);
}

static void
nvme_config_list(nvme_t *nvme)
{
	char	**config_list;
	uint_t	nelem;
	int	rv, i;

	/*
	 * We're following the pattern of 'sd-config-list' here, but extend it.
	 * Instead of two we have three separate strings for "model", "fwrev",
	 * and "name-value-list".
	 */
	rv = ddi_prop_lookup_string_array(DDI_DEV_T_ANY, nvme->n_dip,
	    DDI_PROP_DONTPASS, "nvme-config-list", &config_list, &nelem);

	if (rv != DDI_PROP_SUCCESS) {
		if (rv == DDI_PROP_CANNOT_DECODE) {
			dev_err(nvme->n_dip, CE_WARN,
			    "!nvme-config-list: cannot be decoded");
		}

		return;
	}

	if ((nelem % 3) != 0) {
		dev_err(nvme->n_dip, CE_WARN, "!nvme-config-list: must be "
		    "triplets of <model>/<fwrev>/<name-value-list> strings ");
		goto out;
	}

	for (i = 0; i < nelem; i += 3) {
		char	*model = config_list[i];
		char	*fwrev = config_list[i + 1];
		char	*nvp, *save_nv;
		int	id_model_len, id_fwrev_len;

		id_model_len = nvme_strlen(nvme->n_idctl->id_model,
		    sizeof (nvme->n_idctl->id_model));

		if (strlen(model) != id_model_len)
			continue;

		if (strncmp(model, nvme->n_idctl->id_model, id_model_len) != 0)
			continue;

		id_fwrev_len = nvme_strlen(nvme->n_idctl->id_fwrev,
		    sizeof (nvme->n_idctl->id_fwrev));

		if (strlen(fwrev) != 0) {
			boolean_t match = B_FALSE;
			char *fwr, *last_fw;

			for (fwr = strtok_r(fwrev, ",", &last_fw);
			    fwr != NULL;
			    fwr = strtok_r(NULL, ",", &last_fw)) {
				if (strlen(fwr) != id_fwrev_len)
					continue;

				if (strncmp(fwr, nvme->n_idctl->id_fwrev,
				    id_fwrev_len) == 0)
					match = B_TRUE;
			}

			if (!match)
				continue;
		}

		/*
		 * We should now have a comma-separated list of name:value
		 * pairs.
		 */
		for (nvp = strtok_r(config_list[i + 2], ",", &save_nv);
		    nvp != NULL; nvp = strtok_r(NULL, ",", &save_nv)) {
			char	*name = nvp;
			char	*val = strchr(nvp, ':');

			if (val == NULL || name == val) {
				dev_err(nvme->n_dip, CE_WARN,
				    "!nvme-config-list: <name-value-list> "
				    "for model '%s' is malformed", model);
				goto out;
			}

			/*
			 * Null-terminate 'name', move 'val' past ':' sep.
			 */
			*val++ = '\0';

			/*
			 * Process the name:val pairs that we know about.
			 */
			if (strcmp(name, "ignore-unknown-vendor-status") == 0) {
				nvme_config_boolean(nvme, model, name, val,
				    &nvme->n_ignore_unknown_vendor_status);
			} else if (strcmp(name, "min-phys-block-size") == 0) {
				nvme_config_min_block_size(nvme, model, val);
			} else if (strcmp(name, "volatile-write-cache") == 0) {
				nvme_config_boolean(nvme, model, name, val,
				    &nvme->n_write_cache_enabled);
			} else {
				/*
				 * Unknown 'name'.
				 */
				dev_err(nvme->n_dip, CE_WARN,
				    "!nvme-config-list: unknown config '%s' "
				    "for model '%s', ignoring", name, model);
			}
		}
	}

out:
	ddi_prop_free(config_list);
}

static void
nvme_prepare_devid(nvme_t *nvme, uint32_t nsid)
{
	/*
	 * Section 7.7 of the spec describes how to get a unique ID for
	 * the controller: the vendor ID, the model name and the serial
	 * number shall be unique when combined.
	 *
	 * If a namespace has no EUI64 we use the above and add the hex
	 * namespace ID to get a unique ID for the namespace.
	 */
	char model[sizeof (nvme->n_idctl->id_model) + 1];
	char serial[sizeof (nvme->n_idctl->id_serial) + 1];

	bcopy(nvme->n_idctl->id_model, model, sizeof (nvme->n_idctl->id_model));
	bcopy(nvme->n_idctl->id_serial, serial,
	    sizeof (nvme->n_idctl->id_serial));

	model[sizeof (nvme->n_idctl->id_model)] = '\0';
	serial[sizeof (nvme->n_idctl->id_serial)] = '\0';

	NVME_NSID2NS(nvme, nsid)->ns_devid = kmem_asprintf("%4X-%s-%s-%X",
	    nvme->n_idctl->id_vid, model, serial, nsid);
}

static nvme_identify_nsid_list_t *
nvme_update_nsid_list(nvme_t *nvme, int cns)
{
	nvme_identify_nsid_list_t *nslist;

	/*
	 * We currently don't handle cases where there are more than
	 * 1024 active namespaces, requiring several IDENTIFY commands.
	 */
	if (nvme_identify(nvme, B_FALSE, 0, cns, (void **)&nslist) == 0)
		return (nslist);

	return (NULL);
}

static boolean_t
nvme_allocated_ns(nvme_namespace_t *ns)
{
	nvme_t *nvme = ns->ns_nvme;
	uint32_t i;

	ASSERT(MUTEX_HELD(&nvme->n_mgmt_mutex));

	/*
	 * If supported, update the list of allocated namespace IDs.
	 */
	if (NVME_VERSION_ATLEAST(&nvme->n_version, 1, 2) &&
	    nvme->n_idctl->id_oacs.oa_nsmgmt != 0) {
		nvme_identify_nsid_list_t *nslist = nvme_update_nsid_list(nvme,
		    NVME_IDENTIFY_NSID_ALLOC_LIST);
		boolean_t found = B_FALSE;

		/*
		 * When namespace management is supported, this really shouldn't
		 * be NULL. Treat all namespaces as allocated if it is.
		 */
		if (nslist == NULL)
			return (B_TRUE);

		for (i = 0; i < ARRAY_SIZE(nslist->nl_nsid); i++) {
			if (ns->ns_id == 0)
				break;

			if (ns->ns_id == nslist->nl_nsid[i])
				found = B_TRUE;
		}

		kmem_free(nslist, NVME_IDENTIFY_BUFSIZE);
		return (found);
	} else {
		/*
		 * If namespace management isn't supported, report all
		 * namespaces as allocated.
		 */
		return (B_TRUE);
	}
}

static boolean_t
nvme_active_ns(nvme_namespace_t *ns)
{
	nvme_t *nvme = ns->ns_nvme;
	uint64_t *ptr;
	uint32_t i;

	ASSERT(MUTEX_HELD(&nvme->n_mgmt_mutex));

	/*
	 * If supported, update the list of active namespace IDs.
	 */
	if (NVME_VERSION_ATLEAST(&nvme->n_version, 1, 1)) {
		nvme_identify_nsid_list_t *nslist = nvme_update_nsid_list(nvme,
		    NVME_IDENTIFY_NSID_LIST);
		boolean_t found = B_FALSE;

		/*
		 * When namespace management is supported, this really shouldn't
		 * be NULL. Treat all namespaces as allocated if it is.
		 */
		if (nslist == NULL)
			return (B_TRUE);

		for (i = 0; i < ARRAY_SIZE(nslist->nl_nsid); i++) {
			if (ns->ns_id == 0)
				break;

			if (ns->ns_id == nslist->nl_nsid[i])
				found = B_TRUE;
		}

		kmem_free(nslist, NVME_IDENTIFY_BUFSIZE);
		return (found);
	}

	/*
	 * Workaround for revision 1.0:
	 * Check whether the IDENTIFY NAMESPACE data is zero-filled.
	 */
	for (ptr = (uint64_t *)ns->ns_idns;
	    ptr != (uint64_t *)(ns->ns_idns + 1);
	    ptr++) {
		if (*ptr != 0) {
			return (B_TRUE);
		}
	}

	return (B_FALSE);
}

static int
nvme_init_ns(nvme_t *nvme, int nsid)
{
	nvme_namespace_t *ns = NVME_NSID2NS(nvme, nsid);
	nvme_identify_nsid_t *idns;
	boolean_t was_ignored;
	int last_rp;

	ns->ns_nvme = nvme;

	ASSERT(MUTEX_HELD(&nvme->n_mgmt_mutex));

	if (nvme_identify(nvme, B_FALSE, nsid, NVME_IDENTIFY_NSID,
	    (void **)&idns) != 0) {
		dev_err(nvme->n_dip, CE_WARN,
		    "!failed to identify namespace %d", nsid);
		return (DDI_FAILURE);
	}

	if (ns->ns_idns != NULL)
		kmem_free(ns->ns_idns, sizeof (nvme_identify_nsid_t));

	ns->ns_idns = idns;
	ns->ns_id = nsid;

	was_ignored = ns->ns_ignore;

	ns->ns_allocated = nvme_allocated_ns(ns);
	ns->ns_active = nvme_active_ns(ns);

	ns->ns_block_count = idns->id_nsize;
	ns->ns_block_size =
	    1 << idns->id_lbaf[idns->id_flbas.lba_format].lbaf_lbads;
	ns->ns_best_block_size = ns->ns_block_size;

	/*
	 * Get the EUI64 if present.
	 */
	if (NVME_VERSION_ATLEAST(&nvme->n_version, 1, 1))
		bcopy(idns->id_eui64, ns->ns_eui64, sizeof (ns->ns_eui64));

	/*
	 * Get the NGUID if present.
	 */
	if (NVME_VERSION_ATLEAST(&nvme->n_version, 1, 2))
		bcopy(idns->id_nguid, ns->ns_nguid, sizeof (ns->ns_nguid));

	/*LINTED: E_BAD_PTR_CAST_ALIGN*/
	if (*(uint64_t *)ns->ns_eui64 == 0)
		nvme_prepare_devid(nvme, ns->ns_id);

	(void) snprintf(ns->ns_name, sizeof (ns->ns_name), "%u", ns->ns_id);

	/*
	 * Find the LBA format with no metadata and the best relative
	 * performance. A value of 3 means "degraded", 0 is best.
	 */
	last_rp = 3;
	for (int j = 0; j <= idns->id_nlbaf; j++) {
		if (idns->id_lbaf[j].lbaf_lbads == 0)
			break;
		if (idns->id_lbaf[j].lbaf_ms != 0)
			continue;
		if (idns->id_lbaf[j].lbaf_rp >= last_rp)
			continue;
		last_rp = idns->id_lbaf[j].lbaf_rp;
		ns->ns_best_block_size =
		    1 << idns->id_lbaf[j].lbaf_lbads;
	}

	if (ns->ns_best_block_size < nvme->n_min_block_size)
		ns->ns_best_block_size = nvme->n_min_block_size;

	was_ignored = ns->ns_ignore;

	/*
	 * We currently don't support namespaces that are inactive, or use
	 * either:
	 * - protection information
	 * - illegal block size (< 512)
	 */
	if (!ns->ns_active) {
		ns->ns_ignore = B_TRUE;
	} else if (idns->id_dps.dp_pinfo) {
		dev_err(nvme->n_dip, CE_WARN,
		    "!ignoring namespace %d, unsupported feature: "
		    "pinfo = %d", nsid, idns->id_dps.dp_pinfo);
		ns->ns_ignore = B_TRUE;
	} else if (ns->ns_block_size < 512) {
		dev_err(nvme->n_dip, CE_WARN,
		    "!ignoring namespace %d, unsupported block size %"PRIu64,
		    nsid, (uint64_t)ns->ns_block_size);
		ns->ns_ignore = B_TRUE;
	} else {
		ns->ns_ignore = B_FALSE;
	}

	/*
	 * Keep a count of namespaces which are attachable.
	 * See comments in nvme_bd_driveinfo() to understand its effect.
	 */
	if (was_ignored) {
		/*
		 * Previously ignored, but now not. Count it.
		 */
		if (!ns->ns_ignore)
			nvme->n_namespaces_attachable++;
	} else {
		/*
		 * Wasn't ignored previously, but now needs to be.
		 * Discount it.
		 */
		if (ns->ns_ignore)
			nvme->n_namespaces_attachable--;
	}

	return (DDI_SUCCESS);
}

static int
nvme_attach_ns(nvme_t *nvme, int nsid)
{
	nvme_namespace_t *ns = NVME_NSID2NS(nvme, nsid);

	ASSERT(MUTEX_HELD(&nvme->n_mgmt_mutex));

	if (ns->ns_ignore)
		return (ENOTSUP);

	if (ns->ns_bd_hdl == NULL) {
		bd_ops_t ops = nvme_bd_ops;

		if (!nvme->n_idctl->id_oncs.on_dset_mgmt)
			ops.o_free_space = NULL;

		ns->ns_bd_hdl = bd_alloc_handle(ns, &ops, &nvme->n_prp_dma_attr,
		    KM_SLEEP);

		if (ns->ns_bd_hdl == NULL) {
			dev_err(nvme->n_dip, CE_WARN, "!Failed to get blkdev "
			    "handle for namespace id %d", nsid);
			return (EINVAL);
		}
	}

	if (bd_attach_handle(nvme->n_dip, ns->ns_bd_hdl) != DDI_SUCCESS)
		return (EBUSY);

	ns->ns_attached = B_TRUE;

	return (0);
}

static int
nvme_detach_ns(nvme_t *nvme, int nsid)
{
	nvme_namespace_t *ns = NVME_NSID2NS(nvme, nsid);
	int rv;

	ASSERT(MUTEX_HELD(&nvme->n_mgmt_mutex));

	if (ns->ns_ignore || !ns->ns_attached)
		return (0);

	ASSERT(ns->ns_bd_hdl != NULL);
	rv = bd_detach_handle(ns->ns_bd_hdl);
	if (rv != DDI_SUCCESS)
		return (EBUSY);
	else
		ns->ns_attached = B_FALSE;

	return (0);
}

typedef struct nvme_quirk_table {
	uint16_t nq_vendor_id;
	uint16_t nq_device_id;
	nvme_quirk_t nq_quirks;
} nvme_quirk_table_t;

static const nvme_quirk_table_t nvme_quirks[] = {
	{ 0x1987, 0x5018, NVME_QUIRK_START_CID },	/* Phison E18 */
};

static void
nvme_detect_quirks(nvme_t *nvme)
{
	for (uint_t i = 0; i < ARRAY_SIZE(nvme_quirks); i++) {
		const nvme_quirk_table_t *nqt = &nvme_quirks[i];

		if (nqt->nq_vendor_id == nvme->n_vendor_id &&
		    nqt->nq_device_id == nvme->n_device_id) {
			nvme->n_quirks = nqt->nq_quirks;
			return;
		}
	}
}

static int
nvme_init(nvme_t *nvme)
{
	nvme_reg_cc_t cc = { 0 };
	nvme_reg_aqa_t aqa = { 0 };
	nvme_reg_asq_t asq = { 0 };
	nvme_reg_acq_t acq = { 0 };
	nvme_reg_cap_t cap;
	nvme_reg_vs_t vs;
	nvme_reg_csts_t csts;
	int i = 0;
	uint16_t nqueues;
	uint_t tq_threads;
	char model[sizeof (nvme->n_idctl->id_model) + 1];
	char *vendor, *product;

	/* Check controller version */
	vs.r = nvme_get32(nvme, NVME_REG_VS);
	nvme->n_version.v_major = vs.b.vs_mjr;
	nvme->n_version.v_minor = vs.b.vs_mnr;
	dev_err(nvme->n_dip, CE_CONT, "?NVMe spec version %d.%d",
	    nvme->n_version.v_major, nvme->n_version.v_minor);

	if (nvme->n_version.v_major > nvme_version_major) {
		dev_err(nvme->n_dip, CE_WARN, "!no support for version > %d.x",
		    nvme_version_major);
		if (nvme->n_strict_version)
			goto fail;
	}

	/* retrieve controller configuration */
	cap.r = nvme_get64(nvme, NVME_REG_CAP);

	if ((cap.b.cap_css & NVME_CAP_CSS_NVM) == 0) {
		dev_err(nvme->n_dip, CE_WARN,
		    "!NVM command set not supported by hardware");
		goto fail;
	}

	nvme->n_nssr_supported = cap.b.cap_nssrs;
	nvme->n_doorbell_stride = 4 << cap.b.cap_dstrd;
	nvme->n_timeout = cap.b.cap_to;
	nvme->n_arbitration_mechanisms = cap.b.cap_ams;
	nvme->n_cont_queues_reqd = cap.b.cap_cqr;
	nvme->n_max_queue_entries = cap.b.cap_mqes + 1;

	/*
	 * The MPSMIN and MPSMAX fields in the CAP register use 0 to specify
	 * the base page size of 4k (1<<12), so add 12 here to get the real
	 * page size value.
	 */
	nvme->n_pageshift = MIN(MAX(cap.b.cap_mpsmin + 12, PAGESHIFT),
	    cap.b.cap_mpsmax + 12);
	nvme->n_pagesize = 1UL << (nvme->n_pageshift);

	/*
	 * Set up Queue DMA to transfer at least 1 page-aligned page at a time.
	 */
	nvme->n_queue_dma_attr.dma_attr_align = nvme->n_pagesize;
	nvme->n_queue_dma_attr.dma_attr_minxfer = nvme->n_pagesize;

	/*
	 * Set up PRP DMA to transfer 1 page-aligned page at a time.
	 * Maxxfer may be increased after we identified the controller limits.
	 */
	nvme->n_prp_dma_attr.dma_attr_maxxfer = nvme->n_pagesize;
	nvme->n_prp_dma_attr.dma_attr_minxfer = nvme->n_pagesize;
	nvme->n_prp_dma_attr.dma_attr_align = nvme->n_pagesize;
	nvme->n_prp_dma_attr.dma_attr_seg = nvme->n_pagesize - 1;

	/*
	 * Reset controller if it's still in ready state.
	 */
	if (nvme_reset(nvme, B_FALSE) == B_FALSE) {
		dev_err(nvme->n_dip, CE_WARN, "!unable to reset controller");
		ddi_fm_service_impact(nvme->n_dip, DDI_SERVICE_LOST);
		nvme->n_dead = B_TRUE;
		goto fail;
	}

	/*
	 * Create the cq array with one completion queue to be assigned
	 * to the admin queue pair and a limited number of taskqs (4).
	 */
	if (nvme_create_cq_array(nvme, 1, nvme->n_admin_queue_len, 4) !=
	    DDI_SUCCESS) {
		dev_err(nvme->n_dip, CE_WARN,
		    "!failed to pre-allocate admin completion queue");
		goto fail;
	}
	/*
	 * Create the admin queue pair.
	 */
	if (nvme_alloc_qpair(nvme, nvme->n_admin_queue_len, &nvme->n_adminq, 0)
	    != DDI_SUCCESS) {
		dev_err(nvme->n_dip, CE_WARN,
		    "!unable to allocate admin qpair");
		goto fail;
	}
	nvme->n_ioq = kmem_alloc(sizeof (nvme_qpair_t *), KM_SLEEP);
	nvme->n_ioq[0] = nvme->n_adminq;

	if (nvme->n_quirks & NVME_QUIRK_START_CID)
		nvme->n_adminq->nq_next_cmd++;

	nvme->n_progress |= NVME_ADMIN_QUEUE;

	(void) ddi_prop_update_int(DDI_DEV_T_NONE, nvme->n_dip,
	    "admin-queue-len", nvme->n_admin_queue_len);

	aqa.b.aqa_asqs = aqa.b.aqa_acqs = nvme->n_admin_queue_len - 1;
	asq = nvme->n_adminq->nq_sqdma->nd_cookie.dmac_laddress;
	acq = nvme->n_adminq->nq_cq->ncq_dma->nd_cookie.dmac_laddress;

	ASSERT((asq & (nvme->n_pagesize - 1)) == 0);
	ASSERT((acq & (nvme->n_pagesize - 1)) == 0);

	nvme_put32(nvme, NVME_REG_AQA, aqa.r);
	nvme_put64(nvme, NVME_REG_ASQ, asq);
	nvme_put64(nvme, NVME_REG_ACQ, acq);

	cc.b.cc_ams = 0;	/* use Round-Robin arbitration */
	cc.b.cc_css = 0;	/* use NVM command set */
	cc.b.cc_mps = nvme->n_pageshift - 12;
	cc.b.cc_shn = 0;	/* no shutdown in progress */
	cc.b.cc_en = 1;		/* enable controller */
	cc.b.cc_iosqes = 6;	/* submission queue entry is 2^6 bytes long */
	cc.b.cc_iocqes = 4;	/* completion queue entry is 2^4 bytes long */

	nvme_put32(nvme, NVME_REG_CC, cc.r);

	/*
	 * Wait for the controller to become ready.
	 */
	csts.r = nvme_get32(nvme, NVME_REG_CSTS);
	if (csts.b.csts_rdy == 0) {
		for (i = 0; i != nvme->n_timeout * 10; i++) {
			delay(drv_usectohz(50000));
			csts.r = nvme_get32(nvme, NVME_REG_CSTS);

			if (csts.b.csts_cfs == 1) {
				dev_err(nvme->n_dip, CE_WARN,
				    "!controller fatal status at init");
				ddi_fm_service_impact(nvme->n_dip,
				    DDI_SERVICE_LOST);
				nvme->n_dead = B_TRUE;
				goto fail;
			}

			if (csts.b.csts_rdy == 1)
				break;
		}
	}

	if (csts.b.csts_rdy == 0) {
		dev_err(nvme->n_dip, CE_WARN, "!controller not ready");
		ddi_fm_service_impact(nvme->n_dip, DDI_SERVICE_LOST);
		nvme->n_dead = B_TRUE;
		goto fail;
	}

	/*
	 * Assume an abort command limit of 1. We'll destroy and re-init
	 * that later when we know the true abort command limit.
	 */
	sema_init(&nvme->n_abort_sema, 1, NULL, SEMA_DRIVER, NULL);

	/*
	 * Set up initial interrupt for admin queue.
	 */
	if ((nvme_setup_interrupts(nvme, DDI_INTR_TYPE_MSIX, 1)
	    != DDI_SUCCESS) &&
	    (nvme_setup_interrupts(nvme, DDI_INTR_TYPE_MSI, 1)
	    != DDI_SUCCESS) &&
	    (nvme_setup_interrupts(nvme, DDI_INTR_TYPE_FIXED, 1)
	    != DDI_SUCCESS)) {
		dev_err(nvme->n_dip, CE_WARN,
		    "!failed to set up initial interrupt");
		goto fail;
	}

	/*
	 * Post an asynchronous event command to catch errors.
	 * We assume the asynchronous events are supported as required by
	 * specification (Figure 40 in section 5 of NVMe 1.2).
	 * However, since at least qemu does not follow the specification,
	 * we need a mechanism to protect ourselves.
	 */
	nvme->n_async_event_supported = B_TRUE;
	nvme_async_event(nvme);

	/*
	 * Identify Controller
	 */
	if (nvme_identify(nvme, B_FALSE, 0, NVME_IDENTIFY_CTRL,
	    (void **)&nvme->n_idctl) != 0) {
		dev_err(nvme->n_dip, CE_WARN,
		    "!failed to identify controller");
		goto fail;
	}

	/*
	 * Process nvme-config-list (if present) in nvme.conf.
	 */
	nvme_config_list(nvme);

	/*
	 * Get Vendor & Product ID
	 */
	bcopy(nvme->n_idctl->id_model, model, sizeof (nvme->n_idctl->id_model));
	model[sizeof (nvme->n_idctl->id_model)] = '\0';
	sata_split_model(model, &vendor, &product);

	if (vendor == NULL)
		nvme->n_vendor = strdup("NVMe");
	else
		nvme->n_vendor = strdup(vendor);

	nvme->n_product = strdup(product);

	/*
	 * Get controller limits.
	 */
	nvme->n_async_event_limit = MAX(NVME_MIN_ASYNC_EVENT_LIMIT,
	    MIN(nvme->n_admin_queue_len / 10,
	    MIN(nvme->n_idctl->id_aerl + 1, nvme->n_async_event_limit)));

	(void) ddi_prop_update_int(DDI_DEV_T_NONE, nvme->n_dip,
	    "async-event-limit", nvme->n_async_event_limit);

	nvme->n_abort_command_limit = nvme->n_idctl->id_acl + 1;

	/*
	 * Reinitialize the semaphore with the true abort command limit
	 * supported by the hardware. It's not necessary to disable interrupts
	 * as only command aborts use the semaphore, and no commands are
	 * executed or aborted while we're here.
	 */
	sema_destroy(&nvme->n_abort_sema);
	sema_init(&nvme->n_abort_sema, nvme->n_abort_command_limit - 1, NULL,
	    SEMA_DRIVER, NULL);

	nvme->n_progress |= NVME_CTRL_LIMITS;

	if (nvme->n_idctl->id_mdts == 0)
		nvme->n_max_data_transfer_size = nvme->n_pagesize * 65536;
	else
		nvme->n_max_data_transfer_size =
		    1ull << (nvme->n_pageshift + nvme->n_idctl->id_mdts);

	nvme->n_error_log_len = nvme->n_idctl->id_elpe + 1;

	/*
	 * Limit n_max_data_transfer_size to what we can handle in one PRP.
	 * Chained PRPs are currently unsupported.
	 *
	 * This is a no-op on hardware which doesn't support a transfer size
	 * big enough to require chained PRPs.
	 */
	nvme->n_max_data_transfer_size = MIN(nvme->n_max_data_transfer_size,
	    (nvme->n_pagesize / sizeof (uint64_t) * nvme->n_pagesize));

	nvme->n_prp_dma_attr.dma_attr_maxxfer = nvme->n_max_data_transfer_size;

	/*
	 * Make sure the minimum/maximum queue entry sizes are not
	 * larger/smaller than the default.
	 */

	if (((1 << nvme->n_idctl->id_sqes.qes_min) > sizeof (nvme_sqe_t)) ||
	    ((1 << nvme->n_idctl->id_sqes.qes_max) < sizeof (nvme_sqe_t)) ||
	    ((1 << nvme->n_idctl->id_cqes.qes_min) > sizeof (nvme_cqe_t)) ||
	    ((1 << nvme->n_idctl->id_cqes.qes_max) < sizeof (nvme_cqe_t)))
		goto fail;

	/*
	 * Check for the presence of a Volatile Write Cache. If present,
	 * enable or disable based on the value of the property
	 * volatile-write-cache-enable (default is enabled).
	 */
	nvme->n_write_cache_present =
	    nvme->n_idctl->id_vwc.vwc_present == 0 ? B_FALSE : B_TRUE;

	(void) ddi_prop_update_int(DDI_DEV_T_NONE, nvme->n_dip,
	    "volatile-write-cache-present",
	    nvme->n_write_cache_present ? 1 : 0);

	if (!nvme->n_write_cache_present) {
		nvme->n_write_cache_enabled = B_FALSE;
	} else if (nvme_write_cache_set(nvme, nvme->n_write_cache_enabled)
	    != 0) {
		dev_err(nvme->n_dip, CE_WARN,
		    "!failed to %sable volatile write cache",
		    nvme->n_write_cache_enabled ? "en" : "dis");
		/*
		 * Assume the cache is (still) enabled.
		 */
		nvme->n_write_cache_enabled = B_TRUE;
	}

	(void) ddi_prop_update_int(DDI_DEV_T_NONE, nvme->n_dip,
	    "volatile-write-cache-enable",
	    nvme->n_write_cache_enabled ? 1 : 0);

	/*
	 * Assume LBA Range Type feature is supported. If it isn't this
	 * will be set to B_FALSE by nvme_get_features().
	 */
	nvme->n_lba_range_supported = B_TRUE;

	/*
	 * Check support for Autonomous Power State Transition.
	 */
	if (NVME_VERSION_ATLEAST(&nvme->n_version, 1, 1))
		nvme->n_auto_pst_supported =
		    nvme->n_idctl->id_apsta.ap_sup == 0 ? B_FALSE : B_TRUE;

	/*
	 * Assume Software Progress Marker feature is supported.  If it isn't
	 * this will be set to B_FALSE by nvme_get_features().
	 */
	nvme->n_progress_supported = B_TRUE;

	/*
	 * Get number of supported namespaces and allocate namespace array.
	 */
	nvme->n_namespace_count = nvme->n_idctl->id_nn;

	if (nvme->n_namespace_count == 0) {
		dev_err(nvme->n_dip, CE_WARN,
		    "!controllers without namespaces are not supported");
		goto fail;
	}

	if (nvme->n_namespace_count > NVME_MINOR_MAX) {
		dev_err(nvme->n_dip, CE_WARN,
		    "!too many namespaces: %d, limiting to %d\n",
		    nvme->n_namespace_count, NVME_MINOR_MAX);
		nvme->n_namespace_count = NVME_MINOR_MAX;
	}

	nvme->n_ns = kmem_zalloc(sizeof (nvme_namespace_t) *
	    nvme->n_namespace_count, KM_SLEEP);

	/*
	 * Try to set up MSI/MSI-X interrupts.
	 */
	if ((nvme->n_intr_types & (DDI_INTR_TYPE_MSI | DDI_INTR_TYPE_MSIX))
	    != 0) {
		nvme_release_interrupts(nvme);

		nqueues = MIN(UINT16_MAX, ncpus);

		if ((nvme_setup_interrupts(nvme, DDI_INTR_TYPE_MSIX,
		    nqueues) != DDI_SUCCESS) &&
		    (nvme_setup_interrupts(nvme, DDI_INTR_TYPE_MSI,
		    nqueues) != DDI_SUCCESS)) {
			dev_err(nvme->n_dip, CE_WARN,
			    "!failed to set up MSI/MSI-X interrupts");
			goto fail;
		}
	}

	/*
	 * Create I/O queue pairs.
	 */

	if (nvme_set_nqueues(nvme) != 0) {
		dev_err(nvme->n_dip, CE_WARN,
		    "!failed to set number of I/O queues to %d",
		    nvme->n_intr_cnt);
		goto fail;
	}

	/*
	 * Reallocate I/O queue array
	 */
	kmem_free(nvme->n_ioq, sizeof (nvme_qpair_t *));
	nvme->n_ioq = kmem_zalloc(sizeof (nvme_qpair_t *) *
	    (nvme->n_submission_queues + 1), KM_SLEEP);
	nvme->n_ioq[0] = nvme->n_adminq;

	/*
	 * There should always be at least as many submission queues
	 * as completion queues.
	 */
	ASSERT(nvme->n_submission_queues >= nvme->n_completion_queues);

	nvme->n_ioq_count = nvme->n_submission_queues;

	nvme->n_io_squeue_len =
	    MIN(nvme->n_io_squeue_len, nvme->n_max_queue_entries);

	(void) ddi_prop_update_int(DDI_DEV_T_NONE, nvme->n_dip, "io-squeue-len",
	    nvme->n_io_squeue_len);

	/*
	 * Pre-allocate completion queues.
	 * When there are the same number of submission and completion
	 * queues there is no value in having a larger completion
	 * queue length.
	 */
	if (nvme->n_submission_queues == nvme->n_completion_queues)
		nvme->n_io_cqueue_len = MIN(nvme->n_io_cqueue_len,
		    nvme->n_io_squeue_len);

	nvme->n_io_cqueue_len = MIN(nvme->n_io_cqueue_len,
	    nvme->n_max_queue_entries);

	(void) ddi_prop_update_int(DDI_DEV_T_NONE, nvme->n_dip, "io-cqueue-len",
	    nvme->n_io_cqueue_len);

	/*
	 * Assign the equal quantity of taskq threads to each completion
	 * queue, capping the total number of threads to the number
	 * of CPUs.
	 */
	tq_threads = MIN(UINT16_MAX, ncpus) / nvme->n_completion_queues;

	/*
	 * In case the calculation above is zero, we need at least one
	 * thread per completion queue.
	 */
	tq_threads = MAX(1, tq_threads);

	if (nvme_create_cq_array(nvme, nvme->n_completion_queues + 1,
	    nvme->n_io_cqueue_len, tq_threads) != DDI_SUCCESS) {
		dev_err(nvme->n_dip, CE_WARN,
		    "!failed to pre-allocate completion queues");
		goto fail;
	}

	/*
	 * If we use less completion queues than interrupt vectors return
	 * some of the interrupt vectors back to the system.
	 */
	if (nvme->n_completion_queues + 1 < nvme->n_intr_cnt) {
		nvme_release_interrupts(nvme);

		if (nvme_setup_interrupts(nvme, nvme->n_intr_type,
		    nvme->n_completion_queues + 1) != DDI_SUCCESS) {
			dev_err(nvme->n_dip, CE_WARN,
			    "!failed to reduce number of interrupts");
			goto fail;
		}
	}

	/*
	 * Alloc & register I/O queue pairs
	 */

	for (i = 1; i != nvme->n_ioq_count + 1; i++) {
		if (nvme_alloc_qpair(nvme, nvme->n_io_squeue_len,
		    &nvme->n_ioq[i], i) != DDI_SUCCESS) {
			dev_err(nvme->n_dip, CE_WARN,
			    "!unable to allocate I/O qpair %d", i);
			goto fail;
		}

		if (nvme_create_io_qpair(nvme, nvme->n_ioq[i], i) != 0) {
			dev_err(nvme->n_dip, CE_WARN,
			    "!unable to create I/O qpair %d", i);
			goto fail;
		}
	}

	/*
	 * Post more asynchronous events commands to reduce event reporting
	 * latency as suggested by the spec.
	 */
	if (nvme->n_async_event_supported) {
		for (i = 1; i != nvme->n_async_event_limit; i++)
			nvme_async_event(nvme);
	}

	return (DDI_SUCCESS);

fail:
	(void) nvme_reset(nvme, B_FALSE);
	return (DDI_FAILURE);
}

static uint_t
nvme_intr(caddr_t arg1, caddr_t arg2)
{
	/*LINTED: E_PTR_BAD_CAST_ALIGN*/
	nvme_t *nvme = (nvme_t *)arg1;
	int inum = (int)(uintptr_t)arg2;
	int ccnt = 0;
	int qnum;

	if (inum >= nvme->n_intr_cnt)
		return (DDI_INTR_UNCLAIMED);

	if (nvme->n_dead)
		return (nvme->n_intr_type == DDI_INTR_TYPE_FIXED ?
		    DDI_INTR_UNCLAIMED : DDI_INTR_CLAIMED);

	/*
	 * The interrupt vector a queue uses is calculated as queue_idx %
	 * intr_cnt in nvme_create_io_qpair(). Iterate through the queue array
	 * in steps of n_intr_cnt to process all queues using this vector.
	 */
	for (qnum = inum;
	    qnum < nvme->n_cq_count && nvme->n_cq[qnum] != NULL;
	    qnum += nvme->n_intr_cnt) {
		ccnt += nvme_process_iocq(nvme, nvme->n_cq[qnum]);
	}

	return (ccnt > 0 ? DDI_INTR_CLAIMED : DDI_INTR_UNCLAIMED);
}

static void
nvme_release_interrupts(nvme_t *nvme)
{
	int i;

	for (i = 0; i < nvme->n_intr_cnt; i++) {
		if (nvme->n_inth[i] == NULL)
			break;

		if (nvme->n_intr_cap & DDI_INTR_FLAG_BLOCK)
			(void) ddi_intr_block_disable(&nvme->n_inth[i], 1);
		else
			(void) ddi_intr_disable(nvme->n_inth[i]);

		(void) ddi_intr_remove_handler(nvme->n_inth[i]);
		(void) ddi_intr_free(nvme->n_inth[i]);
	}

	kmem_free(nvme->n_inth, nvme->n_inth_sz);
	nvme->n_inth = NULL;
	nvme->n_inth_sz = 0;

	nvme->n_progress &= ~NVME_INTERRUPTS;
}

static int
nvme_setup_interrupts(nvme_t *nvme, int intr_type, int nqpairs)
{
	int nintrs, navail, count;
	int ret;
	int i;

	if (nvme->n_intr_types == 0) {
		ret = ddi_intr_get_supported_types(nvme->n_dip,
		    &nvme->n_intr_types);
		if (ret != DDI_SUCCESS) {
			dev_err(nvme->n_dip, CE_WARN,
			    "!%s: ddi_intr_get_supported types failed",
			    __func__);
			return (ret);
		}
#ifdef __x86
		if (get_hwenv() == HW_VMWARE)
			nvme->n_intr_types &= ~DDI_INTR_TYPE_MSIX;
#endif
	}

	if ((nvme->n_intr_types & intr_type) == 0)
		return (DDI_FAILURE);

	ret = ddi_intr_get_nintrs(nvme->n_dip, intr_type, &nintrs);
	if (ret != DDI_SUCCESS) {
		dev_err(nvme->n_dip, CE_WARN, "!%s: ddi_intr_get_nintrs failed",
		    __func__);
		return (ret);
	}

	ret = ddi_intr_get_navail(nvme->n_dip, intr_type, &navail);
	if (ret != DDI_SUCCESS) {
		dev_err(nvme->n_dip, CE_WARN, "!%s: ddi_intr_get_navail failed",
		    __func__);
		return (ret);
	}

	/* We want at most one interrupt per queue pair. */
	if (navail > nqpairs)
		navail = nqpairs;

	nvme->n_inth_sz = sizeof (ddi_intr_handle_t) * navail;
	nvme->n_inth = kmem_zalloc(nvme->n_inth_sz, KM_SLEEP);

	ret = ddi_intr_alloc(nvme->n_dip, nvme->n_inth, intr_type, 0, navail,
	    &count, 0);
	if (ret != DDI_SUCCESS) {
		dev_err(nvme->n_dip, CE_WARN, "!%s: ddi_intr_alloc failed",
		    __func__);
		goto fail;
	}

	nvme->n_intr_cnt = count;

	ret = ddi_intr_get_pri(nvme->n_inth[0], &nvme->n_intr_pri);
	if (ret != DDI_SUCCESS) {
		dev_err(nvme->n_dip, CE_WARN, "!%s: ddi_intr_get_pri failed",
		    __func__);
		goto fail;
	}

	for (i = 0; i < count; i++) {
		ret = ddi_intr_add_handler(nvme->n_inth[i], nvme_intr,
		    (void *)nvme, (void *)(uintptr_t)i);
		if (ret != DDI_SUCCESS) {
			dev_err(nvme->n_dip, CE_WARN,
			    "!%s: ddi_intr_add_handler failed", __func__);
			goto fail;
		}
	}

	(void) ddi_intr_get_cap(nvme->n_inth[0], &nvme->n_intr_cap);

	for (i = 0; i < count; i++) {
		if (nvme->n_intr_cap & DDI_INTR_FLAG_BLOCK)
			ret = ddi_intr_block_enable(&nvme->n_inth[i], 1);
		else
			ret = ddi_intr_enable(nvme->n_inth[i]);

		if (ret != DDI_SUCCESS) {
			dev_err(nvme->n_dip, CE_WARN,
			    "!%s: enabling interrupt %d failed", __func__, i);
			goto fail;
		}
	}

	nvme->n_intr_type = intr_type;

	nvme->n_progress |= NVME_INTERRUPTS;

	return (DDI_SUCCESS);

fail:
	nvme_release_interrupts(nvme);

	return (ret);
}

static int
nvme_fm_errcb(dev_info_t *dip, ddi_fm_error_t *fm_error, const void *arg)
{
	_NOTE(ARGUNUSED(arg));

	pci_ereport_post(dip, fm_error, NULL);
	return (fm_error->fme_status);
}

static void
nvme_remove_callback(dev_info_t *dip, ddi_eventcookie_t cookie, void *a,
    void *b)
{
	nvme_t *nvme = a;

	nvme->n_dead = B_TRUE;

	/*
	 * Fail all outstanding commands, including those in the admin queue
	 * (queue 0).
	 */
	for (uint_t i = 0; i < nvme->n_ioq_count + 1; i++) {
		nvme_qpair_t *qp = nvme->n_ioq[i];

		mutex_enter(&qp->nq_mutex);
		for (size_t j = 0; j < qp->nq_nentry; j++) {
			nvme_cmd_t *cmd = qp->nq_cmd[j];
			nvme_cmd_t *u_cmd;

			if (cmd == NULL) {
				continue;
			}

			/*
			 * Since we have the queue lock held the entire time we
			 * iterate over it, it's not possible for the queue to
			 * change underneath us. Thus, we don't need to check
			 * that the return value of nvme_unqueue_cmd matches the
			 * requested cmd to unqueue.
			 */
			u_cmd = nvme_unqueue_cmd(nvme, qp, cmd->nc_sqe.sqe_cid);
			taskq_dispatch_ent(qp->nq_cq->ncq_cmd_taskq,
			    cmd->nc_callback, cmd, TQ_NOSLEEP, &cmd->nc_tqent);

			ASSERT3P(u_cmd, ==, cmd);
		}
		mutex_exit(&qp->nq_mutex);
	}
}

static int
nvme_attach(dev_info_t *dip, ddi_attach_cmd_t cmd)
{
	nvme_t *nvme;
	int instance;
	int nregs;
	off_t regsize;
	int i;
	char name[32];
	boolean_t attached_ns;

	if (cmd != DDI_ATTACH)
		return (DDI_FAILURE);

	instance = ddi_get_instance(dip);

	if (ddi_soft_state_zalloc(nvme_state, instance) != DDI_SUCCESS)
		return (DDI_FAILURE);

	nvme = ddi_get_soft_state(nvme_state, instance);
	ddi_set_driver_private(dip, nvme);
	nvme->n_dip = dip;

	/*
	 * Map PCI config space
	 */
	if (pci_config_setup(dip, &nvme->n_pcicfg_handle) != DDI_SUCCESS) {
		dev_err(dip, CE_WARN, "!failed to map PCI config space");
		goto fail;
	}
	nvme->n_progress |= NVME_PCI_CONFIG;

	/*
	 * Get the various PCI IDs from config space
	 */
	nvme->n_vendor_id =
	    pci_config_get16(nvme->n_pcicfg_handle, PCI_CONF_VENID);
	nvme->n_device_id =
	    pci_config_get16(nvme->n_pcicfg_handle, PCI_CONF_DEVID);
	nvme->n_revision_id =
	    pci_config_get8(nvme->n_pcicfg_handle, PCI_CONF_REVID);
	nvme->n_subsystem_device_id =
	    pci_config_get16(nvme->n_pcicfg_handle, PCI_CONF_SUBSYSID);
	nvme->n_subsystem_vendor_id =
	    pci_config_get16(nvme->n_pcicfg_handle, PCI_CONF_SUBVENID);

	nvme_detect_quirks(nvme);

	/*
	 * Set up event handlers for hot removal. While npe(4D) supports the hot
	 * removal event being injected for devices, the same is not true of all
	 * of our possible parents (i.e. pci(4D) as of this writing). The most
	 * common case this shows up is in some virtualization environments. We
	 * should treat this as non-fatal so that way devices work but leave
	 * this set up in such a way that if a nexus does grow support for this
	 * we're good to go.
	 */
	if (ddi_get_eventcookie(nvme->n_dip, DDI_DEVI_REMOVE_EVENT,
	    &nvme->n_rm_cookie) == DDI_SUCCESS) {
		if (ddi_add_event_handler(nvme->n_dip, nvme->n_rm_cookie,
		    nvme_remove_callback, nvme, &nvme->n_ev_rm_cb_id) !=
		    DDI_SUCCESS) {
			goto fail;
		}
	} else {
		nvme->n_ev_rm_cb_id = NULL;
	}

	mutex_init(&nvme->n_minor_mutex, NULL, MUTEX_DRIVER, NULL);
	nvme->n_progress |= NVME_MUTEX_INIT;

	nvme->n_strict_version = ddi_prop_get_int(DDI_DEV_T_ANY, dip,
	    DDI_PROP_DONTPASS, "strict-version", 1) == 1 ? B_TRUE : B_FALSE;
	nvme->n_ignore_unknown_vendor_status = ddi_prop_get_int(DDI_DEV_T_ANY,
	    dip, DDI_PROP_DONTPASS, "ignore-unknown-vendor-status", 0) == 1 ?
	    B_TRUE : B_FALSE;
	nvme->n_admin_queue_len = ddi_prop_get_int(DDI_DEV_T_ANY, dip,
	    DDI_PROP_DONTPASS, "admin-queue-len", NVME_DEFAULT_ADMIN_QUEUE_LEN);
	nvme->n_io_squeue_len = ddi_prop_get_int(DDI_DEV_T_ANY, dip,
	    DDI_PROP_DONTPASS, "io-squeue-len", NVME_DEFAULT_IO_QUEUE_LEN);
	/*
	 * Double up the default for completion queues in case of
	 * queue sharing.
	 */
	nvme->n_io_cqueue_len = ddi_prop_get_int(DDI_DEV_T_ANY, dip,
	    DDI_PROP_DONTPASS, "io-cqueue-len", 2 * NVME_DEFAULT_IO_QUEUE_LEN);
	nvme->n_async_event_limit = ddi_prop_get_int(DDI_DEV_T_ANY, dip,
	    DDI_PROP_DONTPASS, "async-event-limit",
	    NVME_DEFAULT_ASYNC_EVENT_LIMIT);
	nvme->n_write_cache_enabled = ddi_prop_get_int(DDI_DEV_T_ANY, dip,
	    DDI_PROP_DONTPASS, "volatile-write-cache-enable", 1) != 0 ?
	    B_TRUE : B_FALSE;
	nvme->n_min_block_size = ddi_prop_get_int(DDI_DEV_T_ANY, dip,
	    DDI_PROP_DONTPASS, "min-phys-block-size",
	    NVME_DEFAULT_MIN_BLOCK_SIZE);
	nvme->n_submission_queues = ddi_prop_get_int(DDI_DEV_T_ANY, dip,
	    DDI_PROP_DONTPASS, "max-submission-queues", -1);
	nvme->n_completion_queues = ddi_prop_get_int(DDI_DEV_T_ANY, dip,
	    DDI_PROP_DONTPASS, "max-completion-queues", -1);

	if (!ISP2(nvme->n_min_block_size) ||
	    (nvme->n_min_block_size < NVME_DEFAULT_MIN_BLOCK_SIZE)) {
		dev_err(dip, CE_WARN, "!min-phys-block-size %s, "
		    "using default %d", ISP2(nvme->n_min_block_size) ?
		    "too low" : "not a power of 2",
		    NVME_DEFAULT_MIN_BLOCK_SIZE);
		nvme->n_min_block_size = NVME_DEFAULT_MIN_BLOCK_SIZE;
	}

	if (nvme->n_submission_queues != -1 &&
	    (nvme->n_submission_queues < 1 ||
	    nvme->n_submission_queues > UINT16_MAX)) {
		dev_err(dip, CE_WARN, "!\"submission-queues\"=%d is not "
		    "valid. Must be [1..%d]", nvme->n_submission_queues,
		    UINT16_MAX);
		nvme->n_submission_queues = -1;
	}

	if (nvme->n_completion_queues != -1 &&
	    (nvme->n_completion_queues < 1 ||
	    nvme->n_completion_queues > UINT16_MAX)) {
		dev_err(dip, CE_WARN, "!\"completion-queues\"=%d is not "
		    "valid. Must be [1..%d]", nvme->n_completion_queues,
		    UINT16_MAX);
		nvme->n_completion_queues = -1;
	}

	if (nvme->n_admin_queue_len < NVME_MIN_ADMIN_QUEUE_LEN)
		nvme->n_admin_queue_len = NVME_MIN_ADMIN_QUEUE_LEN;
	else if (nvme->n_admin_queue_len > NVME_MAX_ADMIN_QUEUE_LEN)
		nvme->n_admin_queue_len = NVME_MAX_ADMIN_QUEUE_LEN;

	if (nvme->n_io_squeue_len < NVME_MIN_IO_QUEUE_LEN)
		nvme->n_io_squeue_len = NVME_MIN_IO_QUEUE_LEN;
	if (nvme->n_io_cqueue_len < NVME_MIN_IO_QUEUE_LEN)
		nvme->n_io_cqueue_len = NVME_MIN_IO_QUEUE_LEN;

	if (nvme->n_async_event_limit < 1)
		nvme->n_async_event_limit = NVME_DEFAULT_ASYNC_EVENT_LIMIT;

	nvme->n_reg_acc_attr = nvme_reg_acc_attr;
	nvme->n_queue_dma_attr = nvme_queue_dma_attr;
	nvme->n_prp_dma_attr = nvme_prp_dma_attr;
	nvme->n_sgl_dma_attr = nvme_sgl_dma_attr;

	/*
	 * Set up FMA support.
	 */
	nvme->n_fm_cap = ddi_getprop(DDI_DEV_T_ANY, dip,
	    DDI_PROP_CANSLEEP | DDI_PROP_DONTPASS, "fm-capable",
	    DDI_FM_EREPORT_CAPABLE | DDI_FM_ACCCHK_CAPABLE |
	    DDI_FM_DMACHK_CAPABLE | DDI_FM_ERRCB_CAPABLE);

	ddi_fm_init(dip, &nvme->n_fm_cap, &nvme->n_fm_ibc);

	if (nvme->n_fm_cap) {
		if (nvme->n_fm_cap & DDI_FM_ACCCHK_CAPABLE)
			nvme->n_reg_acc_attr.devacc_attr_access =
			    DDI_FLAGERR_ACC;

		if (nvme->n_fm_cap & DDI_FM_DMACHK_CAPABLE) {
			nvme->n_prp_dma_attr.dma_attr_flags |= DDI_DMA_FLAGERR;
			nvme->n_sgl_dma_attr.dma_attr_flags |= DDI_DMA_FLAGERR;
		}

		if (DDI_FM_EREPORT_CAP(nvme->n_fm_cap) ||
		    DDI_FM_ERRCB_CAP(nvme->n_fm_cap))
			pci_ereport_setup(dip);

		if (DDI_FM_ERRCB_CAP(nvme->n_fm_cap))
			ddi_fm_handler_register(dip, nvme_fm_errcb,
			    (void *)nvme);
	}

	nvme->n_progress |= NVME_FMA_INIT;

	/*
	 * The spec defines several register sets. Only the controller
	 * registers (set 1) are currently used.
	 */
	if (ddi_dev_nregs(dip, &nregs) == DDI_FAILURE ||
	    nregs < 2 ||
	    ddi_dev_regsize(dip, 1, &regsize) == DDI_FAILURE)
		goto fail;

	if (ddi_regs_map_setup(dip, 1, &nvme->n_regs, 0, regsize,
	    &nvme->n_reg_acc_attr, &nvme->n_regh) != DDI_SUCCESS) {
		dev_err(dip, CE_WARN, "!failed to map regset 1");
		goto fail;
	}

	nvme->n_progress |= NVME_REGS_MAPPED;

	/*
	 * Create PRP DMA cache
	 */
	(void) snprintf(name, sizeof (name), "%s%d_prp_cache",
	    ddi_driver_name(dip), ddi_get_instance(dip));
	nvme->n_prp_cache = kmem_cache_create(name, sizeof (nvme_dma_t),
	    0, nvme_prp_dma_constructor, nvme_prp_dma_destructor,
	    NULL, (void *)nvme, NULL, 0);

	if (nvme_init(nvme) != DDI_SUCCESS)
		goto fail;

	/*
	 * Initialize the driver with the UFM subsystem
	 */
	if (ddi_ufm_init(dip, DDI_UFM_CURRENT_VERSION, &nvme_ufm_ops,
	    &nvme->n_ufmh, nvme) != 0) {
		dev_err(dip, CE_WARN, "!failed to initialize UFM subsystem");
		goto fail;
	}
	mutex_init(&nvme->n_fwslot_mutex, NULL, MUTEX_DRIVER, NULL);
	ddi_ufm_update(nvme->n_ufmh);
	nvme->n_progress |= NVME_UFM_INIT;

	mutex_init(&nvme->n_mgmt_mutex, NULL, MUTEX_DRIVER, NULL);
	nvme->n_progress |= NVME_MGMT_INIT;

	/*
	 * Identify namespaces.
	 */
	mutex_enter(&nvme->n_mgmt_mutex);

	for (i = 1; i <= nvme->n_namespace_count; i++) {
		nvme_namespace_t *ns = NVME_NSID2NS(nvme, i);

		/*
		 * Namespaces start out ignored. When nvme_init_ns() checks
		 * their properties and finds they can be used, it will set
		 * ns_ignore to B_FALSE. It will also use this state change
		 * to keep an accurate count of attachable namespaces.
		 */
		ns->ns_ignore = B_TRUE;
		if (nvme_init_ns(nvme, i) != 0) {
			mutex_exit(&nvme->n_mgmt_mutex);
			goto fail;
		}

		if (ddi_create_minor_node(nvme->n_dip, ns->ns_name, S_IFCHR,
		    NVME_MINOR(ddi_get_instance(nvme->n_dip), i),
		    DDI_NT_NVME_ATTACHMENT_POINT, 0) != DDI_SUCCESS) {
			mutex_exit(&nvme->n_mgmt_mutex);
			dev_err(dip, CE_WARN,
			    "!failed to create minor node for namespace %d", i);
			goto fail;
		}
	}

	if (ddi_create_minor_node(dip, "devctl", S_IFCHR,
	    NVME_MINOR(ddi_get_instance(dip), 0), DDI_NT_NVME_NEXUS, 0)
	    != DDI_SUCCESS) {
		mutex_exit(&nvme->n_mgmt_mutex);
		dev_err(dip, CE_WARN, "nvme_attach: "
		    "cannot create devctl minor node");
		goto fail;
	}

	attached_ns = B_FALSE;
	for (i = 1; i <= nvme->n_namespace_count; i++) {
		int rv;

		rv = nvme_attach_ns(nvme, i);
		if (rv == 0) {
			attached_ns = B_TRUE;
		} else if (rv != ENOTSUP) {
			dev_err(nvme->n_dip, CE_WARN,
			    "!failed to attach namespace %d: %d", i, rv);
			/*
			 * Once we have successfully attached a namespace we
			 * can no longer fail the driver attach as there is now
			 * a blkdev child node linked to this device, and
			 * our node is not yet in the attached state.
			 */
			if (!attached_ns) {
				mutex_exit(&nvme->n_mgmt_mutex);
				goto fail;
			}
		}
	}

	mutex_exit(&nvme->n_mgmt_mutex);

	return (DDI_SUCCESS);

fail:
	/* attach successful anyway so that FMA can retire the device */
	if (nvme->n_dead)
		return (DDI_SUCCESS);

	(void) nvme_detach(dip, DDI_DETACH);

	return (DDI_FAILURE);
}

static int
nvme_detach(dev_info_t *dip, ddi_detach_cmd_t cmd)
{
	int instance, i;
	nvme_t *nvme;

	if (cmd != DDI_DETACH)
		return (DDI_FAILURE);

	instance = ddi_get_instance(dip);

	nvme = ddi_get_soft_state(nvme_state, instance);

	if (nvme == NULL)
		return (DDI_FAILURE);

	/*
	 * Remove all minor nodes from the device regardless of the source in
	 * one swoop.
	 */
	ddi_remove_minor_node(dip, NULL);

	/*
	 * We need to remove the event handler as one of the first things that
	 * we do. If we proceed with other teardown without removing the event
	 * handler, we could end up in a very unfortunate race with ourselves.
	 * The DDI does not serialize these with detach (just like timeout(9F)
	 * and others).
	 */
	if (nvme->n_ev_rm_cb_id != NULL) {
		(void) ddi_remove_event_handler(nvme->n_ev_rm_cb_id);
	}
	nvme->n_ev_rm_cb_id = NULL;

	if (nvme->n_ns) {
		for (i = 1; i <= nvme->n_namespace_count; i++) {
			nvme_namespace_t *ns = NVME_NSID2NS(nvme, i);

			if (ns->ns_bd_hdl) {
				(void) bd_detach_handle(ns->ns_bd_hdl);
				bd_free_handle(ns->ns_bd_hdl);
			}

			if (ns->ns_idns)
				kmem_free(ns->ns_idns,
				    sizeof (nvme_identify_nsid_t));
			if (ns->ns_devid)
				strfree(ns->ns_devid);
		}

		kmem_free(nvme->n_ns, sizeof (nvme_namespace_t) *
		    nvme->n_namespace_count);
	}

	if (nvme->n_progress & NVME_MGMT_INIT) {
		mutex_destroy(&nvme->n_mgmt_mutex);
	}

	if (nvme->n_progress & NVME_UFM_INIT) {
		ddi_ufm_fini(nvme->n_ufmh);
		mutex_destroy(&nvme->n_fwslot_mutex);
	}

	if (nvme->n_progress & NVME_INTERRUPTS)
		nvme_release_interrupts(nvme);

	for (i = 0; i < nvme->n_cq_count; i++) {
		if (nvme->n_cq[i]->ncq_cmd_taskq != NULL)
			taskq_wait(nvme->n_cq[i]->ncq_cmd_taskq);
	}

	if (nvme->n_progress & NVME_MUTEX_INIT) {
		mutex_destroy(&nvme->n_minor_mutex);
	}

	if (nvme->n_ioq_count > 0) {
		for (i = 1; i != nvme->n_ioq_count + 1; i++) {
			if (nvme->n_ioq[i] != NULL) {
				/* TODO: send destroy queue commands */
				nvme_free_qpair(nvme->n_ioq[i]);
			}
		}

		kmem_free(nvme->n_ioq, sizeof (nvme_qpair_t *) *
		    (nvme->n_ioq_count + 1));
	}

	if (nvme->n_prp_cache != NULL) {
		kmem_cache_destroy(nvme->n_prp_cache);
	}

	if (nvme->n_progress & NVME_REGS_MAPPED) {
		nvme_shutdown(nvme, B_FALSE);
		(void) nvme_reset(nvme, B_FALSE);
	}

	if (nvme->n_progress & NVME_CTRL_LIMITS)
		sema_destroy(&nvme->n_abort_sema);

	if (nvme->n_progress & NVME_ADMIN_QUEUE)
		nvme_free_qpair(nvme->n_adminq);

	if (nvme->n_cq_count > 0) {
		nvme_destroy_cq_array(nvme, 0);
		nvme->n_cq = NULL;
		nvme->n_cq_count = 0;
	}

	if (nvme->n_idctl)
		kmem_free(nvme->n_idctl, NVME_IDENTIFY_BUFSIZE);

	if (nvme->n_progress & NVME_REGS_MAPPED)
		ddi_regs_map_free(&nvme->n_regh);

	if (nvme->n_progress & NVME_FMA_INIT) {
		if (DDI_FM_ERRCB_CAP(nvme->n_fm_cap))
			ddi_fm_handler_unregister(nvme->n_dip);

		if (DDI_FM_EREPORT_CAP(nvme->n_fm_cap) ||
		    DDI_FM_ERRCB_CAP(nvme->n_fm_cap))
			pci_ereport_teardown(nvme->n_dip);

		ddi_fm_fini(nvme->n_dip);
	}

	if (nvme->n_progress & NVME_PCI_CONFIG)
		pci_config_teardown(&nvme->n_pcicfg_handle);

	if (nvme->n_vendor != NULL)
		strfree(nvme->n_vendor);

	if (nvme->n_product != NULL)
		strfree(nvme->n_product);

	ddi_soft_state_free(nvme_state, instance);

	return (DDI_SUCCESS);
}

static int
nvme_quiesce(dev_info_t *dip)
{
	int instance;
	nvme_t *nvme;

	instance = ddi_get_instance(dip);

	nvme = ddi_get_soft_state(nvme_state, instance);

	if (nvme == NULL)
		return (DDI_FAILURE);

	nvme_shutdown(nvme, B_TRUE);

	(void) nvme_reset(nvme, B_TRUE);

	return (DDI_SUCCESS);
}

static int
nvme_fill_prp(nvme_cmd_t *cmd, ddi_dma_handle_t dma)
{
	nvme_t *nvme = cmd->nc_nvme;
	uint_t nprp_per_page, nprp;
	uint64_t *prp;
	const ddi_dma_cookie_t *cookie;
	uint_t idx;
	uint_t ncookies = ddi_dma_ncookies(dma);

	if (ncookies == 0)
		return (DDI_FAILURE);

	if ((cookie = ddi_dma_cookie_get(dma, 0)) == NULL)
		return (DDI_FAILURE);
	cmd->nc_sqe.sqe_dptr.d_prp[0] = cookie->dmac_laddress;

	if (ncookies == 1) {
		cmd->nc_sqe.sqe_dptr.d_prp[1] = 0;
		return (DDI_SUCCESS);
	} else if (ncookies == 2) {
		if ((cookie = ddi_dma_cookie_get(dma, 1)) == NULL)
			return (DDI_FAILURE);
		cmd->nc_sqe.sqe_dptr.d_prp[1] = cookie->dmac_laddress;
		return (DDI_SUCCESS);
	}

	/*
	 * At this point, we're always operating on cookies at
	 * index >= 1 and writing the addresses of those cookies
	 * into a new page. The address of that page is stored
	 * as the second PRP entry.
	 */
	nprp_per_page = nvme->n_pagesize / sizeof (uint64_t);
	ASSERT(nprp_per_page > 0);

	/*
	 * We currently don't support chained PRPs and set up our DMA
	 * attributes to reflect that. If we still get an I/O request
	 * that needs a chained PRP something is very wrong. Account
	 * for the first cookie here, which we've placed in d_prp[0].
	 */
	nprp = howmany(ncookies - 1, nprp_per_page);
	VERIFY(nprp == 1);

	/*
	 * Allocate a page of pointers, in which we'll write the
	 * addresses of cookies 1 to `ncookies`.
	 */
	cmd->nc_prp = kmem_cache_alloc(nvme->n_prp_cache, KM_SLEEP);
	bzero(cmd->nc_prp->nd_memp, cmd->nc_prp->nd_len);
	cmd->nc_sqe.sqe_dptr.d_prp[1] = cmd->nc_prp->nd_cookie.dmac_laddress;

	prp = (uint64_t *)cmd->nc_prp->nd_memp;
	for (idx = 1; idx < ncookies; idx++) {
		if ((cookie = ddi_dma_cookie_get(dma, idx)) == NULL)
			return (DDI_FAILURE);
		*prp++ = cookie->dmac_laddress;
	}

	(void) ddi_dma_sync(cmd->nc_prp->nd_dmah, 0, cmd->nc_prp->nd_len,
	    DDI_DMA_SYNC_FORDEV);
	return (DDI_SUCCESS);
}

/*
 * The maximum number of requests supported for a deallocate request is
 * NVME_DSET_MGMT_MAX_RANGES (256) -- this is from the NVMe 1.1 spec (and
 * unchanged through at least 1.4a). The definition of nvme_range_t is also
 * from the NVMe 1.1 spec. Together, the result is that all of the ranges for
 * a deallocate request will fit into the smallest supported namespace page
 * (4k).
 */
CTASSERT(sizeof (nvme_range_t) * NVME_DSET_MGMT_MAX_RANGES == 4096);

static int
nvme_fill_ranges(nvme_cmd_t *cmd, bd_xfer_t *xfer, uint64_t blocksize,
    int allocflag)
{
	const dkioc_free_list_t *dfl = xfer->x_dfl;
	const dkioc_free_list_ext_t *exts = dfl->dfl_exts;
	nvme_t *nvme = cmd->nc_nvme;
	nvme_range_t *ranges = NULL;
	uint_t i;

	/*
	 * The number of ranges in the request is 0s based (that is
	 * word10 == 0 -> 1 range, word10 == 1 -> 2 ranges, ...,
	 * word10 == 255 -> 256 ranges). Therefore the allowed values are
	 * [1..NVME_DSET_MGMT_MAX_RANGES]. If blkdev gives us a bad request,
	 * we either provided bad info in nvme_bd_driveinfo() or there is a bug
	 * in blkdev.
	 */
	VERIFY3U(dfl->dfl_num_exts, >, 0);
	VERIFY3U(dfl->dfl_num_exts, <=, NVME_DSET_MGMT_MAX_RANGES);
	cmd->nc_sqe.sqe_cdw10 = (dfl->dfl_num_exts - 1) & 0xff;

	cmd->nc_sqe.sqe_cdw11 = NVME_DSET_MGMT_ATTR_DEALLOCATE;

	cmd->nc_prp = kmem_cache_alloc(nvme->n_prp_cache, allocflag);
	if (cmd->nc_prp == NULL)
		return (DDI_FAILURE);

	bzero(cmd->nc_prp->nd_memp, cmd->nc_prp->nd_len);
	ranges = (nvme_range_t *)cmd->nc_prp->nd_memp;

	cmd->nc_sqe.sqe_dptr.d_prp[0] = cmd->nc_prp->nd_cookie.dmac_laddress;
	cmd->nc_sqe.sqe_dptr.d_prp[1] = 0;

	for (i = 0; i < dfl->dfl_num_exts; i++) {
		uint64_t lba, len;

		lba = (dfl->dfl_offset + exts[i].dfle_start) / blocksize;
		len = exts[i].dfle_length / blocksize;

		VERIFY3U(len, <=, UINT32_MAX);

		/* No context attributes for a deallocate request */
		ranges[i].nr_ctxattr = 0;
		ranges[i].nr_len = len;
		ranges[i].nr_lba = lba;
	}

	(void) ddi_dma_sync(cmd->nc_prp->nd_dmah, 0, cmd->nc_prp->nd_len,
	    DDI_DMA_SYNC_FORDEV);

	return (DDI_SUCCESS);
}

static nvme_cmd_t *
nvme_create_nvm_cmd(nvme_namespace_t *ns, uint8_t opc, bd_xfer_t *xfer)
{
	nvme_t *nvme = ns->ns_nvme;
	nvme_cmd_t *cmd;
	int allocflag;

	/*
	 * Blkdev only sets BD_XFER_POLL when dumping, so don't sleep.
	 */
	allocflag = (xfer->x_flags & BD_XFER_POLL) ? KM_NOSLEEP : KM_SLEEP;
	cmd = nvme_alloc_cmd(nvme, allocflag);

	if (cmd == NULL)
		return (NULL);

	cmd->nc_sqe.sqe_opc = opc;
	cmd->nc_callback = nvme_bd_xfer_done;
	cmd->nc_xfer = xfer;

	switch (opc) {
	case NVME_OPC_NVM_WRITE:
	case NVME_OPC_NVM_READ:
		VERIFY(xfer->x_nblks <= 0x10000);

		cmd->nc_sqe.sqe_nsid = ns->ns_id;

		cmd->nc_sqe.sqe_cdw10 = xfer->x_blkno & 0xffffffffu;
		cmd->nc_sqe.sqe_cdw11 = (xfer->x_blkno >> 32);
		cmd->nc_sqe.sqe_cdw12 = (uint16_t)(xfer->x_nblks - 1);

		if (nvme_fill_prp(cmd, xfer->x_dmah) != DDI_SUCCESS)
			goto fail;
		break;

	case NVME_OPC_NVM_FLUSH:
		cmd->nc_sqe.sqe_nsid = ns->ns_id;
		break;

	case NVME_OPC_NVM_DSET_MGMT:
		cmd->nc_sqe.sqe_nsid = ns->ns_id;

		if (nvme_fill_ranges(cmd, xfer,
		    (uint64_t)ns->ns_block_size, allocflag) != DDI_SUCCESS)
			goto fail;
		break;

	default:
		goto fail;
	}

	return (cmd);

fail:
	nvme_free_cmd(cmd);
	return (NULL);
}

static void
nvme_bd_xfer_done(void *arg)
{
	nvme_cmd_t *cmd = arg;
	bd_xfer_t *xfer = cmd->nc_xfer;
	int error = 0;

	error = nvme_check_cmd_status(cmd);
	nvme_free_cmd(cmd);

	bd_xfer_done(xfer, error);
}

static void
nvme_bd_driveinfo(void *arg, bd_drive_t *drive)
{
	nvme_namespace_t *ns = arg;
	nvme_t *nvme = ns->ns_nvme;
	uint_t ns_count = MAX(1, nvme->n_namespaces_attachable);
	boolean_t mutex_exit_needed = B_TRUE;

	/*
	 * nvme_bd_driveinfo is called by blkdev in two situations:
	 * - during bd_attach_handle(), which we call with the mutex held
	 * - during bd_attach(), which may be called with or without the
	 *   mutex held
	 */
	if (mutex_owned(&nvme->n_mgmt_mutex))
		mutex_exit_needed = B_FALSE;
	else
		mutex_enter(&nvme->n_mgmt_mutex);

	/*
	 * Set the blkdev qcount to the number of submission queues.
	 * It will then create one waitq/runq pair for each submission
	 * queue and spread I/O requests across the queues.
	 */
	drive->d_qcount = nvme->n_ioq_count;

	/*
	 * I/O activity to individual namespaces is distributed across
	 * each of the d_qcount blkdev queues (which has been set to
	 * the number of nvme submission queues). d_qsize is the number
	 * of submitted and not completed I/Os within each queue that blkdev
	 * will allow before it starts holding them in the waitq.
	 *
	 * Each namespace will create a child blkdev instance, for each one
	 * we try and set the d_qsize so that each namespace gets an
	 * equal portion of the submission queue.
	 *
	 * If post instantiation of the nvme drive, n_namespaces_attachable
	 * changes and a namespace is attached it could calculate a
	 * different d_qsize. It may even be that the sum of the d_qsizes is
	 * now beyond the submission queue size. Should that be the case
	 * and the I/O rate is such that blkdev attempts to submit more
	 * I/Os than the size of the submission queue, the excess I/Os
	 * will be held behind the semaphore nq_sema.
	 */
	drive->d_qsize = nvme->n_io_squeue_len / ns_count;

	/*
	 * Don't let the queue size drop below the minimum, though.
	 */
	drive->d_qsize = MAX(drive->d_qsize, NVME_MIN_IO_QUEUE_LEN);

	/*
	 * d_maxxfer is not set, which means the value is taken from the DMA
	 * attributes specified to bd_alloc_handle.
	 */

	drive->d_removable = B_FALSE;
	drive->d_hotpluggable = B_FALSE;

	bcopy(ns->ns_eui64, drive->d_eui64, sizeof (drive->d_eui64));
	drive->d_target = ns->ns_id;
	drive->d_lun = 0;

	drive->d_model = nvme->n_idctl->id_model;
	drive->d_model_len = sizeof (nvme->n_idctl->id_model);
	drive->d_vendor = nvme->n_vendor;
	drive->d_vendor_len = strlen(nvme->n_vendor);
	drive->d_product = nvme->n_product;
	drive->d_product_len = strlen(nvme->n_product);
	drive->d_serial = nvme->n_idctl->id_serial;
	drive->d_serial_len = sizeof (nvme->n_idctl->id_serial);
	drive->d_revision = nvme->n_idctl->id_fwrev;
	drive->d_revision_len = sizeof (nvme->n_idctl->id_fwrev);

	/*
	 * If we support the dataset management command, the only restrictions
	 * on a discard request are the maximum number of ranges (segments)
	 * per single request.
	 */
	if (nvme->n_idctl->id_oncs.on_dset_mgmt)
		drive->d_max_free_seg = NVME_DSET_MGMT_MAX_RANGES;

	if (mutex_exit_needed)
		mutex_exit(&nvme->n_mgmt_mutex);
}

static int
nvme_bd_mediainfo(void *arg, bd_media_t *media)
{
	nvme_namespace_t *ns = arg;
	nvme_t *nvme = ns->ns_nvme;
	boolean_t mutex_exit_needed = B_TRUE;

	if (nvme->n_dead) {
		return (EIO);
	}

	/*
	 * nvme_bd_mediainfo is called by blkdev in various situations,
	 * most of them out of our control. There's one exception though:
	 * When we call bd_state_change() in response to "namespace change"
	 * notification, where the mutex is already being held by us.
	 */
	if (mutex_owned(&nvme->n_mgmt_mutex))
		mutex_exit_needed = B_FALSE;
	else
		mutex_enter(&nvme->n_mgmt_mutex);

	media->m_nblks = ns->ns_block_count;
	media->m_blksize = ns->ns_block_size;
	media->m_readonly = B_FALSE;
	media->m_solidstate = B_TRUE;

	media->m_pblksize = ns->ns_best_block_size;

	if (mutex_exit_needed)
		mutex_exit(&nvme->n_mgmt_mutex);

	return (0);
}

static int
nvme_bd_cmd(nvme_namespace_t *ns, bd_xfer_t *xfer, uint8_t opc)
{
	nvme_t *nvme = ns->ns_nvme;
	nvme_cmd_t *cmd;
	nvme_qpair_t *ioq;
	boolean_t poll;
	int ret;

	if (nvme->n_dead) {
		return (EIO);
	}

	cmd = nvme_create_nvm_cmd(ns, opc, xfer);
	if (cmd == NULL)
		return (ENOMEM);

	cmd->nc_sqid = xfer->x_qnum + 1;
	ASSERT(cmd->nc_sqid <= nvme->n_ioq_count);
	ioq = nvme->n_ioq[cmd->nc_sqid];

	/*
	 * Get the polling flag before submitting the command. The command may
	 * complete immediately after it was submitted, which means we must
	 * treat both cmd and xfer as if they have been freed already.
	 */
	poll = (xfer->x_flags & BD_XFER_POLL) != 0;

	ret = nvme_submit_io_cmd(ioq, cmd);

	if (ret != 0)
		return (ret);

	if (!poll)
		return (0);

	do {
		cmd = nvme_retrieve_cmd(nvme, ioq);
		if (cmd != NULL)
			cmd->nc_callback(cmd);
		else
			drv_usecwait(10);
	} while (ioq->nq_active_cmds != 0);

	return (0);
}

static int
nvme_bd_read(void *arg, bd_xfer_t *xfer)
{
	nvme_namespace_t *ns = arg;

	return (nvme_bd_cmd(ns, xfer, NVME_OPC_NVM_READ));
}

static int
nvme_bd_write(void *arg, bd_xfer_t *xfer)
{
	nvme_namespace_t *ns = arg;

	return (nvme_bd_cmd(ns, xfer, NVME_OPC_NVM_WRITE));
}

static int
nvme_bd_sync(void *arg, bd_xfer_t *xfer)
{
	nvme_namespace_t *ns = arg;

	if (ns->ns_nvme->n_dead)
		return (EIO);

	/*
	 * If the volatile write cache is not present or not enabled the FLUSH
	 * command is a no-op, so we can take a shortcut here.
	 */
	if (!ns->ns_nvme->n_write_cache_present) {
		bd_xfer_done(xfer, ENOTSUP);
		return (0);
	}

	if (!ns->ns_nvme->n_write_cache_enabled) {
		bd_xfer_done(xfer, 0);
		return (0);
	}

	return (nvme_bd_cmd(ns, xfer, NVME_OPC_NVM_FLUSH));
}

static int
nvme_bd_devid(void *arg, dev_info_t *devinfo, ddi_devid_t *devid)
{
	nvme_namespace_t *ns = arg;
	nvme_t *nvme = ns->ns_nvme;

	if (nvme->n_dead) {
		return (EIO);
	}

	if (*(uint64_t *)ns->ns_nguid != 0 ||
	    *(uint64_t *)(ns->ns_nguid + 8) != 0) {
		return (ddi_devid_init(devinfo, DEVID_NVME_NGUID,
		    sizeof (ns->ns_nguid), ns->ns_nguid, devid));
	} else if (*(uint64_t *)ns->ns_eui64 != 0) {
		return (ddi_devid_init(devinfo, DEVID_NVME_EUI64,
		    sizeof (ns->ns_eui64), ns->ns_eui64, devid));
	} else {
		return (ddi_devid_init(devinfo, DEVID_NVME_NSID,
		    strlen(ns->ns_devid), ns->ns_devid, devid));
	}
}

static int
nvme_bd_free_space(void *arg, bd_xfer_t *xfer)
{
	nvme_namespace_t *ns = arg;

	if (xfer->x_dfl == NULL)
		return (EINVAL);

	if (!ns->ns_nvme->n_idctl->id_oncs.on_dset_mgmt)
		return (ENOTSUP);

	return (nvme_bd_cmd(ns, xfer, NVME_OPC_NVM_DSET_MGMT));
}

static int
nvme_open(dev_t *devp, int flag, int otyp, cred_t *cred_p)
{
#ifndef __lock_lint
	_NOTE(ARGUNUSED(cred_p));
#endif
	minor_t minor = getminor(*devp);
	nvme_t *nvme = ddi_get_soft_state(nvme_state, NVME_MINOR_INST(minor));
	int nsid = NVME_MINOR_NSID(minor);
	nvme_minor_state_t *nm;
	int rv = 0;

	if (otyp != OTYP_CHR)
		return (EINVAL);

	if (nvme == NULL)
		return (ENXIO);

	if (nsid > nvme->n_namespace_count)
		return (ENXIO);

	if (nvme->n_dead)
		return (EIO);

	mutex_enter(&nvme->n_minor_mutex);

	/*
	 * First check the devctl node and error out if it's been opened
	 * exclusively already by any other thread.
	 */
	if (nvme->n_minor.nm_oexcl != NULL &&
	    nvme->n_minor.nm_oexcl != curthread) {
		rv = EBUSY;
		goto out;
	}

	nm = nsid == 0 ? &nvme->n_minor : &(NVME_NSID2NS(nvme, nsid)->ns_minor);

	if (flag & FEXCL) {
		if (nm->nm_oexcl != NULL || nm->nm_open) {
			rv = EBUSY;
			goto out;
		}

		/*
		 * If at least one namespace is already open, fail the
		 * exclusive open of the devctl node.
		 */
		if (nsid == 0) {
			for (int i = 1; i <= nvme->n_namespace_count; i++) {
				if (NVME_NSID2NS(nvme, i)->ns_minor.nm_open) {
					rv = EBUSY;
					goto out;
				}
			}
		}

		nm->nm_oexcl = curthread;
	}

	nm->nm_open = B_TRUE;

out:
	mutex_exit(&nvme->n_minor_mutex);
	return (rv);

}

static int
nvme_close(dev_t dev, int flag, int otyp, cred_t *cred_p)
{
#ifndef __lock_lint
	_NOTE(ARGUNUSED(cred_p));
	_NOTE(ARGUNUSED(flag));
#endif
	minor_t minor = getminor(dev);
	nvme_t *nvme = ddi_get_soft_state(nvme_state, NVME_MINOR_INST(minor));
	int nsid = NVME_MINOR_NSID(minor);
	nvme_minor_state_t *nm;

	if (otyp != OTYP_CHR)
		return (ENXIO);

	if (nvme == NULL)
		return (ENXIO);

	if (nsid > nvme->n_namespace_count)
		return (ENXIO);

	nm = nsid == 0 ? &nvme->n_minor : &(NVME_NSID2NS(nvme, nsid)->ns_minor);

	mutex_enter(&nvme->n_minor_mutex);
	if (nm->nm_oexcl != NULL) {
		ASSERT(nm->nm_oexcl == curthread);
		nm->nm_oexcl = NULL;
	}

	ASSERT(nm->nm_open);
	nm->nm_open = B_FALSE;
	mutex_exit(&nvme->n_minor_mutex);

	return (0);
}

static int
nvme_ioctl_identify(nvme_t *nvme, int nsid, nvme_ioctl_t *nioc, int mode,
    cred_t *cred_p)
{
	_NOTE(ARGUNUSED(cred_p));
	int rv = 0;
	void *idctl;

	if ((mode & FREAD) == 0)
		return (EPERM);

	if (nioc->n_len < NVME_IDENTIFY_BUFSIZE)
		return (EINVAL);

	switch (nioc->n_arg) {
	case NVME_IDENTIFY_NSID:
		/*
		 * If we support namespace management, set the nsid to -1 to
		 * retrieve the common namespace capabilities. Otherwise
		 * have a best guess by returning identify data for namespace 1.
		 */
		if (nsid == 0)
			nsid = nvme->n_idctl->id_oacs.oa_nsmgmt == 1 ? -1 : 1;
		break;

	case NVME_IDENTIFY_CTRL:
		/*
		 * Let NVME_IDENTIFY_CTRL work the same on devctl and attachment
		 * point nodes.
		 */
		nsid = 0;
		break;

	case NVME_IDENTIFY_NSID_LIST:
		if (!NVME_VERSION_ATLEAST(&nvme->n_version, 1, 1))
			return (ENOTSUP);

		/*
		 * For now, always try to get the list of active NSIDs starting
		 * at the first namespace. This will have to be revisited should
		 * the need arise to support more than 1024 namespaces.
		 */
		nsid = 0;
		break;

	case NVME_IDENTIFY_NSID_DESC:
		if (!NVME_VERSION_ATLEAST(&nvme->n_version, 1, 3))
			return (ENOTSUP);
		break;

	case NVME_IDENTIFY_NSID_ALLOC:
		if (!NVME_VERSION_ATLEAST(&nvme->n_version, 1, 2) ||
		    (nvme->n_idctl->id_oacs.oa_nsmgmt == 0))
			return (ENOTSUP);

		/*
		 * To make this work on a devctl node, make this return the
		 * identify data for namespace 1. We assume that any NVMe
		 * device supports at least one namespace, which has ID 1.
		 */
		if (nsid == 0)
			nsid = 1;
		break;

	case NVME_IDENTIFY_NSID_ALLOC_LIST:
		if (!NVME_VERSION_ATLEAST(&nvme->n_version, 1, 2) ||
		    (nvme->n_idctl->id_oacs.oa_nsmgmt == 0))
			return (ENOTSUP);

		/*
		 * For now, always try to get the list of allocated NSIDs
		 * starting at the first namespace. This will have to be
		 * revisited should the need arise to support more than 1024
		 * namespaces.
		 */
		nsid = 0;
		break;

	case NVME_IDENTIFY_NSID_CTRL_LIST:
		if (!NVME_VERSION_ATLEAST(&nvme->n_version, 1, 2) ||
		    (nvme->n_idctl->id_oacs.oa_nsmgmt == 0))
			return (ENOTSUP);

		if (nsid == 0)
			return (EINVAL);
		break;

	case NVME_IDENTIFY_CTRL_LIST:
		if (!NVME_VERSION_ATLEAST(&nvme->n_version, 1, 2) ||
		    (nvme->n_idctl->id_oacs.oa_nsmgmt == 0))
			return (ENOTSUP);

		if (nsid != 0)
			return (EINVAL);
		break;

	default:
		return (EINVAL);
	}

	if ((rv = nvme_identify(nvme, B_TRUE, nsid, nioc->n_arg & 0xff,
	    (void **)&idctl)) != 0)
		return (rv);

	if (ddi_copyout(idctl, (void *)nioc->n_buf, NVME_IDENTIFY_BUFSIZE, mode)
	    != 0)
		rv = EFAULT;

	kmem_free(idctl, NVME_IDENTIFY_BUFSIZE);

	return (rv);
}

/*
 * Execute commands on behalf of the various ioctls.
 */
static int
nvme_ioc_cmd(nvme_t *nvme, nvme_sqe_t *sqe, boolean_t is_admin, void *data_addr,
    uint32_t data_len, int rwk, nvme_cqe_t *cqe, uint_t timeout)
{
	nvme_cmd_t *cmd;
	nvme_qpair_t *ioq;
	int rv = 0;

	cmd = nvme_alloc_cmd(nvme, KM_SLEEP);
	if (is_admin) {
		cmd->nc_sqid = 0;
		ioq = nvme->n_adminq;
	} else {
		cmd->nc_sqid = (CPU->cpu_id % nvme->n_ioq_count) + 1;
		ASSERT(cmd->nc_sqid <= nvme->n_ioq_count);
		ioq = nvme->n_ioq[cmd->nc_sqid];
	}

	/*
	 * This function is used to facilitate requests from
	 * userspace, so don't panic if the command fails. This
	 * is especially true for admin passthru commands, where
	 * the actual command data structure is entirely defined
	 * by userspace.
	 */
	cmd->nc_dontpanic = B_TRUE;

	cmd->nc_callback = nvme_wakeup_cmd;
	cmd->nc_sqe = *sqe;

	if ((rwk & (FREAD | FWRITE)) != 0) {
		if (data_addr == NULL) {
			rv = EINVAL;
			goto free_cmd;
		}

		if (nvme_zalloc_dma(nvme, data_len, DDI_DMA_READ,
		    &nvme->n_prp_dma_attr, &cmd->nc_dma) != DDI_SUCCESS) {
			dev_err(nvme->n_dip, CE_WARN,
			    "!nvme_zalloc_dma failed for nvme_ioc_cmd()");

			rv = ENOMEM;
			goto free_cmd;
		}

		if ((rv = nvme_fill_prp(cmd, cmd->nc_dma->nd_dmah)) != 0)
			goto free_cmd;

		if ((rwk & FWRITE) != 0) {
			if (ddi_copyin(data_addr, cmd->nc_dma->nd_memp,
			    data_len, rwk & FKIOCTL) != 0) {
				rv = EFAULT;
				goto free_cmd;
			}
		}
	}

	if (is_admin) {
		nvme_admin_cmd(cmd, timeout);
	} else {
		mutex_enter(&cmd->nc_mutex);

		rv = nvme_submit_io_cmd(ioq, cmd);

		if (rv == EAGAIN) {
			mutex_exit(&cmd->nc_mutex);
			dev_err(cmd->nc_nvme->n_dip, CE_WARN,
			    "!nvme_ioc_cmd() failed, I/O Q full");
			goto free_cmd;
		}

		nvme_wait_cmd(cmd, timeout);

		mutex_exit(&cmd->nc_mutex);
	}

	if (cqe != NULL)
		*cqe = cmd->nc_cqe;

	if ((rv = nvme_check_cmd_status(cmd)) != 0) {
		dev_err(nvme->n_dip, CE_WARN,
		    "!nvme_ioc_cmd() failed with sct = %x, sc = %x",
		    cmd->nc_cqe.cqe_sf.sf_sct, cmd->nc_cqe.cqe_sf.sf_sc);

		goto free_cmd;
	}

	if ((rwk & FREAD) != 0) {
		if (ddi_copyout(cmd->nc_dma->nd_memp,
		    data_addr, data_len, rwk & FKIOCTL) != 0)
			rv = EFAULT;
	}

free_cmd:
	nvme_free_cmd(cmd);

	return (rv);
}

static int
nvme_ioctl_capabilities(nvme_t *nvme, int nsid, nvme_ioctl_t *nioc,
    int mode, cred_t *cred_p)
{
	_NOTE(ARGUNUSED(nsid, cred_p));
	int rv = 0;
	nvme_reg_cap_t cap = { 0 };
	nvme_capabilities_t nc;

	if ((mode & FREAD) == 0)
		return (EPERM);

	if (nioc->n_len < sizeof (nc))
		return (EINVAL);

	cap.r = nvme_get64(nvme, NVME_REG_CAP);

	/*
	 * The MPSMIN and MPSMAX fields in the CAP register use 0 to
	 * specify the base page size of 4k (1<<12), so add 12 here to
	 * get the real page size value.
	 */
	nc.mpsmax = 1 << (12 + cap.b.cap_mpsmax);
	nc.mpsmin = 1 << (12 + cap.b.cap_mpsmin);

	if (ddi_copyout(&nc, (void *)nioc->n_buf, sizeof (nc), mode) != 0)
		rv = EFAULT;

	return (rv);
}

static int
nvme_ioctl_get_logpage(nvme_t *nvme, int nsid, nvme_ioctl_t *nioc,
    int mode, cred_t *cred_p)
{
	_NOTE(ARGUNUSED(cred_p));
	void *log = NULL;
	size_t bufsize = 0;
	int rv = 0;

	if ((mode & FREAD) == 0)
		return (EPERM);

	if (nsid > 0 && !NVME_NSID2NS(nvme, nsid)->ns_active)
		return (EINVAL);

	switch (nioc->n_arg) {
	case NVME_LOGPAGE_ERROR:
		if (nsid != 0)
			return (EINVAL);
		break;
	case NVME_LOGPAGE_HEALTH:
		if (nsid != 0 && nvme->n_idctl->id_lpa.lp_smart == 0)
			return (EINVAL);

		if (nsid == 0)
			nsid = (uint32_t)-1;

		break;
	case NVME_LOGPAGE_FWSLOT:
		if (nsid != 0)
			return (EINVAL);
		break;
	default:
		if (!NVME_IS_VENDOR_SPECIFIC_LOGPAGE(nioc->n_arg))
			return (EINVAL);
		if (nioc->n_len > NVME_VENDOR_SPECIFIC_LOGPAGE_MAX_SIZE) {
			dev_err(nvme->n_dip, CE_NOTE, "!Vendor-specific log "
			    "page size exceeds device maximum supported size: "
			    "%lu", NVME_VENDOR_SPECIFIC_LOGPAGE_MAX_SIZE);
			return (EINVAL);
		}
		if (nioc->n_len == 0)
			return (EINVAL);
		bufsize = nioc->n_len;
		if (nsid == 0)
			nsid = (uint32_t)-1;
	}

	if (nvme_get_logpage(nvme, B_TRUE, &log, &bufsize, nioc->n_arg, nsid)
	    != DDI_SUCCESS)
		return (EIO);

	if (nioc->n_len < bufsize) {
		kmem_free(log, bufsize);
		return (EINVAL);
	}

	if (ddi_copyout(log, (void *)nioc->n_buf, bufsize, mode) != 0)
		rv = EFAULT;

	nioc->n_len = bufsize;
	kmem_free(log, bufsize);

	return (rv);
}

static int
nvme_ioctl_get_features(nvme_t *nvme, int nsid, nvme_ioctl_t *nioc,
    int mode, cred_t *cred_p)
{
	_NOTE(ARGUNUSED(cred_p));
	void *buf = NULL;
	size_t bufsize = 0;
	uint32_t res = 0;
	uint8_t feature;
	int rv = 0;

	if ((mode & FREAD) == 0)
		return (EPERM);

	if (nsid > 0 && !NVME_NSID2NS(nvme, nsid)->ns_active)
		return (EINVAL);

	if ((nioc->n_arg >> 32) > 0xff)
		return (EINVAL);

	feature = (uint8_t)(nioc->n_arg >> 32);

	switch (feature) {
	case NVME_FEAT_ARBITRATION:
	case NVME_FEAT_POWER_MGMT:
	case NVME_FEAT_ERROR:
	case NVME_FEAT_NQUEUES:
	case NVME_FEAT_INTR_COAL:
	case NVME_FEAT_WRITE_ATOM:
	case NVME_FEAT_ASYNC_EVENT:
	case NVME_FEAT_PROGRESS:
		if (nsid != 0)
			return (EINVAL);
		break;

	case NVME_FEAT_TEMPERATURE:
		if (nsid != 0)
			return (EINVAL);
		res = nioc->n_arg & 0xffffffffUL;
		if (NVME_VERSION_ATLEAST(&nvme->n_version, 1, 2)) {
			nvme_temp_threshold_t tt;

			tt.r = res;
			if (tt.b.tt_thsel != NVME_TEMP_THRESH_OVER &&
			    tt.b.tt_thsel != NVME_TEMP_THRESH_UNDER) {
				return (EINVAL);
			}

			if (tt.b.tt_tmpsel > NVME_TEMP_THRESH_MAX_SENSOR) {
				return (EINVAL);
			}
		} else if (res != 0) {
			return (ENOTSUP);
		}
		break;

	case NVME_FEAT_INTR_VECT:
		if (nsid != 0)
			return (EINVAL);

		res = nioc->n_arg & 0xffffffffUL;
		if (res >= nvme->n_intr_cnt)
			return (EINVAL);
		break;

	case NVME_FEAT_LBA_RANGE:
		if (nvme->n_lba_range_supported == B_FALSE)
			return (EINVAL);

		if (nsid == 0 ||
		    nsid > nvme->n_namespace_count)
			return (EINVAL);

		break;

	case NVME_FEAT_WRITE_CACHE:
		if (nsid != 0)
			return (EINVAL);

		if (!nvme->n_write_cache_present)
			return (EINVAL);

		break;

	case NVME_FEAT_AUTO_PST:
		if (nsid != 0)
			return (EINVAL);

		if (!nvme->n_auto_pst_supported)
			return (EINVAL);

		break;

	default:
		return (EINVAL);
	}

	rv = nvme_get_features(nvme, B_TRUE, nsid, feature, &res, &buf,
	    &bufsize);
	if (rv != 0)
		return (rv);

	if (nioc->n_len < bufsize) {
		kmem_free(buf, bufsize);
		return (EINVAL);
	}

	if (buf && ddi_copyout(buf, (void*)nioc->n_buf, bufsize, mode) != 0)
		rv = EFAULT;

	kmem_free(buf, bufsize);
	nioc->n_arg = res;
	nioc->n_len = bufsize;

	return (rv);
}

static int
nvme_ioctl_intr_cnt(nvme_t *nvme, int nsid, nvme_ioctl_t *nioc, int mode,
    cred_t *cred_p)
{
	_NOTE(ARGUNUSED(nsid, mode, cred_p));

	if ((mode & FREAD) == 0)
		return (EPERM);

	nioc->n_arg = nvme->n_intr_cnt;
	return (0);
}

static int
nvme_ioctl_version(nvme_t *nvme, int nsid, nvme_ioctl_t *nioc, int mode,
    cred_t *cred_p)
{
	_NOTE(ARGUNUSED(nsid, cred_p));
	int rv = 0;

	if ((mode & FREAD) == 0)
		return (EPERM);

	if (nioc->n_len < sizeof (nvme->n_version))
		return (ENOMEM);

	if (ddi_copyout(&nvme->n_version, (void *)nioc->n_buf,
	    sizeof (nvme->n_version), mode) != 0)
		rv = EFAULT;

	return (rv);
}

static int
nvme_ioctl_format(nvme_t *nvme, int nsid, nvme_ioctl_t *nioc, int mode,
    cred_t *cred_p)
{
	_NOTE(ARGUNUSED(mode));
	nvme_format_nvm_t frmt = { 0 };
	int c_nsid = nsid != 0 ? nsid : 1;
	nvme_identify_nsid_t *idns;
	nvme_minor_state_t *nm;

	if ((mode & FWRITE) == 0 || secpolicy_sys_config(cred_p, B_FALSE) != 0)
		return (EPERM);

	nm = nsid == 0 ? &nvme->n_minor : &(NVME_NSID2NS(nvme, nsid)->ns_minor);
	if (nm->nm_oexcl != curthread)
		return (EACCES);

	if (nsid != 0) {
		if (NVME_NSID2NS(nvme, nsid)->ns_attached)
			return (EBUSY);
		else if (!NVME_NSID2NS(nvme, nsid)->ns_active)
			return (EINVAL);
	}

	frmt.r = nioc->n_arg & 0xffffffff;

	/*
	 * Check whether the FORMAT NVM command is supported.
	 */
	if (nvme->n_idctl->id_oacs.oa_format == 0)
		return (ENOTSUP);

	/*
	 * Don't allow format or secure erase of individual namespace if that
	 * would cause a format or secure erase of all namespaces.
	 */
	if (nsid != 0 && nvme->n_idctl->id_fna.fn_format != 0)
		return (EINVAL);

	if (nsid != 0 && frmt.b.fm_ses != NVME_FRMT_SES_NONE &&
	    nvme->n_idctl->id_fna.fn_sec_erase != 0)
		return (EINVAL);

	/*
	 * Don't allow formatting with Protection Information.
	 */
	if (frmt.b.fm_pi != 0 || frmt.b.fm_pil != 0 || frmt.b.fm_ms != 0)
		return (EINVAL);

	/*
	 * Don't allow formatting using an illegal LBA format, or any LBA format
	 * that uses metadata.
	 */
	idns = NVME_NSID2NS(nvme, c_nsid)->ns_idns;
	if (frmt.b.fm_lbaf > idns->id_nlbaf ||
	    idns->id_lbaf[frmt.b.fm_lbaf].lbaf_ms != 0)
		return (EINVAL);

	/*
	 * Don't allow formatting using an illegal Secure Erase setting.
	 */
	if (frmt.b.fm_ses > NVME_FRMT_MAX_SES ||
	    (frmt.b.fm_ses == NVME_FRMT_SES_CRYPTO &&
	    nvme->n_idctl->id_fna.fn_crypt_erase == 0))
		return (EINVAL);

	if (nsid == 0)
		nsid = (uint32_t)-1;

	return (nvme_format_nvm(nvme, B_TRUE, nsid, frmt.b.fm_lbaf, B_FALSE, 0,
	    B_FALSE, frmt.b.fm_ses));
}

static int
nvme_ioctl_detach(nvme_t *nvme, int nsid, nvme_ioctl_t *nioc, int mode,
    cred_t *cred_p)
{
	_NOTE(ARGUNUSED(nioc, mode));
	int rv;

	if ((mode & FWRITE) == 0 || secpolicy_sys_config(cred_p, B_FALSE) != 0)
		return (EPERM);

	if (nsid == 0)
		return (EINVAL);

	if (NVME_NSID2NS(nvme, nsid)->ns_minor.nm_oexcl != curthread)
		return (EACCES);

	mutex_enter(&nvme->n_mgmt_mutex);

	rv = nvme_detach_ns(nvme, nsid);

	mutex_exit(&nvme->n_mgmt_mutex);

	return (rv);
}

static int
nvme_ioctl_attach(nvme_t *nvme, int nsid, nvme_ioctl_t *nioc, int mode,
    cred_t *cred_p)
{
	_NOTE(ARGUNUSED(nioc, mode));
	int rv;

	if ((mode & FWRITE) == 0 || secpolicy_sys_config(cred_p, B_FALSE) != 0)
		return (EPERM);

	if (nsid == 0)
		return (EINVAL);

	if (NVME_NSID2NS(nvme, nsid)->ns_minor.nm_oexcl != curthread)
		return (EACCES);

	mutex_enter(&nvme->n_mgmt_mutex);

	if (nvme_init_ns(nvme, nsid) != DDI_SUCCESS) {
		mutex_exit(&nvme->n_mgmt_mutex);
		return (EIO);
	}

	rv = nvme_attach_ns(nvme, nsid);

	mutex_exit(&nvme->n_mgmt_mutex);
	return (rv);
}

static void
nvme_ufm_update(nvme_t *nvme)
{
	mutex_enter(&nvme->n_fwslot_mutex);
	ddi_ufm_update(nvme->n_ufmh);
	if (nvme->n_fwslot != NULL) {
		kmem_free(nvme->n_fwslot, sizeof (nvme_fwslot_log_t));
		nvme->n_fwslot = NULL;
	}
	mutex_exit(&nvme->n_fwslot_mutex);
}

/*
 * Download new firmware to the device's internal staging area. We do not call
 * nvme_ufm_update() here because after a firmware download, there has been no
 * change to any of the actual persistent firmware data. That requires a
 * subsequent ioctl (NVME_IOC_FIRMWARE_COMMIT) to commit the firmware to a slot
 * or to activate a slot.
 */
static int
nvme_ioctl_firmware_download(nvme_t *nvme, int nsid, nvme_ioctl_t *nioc,
    int mode, cred_t *cred_p)
{
	int rv = 0;
	size_t len, copylen;
	offset_t offset;
	uintptr_t buf;
	nvme_cqe_t cqe = { 0 };
	nvme_sqe_t sqe = {
	    .sqe_opc	= NVME_OPC_FW_IMAGE_LOAD
	};

	if ((mode & FWRITE) == 0 || secpolicy_sys_config(cred_p, B_FALSE) != 0)
		return (EPERM);

	if (nvme->n_idctl->id_oacs.oa_firmware == 0)
		return (ENOTSUP);

	if (nsid != 0)
		return (EINVAL);

	/*
	 * The offset (in n_len) is restricted to the number of DWORDs in
	 * 32 bits.
	 */
	if (nioc->n_len > NVME_FW_OFFSETB_MAX)
		return (EINVAL);

	/* Confirm that both offset and length are a multiple of DWORD bytes */
	if ((nioc->n_len & NVME_DWORD_MASK) != 0 ||
	    (nioc->n_arg & NVME_DWORD_MASK) != 0)
		return (EINVAL);

	len = nioc->n_len;
	offset = nioc->n_arg;
	buf = (uintptr_t)nioc->n_buf;

	nioc->n_arg = 0;

	while (len > 0 && rv == 0) {
		/*
		 * nvme_ioc_cmd() does not use SGLs or PRP lists.
		 * It is limited to 2 PRPs per NVM command, so limit
		 * the size of the data to 2 pages.
		 */
		copylen = MIN(2 * nvme->n_pagesize, len);

		sqe.sqe_cdw10 = (uint32_t)(copylen >> NVME_DWORD_SHIFT) - 1;
		sqe.sqe_cdw11 = (uint32_t)(offset >> NVME_DWORD_SHIFT);

		rv = nvme_ioc_cmd(nvme, &sqe, B_TRUE, (void *)buf, copylen,
		    FWRITE, &cqe, nvme_admin_cmd_timeout);

		/*
		 * Regardless of whether the command succeeded or not, whether
		 * there's an errno in rv to be returned, we'll return any
		 * command-specific status code in n_arg.
		 *
		 * As n_arg isn't cleared in all other possible code paths
		 * returning an error, we return the status code as a negative
		 * value so it can be distinguished easily from whatever value
		 * was passed in n_arg originally. This of course only works as
		 * long as arguments passed in n_arg are less than INT64_MAX,
		 * which they currently are.
		 */
		if (cqe.cqe_sf.sf_sct == NVME_CQE_SCT_SPECIFIC)
			nioc->n_arg = (uint64_t)-cqe.cqe_sf.sf_sc;

		buf += copylen;
		offset += copylen;
		len -= copylen;
	}

	return (rv);
}

static int
nvme_ioctl_firmware_commit(nvme_t *nvme, int nsid, nvme_ioctl_t *nioc,
    int mode, cred_t *cred_p)
{
	nvme_firmware_commit_dw10_t fc_dw10 = { 0 };
	uint32_t slot = nioc->n_arg & 0xffffffff;
	uint32_t action = nioc->n_arg >> 32;
	nvme_cqe_t cqe = { 0 };
	nvme_sqe_t sqe = {
	    .sqe_opc	= NVME_OPC_FW_ACTIVATE
	};
	int timeout;
	int rv;

	if ((mode & FWRITE) == 0 || secpolicy_sys_config(cred_p, B_FALSE) != 0)
		return (EPERM);

	if (nvme->n_idctl->id_oacs.oa_firmware == 0)
		return (ENOTSUP);

	if (nsid != 0)
		return (EINVAL);

	/* Validate slot is in range. */
	if (slot < NVME_FW_SLOT_MIN || slot > NVME_FW_SLOT_MAX)
		return (EINVAL);

	switch (action) {
	case NVME_FWC_SAVE:
	case NVME_FWC_SAVE_ACTIVATE:
		if (slot == 1 && nvme->n_idctl->id_frmw.fw_readonly)
			return (EROFS);
		break;
	case NVME_FWC_ACTIVATE:
	case NVME_FWC_ACTIVATE_IMMED:
		break;
	default:
		return (EINVAL);
	}

	/*
	 * Use the extended timeout for all firmware operations here as we've
	 * seen examples in the field where an activate may take longer than the
	 * 1s period that we've asked for.
	 */
	timeout = nvme_commit_save_cmd_timeout;

	fc_dw10.b.fc_slot = slot;
	fc_dw10.b.fc_action = action;
	sqe.sqe_cdw10 = fc_dw10.r;

	nioc->n_arg = 0;
	rv = nvme_ioc_cmd(nvme, &sqe, B_TRUE, NULL, 0, 0, &cqe, timeout);

	/*
	 * Regardless of whether the command succeeded or not, whether
	 * there's an errno in rv to be returned, we'll return any
	 * command-specific status code in n_arg.
	 *
	 * As n_arg isn't cleared in all other possible code paths
	 * returning an error, we return the status code as a negative
	 * value so it can be distinguished easily from whatever value
	 * was passed in n_arg originally. This of course only works as
	 * long as arguments passed in n_arg are less than INT64_MAX,
	 * which they currently are.
	 */
	if (cqe.cqe_sf.sf_sct == NVME_CQE_SCT_SPECIFIC)
		nioc->n_arg = (uint64_t)-cqe.cqe_sf.sf_sc;

	/*
	 * Let the DDI UFM subsystem know that the firmware information for
	 * this device has changed. We perform this unconditionally as an
	 * invalidation doesn't particularly hurt us.
	 */
	nvme_ufm_update(nvme);

	return (rv);
}

/*
 * Helper to copy in a passthru command from userspace, handling
 * different data models.
 */
static int
nvme_passthru_copy_cmd_in(const void *buf, nvme_passthru_cmd_t *cmd, int mode)
{
#ifdef _MULTI_DATAMODEL
	switch (ddi_model_convert_from(mode & FMODELS)) {
	case DDI_MODEL_ILP32: {
		nvme_passthru_cmd32_t cmd32;
		if (ddi_copyin(buf, (void*)&cmd32, sizeof (cmd32), mode) != 0)
			return (-1);
		cmd->npc_opcode = cmd32.npc_opcode;
		cmd->npc_timeout = cmd32.npc_timeout;
		cmd->npc_flags = cmd32.npc_flags;
		cmd->npc_cdw12 = cmd32.npc_cdw12;
		cmd->npc_cdw13 = cmd32.npc_cdw13;
		cmd->npc_cdw14 = cmd32.npc_cdw14;
		cmd->npc_cdw15 = cmd32.npc_cdw15;
		cmd->npc_buflen = cmd32.npc_buflen;
		cmd->npc_buf = cmd32.npc_buf;
		break;
	}
	case DDI_MODEL_NONE:
#endif
	if (ddi_copyin(buf, (void*)cmd, sizeof (nvme_passthru_cmd_t),
	    mode) != 0)
		return (-1);
#ifdef _MULTI_DATAMODEL
		break;
	}
#endif
	return (0);
}

/*
 * Helper to copy out a passthru command result to userspace, handling
 * different data models.
 */
static int
nvme_passthru_copy_cmd_out(const nvme_passthru_cmd_t *cmd, void *buf, int mode)
{
#ifdef _MULTI_DATAMODEL
	switch (ddi_model_convert_from(mode & FMODELS)) {
	case DDI_MODEL_ILP32: {
		nvme_passthru_cmd32_t cmd32;
		bzero(&cmd32, sizeof (cmd32));
		cmd32.npc_opcode = cmd->npc_opcode;
		cmd32.npc_status = cmd->npc_status;
		cmd32.npc_err = cmd->npc_err;
		cmd32.npc_timeout = cmd->npc_timeout;
		cmd32.npc_flags = cmd->npc_flags;
		cmd32.npc_cdw0 = cmd->npc_cdw0;
		cmd32.npc_cdw12 = cmd->npc_cdw12;
		cmd32.npc_cdw13 = cmd->npc_cdw13;
		cmd32.npc_cdw14 = cmd->npc_cdw14;
		cmd32.npc_cdw15 = cmd->npc_cdw15;
		cmd32.npc_buflen = (size32_t)cmd->npc_buflen;
		cmd32.npc_buf = (uintptr32_t)cmd->npc_buf;
		if (ddi_copyout(&cmd32, buf, sizeof (cmd32), mode) != 0)
			return (-1);
		break;
	}
	case DDI_MODEL_NONE:
#endif
		if (ddi_copyout(cmd, buf, sizeof (nvme_passthru_cmd_t),
		    mode) != 0)
			return (-1);
#ifdef _MULTI_DATAMODEL
		break;
	}
#endif
	return (0);
}

/*
 * Run an arbitrary vendor-specific admin command on the device.
 */
static int
nvme_ioctl_passthru(nvme_t *nvme, int nsid, nvme_ioctl_t *nioc, int mode,
    cred_t *cred_p)
{
	int rv = 0;
	uint_t timeout = 0;
	int rwk = 0;
	nvme_passthru_cmd_t cmd;
	size_t expected_passthru_size = 0;
	nvme_sqe_t sqe;
	nvme_cqe_t cqe;

	bzero(&cmd, sizeof (cmd));
	bzero(&sqe, sizeof (sqe));
	bzero(&cqe, sizeof (cqe));

	/*
	 * Basic checks: permissions, data model, argument size.
	 */
	if ((mode & FWRITE) == 0 || secpolicy_sys_config(cred_p, B_FALSE) != 0)
		return (EPERM);

	/*
	 * Compute the expected size of the argument buffer
	 */
#ifdef _MULTI_DATAMODEL
	switch (ddi_model_convert_from(mode & FMODELS)) {
	case DDI_MODEL_ILP32:
		expected_passthru_size = sizeof (nvme_passthru_cmd32_t);
		break;
	case DDI_MODEL_NONE:
#endif
		expected_passthru_size = sizeof (nvme_passthru_cmd_t);
#ifdef _MULTI_DATAMODEL
		break;
	}
#endif

	if (nioc->n_len != expected_passthru_size) {
		cmd.npc_err = NVME_PASSTHRU_ERR_CMD_SIZE;
		rv = EINVAL;
		goto out;
	}

	/*
	 * Ensure the device supports the standard vendor specific
	 * admin command format.
	 */
	if (!nvme->n_idctl->id_nvscc.nv_spec) {
		cmd.npc_err = NVME_PASSTHRU_ERR_NOT_SUPPORTED;
		rv = ENOTSUP;
		goto out;
	}

	if (nvme_passthru_copy_cmd_in((const void*)nioc->n_buf, &cmd, mode))
		return (EFAULT);

	if (!NVME_IS_VENDOR_SPECIFIC_CMD(cmd.npc_opcode)) {
		cmd.npc_err = NVME_PASSTHRU_ERR_INVALID_OPCODE;
		rv = EINVAL;
		goto out;
	}

	/*
	 * This restriction is not mandated by the spec, so future work
	 * could relax this if it's necessary to support commands that both
	 * read and write.
	 */
	if ((cmd.npc_flags & NVME_PASSTHRU_READ) != 0 &&
	    (cmd.npc_flags & NVME_PASSTHRU_WRITE) != 0) {
		cmd.npc_err = NVME_PASSTHRU_ERR_READ_AND_WRITE;
		rv = EINVAL;
		goto out;
	}
	if (cmd.npc_timeout > nvme_vendor_specific_admin_cmd_max_timeout) {
		cmd.npc_err = NVME_PASSTHRU_ERR_INVALID_TIMEOUT;
		rv = EINVAL;
		goto out;
	}
	timeout = cmd.npc_timeout;

	/*
	 * Passed-thru command buffer verification:
	 *  - Size is multiple of DWords
	 *  - Non-null iff the length is non-zero
	 *  - Null if neither reading nor writing data.
	 *  - Non-null if reading or writing.
	 *  - Maximum buffer size.
	 */
	if ((cmd.npc_buflen % sizeof (uint32_t)) != 0) {
		cmd.npc_err = NVME_PASSTHRU_ERR_INVALID_BUFFER;
		rv = EINVAL;
		goto out;
	}
	if (((void*)cmd.npc_buf != NULL && cmd.npc_buflen == 0) ||
	    ((void*)cmd.npc_buf == NULL && cmd.npc_buflen != 0)) {
		cmd.npc_err = NVME_PASSTHRU_ERR_INVALID_BUFFER;
		rv = EINVAL;
		goto out;
	}
	if (cmd.npc_flags == 0 && (void*)cmd.npc_buf != NULL) {
		cmd.npc_err = NVME_PASSTHRU_ERR_INVALID_BUFFER;
		rv = EINVAL;
		goto out;
	}
	if ((cmd.npc_flags != 0) && ((void*)cmd.npc_buf == NULL)) {
		cmd.npc_err = NVME_PASSTHRU_ERR_INVALID_BUFFER;
		rv = EINVAL;
		goto out;
	}
	if (cmd.npc_buflen > nvme_vendor_specific_admin_cmd_size) {
		cmd.npc_err = NVME_PASSTHRU_ERR_INVALID_BUFFER;
		rv = EINVAL;
		goto out;
	}
	if ((cmd.npc_buflen >> NVME_DWORD_SHIFT) > UINT32_MAX) {
		cmd.npc_err = NVME_PASSTHRU_ERR_INVALID_BUFFER;
		rv = EINVAL;
		goto out;
	}

	sqe.sqe_opc = cmd.npc_opcode;
	sqe.sqe_nsid = nsid;
	sqe.sqe_cdw10 = (uint32_t)(cmd.npc_buflen >> NVME_DWORD_SHIFT);
	sqe.sqe_cdw12 = cmd.npc_cdw12;
	sqe.sqe_cdw13 = cmd.npc_cdw13;
	sqe.sqe_cdw14 = cmd.npc_cdw14;
	sqe.sqe_cdw15 = cmd.npc_cdw15;
	if ((cmd.npc_flags & NVME_PASSTHRU_READ) != 0)
		rwk = FREAD;
	else if ((cmd.npc_flags & NVME_PASSTHRU_WRITE) != 0)
		rwk = FWRITE;

	rv = nvme_ioc_cmd(nvme, &sqe, B_TRUE, (void*)cmd.npc_buf,
	    cmd.npc_buflen, rwk, &cqe, timeout);
	cmd.npc_status = cqe.cqe_sf.sf_sc;
	cmd.npc_cdw0 = cqe.cqe_dw0;

out:
	if (nvme_passthru_copy_cmd_out(&cmd, (void*)nioc->n_buf, mode))
		rv = EFAULT;
	return (rv);
}

static int
nvme_ioctl_ns_info(nvme_t *nvme, int nsid, nvme_ioctl_t *nioc, int mode,
    cred_t *cred_p)
{
	_NOTE(ARGUNUSED(cred_p));
	nvme_namespace_t *ns;
	nvme_ns_info_t *info;
	int ret;

	if ((mode & FREAD) == 0)
		return (EPERM);

	if (nioc->n_len < sizeof (nvme_ns_info_t))
		return (EINVAL);

	/*
	 * If we have the controller open (as indicated by nsid set to zero)
	 * then we will allow the caller to specify a namespace id in n_arg.
	 */
	if (nsid == 0) {
		if (nioc->n_arg == 0 || nioc->n_arg > nvme->n_namespace_count)
			return (EINVAL);
		nsid = (int)nioc->n_arg;
	} else if (nioc->n_arg != 0) {
		return (EINVAL);
	}

	ASSERT3S(nsid, >, 0);
	ns = NVME_NSID2NS(nvme, nsid);

	info = kmem_zalloc(sizeof (nvme_ns_info_t), KM_NOSLEEP_LAZY);
	if (info == NULL)
		return (ENOMEM);

	mutex_enter(&nvme->n_mgmt_mutex);

	if (ns->ns_allocated)
		info->nni_state |= NVME_NS_STATE_ALLOCATED;

	if (ns->ns_active)
		info->nni_state |= NVME_NS_STATE_ACTIVE;

	if (ns->ns_ignore)
		info->nni_state |= NVME_NS_STATE_IGNORED;

	if (ns->ns_attached) {
		const char *addr;

		info->nni_state |= NVME_NS_STATE_ATTACHED;
		addr = bd_address(ns->ns_bd_hdl);
		if (strlcpy(info->nni_addr, addr, sizeof (info->nni_addr)) >=
		    sizeof (info->nni_addr)) {
			mutex_exit(&nvme->n_mgmt_mutex);
			ret = EOVERFLOW;
			goto done;
		}
	}

	bcopy(ns->ns_idns, &info->nni_id, sizeof (nvme_identify_nsid_t));
	mutex_exit(&nvme->n_mgmt_mutex);

	if (ddi_copyout(info, (void *)nioc->n_buf, sizeof (nvme_ns_info_t),
	    mode & FKIOCTL) != 0) {
		ret = EFAULT;
	} else {
		ret = 0;
	}

done:
	kmem_free(info, sizeof (nvme_ns_info_t));
	return (ret);
}

static int
nvme_ioctl(dev_t dev, int cmd, intptr_t arg, int mode, cred_t *cred_p,
    int *rval_p)
{
#ifndef __lock_lint
	_NOTE(ARGUNUSED(rval_p));
#endif
	minor_t minor = getminor(dev);
	nvme_t *nvme = ddi_get_soft_state(nvme_state, NVME_MINOR_INST(minor));
	int nsid = NVME_MINOR_NSID(minor);
	int rv = 0;
	nvme_ioctl_t nioc;

	int (*nvme_ioctl[])(nvme_t *, int, nvme_ioctl_t *, int, cred_t *) = {
		NULL,
		nvme_ioctl_identify,
		NULL,
		nvme_ioctl_capabilities,
		nvme_ioctl_get_logpage,
		nvme_ioctl_get_features,
		nvme_ioctl_intr_cnt,
		nvme_ioctl_version,
		nvme_ioctl_format,
		nvme_ioctl_detach,
		nvme_ioctl_attach,
		nvme_ioctl_firmware_download,
		nvme_ioctl_firmware_commit,
		nvme_ioctl_passthru,
		nvme_ioctl_ns_info
	};

	if (nvme == NULL)
		return (ENXIO);

	if (nsid > nvme->n_namespace_count)
		return (ENXIO);

	if (IS_DEVCTL(cmd))
		return (ndi_devctl_ioctl(nvme->n_dip, cmd, arg, mode, 0));

#ifdef _MULTI_DATAMODEL
	switch (ddi_model_convert_from(mode & FMODELS)) {
	case DDI_MODEL_ILP32: {
		nvme_ioctl32_t nioc32;
		if (ddi_copyin((void*)arg, &nioc32, sizeof (nvme_ioctl32_t),
		    mode) != 0)
			return (EFAULT);
		nioc.n_len = nioc32.n_len;
		nioc.n_buf = nioc32.n_buf;
		nioc.n_arg = nioc32.n_arg;
		break;
	}
	case DDI_MODEL_NONE:
#endif
		if (ddi_copyin((void*)arg, &nioc, sizeof (nvme_ioctl_t), mode)
		    != 0)
			return (EFAULT);
#ifdef _MULTI_DATAMODEL
		break;
	}
#endif

	if (nvme->n_dead && cmd != NVME_IOC_DETACH)
		return (EIO);

	if (IS_NVME_IOC(cmd) && nvme_ioctl[NVME_IOC_CMD(cmd)] != NULL)
		rv = nvme_ioctl[NVME_IOC_CMD(cmd)](nvme, nsid, &nioc, mode,
		    cred_p);
	else
		rv = EINVAL;

#ifdef _MULTI_DATAMODEL
	switch (ddi_model_convert_from(mode & FMODELS)) {
	case DDI_MODEL_ILP32: {
		nvme_ioctl32_t nioc32;

		nioc32.n_len = (size32_t)nioc.n_len;
		nioc32.n_buf = (uintptr32_t)nioc.n_buf;
		nioc32.n_arg = nioc.n_arg;

		if (ddi_copyout(&nioc32, (void *)arg, sizeof (nvme_ioctl32_t),
		    mode) != 0)
			return (EFAULT);
		break;
	}
	case DDI_MODEL_NONE:
#endif
		if (ddi_copyout(&nioc, (void *)arg, sizeof (nvme_ioctl_t), mode)
		    != 0)
			return (EFAULT);
#ifdef _MULTI_DATAMODEL
		break;
	}
#endif

	return (rv);
}

/*
 * DDI UFM Callbacks
 */
static int
nvme_ufm_fill_image(ddi_ufm_handle_t *ufmh, void *arg, uint_t imgno,
    ddi_ufm_image_t *img)
{
	nvme_t *nvme = arg;

	if (imgno != 0)
		return (EINVAL);

	ddi_ufm_image_set_desc(img, "Firmware");
	ddi_ufm_image_set_nslots(img, nvme->n_idctl->id_frmw.fw_nslot);

	return (0);
}

/*
 * Fill out firmware slot information for the requested slot.  The firmware
 * slot information is gathered by requesting the Firmware Slot Information log
 * page.  The format of the page is described in section 5.10.1.3.
 *
 * We lazily cache the log page on the first call and then invalidate the cache
 * data after a successful firmware download or firmware commit command.
 * The cached data is protected by a mutex as the state can change
 * asynchronous to this callback.
 */
static int
nvme_ufm_fill_slot(ddi_ufm_handle_t *ufmh, void *arg, uint_t imgno,
    uint_t slotno, ddi_ufm_slot_t *slot)
{
	nvme_t *nvme = arg;
	void *log = NULL;
	size_t bufsize;
	ddi_ufm_attr_t attr = 0;
	char fw_ver[NVME_FWVER_SZ + 1];
	int ret;

	if (imgno > 0 || slotno > (nvme->n_idctl->id_frmw.fw_nslot - 1))
		return (EINVAL);

	mutex_enter(&nvme->n_fwslot_mutex);
	if (nvme->n_fwslot == NULL) {
		ret = nvme_get_logpage(nvme, B_TRUE, &log, &bufsize,
		    NVME_LOGPAGE_FWSLOT, 0);
		if (ret != DDI_SUCCESS ||
		    bufsize != sizeof (nvme_fwslot_log_t)) {
			if (log != NULL)
				kmem_free(log, bufsize);
			mutex_exit(&nvme->n_fwslot_mutex);
			return (EIO);
		}
		nvme->n_fwslot = (nvme_fwslot_log_t *)log;
	}

	/*
	 * NVMe numbers firmware slots starting at 1
	 */
	if (slotno == (nvme->n_fwslot->fw_afi - 1))
		attr |= DDI_UFM_ATTR_ACTIVE;

	if (slotno != 0 || nvme->n_idctl->id_frmw.fw_readonly == 0)
		attr |= DDI_UFM_ATTR_WRITEABLE;

	if (nvme->n_fwslot->fw_frs[slotno][0] == '\0') {
		attr |= DDI_UFM_ATTR_EMPTY;
	} else {
		(void) strncpy(fw_ver, nvme->n_fwslot->fw_frs[slotno],
		    NVME_FWVER_SZ);
		fw_ver[NVME_FWVER_SZ] = '\0';
		ddi_ufm_slot_set_version(slot, fw_ver);
	}
	mutex_exit(&nvme->n_fwslot_mutex);

	ddi_ufm_slot_set_attrs(slot, attr);

	return (0);
}

static int
nvme_ufm_getcaps(ddi_ufm_handle_t *ufmh, void *arg, ddi_ufm_cap_t *caps)
{
	*caps = DDI_UFM_CAP_REPORT;
	return (0);
}
