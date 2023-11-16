/*
 * Copyright 2019-2022, Haiku, Inc. All rights reserved.
 * Distributed under the terms of the MIT License.
 *
 * Authors:
 *		Augustin Cavalier <waddlesplash>
 */


#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <condition_variable.h>
#include <AutoDeleter.h>
#include <kernel.h>
#include <smp.h>
#include <util/AutoLock.h>

#include <fs/devfs.h>
#include <bus/PCI.h>
#include <vm/vm.h>

#include "IORequest.h"

extern "C" {
#include <libnvme/nvme.h>
#include <libnvme/nvme_internal.h>
}


//#define TRACE_NVME_DISK
#ifdef TRACE_NVME_DISK
#	define TRACE(x...) dprintf("nvme_disk: " x)
#else
#	define TRACE(x...) ;
#endif
#define TRACE_ALWAYS(x...)	dprintf("nvme_disk: " x)
#define TRACE_ERROR(x...)	dprintf("\33[33mnvme_disk:\33[0m " x)
#define CALLED() 			TRACE("CALLED %s\n", __PRETTY_FUNCTION__)


static const uint8 kDriveIcon[] = {
	0x6e, 0x63, 0x69, 0x66, 0x08, 0x03, 0x01, 0x00, 0x00, 0x02, 0x00, 0x16,
	0x02, 0x3c, 0xc7, 0xee, 0x38, 0x9b, 0xc0, 0xba, 0x16, 0x57, 0x3e, 0x39,
	0xb0, 0x49, 0x77, 0xc8, 0x42, 0xad, 0xc7, 0x00, 0xff, 0xff, 0xd3, 0x02,
	0x00, 0x06, 0x02, 0x3c, 0x96, 0x32, 0x3a, 0x4d, 0x3f, 0xba, 0xfc, 0x01,
	0x3d, 0x5a, 0x97, 0x4b, 0x57, 0xa5, 0x49, 0x84, 0x4d, 0x00, 0x47, 0x47,
	0x47, 0xff, 0xa5, 0xa0, 0xa0, 0x02, 0x00, 0x16, 0x02, 0xbc, 0x59, 0x2f,
	0xbb, 0x29, 0xa7, 0x3c, 0x0c, 0xe4, 0xbd, 0x0b, 0x7c, 0x48, 0x92, 0xc0,
	0x4b, 0x79, 0x66, 0x00, 0x7d, 0xff, 0xd4, 0x02, 0x00, 0x06, 0x02, 0x38,
	0xdb, 0xb4, 0x39, 0x97, 0x33, 0xbc, 0x4a, 0x33, 0x3b, 0xa5, 0x42, 0x48,
	0x6e, 0x66, 0x49, 0xee, 0x7b, 0x00, 0x59, 0x67, 0x56, 0xff, 0xeb, 0xb2,
	0xb2, 0x03, 0xa7, 0xff, 0x00, 0x03, 0xff, 0x00, 0x00, 0x04, 0x01, 0x80,
	0x07, 0x0a, 0x06, 0x22, 0x3c, 0x22, 0x49, 0x44, 0x5b, 0x5a, 0x3e, 0x5a,
	0x31, 0x39, 0x25, 0x0a, 0x04, 0x22, 0x3c, 0x44, 0x4b, 0x5a, 0x31, 0x39,
	0x25, 0x0a, 0x04, 0x44, 0x4b, 0x44, 0x5b, 0x5a, 0x3e, 0x5a, 0x31, 0x0a,
	0x04, 0x22, 0x3c, 0x22, 0x49, 0x44, 0x5b, 0x44, 0x4b, 0x08, 0x02, 0x27,
	0x43, 0xb8, 0x14, 0xc1, 0xf1, 0x08, 0x02, 0x26, 0x43, 0x29, 0x44, 0x0a,
	0x05, 0x44, 0x5d, 0x49, 0x5d, 0x60, 0x3e, 0x5a, 0x3b, 0x5b, 0x3f, 0x08,
	0x0a, 0x07, 0x01, 0x06, 0x00, 0x0a, 0x00, 0x01, 0x00, 0x10, 0x01, 0x17,
	0x84, 0x00, 0x04, 0x0a, 0x01, 0x01, 0x01, 0x00, 0x0a, 0x02, 0x01, 0x02,
	0x00, 0x0a, 0x03, 0x01, 0x03, 0x00, 0x0a, 0x04, 0x01, 0x04, 0x10, 0x01,
	0x17, 0x85, 0x20, 0x04, 0x0a, 0x06, 0x01, 0x05, 0x30, 0x24, 0xb3, 0x99,
	0x01, 0x17, 0x82, 0x00, 0x04, 0x0a, 0x05, 0x01, 0x05, 0x30, 0x20, 0xb2,
	0xe6, 0x01, 0x17, 0x82, 0x00, 0x04
};


#define NVME_DISK_DRIVER_MODULE_NAME 	"drivers/disk/nvme_disk/driver_v1"
#define NVME_DISK_DEVICE_MODULE_NAME 	"drivers/disk/nvme_disk/device_v1"
#define NVME_DISK_DEVICE_ID_GENERATOR	"nvme_disk/device_id"

#define NVME_MAX_QPAIRS					(16)


static device_manager_info* sDeviceManager;

typedef struct {
	device_node*			node;
	pci_info				info;

	struct nvme_ctrlr*		ctrlr;

	struct nvme_ns*			ns;
	uint64					capacity;
	uint32					block_size;
	uint32					max_io_blocks;
	status_t				media_status;

	DMAResource				dma_resource;
	sem_id					dma_buffers_sem;

	rw_lock					rounded_write_lock;

	ConditionVariable		interrupt;
	int32					polling;

	struct qpair_info {
		struct nvme_qpair*	qpair;
	}						qpairs[NVME_MAX_QPAIRS];
	uint32					qpair_count;
} nvme_disk_driver_info;
typedef nvme_disk_driver_info::qpair_info qpair_info;


typedef struct {
	nvme_disk_driver_info*		info;
} nvme_disk_handle;


static status_t
get_geometry(nvme_disk_handle* handle, device_geometry* geometry)
{
	nvme_disk_driver_info* info = handle->info;

	devfs_compute_geometry_size(geometry, info->capacity, info->block_size);
	geometry->bytes_per_physical_sector = info->block_size;

	geometry->device_type = B_DISK;
	geometry->removable = false;

	geometry->read_only = false;
	geometry->write_once = false;

	TRACE("get_geometry(): %" B_PRId32 ", %" B_PRId32 ", %" B_PRId32 ", %" B_PRId32 ", %d, %d, %d, %d\n",
		geometry->bytes_per_sector, geometry->sectors_per_track,
		geometry->cylinder_count, geometry->head_count, geometry->device_type,
		geometry->removable, geometry->read_only, geometry->write_once);

	return B_OK;
}


static void
nvme_disk_set_capacity(nvme_disk_driver_info* info, uint64 capacity,
	uint32 blockSize)
{
	TRACE("set_capacity(device = %p, capacity = %" B_PRIu64 ", blockSize = %" B_PRIu32 ")\n",
		info, capacity, blockSize);

	info->capacity = capacity;
	info->block_size = blockSize;
}


//	#pragma mark - device module API


static int32 nvme_interrupt_handler(void* _info);


static status_t
nvme_disk_init_device(void* _info, void** _cookie)
{
	CALLED();
	nvme_disk_driver_info* info = (nvme_disk_driver_info*)_info;
	ASSERT(info->ctrlr == NULL);

	pci_device_module_info* pci;
	pci_device* pcidev;
	device_node* parent = sDeviceManager->get_parent_node(info->node);
	sDeviceManager->get_driver(parent, (driver_module_info**)&pci,
		(void**)&pcidev);
	pci->get_pci_info(pcidev, &info->info);
	sDeviceManager->put_node(parent);

	// construct the libnvme pci_device struct
	pci_device* device = new pci_device;
	device->vendor_id = info->info.vendor_id;
	device->device_id = info->info.device_id;
	device->subvendor_id = 0;
	device->subdevice_id = 0;

	device->domain = 0;
	device->bus = info->info.bus;
	device->dev = info->info.device;
	device->func = info->info.function;

	device->pci_info = &info->info;

	// enable busmaster and memory mapped access
	uint16 command = pci->read_pci_config(pcidev, PCI_command, 2);
	command |= PCI_command_master | PCI_command_memory;
	pci->write_pci_config(pcidev, PCI_command, 2, command);

	// open the controller
	info->ctrlr = nvme_ctrlr_open(device, NULL);
	if (info->ctrlr == NULL) {
		TRACE_ERROR("failed to open the controller!\n");
		return B_ERROR;
	}

	struct nvme_ctrlr_stat cstat;
	int err = nvme_ctrlr_stat(info->ctrlr, &cstat);
	if (err != 0) {
		TRACE_ERROR("failed to get controller information!\n");
		nvme_ctrlr_close(info->ctrlr);
		return err;
	}

	TRACE_ALWAYS("attached to NVMe device \"%s (%s)\"\n", cstat.mn, cstat.sn);
	TRACE_ALWAYS("\tmaximum transfer size: %" B_PRIuSIZE "\n", cstat.max_xfer_size);
	TRACE_ALWAYS("\tqpair count: %d\n", cstat.io_qpairs);

	// TODO: export more than just the first namespace!
	info->ns = nvme_ns_open(info->ctrlr, cstat.ns_ids[0]);
	if (info->ns == NULL) {
		TRACE_ERROR("failed to open namespace!\n");
		nvme_ctrlr_close(info->ctrlr);
		return B_ERROR;
	}
	TRACE_ALWAYS("namespace 0\n");

	struct nvme_ns_stat nsstat;
	err = nvme_ns_stat(info->ns, &nsstat);
	if (err != 0) {
		TRACE_ERROR("failed to get namespace information!\n");
		nvme_ctrlr_close(info->ctrlr);
		return err;
	}

	// store capacity information
	TRACE_ALWAYS("\tblock size: %" B_PRIuSIZE ", stripe size: %u\n",
		nsstat.sector_size, info->ns->stripe_size);
	nvme_disk_set_capacity(info, nsstat.sectors, nsstat.sector_size);

	command = pci->read_pci_config(pcidev, PCI_command, 2);
	command &= ~(PCI_command_int_disable);
	pci->write_pci_config(pcidev, PCI_command, 2, command);

	uint8 irq = info->info.u.h0.interrupt_line;
	if (pci->get_msix_count(pcidev)) {
		uint8 msixVector = 0;
		if (pci->configure_msix(pcidev, 1, &msixVector) == B_OK
			&& pci->enable_msix(pcidev) == B_OK) {
			TRACE_ALWAYS("using MSI-X\n");
			irq = msixVector;
		}
	} else if (pci->get_msi_count(pcidev) >= 1) {
		uint8 msiVector = 0;
		if (pci->configure_msi(pcidev, 1, &msiVector) == B_OK
			&& pci->enable_msi(pcidev) == B_OK) {
			TRACE_ALWAYS("using message signaled interrupts\n");
			irq = msiVector;
		}
	}

	if (irq == 0 || irq == 0xFF) {
		TRACE_ERROR("device PCI:%d:%d:%d was assigned an invalid IRQ\n",
			info->info.bus, info->info.device, info->info.function);
		info->polling = 1;
	} else {
		info->polling = 0;
	}
	info->interrupt.Init(NULL, NULL);
	install_io_interrupt_handler(irq, nvme_interrupt_handler, (void*)info, B_NO_HANDLED_INFO);

	if (info->ctrlr->feature_supported[NVME_FEAT_INTERRUPT_COALESCING]) {
		uint32 microseconds = 16, threshold = 32;
		nvme_admin_set_feature(info->ctrlr, false, NVME_FEAT_INTERRUPT_COALESCING,
			((microseconds / 100) << 8) | threshold, 0, NULL);
	}

	// allocate qpairs
	uint32 try_qpairs = cstat.io_qpairs;
	try_qpairs = min_c(try_qpairs, NVME_MAX_QPAIRS);
	if (try_qpairs >= (uint32)smp_get_num_cpus()) {
		try_qpairs = smp_get_num_cpus();
	} else {
		// Find the highest number of qpairs that evenly divides the number of CPUs.
		while ((smp_get_num_cpus() % try_qpairs) != 0)
			try_qpairs--;
	}
	info->qpair_count = 0;
	for (uint32 i = 0; i < try_qpairs; i++) {
		info->qpairs[i].qpair = nvme_ioqp_get(info->ctrlr,
			(enum nvme_qprio)0, 0);
		if (info->qpairs[i].qpair == NULL)
			break;

		info->qpair_count++;
	}
	if (info->qpair_count == 0) {
		TRACE_ERROR("failed to allocate qpairs!\n");
		nvme_ctrlr_close(info->ctrlr);
		return B_NO_MEMORY;
	}
	if (info->qpair_count != try_qpairs) {
		TRACE_ALWAYS("warning: did not get expected number of qpairs\n");
	}

	// allocate DMA buffers
	int buffers = info->qpair_count * 2;

	dma_restrictions restrictions = {};
	restrictions.alignment = B_PAGE_SIZE;
		// Technically, the first and last segments in a transfer can be aligned
		// only on 32-bits, and the rest only need to have sizes that are a multiple
		// of the block size.
	restrictions.max_segment_count = (NVME_MAX_SGL_DESCRIPTORS / 2);
	restrictions.max_transfer_size = cstat.max_xfer_size;
	info->max_io_blocks = cstat.max_xfer_size / nsstat.sector_size;

	err = info->dma_resource.Init(restrictions, B_PAGE_SIZE, buffers, buffers);
	if (err != 0) {
		TRACE_ERROR("failed to initialize DMA resource!\n");
		nvme_ctrlr_close(info->ctrlr);
		return err;
	}

	info->dma_buffers_sem = create_sem(buffers, "nvme buffers sem");
	if (info->dma_buffers_sem < 0) {
		TRACE_ERROR("failed to create DMA buffers semaphore!\n");
		nvme_ctrlr_close(info->ctrlr);
		return info->dma_buffers_sem;
	}

	// set up rounded-write lock
	rw_lock_init(&info->rounded_write_lock, "nvme rounded writes");

	*_cookie = info;
	return B_OK;
}


static void
nvme_disk_uninit_device(void* _cookie)
{
	CALLED();
	nvme_disk_driver_info* info = (nvme_disk_driver_info*)_cookie;

	remove_io_interrupt_handler(info->info.u.h0.interrupt_line,
		nvme_interrupt_handler, (void*)info);

	rw_lock_destroy(&info->rounded_write_lock);

	nvme_ns_close(info->ns);
	nvme_ctrlr_close(info->ctrlr);

	// TODO: Deallocate MSI(-X).
	// TODO: Deallocate PCI.
}


static status_t
nvme_disk_open(void* _info, const char* path, int openMode, void** _cookie)
{
	CALLED();

	nvme_disk_driver_info* info = (nvme_disk_driver_info*)_info;
	nvme_disk_handle* handle = (nvme_disk_handle*)malloc(
		sizeof(nvme_disk_handle));
	if (handle == NULL)
		return B_NO_MEMORY;

	handle->info = info;

	*_cookie = handle;
	return B_OK;
}


static status_t
nvme_disk_close(void* cookie)
{
	CALLED();

	//nvme_disk_handle* handle = (nvme_disk_handle*)cookie;
	return B_OK;
}


static status_t
nvme_disk_free(void* cookie)
{
	CALLED();

	nvme_disk_handle* handle = (nvme_disk_handle*)cookie;
	free(handle);
	return B_OK;
}


// #pragma mark - I/O


static int32
nvme_interrupt_handler(void* _info)
{
	nvme_disk_driver_info* info = (nvme_disk_driver_info*)_info;
	info->interrupt.NotifyAll();
	info->polling = -1;
	return 0;
}


static qpair_info*
get_qpair(nvme_disk_driver_info* info)
{
	return &info->qpairs[smp_get_current_cpu() % info->qpair_count];
}


static void
io_finished_callback(status_t* status, const struct nvme_cpl* cpl)
{
	*status = nvme_cpl_is_error(cpl) ? B_IO_ERROR : B_OK;
}


static void
await_status(nvme_disk_driver_info* info, struct nvme_qpair* qpair, status_t& status)
{
	CALLED();

	ConditionVariableEntry entry;
	int timeouts = 0;
	while (status == EINPROGRESS) {
		info->interrupt.Add(&entry);

		nvme_qpair_poll(qpair, 0);

		if (status != EINPROGRESS)
			return;

		if (info->polling > 0) {
			entry.Wait(B_RELATIVE_TIMEOUT, min_c(5 * 1000 * 1000,
				(1 << timeouts) * 1000));
			timeouts++;
		} else if (entry.Wait(B_RELATIVE_TIMEOUT, 5 * 1000 * 1000) != B_OK) {
			// This should never happen, as we are woken up on every interrupt
			// no matter the qpair or transfer within; so if it does occur,
			// that probably means the controller stalled, or maybe cannot
			// generate interrupts at all.

			TRACE_ERROR("timed out waiting for interrupt!\n");
			if (timeouts++ >= 3) {
				nvme_qpair_fail(qpair);
				status = B_TIMED_OUT;
				return;
			}

			info->polling++;
			if (info->polling > 0) {
				TRACE_ALWAYS("switching to polling mode, performance will be affected!\n");
			}
		}

		nvme_qpair_poll(qpair, 0);
	}
}


struct nvme_io_request {
	status_t status;

	bool write;

	off_t lba_start;
	size_t lba_count;

	physical_entry* iovecs;
	int32 iovec_count;

	int32 iovec_i;
	uint32 iovec_offset;
};


static void
ior_reset_sgl(nvme_io_request* request, uint32_t offset)
{
	TRACE("IOR Reset: %" B_PRIu32 "\n", offset);

	int32 i = 0;
	while (offset > 0 && request->iovecs[i].size <= offset) {
		offset -= request->iovecs[i].size;
		i++;
	}
	request->iovec_i = i;
	request->iovec_offset = offset;
}


static int
ior_next_sge(nvme_io_request* request, uint64_t* address, uint32_t* length)
{
	int32 index = request->iovec_i;
	if (index < 0 || index > request->iovec_count)
		return -1;

	*address = request->iovecs[index].address + request->iovec_offset;
	*length = request->iovecs[index].size - request->iovec_offset;

	TRACE("IOV %d (+ " B_PRIu32 "): 0x%" B_PRIx64 ", %" B_PRIu32 "\n",
		request->iovec_i, request->iovec_offset, *address, *length);

	request->iovec_i++;
	request->iovec_offset = 0;
	return 0;
}


static status_t
do_nvme_io_request(nvme_disk_driver_info* info, nvme_io_request* request)
{
	request->status = EINPROGRESS;

	qpair_info* qpinfo = get_qpair(info);
	int ret = -1;
	if (request->write) {
		ret = nvme_ns_writev(info->ns, qpinfo->qpair, request->lba_start,
			request->lba_count, (nvme_cmd_cb)io_finished_callback, request,
			0, (nvme_req_reset_sgl_cb)ior_reset_sgl,
			(nvme_req_next_sge_cb)ior_next_sge);
	} else {
		ret = nvme_ns_readv(info->ns, qpinfo->qpair, request->lba_start,
			request->lba_count, (nvme_cmd_cb)io_finished_callback, request,
			0, (nvme_req_reset_sgl_cb)ior_reset_sgl,
			(nvme_req_next_sge_cb)ior_next_sge);
	}
	if (ret != 0) {
		TRACE_ERROR("attempt to queue %s I/O at LBA %" B_PRIdOFF " of %" B_PRIuSIZE
			" blocks failed!\n", request->write ? "write" : "read",
			request->lba_start, request->lba_count);

		request->lba_count = 0;
		return ret;
	}

	await_status(info, qpinfo->qpair, request->status);

	if (request->status != B_OK) {
		TRACE_ERROR("%s at LBA %" B_PRIdOFF " of %" B_PRIuSIZE
			" blocks failed!\n", request->write ? "write" : "read",
			request->lba_start, request->lba_count);

		request->lba_count = 0;
	}
	return request->status;
}


static status_t
nvme_disk_bounced_io(nvme_disk_handle* handle, io_request* request)
{
	CALLED();

	WriteLocker writeLocker;
	if (request->IsWrite())
		writeLocker.SetTo(handle->info->rounded_write_lock, false);

	status_t status = acquire_sem(handle->info->dma_buffers_sem);
	if (status != B_OK) {
		request->SetStatusAndNotify(status);
		return status;
	}

	const size_t block_size = handle->info->block_size;

	TRACE("%p: IOR Offset: %" B_PRIdOFF "; Length %" B_PRIuGENADDR
		"; Write %s\n", request, request->Offset(), request->Length(),
		request->IsWrite() ? "yes" : "no");

	nvme_io_request nvme_request;
	while (request->RemainingBytes() > 0) {
		IOOperation operation;
		status = handle->info->dma_resource.TranslateNext(request, &operation, 0);
		if (status != B_OK)
			break;

		do {
			TRACE("%p: IOO offset: %" B_PRIdOFF ", length: %" B_PRIuGENADDR
				", write: %s\n", request, operation.Offset(),
				operation.Length(), operation.IsWrite() ? "yes" : "no");

			nvme_request.write = operation.IsWrite();
			nvme_request.lba_start = operation.Offset() / block_size;
			nvme_request.lba_count = operation.Length() / block_size;
			nvme_request.iovecs = (physical_entry*)operation.Vecs();
			nvme_request.iovec_count = operation.VecCount();

			status = do_nvme_io_request(handle->info, &nvme_request);

			operation.SetStatus(status,
				status == B_OK ? operation.Length() : 0);
		} while (status == B_OK && !operation.Finish());

		if (status == B_OK && operation.Status() != B_OK) {
			TRACE_ERROR("I/O succeeded but IOOperation failed!\n");
			status = operation.Status();
		}

		request->OperationFinished(&operation);

		handle->info->dma_resource.RecycleBuffer(operation.Buffer());

		TRACE("%p: status %s, remaining bytes %" B_PRIuGENADDR "\n", request,
			strerror(status), request->RemainingBytes());
		if (status != B_OK)
			break;
	}

	release_sem(handle->info->dma_buffers_sem);

	// Notify() also takes care of UnlockMemory().
	if (status != B_OK && request->Status() == B_OK)
		request->SetStatusAndNotify(status);
	else
		request->NotifyFinished();
	return status;
}


static status_t
nvme_disk_io(void* cookie, io_request* request)
{
	CALLED();

	nvme_disk_handle* handle = (nvme_disk_handle*)cookie;

	const off_t ns_end = (handle->info->capacity * handle->info->block_size);
	if ((request->Offset() + (off_t)request->Length()) > ns_end)
		return ERANGE;

	nvme_io_request nvme_request;
	memset(&nvme_request, 0, sizeof(nvme_io_request));

	nvme_request.write = request->IsWrite();

	physical_entry* vtophys = NULL;
	MemoryDeleter vtophysDeleter;

	IOBuffer* buffer = request->Buffer();
	status_t status = B_OK;
	if (!buffer->IsPhysical()) {
		status = buffer->LockMemory(request->TeamID(), request->IsWrite());
		if (status != B_OK) {
			TRACE_ERROR("failed to lock memory: %s\n", strerror(status));
			return status;
		}
		// SetStatusAndNotify() takes care of unlocking memory if necessary.

		// This is slightly inefficient, as we could use a BStackOrHeapArray in
		// the optimal case (few physical entries required), but we would not
		// know whether or not that was possible until calling get_memory_map()
		// and then potentially reallocating, which would complicate the logic.

		int32 vtophys_length = (request->Length() / B_PAGE_SIZE) + 2;
		nvme_request.iovecs = vtophys = (physical_entry*)malloc(sizeof(physical_entry)
			* vtophys_length);
		if (vtophys == NULL) {
			TRACE_ERROR("failed to allocate memory for iovecs\n");
			request->SetStatusAndNotify(B_NO_MEMORY);
			return B_NO_MEMORY;
		}
		vtophysDeleter.SetTo(vtophys);

		for (size_t i = 0; i < buffer->VecCount(); i++) {
			generic_io_vec virt = buffer->VecAt(i);
			uint32 entries = vtophys_length - nvme_request.iovec_count;

			// Avoid copies by going straight into the vtophys array.
			status = get_memory_map_etc(request->TeamID(), (void*)virt.base,
				virt.length, vtophys + nvme_request.iovec_count, &entries);
			if (status == B_BUFFER_OVERFLOW) {
				TRACE("vtophys array was too small, reallocating\n");

				vtophysDeleter.Detach();
				vtophys_length *= 2;
				nvme_request.iovecs = vtophys = (physical_entry*)realloc(vtophys,
					sizeof(physical_entry) * vtophys_length);
				vtophysDeleter.SetTo(vtophys);
				if (vtophys == NULL) {
					status = B_NO_MEMORY;
				} else {
					// Try again, with the larger buffer this time.
					i--;
					continue;
				}
			}
			if (status != B_OK) {
				TRACE_ERROR("I/O get_memory_map failed: %s\n", strerror(status));
				request->SetStatusAndNotify(status);
				return status;
			}

			nvme_request.iovec_count += entries;
		}
	} else {
		nvme_request.iovecs = (physical_entry*)buffer->Vecs();
		nvme_request.iovec_count = buffer->VecCount();
	}

	// See if we need to bounce anything other than the first or last vec.
	const size_t block_size = handle->info->block_size;
	bool bounceAll = false;
	for (int32 i = 1; !bounceAll && i < (nvme_request.iovec_count - 1); i++) {
		if ((nvme_request.iovecs[i].address % B_PAGE_SIZE) != 0)
			bounceAll = true;
		if ((nvme_request.iovecs[i].size % B_PAGE_SIZE) != 0)
			bounceAll = true;
	}

	// See if we need to bounce due to the first or last vecs.
	if (nvme_request.iovec_count > 1) {
		// There are middle vecs, so the first and last vecs have different restrictions: they
		// need only be a multiple of the block size, and must end and start on a page boundary,
		// respectively, though the start address must always be 32-bit-aligned.
		physical_entry* entry = &nvme_request.iovecs[0];
		if (!bounceAll && (((entry->address + entry->size) % B_PAGE_SIZE) != 0
				|| (entry->address & 0x3) != 0 || (entry->size % block_size) != 0))
			bounceAll = true;

		entry = &nvme_request.iovecs[nvme_request.iovec_count - 1];
		if (!bounceAll && ((entry->address % B_PAGE_SIZE) != 0
				|| (entry->size % block_size) != 0))
			bounceAll = true;
	} else {
		// There is only one vec. Check that it is a multiple of the block size,
		// and that its address is 32-bit-aligned.
		physical_entry* entry = &nvme_request.iovecs[0];
		if (!bounceAll && ((entry->address & 0x3) != 0 || (entry->size % block_size) != 0))
			bounceAll = true;
	}

	// See if we need to bounce due to rounding.
	const off_t rounded_pos = ROUNDDOWN(request->Offset(), block_size);
	phys_size_t rounded_len = ROUNDUP(request->Length() + (request->Offset()
		- rounded_pos), block_size);
	if (rounded_pos != request->Offset() || rounded_len != request->Length())
		bounceAll = true;

	if (bounceAll) {
		// Let the bounced I/O routine take care of everything from here.
		return nvme_disk_bounced_io(handle, request);
	}

	nvme_request.lba_start = rounded_pos / block_size;
	nvme_request.lba_count = rounded_len / block_size;

	// No bouncing was required.
	ReadLocker readLocker;
	if (nvme_request.write)
		readLocker.SetTo(handle->info->rounded_write_lock, false);

	// Error check before actually doing I/O.
	if (status != B_OK) {
		TRACE_ERROR("I/O failed early: %s\n", strerror(status));
		request->SetStatusAndNotify(status);
		return status;
	}

	const uint32 max_io_blocks = handle->info->max_io_blocks;
	int32 remaining = nvme_request.iovec_count;
	while (remaining > 0) {
		nvme_request.iovec_count = min_c(remaining,
			NVME_MAX_SGL_DESCRIPTORS / 2);

		nvme_request.lba_count = 0;
		for (int i = 0; i < nvme_request.iovec_count; i++) {
			uint32 new_lba_count = nvme_request.lba_count
				+ (nvme_request.iovecs[i].size / block_size);
			if (nvme_request.lba_count > 0 && new_lba_count > max_io_blocks) {
				// We already have a nonzero length, and adding this vec would
				// make us go over (or we already are over.) Stop adding.
				nvme_request.iovec_count = i;
				break;
			}

			nvme_request.lba_count = new_lba_count;
		}

		status = do_nvme_io_request(handle->info, &nvme_request);
		if (status != B_OK)
			break;

		nvme_request.iovecs += nvme_request.iovec_count;
		remaining -= nvme_request.iovec_count;
		nvme_request.lba_start += nvme_request.lba_count;
	}

	if (status != B_OK)
		TRACE_ERROR("I/O failed: %s\n", strerror(status));

	request->SetTransferredBytes(status != B_OK,
		(nvme_request.lba_start * block_size) - rounded_pos);
	request->SetStatusAndNotify(status);
	return status;
}


static status_t
nvme_disk_read(void* cookie, off_t pos, void* buffer, size_t* length)
{
	CALLED();
	nvme_disk_handle* handle = (nvme_disk_handle*)cookie;

	const off_t ns_end = (handle->info->capacity * handle->info->block_size);
	if (pos >= ns_end)
		return B_BAD_VALUE;
	if ((pos + (off_t)*length) > ns_end)
		*length = ns_end - pos;

	IORequest request;
	status_t status = request.Init(pos, (addr_t)buffer, *length, false, 0);
	if (status != B_OK)
		return status;

	status = nvme_disk_io(handle, &request);
	*length = request.TransferredBytes();
	return status;
}


static status_t
nvme_disk_write(void* cookie, off_t pos, const void* buffer, size_t* length)
{
	CALLED();
	nvme_disk_handle* handle = (nvme_disk_handle*)cookie;

	const off_t ns_end = (handle->info->capacity * handle->info->block_size);
	if (pos >= ns_end)
		return B_BAD_VALUE;
	if ((pos + (off_t)*length) > ns_end)
		*length = ns_end - pos;

	IORequest request;
	status_t status = request.Init(pos, (addr_t)buffer, *length, true, 0);
	if (status != B_OK)
		return status;

	status = nvme_disk_io(handle, &request);
	*length = request.TransferredBytes();
	return status;
}


static status_t
nvme_disk_flush(nvme_disk_driver_info* info)
{
	CALLED();
	status_t status = EINPROGRESS;

	qpair_info* qpinfo = get_qpair(info);
	int ret = nvme_ns_flush(info->ns, qpinfo->qpair,
		(nvme_cmd_cb)io_finished_callback, &status);
	if (ret != 0)
		return ret;

	await_status(info, qpinfo->qpair, status);
	return status;
}


static status_t
nvme_disk_trim(nvme_disk_driver_info* info, fs_trim_data* trimData)
{
	CALLED();
	trimData->trimmed_size = 0;

	const off_t deviceSize = info->capacity * info->block_size; // in bytes
	if (deviceSize < 0)
		return B_BAD_VALUE;

	STATIC_ASSERT(sizeof(deviceSize) <= sizeof(uint64));
	ASSERT(deviceSize >= 0);

	// Do not trim past device end.
	for (uint32 i = 0; i < trimData->range_count; i++) {
		uint64 offset = trimData->ranges[i].offset;
		uint64& size = trimData->ranges[i].size;

		if (offset >= (uint64)deviceSize)
			return B_BAD_VALUE;
		size = std::min(size, (uint64)deviceSize - offset);
	}

	// We need contiguous memory for the DSM ranges.
	nvme_dsm_range* dsmRanges = (nvme_dsm_range*)nvme_mem_alloc_node(
		trimData->range_count * sizeof(nvme_dsm_range), 0, 0, NULL);
	if (dsmRanges == NULL)
		return B_NO_MEMORY;
	CObjectDeleter<void, void, nvme_free> dsmRangesDeleter(dsmRanges);

	uint64 trimmingSize = 0;
	for (uint32 i = 0; i < trimData->range_count; i++) {
		uint64 offset = trimData->ranges[i].offset;
		uint64 length = trimData->ranges[i].size;

		// Round up offset and length to the block size.
		// (Some space at the beginning and end may thus not be trimmed.)
		offset = ROUNDUP(offset, info->block_size);
		length -= offset - trimData->ranges[i].offset;
		length = ROUNDDOWN(length, info->block_size);

		if (length == 0)
			continue;
		if ((length / info->block_size) > UINT32_MAX)
			length = uint64(UINT32_MAX) * info->block_size;
			// TODO: Break into smaller trim ranges!

		TRACE("trim %" B_PRIu64 " bytes from %" B_PRIu64 "\n", length, offset);

		dsmRanges[i].attributes = 0;
		dsmRanges[i].length = length / info->block_size;
		dsmRanges[i].starting_lba = offset / info->block_size;

		trimmingSize += length;
	}

	status_t status = EINPROGRESS;
	qpair_info* qpair = get_qpair(info);
	if (nvme_ns_deallocate(info->ns, qpair->qpair, dsmRanges, trimData->range_count,
			(nvme_cmd_cb)io_finished_callback, &status) != 0)
		return B_IO_ERROR;

	await_status(info, qpair->qpair, status);
	if (status != B_OK)
		return status;

	trimData->trimmed_size = trimmingSize;
	return B_OK;
}


static status_t
nvme_disk_ioctl(void* cookie, uint32 op, void* buffer, size_t length)
{
	CALLED();
	nvme_disk_handle* handle = (nvme_disk_handle*)cookie;
	nvme_disk_driver_info* info = handle->info;

	TRACE("ioctl(op = %" B_PRId32 ")\n", op);

	switch (op) {
		case B_GET_MEDIA_STATUS:
		{
			*(status_t *)buffer = info->media_status;
			info->media_status = B_OK;
			return B_OK;
			break;
		}

		case B_GET_DEVICE_SIZE:
		{
			size_t size = info->capacity * info->block_size;
			return user_memcpy(buffer, &size, sizeof(size_t));
		}

		case B_GET_GEOMETRY:
		{
			if (buffer == NULL || length > sizeof(device_geometry))
				return B_BAD_VALUE;

		 	device_geometry geometry;
			status_t status = get_geometry(handle, &geometry);
			if (status != B_OK)
				return status;

			return user_memcpy(buffer, &geometry, length);
		}

		case B_GET_ICON_NAME:
			return user_strlcpy((char*)buffer, "devices/drive-harddisk",
				B_FILE_NAME_LENGTH);

		case B_GET_VECTOR_ICON:
		{
			device_icon iconData;
			if (length != sizeof(device_icon))
				return B_BAD_VALUE;
			if (user_memcpy(&iconData, buffer, sizeof(device_icon)) != B_OK)
				return B_BAD_ADDRESS;

			if (iconData.icon_size >= (int32)sizeof(kDriveIcon)) {
				if (user_memcpy(iconData.icon_data, kDriveIcon,
						sizeof(kDriveIcon)) != B_OK)
					return B_BAD_ADDRESS;
			}

			iconData.icon_size = sizeof(kDriveIcon);
			return user_memcpy(buffer, &iconData, sizeof(device_icon));
		}

		case B_FLUSH_DRIVE_CACHE:
			return nvme_disk_flush(info);

		case B_TRIM_DEVICE:
			ASSERT(IS_KERNEL_ADDRESS(buffer));
			return nvme_disk_trim(info, (fs_trim_data*)buffer);
	}

	return B_DEV_INVALID_IOCTL;
}


//	#pragma mark - driver module API


static float
nvme_disk_supports_device(device_node *parent)
{
	CALLED();

	const char* bus;
	uint16 baseClass, subClass;

	if (sDeviceManager->get_attr_string(parent, B_DEVICE_BUS, &bus, false) != B_OK
		|| sDeviceManager->get_attr_uint16(parent, B_DEVICE_TYPE, &baseClass, false) != B_OK
		|| sDeviceManager->get_attr_uint16(parent, B_DEVICE_SUB_TYPE, &subClass, false) != B_OK)
		return -1.0f;

	if (strcmp(bus, "pci") != 0 || baseClass != PCI_mass_storage)
		return 0.0f;

	if (subClass != PCI_nvm)
		return 0.0f;

	TRACE("NVMe device found!\n");
	return 1.0f;
}


static status_t
nvme_disk_register_device(device_node* parent)
{
	CALLED();

	device_attr attrs[] = {
		{ B_DEVICE_PRETTY_NAME, B_STRING_TYPE, { .string = "NVMe Disk" } },
		{ NULL }
	};

	return sDeviceManager->register_node(parent, NVME_DISK_DRIVER_MODULE_NAME,
		attrs, NULL, NULL);
}


static status_t
nvme_disk_init_driver(device_node* node, void** cookie)
{
	CALLED();

	int ret = nvme_lib_init((enum nvme_log_level)0, (enum nvme_log_facility)0, NULL);
	if (ret != 0) {
		TRACE_ERROR("libnvme initialization failed!\n");
		return ret;
	}

	nvme_disk_driver_info* info = new nvme_disk_driver_info;
	if (info == NULL)
		return B_NO_MEMORY;

	info->media_status = B_OK;
	info->node = node;

	info->ctrlr = NULL;

	*cookie = info;
	return B_OK;
}


static void
nvme_disk_uninit_driver(void* _cookie)
{
	CALLED();

	nvme_disk_driver_info* info = (nvme_disk_driver_info*)_cookie;
	free(info);
}


static status_t
nvme_disk_register_child_devices(void* _cookie)
{
	CALLED();

	nvme_disk_driver_info* info = (nvme_disk_driver_info*)_cookie;
	status_t status;

	int32 id = sDeviceManager->create_id(NVME_DISK_DEVICE_ID_GENERATOR);
	if (id < 0)
		return id;

	char name[64];
	snprintf(name, sizeof(name), "disk/nvme/%" B_PRId32 "/raw",
		id);

	status = sDeviceManager->publish_device(info->node, name,
		NVME_DISK_DEVICE_MODULE_NAME);

	return status;
}


//	#pragma mark -


module_dependency module_dependencies[] = {
	{ B_DEVICE_MANAGER_MODULE_NAME, (module_info**)&sDeviceManager },
	{ NULL }
};

struct device_module_info sNvmeDiskDevice = {
	{
		NVME_DISK_DEVICE_MODULE_NAME,
		0,
		NULL
	},

	nvme_disk_init_device,
	nvme_disk_uninit_device,
	NULL, // remove,

	nvme_disk_open,
	nvme_disk_close,
	nvme_disk_free,
	nvme_disk_read,
	nvme_disk_write,
	nvme_disk_io,
	nvme_disk_ioctl,

	NULL,	// select
	NULL,	// deselect
};

struct driver_module_info sNvmeDiskDriver = {
	{
		NVME_DISK_DRIVER_MODULE_NAME,
		0,
		NULL
	},

	nvme_disk_supports_device,
	nvme_disk_register_device,
	nvme_disk_init_driver,
	nvme_disk_uninit_driver,
	nvme_disk_register_child_devices,
	NULL,	// rescan
	NULL,	// removed
};

module_info* modules[] = {
	(module_info*)&sNvmeDiskDriver,
	(module_info*)&sNvmeDiskDevice,
	NULL
};
