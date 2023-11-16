/* grub_file.c - GRUB file handling interface */
/*
 *  Copyright © 2014-2020 Pete Batard <pete@akeo.ie>
 *  Based on GRUB  --  GRand Unified Bootloader
 *  Copyright © 2001-2014 Free Software Foundation, Inc.
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <grub/err.h>
#include <grub/misc.h>
#include <grub/disk.h>
#include <grub/fs.h>
#include <grub/dl.h>
#include <grub/file.h>

#include "driver.h"

/* The file system list should only ever contain one element */
grub_fs_t grub_fs_list = NULL;

/* Keep track of the mounted filesystems */
LIST_ENTRY FsListHead;

grub_file_filter_t grub_file_filters_all[GRUB_FILE_FILTER_MAX];
grub_file_filter_t grub_file_filters_enabled[GRUB_FILE_FILTER_MAX];

extern EFI_STATUS GrubErrToEFIStatus(grub_err_t err);

/* Don't care about refcounts for a standalone EFI FS driver */
int
grub_dl_ref(grub_dl_t mod) {
	return 0;
}

int
grub_dl_unref(grub_dl_t mod) {
	return 0;
};

/* The following 3 calls are copied verbatim from the GRUB kernel */
grub_ssize_t
grub_file_read (grub_file_t file, void *buf, grub_size_t len)
{
	grub_ssize_t res;
	grub_disk_read_hook_t read_hook;
	void *read_hook_data;

	if (file->offset > file->size)
	{
		grub_error (GRUB_ERR_OUT_OF_RANGE,
			N_("attempt to read past the end of file"));
		return -1;
	}

	if (len == 0)
		return 0;

	if (len > file->size - file->offset)
		len = (grub_size_t) (file->size - file->offset);

	/* Prevent an overflow.  */
	if ((grub_ssize_t) len < 0)
		len >>= 1;

	if (len == 0)
		return 0;
	read_hook = file->read_hook;
	read_hook_data = file->read_hook_data;
	if (!file->read_hook)
	{
		file->read_hook = grub_file_progress_hook;
		file->read_hook_data = file;
		file->progress_offset = file->offset;
	}
	res = (file->fs->fs_read) (file, buf, len);
	file->read_hook = read_hook;
	file->read_hook_data = read_hook_data;
	if (res > 0)
		file->offset += res;

	return res;
}

grub_err_t
grub_file_close(grub_file_t file)
{
	if (file->fs->fs_close)
		(file->fs->fs_close) (file);

	if (file->device)
		grub_device_close (file->device);
	grub_free (file->name);
	grub_free (file);
	return grub_errno;
}

grub_off_t
grub_file_seek(grub_file_t file, grub_off_t offset)
{
	grub_off_t old;

	if (offset > file->size)
	{
		grub_error (GRUB_ERR_OUT_OF_RANGE,
			N_("attempt to seek outside of the file"));
		return (grub_off_t)-1;
	}

	old = file->offset;
	file->offset = offset;

	return old;
}

int
grub_device_iterate(grub_device_iterate_hook_t hook, void *hook_data)
{
	PrintError(L"grub_device_iterate() called\n");
	return 0;
}

grub_disk_read_hook_t grub_file_progress_hook = NULL;

grub_err_t
grub_disk_read(grub_disk_t disk, grub_disk_addr_t sector,
		grub_off_t offset, grub_size_t size, void *buf)
{
	EFI_STATUS Status;
	EFI_FS* FileSystem = (EFI_FS *) disk->data;
	EFI_BLOCK_IO_MEDIA *Media;

	FS_ASSERT(FileSystem != NULL);
	FS_ASSERT(FileSystem->DiskIo != NULL);
	FS_ASSERT(FileSystem->BlockIo != NULL);

	if (FileSystem->BlockIo2 != NULL) {
		Media = FileSystem->BlockIo2->Media;
	} else {
		Media = FileSystem->BlockIo->Media;
	}

	/* NB: We could get the actual blocksize through FileSystem->BlockIo->Media->BlockSize
	 * but GRUB uses the fixed GRUB_DISK_SECTOR_SIZE, so we follow suit
	 */
	if (FileSystem->DiskIo2 != NULL) {
		Status = FileSystem->DiskIo2->ReadDiskEx(FileSystem->DiskIo2, Media->MediaId,
			sector * GRUB_DISK_SECTOR_SIZE + offset, &(FileSystem->DiskIo2Token), size, buf);
	} else {
		Status = FileSystem->DiskIo->ReadDisk(FileSystem->DiskIo, Media->MediaId,
			sector * GRUB_DISK_SECTOR_SIZE + offset, (UINTN)size, buf);
	}

	if (EFI_ERROR(Status)) {
		PrintStatusError(Status, L"Could not read block at address %08x", sector);
		return grub_error (GRUB_ERR_READ_ERROR, N_("Could not read block"));
	}

	return 0;
}

grub_uint64_t
grub_disk_native_sectors (grub_disk_t disk)
{
	EFI_FS* FileSystem = (EFI_FS *) disk->data;

	FS_ASSERT(FileSystem != NULL);
	FS_ASSERT(FileSystem->BlockIo != NULL);

	if (FileSystem->BlockIo2 != NULL) {
		return (FileSystem->BlockIo2->Media->LastBlock + 1) *
			FileSystem->BlockIo2->Media->BlockSize;
	}

	return (FileSystem->BlockIo->Media->LastBlock + 1) *
			FileSystem->BlockIo->Media->BlockSize;
}

grub_disk_dev_t grub_disk_dev_list;

void
grub_disk_dev_register(grub_disk_dev_t dev)
{
	dev->next = grub_disk_dev_list;
	grub_disk_dev_list = dev;
}

void
grub_disk_dev_unregister(grub_disk_dev_t dev)
{
	grub_disk_dev_t *p, q;

	for (p = &grub_disk_dev_list, q = *p; q; p = &(q->next), q = q->next)
		if (q == dev)
		{
			*p = q->next;
			break;
		}
}

grub_device_t
grub_device_open(const char *name)
{
	EFI_DEVICE_PATH* DevicePath = (EFI_DEVICE_PATH*)name;
	EFI_FS *FileSystem;
	struct grub_device* device;

	FS_ASSERT(DevicePath != NULL);

	for (FileSystem = (EFI_FS *) FORWARD_LINK_REF(FsListHead); FileSystem != (EFI_FS *) &FsListHead;
			FileSystem = (EFI_FS *) FileSystem->Flink) {
		if (CompareDevicePaths(FileSystem->DevicePath, DevicePath) == 0)
			break;
	}
	if (FileSystem == (EFI_FS *) &FsListHead)
		return NULL;

	device = grub_zalloc(sizeof(struct grub_device));
	if (device == NULL)
		return NULL;
	device->disk = grub_zalloc(sizeof(struct grub_disk));
	if (device->disk == NULL) {
		grub_free(device);
		return NULL;
	}
	/* The private disk data is a pointer back to our EFI_FS */
	device->disk->data = (void *) FileSystem;
	/* Ideally, we'd fill the other disk data, such as total_sectors, name
	 * and so on, but since we're doing the actual disk access through EFI
	 * DiskIO rather than GRUB's disk.c, this doesn't seem to be needed.
	 */

	return device;
}

grub_err_t
grub_device_close(grub_device_t device)
{
	FS_ASSERT(device != NULL);

	grub_free(device->disk);
	grub_free(device);
	return 0;
}

EFI_STATUS
GrubDeviceInit(EFI_FS *FileSystem)
{
	FS_ASSERT(FileSystem->DevicePath != NULL);

	/* Insert this filesystem in our list */
	InsertTailList(&FsListHead, (LIST_ENTRY *) FileSystem);

	FileSystem->GrubDevice = (VOID *) grub_device_open((const char *)FileSystem->DevicePath);

	if (FileSystem->GrubDevice == NULL) {
		RemoveEntryList((LIST_ENTRY *)FileSystem);
		return EFI_NOT_FOUND;
	}

	return EFI_SUCCESS;
}

EFI_STATUS
GrubDeviceExit(EFI_FS *FileSystem)
{
	grub_device_close((grub_device_t) FileSystem->GrubDevice);
	RemoveEntryList((LIST_ENTRY *)FileSystem);

	return EFI_SUCCESS;
}

EFI_STATUS
GrubCreateFile(EFI_GRUB_FILE **File, EFI_FS *FileSystem)
{
	EFI_GRUB_FILE *NewFile;
	grub_file_t f;

	NewFile = AllocateZeroPool(sizeof(*NewFile));
	if (NewFile == NULL)
		return EFI_OUT_OF_RESOURCES;
	NewFile->GrubFile = AllocateZeroPool(sizeof(*f));
	if (NewFile->GrubFile == NULL) {
		FreePool(NewFile);
		return EFI_OUT_OF_RESOURCES;
	}

	/* Initialize the attributes */
	NewFile->FileSystem = FileSystem;
	FS_ASSERT(FileSystem->RootFile != NULL);
	CopyMem(&NewFile->EfiFile, &FileSystem->RootFile->EfiFile, sizeof(EFI_FILE));

	f = (grub_file_t) NewFile->GrubFile;
	f->device = (grub_device_t) FileSystem->GrubDevice;
	f->fs = grub_fs_list;

	*File = NewFile;
	return EFI_SUCCESS;
}

VOID
GrubDestroyFile(EFI_GRUB_FILE *File)
{
	if (File == NULL)
		return;
	if (File->GrubFile != NULL)
		FreePool(File->GrubFile);
	FreePool(File);
}

UINT64
GrubGetFileSize(EFI_GRUB_FILE *File)
{
	grub_file_t f = (grub_file_t) File->GrubFile;
	return (UINT64) f->size;
}

UINT64
GrubGetFileOffset(EFI_GRUB_FILE *File)
{
	grub_file_t f = (grub_file_t) File->GrubFile;
	return (UINT64) f->offset;
}

VOID
GrubSetFileOffset(EFI_GRUB_FILE *File, UINT64 Offset)
{
	grub_file_t f = (grub_file_t) File->GrubFile;
	f->offset = (grub_off_t) Offset;
}

/*
 * The following provides an EFI interface for each basic GRUB fs call
 */
EFI_STATUS
GrubDir(EFI_GRUB_FILE *File, const CHAR8 *path,
		GRUB_DIRHOOK Hook, VOID *HookData)
{
	grub_fs_t p = grub_fs_list;
	grub_file_t f = (grub_file_t) File->GrubFile;
	grub_err_t rc;

	grub_errno = 0;
	rc = p->fs_dir(f->device, path, (grub_fs_dir_hook_t) Hook, HookData);
	return GrubErrToEFIStatus(rc);
}

EFI_STATUS
GrubOpen(EFI_GRUB_FILE *File)
{
	grub_fs_t p = grub_fs_list;
	grub_file_t f = (grub_file_t) File->GrubFile;
	grub_err_t rc;

	grub_errno = 0;
	rc = p->fs_open(f, File->path);
	return GrubErrToEFIStatus(rc);
}

VOID
GrubClose(EFI_GRUB_FILE *File)
{
	grub_fs_t p = grub_fs_list;
	grub_file_t f = (grub_file_t) File->GrubFile;

	grub_errno = 0;
	p->fs_close(f);
}

EFI_STATUS
GrubRead(EFI_GRUB_FILE *File, VOID *Data, UINTN *Len)
{
	grub_fs_t p = grub_fs_list;
	grub_file_t f = (grub_file_t) File->GrubFile;
	grub_ssize_t len;
	UINTN Remaining;

	/* GRUB may return an error if we request more data than available */
	Remaining = (UINTN)((f->size > f->offset)? f->size - f->offset : 0);

	if (*Len > Remaining)
		*Len = Remaining;

	grub_errno = 0;
	len = p->fs_read(f, (char *) Data, (grub_size_t) *Len);

	if (len < 0) {
		*Len = 0;
		return GrubErrToEFIStatus(grub_errno);
	}

	/* You'd think that GRUB read() would increase the offset... */
	f->offset += len;
	*Len = (UINTN)len;

	return EFI_SUCCESS;
}

EFI_STATUS
GrubLabel(EFI_GRUB_FILE *File, CHAR8 **label)
{
	grub_fs_t p = grub_fs_list;
	grub_file_t f = (grub_file_t) File->GrubFile;
	grub_err_t rc;

	grub_errno = 0;
	rc = p->fs_label(f->device, (char **) label);
	return GrubErrToEFIStatus(rc);
}

/* Helper for GrubFSProbe.  */
static int
probe_dummy_iter (const char *filename __attribute__ ((unused)),
		const struct grub_dirhook_info *info __attribute__ ((unused)),
		void *data __attribute__ ((unused)))
{
	return 1;
}

BOOLEAN
GrubFSProbe(EFI_FS *FileSystem)
{
	grub_fs_t p = grub_fs_list;
	grub_device_t device = (grub_device_t) FileSystem->GrubDevice;

	if ((p == NULL) || (device->disk == NULL)) {
		PrintError(L"GrubFSProbe: uninitialized variables\n");
		return FALSE;
	}

	grub_errno = 0;
	(p->fs_dir)(device, "/", probe_dummy_iter, NULL);
	if (grub_errno != 0) {
		if (LogLevel >= FS_LOGLEVEL_INFO)
			grub_print_error();	/* NB: this call will reset grub_errno */
		return FALSE;
	}
	return TRUE;
}

CHAR16 *
GrubGetUuid(EFI_FS* FileSystem)
{
	EFI_STATUS Status;
	grub_fs_t p = grub_fs_list;
	grub_device_t device = (grub_device_t) FileSystem->GrubDevice;
	static CHAR16 Uuid[36];
	char* uuid;

	if (p->fs_uuid(device, &uuid) || (uuid == NULL))
		return NULL;

	Status = Utf8ToUtf16NoAlloc(uuid, Uuid, ARRAYSIZE(Uuid));
	if (EFI_ERROR(Status)) {
		PrintStatusError(Status, L"Could not convert UUID to UTF-16");
		return NULL;
	}

	return Uuid;
}
