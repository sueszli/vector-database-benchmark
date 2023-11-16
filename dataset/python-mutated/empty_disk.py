from google.cloud import compute_v1

def empty_disk(disk_type: str, disk_size_gb: int, boot: bool=False, auto_delete: bool=True) -> compute_v1.AttachedDisk():
    if False:
        i = 10
        return i + 15
    '\n    Create an AttachedDisk object to be used in VM instance creation. The created disk contains\n    no data and requires formatting before it can be used.\n\n    Args:\n         disk_type: the type of disk you want to create. This value uses the following format:\n            "zones/{zone}/diskTypes/(pd-standard|pd-ssd|pd-balanced|pd-extreme)".\n            For example: "zones/us-west3-b/diskTypes/pd-ssd"\n        disk_size_gb: size of the new disk in gigabytes\n        boot: boolean flag indicating whether this disk should be used as a boot disk of an instance\n        auto_delete: boolean flag indicating whether this disk should be deleted with the VM that uses it\n\n    Returns:\n        AttachedDisk object configured to be created as an empty disk.\n    '
    disk = compute_v1.AttachedDisk()
    initialize_params = compute_v1.AttachedDiskInitializeParams()
    initialize_params.disk_type = disk_type
    initialize_params.disk_size_gb = disk_size_gb
    disk.initialize_params = initialize_params
    disk.auto_delete = auto_delete
    disk.boot = boot
    return disk