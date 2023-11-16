from google.cloud import compute_v1

def disk_from_image(disk_type: str, disk_size_gb: int, boot: bool, source_image: str, auto_delete: bool=True) -> compute_v1.AttachedDisk:
    if False:
        print('Hello World!')
    '\n    Create an AttachedDisk object to be used in VM instance creation. Uses an image as the\n    source for the new disk.\n\n    Args:\n         disk_type: the type of disk you want to create. This value uses the following format:\n            "zones/{zone}/diskTypes/(pd-standard|pd-ssd|pd-balanced|pd-extreme)".\n            For example: "zones/us-west3-b/diskTypes/pd-ssd"\n        disk_size_gb: size of the new disk in gigabytes\n        boot: boolean flag indicating whether this disk should be used as a boot disk of an instance\n        source_image: source image to use when creating this disk. You must have read access to this disk. This can be one\n            of the publicly available images or an image from one of your projects.\n            This value uses the following format: "projects/{project_name}/global/images/{image_name}"\n        auto_delete: boolean flag indicating whether this disk should be deleted with the VM that uses it\n\n    Returns:\n        AttachedDisk object configured to be created using the specified image.\n    '
    boot_disk = compute_v1.AttachedDisk()
    initialize_params = compute_v1.AttachedDiskInitializeParams()
    initialize_params.source_image = source_image
    initialize_params.disk_size_gb = disk_size_gb
    initialize_params.disk_type = disk_type
    boot_disk.initialize_params = initialize_params
    boot_disk.auto_delete = auto_delete
    boot_disk.boot = boot
    return boot_disk