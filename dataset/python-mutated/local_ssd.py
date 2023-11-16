from google.cloud import compute_v1

def local_ssd_disk(zone: str) -> compute_v1.AttachedDisk():
    if False:
        return 10
    '\n    Create an AttachedDisk object to be used in VM instance creation. The created disk contains\n    no data and requires formatting before it can be used.\n\n    Args:\n        zone: The zone in which the local SSD drive will be attached.\n\n    Returns:\n        AttachedDisk object configured as a local SSD disk.\n    '
    disk = compute_v1.AttachedDisk()
    disk.type_ = compute_v1.AttachedDisk.Type.SCRATCH.name
    initialize_params = compute_v1.AttachedDiskInitializeParams()
    initialize_params.disk_type = f'zones/{zone}/diskTypes/local-ssd'
    disk.initialize_params = initialize_params
    disk.auto_delete = True
    return disk