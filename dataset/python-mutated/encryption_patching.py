_torch_encryption_patch = None
is_encryption_patched = False

def patch_encryption():
    if False:
        print('Hello World!')
    '\n    patch_torch is used to patch torch.save and torch.load methods to replace original ones.\n\n    Patched details include:\n\n    | 1. torch.save is now located at bigdl.nano.pytorch.encryption.save\n    | 2. torch.load is now located at bigdl.nano.pytorch.encryption.load\n\n    A key argument is added to torch.save and torch.load which is used to\n    encrypt/decrypt the content before saving/loading it to/from disk.\n\n    .. note::\n\n       Please be noted that the key is only secured in Intel SGX mode.\n    '
    global is_encryption_patched
    if is_encryption_patched:
        return
    mapping_torch = _get_encryption_patch_map()
    for mapping_iter in mapping_torch:
        setattr(mapping_iter[0], mapping_iter[1], mapping_iter[2])
    is_encryption_patched = True

def _get_encryption_patch_map():
    if False:
        print('Hello World!')
    global _torch_encryption_patch
    import torch
    from bigdl.nano.pytorch.encryption import save, load
    _torch_encryption_patch = []
    _torch_encryption_patch += [[torch, 'old_save', torch.save], [torch, 'old_load', torch.load], [torch, 'save', save], [torch, 'load', load]]
    return _torch_encryption_patch