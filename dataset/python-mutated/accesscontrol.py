import os

def fs_standard_access(attr, path, volume):
    if False:
        i = 10
        return i + 15
    '\n    Make dotfiles not readable, not writable, hidden and locked.\n    Should return None to allow for original attribute value, boolean otherwise.\n    This can be used in the :ref:`setting-accessControl` setting.\n    \n    Args:\n        :attr: One of `read`, `write`, `hidden` and `locked`.\n        :path: The path to check against.\n        :volume: The volume responsible for managing the path.\n\n    Returns:\n        ``True`` if `path` has `attr` permissions, ``False`` if not and\n        ``None`` to apply the default permission rules.\n    '
    if os.path.basename(path) in ['.tmb', '.quarantine']:
        return None
    if volume.name() == 'localfilesystem':
        if attr in ['read', 'write'] and os.path.basename(path).startswith('.'):
            return False
        elif attr in ['hidden', 'locked'] and os.path.basename(path).startswith('.'):
            return True