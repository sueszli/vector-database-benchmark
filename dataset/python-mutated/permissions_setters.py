import os
import spack.util.file_permissions as fp

def post_install(spec, explicit=None):
    if False:
        i = 10
        return i + 15
    if not spec.external:
        fp.set_permissions_by_spec(spec.prefix, spec)
        for (root, dirs, files) in os.walk(spec.prefix, followlinks=False):
            for d in dirs:
                if not os.path.islink(os.path.join(root, d)):
                    fp.set_permissions_by_spec(os.path.join(root, d), spec)
            for f in files:
                if not os.path.islink(os.path.join(root, f)):
                    fp.set_permissions_by_spec(os.path.join(root, f), spec)