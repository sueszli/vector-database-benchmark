import spack.verify

def post_install(spec, explicit=None):
    if False:
        print('Hello World!')
    if not spec.external:
        spack.verify.write_manifest(spec)