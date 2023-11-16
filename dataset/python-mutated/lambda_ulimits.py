import resource

def handler(event, context):
    if False:
        print('Hello World!')
    ulimit_names = {'RLIMIT_AS': resource.RLIMIT_AS, 'RLIMIT_CORE': resource.RLIMIT_CORE, 'RLIMIT_CPU': resource.RLIMIT_CPU, 'RLIMIT_DATA': resource.RLIMIT_DATA, 'RLIMIT_FSIZE': resource.RLIMIT_FSIZE, 'RLIMIT_MEMLOCK': resource.RLIMIT_MEMLOCK, 'RLIMIT_NOFILE': resource.RLIMIT_NOFILE, 'RLIMIT_NPROC': resource.RLIMIT_NPROC, 'RLIMIT_RSS': resource.RLIMIT_RSS, 'RLIMIT_STACK': resource.RLIMIT_STACK}
    return {label: resource.getrlimit(res) for (label, res) in ulimit_names.items()}