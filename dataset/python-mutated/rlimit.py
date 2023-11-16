import logging
import resource
logger = logging.getLogger('synapse.app.homeserver')

def change_resource_limit(soft_file_no: int) -> None:
    if False:
        for i in range(10):
            print('nop')
    try:
        (soft, hard) = resource.getrlimit(resource.RLIMIT_NOFILE)
        if not soft_file_no:
            soft_file_no = hard
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft_file_no, hard))
        logger.info('Set file limit to: %d', soft_file_no)
        resource.setrlimit(resource.RLIMIT_CORE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    except (ValueError, resource.error) as e:
        logger.warning('Failed to set file or core limit: %s', e)