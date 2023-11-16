import logging
import subprocess
import utils

def list_devices():
    if False:
        i = 10
        return i + 15
    logging.debug('listing MMAL devices')
    try:
        binary = subprocess.check_output(['which', 'vcgencmd'], stderr=utils.DEV_NULL).strip()
    except subprocess.CalledProcessError:
        return []
    try:
        support = subprocess.check_output([binary, 'get_camera']).strip()
    except subprocess.CalledProcessError:
        return []
    if support.startswith('supported=1 detected=1'):
        logging.debug('MMAL camera detected')
        return [('vc.ril.camera', 'VideoCore Camera')]
    return []