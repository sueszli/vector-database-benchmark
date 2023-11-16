from importlib import import_module
from elfinder.conf import settings as ls

def get_path_driver(hash_, optionset):
    if False:
        print('Hello World!')
    '\n    Given an ``optionset`` and a path ``hash_`` this function returns\n    a mounted volume driver for this path.\n    \n    This method assumes that the driver uses the default driver \n    :func:`elfinder.volumes.base.ElfinderVolumeDriver.id` implementation\n    to generate its id.\n    '
    for root_options in ls.ELFINDER_CONNECTOR_OPTION_SETS[optionset]['roots']:
        if 'driver' in root_options:
            if hash_.startswith('%s%s_' % (root_options['driver']._driver_id, root_options['id'])):
                return instantiate_driver(root_options)

def instantiate_driver(root_options):
    if False:
        print('Hello World!')
    '\n    Instantiate and return a  driver, given its ``root_options``.\n    '
    class_ = root_options['driver'] if 'driver' in root_options else ''
    if 'driverInstance' in root_options and isinstance(root_options['driverInstance'], class_):
        return root_options['driverInstance']
    if isinstance(class_, basestring) and class_:
        try:
            split = class_.split('.')
            storage_module = import_module('.'.join(split[:-1]))
            volume = getattr(storage_module, split[-1])()
        except:
            raise Exception('Could not import driver "%s"' % class_)
    else:
        try:
            volume = class_()
        except TypeError:
            raise Exception('Driver "%s" does not exist' % class_)
    try:
        volume.mount(root_options)
    except Exception as e:
        raise Exception('Driver "%s" " %s' % (class_, e))
    if 'keepAlive' in root_options and root_options['keepAlive']:
        root_options['driverInstance'] = volume
    return volume