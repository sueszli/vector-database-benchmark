"""
Management of Docker volumes

.. versionadded:: 2017.7.0

:depends: docker_ Python module

.. note::
    Older releases of the Python bindings for Docker were called docker-py_ in
    PyPI. All releases of docker_, and releases of docker-py_ >= 1.6.0 are
    supported. These python bindings can easily be installed using
    :py:func:`pip.install <salt.modules.pip.install>`:

    .. code-block:: bash

        salt myminion pip.install docker

    To upgrade from docker-py_ to docker_, you must first uninstall docker-py_,
    and then install docker_:

    .. code-block:: bash

        salt myminion pip.uninstall docker-py
        salt myminion pip.install docker

.. _docker: https://pypi.python.org/pypi/docker
.. _docker-py: https://pypi.python.org/pypi/docker-py

These states were moved from the :mod:`docker <salt.states.docker>` state
module (formerly called **dockerng**) in the 2017.7.0 release.
"""
import logging
import salt.utils.data
log = logging.getLogger(__name__)
__virtualname__ = 'docker_volume'
__virtual_aliases__ = ('moby_volume',)
__deprecated__ = (3009, 'docker', 'https://github.com/saltstack/saltext-docker')

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only load if the docker execution module is available\n    '
    if 'docker.version' in __salt__:
        return __virtualname__
    return (False, __salt__.missing_fun_string('docker.version'))

def _find_volume(name):
    if False:
        while True:
            i = 10
    '\n    Find volume by name on minion\n    '
    docker_volumes = __salt__['docker.volumes']()['Volumes']
    if docker_volumes:
        volumes = [v for v in docker_volumes if v['Name'] == name]
        if volumes:
            return volumes[0]
    return None

def present(name, driver=None, driver_opts=None, force=False):
    if False:
        i = 10
        return i + 15
    "\n    Ensure that a volume is present.\n\n    .. versionadded:: 2015.8.4\n    .. versionchanged:: 2015.8.6\n        This state no longer deletes and re-creates a volume if the existing\n        volume's driver does not match the ``driver`` parameter (unless the\n        ``force`` parameter is set to ``True``).\n    .. versionchanged:: 2017.7.0\n        This state was renamed from **docker.volume_present** to **docker_volume.present**\n\n    name\n        Name of the volume\n\n    driver\n        Type of driver for that volume.  If ``None`` and the volume\n        does not yet exist, the volume will be created using Docker's\n        default driver.  If ``None`` and the volume does exist, this\n        function does nothing, even if the existing volume's driver is\n        not the Docker default driver.  (To ensure that an existing\n        volume's driver matches the Docker default, you must\n        explicitly name Docker's default driver here.)\n\n    driver_opts\n        Options for the volume driver\n\n    force : False\n        If the volume already exists but the existing volume's driver\n        does not match the driver specified by the ``driver``\n        parameter, this parameter controls whether the function errors\n        out (if ``False``) or deletes and re-creates the volume (if\n        ``True``).\n\n        .. versionadded:: 2015.8.6\n\n    Usage Examples:\n\n    .. code-block:: yaml\n\n        volume_foo:\n          docker_volume.present\n\n\n    .. code-block:: yaml\n\n        volume_bar:\n          docker_volume.present\n            - name: bar\n            - driver: local\n            - driver_opts:\n                foo: bar\n\n    .. code-block:: yaml\n\n        volume_bar:\n          docker_volume.present\n            - name: bar\n            - driver: local\n            - driver_opts:\n                - foo: bar\n                - option: value\n\n    "
    ret = {'name': name, 'changes': {}, 'result': False, 'comment': ''}
    if salt.utils.data.is_dictlist(driver_opts):
        driver_opts = salt.utils.data.repack_dictlist(driver_opts)
    volume = _find_volume(name)
    if not volume:
        if __opts__['test']:
            ret['result'] = None
            ret['comment'] = f"The volume '{name}' will be created"
            return ret
        try:
            ret['changes']['created'] = __salt__['docker.create_volume'](name, driver=driver, driver_opts=driver_opts)
        except Exception as exc:
            ret['comment'] = f"Failed to create volume '{name}': {exc}"
            return ret
        else:
            result = True
            ret['result'] = result
            return ret
    if driver is not None and volume['Driver'] != driver:
        if not force:
            ret['comment'] = "Driver for existing volume '{}' ('{}') does not match specified driver ('{}') and force is False".format(name, volume['Driver'], driver)
            ret['result'] = None if __opts__['test'] else False
            return ret
        if __opts__['test']:
            ret['result'] = None
            ret['comment'] = "The volume '{}' will be replaced with a new one using the driver '{}'".format(name, volume)
            return ret
        try:
            ret['changes']['removed'] = __salt__['docker.remove_volume'](name)
        except Exception as exc:
            ret['comment'] = f"Failed to remove volume '{name}': {exc}"
            return ret
        else:
            try:
                ret['changes']['created'] = __salt__['docker.create_volume'](name, driver=driver, driver_opts=driver_opts)
            except Exception as exc:
                ret['comment'] = f"Failed to create volume '{name}': {exc}"
                return ret
            else:
                result = True
                ret['result'] = result
                return ret
    ret['result'] = True
    ret['comment'] = f"Volume '{name}' already exists."
    return ret

def absent(name, driver=None):
    if False:
        i = 10
        return i + 15
    '\n    Ensure that a volume is absent.\n\n    .. versionadded:: 2015.8.4\n    .. versionchanged:: 2017.7.0\n        This state was renamed from **docker.volume_absent** to **docker_volume.absent**\n\n    name\n        Name of the volume\n\n    Usage Examples:\n\n    .. code-block:: yaml\n\n        volume_foo:\n          docker_volume.absent\n\n    '
    ret = {'name': name, 'changes': {}, 'result': False, 'comment': ''}
    volume = _find_volume(name)
    if not volume:
        ret['result'] = True
        ret['comment'] = f"Volume '{name}' already absent"
        return ret
    try:
        ret['changes']['removed'] = __salt__['docker.remove_volume'](name)
        ret['result'] = True
    except Exception as exc:
        ret['comment'] = f"Failed to remove volume '{name}': {exc}"
    return ret