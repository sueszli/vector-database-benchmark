"""
Management of Docker images

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

.. note::
    To pull from a Docker registry, authentication must be configured. See
    :ref:`here <docker-authentication>` for more information on how to
    configure access to docker registries in :ref:`Pillar <pillar>` data.
"""
import logging
import salt.utils.args
import salt.utils.dockermod
from salt.exceptions import CommandExecutionError
log = logging.getLogger(__name__)
__virtualname__ = 'docker_image'
__virtual_aliases__ = ('moby_image',)
__deprecated__ = (3009, 'docker', 'https://github.com/saltstack/saltext-docker')

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only load if the docker execution module is available\n    '
    if 'docker.version' in __salt__:
        return __virtualname__
    return (False, __salt__.missing_fun_string('docker.version'))

def present(name, tag=None, build=None, load=None, force=False, insecure_registry=False, client_timeout=salt.utils.dockermod.CLIENT_TIMEOUT, dockerfile=None, sls=None, base='opensuse/python', saltenv='base', pillarenv=None, pillar=None, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    .. versionchanged:: 2018.3.0\n        The ``tag`` argument has been added. It is now required unless pulling\n        from a registry.\n\n    Ensure that an image is present. The image can either be pulled from a\n    Docker registry, built from a Dockerfile, loaded from a saved image, or\n    built by running SLS files against a base image.\n\n    If none of the ``build``, ``load``, or ``sls`` arguments are used, then Salt\n    will pull from the :ref:`configured registries <docker-authentication>`. If\n    the specified image already exists, it will not be pulled unless ``force``\n    is set to ``True``. Here is an example of a state that will pull an image\n    from the Docker Hub:\n\n    .. code-block:: yaml\n\n        myuser/myimage:\n          docker_image.present:\n            - tag: mytag\n\n    name\n        The name of the docker image.\n\n    tag\n        Tag name for the image. Required when using ``build``, ``load``, or\n        ``sls`` to create the image, but optional if pulling from a repository.\n\n        .. versionadded:: 2018.3.0\n\n    build\n        Path to directory on the Minion containing a Dockerfile\n\n        .. code-block:: yaml\n\n            myuser/myimage:\n              docker_image.present:\n                - build: /home/myuser/docker/myimage\n                - tag: mytag\n\n            myuser/myimage:\n              docker_image.present:\n                - build: /home/myuser/docker/myimage\n                - tag: mytag\n                - dockerfile: Dockerfile.alternative\n\n        The image will be built using :py:func:`docker.build\n        <salt.modules.dockermod.build>` and the specified image name and tag\n        will be applied to it.\n\n        .. versionadded:: 2016.11.0\n        .. versionchanged:: 2018.3.0\n            The ``tag`` must be manually specified using the ``tag`` argument.\n\n    load\n        Loads a tar archive created with :py:func:`docker.save\n        <salt.modules.dockermod.save>` (or the ``docker save`` Docker CLI\n        command), and assigns it the specified repo and tag.\n\n        .. code-block:: yaml\n\n            myuser/myimage:\n              docker_image.present:\n                - load: salt://path/to/image.tar\n                - tag: mytag\n\n        .. versionchanged:: 2018.3.0\n            The ``tag`` must be manually specified using the ``tag`` argument.\n\n    force\n        Set this parameter to ``True`` to force Salt to pull/build/load the\n        image even if it is already present.\n\n    insecure_registry\n        If ``True``, the Docker client will permit the use of insecure\n        (non-HTTPS) registries.\n\n    client_timeout\n        Timeout in seconds for the Docker client. This is not a timeout for\n        the state, but for receiving a response from the API.\n\n    dockerfile\n        Allows for an alternative Dockerfile to be specified.  Path to alternative\n        Dockefile is relative to the build path for the Docker container.\n\n        .. versionadded:: 2016.11.0\n\n    sls\n        Allow for building of image with :py:func:`docker.sls_build\n        <salt.modules.dockermod.sls_build>` by specifying the SLS files with\n        which to build. This can be a list or comma-separated string.\n\n        .. code-block:: yaml\n\n            myuser/myimage:\n              docker_image.present:\n                - tag: latest\n                - sls:\n                    - webapp1\n                    - webapp2\n                - base: centos\n                - saltenv: base\n\n        .. versionadded:: 2017.7.0\n        .. versionchanged:: 2018.3.0\n            The ``tag`` must be manually specified using the ``tag`` argument.\n\n    base\n        Base image with which to start :py:func:`docker.sls_build\n        <salt.modules.dockermod.sls_build>`\n\n        .. versionadded:: 2017.7.0\n\n    saltenv\n        Specify the environment from which to retrieve the SLS indicated by the\n        `mods` parameter.\n\n        .. versionadded:: 2017.7.0\n        .. versionchanged:: 2018.3.0\n            Now uses the effective saltenv if not explicitly passed. In earlier\n            versions, ``base`` was assumed as a default.\n\n    pillarenv\n        Specify a Pillar environment to be used when applying states. This\n        can also be set in the minion config file using the\n        :conf_minion:`pillarenv` option. When neither the\n        :conf_minion:`pillarenv` minion config option nor this CLI argument is\n        used, all Pillar environments will be merged together.\n\n        .. versionadded:: 2018.3.0\n\n    pillar\n        Custom Pillar values, passed as a dictionary of key-value pairs\n\n        .. note::\n            Values passed this way will override Pillar values set via\n            ``pillar_roots`` or an external Pillar source.\n\n        .. versionadded:: 2018.3.0\n\n    kwargs\n        Additional keyword arguments to pass to\n        :py:func:`docker.build <salt.modules.dockermod.build>`\n    '
    ret = {'name': name, 'changes': {}, 'result': False, 'comment': ''}
    if not isinstance(name, str):
        name = str(name)
    num_build_args = len([x for x in (build, load, sls) if x is not None])
    if num_build_args > 1:
        ret['comment'] = "Only one of 'build', 'load', or 'sls' is permitted."
        return ret
    elif num_build_args == 1:
        if not tag:
            ret['comment'] = "The 'tag' argument is required if any one of 'build', 'load', or 'sls' is used."
            return ret
        if not isinstance(tag, str):
            tag = str(tag)
        full_image = ':'.join((name, tag))
    else:
        if tag:
            name = f'{name}:{tag}'
        full_image = name
    try:
        image_info = __salt__['docker.inspect_image'](full_image)
    except CommandExecutionError as exc:
        msg = exc.__str__()
        if '404' in msg:
            image_info = None
        else:
            ret['comment'] = msg
            return ret
    if image_info is not None:
        if not force:
            ret['result'] = True
            ret['comment'] = f'Image {full_image} already present'
            return ret
    if build or sls:
        action = 'built'
    elif load:
        action = 'loaded'
    else:
        action = 'pulled'
    if __opts__['test']:
        ret['result'] = None
        if image_info is not None and force or image_info is None:
            ret['comment'] = f'Image {full_image} will be {action}'
            return ret
    if build:
        argspec = salt.utils.args.get_function_argspec(__salt__['docker.build'])
        build_args = dict(list(zip(argspec.args, argspec.defaults)))
        for k in build_args:
            if k in kwargs.get('kwargs', {}):
                build_args[k] = kwargs.get('kwargs', {}).get(k)
        try:
            build_args['path'] = build
            build_args['repository'] = name
            build_args['tag'] = tag
            build_args['dockerfile'] = dockerfile
            image_update = __salt__['docker.build'](**build_args)
        except Exception as exc:
            ret['comment'] = 'Encountered error building {} as {}: {}'.format(build, full_image, exc)
            return ret
        if image_info is None or image_update['Id'] != image_info['Id'][:12]:
            ret['changes'] = image_update
    elif sls:
        _locals = locals()
        sls_build_kwargs = {k: _locals[k] for k in ('saltenv', 'pillarenv', 'pillar') if _locals[k] is not None}
        try:
            image_update = __salt__['docker.sls_build'](repository=name, tag=tag, base=base, mods=sls, **sls_build_kwargs)
        except Exception as exc:
            ret['comment'] = 'Encountered error using SLS {} for building {}: {}'.format(sls, full_image, exc)
            return ret
        if image_info is None or image_update['Id'] != image_info['Id'][:12]:
            ret['changes'] = image_update
    elif load:
        try:
            image_update = __salt__['docker.load'](path=load, repository=name, tag=tag)
        except Exception as exc:
            ret['comment'] = 'Encountered error loading {} as {}: {}'.format(load, full_image, exc)
            return ret
        if image_info is None or image_update.get('Layers', []):
            ret['changes'] = image_update
    else:
        try:
            image_update = __salt__['docker.pull'](name, insecure_registry=insecure_registry, client_timeout=client_timeout)
        except Exception as exc:
            ret['comment'] = f'Encountered error pulling {full_image}: {exc}'
            return ret
        if image_info is not None and image_info['Id'][:12] == image_update.get('Layers', {}).get('Already_Pulled', [None])[0]:
            pass
        elif image_info is None or image_update.get('Layers', {}).get('Pulled'):
            ret['changes'] = image_update
    error = False
    try:
        __salt__['docker.inspect_image'](full_image)
    except CommandExecutionError as exc:
        msg = exc.__str__()
        if '404' not in msg:
            error = "Failed to inspect image '{}' after it was {}: {}".format(full_image, action, msg)
    if error:
        ret['comment'] = error
    else:
        ret['result'] = True
        if not ret['changes']:
            ret['comment'] = "Image '{}' was {}, but there were no changes".format(name, action)
        else:
            ret['comment'] = f"Image '{full_image}' was {action}"
    return ret

def absent(name=None, images=None, force=False):
    if False:
        print('Hello World!')
    '\n    Ensure that an image is absent from the Minion. Image names can be\n    specified either using ``repo:tag`` notation, or just the repo name (in\n    which case a tag of ``latest`` is assumed).\n\n    name\n        The name of the docker image.\n\n    images\n        Run this state on more than one image at a time. The following two\n        examples accomplish the same thing:\n\n        .. code-block:: yaml\n\n            remove_images:\n              docker_image.absent:\n                - names:\n                  - busybox\n                  - centos:6\n                  - nginx\n\n        .. code-block:: yaml\n\n            remove_images:\n              docker_image.absent:\n                - images:\n                  - busybox\n                  - centos:6\n                  - nginx\n\n        However, the second example will be a bit quicker since Salt will do\n        all the deletions in a single run, rather than executing the state\n        separately on each image (as it would in the first example).\n\n    force\n        Salt will fail to remove any images currently in use by a container.\n        Set this option to true to remove the image even if it is already\n        present.\n\n        .. note::\n\n            This option can also be overridden by Pillar data. If the Minion\n            has a pillar variable named ``docker.running.force`` which is\n            set to ``True``, it will turn on this option. This pillar variable\n            can even be set at runtime. For example:\n\n            .. code-block:: bash\n\n                salt myminion state.sls docker_stuff pillar="{docker.force: True}"\n\n            If this pillar variable is present and set to ``False``, then it\n            will turn off this option.\n\n            For more granular control, setting a pillar variable named\n            ``docker.force.image_name`` will affect only the named image.\n    '
    ret = {'name': name, 'changes': {}, 'result': False, 'comment': ''}
    if not name and (not images):
        ret['comment'] = "One of 'name' and 'images' must be provided"
        return ret
    elif images is not None:
        targets = images
    elif name:
        targets = [name]
    to_delete = []
    for target in targets:
        resolved_tag = __salt__['docker.resolve_tag'](target)
        if resolved_tag is not False:
            to_delete.append(resolved_tag)
    if not to_delete:
        ret['result'] = True
        if len(targets) == 1:
            ret['comment'] = f'Image {name} is not present'
        else:
            ret['comment'] = 'All specified images are not present'
        return ret
    if __opts__['test']:
        ret['result'] = None
        if len(to_delete) == 1:
            ret['comment'] = f'Image {to_delete[0]} will be removed'
        else:
            ret['comment'] = 'The following images will be removed: {}'.format(', '.join(to_delete))
        return ret
    result = __salt__['docker.rmi'](*to_delete, force=force)
    post_tags = __salt__['docker.list_tags']()
    failed = [x for x in to_delete if x in post_tags]
    if failed:
        if [x for x in to_delete if x not in post_tags]:
            ret['changes'] = result
            ret['comment'] = 'The following image(s) failed to be removed: {}'.format(', '.join(failed))
        else:
            ret['comment'] = 'None of the specified images were removed'
            if 'Errors' in result:
                ret['comment'] += '. The following errors were encountered: {}'.format('; '.join(result['Errors']))
    else:
        ret['changes'] = result
        if len(to_delete) == 1:
            ret['comment'] = f'Image {to_delete[0]} was removed'
        else:
            ret['comment'] = 'The following images were removed: {}'.format(', '.join(to_delete))
        ret['result'] = True
    return ret

def mod_watch(name, sfun=None, **kwargs):
    if False:
        return 10
    '\n    The docker_image  watcher, called to invoke the watch command.\n\n    .. note::\n        This state exists to support special handling of the ``watch``\n        :ref:`requisite <requisites>`. It should not be called directly.\n\n        Parameters for this function should be set by the state being triggered.\n    '
    if sfun == 'present':
        kwargs['force'] = True
        return present(name, **kwargs)
    return {'name': name, 'changes': {}, 'result': False, 'comment': f'watch requisite is not implemented for {sfun}'}