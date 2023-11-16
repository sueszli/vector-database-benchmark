"""
Management of OpenStack Glance Images
========================================

.. versionadded:: 2018.3.0

:depends: shade
:configuration: see :py:mod:`salt.modules.glanceng` for setup instructions

Example States

.. code-block:: yaml

    create image:
      glance_image.present:
        - name: cirros
        - filename: cirros.raw
        - image_format: raw

    delete image:
      glance_image.absent:
        - name: cirros
"""
__virtualname__ = 'glance_image'

def __virtual__():
    if False:
        print('Hello World!')
    if 'glanceng.image_get' in __salt__:
        return __virtualname__
    return (False, 'The glanceng execution module failed to load: shade python module is not available')

def present(name, auth=None, **kwargs):
    if False:
        print('Hello World!')
    '\n    Ensure image exists and is up-to-date\n\n    name\n        Name of the image\n\n    enabled\n        Boolean to control if image is enabled\n\n    description\n        An arbitrary description of the image\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': ''}
    kwargs = __utils__['args.clean_kwargs'](**kwargs)
    __salt__['glanceng.setup_clouds'](auth)
    image = __salt__['glanceng.image_get'](name=name)
    if not image:
        if __opts__['test']:
            ret['result'] = None
            ret['changes'] = kwargs
            ret['comment'] = 'Image {} will be created.'.format(name)
            return ret
        kwargs['name'] = name
        image = __salt__['glanceng.image_create'](**kwargs)
        ret['changes'] = image
        ret['comment'] = 'Created image'
        return ret
    return ret

def absent(name, auth=None):
    if False:
        while True:
            i = 10
    '\n    Ensure image does not exist\n\n    name\n        Name of the image\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': ''}
    __salt__['glanceng.setup_clouds'](auth)
    image = __salt__['glanceng.image_get'](name=name)
    if image:
        if __opts__['test']:
            ret['result'] = None
            ret['changes'] = {'name': name}
            ret['comment'] = 'Image {} will be deleted.'.format(name)
            return ret
        __salt__['glanceng.image_delete'](name=image)
        ret['changes']['id'] = image.id
        ret['comment'] = 'Deleted image'
    return ret