"""
Use `Varstack <https://github.com/conversis/varstack>`_ data as a Pillar source

Configuring Varstack
====================

Using varstack in Salt is fairly simple. Just put the following into the
config file of your master:

.. code-block:: yaml

    ext_pillar:
      - varstack: /etc/varstack.yaml

Varstack will then use /etc/varstack.yaml to determine which configuration
data to return as pillar information. From there you can take a look at the
`README <https://github.com/conversis/varstack/blob/master/README.md>`_ of
varstack on how this file is evaluated.
"""
try:
    import varstack
except ImportError:
    varstack = None
__virtualname__ = 'varstack'

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    return (varstack and __virtualname__ or False, 'The varstack module could not be loaded: varstack dependency is missing.')

def ext_pillar(minion_id, pillar, conf):
    if False:
        for i in range(10):
            print('nop')
    '\n    Parse varstack data and return the result\n    '
    vs = varstack.Varstack(config_filename=conf)
    return vs.evaluate(__grains__)