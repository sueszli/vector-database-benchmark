"""
Basic support to use a --dry-run mode w/ invoke tasks.

.. code-block::

    from ._dry_run import DryRunContext

    @task
    def destroy_something(ctx, path, dry_run=False):
        if dry_run:
            ctx = DryRunContext(ctx)

        # -- DRY-RUN MODE: Only echos commands.
        ctx.run("rm -rf {}".format(path))
"""
from __future__ import print_function

class DryRunContext(object):
    PREFIX = 'DRY-RUN: '
    SCHEMA = '{prefix}{command}'
    SCHEMA_WITH_KWARGS = '{prefix}{command} (with kwargs={kwargs})'

    def __init__(self, ctx=None, prefix=None, schema=None):
        if False:
            return 10
        if prefix is None:
            prefix = self.PREFIX
        if schema is None:
            schema = self.SCHEMA
        self.ctx = ctx
        self.prefix = prefix
        self.schema = schema

    def run(self, command, **kwargs):
        if False:
            while True:
                i = 10
        message = self.schema.format(command=command, prefix=self.prefix, kwargs=kwargs)
        print(message)

    def sudo(self, command, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        command2 = 'sudo %s' % command
        self.run(command2, **kwargs)