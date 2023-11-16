import click
from warehouse.cli import warehouse
from warehouse.packaging.tasks import reconcile_file_storages as _reconcile_file_storages

@warehouse.group()
def storages():
    if False:
        for i in range(10):
            print('nop')
    '\n    Manage the Warehouse Storages.\n    '

@storages.command()
@click.pass_obj
@click.option('--batch-size', type=int, default=1)
def reconcile(config, batch_size):
    if False:
        print('Hello World!')
    '\n    Run the storage reconciliation task as a one-off\n    '
    request = config.task(_reconcile_file_storages).get_request()
    request.registry.settings['reconcile_file_storages.batch_size'] = batch_size
    config.task(_reconcile_file_storages).run(request)