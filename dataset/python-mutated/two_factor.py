import click
from warehouse.cli import warehouse
from warehouse.packaging.tasks import compute_2fa_mandate as _compute_2fa_mandate

@warehouse.command()
@click.pass_obj
def compute_2fa_mandate(config):
    if False:
        for i in range(10):
            print('nop')
    '\n    Run a one-off computation of the 2FA-mandated projects\n    '
    request = config.task(_compute_2fa_mandate).get_request()
    config.task(_compute_2fa_mandate).run(request)