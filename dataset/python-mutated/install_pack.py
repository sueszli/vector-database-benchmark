import sys
from oslo_config import cfg
from st2common import config
from st2common import log as logging
from st2common.config import do_register_cli_opts
from st2common.script_setup import setup as common_setup
from st2common.util.pack_management import download_pack
from st2common.util.pack_management import get_and_set_proxy_config
from st2common.util.virtualenvs import setup_pack_virtualenv
__all__ = ['main']
LOG = logging.getLogger(__name__)

def _register_cli_opts():
    if False:
        for i in range(10):
            print('nop')
    cli_opts = [cfg.MultiStrOpt('pack', default=None, required=True, positional=True, help='Name of the pack to install.'), cfg.BoolOpt('verify-ssl', default=True, help='Verify SSL certificate of the Git repo from which the pack is downloaded.'), cfg.BoolOpt('force', default=False, help='True to force pack installation and ignore install lock file if it exists.')]
    do_register_cli_opts(cli_opts)

def main(argv):
    if False:
        i = 10
        return i + 15
    _register_cli_opts()
    common_setup(config=config, setup_db=False, register_mq_exchanges=False, register_internal_trigger_types=False)
    packs = cfg.CONF.pack
    verify_ssl = cfg.CONF.verify_ssl
    force = cfg.CONF.force
    proxy_config = get_and_set_proxy_config()
    for pack in packs:
        LOG.info('Installing pack "%s"' % pack)
        result = download_pack(pack=pack, verify_ssl=verify_ssl, force=force, proxy_config=proxy_config, force_permissions=True)
        pack_name = result[1]
        success = result[2][0]
        if success:
            LOG.info('Successfully installed pack "%s"' % pack_name)
        else:
            error = result[2][1]
            LOG.error('Failed to install pack "%s": %s' % (pack_name, error))
            sys.exit(2)
        LOG.info('Setting up virtualenv for pack "%s"' % pack_name)
        setup_pack_virtualenv(pack_name=pack_name, update=False, logger=LOG, proxy_config=proxy_config, no_download=True)
        LOG.info('Successfully set up virtualenv for pack "%s"' % pack_name)
    return 0