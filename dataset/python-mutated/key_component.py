from ipv8.keyvault.private.libnaclkey import LibNaCLSK
from tribler.core.components.component import Component
from tribler.core.config.tribler_config import TriblerConfig
from tribler.core.utilities.path_util import Path

class KeyComponent(Component):
    primary_key: LibNaCLSK
    secondary_key: LibNaCLSK

    async def run(self):
        config = self.session.config
        primary_private_key_path = config.state_dir / self.get_private_key_filename(config)
        primary_public_key_path = config.state_dir / config.trustchain.ec_keypair_pubfilename
        self.primary_key = self.load_or_create(primary_private_key_path, primary_public_key_path)
        secondary_private_key_path = config.state_dir / config.trustchain.secondary_key_filename
        self.secondary_key = self.load_or_create(secondary_private_key_path)

    @staticmethod
    def load_or_create(private_key_path: Path, public_key_path: Path=None) -> LibNaCLSK:
        if False:
            i = 10
            return i + 15
        if private_key_path.exists():
            return LibNaCLSK(private_key_path.read_bytes())
        key = LibNaCLSK()
        private_key_path.write_bytes(key.key.sk + key.key.seed)
        if public_key_path:
            public_key_path.write_bytes(key.key.pk)
        return key

    @staticmethod
    def get_private_key_filename(config: TriblerConfig):
        if False:
            i = 10
            return i + 15
        if config.general.testnet:
            return config.trustchain.testnet_keypair_filename
        return config.trustchain.ec_keypair_filename