import shutil
import tempfile
import os
import json
from typing import Optional
import asyncio
import inspect
import electrum
from electrum.wallet_db import WalletDBUpgrader, WalletDB, WalletRequiresUpgrade, WalletRequiresSplit
from electrum.wallet import Wallet
from electrum import constants
from electrum import util
from electrum.plugin import Plugins
from electrum.simple_config import SimpleConfig
from . import as_testnet
from .test_wallet import WalletTestCase
WALLET_FILES_DIR = os.path.join(os.path.dirname(__file__), 'test_storage_upgrade')

class TestStorageUpgrade(WalletTestCase):

    def _get_wallet_str(self):
        if False:
            while True:
                i = 10
        test_method_name = inspect.stack()[1][3]
        assert isinstance(test_method_name, str)
        assert test_method_name.startswith('test_upgrade_from_')
        fname = test_method_name[len('test_upgrade_from_'):]
        test_vector_file = os.path.join(WALLET_FILES_DIR, fname)
        with open(test_vector_file, 'r') as f:
            wallet_str = f.read()
        return wallet_str

    async def test_upgrade_from_client_1_9_8_seeded(self):
        """note: this wallet file is not valid json: it tests the ast.literal_eval()
        fallback in wallet_db.load_data()
        """
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_0_4_seeded(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_0_4_importedkeys(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_0_4_watchaddresses(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_0_4_trezor_singleacc(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_0_4_trezor_multiacc(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str, accounts=2)

    async def test_upgrade_from_client_2_0_4_multisig(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_1_1_seeded(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_1_1_importedkeys(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_1_1_watchaddresses(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_1_1_trezor_singleacc(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_1_1_trezor_multiacc(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str, accounts=2)

    async def test_upgrade_from_client_2_1_1_multisig(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_2_0_seeded(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_2_0_importedkeys(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_2_0_watchaddresses(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_2_0_trezor_singleacc(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_2_0_trezor_multiacc(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str, accounts=2)

    async def test_upgrade_from_client_2_2_0_multisig(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_3_2_seeded(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_3_2_importedkeys(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_3_2_watchaddresses(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_3_2_trezor_singleacc(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_3_2_trezor_multiacc(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str, accounts=2)

    async def test_upgrade_from_client_2_3_2_multisig(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_4_3_seeded(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_4_3_importedkeys(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_4_3_watchaddresses(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_4_3_trezor_singleacc(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_4_3_trezor_multiacc(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str, accounts=2)

    async def test_upgrade_from_client_2_4_3_multisig(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_5_4_seeded(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_5_4_importedkeys(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_5_4_watchaddresses(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_5_4_trezor_singleacc(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_5_4_trezor_multiacc(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str, accounts=2)

    async def test_upgrade_from_client_2_5_4_multisig(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_6_4_seeded(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_6_4_importedkeys(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_6_4_watchaddresses(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_6_4_multisig(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_7_18_seeded(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_7_18_importedkeys(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_7_18_watchaddresses(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_7_18_trezor_singleacc(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_7_18_multisig(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_8_3_importedkeys_flawed_previous_upgrade_from_2_7_18(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_8_3_seeded(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_8_3_importedkeys(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_8_3_watchaddresses(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_8_3_trezor_singleacc(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_8_3_multisig(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_9_3_seeded(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    @as_testnet
    async def test_upgrade_from_client_2_9_3_old_seeded_with_realistic_history(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_9_3_importedkeys(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_9_3_watchaddresses(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_9_3_trezor_singleacc(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_2_9_3_multisig(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)

    async def test_upgrade_from_client_3_2_3_ledger_standard_keystore_changes(self):
        wallet_str = self._get_wallet_str()
        db = await self._upgrade_storage(wallet_str)
        wallet = Wallet(db, config=self.config)
        ks = wallet.keystore
        ks._root_fingerprint = 'deadbeef'
        ks.is_requesting_to_be_rewritten_to_wallet_file = True
        await wallet.stop()

    async def test_upgrade_from_client_2_9_3_importedkeys_keystore_changes(self):
        wallet_str = self._get_wallet_str()
        db = await self._upgrade_storage(wallet_str)
        wallet = Wallet(db, config=self.config)
        wallet.import_private_keys(['p2wpkh:L1cgMEnShp73r9iCukoPE3MogLeueNYRD9JVsfT1zVHyPBR3KqBY'], password=None)
        await wallet.stop()

    @as_testnet
    async def test_upgrade_from_client_3_3_8_xpub_with_realistic_history(self):
        wallet_str = self._get_wallet_str()
        await self._upgrade_storage(wallet_str)
    plugins: 'electrum.plugin.Plugins'

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        gui_name = 'cmdline'
        self.plugins = Plugins(self.config, gui_name)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.plugins.stop()
        self.plugins.stopped_event.wait()
        super().tearDown()

    async def _upgrade_storage(self, wallet_json, accounts=1) -> Optional[WalletDB]:
        if accounts == 1:
            try:
                db = self._load_db_from_json_string(wallet_json=wallet_json, upgrade=False)
            except WalletRequiresUpgrade:
                db = self._load_db_from_json_string(wallet_json=wallet_json, upgrade=True)
                await self._sanity_check_upgraded_db(db)
            return db
        else:
            try:
                db = self._load_db_from_json_string(wallet_json=wallet_json, upgrade=False)
            except WalletRequiresSplit as e:
                split_data = e._split_data
                self.assertEqual(accounts, len(split_data))
                for item in split_data:
                    data = json.dumps(item)
                    new_db = WalletDB(data, storage=None, upgrade=True)
                    await self._sanity_check_upgraded_db(new_db)

    async def _sanity_check_upgraded_db(self, db):
        wallet = Wallet(db, config=self.config)
        await wallet.stop()

    @staticmethod
    def _load_db_from_json_string(*, wallet_json, upgrade):
        if False:
            print('Hello World!')
        db = WalletDB(wallet_json, storage=None, upgrade=upgrade)
        return db