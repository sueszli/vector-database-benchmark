import os
import time
from typing import TYPE_CHECKING, Optional
import struct
from electrum import bip32
from electrum.bip32 import BIP32Node, InvalidMasterKeyVersionBytes
from electrum.i18n import _
from electrum.plugin import Device, hook, runs_in_hwd_thread
from electrum.keystore import Hardware_KeyStore, KeyStoreWithMPK
from electrum.transaction import PartialTransaction
from electrum.wallet import Standard_Wallet, Multisig_Wallet, Abstract_Wallet
from electrum.util import bfh, versiontuple, UserFacingException
from electrum.logging import get_logger
from ..hw_wallet import HW_PluginBase, HardwareClientBase
from ..hw_wallet.plugin import LibraryFoundButUnusable, only_hook_if_libraries_available
if TYPE_CHECKING:
    from electrum.plugin import DeviceInfo
    from electrum.wizard import NewWalletWizard
_logger = get_logger(__name__)
try:
    import hid
    from ckcc.protocol import CCProtocolPacker, CCProtocolUnpacker
    from ckcc.protocol import CCProtoError, CCUserRefused, CCBusyError
    from ckcc.constants import MAX_MSG_LEN, MAX_BLK_LEN, MSG_SIGNING_MAX_LENGTH, MAX_TXN_LEN, AF_CLASSIC, AF_P2SH, AF_P2WPKH, AF_P2WSH, AF_P2WPKH_P2SH, AF_P2WSH_P2SH
    from ckcc.client import ColdcardDevice, COINKITE_VID, CKCC_PID, CKCC_SIMULATOR_PATH
    requirements_ok = True

    class ElectrumColdcardDevice(ColdcardDevice):

        def mitm_verify(self, sig, expect_xpub):
            if False:
                while True:
                    i = 10
            pubkey = BIP32Node.from_xkey(expect_xpub).eckey
            return pubkey.verify_message_hash(sig[1:65], self.session_key)
except ImportError as e:
    if not (isinstance(e, ModuleNotFoundError) and e.name == 'ckcc'):
        _logger.exception('error importing coldcard plugin deps')
    requirements_ok = False
    COINKITE_VID = 53566
    CKCC_PID = 52240
CKCC_SIMULATED_PID = CKCC_PID ^ 21930

class CKCCClient(HardwareClientBase):

    def __init__(self, plugin, handler, dev_path, *, is_simulator=False):
        if False:
            return 10
        HardwareClientBase.__init__(self, plugin=plugin)
        self.device = plugin.device
        self.handler = handler
        self._expected_device = None
        if is_simulator:
            self.dev = ElectrumColdcardDevice(dev_path, encrypt=True)
        else:
            hd = hid.device(path=dev_path)
            hd.open_path(dev_path)
            self.dev = ElectrumColdcardDevice(dev=hd, encrypt=True)

    def device_model_name(self) -> Optional[str]:
        if False:
            return 10
        return 'Coldcard'

    def get_soft_device_id(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        try:
            super().get_soft_device_id()
        except CCProtoError:
            return None

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<CKCCClient: xfp=%s label=%r>' % (xfp2str(self.dev.master_fingerprint), self.label())

    @runs_in_hwd_thread
    def verify_connection(self, expected_xfp: int, expected_xpub: str):
        if False:
            print('Hello World!')
        ex = (expected_xfp, expected_xpub)
        if self._expected_device == ex:
            return
        assert expected_xpub
        if self._expected_device is not None or self.dev.master_fingerprint != expected_xfp or self.dev.master_xpub != expected_xpub:
            _logger.info(f'xpubs. reported by device: {self.dev.master_xpub}. stored in file: {expected_xpub}')
            raise RuntimeError("Expecting %s but that's not what's connected?!" % xfp2str(expected_xfp))
        self.dev.check_mitm(expected_xpub=expected_xpub)
        self._expected_device = ex
        _logger.info('Successfully verified against MiTM')

    def is_pairable(self):
        if False:
            print('Hello World!')
        return bool(self.dev.master_xpub)

    @runs_in_hwd_thread
    def close(self):
        if False:
            print('Hello World!')
        self.dev.close()
        self.dev = None

    def is_initialized(self):
        if False:
            return 10
        return bool(self.dev.master_xpub)

    def label(self):
        if False:
            print('Hello World!')
        if self.dev.is_simulator:
            lab = 'Coldcard Simulator ' + xfp2str(self.dev.master_fingerprint)
        elif not self.dev.master_fingerprint:
            lab = 'Coldcard #' + self.dev.serial
        else:
            lab = 'Coldcard ' + xfp2str(self.dev.master_fingerprint)
        return lab

    def _get_ckcc_master_xpub_from_device(self):
        if False:
            i = 10
            return i + 15
        master_xpub = self.dev.master_xpub
        if master_xpub is not None:
            try:
                node = BIP32Node.from_xkey(master_xpub)
            except InvalidMasterKeyVersionBytes:
                raise UserFacingException(_('Invalid xpub magic. Make sure your {} device is set to the correct chain.').format(self.device) + ' ' + _('You might have to unplug and plug it in again.')) from None
            return master_xpub

    @runs_in_hwd_thread
    def has_usable_connection_with_device(self):
        if False:
            while True:
                i = 10
        try:
            self.ping_check()
            return True
        except Exception:
            return False

    @runs_in_hwd_thread
    def get_xpub(self, bip32_path, xtype):
        if False:
            return 10
        assert xtype in ColdcardPlugin.SUPPORTED_XTYPES
        _logger.info('Derive xtype = %r' % xtype)
        xpub = self.dev.send_recv(CCProtocolPacker.get_xpub(bip32_path), timeout=5000)
        try:
            node = BIP32Node.from_xkey(xpub)
        except InvalidMasterKeyVersionBytes:
            raise UserFacingException(_('Invalid xpub magic. Make sure your {} device is set to the correct chain.').format(self.device)) from None
        if xtype != 'standard':
            xpub = node._replace(xtype=xtype).to_xpub()
        return xpub

    @runs_in_hwd_thread
    def ping_check(self):
        if False:
            return 10
        assert self.dev.session_key, 'not encrypted?'
        req = b'1234 Electrum Plugin 4321'
        try:
            echo = self.dev.send_recv(CCProtocolPacker.ping(req))
            assert echo == req
        except Exception:
            raise RuntimeError('Communication trouble with Coldcard')

    @runs_in_hwd_thread
    def show_address(self, path, addr_fmt):
        if False:
            i = 10
            return i + 15
        return self.dev.send_recv(CCProtocolPacker.show_address(path, addr_fmt), timeout=None)

    @runs_in_hwd_thread
    def show_p2sh_address(self, *args, **kws):
        if False:
            for i in range(10):
                print('nop')
        return self.dev.send_recv(CCProtocolPacker.show_p2sh_address(*args, **kws), timeout=None)

    @runs_in_hwd_thread
    def get_version(self):
        if False:
            for i in range(10):
                print('nop')
        return self.dev.send_recv(CCProtocolPacker.version(), timeout=1000).split('\n')

    @runs_in_hwd_thread
    def sign_message_start(self, path, msg):
        if False:
            return 10
        self.dev.send_recv(CCProtocolPacker.sign_message(msg, path), timeout=None)

    @runs_in_hwd_thread
    def sign_message_poll(self):
        if False:
            print('Hello World!')
        return self.dev.send_recv(CCProtocolPacker.get_signed_msg(), timeout=None)

    @runs_in_hwd_thread
    def sign_transaction_start(self, raw_psbt: bytes, *, finalize: bool=False):
        if False:
            while True:
                i = 10
        assert 20 <= len(raw_psbt) < MAX_TXN_LEN, 'PSBT is too big'
        (dlen, chk) = self.dev.upload_file(raw_psbt)
        resp = self.dev.send_recv(CCProtocolPacker.sign_transaction(dlen, chk, finalize=finalize), timeout=None)
        if resp is not None:
            raise ValueError(resp)

    @runs_in_hwd_thread
    def sign_transaction_poll(self):
        if False:
            while True:
                i = 10
        return self.dev.send_recv(CCProtocolPacker.get_signed_txn(), timeout=None)

    @runs_in_hwd_thread
    def download_file(self, length, checksum, file_number=1):
        if False:
            print('Hello World!')
        return self.dev.download_file(length, checksum, file_number=file_number)

class Coldcard_KeyStore(Hardware_KeyStore):
    hw_type = 'coldcard'
    device = 'Coldcard'
    plugin: 'ColdcardPlugin'

    def __init__(self, d):
        if False:
            return 10
        Hardware_KeyStore.__init__(self, d)
        self.ux_busy = False
        self.ckcc_xpub = d.get('ckcc_xpub', None)

    def dump(self):
        if False:
            while True:
                i = 10
        d = Hardware_KeyStore.dump(self)
        d['ckcc_xpub'] = self.ckcc_xpub
        return d

    def get_xfp_int(self) -> int:
        if False:
            return 10
        xfp = self.get_root_fingerprint()
        assert xfp is not None
        return xfp_int_from_xfp_bytes(bfh(xfp))

    def opportunistically_fill_in_missing_info_from_device(self, client: 'CKCCClient'):
        if False:
            while True:
                i = 10
        super().opportunistically_fill_in_missing_info_from_device(client)
        if self.ckcc_xpub is None:
            self.ckcc_xpub = client._get_ckcc_master_xpub_from_device()
            self.is_requesting_to_be_rewritten_to_wallet_file = True

    def get_client(self, *args, **kwargs):
        if False:
            print('Hello World!')
        client = super().get_client(*args, **kwargs)
        if client:
            xfp_int = self.get_xfp_int()
            client.verify_connection(xfp_int, self.ckcc_xpub)
        return client

    def give_error(self, message):
        if False:
            i = 10
            return i + 15
        self.logger.info(message)
        if not self.ux_busy:
            self.handler.show_error(message)
        else:
            self.ux_busy = False
        raise UserFacingException(message)

    def wrap_busy(func):
        if False:
            i = 10
            return i + 15

        def wrapper(self, *args, **kwargs):
            if False:
                return 10
            try:
                self.ux_busy = True
                return func(self, *args, **kwargs)
            finally:
                self.ux_busy = False
        return wrapper

    def decrypt_message(self, pubkey, message, password):
        if False:
            while True:
                i = 10
        raise UserFacingException(_('Encryption and decryption are currently not supported for {}').format(self.device))

    @wrap_busy
    def sign_message(self, sequence, message, password, *, script_type=None):
        if False:
            for i in range(10):
                print('nop')
        try:
            msg = message.encode('ascii', errors='strict')
            assert 1 <= len(msg) <= MSG_SIGNING_MAX_LENGTH
        except (UnicodeError, AssertionError):
            self.handler.show_error('Only short (%d max) ASCII messages can be signed.' % MSG_SIGNING_MAX_LENGTH)
            return b''
        path = self.get_derivation_prefix() + '/%d/%d' % sequence
        try:
            cl = self.get_client()
            try:
                self.handler.show_message('Signing message (using %s)...' % path)
                cl.sign_message_start(path, msg)
                while 1:
                    time.sleep(0.25)
                    resp = cl.sign_message_poll()
                    if resp is not None:
                        break
            finally:
                self.handler.finished()
            assert len(resp) == 2
            (addr, raw_sig) = resp
            assert 40 < len(raw_sig) <= 65
            return raw_sig
        except (CCUserRefused, CCBusyError) as exc:
            self.handler.show_error(str(exc))
        except CCProtoError as exc:
            self.logger.exception('Error showing address')
            self.handler.show_error('{}\n\n{}'.format(_('Error showing address') + ':', str(exc)))
        except Exception as e:
            self.give_error(e)
        return b''

    @wrap_busy
    def sign_transaction(self, tx, password):
        if False:
            return 10
        if tx.is_complete():
            return
        client = self.get_client()
        assert client.dev.master_fingerprint == self.get_xfp_int()
        raw_psbt = tx.serialize_as_bytes()
        try:
            try:
                self.handler.show_message('Authorize Transaction...')
                client.sign_transaction_start(raw_psbt)
                while 1:
                    time.sleep(0.25)
                    resp = client.sign_transaction_poll()
                    if resp is not None:
                        break
                (rlen, rsha) = resp
                raw_resp = client.download_file(rlen, rsha)
            finally:
                self.handler.finished()
        except (CCUserRefused, CCBusyError) as exc:
            self.logger.info(f'Did not sign: {exc}')
            self.handler.show_error(str(exc))
            return
        except BaseException as e:
            self.logger.exception('')
            self.give_error(e)
            return
        tx2 = PartialTransaction.from_raw_psbt(raw_resp)
        tx.combine_with_other_psbt(tx2)

    @staticmethod
    def _encode_txin_type(txin_type):
        if False:
            print('Hello World!')
        return {'standard': AF_CLASSIC, 'p2pkh': AF_CLASSIC, 'p2sh': AF_P2SH, 'p2wpkh-p2sh': AF_P2WPKH_P2SH, 'p2wpkh': AF_P2WPKH, 'p2wsh-p2sh': AF_P2WSH_P2SH, 'p2wsh': AF_P2WSH}[txin_type]

    @wrap_busy
    def show_address(self, sequence, txin_type):
        if False:
            for i in range(10):
                print('nop')
        client = self.get_client()
        address_path = self.get_derivation_prefix()[2:] + '/%d/%d' % sequence
        addr_fmt = self._encode_txin_type(txin_type)
        try:
            try:
                self.handler.show_message(_('Showing address ...'))
                dev_addr = client.show_address(address_path, addr_fmt)
            finally:
                self.handler.finished()
        except CCProtoError as exc:
            self.logger.exception('Error showing address')
            self.handler.show_error('{}\n\n{}'.format(_('Error showing address') + ':', str(exc)))
        except BaseException as exc:
            self.logger.exception('')
            self.handler.show_error(exc)

    @wrap_busy
    def show_p2sh_address(self, M, script, xfp_paths, txin_type):
        if False:
            for i in range(10):
                print('nop')
        client = self.get_client()
        addr_fmt = self._encode_txin_type(txin_type)
        try:
            try:
                self.handler.show_message(_('Showing address ...'))
                dev_addr = client.show_p2sh_address(M, xfp_paths, script, addr_fmt=addr_fmt)
            finally:
                self.handler.finished()
        except CCProtoError as exc:
            self.logger.exception('Error showing address')
            self.handler.show_error('{}.\n{}\n\n{}'.format(_('Error showing address'), _('Make sure you have imported the correct wallet description file on the device for this multisig wallet.'), str(exc)))
        except BaseException as exc:
            self.logger.exception('')
            self.handler.show_error(exc)

class ColdcardPlugin(HW_PluginBase):
    keystore_class = Coldcard_KeyStore
    minimum_library = (0, 7, 7)
    DEVICE_IDS = [(COINKITE_VID, CKCC_PID), (COINKITE_VID, CKCC_SIMULATED_PID)]
    SUPPORTED_XTYPES = ('standard', 'p2wpkh-p2sh', 'p2wpkh', 'p2wsh-p2sh', 'p2wsh')

    def __init__(self, parent, config, name):
        if False:
            return 10
        HW_PluginBase.__init__(self, parent, config, name)
        self.libraries_available = self.check_libraries_available()
        if not self.libraries_available:
            return
        self.device_manager().register_devices(self.DEVICE_IDS, plugin=self)
        self.device_manager().register_enumerate_func(self.detect_simulator)

    def get_library_version(self):
        if False:
            return 10
        import ckcc
        try:
            version = ckcc.__version__
        except AttributeError:
            version = 'unknown'
        if requirements_ok:
            return version
        else:
            raise LibraryFoundButUnusable(library_version=version)

    def detect_simulator(self):
        if False:
            print('Hello World!')
        fn = CKCC_SIMULATOR_PATH
        if os.path.exists(fn):
            return [Device(path=fn, interface_number=-1, id_=fn, product_key=(COINKITE_VID, CKCC_SIMULATED_PID), usage_page=0, transport_ui_string='simulator')]
        return []

    @runs_in_hwd_thread
    def create_client(self, device, handler):
        if False:
            i = 10
            return i + 15
        try:
            rv = CKCCClient(self, handler, device.path, is_simulator=device.product_key[1] == CKCC_SIMULATED_PID)
            return rv
        except Exception as e:
            self.logger.exception('late failure connecting to device?')
            return None

    @runs_in_hwd_thread
    def get_client(self, keystore, force_pair=True, *, devices=None, allow_user_interaction=True) -> Optional['CKCCClient']:
        if False:
            return 10
        client = super().get_client(keystore, force_pair, devices=devices, allow_user_interaction=allow_user_interaction)
        if client is not None:
            client.ping_check()
        return client

    @staticmethod
    def export_ms_wallet(wallet: Multisig_Wallet, fp, name):
        if False:
            print('Hello World!')
        assert isinstance(wallet, Multisig_Wallet)
        print('# Exported from Electrum', file=fp)
        print(f'Name: {name:.20s}', file=fp)
        print(f'Policy: {wallet.m} of {wallet.n}', file=fp)
        print(f'Format: {wallet.txin_type.upper()}', file=fp)
        xpubs = []
        for (xpub, ks) in zip(wallet.get_master_public_keys(), wallet.get_keystores()):
            (fp_bytes, der_full) = ks.get_fp_and_derivation_to_be_used_in_partial_tx(der_suffix=[], only_der_suffix=False)
            fp_hex = fp_bytes.hex().upper()
            der_prefix_str = bip32.convert_bip32_intpath_to_strpath(der_full)
            xpubs.append((fp_hex, xpub, der_prefix_str))
        print('', file=fp)
        assert len(xpubs) == wallet.n
        for (xfp, xpub, der_prefix) in xpubs:
            print(f'Derivation: {der_prefix}', file=fp)
            print(f'{xfp}: {xpub}\n', file=fp)

    def show_address(self, wallet, address, keystore: 'Coldcard_KeyStore'=None):
        if False:
            for i in range(10):
                print('nop')
        if keystore is None:
            keystore = wallet.get_keystore()
        if not self.show_address_helper(wallet, address, keystore):
            return
        txin_type = wallet.get_txin_type(address)
        if type(wallet) is Standard_Wallet:
            sequence = wallet.get_address_index(address)
            keystore.show_address(sequence, txin_type)
        elif type(wallet) is Multisig_Wallet:
            assert isinstance(wallet, Multisig_Wallet)
            pubkey_deriv_info = wallet.get_public_keys_with_deriv_info(address)
            pubkey_hexes = sorted([pk.hex() for pk in list(pubkey_deriv_info)])
            xfp_paths = []
            for pubkey_hex in pubkey_hexes:
                pubkey = bytes.fromhex(pubkey_hex)
                (ks, der_suffix) = pubkey_deriv_info[pubkey]
                (fp_bytes, der_full) = ks.get_fp_and_derivation_to_be_used_in_partial_tx(der_suffix, only_der_suffix=False)
                xfp_int = xfp_int_from_xfp_bytes(fp_bytes)
                xfp_paths.append([xfp_int] + list(der_full))
            script = bfh(wallet.pubkeys_to_scriptcode(pubkey_hexes))
            keystore.show_p2sh_address(wallet.m, script, xfp_paths, txin_type)
        else:
            keystore.handler.show_error(_('This function is only available for standard wallets when using {}.').format(self.device))
            return

    def wizard_entry_for_device(self, device_info: 'DeviceInfo', *, new_wallet=True) -> str:
        if False:
            for i in range(10):
                print('nop')
        if new_wallet:
            return 'coldcard_start' if device_info.initialized else 'coldcard_not_initialized'
        else:
            return 'coldcard_unlock'

    def extend_wizard(self, wizard: 'NewWalletWizard'):
        if False:
            return 10
        views = {'coldcard_start': {'next': 'coldcard_xpub'}, 'coldcard_xpub': {'next': lambda d: wizard.wallet_password_view(d) if wizard.last_cosigner(d) else 'multisig_cosigner_keystore', 'accept': wizard.maybe_master_pubkey, 'last': lambda d: wizard.is_single_password() and wizard.last_cosigner(d)}, 'coldcard_not_initialized': {}, 'coldcard_unlock': {'last': True}}
        wizard.navmap_merge(views)

def xfp_int_from_xfp_bytes(fp_bytes: bytes) -> int:
    if False:
        while True:
            i = 10
    return int.from_bytes(fp_bytes, byteorder='little', signed=False)

def xfp2str(xfp: int) -> str:
    if False:
        for i in range(10):
            print('nop')
    return struct.pack('<I', xfp).hex().lower()