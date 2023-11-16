import time
from typing import TYPE_CHECKING, List, Optional, Union, Dict, Any, Sequence
from decimal import Decimal
import attr
from .json_db import StoredObject, stored_in
from .i18n import _
from .util import age, InvoiceError, format_satoshis
from .bip21 import create_bip21_uri
from .lnutil import hex_to_bytes
from .lnaddr import lndecode, LnAddr
from . import constants
from .bitcoin import COIN, TOTAL_COIN_SUPPLY_LIMIT_IN_BTC
from .bitcoin import address_to_script
from .transaction import PartialTxOutput
from .crypto import sha256d
if TYPE_CHECKING:
    from .paymentrequest import PaymentRequest
PR_UNPAID = 0
PR_EXPIRED = 1
PR_UNKNOWN = 2
PR_PAID = 3
PR_INFLIGHT = 4
PR_FAILED = 5
PR_ROUTING = 6
PR_UNCONFIRMED = 7
PR_BROADCASTING = 8
PR_BROADCAST = 9
pr_color = {PR_UNPAID: (0.7, 0.7, 0.7, 1), PR_PAID: (0.2, 0.9, 0.2, 1), PR_UNKNOWN: (0.7, 0.7, 0.7, 1), PR_EXPIRED: (0.9, 0.2, 0.2, 1), PR_INFLIGHT: (0.9, 0.6, 0.3, 1), PR_FAILED: (0.9, 0.2, 0.2, 1), PR_ROUTING: (0.9, 0.6, 0.3, 1), PR_BROADCASTING: (0.9, 0.6, 0.3, 1), PR_BROADCAST: (0.9, 0.6, 0.3, 1), PR_UNCONFIRMED: (0.9, 0.6, 0.3, 1)}

def pr_tooltips():
    if False:
        while True:
            i = 10
    return {PR_UNPAID: _('Unpaid'), PR_PAID: _('Paid'), PR_UNKNOWN: _('Unknown'), PR_EXPIRED: _('Expired'), PR_INFLIGHT: _('In progress'), PR_BROADCASTING: _('Broadcasting'), PR_BROADCAST: _('Broadcast successfully'), PR_FAILED: _('Failed'), PR_ROUTING: _('Computing route...'), PR_UNCONFIRMED: _('Unconfirmed')}

def pr_expiration_values():
    if False:
        print('Hello World!')
    return {0: _('Never'), 10 * 60: _('10 minutes'), 60 * 60: _('1 hour'), 24 * 60 * 60: _('1 day'), 7 * 24 * 60 * 60: _('1 week')}
PR_DEFAULT_EXPIRATION_WHEN_CREATING = 24 * 60 * 60
assert PR_DEFAULT_EXPIRATION_WHEN_CREATING in pr_expiration_values()

def _decode_outputs(outputs) -> Optional[List[PartialTxOutput]]:
    if False:
        for i in range(10):
            print('nop')
    if outputs is None:
        return None
    ret = []
    for output in outputs:
        if not isinstance(output, PartialTxOutput):
            output = PartialTxOutput.from_legacy_tuple(*output)
        ret.append(output)
    return ret
LN_EXPIRY_NEVER = 100 * 365 * 24 * 60 * 60

@attr.s
class BaseInvoice(StoredObject):
    """
    Base class for Invoice and Request
    In the code, we use 'invoice' for outgoing payments, and 'request' for incoming payments.

    TODO this class is getting too complicated for "attrs"... maybe we should rewrite it without.
    """
    amount_msat = attr.ib(kw_only=True, on_setattr=attr.setters.validate)
    message = attr.ib(type=str, kw_only=True)
    time = attr.ib(type=int, kw_only=True, validator=attr.validators.instance_of(int), on_setattr=attr.setters.validate)
    exp = attr.ib(type=int, kw_only=True, validator=attr.validators.instance_of(int), on_setattr=attr.setters.validate)
    outputs = attr.ib(kw_only=True, converter=_decode_outputs)
    height = attr.ib(type=int, kw_only=True, validator=attr.validators.instance_of(int), on_setattr=attr.setters.validate)
    bip70 = attr.ib(type=str, kw_only=True)

    def is_lightning(self) -> bool:
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def get_address(self) -> Optional[str]:
        if False:
            print('Hello World!')
        'returns the first address, to be displayed in GUI'
        raise NotImplementedError()

    @property
    def rhash(self) -> str:
        if False:
            return 10
        raise NotImplementedError()

    def get_status_str(self, status):
        if False:
            while True:
                i = 10
        status_str = pr_tooltips()[status]
        if status == PR_UNPAID:
            if self.exp > 0 and self.exp != LN_EXPIRY_NEVER:
                expiration = self.get_expiration_date()
                status_str = _('Expires') + ' ' + age(expiration, include_seconds=True)
        return status_str

    def get_outputs(self) -> Sequence[PartialTxOutput]:
        if False:
            while True:
                i = 10
        outputs = self.outputs or []
        if not outputs:
            address = self.get_address()
            amount = self.get_amount_sat()
            if address and amount is not None:
                outputs = [PartialTxOutput.from_address_and_value(address, int(amount))]
        return outputs

    def get_expiration_date(self):
        if False:
            for i in range(10):
                print('nop')
        return self.exp + self.time if self.exp else 0

    @staticmethod
    def _get_cur_time():
        if False:
            print('Hello World!')
        return time.time()

    def has_expired(self) -> bool:
        if False:
            print('Hello World!')
        exp = self.get_expiration_date()
        return bool(exp) and exp < self._get_cur_time()

    def get_amount_msat(self) -> Union[int, str, None]:
        if False:
            print('Hello World!')
        return self.amount_msat

    def get_time(self):
        if False:
            i = 10
            return i + 15
        return self.time

    def get_message(self):
        if False:
            i = 10
            return i + 15
        return self.message

    def get_amount_sat(self) -> Union[int, str, None]:
        if False:
            return 10
        "\n        Returns an integer satoshi amount, or '!' or None.\n        Callers who need msat precision should call get_amount_msat()\n        "
        amount_msat = self.amount_msat
        if amount_msat in [None, '!']:
            return amount_msat
        return int(amount_msat // 1000)

    def set_amount_msat(self, amount_msat: Union[int, str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'The GUI uses this to fill the amount for a zero-amount invoice.'
        if amount_msat == '!':
            amount_sat = amount_msat
        else:
            assert isinstance(amount_msat, int), f'amount_msat={amount_msat!r}'
            assert amount_msat >= 0, amount_msat
            amount_sat = amount_msat // 1000 + int(amount_msat % 1000 > 0)
        if (outputs := self.outputs):
            assert len(self.outputs) == 1, len(self.outputs)
            self.outputs = [PartialTxOutput(scriptpubkey=outputs[0].scriptpubkey, value=amount_sat)]
        self.amount_msat = amount_msat

    @amount_msat.validator
    def _validate_amount(self, attribute, value):
        if False:
            for i in range(10):
                print('nop')
        if value is None:
            return
        if isinstance(value, int):
            if not 0 <= value <= TOTAL_COIN_SUPPLY_LIMIT_IN_BTC * COIN * 1000:
                raise InvoiceError(f'amount is out-of-bounds: {value!r} msat')
        elif isinstance(value, str):
            if value != '!':
                raise InvoiceError(f'unexpected amount: {value!r}')
        else:
            raise InvoiceError(f'unexpected amount: {value!r}')

    @classmethod
    def from_bech32(cls, invoice: str) -> 'Invoice':
        if False:
            i = 10
            return i + 15
        'Constructs Invoice object from BOLT-11 string.\n        Might raise InvoiceError.\n        '
        try:
            lnaddr = lndecode(invoice)
        except Exception as e:
            raise InvoiceError(e) from e
        amount_msat = lnaddr.get_amount_msat()
        timestamp = lnaddr.date
        exp_delay = lnaddr.get_expiry()
        message = lnaddr.get_description()
        return Invoice(message=message, amount_msat=amount_msat, time=timestamp, exp=exp_delay, outputs=None, bip70=None, height=0, lightning_invoice=invoice)

    @classmethod
    def from_bip70_payreq(cls, pr: 'PaymentRequest', *, height: int=0) -> 'Invoice':
        if False:
            i = 10
            return i + 15
        return Invoice(amount_msat=pr.get_amount() * 1000, message=pr.get_memo(), time=pr.get_time(), exp=pr.get_expiration_date() - pr.get_time(), outputs=pr.get_outputs(), bip70=pr.raw.hex(), height=height, lightning_invoice=None)

    def get_id(self) -> str:
        if False:
            return 10
        if self.is_lightning():
            return self.rhash
        else:
            return get_id_from_onchain_outputs(outputs=self.get_outputs(), timestamp=self.time)

    def as_dict(self, status):
        if False:
            return 10
        d = {'is_lightning': self.is_lightning(), 'amount_BTC': format_satoshis(self.get_amount_sat()), 'message': self.message, 'timestamp': self.get_time(), 'expiry': self.exp, 'status': status, 'status_str': self.get_status_str(status), 'id': self.get_id(), 'amount_sat': self.get_amount_sat()}
        if self.is_lightning():
            d['amount_msat'] = self.get_amount_msat()
        return d

@stored_in('invoices')
@attr.s
class Invoice(BaseInvoice):
    lightning_invoice = attr.ib(type=str, kw_only=True)
    __lnaddr = None
    _broadcasting_status = None

    def is_lightning(self):
        if False:
            return 10
        return self.lightning_invoice is not None

    def get_broadcasting_status(self):
        if False:
            i = 10
            return i + 15
        return self._broadcasting_status

    def get_address(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        address = None
        if self.outputs:
            address = self.outputs[0].address if len(self.outputs) > 0 else None
        if not address and self.is_lightning():
            address = self._lnaddr.get_fallback_address() or None
        return address

    @property
    def _lnaddr(self) -> LnAddr:
        if False:
            return 10
        if self.__lnaddr is None:
            self.__lnaddr = lndecode(self.lightning_invoice)
        return self.__lnaddr

    @property
    def rhash(self) -> str:
        if False:
            return 10
        assert self.is_lightning()
        return self._lnaddr.paymenthash.hex()

    @lightning_invoice.validator
    def _validate_invoice_str(self, attribute, value):
        if False:
            for i in range(10):
                print('nop')
        if value is not None:
            lnaddr = lndecode(value)
            self.__lnaddr = lnaddr

    def can_be_paid_onchain(self) -> bool:
        if False:
            return 10
        if self.is_lightning():
            return bool(self._lnaddr.get_fallback_address()) or bool(self.outputs)
        else:
            return True

    def to_debug_json(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        d = self.to_json()
        d['lnaddr'] = self._lnaddr.to_debug_json()
        return d

@stored_in('payment_requests')
@attr.s
class Request(BaseInvoice):
    payment_hash = attr.ib(type=bytes, kw_only=True, converter=hex_to_bytes)

    def is_lightning(self):
        if False:
            print('Hello World!')
        return self.payment_hash is not None

    def get_address(self) -> Optional[str]:
        if False:
            return 10
        address = None
        if self.outputs:
            address = self.outputs[0].address if len(self.outputs) > 0 else None
        return address

    @property
    def rhash(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        assert self.is_lightning()
        return self.payment_hash.hex()

    def get_bip21_URI(self, *, lightning_invoice: Optional[str]=None) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        addr = self.get_address()
        amount = self.get_amount_sat()
        if amount is not None:
            amount = int(amount)
        message = self.message
        extra = {}
        if self.time and self.exp:
            extra['time'] = str(int(self.time))
            extra['exp'] = str(int(self.exp))
        if lightning_invoice:
            extra['lightning'] = lightning_invoice
        if not addr and lightning_invoice:
            return 'bitcoin:?lightning=' + lightning_invoice
        if not addr and (not lightning_invoice):
            return None
        uri = create_bip21_uri(addr, amount, message, extra_query_params=extra)
        return str(uri)

def get_id_from_onchain_outputs(outputs: Sequence[PartialTxOutput], *, timestamp: int) -> str:
    if False:
        while True:
            i = 10
    outputs_str = '\n'.join((f'{txout.scriptpubkey.hex()}, {txout.value}' for txout in outputs))
    return sha256d(outputs_str + '%d' % timestamp).hex()[0:10]