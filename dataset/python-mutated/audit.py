"""
Handle audited data changes.
"""
import logging
from functools import partial
from pymongo.errors import OperationFailure
from .versioned_item import VersionedItem, ChangedItem
from .._util import are_equals
from ..decorators import _get_host
from ..exceptions import NoDataFoundException, ConcurrentModificationException
logger = logging.getLogger(__name__)

class DataChange(object):
    """
    Object representing incoming data change
    """

    def __init__(self, date_range, new_data):
        if False:
            i = 10
            return i + 15
        self.date_range = date_range
        self.new_data = new_data

class ArcticTransaction(object):
    """Use this context manager if you want to modify data in a version store while ensuring that no other writes
    interfere with your own.

    To use, base your modifications on the `base_ts` context manager field and put your newly created timeseries and
    call the `write` method of the context manager to output changes. The changes will only be written when the block
    exits.

    NB changes may be audited.

    Example:
    -------
    with ArcticTransaction(Arctic('hostname')['some_library'], 'symbol') as mt:
        ts_version_info = mt.base_ts
        # do some processing, come up with a new ts for 'symbol' called new_symbol_ts, presumably based on ts_version_info.data
        mt.write('symbol', new_symbol_ts, metadata=new_symbol_metadata)

    The block will raise a ConcurrentModificationException if an inconsistency has been detected. You will have to
    retry the whole block should that happens, as the assumption is that you need to base your changes on a different
    starting timeseries.
    """

    def __init__(self, version_store, symbol, user, log, modify_timeseries=None, audit=True, *args, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Parameters\n        ----------\n        version_store: `VersionStore` Arctic Library\n            Needs to support write, read, list_versions, _delete_version this is the underlying store that we'll\n            be securing for write\n\n        symbol: `str`\n            symbol name for the item that's being modified\n\n        user: `str`\n            user making the change\n\n        log: `str`\n            Log message for the change\n\n        modify_timeseries:\n            if given, it will check the assumption that this is the latest data available for symbol in version_store\n            Should not this be the case, a ConcurrentModificationException will be raised. Use this if you're\n            interacting with code that read in the data already and for some reason you cannot refactor the read-write\n            operation to be contained within this context manager\n\n        audit: `bool`\n            should we 'audit' the transaction. An audited write transaction is equivalent to a snapshot\n            before and after the data change - i.e. we won't prune versions of the data involved in an\n            audited transaction.  This can be used to ensure that the history of certain data changes is\n            preserved indefinitely.\n\n        all other args:\n            Will be passed into the initial read\n        "
        self._version_store = version_store
        self._symbol = symbol
        self._user = user
        self._log = log
        self._audit = audit
        logger.info('MT: {}@{}: [{}] {}: {}'.format(_get_host(version_store).get('l'), _get_host(version_store).get('mhost'), user, log, symbol))
        try:
            self.base_ts = self._version_store.read(self._symbol, *args, **kwargs)
        except NoDataFoundException:
            versions = [x['version'] for x in self._version_store.list_versions(self._symbol, latest_only=True)]
            versions.append(0)
            self.base_ts = VersionedItem(symbol=self._symbol, library=None, version=versions[0], metadata=None, data=None, host=None)
        except OperationFailure:
            self.base_ts = self._version_store.read_metadata(symbol=self._symbol)
        if modify_timeseries is not None and (not are_equals(modify_timeseries, self.base_ts.data)):
            raise ConcurrentModificationException()
        self._do_write = False

    def change(self, symbol, data_changes, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Change, and audit 'data' under the specified 'symbol' name to this library.\n\n        Parameters\n        ----------\n        symbol: `str`\n            symbol name for the item\n\n        data_changes: `list DataChange`\n            list of DataChange objects\n        "
        pass

    def write(self, symbol, data, prune_previous_version=True, metadata=None, **kwargs):
        if False:
            return 10
        '\n        Records a write request to be actioned on context exit. Takes exactly the same parameters as the regular\n        library write call.\n        '
        if data is not None:
            if self.base_ts.data is None or not are_equals(data, self.base_ts.data) or metadata != self.base_ts.metadata:
                self._do_write = True
        self._write = partial(self._version_store.write, symbol, data, prune_previous_version=prune_previous_version, metadata=metadata, **kwargs)

    def __enter__(self):
        if False:
            print('Hello World!')
        return self

    def __exit__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        if self._do_write:
            written_ver = self._write()
            versions = [x['version'] for x in self._version_store.list_versions(self._symbol)]
            versions.append(0)
            versions.reverse()
            base_offset = versions.index(self.base_ts.version)
            new_offset = versions.index(written_ver.version)
            if len(versions[base_offset:new_offset + 1]) != 2:
                self._version_store._delete_version(self._symbol, written_ver.version)
                raise ConcurrentModificationException('Inconsistent Versions: {}: {}->{}'.format(self._symbol, self.base_ts.version, written_ver.version))
            changed = ChangedItem(self._symbol, self.base_ts, written_ver, None)
            if self._audit:
                self._version_store._write_audit(self._user, self._log, changed)