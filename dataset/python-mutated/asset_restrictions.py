import abc
from numpy import vectorize
from functools import partial, reduce
import operator
import pandas as pd
from six import with_metaclass, iteritems
from collections import namedtuple
from toolz import groupby
from zipline.utils.enum import enum
from zipline.utils.numpy_utils import vectorized_is_element
from zipline.assets import Asset
Restriction = namedtuple('Restriction', ['asset', 'effective_date', 'state'])
RESTRICTION_STATES = enum('ALLOWED', 'FROZEN')

class Restrictions(with_metaclass(abc.ABCMeta)):
    """
    Abstract restricted list interface, representing a set of assets that an
    algorithm is restricted from trading.
    """

    @abc.abstractmethod
    def is_restricted(self, assets, dt):
        if False:
            return 10
        '\n        Is the asset restricted (RestrictionStates.FROZEN) on the given dt?\n\n        Parameters\n        ----------\n        asset : Asset of iterable of Assets\n            The asset(s) for which we are querying a restriction\n        dt : pd.Timestamp\n            The timestamp of the restriction query\n\n        Returns\n        -------\n        is_restricted : bool or pd.Series[bool] indexed by asset\n            Is the asset or assets restricted on this dt?\n\n        '
        raise NotImplementedError('is_restricted')

    def __or__(self, other_restriction):
        if False:
            for i in range(10):
                print('nop')
        'Base implementation for combining two restrictions.\n        '
        if isinstance(other_restriction, _UnionRestrictions):
            return other_restriction | self
        return _UnionRestrictions([self, other_restriction])

class _UnionRestrictions(Restrictions):
    """
    A union of a number of sub restrictions.

    Parameters
    ----------
    sub_restrictions : iterable of Restrictions (but not _UnionRestrictions)
        The Restrictions to be added together

    Notes
    -----
    - Consumers should not construct instances of this class directly, but
      instead use the `|` operator to combine restrictions
    """

    def __new__(cls, sub_restrictions):
        if False:
            while True:
                i = 10
        sub_restrictions = [r for r in sub_restrictions if not isinstance(r, NoRestrictions)]
        if len(sub_restrictions) == 0:
            return NoRestrictions()
        elif len(sub_restrictions) == 1:
            return sub_restrictions[0]
        new_instance = super(_UnionRestrictions, cls).__new__(cls)
        new_instance.sub_restrictions = sub_restrictions
        return new_instance

    def __or__(self, other_restriction):
        if False:
            for i in range(10):
                print('nop')
        '\n        Overrides the base implementation for combining two restrictions, of\n        which the left side is a _UnionRestrictions.\n        '
        if isinstance(other_restriction, _UnionRestrictions):
            new_sub_restrictions = self.sub_restrictions + other_restriction.sub_restrictions
        else:
            new_sub_restrictions = self.sub_restrictions + [other_restriction]
        return _UnionRestrictions(new_sub_restrictions)

    def is_restricted(self, assets, dt):
        if False:
            i = 10
            return i + 15
        if isinstance(assets, Asset):
            return any((r.is_restricted(assets, dt) for r in self.sub_restrictions))
        return reduce(operator.or_, (r.is_restricted(assets, dt) for r in self.sub_restrictions))

class NoRestrictions(Restrictions):
    """
    A no-op restrictions that contains no restrictions.
    """

    def is_restricted(self, assets, dt):
        if False:
            i = 10
            return i + 15
        if isinstance(assets, Asset):
            return False
        return pd.Series(index=pd.Index(assets), data=False)

class StaticRestrictions(Restrictions):
    """
    Static restrictions stored in memory that are constant regardless of dt
    for each asset.

    Parameters
    ----------
    restricted_list : iterable of assets
        The assets to be restricted
    """

    def __init__(self, restricted_list):
        if False:
            print('Hello World!')
        self._restricted_set = frozenset(restricted_list)

    def is_restricted(self, assets, dt):
        if False:
            return 10
        '\n        An asset is restricted for all dts if it is in the static list.\n        '
        if isinstance(assets, Asset):
            return assets in self._restricted_set
        return pd.Series(index=pd.Index(assets), data=vectorized_is_element(assets, self._restricted_set))

class HistoricalRestrictions(Restrictions):
    """
    Historical restrictions stored in memory with effective dates for each
    asset.

    Parameters
    ----------
    restrictions : iterable of namedtuple Restriction
        The restrictions, each defined by an asset, effective date and state
    """

    def __init__(self, restrictions):
        if False:
            return 10
        self._restrictions_by_asset = {asset: sorted(restrictions_for_asset, key=lambda x: x.effective_date) for (asset, restrictions_for_asset) in iteritems(groupby(lambda x: x.asset, restrictions))}

    def is_restricted(self, assets, dt):
        if False:
            return 10
        '\n        Returns whether or not an asset or iterable of assets is restricted\n        on a dt.\n        '
        if isinstance(assets, Asset):
            return self._is_restricted_for_asset(assets, dt)
        is_restricted = partial(self._is_restricted_for_asset, dt=dt)
        return pd.Series(index=pd.Index(assets), data=vectorize(is_restricted, otypes=[bool])(assets))

    def _is_restricted_for_asset(self, asset, dt):
        if False:
            print('Hello World!')
        state = RESTRICTION_STATES.ALLOWED
        for r in self._restrictions_by_asset.get(asset, ()):
            if r.effective_date > dt:
                break
            state = r.state
        return state == RESTRICTION_STATES.FROZEN

class SecurityListRestrictions(Restrictions):
    """
    Restrictions based on a security list.

    Parameters
    ----------
    restrictions : zipline.utils.security_list.SecurityList
        The restrictions defined by a SecurityList
    """

    def __init__(self, security_list_by_dt):
        if False:
            return 10
        self.current_securities = security_list_by_dt.current_securities

    def is_restricted(self, assets, dt):
        if False:
            for i in range(10):
                print('nop')
        securities_in_list = self.current_securities(dt)
        if isinstance(assets, Asset):
            return assets in securities_in_list
        return pd.Series(index=pd.Index(assets), data=vectorized_is_element(assets, securities_in_list))