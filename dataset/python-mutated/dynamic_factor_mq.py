"""
Dynamic factor model.

Author: Chad Fulton
License: BSD-3
"""
from collections import OrderedDict
from warnings import warn
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.validation import int_like
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.multivariate.pca import PCA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace._quarterly_ar1 import QuarterlyAR1
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tools.tools import Bunch
from statsmodels.tools.validation import string_like
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.statespace import mlemodel, initialization
from statsmodels.tsa.statespace.tools import companion_matrix, is_invertible, constrain_stationary_univariate, constrain_stationary_multivariate, unconstrain_stationary_univariate, unconstrain_stationary_multivariate
from statsmodels.tsa.statespace.kalman_smoother import SMOOTHER_STATE, SMOOTHER_STATE_COV, SMOOTHER_STATE_AUTOCOV
from statsmodels.base.data import PandasData
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.tableformatting import fmt_params

class FactorBlock(dict):
    """
    Helper class for describing and indexing a block of factors.

    Parameters
    ----------
    factor_names : tuple of str
        Tuple of factor names in the block (in the order that they will appear
        in the state vector).
    factor_order : int
        Order of the vector autoregression governing the factor block dynamics.
    endog_factor_map : pd.DataFrame
        Mapping from endog variable names to factor names.
    state_offset : int
        Offset of this factor block in the state vector.
    has_endog_Q : bool
        Flag if the model contains quarterly data.

    Notes
    -----
    The goal of this class is, in particular, to make it easier to retrieve
    indexes of subsets of the state vector that are associated with a
    particular block of factors.

    - `factors_ix` is a matrix of indices, with rows corresponding to factors
      in the block and columns corresponding to lags
    - `factors` is vec(factors_ix) (i.e. it stacks columns, so that it is
      `factors_ix.ravel(order='F')`). Thinking about a VAR system, the first
       k*p elements correspond to the equation for the first variable. The next
       k*p elements correspond to the equation for the second variable, and so
       on. It contains all of the lags in the state vector, which is max(5, p)
    - `factors_ar` is the subset of `factors` that have nonzero coefficients,
      so it contains lags up to p.
    - `factors_L1` only contains the first lag of the factors
    - `factors_L1_5` contains the first - fifth lags of the factors

    """

    def __init__(self, factor_names, factor_order, endog_factor_map, state_offset, k_endog_Q):
        if False:
            print('Hello World!')
        self.factor_names = factor_names
        self.k_factors = len(self.factor_names)
        self.factor_order = factor_order
        self.endog_factor_map = endog_factor_map.loc[:, factor_names]
        self.state_offset = state_offset
        self.k_endog_Q = k_endog_Q
        if self.k_endog_Q > 0:
            self._factor_order = max(5, self.factor_order)
        else:
            self._factor_order = self.factor_order
        self.k_states = self.k_factors * self._factor_order
        self['factors'] = self.factors
        self['factors_ar'] = self.factors_ar
        self['factors_ix'] = self.factors_ix
        self['factors_L1'] = self.factors_L1
        self['factors_L1_5'] = self.factors_L1_5

    @property
    def factors_ix(self):
        if False:
            return 10
        'Factor state index array, shaped (k_factors, lags).'
        o = self.state_offset
        return np.reshape(o + np.arange(self.k_factors * self._factor_order), (self._factor_order, self.k_factors)).T

    @property
    def factors(self):
        if False:
            print('Hello World!')
        'Factors and all lags in the state vector (max(5, p)).'
        o = self.state_offset
        return np.s_[o:o + self.k_factors * self._factor_order]

    @property
    def factors_ar(self):
        if False:
            while True:
                i = 10
        'Factors and all lags used in the factor autoregression (p).'
        o = self.state_offset
        return np.s_[o:o + self.k_factors * self.factor_order]

    @property
    def factors_L1(self):
        if False:
            return 10
        'Factors (first block / lag only).'
        o = self.state_offset
        return np.s_[o:o + self.k_factors]

    @property
    def factors_L1_5(self):
        if False:
            i = 10
            return i + 15
        'Factors plus four lags.'
        o = self.state_offset
        return np.s_[o:o + self.k_factors * 5]

class DynamicFactorMQStates(dict):
    """
    Helper class for describing and indexing the state vector.

    Parameters
    ----------
    k_endog_M : int
        Number of monthly (or non-time-specific, if k_endog_Q=0) variables.
    k_endog_Q : int
        Number of quarterly variables.
    endog_names : list
        Names of the endogenous variables.
    factors : int, list, or dict
        Integer giving the number of (global) factors, a list with the names of
        (global) factors, or a dictionary with:

        - keys : names of endogenous variables
        - values : lists of factor names.

        If this is an integer, then the factor names will be 0, 1, ....
    factor_orders : int or dict
        Integer describing the order of the vector autoregression (VAR)
        governing all factor block dynamics or dictionary with:

        - keys : factor name or tuples of factor names in a block
        - values : integer describing the VAR order for that factor block

        If a dictionary, this defines the order of the factor blocks in the
        state vector. Otherwise, factors are ordered so that factors that load
        on more variables come first (and then alphabetically, to break ties).
    factor_multiplicities : int or dict
        This argument provides a convenient way to specify multiple factors
        that load identically on variables. For example, one may want two
        "global" factors (factors that load on all variables) that evolve
        jointly according to a VAR. One could specify two global factors in the
        `factors` argument and specify that they are in the same block in the
        `factor_orders` argument, but it is easier to specify a single global
        factor in the `factors` argument, and set the order in the
        `factor_orders` argument, and then set the factor multiplicity to 2.

        This argument must be an integer describing the factor multiplicity for
        all factors or dictionary with:

        - keys : factor name
        - values : integer describing the factor multiplicity for the factors
          in the given block
    idiosyncratic_ar1 : bool
        Whether or not to model the idiosyncratic component for each series as
        an AR(1) process. If False, the idiosyncratic component is instead
        modeled as white noise.

    Attributes
    ----------
    k_endog : int
        Total number of endogenous variables.
    k_states : int
        Total number of state variables (those associated with the factors and
        those associated with the idiosyncratic disturbances).
    k_posdef : int
        Total number of state disturbance terms (those associated with the
        factors and those associated with the idiosyncratic disturbances).
    k_endog_M : int
        Number of monthly (or non-time-specific, if k_endog_Q=0) variables.
    k_endog_Q : int
        Number of quarterly variables.
    k_factors : int
        Total number of factors. Note that factor multiplicities will have
        already been expanded.
    k_states_factors : int
        The number of state variables associated with factors (includes both
        factors and lags of factors included in the state vector).
    k_posdef_factors : int
        The number of state disturbance terms associated with factors.
    k_states_idio : int
        Total number of state variables associated with idiosyncratic
        disturbances.
    k_posdef_idio : int
        Total number of state disturbance terms associated with idiosyncratic
        disturbances.
    k_states_idio_M : int
        The number of state variables associated with idiosyncratic
        disturbances for monthly (or non-time-specific if there are no
        quarterly variables) variables. If the disturbances are AR(1), then
        this will be equal to `k_endog_M`, otherwise it will be equal to zero.
    k_states_idio_Q : int
        The number of state variables associated with idiosyncratic
        disturbances for quarterly variables. This will always be equal to
        `k_endog_Q * 5`, even if the disturbances are not AR(1).
    k_posdef_idio_M : int
        The number of state disturbance terms associated with idiosyncratic
        disturbances for monthly (or non-time-specific if there are no
        quarterly variables) variables. If the disturbances are AR(1), then
        this will be equal to `k_endog_M`, otherwise it will be equal to zero.
    k_posdef_idio_Q : int
        The number of state disturbance terms associated with idiosyncratic
        disturbances for quarterly variables. This will always be equal to
        `k_endog_Q`, even if the disturbances are not AR(1).
    idiosyncratic_ar1 : bool
        Whether or not to model the idiosyncratic component for each series as
        an AR(1) process.
    factor_blocks : list of FactorBlock
        List of `FactorBlock` helper instances for each factor block.
    factor_names : list of str
        List of factor names.
    factors : dict
        Dictionary with:

        - keys : names of endogenous variables
        - values : lists of factor names.

        Note that factor multiplicities will have already been expanded.
    factor_orders : dict
        Dictionary with:

        - keys : tuple of factor names
        - values : integer describing autoregression order

        Note that factor multiplicities will have already been expanded.
    max_factor_order : int
        Maximum autoregression order across all factor blocks.
    factor_block_orders : pd.Series
        Series containing lag orders, with the factor block (a tuple of factor
        names) as the index.
    factor_multiplicities : dict
        Dictionary with:

        - keys : factor name
        - values : integer describing the factor multiplicity for the factors
          in the given block
    endog_factor_map : dict
        Dictionary with:

        - keys : endog name
        - values : list of factor names
    loading_counts : pd.Series
        Series containing number of endogenous variables loading on each
        factor, with the factor name as the index.
    block_loading_counts : dict
        Dictionary with:

        - keys : tuple of factor names
        - values : average number of endogenous variables loading on the block
          (note that average is over the factors in the block)

    Notes
    -----
    The goal of this class is, in particular, to make it easier to retrieve
    indexes of subsets of the state vector.

    Note that the ordering of the factor blocks in the state vector is
    determined by the `factor_orders` argument if a dictionary. Otherwise,
    factors are ordered so that factors that load on more variables come first
    (and then alphabetically, to break ties).

    - `factors_L1` is an array with the indexes of first lag of the factors
      from each block. Ordered first by block, and then by lag.
    - `factors_L1_5` is an array with the indexes contains the first - fifth
      lags of the factors from each block. Ordered first by block, and then by
      lag.
    - `factors_L1_5_ix` is an array shaped (5, k_factors) with the indexes
      of the first - fifth lags of the factors from each block.
    - `idio_ar_L1` is an array with the indexes of the first lag of the
      idiosyncratic AR states, both monthly (if appliable) and quarterly.
    - `idio_ar_M` is a slice with the indexes of the idiosyncratic disturbance
      states for the monthly (or non-time-specific if there are no quarterly
      variables) variables. It is an empty slice if
      `idiosyncratic_ar1 = False`.
    - `idio_ar_Q` is a slice with the indexes of the idiosyncratic disturbance
      states and all lags, for the quarterly variables. It is an empty slice if
      there are no quarterly variable.
    - `idio_ar_Q_ix` is an array shaped (k_endog_Q, 5) with the indexes of the
      first - fifth lags of the idiosyncratic disturbance states for the
      quarterly variables.
    - `endog_factor_iloc` is a list of lists, with entries for each endogenous
      variable. The entry for variable `i`, `endog_factor_iloc[i]` is a list of
      indexes of the factors that variable `i` loads on. This does not include
      any lags, but it can be used with e.g. `factors_L1_5_ix` to get lags.

    """

    def __init__(self, k_endog_M, k_endog_Q, endog_names, factors, factor_orders, factor_multiplicities, idiosyncratic_ar1):
        if False:
            print('Hello World!')
        self.k_endog_M = k_endog_M
        self.k_endog_Q = k_endog_Q
        self.k_endog = self.k_endog_M + self.k_endog_Q
        self.idiosyncratic_ar1 = idiosyncratic_ar1
        factors_is_int = np.issubdtype(type(factors), np.integer)
        factors_is_list = isinstance(factors, (list, tuple))
        orders_is_int = np.issubdtype(type(factor_orders), np.integer)
        if factor_multiplicities is None:
            factor_multiplicities = 1
        mult_is_int = np.issubdtype(type(factor_multiplicities), np.integer)
        if not (factors_is_int or factors_is_list or isinstance(factors, dict)):
            raise ValueError('`factors` argument must an integer number of factors, a list of global factor names, or a dictionary, mapping observed variables to factors.')
        if not (orders_is_int or isinstance(factor_orders, dict)):
            raise ValueError('`factor_orders` argument must either be an integer or a dictionary.')
        if not (mult_is_int or isinstance(factor_multiplicities, dict)):
            raise ValueError('`factor_multiplicities` argument must either be an integer or a dictionary.')
        if factors_is_int or factors_is_list:
            if factors_is_int and factors == 0 or (factors_is_list and len(factors) == 0):
                raise ValueError('The model must contain at least one factor.')
            if factors_is_list:
                factor_names = list(factors)
            else:
                factor_names = [f'{i}' for i in range(factors)]
            factors = {name: factor_names[:] for name in endog_names}
        _factor_names = []
        for val in factors.values():
            _factor_names.extend(val)
        factor_names = set(_factor_names)
        if orders_is_int:
            factor_orders = {factor_name: factor_orders for factor_name in factor_names}
        if mult_is_int:
            factor_multiplicities = {factor_name: factor_multiplicities for factor_name in factor_names}
        (factors, factor_orders) = self._apply_factor_multiplicities(factors, factor_orders, factor_multiplicities)
        self.factors = factors
        self.factor_orders = factor_orders
        self.factor_multiplicities = factor_multiplicities
        self.endog_factor_map = self._construct_endog_factor_map(factors, endog_names)
        self.k_factors = self.endog_factor_map.shape[1]
        if self.k_factors > self.k_endog_M:
            raise ValueError(f'Number of factors ({self.k_factors}) cannot be greater than the number of monthly endogenous variables ({self.k_endog_M}).')
        self.loading_counts = self.endog_factor_map.sum(axis=0).rename('count').reset_index().sort_values(['count', 'factor'], ascending=[False, True]).set_index('factor')
        block_loading_counts = {block: np.atleast_1d(self.loading_counts.loc[list(block), 'count']).mean(axis=0) for block in factor_orders.keys()}
        ix = pd.Index(block_loading_counts.keys(), tupleize_cols=False, name='block')
        self.block_loading_counts = pd.Series(list(block_loading_counts.values()), index=ix, name='count').to_frame().sort_values(['count', 'block'], ascending=[False, True])['count']
        ix = pd.Index(factor_orders.keys(), tupleize_cols=False, name='block')
        self.factor_block_orders = pd.Series(list(factor_orders.values()), index=ix, name='order')
        if orders_is_int:
            keys = self.block_loading_counts.keys()
            self.factor_block_orders = self.factor_block_orders.loc[keys]
            self.factor_block_orders.index.name = 'block'
        factor_names = pd.Series(np.concatenate(list(self.factor_block_orders.index)))
        missing = [name for name in self.endog_factor_map.columns if name not in factor_names.tolist()]
        if len(missing):
            ix = pd.Index([(factor_name,) for factor_name in missing], tupleize_cols=False, name='block')
            default_block_orders = pd.Series(np.ones(len(ix), dtype=int), index=ix, name='order')
            self.factor_block_orders = self.factor_block_orders.append(default_block_orders)
            factor_names = pd.Series(np.concatenate(list(self.factor_block_orders.index)))
        duplicates = factor_names.duplicated()
        if duplicates.any():
            duplicate_names = set(factor_names[duplicates])
            raise ValueError(f'Each factor can be assigned to at most one block of factors in `factor_orders`. Duplicate entries for {duplicate_names}')
        self.factor_names = factor_names.tolist()
        self.max_factor_order = np.max(self.factor_block_orders)
        self.endog_factor_map = self.endog_factor_map.loc[endog_names, factor_names]
        self.k_states_factors = 0
        self.k_posdef_factors = 0
        state_offset = 0
        self.factor_blocks = []
        for (factor_names, factor_order) in self.factor_block_orders.items():
            block = FactorBlock(factor_names, factor_order, self.endog_factor_map, state_offset, self.k_endog_Q)
            self.k_states_factors += block.k_states
            self.k_posdef_factors += block.k_factors
            state_offset += block.k_states
            self.factor_blocks.append(block)
        self.k_states_idio_M = self.k_endog_M if idiosyncratic_ar1 else 0
        self.k_states_idio_Q = self.k_endog_Q * 5
        self.k_states_idio = self.k_states_idio_M + self.k_states_idio_Q
        self.k_posdef_idio_M = self.k_endog_M if self.idiosyncratic_ar1 else 0
        self.k_posdef_idio_Q = self.k_endog_Q
        self.k_posdef_idio = self.k_posdef_idio_M + self.k_posdef_idio_Q
        self.k_states = self.k_states_factors + self.k_states_idio
        self.k_posdef = self.k_posdef_factors + self.k_posdef_idio
        self._endog_factor_iloc = None

    def _apply_factor_multiplicities(self, factors, factor_orders, factor_multiplicities):
        if False:
            i = 10
            return i + 15
        '\n        Expand `factors` and `factor_orders` to account for factor multiplity.\n\n        For example, if there is a `global` factor with multiplicity 2, then\n        this method expands that into `global.1` and `global.2` in both the\n        `factors` and `factor_orders` dictionaries.\n\n        Parameters\n        ----------\n        factors : dict\n            Dictionary of {endog_name: list of factor names}\n        factor_orders : dict\n            Dictionary of {tuple of factor names: factor order}\n        factor_multiplicities : dict\n            Dictionary of {factor name: factor multiplicity}\n\n        Returns\n        -------\n        new_factors : dict\n            Dictionary of {endog_name: list of factor names}, with factor names\n            expanded to incorporate multiplicities.\n        new_factors : dict\n            Dictionary of {tuple of factor names: factor order}, with factor\n            names in each tuple expanded to incorporate multiplicities.\n        '
        new_factors = {}
        for (endog_name, factors_list) in factors.items():
            new_factor_list = []
            for factor_name in factors_list:
                n = factor_multiplicities.get(factor_name, 1)
                if n > 1:
                    new_factor_list += [f'{factor_name}.{i + 1}' for i in range(n)]
                else:
                    new_factor_list.append(factor_name)
            new_factors[endog_name] = new_factor_list
        new_factor_orders = {}
        for (block, factor_order) in factor_orders.items():
            if not isinstance(block, tuple):
                block = (block,)
            new_block = []
            for factor_name in block:
                n = factor_multiplicities.get(factor_name, 1)
                if n > 1:
                    new_block += [f'{factor_name}.{i + 1}' for i in range(n)]
                else:
                    new_block += [factor_name]
            new_factor_orders[tuple(new_block)] = factor_order
        return (new_factors, new_factor_orders)

    def _construct_endog_factor_map(self, factors, endog_names):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct mapping of observed variables to factors.\n\n        Parameters\n        ----------\n        factors : dict\n            Dictionary of {endog_name: list of factor names}\n        endog_names : list of str\n            List of the names of the observed variables.\n\n        Returns\n        -------\n        endog_factor_map : pd.DataFrame\n            Boolean dataframe with `endog_names` as the index and the factor\n            names (computed from the `factors` input) as the columns. Each cell\n            is True if the associated factor is allowed to load on the\n            associated observed variable.\n\n        '
        missing = []
        for (key, value) in factors.items():
            if not isinstance(value, (list, tuple)) or len(value) == 0:
                missing.append(key)
        if len(missing):
            raise ValueError(f'Each observed variable must be mapped to at least one factor in the `factors` dictionary. Variables missing factors are: {missing}.')
        missing = set(endog_names).difference(set(factors.keys()))
        if len(missing):
            raise ValueError(f'If a `factors` dictionary is provided, then it must include entries for each observed variable. Missing variables are: {missing}.')
        factor_names = {}
        for (key, value) in factors.items():
            if isinstance(value, str):
                factor_names[value] = 0
            else:
                factor_names.update({v: 0 for v in value})
        factor_names = list(factor_names.keys())
        k_factors = len(factor_names)
        endog_factor_map = pd.DataFrame(np.zeros((self.k_endog, k_factors), dtype=bool), index=pd.Index(endog_names, name='endog'), columns=pd.Index(factor_names, name='factor'))
        for (key, value) in factors.items():
            endog_factor_map.loc[key, value] = True
        return endog_factor_map

    @property
    def factors_L1(self):
        if False:
            return 10
        'Factors.'
        ix = np.arange(self.k_states_factors)
        iloc = tuple((ix[block.factors_L1] for block in self.factor_blocks))
        return np.concatenate(iloc)

    @property
    def factors_L1_5_ix(self):
        if False:
            return 10
        'Factors plus any lags, index shaped (5, k_factors).'
        ix = np.arange(self.k_states_factors)
        iloc = []
        for block in self.factor_blocks:
            iloc.append(ix[block.factors_L1_5].reshape(5, block.k_factors))
        return np.concatenate(iloc, axis=1)

    @property
    def idio_ar_L1(self):
        if False:
            print('Hello World!')
        'Idiosyncratic AR states, (first block / lag only).'
        ix1 = self.k_states_factors
        if self.idiosyncratic_ar1:
            ix2 = ix1 + self.k_endog
        else:
            ix2 = ix1 + self.k_endog_Q
        return np.s_[ix1:ix2]

    @property
    def idio_ar_M(self):
        if False:
            i = 10
            return i + 15
        'Idiosyncratic AR states for monthly variables.'
        ix1 = self.k_states_factors
        ix2 = ix1
        if self.idiosyncratic_ar1:
            ix2 += self.k_endog_M
        return np.s_[ix1:ix2]

    @property
    def idio_ar_Q(self):
        if False:
            for i in range(10):
                print('nop')
        'Idiosyncratic AR states and all lags for quarterly variables.'
        ix1 = self.k_states_factors
        if self.idiosyncratic_ar1:
            ix1 += self.k_endog_M
        ix2 = ix1 + self.k_endog_Q * 5
        return np.s_[ix1:ix2]

    @property
    def idio_ar_Q_ix(self):
        if False:
            for i in range(10):
                print('nop')
        'Idiosyncratic AR (quarterly) state index, (k_endog_Q, lags).'
        start = self.k_states_factors
        if self.idiosyncratic_ar1:
            start += self.k_endog_M
        return start + np.reshape(np.arange(5 * self.k_endog_Q), (5, self.k_endog_Q)).T

    @property
    def endog_factor_iloc(self):
        if False:
            print('Hello World!')
        'List of list of int, factor indexes for each observed variable.'
        if self._endog_factor_iloc is None:
            ilocs = []
            for i in range(self.k_endog):
                ilocs.append(np.where(self.endog_factor_map.iloc[i])[0])
            self._endog_factor_iloc = ilocs
        return self._endog_factor_iloc

    def __getitem__(self, key):
        if False:
            return 10
        '\n        Use square brackets to access index / slice elements.\n\n        This is convenient in highlighting the indexing / slice quality of\n        these attributes in the code below.\n        '
        if key in ['factors_L1', 'factors_L1_5_ix', 'idio_ar_L1', 'idio_ar_M', 'idio_ar_Q', 'idio_ar_Q_ix']:
            return getattr(self, key)
        else:
            raise KeyError(key)

class DynamicFactorMQ(mlemodel.MLEModel):
    """
    Dynamic factor model with EM algorithm; option for monthly/quarterly data.

    Implementation of the dynamic factor model of Bańbura and Modugno (2014)
    ([1]_) and Bańbura, Giannone, and Reichlin (2011) ([2]_). Uses the EM
    algorithm for parameter fitting, and so can accommodate a large number of
    left-hand-side variables. Specifications can include any collection of
    blocks of factors, including different factor autoregression orders, and
    can include AR(1) processes for idiosyncratic disturbances. Can
    incorporate monthly/quarterly mixed frequency data along the lines of
    Mariano and Murasawa (2011) ([4]_). A special case of this model is the
    Nowcasting model of Bok et al. (2017) ([3]_). Moreover, this model can be
    used to compute the news associated with updated data releases.

    Parameters
    ----------
    endog : array_like
        Observed time-series process :math:`y`. See the "Notes" section for
        details on how to set up a model with monthly/quarterly mixed frequency
        data.
    k_endog_monthly : int, optional
        If specifying a monthly/quarterly mixed frequency model in which the
        provided `endog` dataset contains both the monthly and quarterly data,
        this variable should be used to indicate how many of the variables
        are monthly. Note that when using the `k_endog_monthly` argument, the
        columns with monthly variables in `endog` should be ordered first, and
        the columns with quarterly variables should come afterwards. See the
        "Notes" section for details on how to set up a model with
        monthly/quarterly mixed frequency data.
    factors : int, list, or dict, optional
        Integer giving the number of (global) factors, a list with the names of
        (global) factors, or a dictionary with:

        - keys : names of endogenous variables
        - values : lists of factor names.

        If this is an integer, then the factor names will be 0, 1, .... The
        default is a single factor that loads on all variables. Note that there
        cannot be more factors specified than there are monthly variables.
    factor_orders : int or dict, optional
        Integer describing the order of the vector autoregression (VAR)
        governing all factor block dynamics or dictionary with:

        - keys : factor name or tuples of factor names in a block
        - values : integer describing the VAR order for that factor block

        If a dictionary, this defines the order of the factor blocks in the
        state vector. Otherwise, factors are ordered so that factors that load
        on more variables come first (and then alphabetically, to break ties).
    factor_multiplicities : int or dict, optional
        This argument provides a convenient way to specify multiple factors
        that load identically on variables. For example, one may want two
        "global" factors (factors that load on all variables) that evolve
        jointly according to a VAR. One could specify two global factors in the
        `factors` argument and specify that they are in the same block in the
        `factor_orders` argument, but it is easier to specify a single global
        factor in the `factors` argument, and set the order in the
        `factor_orders` argument, and then set the factor multiplicity to 2.

        This argument must be an integer describing the factor multiplicity for
        all factors or dictionary with:

        - keys : factor name
        - values : integer describing the factor multiplicity for the factors
          in the given block

    idiosyncratic_ar1 : bool
        Whether or not to model the idiosyncratic component for each series as
        an AR(1) process. If False, the idiosyncratic component is instead
        modeled as white noise.
    standardize : bool or tuple, optional
        If a boolean, whether or not to standardize each endogenous variable to
        have mean zero and standard deviation 1 before fitting the model. See
        "Notes" for details about how this option works with postestimation
        output. If a tuple (usually only used internally), then the tuple must
        have length 2, with each element containing a Pandas series with index
        equal to the names of the endogenous variables. The first element
        should contain the mean values and the second element should contain
        the standard deviations. Default is True.
    endog_quarterly : pandas.Series or pandas.DataFrame
        Observed quarterly variables. If provided, must be a Pandas Series or
        DataFrame with a DatetimeIndex or PeriodIndex at the quarterly
        frequency. See the "Notes" section for details on how to set up a model
        with monthly/quarterly mixed frequency data.
    init_t0 : bool, optional
        If True, this option initializes the Kalman filter with the
        distribution for :math:`\\alpha_0` rather than :math:`\\alpha_1`. See
        the "Notes" section for more details. This option is rarely used except
        for testing. Default is False.
    obs_cov_diag : bool, optional
        If True and if `idiosyncratic_ar1 is True`, then this option puts small
        positive values in the observation disturbance covariance matrix. This
        is not required for estimation and is rarely used except for testing.
        (It is sometimes used to prevent numerical errors, for example those
        associated with a positive semi-definite forecast error covariance
        matrix at the first time step when using EM initialization, but state
        space models in Statsmodels switch to the univariate approach in those
        cases, and so do not need to use this trick). Default is False.

    Notes
    -----
    The basic model is:

    .. math::

        y_t & = \\Lambda f_t + \\epsilon_t \\\\
        f_t & = A_1 f_{t-1} + \\dots + A_p f_{t-p} + u_t

    where:

    - :math:`y_t` is observed data at time t
    - :math:`\\epsilon_t` is idiosyncratic disturbance at time t (see below for
      details, including modeling serial correlation in this term)
    - :math:`f_t` is the unobserved factor at time t
    - :math:`u_t \\sim N(0, Q)` is the factor disturbance at time t

    and:

    - :math:`\\Lambda` is referred to as the matrix of factor loadings
    - :math:`A_i` are matrices of autoregression coefficients

    Furthermore, we allow the idiosyncratic disturbances to be serially
    correlated, so that, if `idiosyncratic_ar1=True`,
    :math:`\\epsilon_{i,t} = \\rho_i \\epsilon_{i,t-1} + e_{i,t}`, where
    :math:`e_{i,t} \\sim N(0, \\sigma_i^2)`. If `idiosyncratic_ar1=False`,
    then we instead have :math:`\\epsilon_{i,t} = e_{i,t}`.

    This basic setup can be found in [1]_, [2]_, [3]_, and [4]_.

    We allow for two generalizations of this model:

    1. Following [2]_, we allow multiple "blocks" of factors, which are
       independent from the other blocks of factors. Different blocks can be
       set to load on different subsets of the observed variables, and can be
       specified with different lag orders.
    2. Following [4]_ and [2]_, we allow mixed frequency models in which both
       monthly and quarterly data are used. See the section on "Mixed frequency
       models", below, for more details.

    Additional notes:

    - The observed data may contain arbitrary patterns of missing entries.

    **EM algorithm**

    This model contains a potentially very large number of parameters, and it
    can be difficult and take a prohibitively long time to numerically optimize
    the likelihood function using quasi-Newton methods. Instead, the default
    fitting method in this model uses the EM algorithm, as detailed in [1]_.
    As a result, the model can accommodate datasets with hundreds of
    observed variables.

    **Mixed frequency data**

    This model can handle mixed frequency data in two ways. In this section,
    we only briefly describe this, and refer readers to [2]_ and [4]_ for all
    details.

    First, because there can be arbitrary patterns of missing data in the
    observed vector, one can simply include lower frequency variables as
    observed in a particular higher frequency period, and missing otherwise.
    For example, in a monthly model, one could include quarterly data as
    occurring on the third month of each quarter. To use this method, one
    simply needs to combine the data into a single dataset at the higher
    frequency that can be passed to this model as the `endog` argument.
    However, depending on the type of variables used in the analysis and the
    assumptions about the data generating process, this approach may not be
    valid.

    For example, suppose that we are interested in the growth rate of real GDP,
    which is measured at a quarterly frequency. If the basic factor model is
    specified at a monthly frequency, then the quarterly growth rate in the
    third month of each quarter -- which is what we actually observe -- is
    approximated by a particular weighted average of unobserved monthly growth
    rates. We need to take this particular weight moving average into account
    in constructing our model, and this is what the second approach does.

    The second approach follows [2]_ and [4]_ in constructing a state space
    form to explicitly model the quarterly growth rates in terms of the
    unobserved monthly growth rates. To use this approach, there are two
    methods:

    1. Combine the monthly and quarterly data into a single dataset at the
       monthly frequency, with the monthly data in the first columns and the
       quarterly data in the last columns. Pass this dataset to the model as
       the `endog` argument and give the number of the variables that are
       monthly as the `k_endog_monthly` argument.
    2. Construct a monthly dataset as a Pandas DataFrame with a DatetimeIndex
       or PeriodIndex at the monthly frequency and separately construct a
       quarterly dataset as a Pandas DataFrame with a DatetimeIndex or
       PeriodIndex at the quarterly frequency. Pass the monthly DataFrame to
       the model as the `endog` argument and pass the quarterly DataFrame to
       the model as the `endog_quarterly` argument.

    Note that this only incorporates one particular type of mixed frequency
    data. See also Banbura et al. (2013). "Now-Casting and the Real-Time Data
    Flow." for discussion about other types of mixed frequency data that are
    not supported by this framework.

    **Nowcasting and the news**

    Through its support for monthly/quarterly mixed frequency data, this model
    can allow for the nowcasting of quarterly variables based on monthly
    observations. In particular, [2]_ and [3]_ use this model to construct
    nowcasts of real GDP and analyze the impacts of "the news", derived from
    incoming data on a real-time basis. This latter functionality can be
    accessed through the `news` method of the results object.

    **Standardizing data**

    As is often the case in formulating a dynamic factor model, we do not
    explicitly account for the mean of each observed variable. Instead, the
    default behavior is to standardize each variable prior to estimation. Thus
    if :math:`y_t` are the given observed data, the dynamic factor model is
    actually estimated on the standardized data defined by:

    .. math::

        x_{i, t} = (y_{i, t} - \\bar y_i) / s_i

    where :math:`\\bar y_i` is the sample mean and :math:`s_i` is the sample
    standard deviation.

    By default, if standardization is applied prior to estimation, results such
    as in-sample predictions, out-of-sample forecasts, and the computation of
    the "news"  are reported in the scale of the original data (i.e. the model
    output has the reverse transformation applied before it is returned to the
    user).

    Standardization can be disabled by passing `standardization=False` to the
    model constructor.

    **Identification of factors and loadings**

    The estimated factors and the factor loadings in this model are only
    identified up to an invertible transformation. As described in (the working
    paper version of) [2]_, while it is possible to impose normalizations to
    achieve identification, the EM algorithm does will converge regardless.
    Moreover, for nowcasting and forecasting purposes, identification is not
    required. This model does not impose any normalization to identify the
    factors and the factor loadings.

    **Miscellaneous**

    There are two arguments available in the model constructor that are rarely
    used but which deserve a brief mention: `init_t0` and `obs_cov_diag`. These
    arguments are provided to allow exactly matching the output of other
    packages that have slight differences in how the underlying state space
    model is set up / applied.

    - `init_t0`: state space models in Statsmodels follow Durbin and Koopman in
      initializing the model with :math:`\\alpha_1 \\sim N(a_1, P_1)`. Other
      implementations sometimes initialize instead with
      :math:`\\alpha_0 \\sim N(a_0, P_0)`. We can accommodate this by prepending
      a row of NaNs to the observed dataset.
    - `obs_cov_diag`: the state space form in [1]_ incorporates non-zero (but
      very small) diagonal elements for the observation disturbance covariance
      matrix.

    Examples
    --------
    Constructing and fitting a `DynamicFactorMQ` model.

    >>> data = sm.datasets.macrodata.load_pandas().data.iloc[-100:]
    >>> data.index = pd.period_range(start='1984Q4', end='2009Q3', freq='Q')
    >>> endog = data[['infl', 'tbilrate']].resample('M').last()
    >>> endog_Q = np.log(data[['realgdp', 'realcons']]).diff().iloc[1:] * 400

    **Basic usage**

    In the simplest case, passing only the `endog` argument results in a model
    with a single factor that follows an AR(1) process. Note that because we
    are not also providing an `endog_quarterly` dataset, `endog` can be a numpy
    array or Pandas DataFrame with any index (it does not have to be monthly).

    The `summary` method can be useful in checking the model specification.

    >>> mod = sm.tsa.DynamicFactorMQ(endog)
    >>> print(mod.summary())
                        Model Specification: Dynamic Factor Model
    ==========================================================================
    Model:         Dynamic Factor Model   # of monthly variables:          2
                + 1 factors in 1 blocks   # of factors:                    1
                  + AR(1) idiosyncratic   Idiosyncratic disturbances:  AR(1)
    Sample:                     1984-10   Standardize variables:        True
                              - 2009-09
    Observed variables / factor loadings
    ========================
    Dep. variable          0
    ------------------------
             infl          X
         tbilrate          X
        Factor blocks:
    =====================
         block      order
    ---------------------
             0          1
    =====================

    **Factors**

    With `factors=2`, there will be two independent factors that will each
    evolve according to separate AR(1) processes.

    >>> mod = sm.tsa.DynamicFactorMQ(endog, factors=2)
    >>> print(mod.summary())
                        Model Specification: Dynamic Factor Model
    ==========================================================================
    Model:         Dynamic Factor Model   # of monthly variables:          2
                + 2 factors in 2 blocks   # of factors:                    2
                  + AR(1) idiosyncratic   Idiosyncratic disturbances:  AR(1)
    Sample:                     1984-10   Standardize variables:        True
                              - 2009-09
    Observed variables / factor loadings
    ===================================
    Dep. variable          0          1
    -----------------------------------
             infl          X          X
         tbilrate          X          X
        Factor blocks:
    =====================
         block      order
    ---------------------
             0          1
             1          1
    =====================

    **Factor multiplicities**

    By instead specifying `factor_multiplicities=2`, we would still have two
    factors, but they would be dependent and would evolve jointly according
    to a VAR(1) process.

    >>> mod = sm.tsa.DynamicFactorMQ(endog, factor_multiplicities=2)
    >>> print(mod.summary())
                        Model Specification: Dynamic Factor Model
    ==========================================================================
    Model:         Dynamic Factor Model   # of monthly variables:          2
                + 2 factors in 1 blocks   # of factors:                    2
                  + AR(1) idiosyncratic   Idiosyncratic disturbances:  AR(1)
    Sample:                     1984-10   Standardize variables:        True
                              - 2009-09
    Observed variables / factor loadings
    ===================================
    Dep. variable        0.1        0.2
    -----------------------------------
             infl         X          X
         tbilrate         X          X
        Factor blocks:
    =====================
         block      order
    ---------------------
      0.1, 0.2          1
    =====================

    **Factor orders**

    In either of the above cases, we could extend the order of the (vector)
    autoregressions by using the `factor_orders` argument. For example, the
    below model would contain two independent factors that each evolve
    according to a separate AR(2) process:

    >>> mod = sm.tsa.DynamicFactorMQ(endog, factors=2, factor_orders=2)
    >>> print(mod.summary())
                        Model Specification: Dynamic Factor Model
    ==========================================================================
    Model:         Dynamic Factor Model   # of monthly variables:          2
                + 2 factors in 2 blocks   # of factors:                    2
                  + AR(1) idiosyncratic   Idiosyncratic disturbances:  AR(1)
    Sample:                     1984-10   Standardize variables:        True
                              - 2009-09
    Observed variables / factor loadings
    ===================================
    Dep. variable          0          1
    -----------------------------------
             infl          X          X
         tbilrate          X          X
        Factor blocks:
    =====================
         block      order
    ---------------------
             0          2
             1          2
    =====================

    **Serial correlation in the idiosyncratic disturbances**

    By default, the model allows each idiosyncratic disturbance terms to evolve
    according to an AR(1) process. If preferred, they can instead be specified
    to be serially independent by passing `ididosyncratic_ar1=False`.

    >>> mod = sm.tsa.DynamicFactorMQ(endog, idiosyncratic_ar1=False)
    >>> print(mod.summary())
                        Model Specification: Dynamic Factor Model
    ==========================================================================
    Model:         Dynamic Factor Model   # of monthly variables:          2
                + 1 factors in 1 blocks   # of factors:                    1
                    + iid idiosyncratic   Idiosyncratic disturbances:    iid
    Sample:                     1984-10   Standardize variables:        True
                              - 2009-09
    Observed variables / factor loadings
    ========================
    Dep. variable          0
    ------------------------
             infl          X
         tbilrate          X
        Factor blocks:
    =====================
         block      order
    ---------------------
             0          1
    =====================

    *Monthly / Quarterly mixed frequency*

    To specify a monthly / quarterly mixed frequency model see the (Notes
    section for more details about these models):

    >>> mod = sm.tsa.DynamicFactorMQ(endog, endog_quarterly=endog_Q)
    >>> print(mod.summary())
                        Model Specification: Dynamic Factor Model
    ==========================================================================
    Model:         Dynamic Factor Model   # of monthly variables:          2
                + 1 factors in 1 blocks   # of quarterly variables:        2
                + Mixed frequency (M/Q)   # of factors:                    1
                  + AR(1) idiosyncratic   Idiosyncratic disturbances:  AR(1)
    Sample:                     1984-10   Standardize variables:        True
                              - 2009-09
    Observed variables / factor loadings
    ========================
    Dep. variable          0
    ------------------------
             infl          X
         tbilrate          X
          realgdp          X
         realcons          X
        Factor blocks:
    =====================
         block      order
    ---------------------
             0          1
    =====================

    *Customize observed variable / factor loadings*

    To specify that certain that certain observed variables only load on
    certain factors, it is possible to pass a dictionary to the `factors`
    argument.

    >>> factors = {'infl': ['global']
    ...            'tbilrate': ['global']
    ...            'realgdp': ['global', 'real']
    ...            'realcons': ['global', 'real']}
    >>> mod = sm.tsa.DynamicFactorMQ(endog, endog_quarterly=endog_Q)
    >>> print(mod.summary())
                        Model Specification: Dynamic Factor Model
    ==========================================================================
    Model:         Dynamic Factor Model   # of monthly variables:          2
                + 2 factors in 2 blocks   # of quarterly variables:        2
                + Mixed frequency (M/Q)   # of factor blocks:              2
                  + AR(1) idiosyncratic   Idiosyncratic disturbances:  AR(1)
    Sample:                     1984-10   Standardize variables:        True
                              - 2009-09
    Observed variables / factor loadings
    ===================================
    Dep. variable     global       real
    -----------------------------------
             infl       X
         tbilrate       X
          realgdp       X           X
         realcons       X           X
        Factor blocks:
    =====================
         block      order
    ---------------------
        global          1
          real          1
    =====================

    **Fitting parameters**

    To fit the model, use the `fit` method. This method uses the EM algorithm
    by default.

    >>> mod = sm.tsa.DynamicFactorMQ(endog)
    >>> res = mod.fit()
    >>> print(res.summary())
                              Dynamic Factor Results
    ==========================================================================
    Dep. Variable:      ['infl', 'tbilrate']   No. Observations:         300
    Model:              Dynamic Factor Model   Log Likelihood       -127.909
                     + 1 factors in 1 blocks   AIC                   271.817
                       + AR(1) idiosyncratic   BIC                   301.447
    Date:                   Tue, 04 Aug 2020   HQIC                  283.675
    Time:                           15:59:11   EM Iterations              83
    Sample:                       10-31-1984
                                - 09-30-2009
    Covariance Type:            Not computed
                        Observation equation:
    ==============================================================
    Factor loadings:          0    idiosyncratic: AR(1)       var.
    --------------------------------------------------------------
                infl      -0.67                    0.39       0.73
            tbilrate      -0.63                    0.99       0.01
           Transition: Factor block 0
    =======================================
                     L1.0    error variance
    ---------------------------------------
             0       0.98              0.01
    =======================================
    Warnings:
    [1] Covariance matrix not calculated.

    *Displaying iteration progress*

    To display information about the EM iterations, use the `disp` argument.

    >>> mod = sm.tsa.DynamicFactorMQ(endog)
    >>> res = mod.fit(disp=10)
    EM start iterations, llf=-291.21
    EM iteration 10, llf=-157.17, convergence criterion=0.053801
    EM iteration 20, llf=-128.99, convergence criterion=0.0035545
    EM iteration 30, llf=-127.97, convergence criterion=0.00010224
    EM iteration 40, llf=-127.93, convergence criterion=1.3281e-05
    EM iteration 50, llf=-127.92, convergence criterion=5.4725e-06
    EM iteration 60, llf=-127.91, convergence criterion=2.8665e-06
    EM iteration 70, llf=-127.91, convergence criterion=1.6999e-06
    EM iteration 80, llf=-127.91, convergence criterion=1.1085e-06
    EM converged at iteration 83, llf=-127.91,
       convergence criterion=9.9004e-07 < tolerance=1e-06

    **Results: forecasting, impulse responses, and more**

    One the model is fitted, there are a number of methods available from the
    results object. Some examples include:

    *Forecasting*

    >>> mod = sm.tsa.DynamicFactorMQ(endog)
    >>> res = mod.fit()
    >>> print(res.forecast(steps=5))
                 infl  tbilrate
    2009-10  1.784169  0.260401
    2009-11  1.735848  0.305981
    2009-12  1.730674  0.350968
    2010-01  1.742110  0.395369
    2010-02  1.759786  0.439194

    *Impulse responses*

    >>> mod = sm.tsa.DynamicFactorMQ(endog)
    >>> res = mod.fit()
    >>> print(res.impulse_responses(steps=5))
           infl  tbilrate
    0 -1.511956 -1.341498
    1 -1.483172 -1.315960
    2 -1.454937 -1.290908
    3 -1.427240 -1.266333
    4 -1.400069 -1.242226
    5 -1.373416 -1.218578

    For other available methods (including in-sample prediction, simulation of
    time series, extending the results to incorporate new data, and the news),
    see the documentation for state space models.

    References
    ----------
    .. [1] Bańbura, Marta, and Michele Modugno.
           "Maximum likelihood estimation of factor models on datasets with
           arbitrary pattern of missing data."
           Journal of Applied Econometrics 29, no. 1 (2014): 133-160.
    .. [2] Bańbura, Marta, Domenico Giannone, and Lucrezia Reichlin.
           "Nowcasting."
           The Oxford Handbook of Economic Forecasting. July 8, 2011.
    .. [3] Bok, Brandyn, Daniele Caratelli, Domenico Giannone,
           Argia M. Sbordone, and Andrea Tambalotti. 2018.
           "Macroeconomic Nowcasting and Forecasting with Big Data."
           Annual Review of Economics 10 (1): 615-43.
           https://doi.org/10.1146/annurev-economics-080217-053214.
    .. [4] Mariano, Roberto S., and Yasutomo Murasawa.
           "A coincident index, common factors, and monthly real GDP."
           Oxford Bulletin of Economics and Statistics 72, no. 1 (2010): 27-46.

    """

    def __init__(self, endog, k_endog_monthly=None, factors=1, factor_orders=1, factor_multiplicities=None, idiosyncratic_ar1=True, standardize=True, endog_quarterly=None, init_t0=False, obs_cov_diag=False, **kwargs):
        if False:
            while True:
                i = 10
        if endog_quarterly is not None:
            if k_endog_monthly is not None:
                raise ValueError('If `endog_quarterly` is specified, then `endog` must contain only monthly variables, and so `k_endog_monthly` cannot be specified since it will be inferred from the shape of `endog`.')
            (endog, k_endog_monthly) = self.construct_endog(endog, endog_quarterly)
        endog_is_pandas = _is_using_pandas(endog, None)
        if endog_is_pandas:
            if isinstance(endog, pd.Series):
                endog = endog.to_frame()
        elif np.ndim(endog) < 2:
            endog = np.atleast_2d(endog).T
        if k_endog_monthly is None:
            k_endog_monthly = endog.shape[1]
        if endog_is_pandas:
            endog_names = endog.columns.tolist()
        elif endog.shape[1] == 1:
            endog_names = ['y']
        else:
            endog_names = [f'y{i + 1}' for i in range(endog.shape[1])]
        self.k_endog_M = int_like(k_endog_monthly, 'k_endog_monthly')
        self.k_endog_Q = endog.shape[1] - self.k_endog_M
        s = self._s = DynamicFactorMQStates(self.k_endog_M, self.k_endog_Q, endog_names, factors, factor_orders, factor_multiplicities, idiosyncratic_ar1)
        self.factors = factors
        self.factor_orders = factor_orders
        self.factor_multiplicities = factor_multiplicities
        self.endog_factor_map = self._s.endog_factor_map
        self.factor_block_orders = self._s.factor_block_orders
        self.factor_names = self._s.factor_names
        self.k_factors = self._s.k_factors
        self.k_factor_blocks = len(self.factor_block_orders)
        self.max_factor_order = self._s.max_factor_order
        self.idiosyncratic_ar1 = idiosyncratic_ar1
        self.init_t0 = init_t0
        self.obs_cov_diag = obs_cov_diag
        if self.init_t0:
            if endog_is_pandas:
                ix = pd.period_range(endog.index[0] - 1, endog.index[-1], freq=endog.index.freq)
                endog = endog.reindex(ix)
            else:
                endog = np.c_[[np.nan] * endog.shape[1], endog.T].T
        if isinstance(standardize, tuple) and len(standardize) == 2:
            (endog_mean, endog_std) = standardize
            n = endog.shape[1]
            if isinstance(endog_mean, pd.Series) and (not endog_mean.index.equals(pd.Index(endog_names))):
                raise ValueError(f'Invalid value passed for `standardize`: if a Pandas Series, must have index {endog_names}. Got {endog_mean.index}.')
            else:
                endog_mean = np.atleast_1d(endog_mean)
            if isinstance(endog_std, pd.Series) and (not endog_std.index.equals(pd.Index(endog_names))):
                raise ValueError(f'Invalid value passed for `standardize`: if a Pandas Series, must have index {endog_names}. Got {endog_std.index}.')
            else:
                endog_std = np.atleast_1d(endog_std)
            if np.shape(endog_mean) != (n,) or np.shape(endog_std) != (n,):
                raise ValueError(f'Invalid value passed for `standardize`: each element must be shaped ({n},).')
            standardize = True
            if endog_is_pandas:
                endog_mean = pd.Series(endog_mean, index=endog_names)
                endog_std = pd.Series(endog_std, index=endog_names)
        elif standardize in [1, True]:
            endog_mean = endog.mean(axis=0)
            endog_std = endog.std(axis=0)
        elif standardize in [0, False]:
            endog_mean = np.zeros(endog.shape[1])
            endog_std = np.ones(endog.shape[1])
        else:
            raise ValueError('Invalid value passed for `standardize`.')
        self._endog_mean = endog_mean
        self._endog_std = endog_std
        self.standardize = standardize
        if np.any(self._endog_std < 1e-10):
            ix = np.where(self._endog_std < 1e-10)
            names = np.array(endog_names)[ix[0]].tolist()
            raise ValueError(f'Constant variable(s) found in observed variables, but constants cannot be included in this model. These variables are: {names}.')
        if self.standardize:
            endog = (endog - self._endog_mean) / self._endog_std
        o = self._o = {'M': np.s_[:self.k_endog_M], 'Q': np.s_[self.k_endog_M:]}
        super().__init__(endog, k_states=s.k_states, k_posdef=s.k_posdef, **kwargs)
        if self.standardize:
            self.data.orig_endog = self.data.orig_endog * self._endog_std + self._endog_mean
        if 'initialization' not in kwargs:
            self.ssm.initialize(self._default_initialization())
        if self.idiosyncratic_ar1:
            self['design', o['M'], s['idio_ar_M']] = np.eye(self.k_endog_M)
        multipliers = [1, 2, 3, 2, 1]
        for i in range(len(multipliers)):
            m = multipliers[i]
            self['design', o['Q'], s['idio_ar_Q_ix'][:, i]] = m * np.eye(self.k_endog_Q)
        if self.obs_cov_diag:
            self['obs_cov'] = np.eye(self.k_endog) * 0.0001
        for block in s.factor_blocks:
            if block.k_factors == 1:
                tmp = 0
            else:
                tmp = np.zeros((block.k_factors, block.k_factors))
            self['transition', block['factors'], block['factors']] = companion_matrix([1] + [tmp] * block._factor_order).T
        if self.k_endog_Q == 1:
            tmp = 0
        else:
            tmp = np.zeros((self.k_endog_Q, self.k_endog_Q))
        self['transition', s['idio_ar_Q'], s['idio_ar_Q']] = companion_matrix([1] + [tmp] * 5).T
        ix1 = ix2 = 0
        for block in s.factor_blocks:
            ix2 += block.k_factors
            self['selection', block['factors_ix'][:, 0], ix1:ix2] = np.eye(block.k_factors)
            ix1 = ix2
        if self.idiosyncratic_ar1:
            ix2 = ix1 + self.k_endog_M
            self['selection', s['idio_ar_M'], ix1:ix2] = np.eye(self.k_endog_M)
            ix1 = ix2
        ix2 = ix1 + self.k_endog_Q
        self['selection', s['idio_ar_Q_ix'][:, 0], ix1:ix2] = np.eye(self.k_endog_Q)
        self.params = OrderedDict([('loadings', np.sum(self.endog_factor_map.values)), ('factor_ar', np.sum([block.k_factors ** 2 * block.factor_order for block in s.factor_blocks])), ('factor_cov', np.sum([block.k_factors * (block.k_factors + 1) // 2 for block in s.factor_blocks])), ('idiosyncratic_ar1', self.k_endog if self.idiosyncratic_ar1 else 0), ('idiosyncratic_var', self.k_endog)])
        self.k_params = np.sum(list(self.params.values()))
        ix = np.split(np.arange(self.k_params), np.cumsum(list(self.params.values()))[:-1])
        self._p = dict(zip(self.params.keys(), ix))
        self._loading_constraints = {}
        self._init_keys += ['factors', 'factor_orders', 'factor_multiplicities', 'idiosyncratic_ar1', 'standardize', 'init_t0', 'obs_cov_diag'] + list(kwargs.keys())

    @classmethod
    def construct_endog(cls, endog_monthly, endog_quarterly):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a combined dataset from separate monthly and quarterly data.\n\n        Parameters\n        ----------\n        endog_monthly : array_like\n            Monthly dataset. If a quarterly dataset is given, then this must\n            be a Pandas object with a PeriodIndex or DatetimeIndex at a monthly\n            frequency.\n        endog_quarterly : array_like or None\n            Quarterly dataset. If not None, then this must be a Pandas object\n            with a PeriodIndex or DatetimeIndex at a quarterly frequency.\n\n        Returns\n        -------\n        endog : array_like\n            If both endog_monthly and endog_quarterly were given, this is a\n            Pandas DataFrame with a PeriodIndex at the monthly frequency, with\n            all of the columns from `endog_monthly` ordered first and the\n            columns from `endog_quarterly` ordered afterwards. Otherwise it is\n            simply the input `endog_monthly` dataset.\n        k_endog_monthly : int\n            The number of monthly variables (which are ordered first) in the\n            returned `endog` dataset.\n        '
        if endog_quarterly is not None:
            base_msg = 'If given both monthly and quarterly data then the monthly dataset must be a Pandas object with a date index at a monthly frequency.'
            if not isinstance(endog_monthly, (pd.Series, pd.DataFrame)):
                raise ValueError('Given monthly dataset is not a Pandas object. ' + base_msg)
            elif endog_monthly.index.inferred_type not in ('datetime64', 'period'):
                raise ValueError('Given monthly dataset has an index with non-date values. ' + base_msg)
            elif not getattr(endog_monthly.index, 'freqstr', 'N')[0] == 'M':
                freqstr = getattr(endog_monthly.index, 'freqstr', 'None')
                raise ValueError(f'Index of given monthly dataset has a non-monthly frequency (to check this, examine the `freqstr` attribute of the index of the dataset - it should start with M if it is monthly). Got {freqstr}. ' + base_msg)
            base_msg = 'If a quarterly dataset is given, then it must be a Pandas object with a date index at a quarterly frequency.'
            if not isinstance(endog_quarterly, (pd.Series, pd.DataFrame)):
                raise ValueError('Given quarterly dataset is not a Pandas object. ' + base_msg)
            elif endog_quarterly.index.inferred_type not in ('datetime64', 'period'):
                raise ValueError('Given quarterly dataset has an index with non-date values. ' + base_msg)
            elif not getattr(endog_quarterly.index, 'freqstr', 'N')[0] == 'Q':
                freqstr = getattr(endog_quarterly.index, 'freqstr', 'None')
                raise ValueError(f'Index of given quarterly dataset has a non-quarterly frequency (to check this, examine the `freqstr` attribute of the index of the dataset - it should start with Q if it is quarterly). Got {freqstr}. ' + base_msg)
            if hasattr(endog_monthly.index, 'to_period'):
                endog_monthly = endog_monthly.to_period('M')
            if hasattr(endog_quarterly.index, 'to_period'):
                endog_quarterly = endog_quarterly.to_period('Q')
            endog = pd.concat([endog_monthly, endog_quarterly.resample('M', convention='end').first()], axis=1)
            column_counts = endog.columns.value_counts()
            if column_counts.max() > 1:
                columns = endog.columns.values.astype(object)
                for name in column_counts.index:
                    count = column_counts.loc[name]
                    if count == 1:
                        continue
                    mask = columns == name
                    columns[mask] = [f'{name}{i + 1}' for i in range(count)]
                endog.columns = columns
        else:
            endog = endog_monthly.copy()
        shape = endog_monthly.shape
        k_endog_monthly = shape[1] if len(shape) == 2 else 1
        return (endog, k_endog_monthly)

    def clone(self, endog, k_endog_monthly=None, endog_quarterly=None, retain_standardization=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Clone state space model with new data and optionally new specification.\n\n        Parameters\n        ----------\n        endog : array_like\n            The observed time-series process :math:`y`\n        k_endog_monthly : int, optional\n            If specifying a monthly/quarterly mixed frequency model in which\n            the provided `endog` dataset contains both the monthly and\n            quarterly data, this variable should be used to indicate how many\n            of the variables are monthly.\n        endog_quarterly : array_like, optional\n            Observations of quarterly variables. If provided, must be a\n            Pandas Series or DataFrame with a DatetimeIndex or PeriodIndex at\n            the quarterly frequency.\n        kwargs\n            Keyword arguments to pass to the new model class to change the\n            model specification.\n\n        Returns\n        -------\n        model : DynamicFactorMQ instance\n        '
        if retain_standardization and self.standardize:
            kwargs['standardize'] = (self._endog_mean, self._endog_std)
        mod = self._clone_from_init_kwds(endog, k_endog_monthly=k_endog_monthly, endog_quarterly=endog_quarterly, **kwargs)
        return mod

    @property
    def _res_classes(self):
        if False:
            while True:
                i = 10
        return {'fit': (DynamicFactorMQResults, mlemodel.MLEResultsWrapper)}

    def _default_initialization(self):
        if False:
            for i in range(10):
                print('nop')
        s = self._s
        init = initialization.Initialization(self.k_states)
        for block in s.factor_blocks:
            init.set(block['factors'], 'stationary')
        if self.idiosyncratic_ar1:
            for i in range(s['idio_ar_M'].start, s['idio_ar_M'].stop):
                init.set(i, 'stationary')
        init.set(s['idio_ar_Q'], 'stationary')
        return init

    def _get_endog_names(self, truncate=None, as_string=None):
        if False:
            while True:
                i = 10
        if truncate is None:
            truncate = False if as_string is False or self.k_endog == 1 else 24
        if as_string is False and truncate is not False:
            raise ValueError('Can only truncate endog names if they are returned as a string.')
        if as_string is None:
            as_string = truncate is not False
        endog_names = self.endog_names
        if not isinstance(endog_names, list):
            endog_names = [endog_names]
        if as_string:
            endog_names = [str(name) for name in endog_names]
        if truncate is not False:
            n = truncate
            endog_names = [name if len(name) <= n else name[:n] + '...' for name in endog_names]
        return endog_names

    @property
    def _model_name(self):
        if False:
            i = 10
            return i + 15
        model_name = ['Dynamic Factor Model', f'{self.k_factors} factors in {self.k_factor_blocks} blocks']
        if self.k_endog_Q > 0:
            model_name.append('Mixed frequency (M/Q)')
        error_type = 'AR(1)' if self.idiosyncratic_ar1 else 'iid'
        model_name.append(f'{error_type} idiosyncratic')
        return model_name

    def summary(self, truncate_endog_names=None):
        if False:
            return 10
        '\n        Create a summary table describing the model.\n\n        Parameters\n        ----------\n        truncate_endog_names : int, optional\n            The number of characters to show for names of observed variables.\n            Default is 24 if there is more than one observed variable, or\n            an unlimited number of there is only one.\n        '
        endog_names = self._get_endog_names(truncate=truncate_endog_names, as_string=True)
        title = 'Model Specification: Dynamic Factor Model'
        if self._index_dates:
            ix = self._index
            d = ix[0]
            sample = ['%s' % d]
            d = ix[-1]
            sample += ['- ' + '%s' % d]
        else:
            sample = [str(0), ' - ' + str(self.nobs)]
        model_name = self._model_name
        top_left = []
        top_left.append(('Model:', [model_name[0]]))
        for i in range(1, len(model_name)):
            top_left.append(('', ['+ ' + model_name[i]]))
        top_left += [('Sample:', [sample[0]]), ('', [sample[1]])]
        top_right = []
        if self.k_endog_Q > 0:
            top_right += [('# of monthly variables:', [self.k_endog_M]), ('# of quarterly variables:', [self.k_endog_Q])]
        else:
            top_right += [('# of observed variables:', [self.k_endog])]
        if self.k_factor_blocks == 1:
            top_right += [('# of factors:', [self.k_factors])]
        else:
            top_right += [('# of factor blocks:', [self.k_factor_blocks])]
        top_right += [('Idiosyncratic disturbances:', ['AR(1)' if self.idiosyncratic_ar1 else 'iid']), ('Standardize variables:', [self.standardize])]
        summary = Summary()
        self.model = self
        summary.add_table_2cols(self, gleft=top_left, gright=top_right, title=title)
        table_ix = 1
        del self.model
        data = self.endog_factor_map.replace({True: 'X', False: ''})
        data.index = endog_names
        try:
            items = data.items()
        except AttributeError:
            items = data.iteritems()
        for (name, col) in items:
            data[name] = data[name] + ' ' * (len(name) // 2)
        data.index.name = 'Dep. variable'
        data = data.reset_index()
        params_data = data.values
        params_header = data.columns.map(str).tolist()
        params_stubs = None
        title = 'Observed variables / factor loadings'
        table = SimpleTable(params_data, params_header, params_stubs, txt_fmt=fmt_params, title=title)
        summary.tables.insert(table_ix, table)
        table_ix += 1
        data = self.factor_block_orders.reset_index()
        data['block'] = data['block'].map(lambda factor_names: ', '.join(factor_names))
        try:
            data[['order']] = data[['order']].map(str)
        except AttributeError:
            data[['order']] = data[['order']].applymap(str)
        params_data = data.values
        params_header = data.columns.map(str).tolist()
        params_stubs = None
        title = 'Factor blocks:'
        table = SimpleTable(params_data, params_header, params_stubs, txt_fmt=fmt_params, title=title)
        summary.tables.insert(table_ix, table)
        table_ix += 1
        return summary

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        'Summary tables showing model specification.'
        return str(self.summary())

    @property
    def state_names(self):
        if False:
            print('Hello World!')
        '(list of str) List of human readable names for unobserved states.'
        state_names = []
        for block in self._s.factor_blocks:
            state_names += [f'{name}' for name in block.factor_names[:]]
            for s in range(1, block._factor_order):
                state_names += [f'L{s}.{name}' for name in block.factor_names]
        endog_names = self._get_endog_names()
        if self.idiosyncratic_ar1:
            endog_names_M = endog_names[self._o['M']]
            state_names += [f'eps_M.{name}' for name in endog_names_M]
        endog_names_Q = endog_names[self._o['Q']]
        state_names += [f'eps_Q.{name}' for name in endog_names_Q]
        for s in range(1, 5):
            state_names += [f'L{s}.eps_Q.{name}' for name in endog_names_Q]
        return state_names

    @property
    def param_names(self):
        if False:
            i = 10
            return i + 15
        '(list of str) List of human readable parameter names.'
        param_names = []
        endog_names = self._get_endog_names(as_string=False)
        for endog_name in endog_names:
            for block in self._s.factor_blocks:
                for factor_name in block.factor_names:
                    if self.endog_factor_map.loc[endog_name, factor_name]:
                        param_names.append(f'loading.{factor_name}->{endog_name}')
        for block in self._s.factor_blocks:
            for to_factor in block.factor_names:
                param_names += [f'L{i}.{from_factor}->{to_factor}' for i in range(1, block.factor_order + 1) for from_factor in block.factor_names]
        for i in range(len(self._s.factor_blocks)):
            block = self._s.factor_blocks[i]
            param_names += [f'fb({i}).cov.chol[{j + 1},{k + 1}]' for j in range(block.k_factors) for k in range(j + 1)]
        if self.idiosyncratic_ar1:
            endog_names_M = endog_names[self._o['M']]
            param_names += [f'L1.eps_M.{name}' for name in endog_names_M]
            endog_names_Q = endog_names[self._o['Q']]
            param_names += [f'L1.eps_Q.{name}' for name in endog_names_Q]
        param_names += [f'sigma2.{name}' for name in endog_names]
        return param_names

    @property
    def start_params(self):
        if False:
            i = 10
            return i + 15
        '(array) Starting parameters for maximum likelihood estimation.'
        params = np.zeros(self.k_params, dtype=np.float64)
        endog_factor_map_M = self.endog_factor_map.iloc[:self.k_endog_M]
        factors = []
        endog = np.require(pd.DataFrame(self.endog).interpolate().bfill(), requirements='W')
        for name in self.factor_names:
            endog_ix = np.where(endog_factor_map_M.loc[:, name])[0]
            if len(endog_ix) == 0:
                endog_ix = np.where(self.endog_factor_map.loc[:, name])[0]
            factor_endog = endog[:, endog_ix]
            res_pca = PCA(factor_endog, ncomp=1, method='eig', normalize=False)
            factors.append(res_pca.factors)
            endog[:, endog_ix] -= res_pca.projection
        factors = np.concatenate(factors, axis=1)
        loadings = []
        resid = []
        for i in range(self.k_endog_M):
            factor_ix = self._s.endog_factor_iloc[i]
            factor_exog = factors[:, factor_ix]
            mod_ols = OLS(self.endog[:, i], exog=factor_exog, missing='drop')
            res_ols = mod_ols.fit()
            loadings += res_ols.params.tolist()
            resid.append(res_ols.resid)
        for i in range(self.k_endog_M, self.k_endog):
            factor_ix = self._s.endog_factor_iloc[i]
            factor_exog = lagmat(factors[:, factor_ix], 4, original='in')
            mod_glm = GLM(self.endog[:, i], factor_exog, missing='drop')
            res_glm = mod_glm.fit_constrained(self.loading_constraints(i))
            loadings += res_glm.params[:len(factor_ix)].tolist()
            resid.append(res_glm.resid_response)
        params[self._p['loadings']] = loadings
        stationary = True
        factor_ar = []
        factor_cov = []
        i = 0
        for block in self._s.factor_blocks:
            factors_endog = factors[:, i:i + block.k_factors]
            i += block.k_factors
            if block.factor_order == 0:
                continue
            if block.k_factors == 1:
                mod_factors = SARIMAX(factors_endog, order=(block.factor_order, 0, 0))
                sp = mod_factors.start_params
                block_factor_ar = sp[:-1]
                block_factor_cov = sp[-1:]
                coefficient_matrices = mod_factors.start_params[:-1]
            elif block.k_factors > 1:
                mod_factors = VAR(factors_endog)
                res_factors = mod_factors.fit(maxlags=block.factor_order, ic=None, trend='n')
                block_factor_ar = res_factors.params.T.ravel()
                L = np.linalg.cholesky(res_factors.sigma_u)
                block_factor_cov = L[np.tril_indices_from(L)]
                coefficient_matrices = np.transpose(np.reshape(block_factor_ar, (block.k_factors, block.k_factors, block.factor_order)), (2, 0, 1))
            stationary = is_invertible([1] + list(-coefficient_matrices))
            if not stationary:
                warn(f'Non-stationary starting factor autoregressive parameters found for factor block {block.factor_names}. Using zeros as starting parameters.')
                block_factor_ar[:] = 0
                cov_factor = np.diag(factors_endog.std(axis=0))
                block_factor_cov = cov_factor[np.tril_indices(block.k_factors)]
            factor_ar += block_factor_ar.tolist()
            factor_cov += block_factor_cov.tolist()
        params[self._p['factor_ar']] = factor_ar
        params[self._p['factor_cov']] = factor_cov
        if self.idiosyncratic_ar1:
            idio_ar1 = []
            idio_var = []
            for i in range(self.k_endog_M):
                mod_idio = SARIMAX(resid[i], order=(1, 0, 0), trend='c')
                sp = mod_idio.start_params
                idio_ar1.append(np.clip(sp[1], -0.99, 0.99))
                idio_var.append(np.clip(sp[-1], 1e-05, np.inf))
            for i in range(self.k_endog_M, self.k_endog):
                y = self.endog[:, i].copy()
                y[~np.isnan(y)] = resid[i]
                mod_idio = QuarterlyAR1(y)
                res_idio = mod_idio.fit(maxiter=10, return_params=True, disp=False)
                res_idio = mod_idio.fit_em(res_idio, maxiter=5, return_params=True)
                idio_ar1.append(np.clip(res_idio[0], -0.99, 0.99))
                idio_var.append(np.clip(res_idio[1], 1e-05, np.inf))
            params[self._p['idiosyncratic_ar1']] = idio_ar1
            params[self._p['idiosyncratic_var']] = idio_var
        else:
            idio_var = [np.var(resid[i]) for i in range(self.k_endog_M)]
            for i in range(self.k_endog_M, self.k_endog):
                y = self.endog[:, i].copy()
                y[~np.isnan(y)] = resid[i]
                mod_idio = QuarterlyAR1(y)
                res_idio = mod_idio.fit(return_params=True, disp=False)
                idio_var.append(np.clip(res_idio[1], 1e-05, np.inf))
            params[self._p['idiosyncratic_var']] = idio_var
        return params

    def transform_params(self, unconstrained):
        if False:
            return 10
        '\n        Transform parameters from optimizer space to model space.\n\n        Transform unconstrained parameters used by the optimizer to constrained\n        parameters used in likelihood evaluation.\n\n        Parameters\n        ----------\n        unconstrained : array_like\n            Array of unconstrained parameters used by the optimizer, to be\n            transformed.\n\n        Returns\n        -------\n        constrained : array_like\n            Array of constrained parameters which may be used in likelihood\n            evaluation.\n        '
        constrained = unconstrained.copy()
        unconstrained_factor_ar = unconstrained[self._p['factor_ar']]
        constrained_factor_ar = []
        i = 0
        for block in self._s.factor_blocks:
            length = block.k_factors ** 2 * block.factor_order
            tmp_coeff = np.reshape(unconstrained_factor_ar[i:i + length], (block.k_factors, block.k_factors * block.factor_order))
            tmp_cov = np.eye(block.k_factors)
            (tmp_coeff, _) = constrain_stationary_multivariate(tmp_coeff, tmp_cov)
            constrained_factor_ar += tmp_coeff.ravel().tolist()
            i += length
        constrained[self._p['factor_ar']] = constrained_factor_ar
        if self.idiosyncratic_ar1:
            idio_ar1 = unconstrained[self._p['idiosyncratic_ar1']]
            constrained[self._p['idiosyncratic_ar1']] = [constrain_stationary_univariate(idio_ar1[i:i + 1])[0] for i in range(self.k_endog)]
        constrained[self._p['idiosyncratic_var']] = constrained[self._p['idiosyncratic_var']] ** 2
        return constrained

    def untransform_params(self, constrained):
        if False:
            print('Hello World!')
        '\n        Transform parameters from model space to optimizer space.\n\n        Transform constrained parameters used in likelihood evaluation\n        to unconstrained parameters used by the optimizer.\n\n        Parameters\n        ----------\n        constrained : array_like\n            Array of constrained parameters used in likelihood evaluation, to\n            be transformed.\n\n        Returns\n        -------\n        unconstrained : array_like\n            Array of unconstrained parameters used by the optimizer.\n        '
        unconstrained = constrained.copy()
        constrained_factor_ar = constrained[self._p['factor_ar']]
        unconstrained_factor_ar = []
        i = 0
        for block in self._s.factor_blocks:
            length = block.k_factors ** 2 * block.factor_order
            tmp_coeff = np.reshape(constrained_factor_ar[i:i + length], (block.k_factors, block.k_factors * block.factor_order))
            tmp_cov = np.eye(block.k_factors)
            (tmp_coeff, _) = unconstrain_stationary_multivariate(tmp_coeff, tmp_cov)
            unconstrained_factor_ar += tmp_coeff.ravel().tolist()
            i += length
        unconstrained[self._p['factor_ar']] = unconstrained_factor_ar
        if self.idiosyncratic_ar1:
            idio_ar1 = constrained[self._p['idiosyncratic_ar1']]
            unconstrained[self._p['idiosyncratic_ar1']] = [unconstrain_stationary_univariate(idio_ar1[i:i + 1])[0] for i in range(self.k_endog)]
        unconstrained[self._p['idiosyncratic_var']] = unconstrained[self._p['idiosyncratic_var']] ** 0.5
        return unconstrained

    def update(self, params, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Update the parameters of the model.\n\n        Parameters\n        ----------\n        params : array_like\n            Array of new parameters.\n        transformed : bool, optional\n            Whether or not `params` is already transformed. If set to False,\n            `transform_params` is called. Default is True.\n\n        '
        params = super().update(params, **kwargs)
        o = self._o
        s = self._s
        p = self._p
        loadings = params[p['loadings']]
        start = 0
        for i in range(self.k_endog_M):
            iloc = self._s.endog_factor_iloc[i]
            k_factors = len(iloc)
            factor_ix = s['factors_L1'][iloc]
            self['design', i, factor_ix] = loadings[start:start + k_factors]
            start += k_factors
        multipliers = np.array([1, 2, 3, 2, 1])[:, None]
        for i in range(self.k_endog_M, self.k_endog):
            iloc = self._s.endog_factor_iloc[i]
            k_factors = len(iloc)
            factor_ix = s['factors_L1_5_ix'][:, iloc]
            self['design', i, factor_ix.ravel()] = np.ravel(loadings[start:start + k_factors] * multipliers)
            start += k_factors
        factor_ar = params[p['factor_ar']]
        start = 0
        for block in s.factor_blocks:
            k_params = block.k_factors ** 2 * block.factor_order
            A = np.reshape(factor_ar[start:start + k_params], (block.k_factors, block.k_factors * block.factor_order))
            start += k_params
            self['transition', block['factors_L1'], block['factors_ar']] = A
        factor_cov = params[p['factor_cov']]
        start = 0
        ix1 = 0
        for block in s.factor_blocks:
            k_params = block.k_factors * (block.k_factors + 1) // 2
            L = np.zeros((block.k_factors, block.k_factors), dtype=params.dtype)
            L[np.tril_indices_from(L)] = factor_cov[start:start + k_params]
            start += k_params
            Q = L @ L.T
            ix2 = ix1 + block.k_factors
            self['state_cov', ix1:ix2, ix1:ix2] = Q
            ix1 = ix2
        if self.idiosyncratic_ar1:
            alpha = np.diag(params[p['idiosyncratic_ar1']])
            self['transition', s['idio_ar_L1'], s['idio_ar_L1']] = alpha
        if self.idiosyncratic_ar1:
            self['state_cov', self.k_factors:, self.k_factors:] = np.diag(params[p['idiosyncratic_var']])
        else:
            idio_var = params[p['idiosyncratic_var']]
            self['obs_cov', o['M'], o['M']] = np.diag(idio_var[o['M']])
            self['state_cov', self.k_factors:, self.k_factors:] = np.diag(idio_var[o['Q']])

    @property
    def loglike_constant(self):
        if False:
            return 10
        '\n        Constant term in the joint log-likelihood function.\n\n        Useful in facilitating comparisons to other packages that exclude the\n        constant from the log-likelihood computation.\n        '
        return -0.5 * (1 - np.isnan(self.endog)).sum() * np.log(2 * np.pi)

    def loading_constraints(self, i):
        if False:
            i = 10
            return i + 15
        "\n        Matrix formulation of quarterly variables' factor loading constraints.\n\n        Parameters\n        ----------\n        i : int\n            Index of the `endog` variable to compute constraints for.\n\n        Returns\n        -------\n        R : array (k_constraints, k_factors * 5)\n        q : array (k_constraints,)\n\n        Notes\n        -----\n        If the factors were known, then the factor loadings for the ith\n        quarterly variable would be computed by a linear regression of the form\n\n        y_i = A_i' f + B_i' L1.f + C_i' L2.f + D_i' L3.f + E_i' L4.f\n\n        where:\n\n        - f is (k_i x 1) and collects all of the factors that load on y_i\n        - L{j}.f is (k_i x 1) and collects the jth lag of each factor\n        - A_i, ..., E_i are (k_i x 1) and collect factor loadings\n\n        As the observed variable is quarterly while the factors are monthly, we\n        want to restrict the estimated regression coefficients to be:\n\n        y_i = A_i f + 2 A_i L1.f + 3 A_i L2.f + 2 A_i L3.f + A_i L4.f\n\n        Stack the unconstrained coefficients: \\Lambda_i = [A_i' B_i' ... E_i']'\n\n        Then the constraints can be written as follows, for l = 1, ..., k_i\n\n        - 2 A_{i,l} - B_{i,l} = 0\n        - 3 A_{i,l} - C_{i,l} = 0\n        - 2 A_{i,l} - D_{i,l} = 0\n        - A_{i,l} - E_{i,l} = 0\n\n        So that k_constraints = 4 * k_i. In matrix form the constraints are:\n\n        .. math::\n\n            R \\Lambda_i = q\n\n        where :math:`\\Lambda_i` is shaped `(k_i * 5,)`, :math:`R` is shaped\n        `(k_constraints, k_i * 5)`, and :math:`q` is shaped `(k_constraints,)`.\n\n\n        For example, for the case that k_i = 2, we can write:\n\n        |  2 0   -1  0    0  0    0  0    0  0  |   | A_{i,1} |     | 0 |\n        |  0 2    0 -1    0  0    0  0    0  0  |   | A_{i,2} |     | 0 |\n        |  3 0    0  0   -1  0    0  0    0  0  |   | B_{i,1} |     | 0 |\n        |  0 3    0  0    0 -1    0  0    0  0  |   | B_{i,2} |     | 0 |\n        |  2 0    0  0    0  0   -1  0    0  0  |   | C_{i,1} |  =  | 0 |\n        |  0 2    0  0    0  0    0 -1    0  0  |   | C_{i,2} |     | 0 |\n        |  1 0    0  0    0  0    0  0   -1  0  |   | D_{i,1} |     | 0 |\n        |  0 1    0  0    0  0    0  0    0 -1  |   | D_{i,2} |     | 0 |\n                                                    | E_{i,1} |     | 0 |\n                                                    | E_{i,2} |     | 0 |\n\n        "
        if i < self.k_endog_M:
            raise ValueError('No constraints for monthly variables.')
        if i not in self._loading_constraints:
            k_factors = self.endog_factor_map.iloc[i].sum()
            R = np.zeros((k_factors * 4, k_factors * 5))
            q = np.zeros(R.shape[0])
            multipliers = np.array([1, 2, 3, 2, 1])
            R[:, :k_factors] = np.reshape((multipliers[1:] * np.eye(k_factors)[..., None]).T, (k_factors * 4, k_factors))
            R[:, k_factors:] = np.diag([-1] * (k_factors * 4))
            self._loading_constraints[i] = (R, q)
        return self._loading_constraints[i]

    def fit(self, start_params=None, transformed=True, includes_fixed=False, cov_type='none', cov_kwds=None, method='em', maxiter=500, tolerance=1e-06, em_initialization=True, mstep_method=None, full_output=1, disp=False, callback=None, return_params=False, optim_score=None, optim_complex_step=None, optim_hessian=None, flags=None, low_memory=False, llf_decrease_action='revert', llf_decrease_tolerance=0.0001, **kwargs):
        if False:
            print('Hello World!')
        "\n        Fits the model by maximum likelihood via Kalman filter.\n\n        Parameters\n        ----------\n        start_params : array_like, optional\n            Initial guess of the solution for the loglikelihood maximization.\n            If None, the default is given by Model.start_params.\n        transformed : bool, optional\n            Whether or not `start_params` is already transformed. Default is\n            True.\n        includes_fixed : bool, optional\n            If parameters were previously fixed with the `fix_params` method,\n            this argument describes whether or not `start_params` also includes\n            the fixed parameters, in addition to the free parameters. Default\n            is False.\n        cov_type : str, optional\n            The `cov_type` keyword governs the method for calculating the\n            covariance matrix of parameter estimates. Can be one of:\n\n            - 'opg' for the outer product of gradient estimator\n            - 'oim' for the observed information matrix estimator, calculated\n              using the method of Harvey (1989)\n            - 'approx' for the observed information matrix estimator,\n              calculated using a numerical approximation of the Hessian matrix.\n            - 'robust' for an approximate (quasi-maximum likelihood) covariance\n              matrix that may be valid even in the presence of some\n              misspecifications. Intermediate calculations use the 'oim'\n              method.\n            - 'robust_approx' is the same as 'robust' except that the\n              intermediate calculations use the 'approx' method.\n            - 'none' for no covariance matrix calculation.\n\n            Default is 'none', since computing this matrix can be very slow\n            when there are a large number of parameters.\n        cov_kwds : dict or None, optional\n            A dictionary of arguments affecting covariance matrix computation.\n\n            **opg, oim, approx, robust, robust_approx**\n\n            - 'approx_complex_step' : bool, optional - If True, numerical\n              approximations are computed using complex-step methods. If False,\n              numerical approximations are computed using finite difference\n              methods. Default is True.\n            - 'approx_centered' : bool, optional - If True, numerical\n              approximations computed using finite difference methods use a\n              centered approximation. Default is False.\n        method : str, optional\n            The `method` determines which solver from `scipy.optimize`\n            is used, and it can be chosen from among the following strings:\n\n            - 'em' for the EM algorithm\n            - 'newton' for Newton-Raphson\n            - 'nm' for Nelder-Mead\n            - 'bfgs' for Broyden-Fletcher-Goldfarb-Shanno (BFGS)\n            - 'lbfgs' for limited-memory BFGS with optional box constraints\n            - 'powell' for modified Powell's method\n            - 'cg' for conjugate gradient\n            - 'ncg' for Newton-conjugate gradient\n            - 'basinhopping' for global basin-hopping solver\n\n            The explicit arguments in `fit` are passed to the solver,\n            with the exception of the basin-hopping solver. Each\n            solver has several optional arguments that are not the same across\n            solvers. See the notes section below (or scipy.optimize) for the\n            available arguments and for the list of explicit arguments that the\n            basin-hopping solver supports.\n        maxiter : int, optional\n            The maximum number of iterations to perform.\n        tolerance : float, optional\n            Tolerance to use for convergence checking when using the EM\n            algorithm. To set the tolerance for other methods, pass\n            the optimizer-specific keyword argument(s).\n        full_output : bool, optional\n            Set to True to have all available output in the Results object's\n            mle_retvals attribute. The output is dependent on the solver.\n            See LikelihoodModelResults notes section for more information.\n        disp : bool, optional\n            Set to True to print convergence messages.\n        callback : callable callback(xk), optional\n            Called after each iteration, as callback(xk), where xk is the\n            current parameter vector.\n        return_params : bool, optional\n            Whether or not to return only the array of maximizing parameters.\n            Default is False.\n        optim_score : {'harvey', 'approx'} or None, optional\n            The method by which the score vector is calculated. 'harvey' uses\n            the method from Harvey (1989), 'approx' uses either finite\n            difference or complex step differentiation depending upon the\n            value of `optim_complex_step`, and None uses the built-in gradient\n            approximation of the optimizer. Default is None. This keyword is\n            only relevant if the optimization method uses the score.\n        optim_complex_step : bool, optional\n            Whether or not to use complex step differentiation when\n            approximating the score; if False, finite difference approximation\n            is used. Default is True. This keyword is only relevant if\n            `optim_score` is set to 'harvey' or 'approx'.\n        optim_hessian : {'opg','oim','approx'}, optional\n            The method by which the Hessian is numerically approximated. 'opg'\n            uses outer product of gradients, 'oim' uses the information\n            matrix formula from Harvey (1989), and 'approx' uses numerical\n            approximation. This keyword is only relevant if the\n            optimization method uses the Hessian matrix.\n        low_memory : bool, optional\n            If set to True, techniques are applied to substantially reduce\n            memory usage. If used, some features of the results object will\n            not be available (including smoothed results and in-sample\n            prediction), although out-of-sample forecasting is possible.\n            Note that this option is not available when using the EM algorithm\n            (which is the default for this model). Default is False.\n        llf_decrease_action : {'ignore', 'warn', 'revert'}, optional\n            Action to take if the log-likelihood decreases in an EM iteration.\n            'ignore' continues the iterations, 'warn' issues a warning but\n            continues the iterations, while 'revert' ends the iterations and\n            returns the result from the last good iteration. Default is 'warn'.\n        llf_decrease_tolerance : float, optional\n            Minimum size of the log-likelihood decrease required to trigger a\n            warning or to end the EM iterations. Setting this value slightly\n            larger than zero allows small decreases in the log-likelihood that\n            may be caused by numerical issues. If set to zero, then any\n            decrease will trigger the `llf_decrease_action`. Default is 1e-4.\n        **kwargs\n            Additional keyword arguments to pass to the optimizer.\n\n        Returns\n        -------\n        MLEResults\n\n        See Also\n        --------\n        statsmodels.base.model.LikelihoodModel.fit\n        statsmodels.tsa.statespace.mlemodel.MLEResults\n        "
        if method == 'em':
            return self.fit_em(start_params=start_params, transformed=transformed, cov_type=cov_type, cov_kwds=cov_kwds, maxiter=maxiter, tolerance=tolerance, em_initialization=em_initialization, mstep_method=mstep_method, full_output=full_output, disp=disp, return_params=return_params, low_memory=low_memory, llf_decrease_action=llf_decrease_action, llf_decrease_tolerance=llf_decrease_tolerance, **kwargs)
        else:
            return super().fit(start_params=start_params, transformed=transformed, includes_fixed=includes_fixed, cov_type=cov_type, cov_kwds=cov_kwds, method=method, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, return_params=return_params, optim_score=optim_score, optim_complex_step=optim_complex_step, optim_hessian=optim_hessian, flags=flags, low_memory=low_memory, **kwargs)

    def fit_em(self, start_params=None, transformed=True, cov_type='none', cov_kwds=None, maxiter=500, tolerance=1e-06, disp=False, em_initialization=True, mstep_method=None, full_output=True, return_params=False, low_memory=False, llf_decrease_action='revert', llf_decrease_tolerance=0.0001):
        if False:
            return 10
        '\n        Fits the model by maximum likelihood via the EM algorithm.\n\n        Parameters\n        ----------\n        start_params : array_like, optional\n            Initial guess of the solution for the loglikelihood maximization.\n            The default is to use `DynamicFactorMQ.start_params`.\n        transformed : bool, optional\n            Whether or not `start_params` is already transformed. Default is\n            True.\n        cov_type : str, optional\n            The `cov_type` keyword governs the method for calculating the\n            covariance matrix of parameter estimates. Can be one of:\n\n            - \'opg\' for the outer product of gradient estimator\n            - \'oim\' for the observed information matrix estimator, calculated\n              using the method of Harvey (1989)\n            - \'approx\' for the observed information matrix estimator,\n              calculated using a numerical approximation of the Hessian matrix.\n            - \'robust\' for an approximate (quasi-maximum likelihood) covariance\n              matrix that may be valid even in the presence of some\n              misspecifications. Intermediate calculations use the \'oim\'\n              method.\n            - \'robust_approx\' is the same as \'robust\' except that the\n              intermediate calculations use the \'approx\' method.\n            - \'none\' for no covariance matrix calculation.\n\n            Default is \'none\', since computing this matrix can be very slow\n            when there are a large number of parameters.\n        cov_kwds : dict or None, optional\n            A dictionary of arguments affecting covariance matrix computation.\n\n            **opg, oim, approx, robust, robust_approx**\n\n            - \'approx_complex_step\' : bool, optional - If True, numerical\n              approximations are computed using complex-step methods. If False,\n              numerical approximations are computed using finite difference\n              methods. Default is True.\n            - \'approx_centered\' : bool, optional - If True, numerical\n              approximations computed using finite difference methods use a\n              centered approximation. Default is False.\n        maxiter : int, optional\n            The maximum number of EM iterations to perform.\n        tolerance : float, optional\n            Parameter governing convergence of the EM algorithm. The\n            `tolerance` is the minimum relative increase in the likelihood\n            for which convergence will be declared. A smaller value for the\n            `tolerance` will typically yield more precise parameter estimates,\n            but will typically require more EM iterations. Default is 1e-6.\n        disp : int or bool, optional\n            Controls printing of EM iteration progress. If an integer, progress\n            is printed at every `disp` iterations. A value of True is\n            interpreted as the value of 1. Default is False (nothing will be\n            printed).\n        em_initialization : bool, optional\n            Whether or not to also update the Kalman filter initialization\n            using the EM algorithm. Default is True.\n        mstep_method : {None, \'missing\', \'nonmissing\'}, optional\n            The EM algorithm maximization step. If there are no NaN values\n            in the dataset, this can be set to "nonmissing" (which is slightly\n            faster) or "missing", otherwise it must be "missing". Default is\n            "nonmissing" if there are no NaN values or "missing" if there are.\n        full_output : bool, optional\n            Set to True to have all available output from EM iterations in\n            the Results object\'s mle_retvals attribute.\n        return_params : bool, optional\n            Whether or not to return only the array of maximizing parameters.\n            Default is False.\n        low_memory : bool, optional\n            This option cannot be used with the EM algorithm and will raise an\n            error if set to True. Default is False.\n        llf_decrease_action : {\'ignore\', \'warn\', \'revert\'}, optional\n            Action to take if the log-likelihood decreases in an EM iteration.\n            \'ignore\' continues the iterations, \'warn\' issues a warning but\n            continues the iterations, while \'revert\' ends the iterations and\n            returns the result from the last good iteration. Default is \'warn\'.\n        llf_decrease_tolerance : float, optional\n            Minimum size of the log-likelihood decrease required to trigger a\n            warning or to end the EM iterations. Setting this value slightly\n            larger than zero allows small decreases in the log-likelihood that\n            may be caused by numerical issues. If set to zero, then any\n            decrease will trigger the `llf_decrease_action`. Default is 1e-4.\n\n        Returns\n        -------\n        DynamicFactorMQResults\n\n        See Also\n        --------\n        statsmodels.tsa.statespace.mlemodel.MLEModel.fit\n        statsmodels.tsa.statespace.mlemodel.MLEResults\n        '
        if self._has_fixed_params:
            raise NotImplementedError('Cannot fit using the EM algorithm while holding some parameters fixed.')
        if low_memory:
            raise ValueError('Cannot fit using the EM algorithm when using low_memory option.')
        if start_params is None:
            start_params = self.start_params
            transformed = True
        else:
            start_params = np.array(start_params, ndmin=1)
        if not transformed:
            start_params = self.transform_params(start_params)
        llf_decrease_action = string_like(llf_decrease_action, 'llf_decrease_action', options=['ignore', 'warn', 'revert'])
        disp = int(disp)
        s = self._s
        llf = []
        params = [start_params]
        init = None
        inits = [self.ssm.initialization]
        i = 0
        delta = 0
        terminate = False
        while i < maxiter and (not terminate) and (i < 1 or delta > tolerance):
            out = self._em_iteration(params[-1], init=init, mstep_method=mstep_method)
            new_llf = out[0].llf_obs.sum()
            if not em_initialization:
                self.update(out[1])
                switch_init = []
                T = self['transition']
                init = self.ssm.initialization
                iloc = np.arange(self.k_states)
                if self.k_endog_Q == 0 and (not self.idiosyncratic_ar1):
                    block = s.factor_blocks[0]
                    if init.initialization_type == 'stationary':
                        Tb = T[block['factors'], block['factors']]
                        if not np.all(np.linalg.eigvals(Tb) < 1 - 1e-10):
                            init.set(block['factors'], 'diffuse')
                            switch_init.append(f'factor block: {tuple(block.factor_names)}')
                else:
                    for block in s.factor_blocks:
                        b = tuple(iloc[block['factors']])
                        init_type = init.blocks[b].initialization_type
                        if init_type == 'stationary':
                            Tb = T[block['factors'], block['factors']]
                            if not np.all(np.linalg.eigvals(Tb) < 1 - 1e-10):
                                init.set(block['factors'], 'diffuse')
                                switch_init.append(f'factor block: {tuple(block.factor_names)}')
                if self.idiosyncratic_ar1:
                    endog_names = self._get_endog_names(as_string=True)
                    for j in range(s['idio_ar_M'].start, s['idio_ar_M'].stop):
                        init_type = init.blocks[j,].initialization_type
                        if init_type == 'stationary':
                            if not np.abs(T[j, j]) < 1 - 1e-10:
                                init.set(j, 'diffuse')
                                name = endog_names[j - s['idio_ar_M'].start]
                                switch_init.append(f'idiosyncratic AR(1) for monthly variable: {name}')
                    if self.k_endog_Q > 0:
                        b = tuple(iloc[s['idio_ar_Q']])
                        init_type = init.blocks[b].initialization_type
                        if init_type == 'stationary':
                            Tb = T[s['idio_ar_Q'], s['idio_ar_Q']]
                            if not np.all(np.linalg.eigvals(Tb) < 1 - 1e-10):
                                init.set(s['idio_ar_Q'], 'diffuse')
                                switch_init.append('idiosyncratic AR(1) for the block of quarterly variables')
                if len(switch_init) > 0:
                    warn(f'Non-stationary parameters found at EM iteration {i + 1}, which is not compatible with stationary initialization. Initialization was switched to diffuse for the following:  {switch_init}, and fitting was restarted.')
                    results = self.fit_em(start_params=params[-1], transformed=transformed, cov_type=cov_type, cov_kwds=cov_kwds, maxiter=maxiter, tolerance=tolerance, em_initialization=em_initialization, mstep_method=mstep_method, full_output=full_output, disp=disp, return_params=return_params, low_memory=low_memory, llf_decrease_action=llf_decrease_action, llf_decrease_tolerance=llf_decrease_tolerance)
                    self.ssm.initialize(self._default_initialization())
                    return results
            llf_decrease = i > 0 and new_llf - llf[-1] < -llf_decrease_tolerance
            if llf_decrease_action == 'revert' and llf_decrease:
                warn(f'Log-likelihood decreased at EM iteration {i + 1}. Reverting to the results from EM iteration {i} (prior to the decrease) and returning the solution.')
                i -= 1
                terminate = True
            else:
                if llf_decrease_action == 'warn' and llf_decrease:
                    warn(f'Log-likelihood decreased at EM iteration {i + 1}, which can indicate numerical issues.')
                llf.append(new_llf)
                params.append(out[1])
                if em_initialization:
                    init = initialization.Initialization(self.k_states, 'known', constant=out[0].smoothed_state[..., 0], stationary_cov=out[0].smoothed_state_cov[..., 0])
                    inits.append(init)
                if i > 0:
                    delta = 2 * np.abs(llf[-1] - llf[-2]) / (np.abs(llf[-1]) + np.abs(llf[-2]))
                else:
                    delta = np.inf
                if disp and i == 0:
                    print(f'EM start iterations, llf={llf[-1]:.5g}')
                elif disp and (i + 1) % disp == 0:
                    print(f'EM iteration {i + 1}, llf={llf[-1]:.5g}, convergence criterion={delta:.5g}')
            i += 1
        not_converged = i == maxiter and delta > tolerance
        if not_converged:
            warn(f'EM reached maximum number of iterations ({maxiter}), without achieving convergence: llf={llf[-1]:.5g}, convergence criterion={delta:.5g} (while specified tolerance was {tolerance:.5g})')
        if disp:
            if terminate:
                print(f'EM terminated at iteration {i}, llf={llf[-1]:.5g}, convergence criterion={delta:.5g} (while specified tolerance was {tolerance:.5g})')
            elif not_converged:
                print(f'EM reached maximum number of iterations ({maxiter}), without achieving convergence: llf={llf[-1]:.5g}, convergence criterion={delta:.5g} (while specified tolerance was {tolerance:.5g})')
            else:
                print(f'EM converged at iteration {i}, llf={llf[-1]:.5g}, convergence criterion={delta:.5g} < tolerance={tolerance:.5g}')
        if return_params:
            result = params[-1]
        else:
            if em_initialization:
                base_init = self.ssm.initialization
                self.ssm.initialization = init
            result = self.smooth(params[-1], transformed=True, cov_type=cov_type, cov_kwds=cov_kwds)
            if em_initialization:
                self.ssm.initialization = base_init
            if full_output:
                llf.append(result.llf)
                em_retvals = Bunch(**{'params': np.array(params), 'llf': np.array(llf), 'iter': i, 'inits': inits})
                em_settings = Bunch(**{'method': 'em', 'tolerance': tolerance, 'maxiter': maxiter})
            else:
                em_retvals = None
                em_settings = None
            result._results.mle_retvals = em_retvals
            result._results.mle_settings = em_settings
        return result

    def _em_iteration(self, params0, init=None, mstep_method=None):
        if False:
            i = 10
            return i + 15
        'EM iteration.'
        res = self._em_expectation_step(params0, init=init)
        params1 = self._em_maximization_step(res, params0, mstep_method=mstep_method)
        return (res, params1)

    def _em_expectation_step(self, params0, init=None):
        if False:
            for i in range(10):
                print('nop')
        'EM expectation step.'
        self.update(params0)
        if init is not None:
            base_init = self.ssm.initialization
            self.ssm.initialization = init
        res = self.ssm.smooth(SMOOTHER_STATE | SMOOTHER_STATE_COV | SMOOTHER_STATE_AUTOCOV, update_filter=False)
        res.llf_obs = np.array(self.ssm._kalman_filter.loglikelihood, copy=True)
        if init is not None:
            self.ssm.initialization = base_init
        return res

    def _em_maximization_step(self, res, params0, mstep_method=None):
        if False:
            return 10
        'EM maximization step.'
        s = self._s
        a = res.smoothed_state.T[..., None]
        cov_a = res.smoothed_state_cov.transpose(2, 0, 1)
        acov_a = res.smoothed_state_autocov.transpose(2, 0, 1)
        Eaa = cov_a.copy() + np.matmul(a, a.transpose(0, 2, 1))
        Eaa1 = acov_a[:-1] + np.matmul(a[1:], a[:-1].transpose(0, 2, 1))
        has_missing = np.any(res.nmissing)
        if mstep_method is None:
            mstep_method = 'missing' if has_missing else 'nonmissing'
        mstep_method = mstep_method.lower()
        if mstep_method == 'nonmissing' and has_missing:
            raise ValueError('Cannot use EM algorithm option `mstep_method="nonmissing"` with missing data.')
        if mstep_method == 'nonmissing':
            func = self._em_maximization_obs_nonmissing
        elif mstep_method == 'missing':
            func = self._em_maximization_obs_missing
        else:
            raise ValueError('Invalid maximization step method: "%s".' % mstep_method)
        (Lambda, H) = func(res, Eaa, a, compute_H=not self.idiosyncratic_ar1)
        factor_ar = []
        factor_cov = []
        for b in s.factor_blocks:
            A = Eaa[:-1, b['factors_ar'], b['factors_ar']].sum(axis=0)
            B = Eaa1[:, b['factors_L1'], b['factors_ar']].sum(axis=0)
            C = Eaa[1:, b['factors_L1'], b['factors_L1']].sum(axis=0)
            nobs = Eaa.shape[0] - 1
            try:
                f_A = cho_solve(cho_factor(A), B.T).T
            except LinAlgError:
                f_A = np.linalg.solve(A, B.T).T
            f_Q = (C - f_A @ B.T) / nobs
            factor_ar += f_A.ravel().tolist()
            factor_cov += np.linalg.cholesky(f_Q)[np.tril_indices_from(f_Q)].tolist()
        if self.idiosyncratic_ar1:
            ix = s['idio_ar_L1']
            Ad = Eaa[:-1, ix, ix].sum(axis=0).diagonal()
            Bd = Eaa1[:, ix, ix].sum(axis=0).diagonal()
            Cd = Eaa[1:, ix, ix].sum(axis=0).diagonal()
            nobs = Eaa.shape[0] - 1
            alpha = Bd / Ad
            sigma2 = (Cd - alpha * Bd) / nobs
        else:
            ix = s['idio_ar_L1']
            C = Eaa[:, ix, ix].sum(axis=0)
            sigma2 = np.r_[H.diagonal()[self._o['M']], C.diagonal() / Eaa.shape[0]]
        params1 = np.zeros_like(params0)
        loadings = []
        for i in range(self.k_endog):
            iloc = self._s.endog_factor_iloc[i]
            factor_ix = s['factors_L1'][iloc]
            loadings += Lambda[i, factor_ix].tolist()
        params1[self._p['loadings']] = loadings
        params1[self._p['factor_ar']] = factor_ar
        params1[self._p['factor_cov']] = factor_cov
        if self.idiosyncratic_ar1:
            params1[self._p['idiosyncratic_ar1']] = alpha
        params1[self._p['idiosyncratic_var']] = sigma2
        return params1

    def _em_maximization_obs_nonmissing(self, res, Eaa, a, compute_H=False):
        if False:
            for i in range(10):
                print('nop')
        'EM maximization step, observation equation without missing data.'
        s = self._s
        dtype = Eaa.dtype
        k = s.k_states_factors
        Lambda = np.zeros((self.k_endog, k), dtype=dtype)
        for i in range(self.k_endog):
            y = self.endog[:, i:i + 1]
            iloc = self._s.endog_factor_iloc[i]
            factor_ix = s['factors_L1'][iloc]
            ix = (np.s_[:],) + np.ix_(factor_ix, factor_ix)
            A = Eaa[ix].sum(axis=0)
            B = y.T @ a[:, factor_ix, 0]
            if self.idiosyncratic_ar1:
                ix1 = s.k_states_factors + i
                ix2 = ix1 + 1
                B -= Eaa[:, ix1:ix2, factor_ix].sum(axis=0)
            try:
                Lambda[i, factor_ix] = cho_solve(cho_factor(A), B.T).T
            except LinAlgError:
                Lambda[i, factor_ix] = np.linalg.solve(A, B.T).T
        if compute_H:
            Z = self['design'].copy()
            Z[:, :k] = Lambda
            BL = self.endog.T @ a[..., 0] @ Z.T
            C = self.endog.T @ self.endog
            H = (C + -BL - BL.T + Z @ Eaa.sum(axis=0) @ Z.T) / self.nobs
        else:
            H = np.zeros((self.k_endog, self.k_endog), dtype=dtype) * np.nan
        return (Lambda, H)

    def _em_maximization_obs_missing(self, res, Eaa, a, compute_H=False):
        if False:
            i = 10
            return i + 15
        'EM maximization step, observation equation with missing data.'
        s = self._s
        dtype = Eaa.dtype
        k = s.k_states_factors
        Lambda = np.zeros((self.k_endog, k), dtype=dtype)
        W = 1 - res.missing.T
        mask = W.astype(bool)
        for i in range(self.k_endog_M):
            iloc = self._s.endog_factor_iloc[i]
            factor_ix = s['factors_L1'][iloc]
            m = mask[:, i]
            yt = self.endog[m, i:i + 1]
            ix = np.ix_(m, factor_ix, factor_ix)
            Ai = Eaa[ix].sum(axis=0)
            Bi = yt.T @ a[np.ix_(m, factor_ix)][..., 0]
            if self.idiosyncratic_ar1:
                ix1 = s.k_states_factors + i
                ix2 = ix1 + 1
                Bi -= Eaa[m, ix1:ix2][..., factor_ix].sum(axis=0)
            try:
                Lambda[i, factor_ix] = cho_solve(cho_factor(Ai), Bi.T).T
            except LinAlgError:
                Lambda[i, factor_ix] = np.linalg.solve(Ai, Bi.T).T
        if self.k_endog_Q > 0:
            multipliers = np.array([1, 2, 3, 2, 1])[:, None]
            for i in range(self.k_endog_M, self.k_endog):
                iloc = self._s.endog_factor_iloc[i]
                factor_ix = s['factors_L1_5_ix'][:, iloc].ravel().tolist()
                (R, _) = self.loading_constraints(i)
                iQ = i - self.k_endog_M
                m = mask[:, i]
                yt = self.endog[m, i:i + 1]
                ix = np.ix_(m, factor_ix, factor_ix)
                Ai = Eaa[ix].sum(axis=0)
                BiQ = yt.T @ a[np.ix_(m, factor_ix)][..., 0]
                if self.idiosyncratic_ar1:
                    ix = (np.s_[:],) + np.ix_(s['idio_ar_Q_ix'][iQ], factor_ix)
                    Eepsf = Eaa[ix]
                    BiQ -= (multipliers * Eepsf[m].sum(axis=0)).sum(axis=0)
                try:
                    L_and_lower = cho_factor(Ai)
                    unrestricted = cho_solve(L_and_lower, BiQ.T).T[0]
                    AiiRT = cho_solve(L_and_lower, R.T)
                    L_and_lower = cho_factor(R @ AiiRT)
                    RAiiRTiR = cho_solve(L_and_lower, R)
                    restricted = unrestricted - AiiRT @ RAiiRTiR @ unrestricted
                except LinAlgError:
                    Aii = np.linalg.inv(Ai)
                    unrestricted = (BiQ @ Aii)[0]
                    RARi = np.linalg.inv(R @ Aii @ R.T)
                    restricted = unrestricted - Aii @ R.T @ RARi @ R @ unrestricted
                Lambda[i, factor_ix] = restricted
        if compute_H:
            Z = self['design'].copy()
            Z[:, :Lambda.shape[1]] = Lambda
            y = np.nan_to_num(self.endog)
            C = y.T @ y
            W = W[..., None]
            IW = 1 - W
            WL = W * Z
            WLT = WL.transpose(0, 2, 1)
            BL = y[..., None] @ a.transpose(0, 2, 1) @ WLT
            A = Eaa
            BLT = BL.transpose(0, 2, 1)
            IWT = IW.transpose(0, 2, 1)
            H = (C + (-BL - BLT + WL @ A @ WLT + IW * self['obs_cov'] * IWT).sum(axis=0)) / self.nobs
        else:
            H = np.zeros((self.k_endog, self.k_endog), dtype=dtype) * np.nan
        return (Lambda, H)

    def smooth(self, params, transformed=True, includes_fixed=False, complex_step=False, cov_type='none', cov_kwds=None, return_ssm=False, results_class=None, results_wrapper_class=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Kalman smoothing.\n\n        Parameters\n        ----------\n        params : array_like\n            Array of parameters at which to evaluate the loglikelihood\n            function.\n        transformed : bool, optional\n            Whether or not `params` is already transformed. Default is True.\n        return_ssm : bool,optional\n            Whether or not to return only the state space output or a full\n            results object. Default is to return a full results object.\n        cov_type : str, optional\n            See `MLEResults.fit` for a description of covariance matrix types\n            for results object. Default is None.\n        cov_kwds : dict or None, optional\n            See `MLEResults.get_robustcov_results` for a description required\n            keywords for alternative covariance estimators\n        **kwargs\n            Additional keyword arguments to pass to the Kalman filter. See\n            `KalmanFilter.filter` for more details.\n        '
        return super().smooth(params, transformed=transformed, includes_fixed=includes_fixed, complex_step=complex_step, cov_type=cov_type, cov_kwds=cov_kwds, return_ssm=return_ssm, results_class=results_class, results_wrapper_class=results_wrapper_class, **kwargs)

    def filter(self, params, transformed=True, includes_fixed=False, complex_step=False, cov_type='none', cov_kwds=None, return_ssm=False, results_class=None, results_wrapper_class=None, low_memory=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Kalman filtering.\n\n        Parameters\n        ----------\n        params : array_like\n            Array of parameters at which to evaluate the loglikelihood\n            function.\n        transformed : bool, optional\n            Whether or not `params` is already transformed. Default is True.\n        return_ssm : bool,optional\n            Whether or not to return only the state space output or a full\n            results object. Default is to return a full results object.\n        cov_type : str, optional\n            See `MLEResults.fit` for a description of covariance matrix types\n            for results object. Default is 'none'.\n        cov_kwds : dict or None, optional\n            See `MLEResults.get_robustcov_results` for a description required\n            keywords for alternative covariance estimators\n        low_memory : bool, optional\n            If set to True, techniques are applied to substantially reduce\n            memory usage. If used, some features of the results object will\n            not be available (including in-sample prediction), although\n            out-of-sample forecasting is possible. Default is False.\n        **kwargs\n            Additional keyword arguments to pass to the Kalman filter. See\n            `KalmanFilter.filter` for more details.\n        "
        return super().filter(params, transformed=transformed, includes_fixed=includes_fixed, complex_step=complex_step, cov_type=cov_type, cov_kwds=cov_kwds, return_ssm=return_ssm, results_class=results_class, results_wrapper_class=results_wrapper_class, **kwargs)

    def simulate(self, params, nsimulations, measurement_shocks=None, state_shocks=None, initial_state=None, anchor=None, repetitions=None, exog=None, extend_model=None, extend_kwargs=None, transformed=True, includes_fixed=False, original_scale=True, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Simulate a new time series following the state space model.\n\n        Parameters\n        ----------\n        params : array_like\n            Array of parameters to use in constructing the state space\n            representation to use when simulating.\n        nsimulations : int\n            The number of observations to simulate. If the model is\n            time-invariant this can be any number. If the model is\n            time-varying, then this number must be less than or equal to the\n            number of observations.\n        measurement_shocks : array_like, optional\n            If specified, these are the shocks to the measurement equation,\n            :math:`\\varepsilon_t`. If unspecified, these are automatically\n            generated using a pseudo-random number generator. If specified,\n            must be shaped `nsimulations` x `k_endog`, where `k_endog` is the\n            same as in the state space model.\n        state_shocks : array_like, optional\n            If specified, these are the shocks to the state equation,\n            :math:`\\eta_t`. If unspecified, these are automatically\n            generated using a pseudo-random number generator. If specified,\n            must be shaped `nsimulations` x `k_posdef` where `k_posdef` is the\n            same as in the state space model.\n        initial_state : array_like, optional\n            If specified, this is the initial state vector to use in\n            simulation, which should be shaped (`k_states` x 1), where\n            `k_states` is the same as in the state space model. If unspecified,\n            but the model has been initialized, then that initialization is\n            used. This must be specified if `anchor` is anything other than\n            "start" or 0 (or else you can use the `simulate` method on a\n            results object rather than on the model object).\n        anchor : int, str, or datetime, optional\n            First period for simulation. The simulation will be conditional on\n            all existing datapoints prior to the `anchor`.  Type depends on the\n            index of the given `endog` in the model. Two special cases are the\n            strings \'start\' and \'end\'. `start` refers to beginning the\n            simulation at the first period of the sample, and `end` refers to\n            beginning the simulation at the first period after the sample.\n            Integer values can run from 0 to `nobs`, or can be negative to\n            apply negative indexing. Finally, if a date/time index was provided\n            to the model, then this argument can be a date string to parse or a\n            datetime type. Default is \'start\'.\n        repetitions : int, optional\n            Number of simulated paths to generate. Default is 1 simulated path.\n        exog : array_like, optional\n            New observations of exogenous regressors, if applicable.\n        transformed : bool, optional\n            Whether or not `params` is already transformed. Default is\n            True.\n        includes_fixed : bool, optional\n            If parameters were previously fixed with the `fix_params` method,\n            this argument describes whether or not `params` also includes\n            the fixed parameters, in addition to the free parameters. Default\n            is False.\n        original_scale : bool, optional\n            If the model specification standardized the data, whether or not\n            to return simulations in the original scale of the data (i.e.\n            before it was standardized by the model). Default is True.\n\n        Returns\n        -------\n        simulated_obs : ndarray\n            An array of simulated observations. If `repetitions=None`, then it\n            will be shaped (nsimulations x k_endog) or (nsimulations,) if\n            `k_endog=1`. Otherwise it will be shaped\n            (nsimulations x k_endog x repetitions). If the model was given\n            Pandas input then the output will be a Pandas object. If\n            `k_endog > 1` and `repetitions` is not None, then the output will\n            be a Pandas DataFrame that has a MultiIndex for the columns, with\n            the first level containing the names of the `endog` variables and\n            the second level containing the repetition number.\n        '
        sim = super().simulate(params, nsimulations, measurement_shocks=measurement_shocks, state_shocks=state_shocks, initial_state=initial_state, anchor=anchor, repetitions=repetitions, exog=exog, extend_model=extend_model, extend_kwargs=extend_kwargs, transformed=transformed, includes_fixed=includes_fixed, **kwargs)
        if self.standardize and original_scale:
            use_pandas = isinstance(self.data, PandasData)
            shape = sim.shape
            if use_pandas:
                if len(shape) == 1:
                    sim *= self._endog_std.iloc[0]
                    sim += self._endog_mean.iloc[0]
                elif len(shape) == 2:
                    sim = sim.multiply(self._endog_std, axis=1, level=0).add(self._endog_mean, axis=1, level=0)
            elif len(shape) == 1:
                sim = sim * self._endog_std + self._endog_mean
            elif len(shape) == 2:
                sim = sim * self._endog_std + self._endog_mean
            else:
                std = np.atleast_2d(self._endog_std)[..., None]
                mean = np.atleast_2d(self._endog_mean)[..., None]
                sim = sim * std + mean
        return sim

    def impulse_responses(self, params, steps=1, impulse=0, orthogonalized=False, cumulative=False, anchor=None, exog=None, extend_model=None, extend_kwargs=None, transformed=True, includes_fixed=False, original_scale=True, **kwargs):
        if False:
            return 10
        '\n        Impulse response function.\n\n        Parameters\n        ----------\n        params : array_like\n            Array of model parameters.\n        steps : int, optional\n            The number of steps for which impulse responses are calculated.\n            Default is 1. Note that for time-invariant models, the initial\n            impulse is not counted as a step, so if `steps=1`, the output will\n            have 2 entries.\n        impulse : int or array_like\n            If an integer, the state innovation to pulse; must be between 0\n            and `k_posdef-1`. Alternatively, a custom impulse vector may be\n            provided; must be shaped `k_posdef x 1`.\n        orthogonalized : bool, optional\n            Whether or not to perform impulse using orthogonalized innovations.\n            Note that this will also affect custum `impulse` vectors. Default\n            is False.\n        cumulative : bool, optional\n            Whether or not to return cumulative impulse responses. Default is\n            False.\n        anchor : int, str, or datetime, optional\n            Time point within the sample for the state innovation impulse. Type\n            depends on the index of the given `endog` in the model. Two special\n            cases are the strings \'start\' and \'end\', which refer to setting the\n            impulse at the first and last points of the sample, respectively.\n            Integer values can run from 0 to `nobs - 1`, or can be negative to\n            apply negative indexing. Finally, if a date/time index was provided\n            to the model, then this argument can be a date string to parse or a\n            datetime type. Default is \'start\'.\n        exog : array_like, optional\n            New observations of exogenous regressors for our-of-sample periods,\n            if applicable.\n        transformed : bool, optional\n            Whether or not `params` is already transformed. Default is\n            True.\n        includes_fixed : bool, optional\n            If parameters were previously fixed with the `fix_params` method,\n            this argument describes whether or not `params` also includes\n            the fixed parameters, in addition to the free parameters. Default\n            is False.\n        original_scale : bool, optional\n            If the model specification standardized the data, whether or not\n            to return impulse responses in the original scale of the data (i.e.\n            before it was standardized by the model). Default is True.\n        **kwargs\n            If the model has time-varying design or transition matrices and the\n            combination of `anchor` and `steps` implies creating impulse\n            responses for the out-of-sample period, then these matrices must\n            have updated values provided for the out-of-sample steps. For\n            example, if `design` is a time-varying component, `nobs` is 10,\n            `anchor=1`, and `steps` is 15, a (`k_endog` x `k_states` x 7)\n            matrix must be provided with the new design matrix values.\n\n        Returns\n        -------\n        impulse_responses : ndarray\n            Responses for each endogenous variable due to the impulse\n            given by the `impulse` argument. For a time-invariant model, the\n            impulse responses are given for `steps + 1` elements (this gives\n            the "initial impulse" followed by `steps` responses for the\n            important cases of VAR and SARIMAX models), while for time-varying\n            models the impulse responses are only given for `steps` elements\n            (to avoid having to unexpectedly provide updated time-varying\n            matrices).\n\n        '
        irfs = super().impulse_responses(params, steps=steps, impulse=impulse, orthogonalized=orthogonalized, cumulative=cumulative, anchor=anchor, exog=exog, extend_model=extend_model, extend_kwargs=extend_kwargs, transformed=transformed, includes_fixed=includes_fixed, **kwargs)
        if self.standardize and original_scale:
            use_pandas = isinstance(self.data, PandasData)
            shape = irfs.shape
            if use_pandas:
                if len(shape) == 1:
                    irfs = irfs * self._endog_std.iloc[0]
                elif len(shape) == 2:
                    irfs = irfs.multiply(self._endog_std, axis=1, level=0)
            elif len(shape) == 1:
                irfs = irfs * self._endog_std
            elif len(shape) == 2:
                irfs = irfs * self._endog_std
        return irfs

class DynamicFactorMQResults(mlemodel.MLEResults):
    """
    Results from fitting a dynamic factor model
    """

    def __init__(self, model, params, filter_results, cov_type=None, **kwargs):
        if False:
            while True:
                i = 10
        super(DynamicFactorMQResults, self).__init__(model, params, filter_results, cov_type, **kwargs)

    @property
    def factors(self):
        if False:
            while True:
                i = 10
        '\n        Estimates of unobserved factors.\n\n        Returns\n        -------\n        out : Bunch\n            Has the following attributes shown in Notes.\n\n        Notes\n        -----\n        The output is a bunch of the following format:\n\n        - `filtered`: a time series array with the filtered estimate of\n          the component\n        - `filtered_cov`: a time series array with the filtered estimate of\n          the variance/covariance of the component\n        - `smoothed`: a time series array with the smoothed estimate of\n          the component\n        - `smoothed_cov`: a time series array with the smoothed estimate of\n          the variance/covariance of the component\n        - `offset`: an integer giving the offset in the state vector where\n          this component begins\n        '
        out = None
        if self.model.k_factors > 0:
            iloc = self.model._s.factors_L1
            ix = np.array(self.model.state_names)[iloc].tolist()
            out = Bunch(filtered=self.states.filtered.loc[:, ix], filtered_cov=self.states.filtered_cov.loc[np.s_[ix, :], ix], smoothed=None, smoothed_cov=None)
            if self.smoothed_state is not None:
                out.smoothed = self.states.smoothed.loc[:, ix]
            if self.smoothed_state_cov is not None:
                out.smoothed_cov = self.states.smoothed_cov.loc[np.s_[ix, :], ix]
        return out

    def get_coefficients_of_determination(self, method='individual', which=None):
        if False:
            i = 10
            return i + 15
        '\n        Get coefficients of determination (R-squared) for variables / factors.\n\n        Parameters\n        ----------\n        method : {\'individual\', \'joint\', \'cumulative\'}, optional\n            The type of R-squared values to generate. "individual" plots\n            the R-squared of each variable on each factor; "joint" plots the\n            R-squared of each variable on each factor that it loads on;\n            "cumulative" plots the successive R-squared values as each\n            additional factor is added to the regression, for each variable.\n            Default is \'individual\'.\n        which: {None, \'filtered\', \'smoothed\'}, optional\n            Whether to compute R-squared values based on filtered or smoothed\n            estimates of the factors. Default is \'smoothed\' if smoothed results\n            are available and \'filtered\' otherwise.\n\n        Returns\n        -------\n        rsquared : pd.DataFrame or pd.Series\n            The R-squared values from regressions of observed variables on\n            one or more of the factors. If method=\'individual\' or\n            method=\'cumulative\', this will be a Pandas DataFrame with observed\n            variables as the index and factors as the columns . If\n            method=\'joint\', will be a Pandas Series with observed variables as\n            the index.\n\n        See Also\n        --------\n        plot_coefficients_of_determination\n        coefficients_of_determination\n        '
        from statsmodels.tools import add_constant
        method = string_like(method, 'method', options=['individual', 'joint', 'cumulative'])
        if which is None:
            which = 'filtered' if self.smoothed_state is None else 'smoothed'
        k_endog = self.model.k_endog
        k_factors = self.model.k_factors
        ef_map = self.model._s.endog_factor_map
        endog_names = self.model.endog_names
        factor_names = self.model.factor_names
        if method == 'individual':
            coefficients = np.zeros((k_endog, k_factors))
            for i in range(k_factors):
                exog = add_constant(self.factors[which].iloc[:, i])
                for j in range(k_endog):
                    if ef_map.iloc[j, i]:
                        endog = self.filter_results.endog[j]
                        coefficients[j, i] = OLS(endog, exog, missing='drop').fit().rsquared
                    else:
                        coefficients[j, i] = np.nan
            coefficients = pd.DataFrame(coefficients, index=endog_names, columns=factor_names)
        elif method == 'joint':
            coefficients = np.zeros((k_endog,))
            exog = add_constant(self.factors[which])
            for j in range(k_endog):
                endog = self.filter_results.endog[j]
                ix = np.r_[True, ef_map.iloc[j]].tolist()
                X = exog.loc[:, ix]
                coefficients[j] = OLS(endog, X, missing='drop').fit().rsquared
            coefficients = pd.Series(coefficients, index=endog_names)
        elif method == 'cumulative':
            coefficients = np.zeros((k_endog, k_factors))
            exog = add_constant(self.factors[which])
            for j in range(k_endog):
                endog = self.filter_results.endog[j]
                for i in range(k_factors):
                    if self.model._s.endog_factor_map.iloc[j, i]:
                        ix = np.r_[True, ef_map.iloc[j, :i + 1], [False] * (k_factors - i - 1)]
                        X = exog.loc[:, ix.astype(bool).tolist()]
                        coefficients[j, i] = OLS(endog, X, missing='drop').fit().rsquared
                    else:
                        coefficients[j, i] = np.nan
            coefficients = pd.DataFrame(coefficients, index=endog_names, columns=factor_names)
        return coefficients

    @cache_readonly
    def coefficients_of_determination(self):
        if False:
            i = 10
            return i + 15
        '\n        Individual coefficients of determination (:math:`R^2`).\n\n        Coefficients of determination (:math:`R^2`) from regressions of\n        endogenous variables on individual estimated factors.\n\n        Returns\n        -------\n        coefficients_of_determination : ndarray\n            A `k_endog` x `k_factors` array, where\n            `coefficients_of_determination[i, j]` represents the :math:`R^2`\n            value from a regression of factor `j` and a constant on endogenous\n            variable `i`.\n\n        Notes\n        -----\n        Although it can be difficult to interpret the estimated factor loadings\n        and factors, it is often helpful to use the coefficients of\n        determination from univariate regressions to assess the importance of\n        each factor in explaining the variation in each endogenous variable.\n\n        In models with many variables and factors, this can sometimes lend\n        interpretation to the factors (for example sometimes one factor will\n        load primarily on real variables and another on nominal variables).\n\n        See Also\n        --------\n        get_coefficients_of_determination\n        plot_coefficients_of_determination\n        '
        return self.get_coefficients_of_determination(method='individual')

    def plot_coefficients_of_determination(self, method='individual', which=None, endog_labels=None, fig=None, figsize=None):
        if False:
            while True:
                i = 10
        '\n        Plot coefficients of determination (R-squared) for variables / factors.\n\n        Parameters\n        ----------\n        method : {\'individual\', \'joint\', \'cumulative\'}, optional\n            The type of R-squared values to generate. "individual" plots\n            the R-squared of each variable on each factor; "joint" plots the\n            R-squared of each variable on each factor that it loads on;\n            "cumulative" plots the successive R-squared values as each\n            additional factor is added to the regression, for each variable.\n            Default is \'individual\'.\n        which: {None, \'filtered\', \'smoothed\'}, optional\n            Whether to compute R-squared values based on filtered or smoothed\n            estimates of the factors. Default is \'smoothed\' if smoothed results\n            are available and \'filtered\' otherwise.\n        endog_labels : bool, optional\n            Whether or not to label the endogenous variables along the x-axis\n            of the plots. Default is to include labels if there are 5 or fewer\n            endogenous variables.\n        fig : Figure, optional\n            If given, subplots are created in this figure instead of in a new\n            figure. Note that the grid will be created in the provided\n            figure using `fig.add_subplot()`.\n        figsize : tuple, optional\n            If a figure is created, this argument allows specifying a size.\n            The tuple is (width, height).\n\n        Notes\n        -----\n        The endogenous variables are arranged along the x-axis according to\n        their position in the model\'s `endog` array.\n\n        See Also\n        --------\n        get_coefficients_of_determination\n        '
        from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
        _import_mpl()
        fig = create_mpl_fig(fig, figsize)
        method = string_like(method, 'method', options=['individual', 'joint', 'cumulative'])
        if endog_labels is None:
            endog_labels = self.model.k_endog <= 5
        rsquared = self.get_coefficients_of_determination(method=method, which=which)
        if method in ['individual', 'cumulative']:
            plot_idx = 1
            for (factor_name, coeffs) in rsquared.T.iterrows():
                ax = fig.add_subplot(self.model.k_factors, 1, plot_idx)
                ax.set_ylim((0, 1))
                ax.set(title=f'{factor_name}', ylabel='$R^2$')
                coeffs.plot(ax=ax, kind='bar')
                if plot_idx < len(rsquared.columns) or not endog_labels:
                    ax.xaxis.set_ticklabels([])
                plot_idx += 1
        elif method == 'joint':
            ax = fig.add_subplot(1, 1, 1)
            ax.set_ylim((0, 1))
            ax.set(title='$R^2$ - regression on all loaded factors', ylabel='$R^2$')
            rsquared.plot(ax=ax, kind='bar')
            if not endog_labels:
                ax.xaxis.set_ticklabels([])
        return fig

    def get_prediction(self, start=None, end=None, dynamic=False, information_set='predicted', signal_only=False, original_scale=True, index=None, exog=None, extend_model=None, extend_kwargs=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        In-sample prediction and out-of-sample forecasting.\n\n        Parameters\n        ----------\n        start : int, str, or datetime, optional\n            Zero-indexed observation number at which to start forecasting,\n            i.e., the first forecast is start. Can also be a date string to\n            parse or a datetime type. Default is the the zeroth observation.\n        end : int, str, or datetime, optional\n            Zero-indexed observation number at which to end forecasting, i.e.,\n            the last forecast is end. Can also be a date string to\n            parse or a datetime type. However, if the dates index does not\n            have a fixed frequency, end must be an integer index if you\n            want out of sample prediction. Default is the last observation in\n            the sample.\n        dynamic : bool, int, str, or datetime, optional\n            Integer offset relative to `start` at which to begin dynamic\n            prediction. Can also be an absolute date string to parse or a\n            datetime type (these are not interpreted as offsets).\n            Prior to this observation, true endogenous values will be used for\n            prediction; starting with this observation and continuing through\n            the end of prediction, forecasted endogenous values will be used\n            instead.\n        information_set : str, optional\n            The information set to condition each prediction on. Default is\n            "predicted", which computes predictions of period t values\n            conditional on observed data through period t-1; these are\n            one-step-ahead predictions, and correspond with the typical\n            `fittedvalues` results attribute. Alternatives are "filtered",\n            which computes predictions of period t values conditional on\n            observed data through period t, and "smoothed", which computes\n            predictions of period t values conditional on the entire dataset\n            (including also future observations t+1, t+2, ...).\n        signal_only : bool, optional\n            Whether to compute forecasts of only the "signal" component of\n            the observation equation. Default is False. For example, the\n            observation equation of a time-invariant model is\n            :math:`y_t = d + Z \\alpha_t + \\varepsilon_t`, and the "signal"\n            component is then :math:`Z \\alpha_t`. If this argument is set to\n            True, then forecasts of the "signal" :math:`Z \\alpha_t` will be\n            returned. Otherwise, the default is for forecasts of :math:`y_t`\n            to be returned.\n        original_scale : bool, optional\n            If the model specification standardized the data, whether or not\n            to return predictions in the original scale of the data (i.e.\n            before it was standardized by the model). Default is True.\n        **kwargs\n            Additional arguments may required for forecasting beyond the end\n            of the sample. See `FilterResults.predict` for more details.\n\n        Returns\n        -------\n        forecast : ndarray\n            Array of out of in-sample predictions and / or out-of-sample\n            forecasts. An (npredict x k_endog) array.\n        '
        res = super().get_prediction(start=start, end=end, dynamic=dynamic, information_set=information_set, signal_only=signal_only, index=index, exog=exog, extend_model=extend_model, extend_kwargs=extend_kwargs, **kwargs)
        if self.model.standardize and original_scale:
            prediction_results = res.prediction_results
            (k_endog, _) = prediction_results.endog.shape
            mean = np.array(self.model._endog_mean)
            std = np.array(self.model._endog_std)
            if self.model.k_endog > 1:
                mean = mean[None, :]
                std = std[None, :]
            res._results._predicted_mean = res._results._predicted_mean * std + mean
            if k_endog == 1:
                res._results._var_pred_mean *= std ** 2
            else:
                res._results._var_pred_mean = std * res._results._var_pred_mean * std.T
        return res

    def news(self, comparison, impact_date=None, impacted_variable=None, start=None, end=None, periods=None, exog=None, comparison_type=None, revisions_details_start=False, state_index=None, return_raw=False, tolerance=1e-10, endog_quarterly=None, original_scale=True, **kwargs):
        if False:
            return 10
        '\n        Compute impacts from updated data (news and revisions).\n\n        Parameters\n        ----------\n        comparison : array_like or MLEResults\n            An updated dataset with updated and/or revised data from which the\n            news can be computed, or an updated or previous results object\n            to use in computing the news.\n        impact_date : int, str, or datetime, optional\n            A single specific period of impacts from news and revisions to\n            compute. Can also be a date string to parse or a datetime type.\n            This argument cannot be used in combination with `start`, `end`, or\n            `periods`. Default is the first out-of-sample observation.\n        impacted_variable : str, list, array, or slice, optional\n            Observation variable label or slice of labels specifying that only\n            specific impacted variables should be shown in the News output. The\n            impacted variable(s) describe the variables that were *affected* by\n            the news. If you do not know the labels for the variables, check\n            the `endog_names` attribute of the model instance.\n        start : int, str, or datetime, optional\n            The first period of impacts from news and revisions to compute.\n            Can also be a date string to parse or a datetime type. Default is\n            the first out-of-sample observation.\n        end : int, str, or datetime, optional\n            The last period of impacts from news and revisions to compute.\n            Can also be a date string to parse or a datetime type. Default is\n            the first out-of-sample observation.\n        periods : int, optional\n            The number of periods of impacts from news and revisions to\n            compute.\n        exog : array_like, optional\n            Array of exogenous regressors for the out-of-sample period, if\n            applicable.\n        comparison_type : {None, \'previous\', \'updated\'}\n            This denotes whether the `comparison` argument represents a\n            *previous* results object or dataset or an *updated* results object\n            or dataset. If not specified, then an attempt is made to determine\n            the comparison type.\n        state_index : array_like or "common", optional\n            An optional index specifying a subset of states to use when\n            constructing the impacts of revisions and news. For example, if\n            `state_index=[0, 1]` is passed, then only the impacts to the\n            observed variables arising from the impacts to the first two\n            states will be returned. If the string "common" is passed and the\n            model includes idiosyncratic AR(1) components, news will only be\n            computed based on the common states. Default is to use all states.\n        return_raw : bool, optional\n            Whether or not to return only the specific output or a full\n            results object. Default is to return a full results object.\n        tolerance : float, optional\n            The numerical threshold for determining zero impact. Default is\n            that any impact less than 1e-10 is assumed to be zero.\n        endog_quarterly : array_like, optional\n            New observations of quarterly variables, if `comparison` was\n            provided as an updated monthly dataset. If this argument is\n            provided, it must be a Pandas Series or DataFrame with a\n            DatetimeIndex or PeriodIndex at the quarterly frequency.\n\n        References\n        ----------\n        .. [1] Bańbura, Marta, and Michele Modugno.\n               "Maximum likelihood estimation of factor models on datasets with\n               arbitrary pattern of missing data."\n               Journal of Applied Econometrics 29, no. 1 (2014): 133-160.\n        .. [2] Bańbura, Marta, Domenico Giannone, and Lucrezia Reichlin.\n               "Nowcasting."\n               The Oxford Handbook of Economic Forecasting. July 8, 2011.\n        .. [3] Bańbura, Marta, Domenico Giannone, Michele Modugno, and Lucrezia\n               Reichlin.\n               "Now-casting and the real-time data flow."\n               In Handbook of economic forecasting, vol. 2, pp. 195-237.\n               Elsevier, 2013.\n        '
        if state_index == 'common':
            state_index = np.arange(self.model.k_states - self.model.k_endog)
        news_results = super().news(comparison, impact_date=impact_date, impacted_variable=impacted_variable, start=start, end=end, periods=periods, exog=exog, comparison_type=comparison_type, revisions_details_start=revisions_details_start, state_index=state_index, return_raw=return_raw, tolerance=tolerance, endog_quarterly=endog_quarterly, **kwargs)
        if not return_raw and self.model.standardize and original_scale:
            endog_mean = self.model._endog_mean
            endog_std = self.model._endog_std
            news_results.total_impacts = news_results.total_impacts * endog_std
            news_results.update_impacts = news_results.update_impacts * endog_std
            if news_results.revision_impacts is not None:
                news_results.revision_impacts = news_results.revision_impacts * endog_std
            if news_results.revision_detailed_impacts is not None:
                news_results.revision_detailed_impacts = news_results.revision_detailed_impacts * endog_std
            if news_results.revision_grouped_impacts is not None:
                news_results.revision_grouped_impacts = news_results.revision_grouped_impacts * endog_std
            for name in ['prev_impacted_forecasts', 'news', 'revisions', 'update_realized', 'update_forecasts', 'revised', 'revised_prev', 'post_impacted_forecasts', 'revisions_all', 'revised_all', 'revised_prev_all']:
                dta = getattr(news_results, name)
                orig_name = None
                if hasattr(dta, 'name'):
                    orig_name = dta.name
                dta = dta.multiply(endog_std, level=1)
                if name not in ['news', 'revisions']:
                    dta = dta.add(endog_mean, level=1)
                if orig_name is not None:
                    dta.name = orig_name
                setattr(news_results, name, dta)
            news_results.weights = news_results.weights.divide(endog_std, axis=0, level=1).multiply(endog_std, axis=1, level=1)
            news_results.revision_weights = news_results.revision_weights.divide(endog_std, axis=0, level=1).multiply(endog_std, axis=1, level=1)
        return news_results

    def get_smoothed_decomposition(self, decomposition_of='smoothed_state', state_index=None, original_scale=True):
        if False:
            print('Hello World!')
        '\n        Decompose smoothed output into contributions from observations\n\n        Parameters\n        ----------\n        decomposition_of : {"smoothed_state", "smoothed_signal"}\n            The object to perform a decomposition of. If it is set to\n            "smoothed_state", then the elements of the smoothed state vector\n            are decomposed into the contributions of each observation. If it\n            is set to "smoothed_signal", then the predictions of the\n            observation vector based on the smoothed state vector are\n            decomposed. Default is "smoothed_state".\n        state_index : array_like, optional\n            An optional index specifying a subset of states to use when\n            constructing the decomposition of the "smoothed_signal". For\n            example, if `state_index=[0, 1]` is passed, then only the\n            contributions of observed variables to the smoothed signal arising\n            from the first two states will be returned. Note that if not all\n            states are used, the contributions will not sum to the smoothed\n            signal. Default is to use all states.\n        original_scale : bool, optional\n            If the model specification standardized the data, whether or not\n            to return simulations in the original scale of the data (i.e.\n            before it was standardized by the model). Default is True.\n\n        Returns\n        -------\n        data_contributions : pd.DataFrame\n            Contributions of observations to the decomposed object. If the\n            smoothed state is being decomposed, then `data_contributions` is\n            shaped `(k_states x nobs, k_endog x nobs)` with a `pd.MultiIndex`\n            index corresponding to `state_to x date_to` and `pd.MultiIndex`\n            columns corresponding to `variable_from x date_from`. If the\n            smoothed signal is being decomposed, then `data_contributions` is\n            shaped `(k_endog x nobs, k_endog x nobs)` with `pd.MultiIndex`-es\n            corresponding to `variable_to x date_to` and\n            `variable_from x date_from`.\n        obs_intercept_contributions : pd.DataFrame\n            Contributions of the observation intercept to the decomposed\n            object. If the smoothed state is being decomposed, then\n            `obs_intercept_contributions` is\n            shaped `(k_states x nobs, k_endog x nobs)` with a `pd.MultiIndex`\n            index corresponding to `state_to x date_to` and `pd.MultiIndex`\n            columns corresponding to `obs_intercept_from x date_from`. If the\n            smoothed signal is being decomposed, then\n            `obs_intercept_contributions` is shaped\n            `(k_endog x nobs, k_endog x nobs)` with `pd.MultiIndex`-es\n            corresponding to `variable_to x date_to` and\n            `obs_intercept_from x date_from`.\n        state_intercept_contributions : pd.DataFrame\n            Contributions of the state intercept to the decomposed\n            object. If the smoothed state is being decomposed, then\n            `state_intercept_contributions` is\n            shaped `(k_states x nobs, k_states x nobs)` with a `pd.MultiIndex`\n            index corresponding to `state_to x date_to` and `pd.MultiIndex`\n            columns corresponding to `state_intercept_from x date_from`. If the\n            smoothed signal is being decomposed, then\n            `state_intercept_contributions` is shaped\n            `(k_endog x nobs, k_states x nobs)` with `pd.MultiIndex`-es\n            corresponding to `variable_to x date_to` and\n            `state_intercept_from x date_from`.\n        prior_contributions : pd.DataFrame\n            Contributions of the prior to the decomposed object. If the\n            smoothed state is being decomposed, then `prior_contributions` is\n            shaped `(nobs x k_states, k_states)`, with a `pd.MultiIndex`\n            index corresponding to `state_to x date_to` and columns\n            corresponding to elements of the prior mean (aka "initial state").\n            If the smoothed signal is being decomposed, then\n            `prior_contributions` is shaped `(nobs x k_endog, k_states)`,\n            with a `pd.MultiIndex` index corresponding to\n            `variable_to x date_to` and columns corresponding to elements of\n            the prior mean.\n\n        Notes\n        -----\n        Denote the smoothed state at time :math:`t` by :math:`\\alpha_t`. Then\n        the smoothed signal is :math:`Z_t \\alpha_t`, where :math:`Z_t` is the\n        design matrix operative at time :math:`t`.\n        '
        if self.model.standardize and original_scale:
            cache_obs_intercept = self.model['obs_intercept']
            self.model['obs_intercept'] = self.model._endog_mean
        (data_contributions, obs_intercept_contributions, state_intercept_contributions, prior_contributions) = super().get_smoothed_decomposition(decomposition_of=decomposition_of, state_index=state_index)
        if self.model.standardize and original_scale:
            self.model['obs_intercept'] = cache_obs_intercept
        if decomposition_of == 'smoothed_signal' and self.model.standardize and original_scale:
            endog_std = self.model._endog_std
            data_contributions = data_contributions.multiply(endog_std, axis=0, level=0)
            obs_intercept_contributions = obs_intercept_contributions.multiply(endog_std, axis=0, level=0)
            state_intercept_contributions = state_intercept_contributions.multiply(endog_std, axis=0, level=0)
            prior_contributions = prior_contributions.multiply(endog_std, axis=0, level=0)
        return (data_contributions, obs_intercept_contributions, state_intercept_contributions, prior_contributions)

    def append(self, endog, endog_quarterly=None, refit=False, fit_kwargs=None, copy_initialization=True, retain_standardization=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Recreate the results object with new data appended to original data.\n\n        Creates a new result object applied to a dataset that is created by\n        appending new data to the end of the model's original data. The new\n        results can then be used for analysis or forecasting.\n\n        Parameters\n        ----------\n        endog : array_like\n            New observations from the modeled time-series process.\n        endog_quarterly : array_like, optional\n            New observations of quarterly variables. If provided, must be a\n            Pandas Series or DataFrame with a DatetimeIndex or PeriodIndex at\n            the quarterly frequency.\n        refit : bool, optional\n            Whether to re-fit the parameters, based on the combined dataset.\n            Default is False (so parameters from the current results object\n            are used to create the new results object).\n        fit_kwargs : dict, optional\n            Keyword arguments to pass to `fit` (if `refit=True`) or `filter` /\n            `smooth`.\n        copy_initialization : bool, optional\n            Whether or not to copy the initialization from the current results\n            set to the new model. Default is True.\n        retain_standardization : bool, optional\n            Whether or not to use the mean and standard deviations that were\n            used to standardize the data in the current model in the new model.\n            Default is True.\n        **kwargs\n            Keyword arguments may be used to modify model specification\n            arguments when created the new model object.\n\n        Returns\n        -------\n        results\n            Updated Results object, that includes results from both the\n            original dataset and the new dataset.\n\n        Notes\n        -----\n        The `endog` and `exog` arguments to this method must be formatted in\n        the same way (e.g. Pandas Series versus Numpy array) as were the\n        `endog` and `exog` arrays passed to the original model.\n\n        The `endog` (and, if applicable, `endog_quarterly`) arguments to this\n        method should consist of new observations that occurred directly after\n        the last element of `endog`. For any other kind of dataset, see the\n        `apply` method.\n\n        This method will apply filtering to all of the original data as well\n        as to the new data. To apply filtering only to the new data (which\n        can be much faster if the original dataset is large), see the `extend`\n        method.\n\n        See Also\n        --------\n        extend\n        apply\n        "
        (endog, k_endog_monthly) = DynamicFactorMQ.construct_endog(endog, endog_quarterly)
        k_endog = endog.shape[1] if len(endog.shape) == 2 else 1
        if k_endog_monthly != self.model.k_endog_M or k_endog != self.model.k_endog:
            raise ValueError('Cannot append data of a different dimension to a model.')
        kwargs['k_endog_monthly'] = k_endog_monthly
        return super().append(endog, refit=refit, fit_kwargs=fit_kwargs, copy_initialization=copy_initialization, retain_standardization=retain_standardization, **kwargs)

    def extend(self, endog, endog_quarterly=None, fit_kwargs=None, retain_standardization=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Recreate the results object for new data that extends original data.\n\n        Creates a new result object applied to a new dataset that is assumed to\n        follow directly from the end of the model's original data. The new\n        results can then be used for analysis or forecasting.\n\n        Parameters\n        ----------\n        endog : array_like\n            New observations from the modeled time-series process.\n        endog_quarterly : array_like, optional\n            New observations of quarterly variables. If provided, must be a\n            Pandas Series or DataFrame with a DatetimeIndex or PeriodIndex at\n            the quarterly frequency.\n        fit_kwargs : dict, optional\n            Keyword arguments to pass to `filter` or `smooth`.\n        retain_standardization : bool, optional\n            Whether or not to use the mean and standard deviations that were\n            used to standardize the data in the current model in the new model.\n            Default is True.\n        **kwargs\n            Keyword arguments may be used to modify model specification\n            arguments when created the new model object.\n\n        Returns\n        -------\n        results\n            Updated Results object, that includes results only for the new\n            dataset.\n\n        See Also\n        --------\n        append\n        apply\n\n        Notes\n        -----\n        The `endog` argument to this method should consist of new observations\n        that occurred directly after the last element of the model's original\n        `endog` array. For any other kind of dataset, see the `apply` method.\n\n        This method will apply filtering only to the new data provided by the\n        `endog` argument, which can be much faster than re-filtering the entire\n        dataset. However, the returned results object will only have results\n        for the new data. To retrieve results for both the new data and the\n        original data, see the `append` method.\n        "
        (endog, k_endog_monthly) = DynamicFactorMQ.construct_endog(endog, endog_quarterly)
        k_endog = endog.shape[1] if len(endog.shape) == 2 else 1
        if k_endog_monthly != self.model.k_endog_M or k_endog != self.model.k_endog:
            raise ValueError('Cannot append data of a different dimension to a model.')
        kwargs['k_endog_monthly'] = k_endog_monthly
        return super().extend(endog, fit_kwargs=fit_kwargs, retain_standardization=retain_standardization, **kwargs)

    def apply(self, endog, k_endog_monthly=None, endog_quarterly=None, refit=False, fit_kwargs=None, copy_initialization=False, retain_standardization=True, **kwargs):
        if False:
            return 10
        "\n        Apply the fitted parameters to new data unrelated to the original data.\n\n        Creates a new result object using the current fitted parameters,\n        applied to a completely new dataset that is assumed to be unrelated to\n        the model's original data. The new results can then be used for\n        analysis or forecasting.\n\n        Parameters\n        ----------\n        endog : array_like\n            New observations from the modeled time-series process.\n        k_endog_monthly : int, optional\n            If specifying a monthly/quarterly mixed frequency model in which\n            the provided `endog` dataset contains both the monthly and\n            quarterly data, this variable should be used to indicate how many\n            of the variables are monthly.\n        endog_quarterly : array_like, optional\n            New observations of quarterly variables. If provided, must be a\n            Pandas Series or DataFrame with a DatetimeIndex or PeriodIndex at\n            the quarterly frequency.\n        refit : bool, optional\n            Whether to re-fit the parameters, using the new dataset.\n            Default is False (so parameters from the current results object\n            are used to create the new results object).\n        fit_kwargs : dict, optional\n            Keyword arguments to pass to `fit` (if `refit=True`) or `filter` /\n            `smooth`.\n        copy_initialization : bool, optional\n            Whether or not to copy the initialization from the current results\n            set to the new model. Default is False.\n        retain_standardization : bool, optional\n            Whether or not to use the mean and standard deviations that were\n            used to standardize the data in the current model in the new model.\n            Default is True.\n        **kwargs\n            Keyword arguments may be used to modify model specification\n            arguments when created the new model object.\n\n        Returns\n        -------\n        results\n            Updated Results object, that includes results only for the new\n            dataset.\n\n        See Also\n        --------\n        statsmodels.tsa.statespace.mlemodel.MLEResults.append\n        statsmodels.tsa.statespace.mlemodel.MLEResults.apply\n\n        Notes\n        -----\n        The `endog` argument to this method should consist of new observations\n        that are not necessarily related to the original model's `endog`\n        dataset. For observations that continue that original dataset by follow\n        directly after its last element, see the `append` and `extend` methods.\n        "
        mod = self.model.clone(endog, k_endog_monthly=k_endog_monthly, endog_quarterly=endog_quarterly, retain_standardization=retain_standardization, **kwargs)
        if copy_initialization:
            init = initialization.Initialization.from_results(self.filter_results)
            mod.ssm.initialization = init
        res = self._apply(mod, refit=refit, fit_kwargs=fit_kwargs)
        return res

    def summary(self, alpha=0.05, start=None, title=None, model_name=None, display_params=True, display_diagnostics=False, display_params_as_list=False, truncate_endog_names=None, display_max_endog=3):
        if False:
            return 10
        '\n        Summarize the Model.\n\n        Parameters\n        ----------\n        alpha : float, optional\n            Significance level for the confidence intervals. Default is 0.05.\n        start : int, optional\n            Integer of the start observation. Default is 0.\n        title : str, optional\n            The title used for the summary table.\n        model_name : str, optional\n            The name of the model used. Default is to use model class name.\n\n        Returns\n        -------\n        summary : Summary instance\n            This holds the summary table and text, which can be printed or\n            converted to various output formats.\n\n        See Also\n        --------\n        statsmodels.iolib.summary.Summary\n        '
        mod = self.model
        if title is None:
            title = 'Dynamic Factor Results'
        if model_name is None:
            model_name = self.model._model_name
        endog_names = self.model._get_endog_names(truncate=truncate_endog_names)
        extra_top_left = None
        extra_top_right = []
        mle_retvals = getattr(self, 'mle_retvals', None)
        mle_settings = getattr(self, 'mle_settings', None)
        if mle_settings is not None and mle_settings.method == 'em':
            extra_top_right += [('EM Iterations', [f'{mle_retvals.iter}'])]
        summary = super().summary(alpha=alpha, start=start, title=title, model_name=model_name, display_params=display_params and display_params_as_list, display_diagnostics=display_diagnostics, truncate_endog_names=truncate_endog_names, display_max_endog=display_max_endog, extra_top_left=extra_top_left, extra_top_right=extra_top_right)
        table_ix = 1
        if not display_params_as_list:
            data = pd.DataFrame(self.filter_results.design[:, mod._s['factors_L1'], 0], index=endog_names, columns=mod.factor_names)
            try:
                data = data.map(lambda s: '%.2f' % s)
            except AttributeError:
                data = data.applymap(lambda s: '%.2f' % s)
            k_idio = 1
            if mod.idiosyncratic_ar1:
                data['   idiosyncratic: AR(1)'] = self.params[mod._p['idiosyncratic_ar1']]
                k_idio += 1
            data['var.'] = self.params[mod._p['idiosyncratic_var']]
            try:
                data.iloc[:, -k_idio:] = data.iloc[:, -k_idio:].map(lambda s: f'{s:.2f}')
            except AttributeError:
                data.iloc[:, -k_idio:] = data.iloc[:, -k_idio:].applymap(lambda s: f'{s:.2f}')
            data.index.name = 'Factor loadings:'
            base_iloc = np.arange(mod.k_factors)
            for i in range(mod.k_endog):
                iloc = [j for j in base_iloc if j not in mod._s.endog_factor_iloc[i]]
                data.iloc[i, iloc] = '.'
            data = data.reset_index()
            params_data = data.values
            params_header = data.columns.tolist()
            params_stubs = None
            title = 'Observation equation:'
            table = SimpleTable(params_data, params_header, params_stubs, txt_fmt=fmt_params, title=title)
            summary.tables.insert(table_ix, table)
            table_ix += 1
            ix1 = 0
            ix2 = 0
            for i in range(len(mod._s.factor_blocks)):
                block = mod._s.factor_blocks[i]
                ix2 += block.k_factors
                T = self.filter_results.transition
                lag_names = []
                for j in range(block.factor_order):
                    lag_names += [f'L{j + 1}.{name}' for name in block.factor_names]
                data = pd.DataFrame(T[block.factors_L1, block.factors_ar, 0], index=block.factor_names, columns=lag_names)
                data.index.name = ''
                try:
                    data = data.map(lambda s: '%.2f' % s)
                except AttributeError:
                    data = data.applymap(lambda s: '%.2f' % s)
                Q = self.filter_results.state_cov
                if block.k_factors == 1:
                    data['   error variance'] = Q[ix1, ix1]
                else:
                    data['   error covariance'] = block.factor_names
                    for j in range(block.k_factors):
                        data[block.factor_names[j]] = Q[ix1:ix2, ix1 + j]
                try:
                    formatted_vals = data.iloc[:, -block.k_factors:].map(lambda s: f'{s:.2f}')
                except AttributeError:
                    formatted_vals = data.iloc[:, -block.k_factors:].applymap(lambda s: f'{s:.2f}')
                data.iloc[:, -block.k_factors:] = formatted_vals
                data = data.reset_index()
                params_data = data.values
                params_header = data.columns.tolist()
                params_stubs = None
                title = f'Transition: Factor block {i}'
                table = SimpleTable(params_data, params_header, params_stubs, txt_fmt=fmt_params, title=title)
                summary.tables.insert(table_ix, table)
                table_ix += 1
                ix1 = ix2
        return summary