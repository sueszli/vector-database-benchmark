"""Built-in cosmologies.

See :attr:`~astropy.cosmology.realizations.available` for a full list.
"""
from __future__ import annotations
__all__ = ['available', 'default_cosmology', 'WMAP1', 'WMAP3', 'WMAP5', 'WMAP7', 'WMAP9', 'Planck13', 'Planck15', 'Planck18']
import pathlib
import sys
from astropy.utils.data import get_pkg_data_path
from astropy.utils.state import ScienceState
from . import _io
from .core import Cosmology
__doctest_requires__ = {'*': ['scipy']}
_COSMOLOGY_DATA_DIR = pathlib.Path(get_pkg_data_path('cosmology', 'data', package='astropy'))
available = ('WMAP1', 'WMAP3', 'WMAP5', 'WMAP7', 'WMAP9', 'Planck13', 'Planck15', 'Planck18')

def __getattr__(name):
    if False:
        while True:
            i = 10
    'Make specific realizations from data files with lazy import from ``PEP 562``.\n\n    Raises\n    ------\n    AttributeError\n        If "name" is not in :mod:`astropy.cosmology.realizations`\n    '
    if name not in available:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}.')
    cosmo = Cosmology.read(str(_COSMOLOGY_DATA_DIR / name) + '.ecsv', format='ascii.ecsv')
    cosmo.__doc__ = f"{name} instance of {cosmo.__class__.__qualname__} cosmology\n(from {cosmo.meta['reference']})"
    setattr(sys.modules[__name__], name, cosmo)
    return cosmo

def __dir__():
    if False:
        while True:
            i = 10
    'Directory, including lazily-imported objects.'
    return __all__

class default_cosmology(ScienceState):
    """The default cosmology to use.

    To change it::

        >>> from astropy.cosmology import default_cosmology, WMAP7
        >>> with default_cosmology.set(WMAP7):
        ...     # WMAP7 cosmology in effect
        ...     pass

    Or, you may use a string::

        >>> with default_cosmology.set('WMAP7'):
        ...     # WMAP7 cosmology in effect
        ...     pass

    To get the default cosmology:

        >>> default_cosmology.get()
        FlatLambdaCDM(name='Planck18', H0=<Quantity 67.66 km / (Mpc s)>,
                      Om0=0.30966, ...
    """
    _default_value = 'Planck18'
    _value = 'Planck18'

    @classmethod
    def validate(cls, value: Cosmology | str | None) -> Cosmology | None:
        if False:
            print('Hello World!')
        'Return a Cosmology given a value.\n\n        Parameters\n        ----------\n        value : None, str, or `~astropy.cosmology.Cosmology`\n\n        Returns\n        -------\n        `~astropy.cosmology.Cosmology` instance\n\n        Raises\n        ------\n        TypeError\n            If ``value`` is not a string or |Cosmology|.\n        '
        if value is None:
            value = cls._default_value
        if isinstance(value, str):
            if value == 'no_default':
                value = None
            else:
                value = cls._get_from_registry(value)
        elif not isinstance(value, Cosmology):
            raise TypeError(f'default_cosmology must be a string or Cosmology instance, not {value}.')
        return value

    @classmethod
    def _get_from_registry(cls, name: str) -> Cosmology:
        if False:
            for i in range(10):
                print('nop')
        'Get a registered Cosmology realization.\n\n        Parameters\n        ----------\n        name : str\n            The built-in |Cosmology| realization to retrieve.\n\n        Returns\n        -------\n        `astropy.cosmology.Cosmology`\n            The cosmology realization of `name`.\n\n        Raises\n        ------\n        ValueError\n            If ``name`` is a str, but not for a built-in Cosmology.\n        TypeError\n            If ``name`` is for a non-Cosmology object.\n        '
        try:
            value = getattr(sys.modules[__name__], name)
        except AttributeError:
            raise ValueError(f'Unknown cosmology {name!r}. Valid cosmologies:\n{available}')
        if not isinstance(value, Cosmology):
            raise TypeError(f'cannot find a Cosmology realization called {name}.')
        return value