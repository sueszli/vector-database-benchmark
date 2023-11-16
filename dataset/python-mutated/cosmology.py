"""|Cosmology| I/O, using |Cosmology.to_format| and |Cosmology.from_format|.

This module provides functions to transform a |Cosmology| object to and from another
|Cosmology| object. The functions are registered with ``convert_registry`` under the
format name "astropy.cosmology". You probably won't need to use these functions as they
are present mainly for completeness and testing.

    >>> from astropy.cosmology import Cosmology, Planck18
    >>> Planck18.to_format("astropy.cosmology") is Planck18
    True
    >>> Cosmology.from_format(Planck18) is Planck18
    True
"""
from astropy.cosmology.connect import convert_registry
from astropy.cosmology.core import _COSMOLOGY_CLASSES, Cosmology
__all__ = []

def from_cosmology(cosmo, /, **kwargs):
    if False:
        return 10
    'Return the |Cosmology| unchanged.\n\n    Parameters\n    ----------\n    cosmo : `~astropy.cosmology.Cosmology`\n        The cosmology to return.\n    **kwargs\n        This argument is required for compatibility with the standard set of\n        keyword arguments in format `~astropy.cosmology.Cosmology.from_format`,\n        e.g. "cosmology". If "cosmology" is included and is not `None`,\n        ``cosmo`` is checked for correctness.\n\n    Returns\n    -------\n    `~astropy.cosmology.Cosmology` subclass instance\n        Just ``cosmo`` passed through.\n\n    Raises\n    ------\n    TypeError\n        If the |Cosmology| object is not an instance of ``cosmo`` (and\n        ``cosmology`` is not `None`).\n\n    Examples\n    --------\n    >>> from astropy.cosmology import Cosmology, Planck18\n    >>> print(Cosmology.from_format(Planck18))\n    FlatLambdaCDM(name="Planck18", H0=67.66 km / (Mpc s), Om0=0.30966,\n                  Tcmb0=2.7255 K, Neff=3.046, m_nu=[0. 0. 0.06] eV, Ob0=0.04897)\n    '
    cosmology = kwargs.get('cosmology')
    if isinstance(cosmology, str):
        cosmology = _COSMOLOGY_CLASSES[cosmology]
    if cosmology is not None and (not isinstance(cosmo, cosmology)):
        raise TypeError(f'cosmology {cosmo} is not an {cosmology} instance.')
    return cosmo

def to_cosmology(cosmo, *args):
    if False:
        while True:
            i = 10
    'Return the |Cosmology| unchanged.\n\n    Parameters\n    ----------\n    cosmo : `~astropy.cosmology.Cosmology`\n        The cosmology to return.\n    *args\n        Not used.\n\n    Returns\n    -------\n    `~astropy.cosmology.Cosmology` subclass instance\n        Just ``cosmo`` passed through.\n\n    Examples\n    --------\n    >>> from astropy.cosmology import Planck18\n    >>> Planck18.to_format("astropy.cosmology") is Planck18\n    True\n    '
    return cosmo

def cosmology_identify(origin, format, *args, **kwargs):
    if False:
        print('Hello World!')
    'Identify if object is a `~astropy.cosmology.Cosmology`.\n\n    Returns\n    -------\n    bool\n    '
    itis = False
    if origin == 'read':
        itis = isinstance(args[1], Cosmology) and format in (None, 'astropy.cosmology')
    return itis
convert_registry.register_reader('astropy.cosmology', Cosmology, from_cosmology)
convert_registry.register_writer('astropy.cosmology', Cosmology, to_cosmology)
convert_registry.register_identifier('astropy.cosmology', Cosmology, cosmology_identify)