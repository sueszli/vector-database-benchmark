from abc import ABCMeta, abstractmethod
__all__ = ['NDDataBase']

class NDDataBase(metaclass=ABCMeta):
    """Base metaclass that defines the interface for N-dimensional datasets
    with associated meta information used in ``astropy``.

    All properties and ``__init__`` have to be overridden in subclasses. See
    `NDData` for a subclass that defines this interface on `numpy.ndarray`-like
    ``data``.

    See also: https://docs.astropy.org/en/stable/nddata/

    """

    @abstractmethod
    def __init__(self):
        if False:
            i = 10
            return i + 15
        pass

    @property
    @abstractmethod
    def data(self):
        if False:
            return 10
        'The stored dataset.'

    @property
    @abstractmethod
    def mask(self):
        if False:
            i = 10
            return i + 15
        'Mask for the dataset.\n\n        Masks should follow the ``numpy`` convention that **valid** data points\n        are marked by ``False`` and **invalid** ones with ``True``.\n        '
        return None

    @property
    @abstractmethod
    def unit(self):
        if False:
            return 10
        'Unit for the dataset.'
        return None

    @property
    @abstractmethod
    def wcs(self):
        if False:
            print('Hello World!')
        'World coordinate system (WCS) for the dataset.'
        return None

    @property
    def psf(self):
        if False:
            i = 10
            return i + 15
        'Image representation of the PSF for the dataset.\n\n        Should be `ndarray`-like.\n        '
        return None

    @property
    @abstractmethod
    def meta(self):
        if False:
            print('Hello World!')
        'Additional meta information about the dataset.\n\n        Should be `dict`-like.\n        '
        return None

    @property
    @abstractmethod
    def uncertainty(self):
        if False:
            return 10
        'Uncertainty in the dataset.\n\n        Should have an attribute ``uncertainty_type`` that defines what kind of\n        uncertainty is stored, such as ``"std"`` for standard deviation or\n        ``"var"`` for variance.\n        '
        return None