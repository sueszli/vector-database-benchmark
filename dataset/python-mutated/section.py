import numpy as np
from astropy.io.fits.hdu.base import BITPIX2DTYPE
from astropy.io.fits.hdu.compressed._tiled_compression import decompress_image_data_section
from astropy.io.fits.hdu.compressed.utils import _data_shape, _n_tiles, _tile_shape
from astropy.utils.shapes import simplify_basic_index

class CompImageSection:
    """
    Class enabling subsets of CompImageHDU data to be loaded lazily via slicing.

    Slices of this object load the corresponding section of an image array from
    the underlying FITS file, and applies any BSCALE/BZERO factors.

    Section slices cannot be assigned to, and modifications to a section are
    not saved back to the underlying file.

    See the :ref:`astropy:data-sections` section of the Astropy documentation
    for more details.
    """

    def __init__(self, hdu):
        if False:
            i = 10
            return i + 15
        self.hdu = hdu
        self._data_shape = _data_shape(self.hdu._header)
        self._tile_shape = _tile_shape(self.hdu._header)
        self._n_dim = len(self._data_shape)
        self._n_tiles = np.array(_n_tiles(self._data_shape, self._tile_shape), dtype=int)

    @property
    def shape(self):
        if False:
            while True:
                i = 10
        return tuple(self._data_shape)

    @property
    def ndim(self):
        if False:
            for i in range(10):
                print('nop')
        return self.hdu._header['ZNAXIS']

    @property
    def dtype(self):
        if False:
            i = 10
            return i + 15
        return BITPIX2DTYPE[self.hdu._header['ZBITPIX']]

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        if index is Ellipsis:
            first_tile_index = np.zeros(self._n_dim, dtype=int)
            last_tile_index = self._n_tiles - 1
            data = decompress_image_data_section(self.hdu.compressed_data, self.hdu.compression_type, self.hdu._header, self.hdu, first_tile_index, last_tile_index)
            return self.hdu._scale_data(data)
        index = simplify_basic_index(index, shape=self._data_shape)
        first_tile_index = np.zeros(self._n_dim, dtype=int)
        last_tile_index = np.zeros(self._n_dim, dtype=int)
        final_array_index = []
        for (dim, idx) in enumerate(index):
            if isinstance(idx, slice):
                if idx.step > 0:
                    first_tile_index[dim] = idx.start // self._tile_shape[dim]
                    last_tile_index[dim] = (idx.stop - 1) // self._tile_shape[dim]
                else:
                    stop = 0 if idx.stop is None else max(idx.stop - 1, 0)
                    first_tile_index[dim] = stop // self._tile_shape[dim]
                    last_tile_index[dim] = idx.start // self._tile_shape[dim]
                last_tile_index = np.maximum(last_tile_index, first_tile_index)
                if idx.step < 0 and idx.stop is None:
                    final_array_index.append(idx)
                else:
                    final_array_index.append(slice(idx.start - self._tile_shape[dim] * first_tile_index[dim], idx.stop - self._tile_shape[dim] * first_tile_index[dim], idx.step))
            else:
                first_tile_index[dim] = idx // self._tile_shape[dim]
                last_tile_index[dim] = first_tile_index[dim]
                final_array_index.append(idx - self._tile_shape[dim] * first_tile_index[dim])
        data = decompress_image_data_section(self.hdu.compressed_data, self.hdu.compression_type, self.hdu._header, self.hdu, first_tile_index, last_tile_index)
        return self.hdu._scale_data(data[tuple(final_array_index)])