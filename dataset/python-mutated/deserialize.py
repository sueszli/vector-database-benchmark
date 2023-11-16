import numpy as np
from typing import List, Tuple, Union, Optional
from deeplake.core.chunk.base_chunk import BaseChunk
from deeplake.core.meta.encode.tile import TileEncoder

def coalesce_tiles(tiles: np.ndarray, tile_shape: Tuple[int, ...], sample_shape: Optional[Tuple[int, ...]], dtype: Optional[Union[str, np.dtype]]=None) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Coalesce tiles into a single array of shape `sample_shape`.\n    Args:\n        tiles (np.ndarray): numpy object array of tiles.\n        tile_shape (Tuple[int, ...]): Tile shape. Corner tiles may be smaller than this.\n        sample_shape (Optional, Tuple[int, ...]): Shape of the output array. The sum of all actual tile shapes are expected to be equal to this.\n        dtype (Optional, Union[str, np.dtype]): Dtype of the output array. Should match dtype of tiles.\n    Raises:\n        TypeError: If `tiles` is not deserialized.\n    Returns:\n        np.ndarray: Sample array from tiles.\n    '
    if dtype is None:
        dtype = next(iter(tiles.flat)).dtype
    ndim = tiles.ndim
    sample_shape = sample_shape or tuple((sum((tile.shape[i] for tile in tiles[tuple((slice(None) if j == i else 0 for j in range(ndim)))])) for i in range(ndim)))
    sample = np.empty(sample_shape, dtype=dtype)
    if tiles.size <= 0:
        return sample
    for (tile_coords, tile) in np.ndenumerate(tiles):
        low = np.multiply(tile_coords, tile_shape)
        high = low + tile.shape
        idx = tuple((slice(l, h) for (l, h) in zip(low, high)))
        view = sample[idx]
        view[:] = tile
    return sample

def combine_chunks(chunks: List[BaseChunk], sample_index: int, tile_encoder: TileEncoder) -> np.ndarray:
    if False:
        print('Hello World!')
    dtype = chunks[0].dtype
    shape = tile_encoder.get_sample_shape(sample_index)
    tile_shape = tile_encoder.get_tile_shape(sample_index)
    layout_shape = tile_encoder.get_tile_layout_shape(sample_index)
    tiled_arrays = [chunk.read_sample(0, is_tile=True) for chunk in chunks]
    return np_list_to_sample(tiled_arrays, shape, tile_shape, layout_shape, dtype)

def np_list_to_sample(tiled_arrays: List[np.ndarray], shape, tile_shape, layout_shape, dtype=None) -> np.ndarray:
    if False:
        return 10
    num_tiles = len(tiled_arrays)
    tiles = np.empty((num_tiles,), dtype=object)
    tiles[:] = tiled_arrays[:]
    tiles = np.reshape(tiles, layout_shape)
    return coalesce_tiles(tiles, tile_shape, shape, dtype)

def translate_slices(slices: List[Union[slice, int, List[int]]], sample_shape: Tuple[int, ...], tile_shape: Tuple[int, ...]) -> Tuple[Tuple, Tuple]:
    if False:
        while True:
            i = 10
    'Translates slices from sample space to tile space\n    Args:\n        sample_shape (Tuple[int, ...]): Sample shape.\n        tile_shape (Tuple[int, ...]): Tile shape.\n    Raises:\n        NotImplementedError: For stepping slices\n    '
    tiles_index: List[slice] = []
    sample_index: List[Union[int, slice, List[int]]] = []
    for (i, s) in enumerate(slices):
        if isinstance(s, int):
            if s < 0:
                s += sample_shape[i]
            ts = s // tile_shape[i]
            tiles_index.append(slice(ts, ts + 1))
            sample_index.append(s % tile_shape[i])
        elif isinstance(s, list):
            s = [x + sample_shape[i] if x < 0 else x for x in s]
            (mn, mx) = (min(s), max(s))
            if s != list(range(mn, mx + 1)):
                raise NotImplementedError('Non-contiguous indexing for tiled samples is not supported yet.')
            tiles_index.append(slice(mn // tile_shape[i], mx // tile_shape[i] + 1))
            offset = mn - mn % tile_shape[i]
            sample_index.append([x - offset for x in s])
        elif isinstance(s, slice):
            (start, stop, step) = (s.start, s.stop, s.step)
            if start is None:
                start = 0
            elif start < 0:
                start += sample_shape[i]
            if stop is None:
                stop = sample_shape[i]
            elif stop < 0:
                stop += sample_shape[i]
            else:
                stop = min(stop, sample_shape[i])
            if step not in (1, None):
                raise NotImplementedError('Stepped indexing for tiled samples is not supported yet.')
            ts = slice(start // tile_shape[i], (stop - 1) // tile_shape[i] + 1)
            tiles_index.append(ts)
            offset = ts.start * tile_shape[i]
            sample_index.append(slice(start - offset, stop - offset))
    return (tuple(tiles_index), tuple(sample_index))