"""Cutting utilities working with audio."""
import numpy as np

def find_audio_period(clip, min_time=0.1, max_time=2, time_resolution=0.01):
    if False:
        return 10
    'Finds the period, in seconds of an audioclip.\n\n    Parameters\n    ----------\n\n    min_time : float, optional\n      Minimum bound for the returned value.\n\n    max_time : float, optional\n      Maximum bound for the returned value.\n\n    time_resolution : float, optional\n      Numerical precision.\n    '
    chunksize = int(time_resolution * clip.fps)
    chunk_duration = 1.0 * chunksize / clip.fps
    v = np.array([(chunk ** 2).sum() for chunk in clip.iter_chunks(chunksize)])
    v = v - v.mean()
    corrs = np.correlate(v, v, mode='full')[-len(v):]
    corrs[:int(min_time / chunk_duration)] = 0
    corrs[int(max_time / chunk_duration):] = 0
    return chunk_duration * np.argmax(corrs)