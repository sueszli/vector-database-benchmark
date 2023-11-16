"""
==========================
Efficient patch generation
==========================

This notebook demonstrates how to efficiently generate fixed-duration
excerpts of a signal using `librosa.util.frame`.
This can be helpful in machine learning applications where a model may
expect inputs of a certain size during training, but your data may be
of arbitrary and variable length.

Aside from being a convenient helper method for patch sampling, the
`librosa.util.frame` function can do this *efficiently* by avoiding
memory copies.
The patch array produced below is a *view* of the original data array,
not a copy.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import librosa
(y, sr) = librosa.load(librosa.ex('libri1'))
melspec = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr), ref=np.max)
print(f'Mel spectrogram shape: {melspec.shape}')
(fig, ax) = plt.subplots()
librosa.display.specshow(melspec, x_axis='time', y_axis='mel', ax=ax)
ax.set(title='Full Mel spectrogram')
frame_length = librosa.time_to_frames(5.0)
hop_length = librosa.time_to_frames(0.1)
print(f'Frame length={frame_length}, hop length={hop_length}')
patches = librosa.util.frame(melspec, frame_length=frame_length, hop_length=hop_length)
print(f'Patch array shape: {patches.shape}')
(fig, ax) = plt.subplot_mosaic([list('AAA'), list('012')])
librosa.display.specshow(melspec, x_axis='time', y_axis='mel', ax=ax['A'])
ax['A'].set(title='Full spectrogram', xlabel=None)
for index in [0, 1, 2]:
    librosa.display.specshow(patches[..., index], x_axis='time', y_axis='mel', ax=ax[str(index)])
    ax[str(index)].set(title=f'Patch #{index}')
    ax[str(index)].label_outer()
(fig, ax) = plt.subplots()
mesh = librosa.display.specshow(patches[..., 0], x_axis='time', y_axis='mel', ax=ax)

def _update(num):
    if False:
        print('Hello World!')
    mesh.set_array(patches[..., num])
    return (mesh,)
ani = animation.FuncAnimation(fig, func=_update, frames=patches.shape[-1], interval=100, blit=True)