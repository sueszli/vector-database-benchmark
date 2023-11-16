import numpy as np

class Utterance:

    def __init__(self, frames_fpath, wave_fpath):
        if False:
            return 10
        self.frames_fpath = frames_fpath
        self.wave_fpath = wave_fpath

    def get_frames(self):
        if False:
            for i in range(10):
                print('nop')
        return np.load(self.frames_fpath)

    def random_partial(self, n_frames):
        if False:
            while True:
                i = 10
        '\n        Crops the frames into a partial utterance of n_frames\n        \n        :param n_frames: The number of frames of the partial utterance\n        :return: the partial utterance frames and a tuple indicating the start and end of the \n        partial utterance in the complete utterance.\n        '
        frames = self.get_frames()
        if frames.shape[0] == n_frames:
            start = 0
        else:
            start = np.random.randint(0, frames.shape[0] - n_frames)
        end = start + n_frames
        return (frames[start:end], (start, end))