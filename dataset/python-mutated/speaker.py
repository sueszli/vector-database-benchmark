from encoder.data_objects.random_cycler import RandomCycler
from encoder.data_objects.utterance import Utterance
from pathlib import Path

class Speaker:

    def __init__(self, root: Path):
        if False:
            i = 10
            return i + 15
        self.root = root
        self.name = root.name
        self.utterances = None
        self.utterance_cycler = None

    def _load_utterances(self):
        if False:
            return 10
        with self.root.joinpath('_sources.txt').open('r') as sources_file:
            sources = [l.split(',') for l in sources_file]
        sources = {frames_fname: wave_fpath for (frames_fname, wave_fpath) in sources}
        self.utterances = [Utterance(self.root.joinpath(f), w) for (f, w) in sources.items()]
        self.utterance_cycler = RandomCycler(self.utterances)

    def random_partial(self, count, n_frames):
        if False:
            for i in range(10):
                print('nop')
        '\n        Samples a batch of <count> unique partial utterances from the disk in a way that all \n        utterances come up at least once every two cycles and in a random order every time.\n        \n        :param count: The number of partial utterances to sample from the set of utterances from \n        that speaker. Utterances are guaranteed not to be repeated if <count> is not larger than \n        the number of utterances available.\n        :param n_frames: The number of frames in the partial utterance.\n        :return: A list of tuples (utterance, frames, range) where utterance is an Utterance, \n        frames are the frames of the partial utterances and range is the range of the partial \n        utterance with regard to the complete utterance.\n        '
        if self.utterances is None:
            self._load_utterances()
        utterances = self.utterance_cycler.sample(count)
        a = [(u,) + u.random_partial(n_frames) for u in utterances]
        return a