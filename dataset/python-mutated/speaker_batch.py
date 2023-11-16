import numpy as np
from typing import List
from encoder.data_objects.speaker import Speaker

class SpeakerBatch:

    def __init__(self, speakers: List[Speaker], utterances_per_speaker: int, n_frames: int):
        if False:
            return 10
        self.speakers = speakers
        self.partials = {s: s.random_partial(utterances_per_speaker, n_frames) for s in speakers}
        self.data = np.array([frames for s in speakers for (_, frames, _) in self.partials[s]])