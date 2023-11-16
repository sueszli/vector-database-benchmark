class Vocabulary:

    def __init__(self, counter, min_freq=0, reserved_tokens=None):
        if False:
            while True:
                i = 10
        if reserved_tokens is None:
            reserved_tokens = []
        self.token_freqs = sorted(counter.items(), key=lambda x: x[0])
        self.token_freqs.sort(key=lambda x: x[1], reverse=True)
        (self.unk, uniq_tokens) = (0, ['<unk>'] + reserved_tokens)
        uniq_tokens += [token for (token, freq) in self.token_freqs if freq >= min_freq and token not in uniq_tokens]
        (self.idx_to_token, self.token_to_idx) = ([], dict())
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.idx_to_token)

    def to_indices(self, tokens):
        if False:
            return 10
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.to_indices(token) for token in tokens]