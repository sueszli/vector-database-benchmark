import math
import re
try:
    import numpy
except ImportError:
    pass
from nltk.tokenize.api import TokenizerI
(BLOCK_COMPARISON, VOCABULARY_INTRODUCTION) = (0, 1)
(LC, HC) = (0, 1)
DEFAULT_SMOOTHING = [0]

class TextTilingTokenizer(TokenizerI):
    """Tokenize a document into topical sections using the TextTiling algorithm.
    This algorithm detects subtopic shifts based on the analysis of lexical
    co-occurrence patterns.

    The process starts by tokenizing the text into pseudosentences of
    a fixed size w. Then, depending on the method used, similarity
    scores are assigned at sentence gaps. The algorithm proceeds by
    detecting the peak differences between these scores and marking
    them as boundaries. The boundaries are normalized to the closest
    paragraph break and the segmented text is returned.

    :param w: Pseudosentence size
    :type w: int
    :param k: Size (in sentences) of the block used in the block comparison method
    :type k: int
    :param similarity_method: The method used for determining similarity scores:
       `BLOCK_COMPARISON` (default) or `VOCABULARY_INTRODUCTION`.
    :type similarity_method: constant
    :param stopwords: A list of stopwords that are filtered out (defaults to NLTK's stopwords corpus)
    :type stopwords: list(str)
    :param smoothing_method: The method used for smoothing the score plot:
      `DEFAULT_SMOOTHING` (default)
    :type smoothing_method: constant
    :param smoothing_width: The width of the window used by the smoothing method
    :type smoothing_width: int
    :param smoothing_rounds: The number of smoothing passes
    :type smoothing_rounds: int
    :param cutoff_policy: The policy used to determine the number of boundaries:
      `HC` (default) or `LC`
    :type cutoff_policy: constant

    >>> from nltk.corpus import brown
    >>> tt = TextTilingTokenizer(demo_mode=True)
    >>> text = brown.raw()[:4000]
    >>> s, ss, d, b = tt.tokenize(text)
    >>> b
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    """

    def __init__(self, w=20, k=10, similarity_method=BLOCK_COMPARISON, stopwords=None, smoothing_method=DEFAULT_SMOOTHING, smoothing_width=2, smoothing_rounds=1, cutoff_policy=HC, demo_mode=False):
        if False:
            for i in range(10):
                print('nop')
        if stopwords is None:
            from nltk.corpus import stopwords
            stopwords = stopwords.words('english')
        self.__dict__.update(locals())
        del self.__dict__['self']

    def tokenize(self, text):
        if False:
            i = 10
            return i + 15
        'Return a tokenized copy of *text*, where each "token" represents\n        a separate topic.'
        lowercase_text = text.lower()
        paragraph_breaks = self._mark_paragraph_breaks(text)
        text_length = len(lowercase_text)
        nopunct_text = ''.join((c for c in lowercase_text if re.match("[a-z\\-' \\n\\t]", c)))
        nopunct_par_breaks = self._mark_paragraph_breaks(nopunct_text)
        tokseqs = self._divide_to_tokensequences(nopunct_text)
        for ts in tokseqs:
            ts.wrdindex_list = [wi for wi in ts.wrdindex_list if wi[0] not in self.stopwords]
        token_table = self._create_token_table(tokseqs, nopunct_par_breaks)
        if self.similarity_method == BLOCK_COMPARISON:
            gap_scores = self._block_comparison(tokseqs, token_table)
        elif self.similarity_method == VOCABULARY_INTRODUCTION:
            raise NotImplementedError('Vocabulary introduction not implemented')
        else:
            raise ValueError(f'Similarity method {self.similarity_method} not recognized')
        if self.smoothing_method == DEFAULT_SMOOTHING:
            smooth_scores = self._smooth_scores(gap_scores)
        else:
            raise ValueError(f'Smoothing method {self.smoothing_method} not recognized')
        depth_scores = self._depth_scores(smooth_scores)
        segment_boundaries = self._identify_boundaries(depth_scores)
        normalized_boundaries = self._normalize_boundaries(text, segment_boundaries, paragraph_breaks)
        segmented_text = []
        prevb = 0
        for b in normalized_boundaries:
            if b == 0:
                continue
            segmented_text.append(text[prevb:b])
            prevb = b
        if prevb < text_length:
            segmented_text.append(text[prevb:])
        if not segmented_text:
            segmented_text = [text]
        if self.demo_mode:
            return (gap_scores, smooth_scores, depth_scores, segment_boundaries)
        return segmented_text

    def _block_comparison(self, tokseqs, token_table):
        if False:
            return 10
        'Implements the block comparison method'

        def blk_frq(tok, block):
            if False:
                for i in range(10):
                    print('nop')
            ts_occs = filter(lambda o: o[0] in block, token_table[tok].ts_occurences)
            freq = sum((tsocc[1] for tsocc in ts_occs))
            return freq
        gap_scores = []
        numgaps = len(tokseqs) - 1
        for curr_gap in range(numgaps):
            (score_dividend, score_divisor_b1, score_divisor_b2) = (0.0, 0.0, 0.0)
            score = 0.0
            if curr_gap < self.k - 1:
                window_size = curr_gap + 1
            elif curr_gap > numgaps - self.k:
                window_size = numgaps - curr_gap
            else:
                window_size = self.k
            b1 = [ts.index for ts in tokseqs[curr_gap - window_size + 1:curr_gap + 1]]
            b2 = [ts.index for ts in tokseqs[curr_gap + 1:curr_gap + window_size + 1]]
            for t in token_table:
                score_dividend += blk_frq(t, b1) * blk_frq(t, b2)
                score_divisor_b1 += blk_frq(t, b1) ** 2
                score_divisor_b2 += blk_frq(t, b2) ** 2
            try:
                score = score_dividend / math.sqrt(score_divisor_b1 * score_divisor_b2)
            except ZeroDivisionError:
                pass
            gap_scores.append(score)
        return gap_scores

    def _smooth_scores(self, gap_scores):
        if False:
            for i in range(10):
                print('nop')
        'Wraps the smooth function from the SciPy Cookbook'
        return list(smooth(numpy.array(gap_scores[:]), window_len=self.smoothing_width + 1))

    def _mark_paragraph_breaks(self, text):
        if False:
            for i in range(10):
                print('nop')
        'Identifies indented text or line breaks as the beginning of\n        paragraphs'
        MIN_PARAGRAPH = 100
        pattern = re.compile('[ \t\r\x0c\x0b]*\n[ \t\r\x0c\x0b]*\n[ \t\r\x0c\x0b]*')
        matches = pattern.finditer(text)
        last_break = 0
        pbreaks = [0]
        for pb in matches:
            if pb.start() - last_break < MIN_PARAGRAPH:
                continue
            else:
                pbreaks.append(pb.start())
                last_break = pb.start()
        return pbreaks

    def _divide_to_tokensequences(self, text):
        if False:
            print('Hello World!')
        'Divides the text into pseudosentences of fixed size'
        w = self.w
        wrdindex_list = []
        matches = re.finditer('\\w+', text)
        for match in matches:
            wrdindex_list.append((match.group(), match.start()))
        return [TokenSequence(i / w, wrdindex_list[i:i + w]) for i in range(0, len(wrdindex_list), w)]

    def _create_token_table(self, token_sequences, par_breaks):
        if False:
            return 10
        'Creates a table of TokenTableFields'
        token_table = {}
        current_par = 0
        current_tok_seq = 0
        pb_iter = par_breaks.__iter__()
        current_par_break = next(pb_iter)
        if current_par_break == 0:
            try:
                current_par_break = next(pb_iter)
            except StopIteration as e:
                raise ValueError('No paragraph breaks were found(text too short perhaps?)') from e
        for ts in token_sequences:
            for (word, index) in ts.wrdindex_list:
                try:
                    while index > current_par_break:
                        current_par_break = next(pb_iter)
                        current_par += 1
                except StopIteration:
                    pass
                if word in token_table:
                    token_table[word].total_count += 1
                    if token_table[word].last_par != current_par:
                        token_table[word].last_par = current_par
                        token_table[word].par_count += 1
                    if token_table[word].last_tok_seq != current_tok_seq:
                        token_table[word].last_tok_seq = current_tok_seq
                        token_table[word].ts_occurences.append([current_tok_seq, 1])
                    else:
                        token_table[word].ts_occurences[-1][1] += 1
                else:
                    token_table[word] = TokenTableField(first_pos=index, ts_occurences=[[current_tok_seq, 1]], total_count=1, par_count=1, last_par=current_par, last_tok_seq=current_tok_seq)
            current_tok_seq += 1
        return token_table

    def _identify_boundaries(self, depth_scores):
        if False:
            print('Hello World!')
        'Identifies boundaries at the peaks of similarity score\n        differences'
        boundaries = [0 for x in depth_scores]
        avg = sum(depth_scores) / len(depth_scores)
        stdev = numpy.std(depth_scores)
        if self.cutoff_policy == LC:
            cutoff = avg - stdev
        else:
            cutoff = avg - stdev / 2.0
        depth_tuples = sorted(zip(depth_scores, range(len(depth_scores))))
        depth_tuples.reverse()
        hp = list(filter(lambda x: x[0] > cutoff, depth_tuples))
        for dt in hp:
            boundaries[dt[1]] = 1
            for dt2 in hp:
                if dt[1] != dt2[1] and abs(dt2[1] - dt[1]) < 4 and (boundaries[dt2[1]] == 1):
                    boundaries[dt[1]] = 0
        return boundaries

    def _depth_scores(self, scores):
        if False:
            return 10
        "Calculates the depth of each gap, i.e. the average difference\n        between the left and right peaks and the gap's score"
        depth_scores = [0 for x in scores]
        clip = min(max(len(scores) // 10, 2), 5)
        index = clip
        for gapscore in scores[clip:-clip]:
            lpeak = gapscore
            for score in scores[index::-1]:
                if score >= lpeak:
                    lpeak = score
                else:
                    break
            rpeak = gapscore
            for score in scores[index:]:
                if score >= rpeak:
                    rpeak = score
                else:
                    break
            depth_scores[index] = lpeak + rpeak - 2 * gapscore
            index += 1
        return depth_scores

    def _normalize_boundaries(self, text, boundaries, paragraph_breaks):
        if False:
            i = 10
            return i + 15
        "Normalize the boundaries identified to the original text's\n        paragraph breaks"
        norm_boundaries = []
        (char_count, word_count, gaps_seen) = (0, 0, 0)
        seen_word = False
        for char in text:
            char_count += 1
            if char in ' \t\n' and seen_word:
                seen_word = False
                word_count += 1
            if char not in ' \t\n' and (not seen_word):
                seen_word = True
            if gaps_seen < len(boundaries) and word_count > max(gaps_seen * self.w, self.w):
                if boundaries[gaps_seen] == 1:
                    best_fit = len(text)
                    for br in paragraph_breaks:
                        if best_fit > abs(br - char_count):
                            best_fit = abs(br - char_count)
                            bestbr = br
                        else:
                            break
                    if bestbr not in norm_boundaries:
                        norm_boundaries.append(bestbr)
                gaps_seen += 1
        return norm_boundaries

class TokenTableField:
    """A field in the token table holding parameters for each token,
    used later in the process"""

    def __init__(self, first_pos, ts_occurences, total_count=1, par_count=1, last_par=0, last_tok_seq=None):
        if False:
            i = 10
            return i + 15
        self.__dict__.update(locals())
        del self.__dict__['self']

class TokenSequence:
    """A token list with its original length and its index"""

    def __init__(self, index, wrdindex_list, original_length=None):
        if False:
            return 10
        original_length = original_length or len(wrdindex_list)
        self.__dict__.update(locals())
        del self.__dict__['self']

def smooth(x, window_len=11, window='flat'):
    if False:
        i = 10
        return i + 15
    "smooth the data using a window with requested size.\n\n    This method is based on the convolution of a scaled window with the signal.\n    The signal is prepared by introducing reflected copies of the signal\n    (with the window size) in both ends so that transient parts are minimized\n    in the beginning and end part of the output signal.\n\n    :param x: the input signal\n    :param window_len: the dimension of the smoothing window; should be an odd integer\n    :param window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'\n        flat window will produce a moving average smoothing.\n\n    :return: the smoothed signal\n\n    example::\n\n        t=linspace(-2,2,0.1)\n        x=sin(t)+randn(len(t))*0.1\n        y=smooth(x)\n\n    :see also: numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve,\n        scipy.signal.lfilter\n\n    TODO: the window parameter could be the window itself if an array instead of a string\n    "
    if x.ndim != 1:
        raise ValueError('smooth only accepts 1 dimension arrays.')
    if x.size < window_len:
        raise ValueError('Input vector needs to be bigger than window size.')
    if window_len < 3:
        return x
    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = numpy.r_[2 * x[0] - x[window_len:1:-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    if window == 'flat':
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')
    y = numpy.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]

def demo(text=None):
    if False:
        for i in range(10):
            print('nop')
    from matplotlib import pylab
    from nltk.corpus import brown
    tt = TextTilingTokenizer(demo_mode=True)
    if text is None:
        text = brown.raw()[:10000]
    (s, ss, d, b) = tt.tokenize(text)
    pylab.xlabel('Sentence Gap index')
    pylab.ylabel('Gap Scores')
    pylab.plot(range(len(s)), s, label='Gap Scores')
    pylab.plot(range(len(ss)), ss, label='Smoothed Gap scores')
    pylab.plot(range(len(d)), d, label='Depth scores')
    pylab.stem(range(len(b)), b)
    pylab.legend()
    pylab.show()