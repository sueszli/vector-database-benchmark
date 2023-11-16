import re
import bangla
from bnnumerizer import numerize
from bnunicodenormalizer import Normalizer
bnorm = Normalizer()
attribution_dict = {'সাঃ': 'সাল্লাল্লাহু আলাইহি ওয়া সাল্লাম', 'আঃ': 'আলাইহিস সালাম', 'রাঃ': 'রাদিআল্লাহু আনহু', 'রহঃ': 'রহমাতুল্লাহি আলাইহি', 'রহিঃ': 'রহিমাহুল্লাহ', 'হাফিঃ': 'হাফিযাহুল্লাহ', 'বায়ান': 'বাইআন', 'দাঃবাঃ': 'দামাত বারাকাতুহুম,দামাত বারাকাতুল্লাহ', '’': '', '‘': '', '/': ' বাই '}

def tag_text(text: str):
    if False:
        return 10
    text = re.sub(' +', ' ', text)
    text = 'start' + text + 'end'
    parts = re.split('[\u0600-ۿ]+', text)
    parts = [p for p in parts if p.strip()]
    parts = set(parts)
    for m in parts:
        if len(m.strip()) > 1:
            text = text.replace(m, f'{m}')
    text = text.replace('start', '')
    text = text.replace('end', '')
    return text

def normalize(sen):
    if False:
        return 10
    global bnorm
    _words = [bnorm(word)['normalized'] for word in sen.split()]
    return ' '.join([word for word in _words if word is not None])

def expand_full_attribution(text):
    if False:
        while True:
            i = 10
    for (word, attr) in attribution_dict.items():
        if word in text:
            text = text.replace(word, normalize(attr))
    return text

def collapse_whitespace(text):
    if False:
        print('Hello World!')
    _whitespace_re = re.compile('\\s+')
    return re.sub(_whitespace_re, ' ', text)

def bangla_text_to_phonemes(text: str) -> str:
    if False:
        return 10
    res = re.search('[0-9]', text)
    if res is not None:
        text = bangla.convert_english_digit_to_bangla_digit(text)
    pattern = '[০, ১, ২, ৩, ৪, ৫, ৬, ৭, ৮, ৯]:[০, ১, ২, ৩, ৪, ৫, ৬, ৭, ৮, ৯]'
    matches = re.findall(pattern, text)
    for m in matches:
        r = m.replace(':', ' এর ')
        text = text.replace(m, r)
    text = numerize(text)
    text = tag_text(text)
    if '' in text:
        text = text.replace('', '').replace('', '')
    bn_text = text.strip()
    sentenceEnders = re.compile('[।!?]')
    sentences = sentenceEnders.split(str(bn_text))
    data = ''
    for sent in sentences:
        res = re.sub('\n', '', sent)
        res = normalize(res)
        res = expand_full_attribution(res)
        res = collapse_whitespace(res)
        res += '।'
        data += res
    return data