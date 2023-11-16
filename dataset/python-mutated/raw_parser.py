import logging
import re
from module.models import Episode
logger = logging.getLogger(__name__)
EPISODE_RE = re.compile('\\d+')
TITLE_RE = re.compile('(.*|\\[.*])( -? \\d+|\\[\\d+]|\\[\\d+.?[vV]\\d]|第\\d+[话話集]|\\[第?\\d+[话話集]]|\\[\\d+.?END]|[Ee][Pp]?\\d+)(.*)')
RESOLUTION_RE = re.compile('1080|720|2160|4K')
SOURCE_RE = re.compile('B-Global|[Bb]aha|[Bb]ilibili|AT-X|Web')
SUB_RE = re.compile('[简繁日字幕]|CH|BIG5|GB')
PREFIX_RE = re.compile('[^\\w\\s\\u4e00-\\u9fff\\u3040-\\u309f\\u30a0-\\u30ff-]')
CHINESE_NUMBER_MAP = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}

def get_group(name: str) -> str:
    if False:
        return 10
    return re.split('[\\[\\]]', name)[1]

def pre_process(raw_name: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    return raw_name.replace('【', '[').replace('】', ']')

def prefix_process(raw: str, group: str) -> str:
    if False:
        i = 10
        return i + 15
    raw = re.sub(f'.{group}.', '', raw)
    raw_process = PREFIX_RE.sub('/', raw)
    arg_group = raw_process.split('/')
    while '' in arg_group:
        arg_group.remove('')
    if len(arg_group) == 1:
        arg_group = arg_group[0].split(' ')
    for arg in arg_group:
        if re.search('新番|月?番', arg) and len(arg) <= 5:
            raw = re.sub(f'.{arg}.', '', raw)
        elif re.search('港澳台地区', arg):
            raw = re.sub(f'.{arg}.', '', raw)
    return raw

def season_process(season_info: str):
    if False:
        return 10
    name_season = season_info
    season_rule = 'S\\d{1,2}|Season \\d{1,2}|[第].[季期]'
    name_season = re.sub('[\\[\\]]', ' ', name_season)
    seasons = re.findall(season_rule, name_season)
    if not seasons:
        return (name_season, '', 1)
    name = re.sub(season_rule, '', name_season)
    for season in seasons:
        season_raw = season
        if re.search('Season|S', season) is not None:
            season = int(re.sub('Season|S', '', season))
            break
        elif re.search('[第 ].*[季期(部分)]|部分', season) is not None:
            season_pro = re.sub('[第季期 ]', '', season)
            try:
                season = int(season_pro)
            except ValueError:
                season = CHINESE_NUMBER_MAP[season_pro]
                break
    return (name, season_raw, season)

def name_process(name: str):
    if False:
        return 10
    (name_en, name_zh, name_jp) = (None, None, None)
    name = name.strip()
    name = re.sub('[(（]仅限港澳台地区[）)]', '', name)
    split = re.split('/|\\s{2}|-\\s{2}', name)
    while '' in split:
        split.remove('')
    if len(split) == 1:
        if re.search('_{1}', name) is not None:
            split = re.split('_', name)
        elif re.search(' - {1}', name) is not None:
            split = re.split('-', name)
    if len(split) == 1:
        split_space = split[0].split(' ')
        for idx in [0, -1]:
            if re.search('^[\\u4e00-\\u9fa5]{2,}', split_space[idx]) is not None:
                chs = split_space[idx]
                split_space.remove(chs)
                split = [chs, ' '.join(split_space)]
                break
    for item in split:
        if re.search('[\\u0800-\\u4e00]{2,}', item) and (not name_jp):
            name_jp = item.strip()
        elif re.search('[\\u4e00-\\u9fa5]{2,}', item) and (not name_zh):
            name_zh = item.strip()
        elif re.search('[a-zA-Z]{3,}', item) and (not name_en):
            name_en = item.strip()
    return (name_en, name_zh, name_jp)

def find_tags(other):
    if False:
        for i in range(10):
            print('nop')
    elements = re.sub('[\\[\\]()（）]', ' ', other).split(' ')
    (sub, resolution, source) = (None, None, None)
    for element in filter(lambda x: x != '', elements):
        if SUB_RE.search(element):
            sub = element
        elif RESOLUTION_RE.search(element):
            resolution = element
        elif SOURCE_RE.search(element):
            source = element
    return (clean_sub(sub), resolution, source)

def clean_sub(sub: str | None) -> str | None:
    if False:
        for i in range(10):
            print('nop')
    if sub is None:
        return sub
    return re.sub('_MP4|_MKV', '', sub)

def process(raw_title: str):
    if False:
        return 10
    raw_title = raw_title.strip()
    content_title = pre_process(raw_title)
    group = get_group(content_title)
    match_obj = TITLE_RE.match(content_title)
    (season_info, episode_info, other) = list(map(lambda x: x.strip(), match_obj.groups()))
    process_raw = prefix_process(season_info, group)
    (raw_name, season_raw, season) = season_process(process_raw)
    (name_en, name_zh, name_jp) = ('', '', '')
    try:
        (name_en, name_zh, name_jp) = name_process(raw_name)
    except ValueError:
        pass
    raw_episode = EPISODE_RE.search(episode_info)
    episode = 0
    if raw_episode is not None:
        episode = int(raw_episode.group())
    (sub, dpi, source) = find_tags(other)
    return (name_en, name_zh, name_jp, season, season_raw, episode, sub, dpi, source, group)

def raw_parser(raw: str) -> Episode | None:
    if False:
        while True:
            i = 10
    ret = process(raw)
    if ret is None:
        logger.error(f'Parser cannot analyse {raw}')
        return None
    (name_en, name_zh, name_jp, season, sr, episode, sub, dpi, source, group) = ret
    return Episode(name_en, name_zh, name_jp, season, sr, episode, sub, group, dpi, source)
if __name__ == '__main__':
    title = '[动漫国字幕组&LoliHouse] THE MARGINAL SERVICE - 08 [WebRip 1080p HEVC-10bit AAC][简繁内封字幕]'
    print(raw_parser(title))