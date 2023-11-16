import logging
import re
from pathlib import Path
from module.models import EpisodeFile, SubtitleFile
logger = logging.getLogger(__name__)
PLATFORM = 'Unix'
RULES = ['(.*) - (\\d{1,4}(?!\\d|p)|\\d{1,4}\\.\\d{1,2}(?!\\d|p))(?:v\\d{1,2})?(?: )?(?:END)?(.*)', '(.*)[\\[\\ E](\\d{1,4}|\\d{1,4}\\.\\d{1,2})(?:v\\d{1,2})?(?: )?(?:END)?[\\]\\ ](.*)', '(.*)\\[(?:第)?(\\d*\\.*\\d*)[话集話](?:END)?\\](.*)', '(.*)第?(\\d*\\.*\\d*)[话話集](?:END)?(.*)', '(.*)(?:S\\d{2})?EP?(\\d+)(.*)']
SUBTITLE_LANG = {'zh-tw': ['tc', 'cht', '繁', 'zh-tw'], 'zh': ['sc', 'chs', '简', 'zh']}

def get_path_basename(torrent_path: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the basename of a path string.\n\n    :param torrent_path: A string representing a path to a file.\n    :type torrent_path: str\n    :return: A string representing the basename of the given path.\n    :rtype: str\n    '
    return Path(torrent_path).name

def get_group(group_and_title) -> tuple[str | None, str]:
    if False:
        i = 10
        return i + 15
    n = re.split('[\\[\\]()【】（）]', group_and_title)
    while '' in n:
        n.remove('')
    if len(n) > 1:
        if re.match('\\d+', n[1]):
            return (None, group_and_title)
        return (n[0], n[1])
    else:
        return (None, n[0])

def get_season_and_title(season_and_title) -> tuple[str, int]:
    if False:
        return 10
    title = re.sub('([Ss]|Season )\\d{1,3}', '', season_and_title).strip()
    try:
        season = re.search('([Ss]|Season )(\\d{1,3})', season_and_title, re.I).group(2)
    except AttributeError:
        season = 1
    return (title, int(season))

def get_subtitle_lang(subtitle_name: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    for (key, value) in SUBTITLE_LANG.items():
        for v in value:
            if v in subtitle_name.lower():
                return key

def torrent_parser(torrent_path: str, torrent_name: str | None=None, season: int | None=None, file_type: str='media') -> EpisodeFile | SubtitleFile:
    if False:
        for i in range(10):
            print('nop')
    media_path = get_path_basename(torrent_path)
    for rule in RULES:
        if torrent_name:
            match_obj = re.match(rule, torrent_name, re.I)
        else:
            match_obj = re.match(rule, media_path, re.I)
        if match_obj:
            (group, title) = get_group(match_obj.group(1))
            if not season:
                (title, season) = get_season_and_title(title)
            else:
                (title, _) = get_season_and_title(title)
            episode = int(match_obj.group(2))
            suffix = Path(torrent_path).suffix
            if file_type == 'media':
                return EpisodeFile(media_path=torrent_path, group=group, title=title, season=season, episode=episode, suffix=suffix)
            elif file_type == 'subtitle':
                language = get_subtitle_lang(media_path)
                return SubtitleFile(media_path=torrent_path, group=group, title=title, season=season, language=language, episode=episode, suffix=suffix)